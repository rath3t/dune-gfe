#ifndef LFE_ORDER
    #define LFE_ORDER 2
#endif

#ifndef GFE_ORDER
    #define GFE_ORDER 1
#endif

#ifndef MIXED_SPACE
    #if LFE_ORDER != GFE_ORDER
    #define MIXED_SPACE 1
    #else
    #define MIXED_SPACE 0
    #endif
#endif

//#define PROJECTED_INTERPOLATION

#include <config.h>

#include <fenv.h>
#include <array>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>    // use of "Easy to Use" drivers
#include <adolc/taping.h>

#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/typetraits.hh>
#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/common/tuplevector.hh>
#include <dune/common/version.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#if HAVE_DUNE_FOAMGRID
#include <dune/foamgrid/foamgrid.hh>
#endif

#include <dune/istl/multitypeblockvector.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/compositebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>

#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/functiontools/boundarydofs.hh>
#include <dune/fufem/dunepython.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/localprojectedfefunction.hh>
#include <dune/gfe/assemblers/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/assemblers/cosseratenergystiffness.hh>
#include <dune/gfe/assemblers/nonplanarcosseratshellenergy.hh>
#include <dune/gfe/cosseratvtkwriter.hh>
#include <dune/gfe/cosseratvtkreader.hh>
#include <dune/gfe/assemblers/geodesicfeassembler.hh>
#include <dune/gfe/embeddedglobalgfefunction.hh>
#include <dune/gfe/assemblers/mixedgfeassembler.hh>

#if MIXED_SPACE
#include <dune/gfe/mixedriemannianpnsolver.hh>
#include <dune/gfe/mixedriemanniantrsolver.hh>
#else
#include <dune/gfe/assemblers/geodesicfeassemblerwrapper.hh>
#include <dune/gfe/riemannianpnsolver.hh>
#include <dune/gfe/riemanniantrsolver.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>
#endif

#if HAVE_DUNE_VTK
#include <dune/vtk/vtkreader.hh>
#endif

#include <dune/gmsh4/gmsh4reader.hh>
#include <dune/gmsh4/gridcreators/lagrangegridcreator.hh>

using namespace Dune;

// grid dimension
const int dim = GRID_DIM;
const int dimworld = WORLD_DIM;

// Order of the approximation space for the displacement
const int displacementOrder = LFE_ORDER;

// Order of the approximation space for the microrotations
const int rotationOrder = GFE_ORDER;

#if !MIXED_SPACE
static_assert(displacementOrder==rotationOrder, "displacement and rotation order do not match!");
#endif

// Image space of the geodesic fe functions
using TargetSpace = GFE::ProductManifold<RealTuple<double,3>,Rotation<double,3> >;


int main (int argc, char *argv[]) try
{
  // initialize MPI, finalize is done automatically on exit
  Dune::MPIHelper& mpiHelper = MPIHelper::instance(argc, argv);

  if (mpiHelper.rank()==0) {
    std::cout << "DISPLACEMENTORDER = " << displacementOrder << ", ROTATIONORDER = " << rotationOrder << std::endl;
    std::cout << "dim = " << dim << ", dimworld = " << dimworld << std::endl;
    #if MIXED_SPACE
    std::cout << "MIXED_SPACE = 1" << std::endl;
    #else
    std::cout << "MIXED_SPACE = 0" << std::endl;
    #endif
  }

  // Check for appropriate number of command line arguments
  if (argc < 3)
    DUNE_THROW(Exception, "Usage: ./cosserat-continuum <python path> <parameter file>");

  // Start Python interpreter
  Python::start();
  auto pyMain = Python::main();
  Python::run("import math");

  //feenableexcept(FE_INVALID);
  Python::runStream()
    << std::endl << "import sys"
    << std::endl << "sys.path.append('" << argv[1] << "')"
    << std::endl;

  // parse data file
  auto pyModule = pyMain.import(argv[2]);

  // Get main parameter set
  ParameterTree parameterSet;
  pyModule.get("parameterSet").toC(parameterSet);

  ParameterTreeParser::readOptions(argc, argv, parameterSet);

  // read solver settings
  const int numLevels                   = parameterSet.get<int>("numLevels");
  int numHomotopySteps                  = parameterSet.get<int>("numHomotopySteps");
  const double tolerance                = parameterSet.get<double>("tolerance");
  const int maxSolverSteps              = parameterSet.get<int>("maxSolverSteps");
  const double initialRegularization    = parameterSet.get<double>("initialRegularization", 1000);
  const double initialTrustRegionRadius = parameterSet.get<double>("initialTrustRegionRadius");
  const int multigridIterations         = parameterSet.get<int>("numIt");
  const int nu1                         = parameterSet.get<int>("nu1");
  const int nu2                         = parameterSet.get<int>("nu2");
  const int mu                          = parameterSet.get<int>("mu");
  const int baseIterations              = parameterSet.get<int>("baseIt");
  const double mgTolerance              = parameterSet.get<double>("mgTolerance");
  const double baseTolerance            = parameterSet.get<double>("baseTolerance");
  const bool instrumented               = parameterSet.get<bool>("instrumented");
  const bool adolcScalarMode            = parameterSet.get<bool>("adolcScalarMode", false);
  const std::string resultPath          = parameterSet.get("resultPath", "");

  using namespace Dune::Indices;
  using SolutionType = TupleVector<std::vector<RealTuple<double,3> >,
      std::vector<Rotation<double,3> > >;

  // ///////////////////////////////////////
  //    Create the grid
  // ///////////////////////////////////////
#if HAVE_DUNE_FOAMGRID
  typedef std::conditional<dim==dimworld,UGGrid<dim>, FoamGrid<dim,dimworld> >::type GridType;
#else
  static_assert(dim==dimworld, "FoamGrid needs to be installed to allow problems with dim != dimworld.");
  typedef UGGrid<dim> GridType;
#endif

  std::shared_ptr<GridType> grid;

  GridFactory<GridType> factory;
  Gmsh4::LagrangeGridCreator creator{factory};

  FieldVector<double,dimworld> lower(0), upper(1);

  std::string structuredGridType = parameterSet["structuredGrid"];
  if (structuredGridType == "cube") {
    if (dim!=dimworld)
      DUNE_THROW(GridError, "Please use FoamGrid and read in a grid for problems with dim != dimworld.");

    lower = parameterSet.get<FieldVector<double,dimworld> >("lower");
    upper = parameterSet.get<FieldVector<double,dimworld> >("upper");

    auto elements = parameterSet.get<std::array<unsigned int,dim> >("elements");
    grid = StructuredGridFactory<GridType>::createCubeGrid(lower, upper, elements);

  } else {
    std::string path                = parameterSet.get<std::string>("path", "");
    std::string gridFile            = parameterSet.get<std::string>("gridFile");

    // Guess the grid file format by looking at the file name suffix
    auto dotPos = gridFile.rfind('.');
    if (dotPos == std::string::npos)
      DUNE_THROW(IOError, "Could not determine grid input file format");
    std::string suffix = gridFile.substr(dotPos, gridFile.length()-dotPos);

    if (suffix == ".msh") {
      Gmsh4Reader reader{creator};
      reader.read(path + "/" + gridFile);
      grid = factory.createGrid();
    } else if (suffix == ".vtu" or suffix == ".vtp")
#if HAVE_DUNE_VTK
#if DUNE_VERSION_GTE(DUNE_VTK, 2, 10)
      grid = Vtk::VtkReader<GridType>::createGridFromFile(path + "/" + gridFile);
#else
      grid = VtkReader<GridType>::createGridFromFile(path + "/" + gridFile);
#endif
#else
      DUNE_THROW(NotImplemented, "Please install dune-vtk for VTK reading support!");
#endif
  }

  grid->globalRefine(numLevels-1);

  grid->loadBalance();

  if (mpiHelper.rank()==0)
    std::cout << "There are " << grid->leafGridView().comm().size() << " processes" << std::endl;

  typedef GridType::LeafGridView GridView;
  GridView gridView = grid->leafGridView();

  using namespace Dune::Functions::BasisFactory;

  const int dimRotation = Rotation<double,3>::embeddedDim;
  auto compositeBasis = makeBasis(
    gridView,
    composite(
      power<3>(
        lagrange<displacementOrder>()
        ),
      power<dimRotation>(
        lagrange<rotationOrder>()
        )
      ));
  using CompositeBasis = decltype(compositeBasis);

  auto deformationPowerBasis = makeBasis(
    gridView,
    power<3>(
      lagrange<displacementOrder>()
      ));

  // Vector-valued basis to represent directors
  auto directorBasis = makeBasis(
    gridView,
    power<3>(
      lagrange<rotationOrder>()
      ));

  // Matrix-valued basis for treating the microrotation as a matrix field
  auto orientationMatrixBasis = makeBasis(
    gridView,
    power<3>(
      power<3>(
        lagrange<rotationOrder>()
        )
      ));

  BlockVector<FieldVector<double,dimworld> > identityDeformation(compositeBasis.size({0}));
  auto identityDeformationBasis = makeBasis(
    gridView,
    power<dimworld>(
      lagrange<displacementOrder>()
      ));
  Dune::Functions::interpolate(identityDeformationBasis, identityDeformation, [&](FieldVector<double,dimworld> x){
    return x;
  });


  BlockVector<FieldVector<double,dimworld> > identityRotation(compositeBasis.size({1}));
  auto identityRotationBasis = makeBasis(
    gridView,
    power<dimworld>(
      lagrange<rotationOrder>()
      ));
  Dune::Functions::interpolate(identityRotationBasis, identityRotation, [&](FieldVector<double,dimworld> x){
    return x;
  });

  typedef Dune::Functions::LagrangeBasis<GridView,displacementOrder> DeformationFEBasis;
  typedef Dune::Functions::LagrangeBasis<GridView,rotationOrder> OrientationFEBasis;

  DeformationFEBasis deformationFEBasis(gridView);
  OrientationFEBasis orientationFEBasis(gridView);

  // /////////////////////////////////////////
  //   Read Dirichlet values
  // /////////////////////////////////////////

  BitSetVector<1> neumannVertices(gridView.size(dim), false);
  BitSetVector<3> deformationDirichletDofs(deformationFEBasis.size(), false);
  BitSetVector<3> orientationDirichletDofs(orientationFEBasis.size(), false);

  const GridView::IndexSet& indexSet = gridView.indexSet();

  // Make Python function that computes which vertices are on the Dirichlet boundary, based on the vertex positions.
  std::string lambda = std::string("lambda x: (") + parameterSet.get<std::string>("dirichletVerticesPredicate") + std::string(")");
  auto pythonDirichletVertices = Python::make_function<FieldVector<bool,3> >(Python::evaluate(lambda));

  lambda = std::string("lambda x: (") + parameterSet.get<std::string>("dirichletRotationVerticesPredicate") + std::string(")");
  auto pythonOrientationDirichletVertices = Python::make_function<bool>(Python::evaluate(lambda));

  // Same for the Neumann boundary
  lambda = std::string("lambda x: (") + parameterSet.get<std::string>("neumannVerticesPredicate", "0") + std::string(")");
  auto pythonNeumannVertices = Python::make_function<bool>(Python::evaluate(lambda));

  for (auto&& vertex : vertices(gridView))
  {
    bool isNeumann = pythonNeumannVertices(vertex.geometry().corner(0));
    neumannVertices[indexSet.index(vertex)] = isNeumann;
  }

  BoundaryPatch<GridView> neumannBoundary(gridView, neumannVertices);

  std::cout << "On rank " << mpiHelper.rank() << ": Neumann boundary has " << neumannBoundary.numFaces()
            << " faces and " << neumannVertices.count() << " degrees of freedom.\n";

  BitSetVector<1> neumannNodes(deformationFEBasis.size(), false);
#if DUNE_VERSION_GTE(DUNE_FUFEM, 2, 10)
  Fufem::markBoundaryPatchDofs(neumannBoundary,deformationFEBasis,neumannNodes);
#else
  constructBoundaryDofs(neumannBoundary,deformationFEBasis,neumannNodes);
#endif

  for (size_t i=0; i<deformationFEBasis.size(); i++) {
    FieldVector<bool,3> isDirichlet;
    isDirichlet = pythonDirichletVertices(identityDeformation[i]);
    for (size_t j=0; j<3; j++)
      deformationDirichletDofs[i][j] = isDirichlet[j];
  }


  for (size_t i=0; i<orientationFEBasis.size(); i++) {
    bool isDirichletOrientation = pythonOrientationDirichletVertices(identityRotation[i]);
    for (size_t j=0; j<3; j++)
      orientationDirichletDofs[i][j] = isDirichletOrientation;
  }

  std::cout << "On rank " << mpiHelper.rank() << ": Dirichlet boundary has " << deformationDirichletDofs.count() << " degrees of freedom.\n";
  std::cout << "On rank " << mpiHelper.rank() << ": Dirichlet boundary (orientation) has " << orientationDirichletDofs.count() << " degrees of freedom.\n";

  // //////////////////////////
  //   Initial iterate
  // //////////////////////////

  SolutionType x;

  x[_0].resize(compositeBasis.size({0}));

  lambda = std::string("lambda x: (") + parameterSet.get<std::string>("initialDeformation") + std::string(")");
  auto pythonInitialDeformation = Python::make_function<FieldVector<double,3> >(Python::evaluate(lambda));

  std::vector<FieldVector<double,3> > v;
  Functions::interpolate(deformationPowerBasis, v, pythonInitialDeformation);

  for (size_t i=0; i<x[_0].size(); i++)
    x[_0][i] = v[i];

  x[_1].resize(compositeBasis.size({1}));
#if !MIXED_SPACE
  if (parameterSet.hasKey("startFromFile"))
  {
    std::shared_ptr<GridType> initialIterateGrid;
    if (parameterSet.get<bool>("structuredGrid"))
    {
      std::array<unsigned int,dim> elements = parameterSet.get<std::array<unsigned int,dim> >("elements");
      initialIterateGrid = StructuredGridFactory<GridType>::createCubeGrid(lower, upper, elements);
    }
    else
    {
      std::string path                       = parameterSet.get<std::string>("path");
      std::string initialIterateGridFilename = parameterSet.get<std::string>("initialIterateGridFilename");

      initialIterateGrid = std::shared_ptr<GridType>(Gmsh4Reader<GridType>::createGridFromFile(path + "/" + initialIterateGridFilename));
    }

    std::vector<TargetSpace> initialIterate;
    GFE::CosseratVTKReader::read(initialIterate, parameterSet.get<std::string>("initialIterateFilename"));

    typedef Dune::Functions::LagrangeBasis<typename GridType::LeafGridView, 2> InitialBasis;
    InitialBasis initialBasis(initialIterateGrid->leafGridView());

#ifdef PROJECTED_INTERPOLATION
    using LocalInterpolationRule  = GFE::LocalProjectedFEFunction<dim, double, DeformationFEBasis::LocalView::Tree::FiniteElement, TargetSpace>;
#else
    using LocalInterpolationRule  = LocalGeodesicFEFunction<dim, double, DeformationFEBasis::LocalView::Tree::FiniteElement, TargetSpace>;
#endif
    GFE::EmbeddedGlobalGFEFunction<InitialBasis,LocalInterpolationRule,TargetSpace> initialFunction(initialBasis,initialIterate);
    auto powerBasis = makeBasis(
      gridView,
      power<7>(
        lagrange<displacementOrder>()
        ));
    std::vector<FieldVector<double,7> > v;
    //TODO: Interpolate does not work with an GFE:EmbeddedGlobalGFEFunction
    //Dune::Functions::interpolate(powerBasis,v,initialFunction);

    for (size_t i=0; i<x.size(); i++) {
      auto vTargetSpace = TargetSpace(v[i]);
      x[_0][i] = vTargetSpace[_0];
      x[_1][i] = vTargetSpace[_1];
    }
  }
#endif

  ////////////////////////////////////////////////////////
  //   Main homotopy loop
  ////////////////////////////////////////////////////////

  // Output initial iterate (of homotopy loop)

  // Compute the displacement from the deformation, because that's more easily visualized
  // in ParaView
  BlockVector<FieldVector<double,3> > displacement(x[_0].size());
  for (size_t i=0; i<x[_0].size(); i++)
  {
    for (int j = 0; j < dimworld; j ++)
      displacement[i][j] = x[_0][i][j] - identityDeformation[i][j];
    for (int j = dimworld; j<3; j++)
      displacement[i][j] = x[_0][i][j];
  }

  auto displacementFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,3> >(deformationPowerBasis, displacement);

  if (dim == dimworld) {
    CosseratVTKWriter<GridView>::write(gridView,
                                       displacementFunction,
                                       directorBasis,
                                       x[_1],
                                       std::max(LFE_ORDER, GFE_ORDER),
                                       resultPath + "cosserat_homotopy_0_l" + std::to_string(numLevels));
  } else if (dim == 2 && dimworld == 3) {
#if MIXED_SPACE
    CosseratVTKWriter<GridView>::write<DeformationFEBasis>(deformationFEBasis, x[_0], resultPath + "cosserat_homotopy_0_l" + std::to_string(numLevels));
#else
    CosseratVTKWriter<GridView>::write<DeformationFEBasis>(deformationFEBasis, x, resultPath + "cosserat_homotopy_0_l" + std::to_string(numLevels));
#endif
  }

  for (int i=0; i<numHomotopySteps; i++) {

    double homotopyParameter = (i+1)*(1.0/numHomotopySteps);
    if (mpiHelper.rank()==0)
      std::cout << "Homotopy step: " << i << ",    parameter: " << homotopyParameter << std::endl;


    // ////////////////////////////////////////////////////////////
    //   Create an assembler for the energy functional
    // ////////////////////////////////////////////////////////////

    const ParameterTree& materialParameters = parameterSet.sub("materialParameters");
    FieldVector<double,3> neumannValues {0,0,0};
    if (parameterSet.hasKey("neumannValues"))
      neumannValues = parameterSet.get<FieldVector<double,3> >("neumannValues");

    auto neumannFunction = [&]( FieldVector<double,dimworld> ) {
                             auto nV = neumannValues;
                             nV *= (-homotopyParameter);
                             return nV;
                           };

    Python::Reference volumeLoadClass = Python::import(parameterSet.get<std::string>("volumeLoadPythonFunction", "zero-volume-load"));
    Python::Callable volumeLoadCallable = volumeLoadClass.get("VolumeLoad");

    // Call a constructor
    Python::Reference volumeLoadPythonObject = volumeLoadCallable(homotopyParameter);

    // Extract object member functions as Dune functions
    auto volumeLoad = Python::make_function<FieldVector<double,3> > (volumeLoadPythonObject.get("volumeLoad"));

    if (mpiHelper.rank() == 0) {
      std::cout << "Material parameters:" << std::endl;
      materialParameters.report();
    }

    ////////////////////////////////////////////////////////
    //   Set Dirichlet values
    ////////////////////////////////////////////////////////

    // Dirichlet boundary value function, depending on the homotopy parameter
    Python::Callable dirichletValuesPythonClass = pyModule.get("DirichletValues");

    // Construct with a particular value of the homotopy parameter
    Python::Reference dirichletValuesPythonObject = dirichletValuesPythonClass(homotopyParameter);

    // Extract object member functions as Dune functions
    auto deformationDirichletValues = Python::make_function<FieldVector<double,3> >   (dirichletValuesPythonObject.get("deformation"));
    auto orientationDirichletValues = Python::make_function<FieldMatrix<double,3,3> > (dirichletValuesPythonObject.get("orientation"));

    BlockVector<FieldVector<double,3> > ddV;
    Dune::Functions::interpolate(deformationPowerBasis, ddV, deformationDirichletValues);

    BlockVector<FieldMatrix<double,3,3> > dOV;
    Functions::interpolate(orientationMatrixBasis, dOV, orientationDirichletValues);

    for (std::size_t i = 0; i < compositeBasis.size({0}); i++) {
      FieldVector<double,3> x0i = x[_0][i].globalCoordinates();
      for (int j=0; j<3; j++) {
        if (deformationDirichletDofs[i][j])
          x0i[j] = ddV[i][j];
      }
      x[_0][i] = x0i;
    }
    for (std::size_t i = 0; i < compositeBasis.size({1}); i++)
      if (orientationDirichletDofs[i][0])
        x[_1][i].set(dOV[i]);

    if (dim==dimworld) {
      auto localCosseratEnergy = std::make_shared<CosseratEnergyLocalStiffness<CompositeBasis, 3,adouble> > (materialParameters,
                                                                                                             &neumannBoundary,
                                                                                                             neumannFunction,
                                                                                                             volumeLoad);

      LocalGeodesicFEADOLCStiffness<CompositeBasis,TargetSpace> localGFEADOLCStiffness(localCosseratEnergy,
                                                                                       adolcScalarMode);
      MixedGFEAssembler<CompositeBasis,TargetSpace> mixedAssembler(compositeBasis, localGFEADOLCStiffness);
#if MIXED_SPACE
      if (parameterSet.get<std::string>("solvertype", "trustRegion") == "trustRegion")
      {

        MixedRiemannianTrustRegionSolver<GridType,
            CompositeBasis,
            DeformationFEBasis, RealTuple<double,3>,
            OrientationFEBasis, Rotation<double,3> > solver;
        solver.setup(*grid,
                     &mixedAssembler,
                     deformationFEBasis,
                     orientationFEBasis,
                     x,
                     deformationDirichletDofs,
                     orientationDirichletDofs, tolerance,
                     maxSolverSteps,
                     initialTrustRegionRadius,
                     multigridIterations,
                     mgTolerance,
                     mu, nu1, nu2,
                     baseIterations,
                     baseTolerance,
                     instrumented);

        solver.setScaling(parameterSet.get<FieldVector<double,6> >("solverScaling"));

        solver.setInitialIterate(x);
        solver.solve();
        x = solver.getSol();
      }
      else
      {
        //Create BitVector matching the tangential space
        const int dimRotationTangent = Rotation<double,3>::TangentVector::dimension;
        using VectorForBit = MultiTypeBlockVector<std::vector<FieldVector<double,3> >, std::vector<FieldVector<double,dimRotationTangent> > >;
        using BitVector = Solvers::DefaultBitVector_t<VectorForBit>;
        BitVector dirichletDofs;
        dirichletDofs[_0].resize(compositeBasis.size({0}));
        dirichletDofs[_1].resize(compositeBasis.size({1}));
        for (size_t i = 0; i < compositeBasis.size({0}); i++) {
          for (size_t j = 0; j < 3; j++)
            dirichletDofs[_0][i][j] = deformationDirichletDofs[i][j];
        }
        for (size_t i = 0; i < compositeBasis.size({1}); i++) {
          for (int j = 0; j < dimRotationTangent; j++)
            dirichletDofs[_1][i][j] = orientationDirichletDofs[i][j];
        }
        GFE::MixedRiemannianProximalNewtonSolver<CompositeBasis, DeformationFEBasis, RealTuple<double,3>, OrientationFEBasis, Rotation<double,3>, BitVector> solver;
        solver.setup(*grid,
                     &mixedAssembler,
                     x,
                     dirichletDofs,
                     tolerance,
                     maxSolverSteps,
                     initialRegularization,
                     instrumented);
        solver.setInitialIterate(x);
        solver.solve();
        x = solver.getSol();
      }
#else
      //The MixedRiemannianTrustRegionSolver can treat the Displacement and Orientation Space as separate ones
      //The RiemannianTrustRegionSolver can only treat the Displacement and Rotation together in a ProductManifold.
      //Therefore, x and the dirichletDofs are converted to a ProductManifold structure, as well as the Hessian and Gradient that are returned by the assembler
      std::vector<TargetSpace> xTargetSpace(compositeBasis.size({0}));
      BitSetVector<TargetSpace::TangentVector::dimension> dirichletDofsTargetSpace(compositeBasis.size({0}), false);
      for (std::size_t i = 0; i < compositeBasis.size({0}); i++) {
        for (int j = 0; j < 3; j ++) {       // Displacement part
          xTargetSpace[i][_0].globalCoordinates()[j] = x[_0][i][j];
          dirichletDofsTargetSpace[i][j] = deformationDirichletDofs[i][j];
        }
        xTargetSpace[i][_1] = x[_1][i];       // Rotation part
        for (int j = 3; j < TargetSpace::TangentVector::dimension; j ++)
          dirichletDofsTargetSpace[i][j] = orientationDirichletDofs[i][j-3];
      }

      using GFEAssemblerWrapper = Dune::GFE::GeodesicFEAssemblerWrapper<CompositeBasis, DeformationFEBasis, TargetSpace>;
      GFEAssemblerWrapper assembler(&mixedAssembler, deformationFEBasis);
      if (parameterSet.get<std::string>("solvertype", "trustRegion") == "trustRegion") {
        RiemannianTrustRegionSolver<DeformationFEBasis, TargetSpace, GFEAssemblerWrapper> solver;
        solver.setup(*grid,
                     &assembler,
                     xTargetSpace,
                     dirichletDofsTargetSpace,
                     tolerance,
                     maxSolverSteps,
                     initialTrustRegionRadius,
                     multigridIterations,
                     mgTolerance,
                     mu, nu1, nu2,
                     baseIterations,
                     baseTolerance,
                     instrumented);

        solver.setScaling(parameterSet.get<FieldVector<double,6> >("solverScaling"));
        solver.setInitialIterate(xTargetSpace);
        solver.solve();
        xTargetSpace = solver.getSol();
      } else {
        RiemannianProximalNewtonSolver<DeformationFEBasis, TargetSpace, GFEAssemblerWrapper> solver;
        solver.setup(*grid,
                     &assembler,
                     xTargetSpace,
                     dirichletDofsTargetSpace,
                     tolerance,
                     maxSolverSteps,
                     initialRegularization,
                     instrumented);
        solver.setScaling(parameterSet.get<FieldVector<double,6> >("solverScaling"));
        solver.setInitialIterate(xTargetSpace);
        solver.solve();
        xTargetSpace = solver.getSol();
      }
      for (std::size_t i = 0; i < xTargetSpace.size(); i++) {
        x[_0][i] = xTargetSpace[i][_0];
        x[_1][i] = xTargetSpace[i][_1];
      }
#endif
    } else {     //dim != dimworld
      using StiffnessType = LocalGeodesicFEADOLCStiffness<CompositeBasis, TargetSpace>;
      std::shared_ptr<StiffnessType> localGFEStiffness;

#if HAVE_DUNE_CURVEDGEOMETRY && WORLD_DIM == 3 && GRID_DIM == 2
      auto localCosseratEnergy = std::make_shared<NonplanarCosseratShellEnergy<CompositeBasis, 3, adouble, decltype(creator)> >(materialParameters,
                                                                                                                                &creator,
                                                                                                                                &neumannBoundary,
                                                                                                                                neumannFunction,
                                                                                                                                volumeLoad);

      localGFEStiffness = std::make_shared<StiffnessType>(localCosseratEnergy, adolcScalarMode);
#endif
      MixedGFEAssembler<CompositeBasis,TargetSpace> mixedAssembler(compositeBasis, localGFEStiffness);
#if MIXED_SPACE
      MixedRiemannianTrustRegionSolver<GridType,
          CompositeBasis,
          DeformationFEBasis, RealTuple<double,3>,
          OrientationFEBasis, Rotation<double,3> > solver;
      solver.setup(*grid,
                   &mixedAssembler,
                   deformationFEBasis,
                   orientationFEBasis,
                   x,
                   deformationDirichletDofs,
                   orientationDirichletDofs, tolerance,
                   maxSolverSteps,
                   initialTrustRegionRadius,
                   multigridIterations,
                   mgTolerance,
                   mu, nu1, nu2,
                   baseIterations,
                   baseTolerance,
                   instrumented);

      solver.setScaling(parameterSet.get<FieldVector<double,6> >("solverScaling"));

      solver.setInitialIterate(x);
      solver.solve();
      x = solver.getSol();
#else
      //The MixedRiemannianTrustRegionSolver can treat the Displacement and Orientation Space as separate ones
      //The RiemannianTrustRegionSolver can only treat the Displacement and Rotation together in a ProductManifold.
      //Therefore, x and the dirichletDofs are converted to a ProductManifold structure, as well as the Hessian and Gradient that are returned by the assembler
      std::vector<TargetSpace> xTargetSpace(compositeBasis.size({0}));
      BitSetVector<TargetSpace::TangentVector::dimension> dirichletDofsTargetSpace(compositeBasis.size({0}), false);
      for (std::size_t i = 0; i < compositeBasis.size({0}); i++) {
        for (int j = 0; j < 3; j ++) {       // Displacement part
          xTargetSpace[i][_0].globalCoordinates()[j] = x[_0][i][j];
          dirichletDofsTargetSpace[i][j] = deformationDirichletDofs[i][j];
        }
        xTargetSpace[i][_1] = x[_1][i];       // Rotation part
        for (int j = 3; j < TargetSpace::TangentVector::dimension; j ++)
          dirichletDofsTargetSpace[i][j] = orientationDirichletDofs[i][j-3];
      }

      using GFEAssemblerWrapper = Dune::GFE::GeodesicFEAssemblerWrapper<CompositeBasis, DeformationFEBasis, TargetSpace>;
      GFEAssemblerWrapper assembler(&mixedAssembler, deformationFEBasis);
      if (parameterSet.get<std::string>("solvertype", "trustRegion") == "trustRegion") {
        RiemannianTrustRegionSolver<DeformationFEBasis, TargetSpace, GFEAssemblerWrapper> solver;
        solver.setup(*grid,
                     &assembler,
                     xTargetSpace,
                     dirichletDofsTargetSpace,
                     tolerance,
                     maxSolverSteps,
                     initialTrustRegionRadius,
                     multigridIterations,
                     mgTolerance,
                     mu, nu1, nu2,
                     baseIterations,
                     baseTolerance,
                     instrumented);

        solver.setScaling(parameterSet.get<FieldVector<double,6> >("solverScaling"));
        solver.setInitialIterate(xTargetSpace);
        solver.solve();
        xTargetSpace = solver.getSol();
      } else {       //parameterSet.get<std::string>("solvertype") == "proximalNewton"
        RiemannianProximalNewtonSolver<DeformationFEBasis, TargetSpace, GFEAssemblerWrapper> solver;
        solver.setup(*grid,
                     &assembler,
                     xTargetSpace,
                     dirichletDofsTargetSpace,
                     tolerance,
                     maxSolverSteps,
                     initialRegularization,
                     instrumented);
        solver.setScaling(parameterSet.get<FieldVector<double,6> >("solverScaling"));
        solver.setInitialIterate(xTargetSpace);
        solver.solve();
        xTargetSpace = solver.getSol();
      }

      for (std::size_t i = 0; i < xTargetSpace.size(); i++) {
        x[_0][i] = xTargetSpace[i][_0];
        x[_1][i] = xTargetSpace[i][_1];
      }
#endif
    }
    // Output result of each homotopy step

    // Compute the displacement from the deformation, because that's more easily visualized
    // in ParaView
    BlockVector<FieldVector<double,3> > displacement(x[_0].size());
    for (size_t i=0; i<x[_0].size(); i++)
    {
      for (int j = 0; j < dimworld; j ++)
        displacement[i][j] = x[_0][i][j] - identityDeformation[i][j];
      for (int j = dimworld; j<3; j++)
        displacement[i][j] = x[_0][i][j];
    }

    auto displacementFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,3> >(deformationPowerBasis, displacement);

    if (dim == dimworld) {
      CosseratVTKWriter<GridView>::write(gridView,
                                         displacementFunction,
                                         directorBasis,
                                         x[_1],
                                         std::max(LFE_ORDER, GFE_ORDER),
                                         resultPath + "cosserat_homotopy_" + std::to_string(i+1) + "_l" + std::to_string(numLevels));
    } else if (dim == 2 && dimworld == 3) {
#if MIXED_SPACE
      CosseratVTKWriter<GridView>::write<DeformationFEBasis>(deformationFEBasis, x[_0], resultPath + "cosserat_homotopy_" + std::to_string(i+1) + "_l" + std::to_string(numLevels));
#else
      CosseratVTKWriter<GridView>::write<DeformationFEBasis>(deformationFEBasis, x, resultPath + "cosserat_homotopy_" + std::to_string(i+1) + "_l" + std::to_string(numLevels));
#endif
    }
  }

  // //////////////////////////////
  //   Output result
  // //////////////////////////////

#if !MIXED_SPACE
  // Write the corresponding coefficient vector: verbatim in binary, to be completely lossless
  // This data may be used by other applications measuring the discretization error
  BlockVector<TargetSpace::CoordinateType> xEmbedded(compositeBasis.size({0}));
  for (size_t i=0; i<compositeBasis.size({0}); i++)
    for (size_t j=0; j<TargetSpace::TangentVector::dimension; j++)
      xEmbedded[i][j] = j < 3 ? x[_0][i][j] : x[_1][i].globalCoordinates()[j-3];

  std::ofstream outFile("cosserat-continuum-result-" + std::to_string(numLevels) + ".data", std::ios_base::binary);
  MatrixVector::Generic::writeBinary(outFile, xEmbedded);
  outFile.close();
#endif

  if (parameterSet.get<bool>("computeAverageNeumannDeflection", false) and parameterSet.hasKey("neumannValues")) {
    // finally: compute the average deformation of the Neumann boundary
    // That is what we need for the locking tests
    FieldVector<double,3> averageDef(0);
    for (size_t i=0; i<x[_0].size(); i++)
      if (neumannNodes[i][0])
        averageDef += x[_0][i].globalCoordinates();
    averageDef /= neumannNodes.count();

    if (mpiHelper.rank()==0 and parameterSet.hasKey("neumannValues"))
    {
      std::cout << "Neumann values = " << parameterSet.get<FieldVector<double, 3> >("neumannValues") << "  "
                << ",  average deflection: " << averageDef << std::endl;
    }
  }

}
catch (Exception& e)
{
  std::cout << e.what() << std::endl;
}
