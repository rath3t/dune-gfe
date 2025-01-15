#ifndef LFE_ORDER
    #define LFE_ORDER 2
#endif

#ifndef GFE_ORDER
    #define GFE_ORDER 2
#endif

#ifndef MIXED_SPACE
    #if LFE_ORDER != GFE_ORDER
    #define MIXED_SPACE 1
    #else
    #define MIXED_SPACE 0
    #endif
#endif

#include <iostream>
#include <fstream>

#include <config.h>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>    // use of "Easy to Use" drivers
#include <adolc/taping.h>

#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/bitsetvector.hh>
#include <dune/common/indices.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/common/timer.hh>
#include <dune/common/tuplevector.hh>
#include <dune/common/version.hh>

#if DUNE_VERSION_GTE(DUNE_ELASTICITY, 2, 11)
#include <dune/elasticity/densities/exphenckydensity.hh>
#include <dune/elasticity/densities/henckydensity.hh>
#include <dune/elasticity/densities/mooneyrivlindensity.hh>
#include <dune/elasticity/densities/neohookedensity.hh>
#include <dune/elasticity/densities/stvenantkirchhoffdensity.hh>
#else
#include <dune/elasticity/materials/exphenckydensity.hh>
#include <dune/elasticity/materials/henckydensity.hh>
#include <dune/elasticity/materials/mooneyrivlindensity.hh>
#include <dune/elasticity/materials/neohookedensity.hh>
#include <dune/elasticity/materials/stvenantkirchhoffdensity.hh>
#endif

#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
#include <dune/functions/functionspacebases/compositebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>

#include <dune/fufem/functiontools/boundarydofs.hh>
#include <dune/fufem/dunepython.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/io/file/vtk.hh>

#include <dune/gfe/cosseratvtkwriter.hh>
#include <dune/gfe/assemblers/localintegralenergy.hh>
#include <dune/gfe/assemblers/mixedgfeassembler.hh>
#include <dune/gfe/assemblers/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/neumannenergy.hh>
#include <dune/gfe/assemblers/surfacecosseratenergy.hh>
#include <dune/gfe/assemblers/sumenergy.hh>
#include <dune/gfe/densities/duneelasticitydensity.hh>

#if MIXED_SPACE
#include <dune/gfe/mixedriemannianpnsolver.hh>
#include <dune/gfe/mixedriemanniantrsolver.hh>
#else
#include <dune/gfe/assemblers/geodesicfeassemblerwrapper.hh>
#include <dune/gfe/riemannianpnsolver.hh>
#include <dune/gfe/riemanniantrsolver.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#endif

#include <dune/istl/multitypeblockvector.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>


const int dim = WORLD_DIM;

const int targetDim = WORLD_DIM;

const int displacementOrder = GFE_ORDER;
const int rotationOrder = LFE_ORDER;

const int stressFreeDataOrder = 2;

#if !MIXED_SPACE
static_assert(displacementOrder==rotationOrder, "displacement and rotation order do not match!");
#endif

//differentiation method
typedef adouble ValueType;

using namespace Dune;

int main (int argc, char *argv[]) try
{
  Dune::Timer overallTimer;
  // initialize MPI, finalize is done automatically on exit
  Dune::MPIHelper& mpiHelper = MPIHelper::instance(argc, argv);

  if (mpiHelper.rank()==0) {
    std::cout << "DISPLACEMENTORDER = " << displacementOrder << ", ROTATIONORDER = " << rotationOrder << std::endl;
#if MIXED_SPACE
    std::cout << "MIXED_SPACE = 1" << std::endl;
#else
    std::cout << "MIXED_SPACE = 0" << std::endl;
#endif
  }

  if (argc < 3)
    DUNE_THROW(Exception, "Usage: ./film-on-substrate <python path> <parameter file>");

  // Start Python interpreter
  Python::start();
  Python::Reference main = Python::import("__main__");
  Python::run("import math");

  //feenableexcept(FE_INVALID);
  Python::runStream()
    << std::endl << "import sys"
    << std::endl << "sys.path.append('" << argv[1] << "')"
    << std::endl;

  // parse data file
  ParameterTree parameterSet;

  ParameterTreeParser::readINITree(argv[2], parameterSet);

  ParameterTreeParser::readOptions(argc, argv, parameterSet);

  // read solver settings
  int numLevels                         = parameterSet.get<int>("numLevels");
  int numHomotopySteps                  = parameterSet.get<int>("numHomotopySteps");
  const double tolerance                = parameterSet.get<double>("tolerance");
  const int maxSolverSteps              = parameterSet.get<int>("maxSolverSteps");
  const double initialTrustRegionRadius = parameterSet.get<double>("initialTrustRegionRadius");
  const double initialRegularization    = parameterSet.get<double>("initialRegularization");
  const int multigridIterations         = parameterSet.get<int>("numIt");
  const int nu1                         = parameterSet.get<int>("nu1");
  const int nu2                         = parameterSet.get<int>("nu2");
  const int mu                          = parameterSet.get<int>("mu");
  const int baseIterations              = parameterSet.get<int>("baseIt");
  const double mgTolerance              = parameterSet.get<double>("mgTolerance");
  const double baseTolerance            = parameterSet.get<double>("baseTolerance");
  const bool instrumented               = parameterSet.get<bool>("instrumented");
  const bool startFromFile              = parameterSet.get<bool>("startFromFile");
  const std::string resultPath          = parameterSet.get("resultPath", "");

  /////////////////////////////////////////////////////////////
  //                      CREATE THE GRID
  /////////////////////////////////////////////////////////////
  typedef UGGrid<dim> GridType;

  std::shared_ptr<GridType> grid;

  FieldVector<double,dim> lower(0), upper(1);


  if (parameterSet.get<bool>("structuredGrid")) {

    lower = parameterSet.get<FieldVector<double,dim> >("lower");
    upper = parameterSet.get<FieldVector<double,dim> >("upper");

    std::array<unsigned int,dim> elements = parameterSet.get<std::array<unsigned int,dim> >("elements");
    grid = StructuredGridFactory<GridType>::createCubeGrid(lower, upper, elements);

  } else {
    std::string path                = parameterSet.get<std::string>("path");
    std::string gridFile            = parameterSet.get<std::string>("gridFile");
    grid = std::shared_ptr<GridType>(GmshReader<GridType>::read(path + "/" + gridFile));
  }

  grid->setRefinementType(GridType::RefinementType::COPY);

  // Make Python function that computes which vertices are on the Dirichlet boundary,
  // based on the vertex positions.
  std::string lambda = std::string("lambda x: (") + parameterSet.get<std::string>("dirichletVerticesPredicate") + std::string(")");
  auto pythonDirichletVertices = Python::make_function<FieldVector<bool,dim> >(Python::evaluate(lambda));

  // Same for the Neumann boundary
  lambda = std::string("lambda x: (") + parameterSet.get<std::string>("neumannVerticesPredicate", "0") + std::string(")");
  auto pythonNeumannVertices = Python::make_function<bool>(Python::evaluate(lambda));

  // Same for the Surface Shell Boundary
  lambda = std::string("lambda x: (") + parameterSet.get<std::string>("surfaceShellVerticesPredicate", "0") + std::string(")");
  auto pythonSurfaceShellVertices = Python::make_function<bool>(Python::evaluate(lambda));

  // Same for adaptive refinement vertices
  lambda = std::string("lambda x: (") + parameterSet.get<std::string>("adaptiveRefinementVerticesPredicate", "0") + std::string(")");
  auto pythonAdaptiveRefinementVertices = Python::make_function<bool>(Python::evaluate(lambda));

  while (numLevels > 0) {
    for (auto&& e : elements(grid->leafGridView())) {
      bool refineHere = false;
      for (int i = 0; i < e.geometry().corners(); i++) {
        refineHere = refineHere || pythonSurfaceShellVertices(e.geometry().corner(i)) || pythonAdaptiveRefinementVertices(e.geometry().corner(i));
      }
      grid->mark(refineHere ? 1 : 0, e);
    }

    grid->adapt();

    numLevels--;
  }

  grid->loadBalance();

  if (mpiHelper.rank()==0)
    std::cout << "There are " << grid->leafGridView().comm().size() << " processes" << std::endl;

  typedef GridType::LeafGridView GridView;
  GridView gridView = grid->leafGridView();

  /////////////////////////////////////////////////////////////
  //                      DATA TYPES
  /////////////////////////////////////////////////////////////

  using namespace Dune::Indices;

  typedef std::vector<GFE::RealTuple<double,dim> > DisplacementVector;
  typedef std::vector<GFE::Rotation<double,dim> > RotationVector;
  const int dimRotation = GFE::Rotation<double,dim>::embeddedDim;
  typedef TupleVector<DisplacementVector, RotationVector> SolutionType;

  /////////////////////////////////////////////////////////////
  //                      FUNCTION SPACE
  /////////////////////////////////////////////////////////////

  using namespace Dune::Functions::BasisFactory;
  auto compositeBasis = makeBasis(
    gridView,
    composite(
      power<dim>(
        lagrange<displacementOrder>()
        ),
      power<dimRotation>(
        lagrange<rotationOrder>()
        )
      ));

  auto deformationPowerBasis = makeBasis(
    gridView,
    power<dim>(
      lagrange<displacementOrder>()
      ));

  auto orientationPowerBasis = makeBasis(
    gridView,
    power<dim>(
      power<dim>(
        lagrange<rotationOrder>()
        )
      ));

  typedef Dune::Functions::LagrangeBasis<GridView,displacementOrder> DeformationFEBasis;
  typedef Dune::Functions::LagrangeBasis<GridView,rotationOrder> OrientationFEBasis;
  DeformationFEBasis deformationFEBasis(gridView);
  OrientationFEBasis orientationFEBasis(gridView);

  using CompositeBasis = decltype(compositeBasis);

  /////////////////////////////////////////////////////////////
  //                      BOUNDARY DATA
  /////////////////////////////////////////////////////////////

  BitSetVector<1> dirichletVerticesX(gridView.size(dim), false);
  BitSetVector<1> dirichletVerticesY(gridView.size(dim), false);
  BitSetVector<1> dirichletVerticesZ(gridView.size(dim), false);
  BitSetVector<1> neumannVertices(gridView.size(dim), false);
  BitSetVector<1> surfaceShellVertices(gridView.size(dim), false);

  const GridView::IndexSet& indexSet = gridView.indexSet();

  for (auto&& v : vertices(gridView))
  {
    FieldVector<bool,dim> isDirichlet = pythonDirichletVertices(v.geometry().corner(0));
    dirichletVerticesX[indexSet.index(v)] = isDirichlet[0];
    dirichletVerticesY[indexSet.index(v)] = isDirichlet[1];
    dirichletVerticesZ[indexSet.index(v)] = isDirichlet[2];

    bool isNeumann = pythonNeumannVertices(v.geometry().corner(0));
    neumannVertices[indexSet.index(v)] = isNeumann;

    bool isSurfaceShell = pythonSurfaceShellVertices(v.geometry().corner(0));
    surfaceShellVertices[indexSet.index(v)] = isSurfaceShell;
  }

  BoundaryPatch<GridView> dirichletBoundaryX(gridView, dirichletVerticesX);
  BoundaryPatch<GridView> dirichletBoundaryY(gridView, dirichletVerticesY);
  BoundaryPatch<GridView> dirichletBoundaryZ(gridView, dirichletVerticesZ);
  auto neumannBoundary = std::make_shared<BoundaryPatch<GridView> >(gridView, neumannVertices);
  BoundaryPatch<GridView> surfaceShellBoundary(gridView, surfaceShellVertices);

  std::cout << "On rank " << mpiHelper.rank() << ": Dirichlet boundary has [" << dirichletBoundaryX.numFaces() <<
    ", " << dirichletBoundaryY.numFaces() <<
    ", " << dirichletBoundaryZ.numFaces() <<"] faces\n";
  std::cout << "On rank " << mpiHelper.rank() << ": Neumann boundary has " << neumannBoundary->numFaces() << " faces\n";
  std::cout << "On rank " << mpiHelper.rank() << ": Shell boundary has " << surfaceShellBoundary.numFaces() << " faces\n";

  BitSetVector<1> dirichletNodesX(compositeBasis.size({0}),false);
  BitSetVector<1> dirichletNodesY(compositeBasis.size({0}),false);
  BitSetVector<1> dirichletNodesZ(compositeBasis.size({0}),false);
  BitSetVector<1> surfaceShellNodes(compositeBasis.size({1}),false);
#if DUNE_VERSION_GTE(DUNE_FUFEM, 2, 10)
  Fufem::markBoundaryPatchDofs(dirichletBoundaryX,deformationFEBasis,dirichletNodesX);
  Fufem::markBoundaryPatchDofs(dirichletBoundaryY,deformationFEBasis,dirichletNodesY);
  Fufem::markBoundaryPatchDofs(dirichletBoundaryZ,deformationFEBasis,dirichletNodesZ);
  Fufem::markBoundaryPatchDofs(surfaceShellBoundary,orientationFEBasis,surfaceShellNodes);
#else
  constructBoundaryDofs(dirichletBoundaryX,deformationFEBasis,dirichletNodesX);
  constructBoundaryDofs(dirichletBoundaryY,deformationFEBasis,dirichletNodesY);
  constructBoundaryDofs(dirichletBoundaryZ,deformationFEBasis,dirichletNodesZ);
  constructBoundaryDofs(surfaceShellBoundary,orientationFEBasis,surfaceShellNodes);
#endif

  //Create BitVector matching the tangential space
  const int dimRotationTangent = GFE::Rotation<double,dim>::TangentVector::dimension;
  typedef MultiTypeBlockVector<std::vector<FieldVector<double,dim> >, std::vector<FieldVector<double,dimRotationTangent> > > VectorForBit;
  typedef Solvers::DefaultBitVector_t<VectorForBit> BitVector;

  BitVector dirichletDofs;
  dirichletDofs[_0].resize(compositeBasis.size({0}));
  dirichletDofs[_1].resize(compositeBasis.size({1}));
  for (size_t i = 0; i < compositeBasis.size({0}); i++) {
    dirichletDofs[_0][i][0] = dirichletNodesX[i][0];
    dirichletDofs[_0][i][1] = dirichletNodesY[i][0];
    dirichletDofs[_0][i][2] = dirichletNodesZ[i][0];
  }
  for (size_t i = 0; i < compositeBasis.size({1}); i++) {
    for (int j = 0; j < dimRotationTangent; j++) {
      dirichletDofs[_1][i][j] = not surfaceShellNodes[i][0];
    }
  }

  /////////////////////////////////////////////////////////////
  //                      INITIAL DATA
  /////////////////////////////////////////////////////////////

  SolutionType x;
  x[_0].resize(compositeBasis.size({0}));
  x[_1].resize(compositeBasis.size({1}));

  //Initial deformation of the underlying substrate
  BlockVector<FieldVector<double,dim> > displacement(compositeBasis.size({0}));
  lambda = std::string("lambda x: (") + parameterSet.get<std::string>("initialDeformation") + std::string(")");
  auto pythonInitialDeformation = Python::make_function<FieldVector<double,dim> >(Python::evaluate(lambda));
  Dune::Functions::interpolate(deformationPowerBasis, displacement, pythonInitialDeformation);

  BlockVector<FieldVector<double,dim> > identity(compositeBasis.size({0}));
  Dune::Functions::interpolate(deformationPowerBasis, identity, [](FieldVector<double,dim> x){
    return x;
  });

  double max_x = 0;
  double initial_max_x = 0;
  for (std::size_t i = 0; i < displacement.size(); i++) {
    x[_0][i] = displacement[i]; //Copy over the initial deformation to the deformation part of x
    initial_max_x = std::max(x[_0][i][0], initial_max_x);
    displacement[i] -= identity[i]; //Subtract identity to get the initial displacement as a function
  }
  auto displacementFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,dim> >(deformationPowerBasis, displacement);
  //  We need to subsample, because VTK cannot natively display real second-order functions
  SubsamplingVTKWriter<GridView> vtkWriter(gridView, Dune::refinementLevels(displacementOrder-1));
  vtkWriter.addVertexData(displacementFunction, VTK::FieldInfo("displacement", VTK::FieldInfo::Type::scalar, dim));
  vtkWriter.write(resultPath + "finite-strain_homotopy_" + parameterSet.get<std::string>("energy") + "_0");

  /////////////////////////////////////////////////////////////
  //               STRESS-FREE SURFACE SHELL DATA
  /////////////////////////////////////////////////////////////
  auto stressFreeFEBasis = makeBasis(
    gridView,
    power<dim>(
      lagrange<stressFreeDataOrder>(),
      blockedInterleaved()
      ));

  GlobalIndexSet<GridView> globalVertexIndexSet(gridView,dim);
  BlockVector<FieldVector<double,dim> > stressFreeShellVector(stressFreeFEBasis.size());

  if (startFromFile) {
    const std::string pathToGridDeformationFile = parameterSet.get("pathToGridDeformationFile", "");

    std::unordered_map<std::string, FieldVector<double,3> > deformationMap;
    std::string line, displacement, entry;
    if (mpiHelper.rank() == 0)
      std::cout << "Reading in deformation file ("  << "order is "  << stressFreeDataOrder  << "): " << pathToGridDeformationFile + parameterSet.get<std::string>("gridDeformationFile") << std::endl;
    // Read grid deformation information from the file specified in the parameter set via gridDeformationFile

    std::ifstream file(pathToGridDeformationFile + parameterSet.get<std::string>("gridDeformationFile"), std::ios::in);
    if (file.is_open()) {
      while (std::getline(file, line)) {
        size_t j = 0;
        size_t pos = line.find(":");
        displacement = line.substr(pos + 1);
        line.erase(pos);
        std::stringstream entries(displacement);
        FieldVector<double,3> displacementVector(0);
        while(entries >> entry)
          displacementVector[j++] = std::stod(entry);
        deformationMap.insert({line,displacementVector});
      }
      if (mpiHelper.rank() == 0)
        std::cout << "... done: The grid has " << globalVertexIndexSet.size(dim) << " vertices and the defomation file has " << deformationMap.size() << " entries." << std::endl;
      if (stressFreeDataOrder == 1 && deformationMap.size() != globalVertexIndexSet.size(dim))
        DUNE_THROW(Exception, "Error: Grid and deformation vector do not match!");
      file.close();
    } else {
      DUNE_THROW(Exception, "Error: Could not open the file containing the deformation vector!");
    }
    Dune::Functions::interpolate(stressFreeFEBasis, stressFreeShellVector, [](FieldVector<double,dim> x){
      return x;
    });

    for (auto& entry : stressFreeShellVector) {
      std::stringstream stream;
      stream << entry;
      entry += deformationMap.at(stream.str()); //Look up the displacement for this vertex in the deformationMap
    }
  } else {
    // Read grid deformation from deformation function
    auto gridDeformationLambda = std::string("lambda x: (") + parameterSet.get<std::string>("gridDeformation") + std::string(")");
    auto gridDeformation = Python::make_function<FieldVector<double,dim> >(Python::evaluate(gridDeformationLambda));
    Dune::Functions::interpolate(stressFreeFEBasis, stressFreeShellVector, gridDeformation);
  }

  auto stressFreeShellFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,dim> >(stressFreeFEBasis, stressFreeShellVector);

  if (parameterSet.hasKey("writeOutStressFreeData") && parameterSet.get<bool>("writeOutStressFreeData")) {
    BlockVector<FieldVector<double,dim> > stressFreeDisplacement(stressFreeFEBasis.size());
    Dune::Functions::interpolate(stressFreeFEBasis, stressFreeDisplacement, [](FieldVector<double,dim> x){
      return (-1.0)*x;
    });

    for (std::size_t i = 0; i < stressFreeFEBasis.size(); i++) {
      stressFreeDisplacement[i] += stressFreeShellVector[i];
    }
    auto stressFreeDisplacementFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,dim> >(stressFreeFEBasis, stressFreeDisplacement);
    //Write out the stress-free shell function that was read in
    SubsamplingVTKWriter<GridView> vtkWriterStressFree(gridView, Dune::refinementLevels(1));
    vtkWriterStressFree.addVertexData(stressFreeDisplacementFunction, VTK::FieldInfo("displacement", VTK::FieldInfo::Type::scalar, dim));
    vtkWriterStressFree.write("stress-free-shell-function");
  }

  /////////////////////////////////////////////////////////////
  //                      MAIN HOMOTOPY LOOP
  /////////////////////////////////////////////////////////////
  FieldVector<double,dim> neumannValues {0,0,0};
  if (parameterSet.hasKey("neumannValues"))
    neumannValues = parameterSet.get<FieldVector<double,dim> >("neumannValues");
  std::cout << "Neumann values: " << neumannValues << std::endl;

  //Function for the Cosserat material parameters
  const ParameterTree& materialParameters = parameterSet.sub("materialParameters");
  Python::Reference surfaceShellClass = Python::import(materialParameters.get<std::string>("surfaceShellParameters"));
  Python::Callable surfaceShellCallable = surfaceShellClass.get("SurfaceShellParameters");
  Python::Reference pythonObject = surfaceShellCallable();
  auto fThickness = Python::make_function<double>(pythonObject.get("thickness"));
  auto fLame = Python::make_function<FieldVector<double, 2> >(pythonObject.get("lame"));

  for (int i = 0; i < numHomotopySteps; i++)
  {
    double homotopyParameter = (i+1)*(1.0/numHomotopySteps);

    // ////////////////////////////////////////////////////////////
    //   Create an assembler for the energy functional
    // ////////////////////////////////////////////////////////////


    // A constant vector-valued function, for simple Neumann boundary values
    auto neumannFunctionPtr = std::function<Dune::FieldVector<double,dim>(Dune::FieldVector<double,dim>)>([&](FieldVector<double,dim> ) {
      return neumannValues * (-homotopyParameter);
    });

    const ParameterTree& materialParameters = parameterSet.sub("materialParameters");
    if (mpiHelper.rank() == 0)
    {
      std::cout << "Homotopy step: " << i << ",    parameter: " << homotopyParameter << std::endl;
      std::cout << "Material parameters:" << std::endl;
      materialParameters.report();
    }

    std::cout << "Selected energy is: " << parameterSet.get<std::string>("energy") << std::endl;

    // /////////////////////////////////////////////////
    //   Create the energy functional
    // /////////////////////////////////////////////////

    std::shared_ptr<Elasticity::LocalDensity<dim,ValueType> > elasticDensity;
    if (parameterSet.get<std::string>("energy") == "stvenantkirchhoff")
      elasticDensity = std::make_shared<Elasticity::StVenantKirchhoffDensity<dim,ValueType> >(materialParameters);
    if (parameterSet.get<std::string>("energy") == "neohooke")
      elasticDensity = std::make_shared<Elasticity::NeoHookeDensity<dim,ValueType> >(materialParameters);
    if (parameterSet.get<std::string>("energy") == "hencky")
      elasticDensity = std::make_shared<Elasticity::HenckyDensity<dim,ValueType> >(materialParameters);
    if (parameterSet.get<std::string>("energy") == "exphencky")
      elasticDensity = std::make_shared<Elasticity::ExpHenckyDensity<dim,ValueType> >(materialParameters);
    if (parameterSet.get<std::string>("energy") == "mooneyrivlin")
      elasticDensity = std::make_shared<Elasticity::MooneyRivlinDensity<dim,ValueType> >(materialParameters);


    if(!elasticDensity)
      DUNE_THROW(Exception, "Error: Selected energy not available!");

    using ActiveRigidBodyMotion = GFE::ProductManifold<GFE::RealTuple<adouble,dim>, GFE::Rotation<adouble,dim> >;

    // Wrap dune-elasticity density as dune-gfe density
    using Element = GridView::Codim<0>::Entity;
    auto elasticDensityWrapped = std::make_shared<GFE::DuneElasticityDensity<Element,ActiveRigidBodyMotion,0> >(elasticDensity);


    // Select which type of geometric interpolation to use
    using LocalDeformationInterpolationRule = GFE::LocalGeodesicFEFunction<dim, GridType::ctype, decltype(deformationFEBasis.localView().tree().finiteElement()), GFE::RealTuple<adouble,dim> >;
    using LocalOrientationInterpolationRule = GFE::LocalGeodesicFEFunction<dim, GridType::ctype, decltype(orientationFEBasis.localView().tree().finiteElement()), GFE::Rotation<adouble,dim> >;

    using LocalInterpolationRule = std::tuple<LocalDeformationInterpolationRule,LocalOrientationInterpolationRule>;

    auto elasticEnergy = std::make_shared<GFE::LocalIntegralEnergy<CompositeBasis, LocalInterpolationRule, ActiveRigidBodyMotion> >(elasticDensityWrapped);
    auto neumannEnergy = std::make_shared<GFE::NeumannEnergy<CompositeBasis, GFE::RealTuple<ValueType,targetDim>, GFE::Rotation<ValueType,dim> > >(neumannBoundary,neumannFunctionPtr);

    // The energy of the surface shell
    using Intersection = typename GridView::Intersection;
    auto cosseratShellDensity = std::make_shared<GFE::CosseratShellDensity<Intersection, adouble> >(
      materialParameters,
      fThickness,
      fLame);

    auto surfaceCosseratEnergy = std::make_shared<GFE::SurfaceCosseratEnergy<
        decltype(stressFreeShellFunction), CompositeBasis, ActiveRigidBodyMotion> >(
      cosseratShellDensity,
      &surfaceShellBoundary,
      stressFreeShellFunction);

    using RBM = GFE::ProductManifold<GFE::RealTuple<double, dim>,GFE::Rotation<double,dim> >;

    auto sumEnergy = std::make_shared<GFE::SumEnergy<CompositeBasis, GFE::RealTuple<ValueType,targetDim>, GFE::Rotation<ValueType,targetDim> > >();
    sumEnergy->addLocalEnergy(neumannEnergy);
    sumEnergy->addLocalEnergy(elasticEnergy);
    sumEnergy->addLocalEnergy(surfaceCosseratEnergy);

    GFE::LocalGeodesicFEADOLCStiffness<CompositeBasis,RBM> localGFEADOLCStiffness(sumEnergy);
    GFE::MixedGFEAssembler<CompositeBasis,RBM> mixedAssembler(compositeBasis, localGFEADOLCStiffness);

    ////////////////////////////////////////////////////////
    //   Set Dirichlet values
    ////////////////////////////////////////////////////////

    BitSetVector<GFE::RealTuple<double,dim>::TangentVector::dimension> deformationDirichletDofs(dirichletDofs[_0].size(), false);
    for (std::size_t i = 0; i < dirichletDofs[_0].size(); i++)
      for (int j = 0; j < GFE::RealTuple<double,dim>::TangentVector::dimension; j++)
        deformationDirichletDofs[i][j] = dirichletDofs[_0][i][j];
    BitSetVector<GFE::Rotation<double,dim>::TangentVector::dimension> orientationDirichletDofs(dirichletDofs[_1].size(), false);
    for (std::size_t i = 0; i < dirichletDofs[_1].size(); i++)
      for (int j = 0; j < GFE::Rotation<double,dim>::TangentVector::dimension; j++)
        orientationDirichletDofs[i][j] = dirichletDofs[_1][i][j];

    Python::Reference dirichletValuesClass = Python::import(parameterSet.get<std::string>("dirichletValues"));
    Python::Callable C = dirichletValuesClass.get("DirichletValues");

    // Call a constructor.
    Python::Reference dirichletValuesPythonObject = C(homotopyParameter);

    // Extract object member functions as Dune functions
    auto deformationDirichletValues = Python::make_function<FieldVector<double,targetDim> >   (dirichletValuesPythonObject.get("deformation"));
    auto rotationalDirichletValues = Python::make_function<FieldMatrix<double,targetDim,targetDim> > (dirichletValuesPythonObject.get("orientation"));

    BlockVector<FieldVector<double,targetDim> > ddV;
    Dune::Functions::interpolate(deformationPowerBasis, ddV, deformationDirichletValues, deformationDirichletDofs);

    BlockVector<FieldMatrix<double,targetDim,targetDim> > dOV;
    Dune::Functions::interpolate(orientationPowerBasis, dOV, rotationalDirichletValues);

    for (std::size_t i = 0; i < compositeBasis.size({0}); i++) {
      FieldVector<double,3> x0i = x[_0][i].globalCoordinates();
      for (int j=0; j<3; j++)
        if (deformationDirichletDofs[i][j])
          x0i[j] = ddV[i][j];
      x[_0][i] = x0i;
    }
    for (std::size_t i = 0; i < compositeBasis.size({1}); i++)
      if (dirichletDofs[_1][i][0])
        x[_1][i].set(dOV[i]);

#if !MIXED_SPACE
    //The MixedRiemannianTrustRegionSolver can treat the Deformation and Orientation Space as separate ones
    //The RiemannianTrustRegionSolver can only treat the Deformation and Rotation together in a ProductManifold
    //Therefore, x and the dirichletDofs are converted to a ProductManifold structure, as well as the Hessian and Gradient that are returned by the assembler
    std::vector<RBM> xRBM(compositeBasis.size({0}));
    BitSetVector<RBM::TangentVector::dimension> dirichletDofsRBM(compositeBasis.size({0}), false);
    for (std::size_t i = 0; i < compositeBasis.size({0}); i++) {
      for (int j = 0; j < dim; j ++) { // Displacement part
        xRBM[i][_0].globalCoordinates()[j] = x[_0][i][j];
        dirichletDofsRBM[i][j] = dirichletDofs[_0][i][j];
      }
      xRBM[i][_1] = x[_1][i]; // Rotation part
      for (int j = dim; j < RBM::TangentVector::dimension; j ++)
        dirichletDofsRBM[i][j] = dirichletDofs[_1][i][j-dim];
    }
    typedef Dune::GFE::GeodesicFEAssemblerWrapper<CompositeBasis, DeformationFEBasis, RBM> GFEAssemblerWrapper;
    GFEAssemblerWrapper assembler(&mixedAssembler, deformationFEBasis);
#endif

    ///////////////////////////////////////////////////
    //   Create the chosen solver and solve!
    ///////////////////////////////////////////////////

    if (parameterSet.get<std::string>("solvertype", "trustRegion") == "trustRegion") {
#if MIXED_SPACE
      GFE::MixedRiemannianTrustRegionSolver<GridType, CompositeBasis, DeformationFEBasis, GFE::RealTuple<double,dim>, OrientationFEBasis, GFE::Rotation<double,dim> > solver;
      solver.setup(*grid,
                   &mixedAssembler,
                   deformationFEBasis,
                   orientationFEBasis,
                   x,
                   deformationDirichletDofs,
                   orientationDirichletDofs,
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
      solver.setInitialIterate(x);
      solver.solve();
      x = solver.getSol();
#else
      GFE::RiemannianTrustRegionSolver<DeformationFEBasis, RBM, GFEAssemblerWrapper> solver;
      solver.setup(*grid,
                   &assembler,
                   xRBM,
                   dirichletDofsRBM,
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
      solver.setInitialIterate(xRBM);
      solver.solve();
      xRBM = solver.getSol();
      for (std::size_t i = 0; i < xRBM.size(); i++) {
        x[_0][i] = xRBM[i][_0];
        x[_1][i] = xRBM[i][_1];
      }
#endif
    } else { //parameterSet.get<std::string>("solvertype") == "proximalNewton"

#if MIXED_SPACE
      GFE::MixedRiemannianProximalNewtonSolver<CompositeBasis, DeformationFEBasis, GFE::RealTuple<double,dim>, OrientationFEBasis, GFE::Rotation<double,dim>, BitVector> solver;
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
#else
      GFE::RiemannianProximalNewtonSolver<DeformationFEBasis, RBM, GFEAssemblerWrapper> solver;
      solver.setup(*grid,
                   &assembler,
                   xRBM,
                   dirichletDofsRBM,
                   tolerance,
                   maxSolverSteps,
                   initialRegularization,
                   instrumented);
      solver.setScaling(parameterSet.get<FieldVector<double,6> >("solverScaling"));
      solver.setInitialIterate(xRBM);
      solver.solve();
      xRBM = solver.getSol();
      for (std::size_t i = 0; i < xRBM.size(); i++) {
        x[_0][i] = xRBM[i][_0];
        x[_1][i] = xRBM[i][_1];
      }
#endif
    }

    std::cout << "Overall calculation took " << overallTimer.elapsed() << " sec." << std::endl;

    /////////////////////////////////
    //   Output result
    /////////////////////////////////

    // Compute the displacement
    for (std::size_t i = 0; i < compositeBasis.size({0}); i++) {
      for (int j = 0; j  < dim; j++) {
        displacement[i][j] = x[_0][i][j];
      }
      displacement[i] -= identity[i];
    }
    auto displacementFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,dim> >(deformationPowerBasis, displacement);

    //  We need to subsample, because VTK cannot natively display real second-order functions
    SubsamplingVTKWriter<GridView> vtkWriter(gridView, Dune::refinementLevels(displacementOrder-1));
    vtkWriter.addVertexData(displacementFunction, VTK::FieldInfo("displacement", VTK::FieldInfo::Type::scalar, dim));
    vtkWriter.write(resultPath + "finite-strain_homotopy_" + parameterSet.get<std::string>("energy") + "_" + std::to_string(neumannValues[0]) + "_" + std::to_string(i+1));
  }
  for (std::size_t i = 0; i < x[_0].size(); i++) {
    max_x = std::max(x[_0][i][0], max_x);
  }

  if (mpiHelper.rank()==0) {
    std::cout << "Maximal value in x-direction: " << max_x << ", this is a stretch in of " << 100*(max_x - initial_max_x)/initial_max_x << " %." << std::endl;
  }

  std::string ending = grid->leafGridView().comm().size() > 1 ? std::to_string(mpiHelper.rank()) : "";
  std::ofstream file;
  std::string pathToOutput = parameterSet.hasKey("pathToOutput") ?  parameterSet.get<std::string>("pathToOutput") : "./";
  std::string deformationOutput = parameterSet.hasKey("deformationOutput") ?  parameterSet.get<std::string>("deformationOutput") : "deformation";
  std::string rotationOutput = parameterSet.hasKey("rotationOutput") ?  parameterSet.get<std::string>("rotationOutput") : "rotation";

  deformationOutput = pathToOutput + deformationOutput;
  rotationOutput = pathToOutput + rotationOutput;

  file.open(deformationOutput + ending);
  for (std::size_t i = 0; i < identity.size(); i++) {
    file << identity[i] << ":" << displacement[i] << "\n";
  }

  file.close();

  BlockVector<FieldVector<double,dim> > identityRotation(orientationFEBasis.size());
  auto identityRotationPowerBasis = makeBasis(
    gridView,
    power<dim>(
      lagrange<rotationOrder>()
      ));
  Dune::Functions::interpolate(identityRotationPowerBasis, identityRotation, [](FieldVector<double,dim> x){
    return x;
  });

  file.open(rotationOutput + ending);
  for (std::size_t i = 0; i < identityRotation.size(); i++) {
    file << identityRotation[i] << ":" << x[_1][i] << "\n";
  }

  file.close();

  MPI_Barrier(grid->leafGridView().comm());

  if (grid->leafGridView().comm().size() > 1 && mpiHelper.rank() == 0) {
    file.open(deformationOutput);
    for (int i = 0; i < grid->leafGridView().comm().size(); i++) {
      std::ifstream deformationInput(deformationOutput + std::to_string(i));
      if (deformationInput.is_open()) {
        file << deformationInput.rdbuf();
      }
      deformationInput.close();
      std::remove((deformationOutput + std::to_string(i)).c_str());
    }
    file.close();

    file.open(rotationOutput);
    for (int i = 0; i < grid->leafGridView().comm().size(); i++) {
      std::ifstream rotationInput(rotationOutput + std::to_string(i));
      if (rotationInput.is_open()) {
        file << rotationInput.rdbuf();
      }
      rotationInput.close();
      std::remove((rotationOutput + std::to_string(i)).c_str());
    }
    file.close();
  }

}
catch (Exception& e) {
  std::cout << e.what() << std::endl;
}
