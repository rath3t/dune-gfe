#define MIXED_SPACE 0
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

#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/grid/io/file/gmshreader.hh>

#if HAVE_DUNE_FOAMGRID
#include <dune/foamgrid/foamgrid.hh>
#endif

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
#include <dune/gfe/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/mixedlocalgfeadolcstiffness.hh>
#include <dune/gfe/cosseratenergystiffness.hh>
#include <dune/gfe/nonplanarcosseratshellenergy.hh>
#include <dune/gfe/cosseratvtkwriter.hh>
#include <dune/gfe/cosseratvtkreader.hh>
#include <dune/gfe/geodesicfeassembler.hh>
#include <dune/gfe/embeddedglobalgfefunction.hh>
#include <dune/gfe/mixedgfeassembler.hh>

#if MIXED_SPACE
#include <dune/gfe/mixedriemanniantrsolver.hh>
#else
#include <dune/gfe/geodesicfeassemblerwrapper.hh>
#include <dune/gfe/riemannianpnsolver.hh>
#include <dune/gfe/riemanniantrsolver.hh>
#include <dune/gfe/rigidbodymotion.hh>
#endif

#if HAVE_DUNE_VTK
#include <dune/vtk/vtkreader.hh>
#endif


// grid dimension
const int dim = 2;
const int dimworld = 2;

// Order of the approximation space for the displacement
const int displacementOrder = 2;

// Order of the approximation space for the microrotations
const int rotationOrder = 2;

#if !MIXED_SPACE
static_assert(displacementOrder==rotationOrder, "displacement and rotation order do not match!");

// Image space of the geodesic fe functions
using TargetSpace = RigidBodyMotion<double, 3>;
#endif

using namespace Dune;

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
    // Start Python interpreter
    Python::start();
    Python::Reference main = Python::import("__main__");
    Python::run("import math");

    //feenableexcept(FE_INVALID);
    Python::runStream()
        << std::endl << "import sys"
        << std::endl << "import os"
        << std::endl << "sys.path.append(os.getcwd() + '/../../problems/')"
        << std::endl;

    using namespace TypeTree::Indices;
    using SolutionType = TupleVector<std::vector<RealTuple<double,3> >,
                                     std::vector<Rotation<double,3> > >;

    // parse data file
    ParameterTree parameterSet;
    if (argc < 2)
      DUNE_THROW(Exception, "Usage: ./cosserat-continuum <parameter file>");

    ParameterTreeParser::readINITree(argv[1], parameterSet);

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
    std::string resultPath                = parameterSet.get("resultPath", "");

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
        std::string path                = parameterSet.get<std::string>("path");
        std::string gridFile            = parameterSet.get<std::string>("gridFile");

        // Guess the grid file format by looking at the file name suffix
        auto dotPos = gridFile.rfind('.');
        if (dotPos == std::string::npos)
          DUNE_THROW(IOError, "Could not determine grid input file format");
        std::string suffix = gridFile.substr(dotPos, gridFile.length()-dotPos);

        if (suffix == ".msh")
            grid = std::shared_ptr<GridType>(GmshReader<GridType>::read(path + "/" + gridFile));
        else if (suffix == ".vtu" or suffix == ".vtp")
#if HAVE_DUNE_VTK
            grid = VtkReader<GridType>::createGridFromFile(path + "/" + gridFile);
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

    auto orientationPowerBasis = makeBasis(
        gridView,
        power<3>(
            power<3>(
            lagrange<rotationOrder>()
        )
    ));

    typedef Dune::Functions::LagrangeBasis<GridView,displacementOrder> DeformationFEBasis;
    typedef Dune::Functions::LagrangeBasis<GridView,rotationOrder> OrientationFEBasis;

    DeformationFEBasis deformationFEBasis(gridView);
    OrientationFEBasis orientationFEBasis(gridView);


    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////

    BitSetVector<1> dirichletVertices(gridView.size(dim), false);
    BitSetVector<1> neumannVertices(gridView.size(dim), false);

    const GridView::IndexSet& indexSet = gridView.indexSet();

    // Make Python function that computes which vertices are on the Dirichlet boundary,
    // based on the vertex positions.
    std::string lambda = std::string("lambda x: (") + parameterSet.get<std::string>("dirichletVerticesPredicate") + std::string(")");
    auto pythonDirichletVertices = Python::make_function<bool>(Python::evaluate(lambda));

    // Same for the Neumann boundary
    lambda = std::string("lambda x: (") + parameterSet.get<std::string>("neumannVerticesPredicate", "0") + std::string(")");
    auto pythonNeumannVertices = Python::make_function<bool>(Python::evaluate(lambda));

    for (auto&& vertex : vertices(gridView))
    {
        bool isDirichlet = pythonDirichletVertices(vertex.geometry().corner(0));
        dirichletVertices[indexSet.index(vertex)] = isDirichlet;

        bool isNeumann = pythonNeumannVertices(vertex.geometry().corner(0));
        neumannVertices[indexSet.index(vertex)] = isNeumann;
    }

    BoundaryPatch<GridView> dirichletBoundary(gridView, dirichletVertices);
    BoundaryPatch<GridView> neumannBoundary(gridView, neumannVertices);

    std::cout << "On rank " << mpiHelper.rank() << ": Dirichlet boundary has " << dirichletBoundary.numFaces()
              << " faces and " << dirichletVertices.count() << " nodes.\n";
    std::cout << "On rank " << mpiHelper.rank() << ": Neumann boundary has " << neumannBoundary.numFaces() << " faces.\n";
  
    BitSetVector<1> deformationDirichletNodes(deformationFEBasis.size(), false);
    constructBoundaryDofs(dirichletBoundary,deformationFEBasis,deformationDirichletNodes);

    BitSetVector<1> neumannNodes(deformationFEBasis.size(), false);
    constructBoundaryDofs(neumannBoundary,deformationFEBasis,neumannNodes);

    BitSetVector<3> deformationDirichletDofs(deformationFEBasis.size(), false);
    for (size_t i=0; i<deformationFEBasis.size(); i++)
      if (deformationDirichletNodes[i][0])
        for (int j=0; j<3; j++)
          deformationDirichletDofs[i][j] = true;

    BitSetVector<1> orientationDirichletNodes(orientationFEBasis.size(), false);
    constructBoundaryDofs(dirichletBoundary,orientationFEBasis,orientationDirichletNodes);

    BitSetVector<3> orientationDirichletDofs(orientationFEBasis.size(), false);
    for (size_t i=0; i<orientationFEBasis.size(); i++)
      if (orientationDirichletNodes[i][0])
        for (int j=0; j<3; j++)
          orientationDirichletDofs[i][j] = true;

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

        initialIterateGrid = std::shared_ptr<GridType>(GmshReader<GridType>::read(path + "/" + initialIterateGridFilename));
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
        x[_0][i] = vTargetSpace.r;
        x[_1][i] = vTargetSpace.q;
      }
    }
#endif

    ////////////////////////////////////////////////////////
    //   Main homotopy loop
    ////////////////////////////////////////////////////////

    // Output initial iterate (of homotopy loop)
    BlockVector<FieldVector<double,3> > identity(compositeBasis.size({0}));
    Dune::Functions::interpolate(deformationPowerBasis, identity, [&](FieldVector<double,dimworld> x){ return x;});
    BlockVector<FieldVector<double,3> > displacement(compositeBasis.size({0}));

    if (dim == dimworld) {
    CosseratVTKWriter<GridType>::writeMixed<DeformationFEBasis,OrientationFEBasis>(deformationFEBasis,x[_0],
                                                                                   orientationFEBasis,x[_1],
                                                                                   resultPath + "mixed-cosserat_homotopy_0");
    } else {
#if MIXED_SPACE
        for (int i = 0; i < displacement.size(); i++) {
            for (int j = 0; j < 3; j++)
                displacement[i][j] = x[_0][i][j];
            displacement[i] -= identity[i];
        }
        auto displacementFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,3>>(deformationPowerBasis, displacement);
    
        SubsamplingVTKWriter<GridView> vtkWriter(gridView, Dune::refinementLevels(0));
        vtkWriter.addVertexData(displacementFunction, VTK::FieldInfo("displacement", VTK::FieldInfo::Type::scalar, dimworld));
        vtkWriter.write(resultPath + "mixed-cosserat_homotopy_0");
#else
    CosseratVTKWriter<GridType>::write<DeformationFEBasis>(deformationFEBasis, x, resultPath + "cosserat_homotopy_0");
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

        Python::Reference dirichletValuesClass = Python::import(parameterSet.get<std::string>("problem") + "-dirichlet-values");

        Python::Callable C = dirichletValuesClass.get("DirichletValues");

        // Call a constructor.
        Python::Reference dirichletValuesPythonObject = C(homotopyParameter);

        // Extract object member functions as Dune functions
        auto deformationDirichletValues = Python::make_function<FieldVector<double,3> >   (dirichletValuesPythonObject.get("deformation"));
        auto orientationDirichletValues = Python::make_function<FieldMatrix<double,3,3> > (dirichletValuesPythonObject.get("orientation"));
    
        BlockVector<FieldVector<double,3> > ddV;
        Dune::Functions::interpolate(deformationPowerBasis, ddV, deformationDirichletValues, deformationDirichletDofs);
    
        BlockVector<FieldMatrix<double,3,3> > dOV;
        Dune::Functions::interpolate(orientationPowerBasis, dOV, orientationDirichletValues);
    
        for (int i = 0; i < compositeBasis.size({0}); i++) {
          if (deformationDirichletDofs[i][0])
            x[_0][i] = ddV[i];
        }
        for (int i = 0; i < compositeBasis.size({1}); i++)
          if (orientationDirichletDofs[i][0])
            x[_1][i].set(dOV[i]);

        if (dim==dimworld) {
            CosseratEnergyLocalStiffness<CompositeBasis, 3,adouble> localCosseratEnergy(materialParameters,
                                                                                            &neumannBoundary,
                                                                                            neumannFunction,
                                                                                            volumeLoad);
            MixedLocalGFEADOLCStiffness<CompositeBasis,
                                RealTuple<double,3>,
                                Rotation<double,3> > localGFEADOLCStiffness(&localCosseratEnergy, adolcScalarMode);
            MixedGFEAssembler<CompositeBasis,
                      RealTuple<double,3>,
                      Rotation<double,3> > mixedAssembler(compositeBasis, &localGFEADOLCStiffness);
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

            solver.setScaling(parameterSet.get<FieldVector<double,6> >("trustRegionScaling"));

            solver.setInitialIterate(x);
            solver.solve();
            x = solver.getSol();
#else
            //The MixedRiemannianTrustRegionSolver can treat the Displacement and Orientation Space as separate ones
            //The RiemannianTrustRegionSolver can only treat the Displacement and Rotation together in a RigidBodyMotion
            //Therefore, x and the dirichletDofs are converted to a RigidBodyMotion structure, as well as the Hessian and Gradient that are returned by the assembler
            std::vector<TargetSpace> xTargetSpace(compositeBasis.size({0}));
            BitSetVector<TargetSpace::TangentVector::dimension> dirichletDofsTargetSpace(compositeBasis.size({0}), false);
            for (int i = 0; i < compositeBasis.size({0}); i++) {
              for (int j = 0; j < 3; j ++) { // Displacement part
                xTargetSpace[i].r[j] = x[_0][i][j];
                dirichletDofsTargetSpace[i][j] = deformationDirichletDofs[i][j];
              }
              xTargetSpace[i].q = x[_1][i]; // Rotation part
              for (int j = 3; j < TargetSpace::TangentVector::dimension; j ++)
                dirichletDofsTargetSpace[i][j] = orientationDirichletDofs[i][j-3];
            }

            using GFEAssemblerWrapper = Dune::GFE::GeodesicFEAssemblerWrapper<CompositeBasis, DeformationFEBasis, TargetSpace, RealTuple<double, 3>, Rotation<double,3>>;
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
    
                solver.setScaling(parameterSet.get<FieldVector<double,6> >("trustRegionScaling"));
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
                solver.setInitialIterate(xTargetSpace);
                solver.solve();
                xTargetSpace = solver.getSol();
            }
            for (int i = 0; i < xTargetSpace.size(); i++) {
              x[_0][i] = xTargetSpace[i].r;
              x[_1][i] = xTargetSpace[i].q;
            }
#endif
        } else {
#if MIXED_SPACE
            DUNE_THROW(Exception, "Problems with dim != dimworld do not work for MIXED_SPACE = 1!");
#else
            std::shared_ptr<LocalGeodesicFEADOLCStiffness<DeformationFEBasis, TargetSpace>> localGFEADOLCStiffness;

            NonplanarCosseratShellEnergy<DeformationFEBasis, 3, adouble> localCosseratEnergyPlanar(materialParameters,
                                                                                                    nullptr,
                                                                                                    &neumannBoundary,
                                                                                                    neumannFunction,
                                                                                                    volumeLoad);
              localGFEADOLCStiffness = std::make_shared<LocalGeodesicFEADOLCStiffness<DeformationFEBasis, TargetSpace>>(&localCosseratEnergyPlanar, adolcScalarMode);

            GeodesicFEAssembler<DeformationFEBasis,TargetSpace> assembler(gridView, *localGFEADOLCStiffness);
            //The MixedRiemannianTrustRegionSolver can treat the Displacement and Orientation Space as separate ones
            //The RiemannianTrustRegionSolver can only treat the Displacement and Rotation together in a RigidBodyMotion
            //Therefore, x and the dirichletDofs are converted to a RigidBodyMotion structure, as well as the Hessian and Gradient that are returned by the assembler
            std::vector<TargetSpace> xTargetSpace(compositeBasis.size({0}));
            BitSetVector<TargetSpace::TangentVector::dimension> dirichletDofsTargetSpace(compositeBasis.size({0}), false);
            for (int i = 0; i < compositeBasis.size({0}); i++) {
              for (int j = 0; j < 3; j ++) { // Displacement part
                xTargetSpace[i].r[j] = x[_0][i][j];
                dirichletDofsTargetSpace[i][j] = deformationDirichletDofs[i][j];
              }
              xTargetSpace[i].q = x[_1][i]; // Rotation part
              for (int j = 3; j < TargetSpace::TangentVector::dimension; j ++)
                dirichletDofsTargetSpace[i][j] = orientationDirichletDofs[i][j-3];
            }
            if (parameterSet.get<std::string>("solvertype", "trustRegion") == "trustRegion") {

                RiemannianTrustRegionSolver<DeformationFEBasis, TargetSpace> solver;
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

                solver.setScaling(parameterSet.get<FieldVector<double,6> >("trustRegionScaling"));
                solver.setInitialIterate(xTargetSpace);
                solver.solve();
                xTargetSpace = solver.getSol();
            } else { //parameterSet.get<std::string>("solvertype") == "proximalNewton"
                RiemannianProximalNewtonSolver<DeformationFEBasis, TargetSpace> solver;
                solver.setup(*grid,
                             &assembler,
                             xTargetSpace,
                             dirichletDofsTargetSpace,
                             tolerance,
                             maxSolverSteps,
                             initialRegularization,
                             instrumented);
                solver.setInitialIterate(xTargetSpace);
                solver.solve();
                xTargetSpace = solver.getSol();
            }

            for (int i = 0; i < xTargetSpace.size(); i++) {
              x[_0][i] = xTargetSpace[i].r;
              x[_1][i] = xTargetSpace[i].q;
            }
#endif
        }
        // Output result of each homotopy step
        std::stringstream iAsAscii;
        iAsAscii << i+1;
        if (dim == dimworld) {
        CosseratVTKWriter<GridType>::writeMixed<DeformationFEBasis,OrientationFEBasis>(deformationFEBasis,x[_0],
                                                                                       orientationFEBasis,x[_1],
                                                                                       resultPath + "mixed-cosserat_homotopy_" + iAsAscii.str());
        } else {
#if MIXED_SPACE
            for (int i = 0; i < displacement.size(); i++) {
               for (int j = 0; j  < 3; j++) {
                displacement[i][j] = x[_0][i][j];
              }
              displacement[i] -= identity[i];
            }
            auto displacementFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,3>>(deformationPowerBasis, displacement);
    
            //  We need to subsample, because VTK cannot natively display real second-order functions
            SubsamplingVTKWriter<GridView> vtkWriter(gridView, Dune::refinementLevels(displacementOrder-1));
            vtkWriter.addVertexData(displacementFunction, VTK::FieldInfo("displacement", VTK::FieldInfo::Type::scalar, 3));
            vtkWriter.write(resultPath + "cosserat_homotopy_" + std::to_string(i+1));
#else 
        CosseratVTKWriter<GridType>::write<DeformationFEBasis>(deformationFEBasis, x, resultPath + "cosserat_homotopy_" + std::to_string(i+1));
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

} catch (Exception& e)
{
    std::cout << e.what() << std::endl;
}
