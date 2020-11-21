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
#include <dune/fufem/functionspacebases/dunefunctionsbasis.hh>
#include <dune/fufem/dunepython.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/mixedlocalgfeadolcstiffness.hh>
#include <dune/gfe/cosseratenergystiffness.hh>
#include <dune/gfe/nonplanarcosseratshellenergy.hh>
#include <dune/gfe/cosseratvtkwriter.hh>
#include <dune/gfe/cosseratvtkreader.hh>
#include <dune/gfe/geodesicfeassembler.hh>
#include <dune/gfe/riemanniantrsolver.hh>
#include <dune/gfe/embeddedglobalgfefunction.hh>
#include <dune/gfe/mixedgfeassembler.hh>
#include <dune/gfe/mixedriemanniantrsolver.hh>

#if HAVE_DUNE_VTK
#include <dune/vtk/vtkreader.hh>
#endif


// grid dimension
const int dim = 2;
const int dimworld = 2;

//#define MIXED_SPACE

// Order of the approximation space for the displacement
const int displacementOrder = 2;

// Order of the approximation space for the microrotations
const int rotationOrder = 2;

#ifndef MIXED_SPACE
static_assert(displacementOrder==rotationOrder, "displacement and rotation order do not match!");

// Image space of the geodesic fe functions
typedef RigidBodyMotion<double,3> TargetSpace;

// Tangent vector of the image space
const int blocksize = TargetSpace::TangentVector::dimension;
#endif

using namespace Dune;

int main (int argc, char *argv[]) try
{
    // initialize MPI, finalize is done automatically on exit
    Dune::MPIHelper& mpiHelper = MPIHelper::instance(argc, argv);

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
#ifdef MIXED_SPACE
    using SolutionType = TupleVector<std::vector<RealTuple<double,3> >,
                                     std::vector<Rotation<double,3> > >;
#else
    typedef std::vector<TargetSpace> SolutionType;
#endif

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
    const int maxTrustRegionSteps         = parameterSet.get<int>("maxTrustRegionSteps");
    const double initialTrustRegionRadius = parameterSet.get<double>("initialTrustRegionRadius");
    const int multigridIterations         = parameterSet.get<int>("numIt");
    const int nu1                         = parameterSet.get<int>("nu1");
    const int nu2                         = parameterSet.get<int>("nu2");
    const int mu                          = parameterSet.get<int>("mu");
    const int baseIterations              = parameterSet.get<int>("baseIt");
    const double mgTolerance              = parameterSet.get<double>("mgTolerance");
    const double baseTolerance            = parameterSet.get<double>("baseTolerance");
    const bool instrumented               = parameterSet.get<bool>("instrumented");
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
#ifdef MIXED_SPACE
    const int dimRotation = Rotation<double,dim>::embeddedDim;
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

    typedef Dune::Functions::LagrangeBasis<GridView,displacementOrder> DeformationFEBasis;
    typedef Dune::Functions::LagrangeBasis<GridView,rotationOrder> OrientationFEBasis;

    DeformationFEBasis deformationFEBasis(gridView);
    OrientationFEBasis orientationFEBasis(gridView);

    // Construct fufem-style function space bases to ease the transition to dune-functions
    typedef DuneFunctionsBasis<DeformationFEBasis> FufemDeformationFEBasis;
    FufemDeformationFEBasis fufemDeformationFEBasis(deformationFEBasis);

    typedef DuneFunctionsBasis<OrientationFEBasis> FufemOrientationFEBasis;
    FufemOrientationFEBasis fufemOrientationFEBasis(orientationFEBasis);
#else
    typedef Dune::Functions::LagrangeBasis<typename GridType::LeafGridView, displacementOrder> FEBasis;
    FEBasis feBasis(gridView);

    typedef DuneFunctionsBasis<FEBasis> FufemFEBasis;
    FufemFEBasis fufemFeBasis(feBasis);
#endif

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

    if (mpiHelper.rank()==0)
      std::cout << "Neumann boundary has " << neumannBoundary.numFaces() << " faces\n";

#ifdef MIXED_SPACE
    BitSetVector<1> deformationDirichletNodes(deformationFEBasis.size(), false);
    constructBoundaryDofs(dirichletBoundary,fufemDeformationFEBasis,deformationDirichletNodes);

    BitSetVector<1> neumannNodes(deformationFEBasis.size(), false);
    constructBoundaryDofs(neumannBoundary,fufemDeformationFEBasis,neumannNodes);

    BitSetVector<3> deformationDirichletDofs(deformationFEBasis.size(), false);
    for (size_t i=0; i<deformationFEBasis.size(); i++)
      if (deformationDirichletNodes[i][0])
        for (int j=0; j<3; j++)
          deformationDirichletDofs[i][j] = true;

    BitSetVector<1> orientationDirichletNodes(orientationFEBasis.size(), false);
    constructBoundaryDofs(dirichletBoundary,fufemOrientationFEBasis,orientationDirichletNodes);

    BitSetVector<3> orientationDirichletDofs(orientationFEBasis.size(), false);
    for (size_t i=0; i<orientationFEBasis.size(); i++)
      if (orientationDirichletNodes[i][0])
        for (int j=0; j<3; j++)
          orientationDirichletDofs[i][j] = true;
#else
    BitSetVector<1> dirichletNodes(feBasis.size(), false);
#if DUNE_VERSION_LT(DUNE_GEOMETRY, 2, 7)
    constructBoundaryDofs(dirichletBoundary,fufemFeBasis,dirichletNodes);
#else
    constructBoundaryDofs(dirichletBoundary,feBasis,dirichletNodes);
#endif

    BitSetVector<1> neumannNodes(feBasis.size(), false);
#if DUNE_VERSION_LT(DUNE_GEOMETRY, 2, 7)
    constructBoundaryDofs(neumannBoundary,fufemFeBasis,neumannNodes);
#else
    constructBoundaryDofs(neumannBoundary,feBasis,neumannNodes);
#endif

    BitSetVector<blocksize> dirichletDofs(feBasis.size(), false);
    for (size_t i=0; i<feBasis.size(); i++)
      if (dirichletNodes[i][0])
        for (int j=0; j<5; j++)
          dirichletDofs[i][j] = true;
#endif

    // //////////////////////////
    //   Initial iterate
    // //////////////////////////

#ifdef MIXED_SPACE
    SolutionType x;

    x[_0].resize(deformationFEBasis.size());

    lambda = std::string("lambda x: (") + parameterSet.get<std::string>("initialDeformation") + std::string(")");
    auto pythonInitialDeformation = Python::make_function<FieldVector<double,3> >(Python::evaluate(lambda));

    std::vector<FieldVector<double,3> > v;
    Functions::interpolate(deformationFEBasis, v, pythonInitialDeformation);

    for (size_t i=0; i<x[_0].size(); i++)
      x[_0][i] = v[i];

    x[_1].resize(orientationFEBasis.size());
#else
    SolutionType x(feBasis.size());

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

      SolutionType initialIterate;
      GFE::CosseratVTKReader::read(initialIterate, parameterSet.get<std::string>("initialIterateFilename"));

      typedef Dune::Functions::LagrangeBasis<typename GridType::LeafGridView, 2> InitialBasis;
      InitialBasis initialBasis(initialIterateGrid->leafGridView());

#ifdef PROJECTED_INTERPOLATION
      using LocalInterpolationRule  = LocalProjectedFEFunction<dim, double, FEBasis::LocalView::Tree::FiniteElement, TargetSpace>;
#else
      using LocalInterpolationRule  = LocalGeodesicFEFunction<dim, double, FEBasis::LocalView::Tree::FiniteElement, TargetSpace>;
#endif
      GFE::EmbeddedGlobalGFEFunction<InitialBasis,LocalInterpolationRule,TargetSpace> initialFunction(initialBasis,initialIterate);

      std::vector<FieldVector<double,7> > v;
      Dune::Functions::interpolate(feBasis,v,initialFunction);
      DUNE_THROW(NotImplemented, "Replace scalar basis by power basis!");

      for (size_t i=0; i<x.size(); i++)
        x[i] = TargetSpace(v[i]);

    } else {
    lambda = std::string("lambda x: (") + parameterSet.get<std::string>("initialDeformation") + std::string(")");
      auto pythonInitialDeformation = Python::make_function<FieldVector<double,3> >(Python::evaluate(lambda));

    std::vector<FieldVector<double,3> > v;
      Functions::interpolate(feBasis, v, pythonInitialDeformation);

    for (size_t i=0; i<x.size(); i++)
      x[i].r = v[i];
    }
#endif

    ////////////////////////////////////////////////////////
    //   Main homotopy loop
    ////////////////////////////////////////////////////////

    // Output initial iterate (of homotopy loop)
#ifdef MIXED_SPACE
    CosseratVTKWriter<GridType>::writeMixed<DeformationFEBasis,OrientationFEBasis>(deformationFEBasis,x[_0],
                                                                                   orientationFEBasis,x[_1],
                                                                                   resultPath + "mixed-cosserat_homotopy_0");
#else
    CosseratVTKWriter<GridType>::write<FEBasis>(feBasis,x, resultPath + "cosserat_homotopy_0");
#endif

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

    auto neumannFunction = [&]( FieldVector<double,dim> ) {
      auto nV = neumannValues;
      nV *= homotopyParameter;
      return nV;
    };

    FieldVector<double,3> volumeLoadValues {0,0,0};
    if (parameterSet.hasKey("volumeLoad"))
        volumeLoadValues = parameterSet.get<FieldVector<double,3> >("volumeLoad");

    auto volumeLoad = [&]( FieldVector<double,dim>) {
      auto vL = volumeLoadValues;
      vL *= homotopyParameter;
      return vL;
    };

    if (mpiHelper.rank() == 0) {
        std::cout << "Material parameters:" << std::endl;
        materialParameters.report();
    }

    // Assembler using ADOL-C
#ifdef MIXED_SPACE
    CosseratEnergyLocalStiffness<decltype(compositeBasis),
                        3,adouble> cosseratEnergyADOLCLocalStiffness(materialParameters,
                                                                     &neumannBoundary,
                                                                     neumannFunction,
                                                                     volumeLoad);

    MixedLocalGFEADOLCStiffness<decltype(compositeBasis),
                                RealTuple<double,3>,
                                Rotation<double,3> > localGFEADOLCStiffness(&cosseratEnergyADOLCLocalStiffness);

    MixedGFEAssembler<decltype(compositeBasis),
                      RealTuple<double,3>,
                      Rotation<double,3> > assembler(compositeBasis, &localGFEADOLCStiffness);
#else
    std::shared_ptr<GFE::LocalEnergy<FEBasis,RigidBodyMotion<adouble,3> > > localCosseratEnergy;

    if (dim==dimworld)
    {
      localCosseratEnergy = std::make_shared<CosseratEnergyLocalStiffness<FEBasis,3,adouble> >(materialParameters,
                                                                                               &neumannBoundary,
                                                                                               neumannFunction,
                                                                                               volumeLoad);
    }
    else
    {
      localCosseratEnergy = std::make_shared<NonplanarCosseratShellEnergy<FEBasis,3,adouble> >(materialParameters,
                                                                                               &neumannBoundary,
                                                                                               neumannFunction,
                                                                                               volumeLoad);
    }

    LocalGeodesicFEADOLCStiffness<FEBasis,
                                  TargetSpace> localGFEADOLCStiffness(localCosseratEnergy.get());

    GeodesicFEAssembler<FEBasis,TargetSpace> assembler(gridView, localGFEADOLCStiffness);
#endif

    // /////////////////////////////////////////////////
    //   Create a Riemannian trust-region solver
    // /////////////////////////////////////////////////

#ifdef MIXED_SPACE
    MixedRiemannianTrustRegionSolver<GridType,
                                     decltype(compositeBasis),
                                     DeformationFEBasis, RealTuple<double,3>,
                                     OrientationFEBasis, Rotation<double,3> > solver;
#else
    RiemannianTrustRegionSolver<FEBasis,TargetSpace> solver;
#endif
    solver.setup(*grid,
                 &assembler,
#ifdef MIXED_SPACE
                 deformationFEBasis,
                 orientationFEBasis,
#endif
                 x,
#ifdef MIXED_SPACE
                 deformationDirichletDofs,
                 orientationDirichletDofs,
#else
                 dirichletDofs,
#endif
                 tolerance,
                 maxTrustRegionSteps,
                 initialTrustRegionRadius,
                 multigridIterations,
                 mgTolerance,
                 mu, nu1, nu2,
                 baseIterations,
                 baseTolerance,
                 instrumented);

        solver.setScaling(parameterSet.get<FieldVector<double,6> >("trustRegionScaling"));

        ////////////////////////////////////////////////////////
        //   Set Dirichlet values
        ////////////////////////////////////////////////////////

        Python::Reference dirichletValuesClass = Python::import(parameterSet.get<std::string>("problem") + "-dirichlet-values");

        Python::Callable C = dirichletValuesClass.get("DirichletValues");

        // Call a constructor.
        Python::Reference dirichletValuesPythonObject = C(homotopyParameter);

        // Extract object member functions as Dune functions
        auto deformationDirichletValues = Python::make_function<FieldVector<double,3> >(dirichletValuesPythonObject.get("deformation"));
        auto orientationDirichletValues = Python::make_function<FieldMatrix<double,3,3> >(dirichletValuesPythonObject.get("orientation"));

        std::vector<FieldVector<double,3> > ddV;
        std::vector<FieldMatrix<double,3,3> > dOV;

#ifdef MIXED_SPACE
        Functions::interpolate(deformationFEBasis, ddV, deformationDirichletValues, deformationDirichletDofs);
        Functions::interpolate(orientationFEBasis, dOV, orientationDirichletValues, orientationDirichletDofs);

        for (size_t j=0; j<x[_0].size(); j++)
          if (deformationDirichletNodes[j][0])
            x[_0][j] = ddV[j];

        for (size_t j=0; j<x[_1].size(); j++)
          if (orientationDirichletNodes[j][0])
            x[_1][j].set(dOV[j]);
#else
        Functions::interpolate(feBasis, ddV, deformationDirichletValues, dirichletDofs);
        Functions::interpolate(feBasis, dOV, orientationDirichletValues, dirichletDofs);

        for (size_t j=0; j<x.size(); j++)
          if (dirichletNodes[j][0])
          {
            x[j].r = ddV[j];
            x[j].q.set(dOV[j]);
          }
#endif

        // /////////////////////////////////////////////////////
        //   Solve!
        // /////////////////////////////////////////////////////

        solver.setInitialIterate(x);
        solver.solve();

        x = solver.getSol();

        // Output result of each homotopy step
        std::stringstream iAsAscii;
        iAsAscii << i+1;
#ifdef MIXED_SPACE
        CosseratVTKWriter<GridType>::writeMixed<DeformationFEBasis,OrientationFEBasis>(deformationFEBasis,x[_0],
                                                                                       orientationFEBasis,x[_1],
                                                                                       resultPath + "mixed-cosserat_homotopy_" + iAsAscii.str());
#else
        CosseratVTKWriter<GridType>::write<FEBasis>(feBasis,x, resultPath + "cosserat_homotopy_" + iAsAscii.str());
#endif

    }

    // //////////////////////////////
    //   Output result
    // //////////////////////////////

#ifndef MIXED_SPACE
    // Write the corresponding coefficient vector: verbatim in binary, to be completely lossless
    // This data may be used by other applications measuring the discretization error
    BlockVector<TargetSpace::CoordinateType> xEmbedded(x.size());
    for (size_t i=0; i<x.size(); i++)
      xEmbedded[i] = x[i].globalCoordinates();

    std::ofstream outFile("cosserat-continuum-result-" + std::to_string(numLevels) + ".data", std::ios_base::binary);
    MatrixVector::Generic::writeBinary(outFile, xEmbedded);
    outFile.close();
#endif

    // finally: compute the average deformation of the Neumann boundary
    // That is what we need for the locking tests
    FieldVector<double,3> averageDef(0);
#ifdef MIXED_SPACE
    for (size_t i=0; i<x[_0].size(); i++)
        if (neumannNodes[i][0])
            averageDef += x[_0][i].globalCoordinates();
#else
    for (size_t i=0; i<x.size(); i++)
        if (neumannNodes[i][0])
            averageDef += x[i].r;
#endif
    averageDef /= neumannNodes.count();

    if (mpiHelper.rank()==0 and parameterSet.hasKey("neumannValues"))
    {
      std::cout << "Neumann values = " << parameterSet.get<FieldVector<double, 3> >("neumannValues") << "  "
                << ",  average deflection: " << averageDef << std::endl;
    }

} catch (Exception& e)
{
    std::cout << e.what() << std::endl;
}
