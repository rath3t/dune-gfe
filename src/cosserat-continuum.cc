#include <config.h>

#include <fenv.h>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>    // use of "Easy to Use" drivers
#include <adolc/taping.h>

#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/grid/io/file/gmshreader.hh>

#if HAVE_DUNE_FOAMGRID
#include <dune/foamgrid/foamgrid.hh>
#endif

#include <dune/functions/functionspacebases/pqknodalbasis.hh>

#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/functiontools/boundarydofs.hh>
#include <dune/fufem/functiontools/basisinterpolator.hh>
#include <dune/fufem/functionspacebases/dunefunctionsbasis.hh>
#include <dune/fufem/dunepython.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/cosseratenergystiffness.hh>
#include <dune/gfe/cosseratvtkwriter.hh>
#include <dune/gfe/cosseratvtkreader.hh>
#include <dune/gfe/geodesicfeassembler.hh>
#include <dune/gfe/riemanniantrsolver.hh>

// grid dimension
const int dim = 2;
const int dimworld = 2;

// Order of the approximation space
const int order = 2;

// Image space of the geodesic fe functions
typedef RigidBodyMotion<double,3> TargetSpace;

// Tangent vector of the image space
const int blocksize = TargetSpace::TangentVector::dimension;

using namespace Dune;

/** \brief A constant vector-valued function, for simple Neumann boundary values */
struct NeumannFunction
    : public Dune::VirtualFunction<FieldVector<double,dimworld>, FieldVector<double,3> >
{
    NeumannFunction(const FieldVector<double,3> values,
                    double homotopyParameter)
    : values_(values),
      homotopyParameter_(homotopyParameter)
    {}

    void evaluate(const FieldVector<double, dimworld>& x, FieldVector<double,3>& out) const {
        out = 0;
        out.axpy(homotopyParameter_, values_);
    }

    FieldVector<double,3> values_;
    double homotopyParameter_;
};

/** \brief A constant vector-valued function, for simple volume loads */
struct VolumeLoad
    : public Dune::VirtualFunction<FieldVector<double,dimworld>, FieldVector<double,3> >
{
    VolumeLoad(const FieldVector<double,3> values,
               double homotopyParameter)
    : values_(values),
      homotopyParameter_(homotopyParameter)
    {}

    void evaluate(const FieldVector<double, dimworld>& x, FieldVector<double,3>& out) const {
        out = 0;
        out.axpy(homotopyParameter_, values_);
    }

    FieldVector<double,3> values_;
    double homotopyParameter_;
};


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
        << std::endl << "sys.path.append('/home/sander/dune/dune-gfe/problems/')"
        << std::endl;

    typedef std::vector<TargetSpace> SolutionType;

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

    shared_ptr<GridType> grid;

    FieldVector<double,dimworld> lower(0), upper(1);

    if (parameterSet.get<bool>("structuredGrid")) {

        lower = parameterSet.get<FieldVector<double,dimworld> >("lower");
        upper = parameterSet.get<FieldVector<double,dimworld> >("upper");

        array<unsigned int,dim> elements = parameterSet.get<array<unsigned int,dim> >("elements");
        grid = StructuredGridFactory<GridType>::createCubeGrid(lower, upper, elements);

    } else {
        std::string path                = parameterSet.get<std::string>("path");
        std::string gridFile            = parameterSet.get<std::string>("gridFile");
        grid = shared_ptr<GridType>(GmshReader<GridType>::read(path + "/" + gridFile));
    }

    grid->globalRefine(numLevels-1);

    grid->loadBalance();

    if (mpiHelper.rank()==0)
      std::cout << "There are " << grid->leafGridView().comm().size() << " processes" << std::endl;

    typedef GridType::LeafGridView GridView;
    GridView gridView = grid->leafGridView();

    typedef Dune::Functions::PQkNodalBasis<typename GridType::LeafGridView, order> FEBasis;
    FEBasis feBasis(gridView);

    typedef DuneFunctionsBasis<FEBasis> FufemFEBasis;
    FufemFEBasis fufemFeBasis(feBasis);

    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////

    BitSetVector<1> dirichletVertices(gridView.size(dim), false);
    BitSetVector<1> neumannVertices(gridView.size(dim), false);

    GridType::Codim<dim>::LeafIterator vIt    = gridView.begin<dim>();
    GridType::Codim<dim>::LeafIterator vEndIt = gridView.end<dim>();

    const GridView::IndexSet& indexSet = gridView.indexSet();

    // Make Python function that computes which vertices are on the Dirichlet boundary,
    // based on the vertex positions.
    std::string lambda = std::string("lambda x: (") + parameterSet.get<std::string>("dirichletVerticesPredicate") + std::string(")");
    PythonFunction<FieldVector<double,dimworld>, bool> pythonDirichletVertices(Python::evaluate(lambda));

    // Same for the Neumann boundary
    lambda = std::string("lambda x: (") + parameterSet.get<std::string>("neumannVerticesPredicate", "0") + std::string(")");
    PythonFunction<FieldVector<double,dimworld>, bool> pythonNeumannVertices(Python::evaluate(lambda));

    for (; vIt!=vEndIt; ++vIt) {

        bool isDirichlet;
        pythonDirichletVertices.evaluate(vIt->geometry().corner(0), isDirichlet);
        dirichletVertices[indexSet.index(*vIt)] = isDirichlet;

        bool isNeumann;
        pythonNeumannVertices.evaluate(vIt->geometry().corner(0), isNeumann);
        neumannVertices[indexSet.index(*vIt)] = isNeumann;

    }

    BoundaryPatch<GridView> dirichletBoundary(gridView, dirichletVertices);
    BoundaryPatch<GridView> neumannBoundary(gridView, neumannVertices);

    if (mpiHelper.rank()==0)
      std::cout << "Neumann boundary has " << neumannBoundary.numFaces() << " faces\n";


    BitSetVector<1> dirichletNodes(feBasis.indexSet().size(), false);
    constructBoundaryDofs(dirichletBoundary,fufemFeBasis,dirichletNodes);

    BitSetVector<1> neumannNodes(feBasis.indexSet().size(), false);
    constructBoundaryDofs(neumannBoundary,fufemFeBasis,neumannNodes);

    BitSetVector<blocksize> dirichletDofs(feBasis.indexSet().size(), false);
    for (size_t i=0; i<feBasis.indexSet().size(); i++)
      if (dirichletNodes[i][0])
        for (int j=0; j<5; j++)
          dirichletDofs[i][j] = true;

    // //////////////////////////
    //   Initial iterate
    // //////////////////////////

    SolutionType x(feBasis.indexSet().size());

    if (parameterSet.hasKey("startFromFile"))
    {
      GFE::CosseratVTKReader::read(x, parameterSet.get<std::string>("initialIterateFilename"));
    } else {
    lambda = std::string("lambda x: (") + parameterSet.get<std::string>("initialDeformation") + std::string(")");
    PythonFunction<FieldVector<double,dimworld>, FieldVector<double,3> > pythonInitialDeformation(Python::evaluate(lambda));

    std::vector<FieldVector<double,3> > v;
      ::Functions::interpolate(fufemFeBasis, v, pythonInitialDeformation);

    for (size_t i=0; i<x.size(); i++)
      x[i].r = v[i];
    }

    ////////////////////////////////////////////////////////
    //   Main homotopy loop
    ////////////////////////////////////////////////////////

    // Output initial iterate (of homotopy loop)
    CosseratVTKWriter<GridType>::write<FEBasis>(feBasis,x, resultPath + "cosserat_homotopy_0");

    for (int i=0; i<numHomotopySteps; i++) {

        double homotopyParameter = (i+1)*(1.0/numHomotopySteps);
        if (mpiHelper.rank()==0)
            std::cout << "Homotopy step: " << i << ",    parameter: " << homotopyParameter << std::endl;


    // ////////////////////////////////////////////////////////////
    //   Create an assembler for the energy functional
    // ////////////////////////////////////////////////////////////

    const ParameterTree& materialParameters = parameterSet.sub("materialParameters");
    shared_ptr<NeumannFunction> neumannFunction;
    if (parameterSet.hasKey("neumannValues"))
        neumannFunction = make_shared<NeumannFunction>(parameterSet.get<FieldVector<double,3> >("neumannValues"),
                                                       homotopyParameter);

        shared_ptr<VolumeLoad> volumeLoad;
        if (parameterSet.hasKey("volumeLoad"))
            volumeLoad = make_shared<VolumeLoad>(parameterSet.get<FieldVector<double,3> >("volumeLoad"),
                                                                                          homotopyParameter);

        if (mpiHelper.rank() == 0) {
            std::cout << "Material parameters:" << std::endl;
            materialParameters.report();
        }

    // Assembler using ADOL-C
    CosseratEnergyLocalStiffness<FEBasis,
                                 3,adouble> cosseratEnergyADOLCLocalStiffness(materialParameters,
                                                                              &neumannBoundary,
                                                                              neumannFunction,
                                                                              volumeLoad);
    LocalGeodesicFEADOLCStiffness<FEBasis,
                                  TargetSpace> localGFEADOLCStiffness(&cosseratEnergyADOLCLocalStiffness);

    GeodesicFEAssembler<FEBasis,TargetSpace> assembler(gridView, &localGFEADOLCStiffness);

    // /////////////////////////////////////////////////
    //   Create a Riemannian trust-region solver
    // /////////////////////////////////////////////////

    RiemannianTrustRegionSolver<FEBasis,TargetSpace> solver;
    solver.setup(*grid,
                 &assembler,
                 x,
                 dirichletDofs,
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
        PythonFunction<FieldVector<double,dimworld>, FieldVector<double,3> >   deformationDirichletValues(dirichletValuesPythonObject.get("deformation"));
        PythonFunction<FieldVector<double,dimworld>, FieldMatrix<double,3,3> > orientationDirichletValues(dirichletValuesPythonObject.get("orientation"));

        std::vector<FieldVector<double,3> > ddV;
        ::Functions::interpolate(fufemFeBasis, ddV, deformationDirichletValues, dirichletDofs);

        std::vector<FieldMatrix<double,3,3> > dOV;
        ::Functions::interpolate(fufemFeBasis, dOV, orientationDirichletValues, dirichletDofs);

        for (size_t j=0; j<x.size(); j++)
          if (dirichletNodes[j][0])
          {
            x[j].r = ddV[j];
            x[j].q.set(dOV[j]);
          }

        // /////////////////////////////////////////////////////
        //   Solve!
        // /////////////////////////////////////////////////////

        solver.setInitialIterate(x);
        solver.solve();

        x = solver.getSol();

        // Output result of each homotopy step
        std::stringstream iAsAscii;
        iAsAscii << i+1;
        CosseratVTKWriter<GridType>::write<FEBasis>(feBasis,x, resultPath + "cosserat_homotopy_" + iAsAscii.str());

    }

    // //////////////////////////////
    //   Output result
    // //////////////////////////////

    // Write the corresponding coefficient vector: verbatim in binary, to be completely lossless
    // This data may be used by other applications measuring the discretization error
    BlockVector<TargetSpace::CoordinateType> xEmbedded(x.size());
    for (size_t i=0; i<x.size(); i++)
      xEmbedded[i] = x[i].globalCoordinates();

    std::ofstream outFile("cosserat-continuum-result-" + std::to_string(numLevels) + ".data", std::ios_base::binary);
    GenericVector::writeBinary(outFile, xEmbedded);
    outFile.close();

    // finally: compute the average deformation of the Neumann boundary
    // That is what we need for the locking tests
    FieldVector<double,3> averageDef(0);
    for (size_t i=0; i<x.size(); i++)
        if (neumannNodes[i][0])
            averageDef += x[i].r;
    averageDef /= neumannNodes.count();

    if (mpiHelper.rank()==0)
    {
      std::cout << "Neumann values = " << parameterSet.get<FieldVector<double, 3> >("neumannValues") << "  "
                << ",  average deflection: " << averageDef << std::endl;
    }

    // //////////////////////////////
 } catch (Exception e) {

    std::cout << e << std::endl;

 }
