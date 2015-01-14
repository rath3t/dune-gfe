#include <config.h>

#include <fenv.h>

//#define UNITVECTOR2
#define UNITVECTOR3
//#define UNITVECTOR4
//#define ROTATION2
//#define ROTATION3
//#define REALTUPLE1

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <dune/gfe/adolcnamespaceinjections.hh>

#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/onedgrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/grid/io/file/amirameshreader.hh>
#include <dune/grid/io/file/vtk.hh>

#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/functions/vtkbasisgridfunction.hh>
#include <dune/fufem/functiontools/basisinterpolator.hh>
#include <dune/fufem/dunepython.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/unitvector.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/harmonicenergystiffness.hh>
#include <dune/gfe/chiralskyrmionenergy.hh>
#include <dune/gfe/geodesicfeassembler.hh>
#include <dune/gfe/riemanniantrsolver.hh>

// grid dimension
const int dim = 2;

// Image space of the geodesic fe functions
#ifdef ROTATION2
typedef Rotation<double,2> TargetSpace;
#endif
#ifdef ROTATION3
typedef Rotation<double,3> TargetSpace;
#endif
#ifdef UNITVECTOR2
typedef UnitVector<double,2> TargetSpace;
#endif
#ifdef UNITVECTOR3
typedef UnitVector<double,3> TargetSpace;
#endif
#ifdef UNITVECTOR4
typedef UnitVector<double,4> TargetSpace;
#endif
#ifdef REALTUPLE1
typedef RealTuple<double,1> TargetSpace;
#endif

// Tangent vector of the image space
const int blocksize = TargetSpace::TangentVector::dimension;

using namespace Dune;

BlockVector<TargetSpace::CoordinateType>
computeEmbeddedDifference(const std::vector<TargetSpace>& a, const std::vector<TargetSpace>& b)
{
    assert(a.size() == b.size());

    BlockVector<TargetSpace::CoordinateType> difference(a.size());

    for (size_t i=0; i<a.size(); i++)
        difference[i] = a[i].globalCoordinates() - b[i].globalCoordinates();

    return difference;
}

int main (int argc, char *argv[]) try
{
    //feenableexcept(FE_INVALID);
    // Start Python interpreter
    Python::start();
    Python::Reference main = Python::import("__main__");
    Python::run("import math");

    //feenableexcept(FE_INVALID);
    Python::runStream()
        << std::endl << "import sys"
        << std::endl << "sys.path.append('/home/sander/dune/dune-gfe/src')"
        << std::endl;


    typedef std::vector<TargetSpace> SolutionType;

    // parse data file
    ParameterTree parameterSet;
    if (argc==2)
        ParameterTreeParser::readINITree(argv[1], parameterSet);
    else
        ParameterTreeParser::readINITree("harmonicmaps.parset", parameterSet);

    // read solver settings
    const int numLevels                   = parameterSet.get<int>("numLevels");
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
    typedef std::conditional<dim==1,OneDGrid,UGGrid<dim> >::type GridType;

    shared_ptr<GridType> grid;
    FieldVector<double,dim> lower(0), upper(1);

    if (parameterSet.get<bool>("structuredGrid")) {

        lower = parameterSet.get<FieldVector<double,dim> >("lower");
        upper = parameterSet.get<FieldVector<double,dim> >("upper");

        array<unsigned int,dim> elements = parameterSet.get<array<unsigned int,dim> >("elements");
        grid = StructuredGridFactory<GridType>::createCubeGrid(lower, upper, elements);

    } else {

        std::string path                = parameterSet.get<std::string>("path");
        std::string gridFile            = parameterSet.get<std::string>("gridFile");

        grid = shared_ptr<GridType>(AmiraMeshReader<GridType>::read(path + gridFile));
    }

    grid->globalRefine(numLevels-1);

    SolutionType x(grid->size(dim));

    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////

    BitSetVector<1> allNodes(grid->size(dim));
    allNodes.setAll();
    BoundaryPatch<GridType::LeafGridView> dirichletBoundary(grid->leafGridView(), allNodes);

    BitSetVector<blocksize> dirichletNodes(grid->size(dim));
    for (size_t i=0; i<dirichletNodes.size(); i++)
        dirichletNodes[i] = dirichletBoundary.containsVertex(i);

    // //////////////////////////
    //   Initial iterate
    // //////////////////////////

    typedef P1NodalBasis<typename GridType::LeafGridView,double> FEBasis;
    FEBasis feBasis(grid->leafGridView());

    std::string lambda = std::string("lambda x: (") + parameterSet.get<std::string>("initialIterate") + std::string(")");
    PythonFunction<FieldVector<double,dim>, TargetSpace::CoordinateType > pythonInitialIterate(Python::evaluate(lambda));

    std::vector<TargetSpace::CoordinateType> v;
    Functions::interpolate(feBasis, v, pythonInitialIterate);

    for (size_t i=0; i<x.size(); i++)
      x[i] = v[i];

    // backup for error measurement later
    SolutionType initialIterate = x;

    // ////////////////////////////////////////////////////////////
    //   Create an assembler for the Harmonic Energy Functional
    // ////////////////////////////////////////////////////////////

    GFE::ChiralSkyrmionEnergy<GridType::LeafGridView, FEBasis::LocalFiniteElement, double> chiralSkyrmionEnergy;
    // Assembler using ADOL-C
    typedef TargetSpace::rebind<adouble>::other ATargetSpace;
    HarmonicEnergyLocalStiffness<GridType::LeafGridView, FEBasis::LocalFiniteElement, ATargetSpace> harmonicEnergyADOLCLocalStiffness;
    LocalGeodesicFEADOLCStiffness<GridType::LeafGridView,
                                  FEBasis::LocalFiniteElement,
                                  TargetSpace> localGFEADOLCStiffness(&harmonicEnergyADOLCLocalStiffness);

    GeodesicFEAssembler<FEBasis,TargetSpace> assembler(grid->leafGridView(), &localGFEADOLCStiffness);

    // /////////////////////////////////////////////////
    //   Create a Riemannian trust-region solver
    // /////////////////////////////////////////////////

    RiemannianTrustRegionSolver<GridType,TargetSpace> solver;
    solver.setup(*grid,
                 &assembler,
                 x,
                 dirichletNodes,
                 tolerance,
                 maxTrustRegionSteps,
                 initialTrustRegionRadius,
                 multigridIterations,
                 mgTolerance,
                 mu, nu1, nu2,
                 baseIterations,
                 baseTolerance,
                 instrumented);

    // /////////////////////////////////////////////////////
    //   Solve!
    // /////////////////////////////////////////////////////

    std::cout << "Energy: " << assembler.computeEnergy(x) << std::endl;
    //exit(0);

    solver.setInitialIterate(x);
    solver.solve();

    x = solver.getSol();

    // //////////////////////////////
    //   Output result
    // //////////////////////////////

    typedef BlockVector<FieldVector<double,3> > EmbeddedVectorType;
    EmbeddedVectorType xEmbedded(x.size());
    for (size_t i=0; i<x.size(); i++) {
#ifdef UNITVECTOR2
        xEmbedded[i][0] = x[i].globalCoordinates()[0];
        xEmbedded[i][1] = 0;
        xEmbedded[i][2] = x[i].globalCoordinates()[1];
#endif
#ifdef UNITVECTOR3
        xEmbedded[i][0] = x[i].globalCoordinates()[0];
        xEmbedded[i][1] = x[i].globalCoordinates()[1];
        xEmbedded[i][2] = x[i].globalCoordinates()[2];
#endif
#ifdef ROTATION2
        xEmbedded[i][0] = std::sin(x[i].angle_);
        xEmbedded[i][1] = 0;
        xEmbedded[i][2] = std::cos(x[i].angle_);
#endif
#ifdef REALTUPLE1
        xEmbedded[i][0] = std::sin(x[i].globalCoordinates()[0]);
        xEmbedded[i][1] = 0;
        xEmbedded[i][2] = std::cos(x[i].globalCoordinates()[0]);
#endif
    }

    VTKWriter<GridType::LeafGridView> vtkWriter(grid->leafGridView());
    Dune::shared_ptr<VTKBasisGridFunction<FEBasis,EmbeddedVectorType> > vtkVectorField
        = Dune::shared_ptr<VTKBasisGridFunction<FEBasis,EmbeddedVectorType> >
               (new VTKBasisGridFunction<FEBasis,EmbeddedVectorType>(feBasis, xEmbedded, "orientation"));
    vtkWriter.addVertexData(vtkVectorField);

    vtkWriter.write("resultGrid");

    // //////////////////////////////////////////////////////////
    //   Recompute and compare against exact solution
    // //////////////////////////////////////////////////////////

    SolutionType exactSolution = x;

    // //////////////////////////////////////////////////////////////////////
    //   Compute mass matrix and laplace matrix to emulate L2 and H1 norms
    // //////////////////////////////////////////////////////////////////////

    typedef P1NodalBasis<GridType::LeafGridView,double> FEBasis;
    FEBasis basis(grid->leafGridView());
    OperatorAssembler<FEBasis,FEBasis> operatorAssembler(basis, basis);

    LaplaceAssembler<GridType, FEBasis::LocalFiniteElement, FEBasis::LocalFiniteElement> laplaceLocalAssembler;
    MassAssembler<GridType, FEBasis::LocalFiniteElement, FEBasis::LocalFiniteElement> massMatrixLocalAssembler;

    typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> > ScalarMatrixType;
    ScalarMatrixType laplaceMatrix, massMatrix;

    operatorAssembler.assemble(laplaceLocalAssembler, laplaceMatrix);
    operatorAssembler.assemble(massMatrixLocalAssembler, massMatrix);


    double error = std::numeric_limits<double>::max();

    SolutionType intermediateSolution(x.size());

    // Create statistics file
    std::ofstream statisticsFile((resultPath + "trStatistics").c_str());

    // Compute error of the initial iterate
    typedef BlockVector<TargetSpace::CoordinateType> DifferenceType;
    DifferenceType difference = computeEmbeddedDifference(exactSolution, initialIterate);

    H1SemiNorm< BlockVector<TargetSpace::CoordinateType> > h1Norm(laplaceMatrix);
    H1SemiNorm< BlockVector<TargetSpace::CoordinateType> > l2Norm(massMatrix);

    //double oldError = std::sqrt(EnergyNorm<BCRSMatrix<FieldMatrix<double, blocksize, blocksize> >, BlockVector<FieldVector<double,blocksize> > >::normSquared(geodesicDifference, hessian));
    double oldError = h1Norm(difference);
    int i;
    for (i=0; i<maxTrustRegionSteps; i++) {

        // /////////////////////////////////////////////////////
        //   Read iteration from file
        // /////////////////////////////////////////////////////
        char iSolFilename[100];
        sprintf(iSolFilename, "tmp/intermediateSolution_%04d", i);

        FILE* fp = fopen(iSolFilename, "rb");
        if (!fp)
            DUNE_THROW(IOError, "Couldn't open intermediate solution '" << iSolFilename << "'");
        for (size_t j=0; j<intermediateSolution.size(); j++) {
            fread(&intermediateSolution[j], sizeof(TargetSpace), 1, fp);
        }

        fclose(fp);

        // /////////////////////////////////////////////////////
        //   Compute error
        // /////////////////////////////////////////////////////

        difference = computeEmbeddedDifference(exactSolution, intermediateSolution);

        //error = std::sqrt(EnergyNorm<BCRSMatrix<FieldMatrix<double, blocksize, blocksize> >, BlockVector<FieldVector<double,blocksize> > >::normSquared(geodesicDifference, hessian));
        error = h1Norm(difference);

        double convRate = error / oldError;

        // Output
        std::cout << "Trust-region iteration: " << i << "  error : " << error << ",      "
                  << "convrate " << convRate << std::endl;
        statisticsFile << i << "  " << error << "  " << convRate << std::endl;

        if (error < 1e-12)
          break;

        oldError = error;

    }

    // //////////////////////////////
 } catch (Exception e) {

    std::cout << e << std::endl;

 }
