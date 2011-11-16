#include <config.h>

#include <fenv.h>

//#define LAPLACE_DEBUG
//#define HARMONIC_ENERGY_FD_GRADIENT

//#define UNITVECTOR2
#define UNITVECTOR3
//#define ROTATION2
//#define ROTATION3
//#define REALTUPLE1

#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/onedgrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/grid/io/file/amirameshreader.hh>
#include <dune/grid/io/file/amirameshwriter.hh>

#include <dune/fufem/boundarypatch.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/unitvector.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/harmonicenergystiffness.hh>
#include <dune/gfe/geodesicfeassembler.hh>
#include <dune/gfe/riemanniantrsolver.hh>

// grid dimension
const int dim = 3;

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
#ifdef REALTUPLE1
typedef RealTuple<double,1> TargetSpace;
#endif

// Tangent vector of the image space
const int blocksize = TargetSpace::TangentVector::dimension;

using namespace Dune;

BlockVector<FieldVector<double,3> >
computeEmbeddedDifference(const std::vector<TargetSpace>& a, const std::vector<TargetSpace>& b)
{
    assert(a.size() == b.size());
    
    BlockVector<FieldVector<double,3> > difference(a.size());
    
    for (int i=0; i<a.size(); i++)
        difference[i] = a[i].globalCoordinates() - b[i].globalCoordinates();

    return difference;
}

int main (int argc, char *argv[]) try
{
    //feenableexcept(FE_INVALID);

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

    // read problem settings
    std::string path                = parameterSet.get<std::string>("path");
    std::string gridFile            = parameterSet.get<std::string>("gridFile");

    // ///////////////////////////////////////
    //    Create the grid
    // ///////////////////////////////////////
    typedef std::conditional<dim==1,OneDGrid,UGGrid<dim> >::type GridType;
    
    shared_ptr<GridType> gridPtr;
    if (parameterSet.get<std::string>("gridType")=="structured") {
        array<unsigned int,dim> elements;
        elements.fill(3);
        gridPtr = StructuredGridFactory<GridType>::createSimplexGrid(FieldVector<double,dim>(0),
                                                                     FieldVector<double,dim>(1),
                                                                     elements);
    } else {
        gridPtr = shared_ptr<GridType>(AmiraMeshReader<GridType>::read(path + gridFile));
    }

    GridType& grid = *gridPtr.get();

    grid.globalRefine(numLevels-1);

    SolutionType x(grid.size(dim));

    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////

    BitSetVector<1> allNodes(grid.size(dim));
    allNodes.setAll();
    BoundaryPatch<GridType::LeafGridView> dirichletBoundary(grid.leafView(), allNodes);

    BitSetVector<blocksize> dirichletNodes(grid.size(dim));
    for (int i=0; i<dirichletNodes.size(); i++)
        dirichletNodes[i] = dirichletBoundary.containsVertex(i);
    
    // //////////////////////////
    //   Initial solution
    // //////////////////////////

    FieldVector<double,3> yAxis(0);
    yAxis[1] = 1;

    GridType::LeafGridView::Codim<dim>::Iterator vIt    = grid.leafbegin<dim>();
    GridType::LeafGridView::Codim<dim>::Iterator vEndIt = grid.leafend<dim>();

    for (; vIt!=vEndIt; ++vIt) {
        int idx = grid.leafIndexSet().index(*vIt);

#ifdef REALTUPLE1
        FieldVector<double,1> v;
#elif defined UNITVECTOR3
        FieldVector<double,3> v;
#else
        FieldVector<double,2> v;
#endif
        FieldVector<double,dim> pos = vIt->geometry().corner(0);
        FieldVector<double,3> axis;
        axis[0] = pos[0];  axis[1] = pos[1]; axis[2] = 1;
        Rotation<3,double> rotation(axis, pos.two_norm()*M_PI*1.5);

        //dirichletNodes[idx] = pos[0]<0.01 || pos[0] > 0.99;

        if (dirichletNodes[idx][0]) {
//             FieldMatrix<double,3,3> rMat;
//             rotation.matrix(rMat);
//             v = rMat[2];
#ifdef UNITVECTOR3
            v[0] = std::sin(pos[0]*M_PI);
            v[1] = 0;
            v[2] = std::cos(pos[0]*M_PI);
#endif
#ifdef UNITVECTOR2
            v[0] = std::sin(pos[0]*M_PI);
            v[1] = std::cos(pos[0]*M_PI);
#endif
#if defined ROTATION2 || defined REALTUPLE1
            v[0] = pos[0]/*M_PI*/;
#endif
        } else {
#ifdef UNITVECTOR2
            v[0] = 1;
            v[1] = 0;
#endif
#ifdef UNITVECTOR3
            v[0] = 1;
            v[1] = 0;
            v[2] = 0;
#endif
#if defined ROTATION2 || defined REALTUPLE1
            v[0] = 0.5*M_PI;
#endif
        }            

#if defined UNITVECTOR2 || defined UNITVECTOR3 || defined REALTUPLE1
        x[idx] = v;
#endif
#if defined ROTATION2 
        x[idx] = v[0];
#endif

    }
        
    // backup for error measurement later
    SolutionType initialIterate = x;

    // ////////////////////////////////////////////////////////////
    //   Create an assembler for the Harmonic Energy Functional
    // ////////////////////////////////////////////////////////////

    HarmonicEnergyLocalStiffness<GridType::LeafGridView,TargetSpace> harmonicEnergyLocalStiffness;

    GeodesicFEAssembler<GridType::LeafGridView,TargetSpace> assembler(grid.leafView(),
                                                                      &harmonicEnergyLocalStiffness);

    // /////////////////////////////////////////////////
    //   Create a Riemannian trust-region solver
    // /////////////////////////////////////////////////

    RiemannianTrustRegionSolver<GridType,TargetSpace> solver;
    solver.setup(grid, 
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

    solver.setInitialSolution(x);
    solver.solve();

    x = solver.getSol();

    // //////////////////////////////
    //   Output result
    // //////////////////////////////

    BlockVector<FieldVector<double,3> > xEmbedded(x.size());
    for (int i=0; i<x.size(); i++) {
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

    LeafAmiraMeshWriter<GridType> amiramesh;
    amiramesh.addGrid(grid.leafView());
    amiramesh.addVertexData(xEmbedded, grid.leafView());
    amiramesh.write("resultGrid", 1);
    
    // //////////////////////////////////////////////////////////
    //   Recompute and compare against exact solution
    // //////////////////////////////////////////////////////////
    
    SolutionType exactSolution = x;

    // //////////////////////////////////////////////////////////////////////
    //   Compute mass matrix and laplace matrix to emulate L2 and H1 norms
    // //////////////////////////////////////////////////////////////////////

    typedef P1NodalBasis<GridType::LeafGridView,double> FEBasis;
    FEBasis basis(grid.leafView());
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
    typedef BlockVector<FieldVector<double,3> > DifferenceType;
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
        for (int j=0; j<intermediateSolution.size(); j++) {
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
