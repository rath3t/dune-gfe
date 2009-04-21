#include <config.h>

#include <dune/common/bitsetvector.hh>
#include <dune/common/configparser.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/io/file/amirameshreader.hh>
#include <dune/grid/io/file/amirameshwriter.hh>

#include <dune-solvers/solvers/iterativesolver.hh>
#include <dune-solvers/norms/energynorm.hh>

#include "src/geodesicdifference.hh"
#include "src/rodwriter.hh"
#include "src/rotation.hh"
#include "src/harmonicenergystiffness.hh"
#include "src/geodesicfeassembler.hh"
#include "src/riemanniantrsolver.hh"

// grid dimension
const int dim = 3;

// Image space of the geodesic fe functions
typedef Rotation<3,double> TargetSpace;

// Tangent vector of the image space
const int blocksize = TargetSpace::TangentVector::size;

using namespace Dune;

int main (int argc, char *argv[]) try
{
    typedef std::vector<TargetSpace> SolutionType;

    // parse data file
    ConfigParser parameterSet;
    if (argc==2)
        parameterSet.parseFile(argv[1]);
    else
        parameterSet.parseFile("harmonicmaps.parset");

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
    typedef UGGrid<dim> GridType;
    GridType grid;
    grid.setRefinementType(GridType::COPY);
    AmiraMeshReader<GridType>::read(grid, path + gridFile);

    grid.globalRefine(numLevels-1);

    SolutionType x(grid.size(dim));

    // //////////////////////////
    //   Initial solution
    // //////////////////////////

    FieldVector<double,3> yAxis(0);
    yAxis[1] = 1;

    GridType::LeafGridView::Codim<dim>::Iterator vIt    = grid.leafbegin<dim>();
    GridType::LeafGridView::Codim<dim>::Iterator vEndIt = grid.leafend<dim>();

    for (; vIt!=vEndIt; ++vIt) {
        int idx = grid.leafIndexSet().index(*vIt);
        double angle = vIt->geometry().corner(0)[0];
        x[idx] = Rotation<3,double>(yAxis,angle);
    }
        
    // backup for error measurement later
    SolutionType initialIterate = x;

    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////

    BitSetVector<1> allNodes(grid.size(dim));
    allNodes.setAll();
    LeafBoundaryPatch<GridType> dirichletBoundary(grid, allNodes);

    BitSetVector<blocksize> dirichletNodes(grid.size(dim));
    for (int i=0; i<dirichletNodes.size(); i++)
        dirichletNodes[i] = dirichletBoundary.containsVertex(i);
    
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
    LeafAmiraMeshWriter<GridType> amiramesh;
    amiramesh.addGrid(grid.leafView());
    amiramesh.write("resultGrid", 1);

    solver.setInitialSolution(x);
    solver.solve();

    x = solver.getSol();

    // //////////////////////////////
    //   Output result
    // //////////////////////////////
#if 0
    writeRod(x, resultPath + "rod3d.result");
#endif

    // //////////////////////////////////////////////////////////
    //   Recompute and compare against exact solution
    // //////////////////////////////////////////////////////////
    
    SolutionType exactSolution = x;

    // //////////////////////////////////////////////////////////
    //   Compute hessian of the rod functional at the exact solution
    //   for use of the energy norm it creates.
    // //////////////////////////////////////////////////////////

    BCRSMatrix<FieldMatrix<double, blocksize, blocksize> > hessian;
    assembler.assembleMatrix(exactSolution, hessian);


    double error = std::numeric_limits<double>::max();

    SolutionType intermediateSolution(x.size());

    // Create statistics file
    std::ofstream statisticsFile((resultPath + "trStatistics").c_str());

    // Compute error of the initial iterate
    typedef BlockVector<FieldVector<double,blocksize> > DifferenceType;
    DifferenceType geodesicDifference = computeGeodesicDifference(exactSolution, initialIterate);
    double oldError = std::sqrt(EnergyNorm<BCRSMatrix<FieldMatrix<double, blocksize, blocksize> >, BlockVector<FieldVector<double,blocksize> > >::normSquared(geodesicDifference, hessian));

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
            fread(&intermediateSolution[j], sizeof(double), 4, fp);
        }
        
        fclose(fp);

        // /////////////////////////////////////////////////////
        //   Compute error
        // /////////////////////////////////////////////////////

        geodesicDifference = computeGeodesicDifference(exactSolution, intermediateSolution);
        
        error = std::sqrt(EnergyNorm<BCRSMatrix<FieldMatrix<double, blocksize, blocksize> >, BlockVector<FieldVector<double,blocksize> > >::normSquared(geodesicDifference, hessian));
        

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
