#include <config.h>

#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/grid/onedgrid.hh>

#include <dune/istl/io.hh>


#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/geodesicdifference.hh>
#include <dune/gfe/rodwriter.hh>
#include <dune/gfe/rotation.hh>
#include <dune/gfe/rodassembler.hh>
#include <dune/gfe/riemanniantrsolver.hh>

typedef RigidBodyMotion<double,3> TargetSpace;

const int blocksize = TargetSpace::TangentVector::dimension;

using namespace Dune;
using std::string;

int main (int argc, char *argv[]) try
{
    typedef std::vector<RigidBodyMotion<double,3> > SolutionType;

    // parse data file
    ParameterTree parameterSet;
    if (argc==2)
        ParameterTreeParser::readINITree(argv[1], parameterSet);
    else
        ParameterTreeParser::readINITree("rod3d.parset", parameterSet);

    // read solver settings
    const int numLevels        = parameterSet.get<int>("numLevels");
    const double tolerance        = parameterSet.get<double>("tolerance");
    const int maxTrustRegionSteps   = parameterSet.get<int>("maxNewtonSteps");
    const double initialTrustRegionRadius = parameterSet.get<double>("initialTrustRegionRadius");
    const int multigridIterations   = parameterSet.get<int>("numIt");
    const int nu1              = parameterSet.get<int>("nu1");
    const int nu2              = parameterSet.get<int>("nu2");
    const int mu               = parameterSet.get<int>("mu");
    const int baseIterations      = parameterSet.get<int>("baseIt");
    const double mgTolerance        = parameterSet.get<double>("mgTolerance");
    const double baseTolerance    = parameterSet.get<double>("baseTolerance");
    const bool instrumented      = parameterSet.get<bool>("instrumented");
    std::string resultPath           = parameterSet.get("resultPath", "");

    // read rod parameter settings
    const double A               = parameterSet.get<double>("A");
    const double J1              = parameterSet.get<double>("J1");
    const double J2              = parameterSet.get<double>("J2");
    const double E               = parameterSet.get<double>("E");
    const double nu              = parameterSet.get<double>("nu");
    const int numRodBaseElements = parameterSet.get<int>("numRodBaseElements");
    
    // ///////////////////////////////////////
    //    Create the grid
    // ///////////////////////////////////////
    typedef OneDGrid GridType;
    GridType grid(numRodBaseElements, 0, 1);

    grid.globalRefine(numLevels-1);

    using GridView = GridType::LeafGridView;
    GridView gridView = grid.leafGridView();

    using FEBasis = Functions::LagrangeBasis<GridView,1>;
    FEBasis feBasis(gridView);

    SolutionType x(feBasis.size());

    // //////////////////////////
    //   Initial solution
    // //////////////////////////

    for (size_t i=0; i<x.size(); i++) {
        x[i].r[0] = 0;
        x[i].r[1] = 0;
        x[i].r[2] = double(i)/(x.size()-1);
        x[i].q    = Rotation<double,3>::identity();
    }

    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////
    x.back().r[0] = parameterSet.get<double>("dirichletValueX");
    x.back().r[1] = parameterSet.get<double>("dirichletValueY");
    x.back().r[2] = parameterSet.get<double>("dirichletValueZ");

    FieldVector<double,3> axis;
    axis[0] = parameterSet.get<double>("dirichletAxisX");
    axis[1] = parameterSet.get<double>("dirichletAxisY");
    axis[2] = parameterSet.get<double>("dirichletAxisZ");
    double angle = parameterSet.get<double>("dirichletAngle");

    x.back().q = Rotation<double,3>(axis, M_PI*angle/180);

    // backup for error measurement later
    SolutionType initialIterate = x;

    std::cout << "Left boundary orientation:" << std::endl;
    std::cout << "director 0:  " << x[0].q.director(0) << std::endl;
    std::cout << "director 1:  " << x[0].q.director(1) << std::endl;
    std::cout << "director 2:  " << x[0].q.director(2) << std::endl;
    std::cout << std::endl;
    std::cout << "Right boundary orientation:" << std::endl;
    std::cout << "director 0:  " << x[x.size()-1].q.director(0) << std::endl;
    std::cout << "director 1:  " << x[x.size()-1].q.director(1) << std::endl;
    std::cout << "director 2:  " << x[x.size()-1].q.director(2) << std::endl;

    BitSetVector<blocksize> dirichletNodes(feBasis.size());
    dirichletNodes.unsetAll();
        
    dirichletNodes[0] = true;
    dirichletNodes.back() = true;
    
    // ///////////////////////////////////////////
    //   Create a solver for the rod problem
    // ///////////////////////////////////////////

    RodLocalStiffness<GridView,double> localStiffness(gridView,
                                                      A, J1, J2, E, nu);

    RodAssembler<FEBasis,3> rodAssembler(gridView, &localStiffness);

    RiemannianTrustRegionSolver<FEBasis,RigidBodyMotion<double,3> > rodSolver;

    rodSolver.setup(grid, 
                    &rodAssembler,
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

    std::cout << "Energy: " << rodAssembler.computeEnergy(x) << std::endl;
    
    rodSolver.setInitialIterate(x);
    rodSolver.solve();

    x = rodSolver.getSol();
        
    // //////////////////////////////
    //   Output result
    // //////////////////////////////
    writeRod(x, resultPath + "rod3d.result");
    BlockVector<FieldVector<double, 6> > strain(x.size()-1);
    rodAssembler.getStrain(x,strain);

    // If convergence measurement is not desired stop here
    if (!instrumented)
        exit(0);

    // //////////////////////////////////////////////////////////
    //   Recompute and compare against exact solution
    // //////////////////////////////////////////////////////////
    
    SolutionType exactSolution = x;

    // //////////////////////////////////////////////////////////
    //   Compute hessian of the rod functional at the exact solution
    //   for use of the energy norm it creates.
    // //////////////////////////////////////////////////////////

    BCRSMatrix<FieldMatrix<double, 6, 6> > hessian;
    MatrixIndexSet indices(exactSolution.size(), exactSolution.size());
    rodAssembler.getNeighborsPerVertex(indices);
    indices.exportIdx(hessian);
    BlockVector<FieldVector<double,6> > dummyRhs(x.size());
    rodAssembler.assembleGradientAndHessian(exactSolution, dummyRhs, hessian);


    double error = std::numeric_limits<double>::max();

    SolutionType intermediateSolution(x.size());

    // Create statistics file
    std::ofstream statisticsFile((resultPath + "trStatistics").c_str());

    // Compute error of the initial iterate
    typedef BlockVector<FieldVector<double,6> > RodDifferenceType;
    RodDifferenceType rodDifference = computeGeodesicDifference(exactSolution, initialIterate);
    double oldError = std::sqrt(EnergyNorm<BCRSMatrix<FieldMatrix<double, blocksize, blocksize> >, BlockVector<FieldVector<double,blocksize> > >::normSquared(rodDifference, hessian));

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
            fread(&intermediateSolution[j].r, sizeof(double), 3, fp);
            fread(&intermediateSolution[j].q, sizeof(double), 4, fp);
        }
        
        fclose(fp);

        // /////////////////////////////////////////////////////
        //   Compute error
        // /////////////////////////////////////////////////////

        rodDifference = computeGeodesicDifference(exactSolution, intermediateSolution);
        
        error = std::sqrt(EnergyNorm<BCRSMatrix<FieldMatrix<double, blocksize, blocksize> >, BlockVector<FieldVector<double,blocksize> > >::normSquared(rodDifference, hessian));
        

        double convRate = error / oldError;

        // Output
        std::cout << "Trust-region iteration: " << i << "  error : " << error << ",      "
                  << "convrate " << convRate << std::endl;
        statisticsFile << i << "  " << error << "  " << convRate << std::endl;

        if (error < 1e-12)
          break;

        oldError = error;
        
    }            

} catch (Exception& e)
{
    std::cout << e.what() << std::endl;
}
