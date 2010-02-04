#include <config.h>

#include <dune/common/bitsetvector.hh>
#include <dune/common/configparser.hh>

#include <dune/grid/onedgrid.hh>

#include <dune/istl/io.hh>

#include <dune/disc/operators/p1operator.hh>

#include <dune/solvers/iterationsteps/projectedblockgsstep.hh>
#include <dune/solvers/iterationsteps/mmgstep.hh>
#include <dune/solvers/solvers/loopsolver.hh>
#include <dune/solvers/norms/energynorm.hh>
#include <dune/solvers/transferoperators/mandelobsrestrictor.hh>
#include <dune/solvers/transferoperators/truncatedcompressedmgtransfer.hh>
#include <dune/ag-common/estimators/geometricmarking.hh>
#include <dune/ag-common/boundarypatch.hh>

#include "src/rodwriter.hh"
#include "src/planarrodassembler.hh"


// Number of degrees of freedom: 
// 3 (x, y, theta) for a planar rod
const int blocksize = 3;

using namespace Dune;
using std::string;

void setTrustRegionObstacles(double trustRegionRadius,
                             std::vector<BoxConstraint<double,blocksize> >& trustRegionObstacles,
                             const std::vector<BoxConstraint<double,blocksize> >& trueObstacles,
                             const BitSetVector<blocksize>& dirichletNodes)
{
    //std::cout << "True obstacles\n" << trueObstacles << std::endl;

    for (int j=0; j<trustRegionObstacles.size(); j++) {

        for (int k=0; k<blocksize; k++) {

            if (dirichletNodes[j][k])
                continue;

            trustRegionObstacles[j].lower(k) =
                (trueObstacles[j].lower(k) < -1e10)
                ? std::min(-trustRegionRadius, trueObstacles[j].upper(k) - trustRegionRadius)
                : trueObstacles[j].lower(k);
                
            trustRegionObstacles[j].upper(k) =
                (trueObstacles[j].upper(k) >  1e10) 
                ? std::max(trustRegionRadius,trueObstacles[j].lower(k) + trustRegionRadius)
                : trueObstacles[j].upper(k);

        }

    }

    //std::cout << "TrustRegion obstacles\n" << trustRegionObstacles << std::endl;
}

bool refineCondition(const FieldVector<double,1>& pos) {
    return pos[2] > -2 && pos[2] < -0.5;
}

bool refineAll(const FieldVector<double,1>& pos) {
    return true;
}

int main (int argc, char *argv[]) try
{
    // Some types that I need
    typedef BCRSMatrix<FieldMatrix<double, blocksize, blocksize> > MatrixType;
    typedef BlockVector<FieldVector<double, blocksize> >     VectorType;

    // parse data file
    ConfigParser parameterSet;
    parameterSet.parseFile("staticrod2.parset");

    // read solver settings
    const int minLevel         = parameterSet.get<int>("minLevel");
    const int maxLevel         = parameterSet.get<int>("maxLevel");
    const int maxNewtonSteps   = parameterSet.get<int>("maxNewtonSteps");
    const int numIt            = parameterSet.get<int>("numIt");
    const int nu1              = parameterSet.get<int>("nu1");
    const int nu2              = parameterSet.get<int>("nu2");
    const int mu               = parameterSet.get<int>("mu");
    const int baseIt           = parameterSet.get<int>("baseIt");
    const double tolerance     = parameterSet.get<double>("tolerance");
    const double baseTolerance = parameterSet.get<double>("baseTolerance");
    
    // Problem settings
    const int numRodBaseElements = parameterSet.get<int>("numRodBaseElements");
    
    // ///////////////////////////////////////
    //    Create the two grids
    // ///////////////////////////////////////
    typedef OneDGrid GridType;
    GridType grid(numRodBaseElements, 0, 1);

    std::cout << "Grid has " << grid.size(0,0) << " elements and " << grid.size(0,1) << " vertices." << std::endl;

    std::vector<std::vector<BoxConstraint<double,3> > > trustRegionObstacles(1);
    std::vector<BitSetVector<1> > hasObstacle(1);
    std::vector<BitSetVector<blocksize> > dirichletNodes(1);

    // ////////////////////////////////
    //   Create a multigrid solver
    // ////////////////////////////////

    // First create a gauss-seidel base solver
    ProjectedBlockGSStep<MatrixType, VectorType> baseSolverStep;

    EnergyNorm<MatrixType, VectorType> baseEnergyNorm(baseSolverStep);

    LoopSolver<VectorType> baseSolver(&baseSolverStep,
                                                       baseIt,
                                                       baseTolerance,
                                                       &baseEnergyNorm,
                                                       Solver::QUIET);

    // Make pre and postsmoothers
    ProjectedBlockGSStep<MatrixType, VectorType> presmoother;
    ProjectedBlockGSStep<MatrixType, VectorType> postsmoother;

    MonotoneMGStep<MatrixType, VectorType> multigridStep(1);

    multigridStep.setMGType(mu, nu1, nu2);
    multigridStep.ignoreNodes_       = &dirichletNodes[0];
    multigridStep.basesolver_        = &baseSolver;
    multigridStep.setSmoother(&presmoother, &postsmoother);
    multigridStep.hasObstacle_       = &hasObstacle;
    multigridStep.obstacles_         = &trustRegionObstacles;
    multigridStep.verbosity_         = Solver::QUIET;
    multigridStep.obstacleRestrictor_ = new MandelObstacleRestrictor<VectorType>;


    EnergyNorm<MatrixType, VectorType> energyNorm(multigridStep);

    LoopSolver<VectorType> solver(&multigridStep,
                                                   numIt,
                                                   tolerance,
                                                   &energyNorm,
                                                   Solver::FULL);

    double trustRegionRadius = 0.1;

    VectorType rhs;
    VectorType x(grid.size(0,1));
    VectorType corr;

    // //////////////////////////
    //   Initial solution
    // //////////////////////////
    x = 0;
    x[x.size()-1][2] = 2*M_PI;


    // /////////////////////////////////////////////////////////////////////
    //   Refinement Loop
    // /////////////////////////////////////////////////////////////////////
    
    for (int toplevel=0; toplevel<=maxLevel; toplevel++) {
        
        std::cout << "####################################################" << std::endl;
        std::cout << "      Solving on level: " << toplevel << std::endl;
        std::cout << "####################################################" << std::endl;
    
        dirichletNodes.resize(toplevel+1);
        for (int i=0; i<=toplevel; i++) {
            
            dirichletNodes[i].resize( grid.size(i,1), false );
            
            dirichletNodes[i][0]     = true;
            dirichletNodes[i].back() = true;

        }
        
        // ////////////////////////////////////////////////////////////
        //    Create solution and rhs vectors
        // ////////////////////////////////////////////////////////////


        MatrixType hessianMatrix;
        PlanarRodAssembler<GridType,4> rodAssembler(grid);
        
        rodAssembler.setParameters(1, 350000, 350000);
        
        MatrixIndexSet indices(grid.size(toplevel,1), grid.size(toplevel,1));
        rodAssembler.getNeighborsPerVertex(indices);
        indices.exportIdx(hessianMatrix);
        
        rhs.resize(grid.size(toplevel,1));
        corr.resize(grid.size(toplevel,1));
    

        // //////////////////////////////////////////////////////////
        //   Create obstacles
        // //////////////////////////////////////////////////////////
        
        hasObstacle.resize(toplevel+1);
        for (int i=0; i<hasObstacle.size(); i++) {
            hasObstacle[i].resize(grid.size(i, 1));
            hasObstacle[i].setAll();
        }
        
        std::vector<std::vector<BoxConstraint<double,3> > > trueObstacles(toplevel+1);
        trustRegionObstacles.resize(toplevel+1);
        
        for (int i=0; i<toplevel+1; i++) {
            trueObstacles[i].resize(grid.size(i,1));
            trustRegionObstacles[i].resize(grid.size(i,1));
        }
        
        for (int i=0; i<trueObstacles[toplevel].size(); i++) {
            trueObstacles[toplevel][i].clear();
            //trueObstacles[toplevel][i].val[0] =     - x[i][0];
            trueObstacles[toplevel][i].upper(0) = 0.1 - x[i][0];
        }
        

        trustRegionObstacles.resize(toplevel+1);
        for (int i=0; i<=toplevel; i++)
            trustRegionObstacles[i].resize(grid.size(i, 1));

        // ////////////////////////////////////
        //   Create the transfer operators
        // ////////////////////////////////////
        for (int k=0; k<multigridStep.mgTransfer_.size(); k++)
            delete(multigridStep.mgTransfer_[k]);

        multigridStep.mgTransfer_.resize(toplevel);

        for (int i=0; i<multigridStep.mgTransfer_.size(); i++){
            TruncatedCompressedMGTransfer<VectorType>* newTransferOp = new TruncatedCompressedMGTransfer<VectorType>;
            newTransferOp->setup(grid,i,i+1);
            multigridStep.mgTransfer_[i] = newTransferOp;
        }

        // /////////////////////////////////////////////////////
        //   Trust-Region Solver
        // /////////////////////////////////////////////////////
        for (int i=0; i<maxNewtonSteps; i++) {

            std::cout << "----------------------------------------------------" << std::endl;
            std::cout << "      Trust-Region Step Number: " << i << std::endl;
            std::cout << "----------------------------------------------------" << std::endl;


            rhs = 0;
            corr = 0;
            
            rodAssembler.assembleGradient(x, rhs);
            rodAssembler.assembleMatrix(x, hessianMatrix);
            
            rhs *= -1;

            // Create trust-region obstacle on grid0.maxLevel()
            setTrustRegionObstacles(trustRegionRadius,
                                    trustRegionObstacles[toplevel],
                                    trueObstacles[toplevel],
                                    dirichletNodes[toplevel]);

            dynamic_cast<MultigridStep<MatrixType,VectorType>*>(solver.iterationStep_)->setProblem(hessianMatrix, corr, rhs, toplevel+1);

            solver.preprocess();

            multigridStep.preprocess();


            // /////////////////////////////
            //    Solve !
            // /////////////////////////////
             solver.solve();

             corr = multigridStep.getSol();

             printf("infinity norm of the correction: %g\n", corr.infinity_norm());
             if (corr.infinity_norm() < 1e-5) {
                 std::cout << "CORRECTION IS SMALL ENOUGH" << std::endl;
                 break;
             }

             // ////////////////////////////////////////////////////
             //   Check whether trust-region step can be accepted
             // ////////////////////////////////////////////////////
             /** \todo Faster with expression templates */
             VectorType newIterate = x;  newIterate += corr;

             /** \todo Don't always recompute oldEnergy */
             double oldEnergy = rodAssembler.computeEnergy(x); 
             double energy    = rodAssembler.computeEnergy(newIterate); 

             if (energy >= oldEnergy) {
                 printf("Richtung ist keine Abstiegsrichtung!\n");
//                  std::cout << "corr[0]\n" << corr[0] << std::endl;

                 exit(0);
             }
                 
             //  Add correction to the current solution
             x += corr;

             // Subtract correction from the current obstacle
             for (int k=0; k<corr.size(); k++)
                 trueObstacles[grid.maxLevel()][k] -= corr[k];

        }
        
        // //////////////////////////////
        //   Output result
        // //////////////////////////////
        
        // Write Lagrange multiplyers
        std::stringstream levelAsAscii;
        levelAsAscii << toplevel;
        std::string lagrangeFilename = "pressure/lagrange_" + levelAsAscii.str();
        std::ofstream lagrangeFile(lagrangeFilename.c_str());
        
        VectorType lagrangeMultipliers;
        rodAssembler.assembleGradient(x, lagrangeMultipliers);
        lagrangeFile << lagrangeMultipliers << std::endl;
        
        // Write result grid
        std::string solutionFilename = "solutions/rod_" + levelAsAscii.str() + ".result";
        writeRod(x, solutionFilename);
        
        // ////////////////////////////////////////////////////////////////////////////
        //    Refine locally and transfer the current solution to the new leaf level
        // ////////////////////////////////////////////////////////////////////////////
        
        GeometricEstimator<GridType> estimator;
        
        estimator.estimate(grid, (toplevel<=minLevel) ? refineAll : refineCondition);

        P1FunctionManager<GridType,double> functionManager(grid);
        LeafP1Function<GridType,double,blocksize> sol(grid);
        *sol = x;

        grid.preAdapt();
        sol.preAdapt();
        grid.adapt();

        sol.postAdapt(functionManager);
        grid.postAdapt();

        x = *sol;

        //writeRod(x, "solutions/rod_1.result");
    }

 } catch (Exception e) {

    std::cout << e << std::endl;

 }
