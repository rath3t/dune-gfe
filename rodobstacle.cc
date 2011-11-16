#include <config.h>

#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/grid/onedgrid.hh>

#include <dune/istl/io.hh>

#include <dune/solvers/iterationsteps/projectedblockgsstep.hh>
#include <dune/solvers/iterationsteps/mmgstep.hh>
#include <dune/solvers/solvers/loopsolver.hh>
#include <dune/solvers/norms/energynorm.hh>
#include <dune/solvers/transferoperators/mandelobsrestrictor.hh>
#include <dune/solvers/transferoperators/truncatedcompressedmgtransfer.hh>
#include <dune/fufem/estimators/geometricmarking.hh>
#include <dune/fufem/boundarypatch.hh>

#include <dune/gfe/rodwriter.hh>
#include <dune/gfe/rodassembler.hh>


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
    typedef BlockVector<FieldVector<double, blocksize> >           CorrectionType;
    typedef std::vector<RigidBodyMotion<double,2> >                SolutionType;

    // parse data file
    ParameterTree parameterSet;
    ParameterTreeParser::readINITree("rodobstacle.parset", parameterSet);

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
    GridType grid(numRodBaseElements, 0, numRodBaseElements);

    grid.globalRefine(minLevel);

    std::vector<std::vector<BoxConstraint<double,3> > > trustRegionObstacles(minLevel+1);
    std::vector<BitSetVector<1> > hasObstacle(minLevel+1);
    BitSetVector<blocksize> dirichletNodes;

    // ////////////////////////////////
    //   Create a multigrid solver
    // ////////////////////////////////

    // First create a gauss-seidel base solver
    ProjectedBlockGSStep<MatrixType, CorrectionType> baseSolverStep;

    EnergyNorm<MatrixType, CorrectionType> baseEnergyNorm(baseSolverStep);

    LoopSolver<CorrectionType> baseSolver(&baseSolverStep,
                                                       baseIt,
                                                       baseTolerance,
                                                       &baseEnergyNorm,
                                                       Solver::QUIET);

    // Make pre and postsmoothers
    ProjectedBlockGSStep<MatrixType, CorrectionType> presmoother;
    ProjectedBlockGSStep<MatrixType, CorrectionType> postsmoother;

    MonotoneMGStep<MatrixType, CorrectionType> multigridStep(1);

    multigridStep.setMGType(mu, nu1, nu2);
    multigridStep.ignoreNodes_       = &dirichletNodes;
    multigridStep.basesolver_        = &baseSolver;
    multigridStep.hasObstacle_       = &hasObstacle;
    multigridStep.obstacles_         = &trustRegionObstacles;
    multigridStep.verbosity_         = Solver::QUIET;
    multigridStep.obstacleRestrictor_ = new MandelObstacleRestrictor<CorrectionType>;


    EnergyNorm<MatrixType, CorrectionType> energyNorm(multigridStep);

    LoopSolver<CorrectionType> solver(&multigridStep,
                                                   numIt,
                                                   tolerance,
                                                   &energyNorm,
                                                   Solver::FULL);

    double trustRegionRadius = 0.1;

    CorrectionType rhs;
    SolutionType x(grid.size(1));
    CorrectionType corr;

    // //////////////////////////
    //   Initial solution
    // //////////////////////////

    for (int i=0; i<x.size(); i++) {
        x[i].r[0] = 0;
        x[i].r[1] = i;//double(i)/(x.size()-1);
        x[i].q    = Rotation<double,2>::identity();
    }

    x.back().r[1] += 1;

    // /////////////////////////////////////////////////////////////////////
    //   Refinement Loop
    // /////////////////////////////////////////////////////////////////////
    
    for (int toplevel=minLevel; toplevel<=maxLevel; toplevel++) {
        
        std::cout << "####################################################" << std::endl;
        std::cout << "      Solving on level: " << toplevel << std::endl;
        std::cout << "####################################################" << std::endl;
    
        dirichletNodes.resize( grid.size(1) );
        dirichletNodes.unsetAll();
            
        dirichletNodes[0]     = true;
        dirichletNodes.back() = true;

        // ////////////////////////////////////////////////////////////
        //    Create solution and rhs vectors
        // ////////////////////////////////////////////////////////////


        MatrixType hessianMatrix;
        RodAssembler<GridType::LeafGridView,2> rodAssembler(grid.leafView());
        
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
            trueObstacles[toplevel][i].upper(0) = 0.1 - x[i].r[0];
        }
        

        trustRegionObstacles.resize(toplevel+1);
        for (int i=0; i<=toplevel; i++)
            trustRegionObstacles[i].resize(grid.size(i, 1));

        // ////////////////////////////////////////////
        //   Adjust the solver to the new hierarchy
        // ////////////////////////////////////////////

        multigridStep.setNumberOfLevels(toplevel+1);
        multigridStep.ignoreNodes_ = &dirichletNodes;
        multigridStep.setSmoother(&presmoother, &postsmoother);

        for (int k=0; k<multigridStep.mgTransfer_.size(); k++)
            delete(multigridStep.mgTransfer_[k]);

        multigridStep.mgTransfer_.resize(toplevel);

        for (int i=0; i<multigridStep.mgTransfer_.size(); i++){
            TruncatedCompressedMGTransfer<CorrectionType>* newTransferOp = new TruncatedCompressedMGTransfer<CorrectionType>;
            newTransferOp->setup(grid,i,i+1);
            multigridStep.mgTransfer_[i] = newTransferOp;
        }

        // /////////////////////////////////////////////////////
        //   Trust-Region Solver
        // /////////////////////////////////////////////////////
        for (int i=0; i<maxNewtonSteps; i++) {

            std::cout << "-----------------------------------------------------------------------------" << std::endl;
            std::cout << "      Trust-Region Step Number: " << i 
                      << ",     radius: " << trustRegionRadius
                      << ",     energy: " << rodAssembler.computeEnergy(x) << std::endl;
            std::cout << "-----------------------------------------------------------------------------" << std::endl;

            rhs = 0;
            corr = 0;
            
            rodAssembler.assembleGradient(x, rhs);
            rodAssembler.assembleMatrix(x, hessianMatrix);
            
            rhs *= -1;

            // Create trust-region obstacle on grid0.maxLevel()
            setTrustRegionObstacles(trustRegionRadius,
                                    trustRegionObstacles[toplevel],
                                    trueObstacles[toplevel],
                                    dirichletNodes);

            dynamic_cast<MultigridStep<MatrixType,CorrectionType>*>(solver.iterationStep_)->setProblem(hessianMatrix, corr, rhs, toplevel+1);

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

             SolutionType newIterate = x;
             for (int j=0; j<newIterate.size(); j++) 
                 newIterate[j] = RigidBodyMotion<double,2>::exp(newIterate[j], corr[j]);

             /** \todo Don't always recompute oldEnergy */
             double oldEnergy = rodAssembler.computeEnergy(x); 
             double energy    = rodAssembler.computeEnergy(newIterate); 

             if (energy >= oldEnergy) 
                 DUNE_THROW(SolverError, "Richtung ist keine Abstiegsrichtung!");
                 
             //  Add correction to the current solution
             for (int j=0; j<x.size(); j++) 
                 x[j] = RigidBodyMotion<double,2>::exp(x[j], corr[j]);

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
        
        CorrectionType lagrangeMultipliers;
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

        std::cout << "  #### WARNING: function not transferred to the next level! #### " << std::endl;
        grid.adapt();
        x.resize(grid.size(1));

        //writeRod(x, "solutions/rod_1.result");
    }

 } catch (Exception e) {

    std::cout << e << std::endl;

 }
