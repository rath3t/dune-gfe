#include <config.h>

//#define DUNE_EXPRESSIONTEMPLATES
#include <dune/common/bitfield.hh>

#include <dune/grid/onedgrid.hh>

#include <dune/istl/io.hh>

#include <dune/disc/operators/p1operator.hh>

#include "../common/boundarypatch.hh"

#include "src/planarrodassembler.hh"
#include "../common/projectedblockgsstep.hh"
#include "../contact/src/contactmmgstep.hh"

#include "../common/iterativesolver.hh"

#include "../common/geomestimator.hh"
#include "../common/energynorm.hh"
#include "src/rodwriter.hh"

#include <dune/common/configparser.hh>

// Number of degrees of freedom: 
// 3 (x, y, theta) for a planar rod
const int blocksize = 3;

using namespace Dune;
using std::string;

void setTrustRegionObstacles(double trustRegionRadius,
                             std::vector<BoxConstraint<blocksize> >& trustRegionObstacles,
                             const std::vector<BoxConstraint<blocksize> >& trueObstacles,
                             const BitField& dirichletNodes)
{
    //std::cout << "True obstacles\n" << trueObstacles << std::endl;

    for (int j=0; j<trustRegionObstacles.size(); j++) {

        for (int k=0; k<blocksize; k++) {

            if (dirichletNodes[j*blocksize+k])
                continue;

            trustRegionObstacles[j].val[2*k] =
                (trueObstacles[j].val[2*k] < -1e10)
                ? std::min(-trustRegionRadius, trueObstacles[j].val[2*k+1] - trustRegionRadius)
                : trueObstacles[j].val[2*k];
                
            trustRegionObstacles[j].val[2*k+1] =
                (trueObstacles[j].val[2*k+1] >  1e10) 
                ? std::max(trustRegionRadius,trueObstacles[j].val[2*k] + trustRegionRadius)
                : trueObstacles[j].val[2*k+1];

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
    const int minLevel         = parameterSet.get("minLevel", int(0));
    const int maxLevel         = parameterSet.get("maxLevel", int(0));
    const int maxNewtonSteps   = parameterSet.get("maxNewtonSteps", int(0));
    const int numIt            = parameterSet.get("numIt", int(0));
    const int nu1              = parameterSet.get("nu1", int(0));
    const int nu2              = parameterSet.get("nu2", int(0));
    const int mu               = parameterSet.get("mu", int(0));
    const int baseIt           = parameterSet.get("baseIt", int(0));
    const double tolerance     = parameterSet.get("tolerance", double(0));
    const double baseTolerance = parameterSet.get("baseTolerance", double(0));
    
    // Problem settings
    const int numRodBaseElements = parameterSet.get("numRodBaseElements", int(0));
    
    // ///////////////////////////////////////
    //    Create the two grids
    // ///////////////////////////////////////
    typedef OneDGrid GridType;
    GridType grid(numRodBaseElements, 0, 1);

    std::vector<std::vector<BoxConstraint<3> > > trustRegionObstacles(1);
    std::vector<BitField> hasObstacle(1);
    std::vector<BitField> dirichletNodes(1);

    // ////////////////////////////////
    //   Create a multigrid solver
    // ////////////////////////////////

    // First create a gauss-seidel base solver
    ProjectedBlockGSStep<MatrixType, VectorType> baseSolverStep;

    EnergyNorm<MatrixType, VectorType> baseEnergyNorm(baseSolverStep);

    IterativeSolver<MatrixType, VectorType> baseSolver(&baseSolverStep,
                                                       baseIt,
                                                       baseTolerance,
                                                       &baseEnergyNorm,
                                                       Solver::QUIET);

    // Make pre and postsmoothers
    ProjectedBlockGSStep<MatrixType, VectorType> presmoother;
    ProjectedBlockGSStep<MatrixType, VectorType> postsmoother;

    ContactMMGStep<MatrixType, VectorType> contactMMGStep(1);

    contactMMGStep.setMGType(mu, nu1, nu2);
    contactMMGStep.dirichletNodes_    = &dirichletNodes;
    contactMMGStep.basesolver_        = &baseSolver;
    contactMMGStep.presmoother_       = &presmoother;
    contactMMGStep.postsmoother_      = &postsmoother;    
    contactMMGStep.hasObstacle_       = &hasObstacle;
    contactMMGStep.obstacles_         = &trustRegionObstacles;
    contactMMGStep.verbosity_         = Solver::QUIET;



    EnergyNorm<MatrixType, VectorType> energyNorm(contactMMGStep);

    IterativeSolver<MatrixType, VectorType> solver(&contactMMGStep,
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
            
            dirichletNodes[i].resize( blocksize * grid.size(i,1), false );
            
            for (int j=0; j<blocksize; j++) {
                dirichletNodes[i][j] = true;
                dirichletNodes[i][dirichletNodes[i].size()-1-j] = true;
            }
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
        
        std::vector<std::vector<BoxConstraint<3> > > trueObstacles(toplevel+1);
        trustRegionObstacles.resize(toplevel+1);
        
        for (int i=0; i<toplevel+1; i++) {
            trueObstacles[i].resize(grid.size(i,1));
            trustRegionObstacles[i].resize(grid.size(i,1));
        }
        
        for (int i=0; i<trueObstacles[toplevel].size(); i++) {
            trueObstacles[toplevel][i].clear();
            //trueObstacles[toplevel][i].val[0] =     - x[i][0];
            trueObstacles[toplevel][i].val[1] = 0.1 - x[i][0];
        }
        

        trustRegionObstacles.resize(toplevel+1);
        for (int i=0; i<=toplevel; i++)
            trustRegionObstacles[i].resize(grid.size(i, 1));

        // ////////////////////////////////////
        //   Create the transfer operators
        // ////////////////////////////////////
        for (int k=0; k<contactMMGStep.mgTransfer_.size(); k++)
            delete(contactMMGStep.mgTransfer_[k]);

        contactMMGStep.mgTransfer_.resize(toplevel);

        for (int i=0; i<contactMMGStep.mgTransfer_.size(); i++){
            TruncatedMGTransfer<VectorType>* newTransferOp = new TruncatedMGTransfer<VectorType>;
            newTransferOp->setup(grid,i,i+1);
            contactMMGStep.mgTransfer_[i] = newTransferOp;
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

            contactMMGStep.preprocess();


            // /////////////////////////////
            //    Solve !
            // /////////////////////////////
             solver.solve();

             corr = contactMMGStep.getSol();

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
