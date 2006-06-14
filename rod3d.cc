#include <config.h>

//#define DUNE_EXPRESSIONTEMPLATES
#include <dune/grid/onedgrid.hh>

#include <dune/istl/io.hh>

#include <dune/common/bitfield.hh>
#include "src/quaternion.hh"

#include "src/rodassembler.hh"
#include "../common/trustregiongsstep.hh"
#include "../contact/src/contactmmgstep.hh"

#include "../solver/iterativesolver.hh"

#include "../common/geomestimator.hh"
#include "../common/refinegrid.hh"
#include "../common/energynorm.hh"

#include <dune/common/configparser.hh>
#include "src/configuration.hh"
#include "src/rodwriter.hh"

// Number of degrees of freedom: 
// 7 (x, y, z, q_1, q_2, q_3, q_4) for a spatial rod
const int blocksize = 6;

using namespace Dune;
using std::string;


void setTrustRegionObstacles(double trustRegionRadius,
                             SimpleVector<BoxConstraint<blocksize> >& trustRegionObstacles)
{
    for (int j=0; j<trustRegionObstacles.size(); j++) {

        for (int k=0; k<blocksize; k++) {

//             if (totalDirichletNodes[j*dim+k])
//                 continue;

            trustRegionObstacles[j].val[2*k]   = -trustRegionRadius;
            trustRegionObstacles[j].val[2*k+1] =  trustRegionRadius;

        }
        
    }

    //std::cout << trustRegionObstacles << std::endl;
//     exit(0);
}


int main (int argc, char *argv[]) try
{
    // Some types that I need
    typedef BCRSMatrix<FieldMatrix<double, blocksize, blocksize> > MatrixType;
    typedef BlockVector<FieldVector<double, blocksize> >           CorrectionType;
    typedef std::vector<Configuration>                              SolutionType;

    // parse data file
    ConfigParser parameterSet;
    parameterSet.parseFile("rod3d.parset");

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
    //    Create the grid
    // ///////////////////////////////////////
    typedef OneDGrid<1,1> GridType;
    GridType grid(numRodBaseElements, 0, 1);

    grid.globalRefine(minLevel);

    Array<SimpleVector<BoxConstraint<blocksize> > > trustRegionObstacles(1);
    Array<BitField> hasObstacle(1);
    Array<BitField> dirichletNodes(1);

    // ////////////////////////////////
    //   Create a multigrid solver
    // ////////////////////////////////

    // First create a gauss-seidel base solver
    TrustRegionGSStep<MatrixType, CorrectionType> baseSolverStep;

    EnergyNorm<MatrixType, CorrectionType> baseEnergyNorm(baseSolverStep);

    IterativeSolver<MatrixType, CorrectionType> baseSolver;
    baseSolver.iterationStep = &baseSolverStep;
    baseSolver.numIt = baseIt;
    baseSolver.verbosity_ = Solver::QUIET;
    baseSolver.errorNorm_ = &baseEnergyNorm;
    baseSolver.tolerance_ = baseTolerance;

    // Make pre and postsmoothers
    TrustRegionGSStep<MatrixType, CorrectionType> presmoother;
    TrustRegionGSStep<MatrixType, CorrectionType> postsmoother;

    ContactMMGStep<MatrixType, CorrectionType> contactMMGStep(1);

    contactMMGStep.setMGType(mu, nu1, nu2);
    contactMMGStep.dirichletNodes_    = &dirichletNodes;
    contactMMGStep.basesolver_        = &baseSolver;
    contactMMGStep.presmoother_       = &presmoother;
    contactMMGStep.postsmoother_      = &postsmoother;    
    contactMMGStep.hasObstacle_       = &hasObstacle;
    contactMMGStep.obstacles_         = &trustRegionObstacles;
    contactMMGStep.verbosity_         = Solver::FULL;



    EnergyNorm<MatrixType, CorrectionType> energyNorm(contactMMGStep);

    IterativeSolver<MatrixType, CorrectionType> solver;
    solver.iterationStep = &contactMMGStep;
    solver.numIt = numIt;
    solver.verbosity_ = Solver::FULL;
    solver.errorNorm_ = &energyNorm;
    solver.tolerance_ = tolerance;

    double trustRegionRadius = 1;

    SolutionType x(grid.size(grid.maxLevel(),1));

    // //////////////////////////
    //   Initial solution
    // //////////////////////////

    for (int i=0; i<x.size(); i++) {
        x[i].r[0] = 0;    // x
        x[i].r[1] = 0;                 // y
        x[i].r[2] = double(i)/(x.size()-1);                 // z
        //x[i].r[2] = i+5;
        x[i].q = Quaternion<double>::identity();
    }

    x[x.size()-1].r[0] = 0;
    x[x.size()-1].r[1] = 0.1;
    x[x.size()-1].r[2] = 0;
#if 0
    x[x.size()-1].q[0] = 0;
    x[x.size()-1].q[1] = 0;
    x[x.size()-1].q[2] = 1/sqrt(2);
    x[x.size()-1].q[3] = 1/sqrt(2);
#endif

    std::cout << "Left boundary orientation:" << std::endl;
    std::cout << "director 0:  " << x[0].q.director(0) << std::endl;
    std::cout << "director 1:  " << x[0].q.director(1) << std::endl;
    std::cout << "director 2:  " << x[0].q.director(2) << std::endl;
    std::cout << std::endl;
    std::cout << "Right boundary orientation:" << std::endl;
    std::cout << "director 0:  " << x[x.size()-1].q.director(0) << std::endl;
    std::cout << "director 1:  " << x[x.size()-1].q.director(1) << std::endl;
    std::cout << "director 2:  " << x[x.size()-1].q.director(2) << std::endl;
//     exit(0);

    //x[0].r[2] = -1;

    // /////////////////////////////////////////////////////////////////////
    //   Refinement Loop
    // /////////////////////////////////////////////////////////////////////
    
    for (int toplevel=minLevel; toplevel<=maxLevel; toplevel++) {
        
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
        RodAssembler<GridType> rodAssembler(grid);
        rodAssembler.setShapeAndMaterial(0.01, 0.0001, 0.0001, 2.5e5, 0.3);

        std::cout << "Energy: " << rodAssembler.computeEnergy(x) << std::endl;

        MatrixIndexSet indices(grid.size(toplevel,1), grid.size(toplevel,1));
        rodAssembler.getNeighborsPerVertex(indices);
        indices.exportIdx(hessianMatrix);

        CorrectionType rhs, corr;
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
        
        trustRegionObstacles.resize(toplevel+1);
        
        for (int i=0; i<toplevel+1; i++) {
            trustRegionObstacles[i].resize(grid.size(i,1));
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
            TruncatedMGTransfer<CorrectionType>* newTransferOp = new TruncatedMGTransfer<CorrectionType>;
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

            std::cout << "### Trust-Region Radius: " << trustRegionRadius << " ###" << std::endl;

            rhs = 0;
            corr = 0;
            
            rodAssembler.assembleGradient(x, rhs);
            rodAssembler.assembleMatrix(x, hessianMatrix);
            
            rhs *= -1;

            // Create trust-region obstacle on maxlevel
            setTrustRegionObstacles(trustRegionRadius,
                                    trustRegionObstacles[toplevel]);

            dynamic_cast<MultigridStep<MatrixType,CorrectionType>*>(solver.iterationStep)->setProblem(hessianMatrix, corr, rhs, toplevel+1);

            solver.preprocess();

            contactMMGStep.preprocess();


            // /////////////////////////////
            //    Solve !
            // /////////////////////////////
             solver.solve();

             corr = contactMMGStep.getSol();

             //std::cout << "Correction: " << std::endl << corr << std::endl;

             printf("infinity norm of the correction: %g\n", corr.infinity_norm());
             if (corr.infinity_norm() < 1e-5) {
                 std::cout << "CORRECTION IS SMALL ENOUGH" << std::endl;
                 break;
             }

             // ////////////////////////////////////////////////////
             //   Check whether trust-region step can be accepted
             // ////////////////////////////////////////////////////
             
             SolutionType newIterate = x;
             for (int j=0; j<newIterate.size(); j++) {

                 // Add translational correction
                 for (int k=0; k<3; k++)
                     newIterate[j].r[k] += corr[j][k];

                 // Add rotational correction
                 Quaternion<double> qCorr = Quaternion<double>::exp(corr[j][3], corr[j][4], corr[j][5]);
                 newIterate[j].q = newIterate[j].q.mult(qCorr);

             }

#if 0
             std::cout << "newIterate: \n";
             for (int j=0; j<newIterate.size(); j++)
                 printf("%d:  (%g %g %g)  (%g %g %g %g)\n", j,
                        newIterate[j].r[0],newIterate[j].r[1],newIterate[j].r[2],
                        newIterate[j].q[0],newIterate[j].q[1],newIterate[j].q[2],newIterate[j].q[3]);
#endif     
            
             /** \todo Don't always recompute oldEnergy */
             double oldEnergy = rodAssembler.computeEnergy(x); 
             double energy    = rodAssembler.computeEnergy(newIterate); 

             // compute the model decrease
             // It is $ m(x) - m(x+s) = -<g,s> - 0.5 <s, Hs>
             // Note that rhs = -g
             CorrectionType tmp(corr.size());
             tmp = 0;
             hessianMatrix.mmv(corr, tmp);
             double modelDecrease = (rhs*corr) - 0.5 * (corr*tmp);
             
             std::cout << "Model decrease: " << modelDecrease 
                       << ",  functional decrease: " << oldEnergy - energy << std::endl;

             assert(modelDecrease >= 0);

             if (energy >= oldEnergy) {
                 printf("Richtung ist keine Abstiegsrichtung!\n");
//                  std::cout << "corr[0]\n" << corr[0] << std::endl;
                 //exit(0);
             }
              
             // //////////////////////////////////////////////
             //   Check for acceptance of the step
             // //////////////////////////////////////////////
             if ( (oldEnergy-energy) / modelDecrease > 0.9) {
                 // very successful iteration

                  x = newIterate;
                  trustRegionRadius *= 2;

             } else if ( (oldEnergy-energy) / modelDecrease > 0.01) {
                 // successful iteration
                  x = newIterate;

             } else {
                 // unsuccessful iteration
                 trustRegionRadius /= 2;
                 std::cout << "Unsuccessful iteration!" << std::endl;
             }

             //  Write current energy
             std::cout << "--- Current energy: " << energy << " ---" << std::endl;
        }
        
        // //////////////////////////////
        //   Output result
        // //////////////////////////////
        writeRod(x, "rod3d.result");
        BlockVector<FieldVector<double, 6> > strain(x.size()-1);
        rodAssembler.getStrain(x,strain);
        //std::cout << strain << std::endl;
        //exit(0);

        writeRod(x, strain, "rod3d.strain");

    }


 } catch (Exception e) {

    std::cout << e << std::endl;

 }
