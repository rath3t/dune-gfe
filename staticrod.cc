#include <config.h>

//#define DUNE_EXPRESSIONTEMPLATES
#include <dune/grid/onedgrid.hh>

#include <dune/istl/io.hh>

#include "../common/boundarypatch.hh"
#include <dune/common/bitfield.hh>

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
                             std::vector<BoxConstraint<double,blocksize> >& trustRegionObstacles,
                             const std::vector<BoxConstraint<double,blocksize> >& trueObstacles,
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

int main (int argc, char *argv[]) try
{
    // Some types that I need
    typedef BCRSMatrix<FieldMatrix<double, blocksize, blocksize> > MatrixType;
    typedef BlockVector<FieldVector<double, blocksize> >     VectorType;

    // parse data file
    ConfigParser parameterSet;
    parameterSet.parseFile("staticrod.parset");

    // read solver settings
    const int maxLevel         = parameterSet.get("maxLevel", int(0));
    double loadIncrement       = parameterSet.get("loadIncrement", double(0));
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
    typedef OneDGrid RodGridType;
    RodGridType rod(numRodBaseElements, 0, 1);

    // refine uniformly until maxLevel
    for (int i=0; i<maxLevel; i++)
        rod.globalRefine(1);

    int maxlevel = rod.maxLevel();
    int numRodElements = rod.size(maxlevel, 0);

    
    std::vector<BitField> dirichletNodes;
    dirichletNodes.resize(maxLevel+1);
    for (int i=0; i<=maxlevel; i++) {

        dirichletNodes[i].resize( blocksize * rod.size(i,1), false );

        for (int j=0; j<blocksize; j++) {
            dirichletNodes[i][j] = true;
            dirichletNodes[i][dirichletNodes[i].size()-1-j] = true;
        }
    }

    // ////////////////////////////////////////////////////////////
    //    Create solution and rhs vectors
    // ////////////////////////////////////////////////////////////

    VectorType rhs;
    VectorType x;
    VectorType corr;

    MatrixType hessianMatrix;
    PlanarRodAssembler<RodGridType,4> rodAssembler(rod);
    
    rodAssembler.setParameters(1, 100, 100);

    MatrixIndexSet indices(numRodElements+1, numRodElements+1);
    rodAssembler.getNeighborsPerVertex(indices);
    indices.exportIdx(hessianMatrix);

    rhs.resize(rod.size(maxlevel,1));
    x.resize(rod.size(maxlevel,1));
    corr.resize(rod.size(maxlevel,1));
    
    // Initial solution
    x = 0;

    for (int i=0; i<numRodElements+1; i++) {
        x[i][0] = i/((double)numRodElements);
        x[i][1] = 0;
        x[i][2] = M_PI/2;
    }

    x[0][0] = x[numRodElements][0] = 0;
    x[0][1] = x[numRodElements][1] = 0;

    x[0][2] = 0;
    x[numRodElements][2] = 2*M_PI;

    // //////////////////////////////////////////////////////////
    //   Create obstacles
    // //////////////////////////////////////////////////////////

    std::vector<BitField> hasObstacle;
    hasObstacle.resize(maxLevel+1);
    for (int i=0; i<hasObstacle.size(); i++) {
        hasObstacle[i].resize(rod.size(i, 1));
        hasObstacle[i].setAll();
    }

    std::vector<std::vector<BoxConstraint<double,3> > > trueObstacles(maxlevel+1);
    std::vector<std::vector<BoxConstraint<double,3> > > trustRegionObstacles(maxlevel+1);

    for (int i=0; i<maxlevel+1; i++) {
        trueObstacles[i].resize(rod.size(i,1));
        trustRegionObstacles[i].resize(rod.size(i,1));
    }

    for (int i=0; i<trueObstacles[maxlevel].size(); i++) {
        trueObstacles[maxlevel][i].clear();
        //trueObstacles[maxlevel][i].val[0] =     - x[i][0];
        trueObstacles[maxlevel][i].val[1] = 0.1 - x[i][0];
    }

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

    ContactMMGStep<MatrixType, VectorType> contactMMGStep(maxlevel+1);

    contactMMGStep.setMGType(mu, nu1, nu2);
    contactMMGStep.dirichletNodes_    = &dirichletNodes;
    contactMMGStep.basesolver_        = &baseSolver;
    contactMMGStep.presmoother_       = &presmoother;
    contactMMGStep.postsmoother_      = &postsmoother;    
    contactMMGStep.hasObstacle_       = &hasObstacle;
    contactMMGStep.obstacles_         = &trustRegionObstacles;

    // Create the transfer operators
    contactMMGStep.mgTransfer_.resize(maxlevel);
    for (int i=0; i<contactMMGStep.mgTransfer_.size(); i++){
        TruncatedMGTransfer<VectorType>* newTransferOp = new TruncatedMGTransfer<VectorType>;
        newTransferOp->setup(rod,i,i+1);
        contactMMGStep.mgTransfer_[i] = newTransferOp;
    }

    EnergyNorm<MatrixType, VectorType> energyNorm(contactMMGStep);

    IterativeSolver<MatrixType, VectorType> solver(&contactMMGStep,
                                                   numIt,
                                                   tolerance,
                                                   &energyNorm,
                                                   Solver::QUIET);

    // ///////////////////////////////////////////////////
    //   Do a homotopy of the material parameters
    // ///////////////////////////////////////////////////
    double loadFactor = 0;
    double trustRegionRadius = 0.1;

    do {

        loadFactor += loadIncrement;

        std::cout << "####################################################" << std::endl;
        std::cout << "New load factor: " << loadFactor 
                  << "    new load increment: " << loadIncrement << std::endl;
        std::cout << "####################################################" << std::endl;

        // The continuation variable determines the material parameters
        double A1 = loadFactor * 10000;
        double A3 = loadFactor * 10000;
        rodAssembler.setParameters(1, A1, A3);

        // /////////////////////////////////////////////////////
        //   Newton Solver
        // /////////////////////////////////////////////////////

        for (int j=0; j<maxNewtonSteps; j++) {

            std::cout << "----------------------------------------------------" << std::endl;
            std::cout << "      Newton Step Number: " << j << std::endl;
            std::cout << "----------------------------------------------------" << std::endl;

            rhs = 0;
            corr = 0;

            //std::cout <<"Solution: " << x << std::endl;
            //exit(0);
            rodAssembler.assembleGradient(x, rhs);
            rodAssembler.assembleMatrix(x, hessianMatrix);

            rhs *= -1;

            // Apply trust-region obstacles
            setTrustRegionObstacles(trustRegionRadius,
                                    trustRegionObstacles[maxlevel],
                                    trueObstacles[maxlevel],
                                    dirichletNodes[maxlevel]);

            //std::cout << "rhs: " << std::endl << rhs << std::endl;
            //std::cout << "Trust Region obstacles:" << std::endl;
            //std::cout << (*contactMMGStep.obstacles_)[maxlevel] << std::endl;

            solver.iterationStep_->setProblem(hessianMatrix, corr, rhs);

            solver.preprocess();
            contactMMGStep.preprocess();

            // /////////////////////////////
            //    Solve !
            // /////////////////////////////
             solver.solve();

             corr = contactMMGStep.getSol();

             //std::cout << "Correction: \n" << corr << std::endl;

             // line search
             printf("------  Line Search ---------\n");
             int lSSteps = 10;
             double smallestEnergy = std::numeric_limits<double>::max();
             double smallestFactor = 1;
             for (int k=0; k<lSSteps; k++) {

                 double factor = double(k)/(lSSteps-1);
                 VectorType sCorr = corr;
                 sCorr *= factor;
                 sCorr += x;

                 double energy = rodAssembler.computeEnergy(sCorr);

                 if (energy < smallestEnergy) {
                     smallestEnergy = energy;
                     smallestFactor = factor;
                 }
                 printf("factor: %g,  energy: %e\n", factor, energy);
             }

             std::cout << "Damping factor: " << smallestFactor << std::endl;
             //exit(0);
             //  Add correction to the current solution
             x.axpy(smallestFactor, corr);

             // Output result
             //std::cout << "Solution:" << std::endl << x << std::endl;

             printf("infinity norm of the correction: %g\n", smallestFactor*corr.infinity_norm());
             if (smallestFactor*corr.infinity_norm() < 1e-8)
                 break;

             // Subtract correction from the current obstacle
             for (int k=0; k<corr.size(); k++) {
                 FieldVector<double, blocksize> tmp = corr[k];
                 tmp *= smallestFactor;
                 trueObstacles[maxlevel][k] -= tmp;
             }

        }
        
        // Write Lagrange multiplyers
        std::stringstream a1AsAscii, a3AsAscii;
        a1AsAscii << A1;
        a3AsAscii << A3;
        std::string lagrangeFilename = "pressure/lagrange_" + a1AsAscii.str() + "_" + a3AsAscii.str();
        std::ofstream lagrangeFile(lagrangeFilename.c_str());
        
        VectorType lagrangeMultipliers;
        rodAssembler.assembleGradient(x, lagrangeMultipliers);
        lagrangeFile << lagrangeMultipliers << std::endl;
        
        // Write result grid
        std::string solutionFilename = "solutions/rod_" + a1AsAscii.str() 
            + "_" + a3AsAscii.str() + ".result";
        writeRod(x, solutionFilename);

        //break;
    } while (loadFactor < 1);

 } catch (Exception e) {

    std::cout << e << std::endl;

 }
