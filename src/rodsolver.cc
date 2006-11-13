#include <dune/common/bitfield.hh>

#include <dune/istl/io.hh>

#include "../common/trustregiongsstep.hh"
#include "../contact/src/contactmmgstep.hh"

#include "../solver/iterativesolver.hh"

#include "../common/energynorm.hh"

#include "configuration.hh"
#include "quaternion.hh"

// Number of degrees of freedom: 
// 7 (x, y, z, q_1, q_2, q_3, q_4) for a spatial rod
//const int blocksize = 6;

template <class GridType>
void RodSolver<GridType>::
setTrustRegionObstacles(double trustRegionRadius,
                        std::vector<BoxConstraint<blocksize> >& trustRegionObstacles)
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


template <class GridType>
void RodSolver<GridType>::setup(const GridType& grid,
                                const Dune::RodAssembler<GridType>* rodAssembler,
                                const SolutionType& x,
                                int maxTrustRegionSteps,
                                double initialTrustRegionRadius,
                                int multigridIterations,
                                double mgTolerance,
                                int mu, 
                                int nu1,
                                int nu2,
                                int baseIterations,
                                double baseTolerance)
{
    using namespace Dune;

    grid_ = &grid;
    rodAssembler_             = rodAssembler;
    x_                        = x;
    maxTrustRegionSteps_      = maxTrustRegionSteps;
    initialTrustRegionRadius_ = initialTrustRegionRadius;
    multigridIterations_      = multigridIterations;
    qpTolerance_              = mgTolerance;
    mu_                       = mu;
    nu1_                      = nu1;
    nu2_                      = nu2;
    baseIt_                   = baseIterations;
    baseTolerance_            = baseTolerance;

    int numLevels = grid_->maxLevel()+1;

    // ////////////////////////////////////////////
    //   Construct array with the Dirichlet nodes
    // ////////////////////////////////////////////
    dirichletNodes_.resize(numLevels);
    for (int i=0; i<numLevels; i++) {
        
        dirichletNodes_[i].resize( blocksize * grid.size(i,1), false );
        
        for (int j=0; j<blocksize; j++) {
            dirichletNodes_[i][j] = true;
            dirichletNodes_[i][dirichletNodes_[i].size()-1-j] = true;
        }

    }

    // ////////////////////////////////
    //   Create a multigrid solver
    // ////////////////////////////////

    // First create a gauss-seidel base solver
    TrustRegionGSStep<MatrixType, CorrectionType>* baseSolverStep = new TrustRegionGSStep<MatrixType, CorrectionType>;

    EnergyNorm<MatrixType, CorrectionType>* baseEnergyNorm = new EnergyNorm<MatrixType, CorrectionType>(*baseSolverStep);

    IterativeSolver<MatrixType, CorrectionType>* baseSolver = new IterativeSolver<MatrixType, CorrectionType>;
    baseSolver->iterationStep = baseSolverStep;
    baseSolver->numIt = baseIt_;
    baseSolver->verbosity_ = Solver::QUIET;
    baseSolver->errorNorm_ = baseEnergyNorm;
    baseSolver->tolerance_ = baseTolerance_;

    // Make pre and postsmoothers
    TrustRegionGSStep<MatrixType, CorrectionType>* presmoother  = new TrustRegionGSStep<MatrixType, CorrectionType>;
    TrustRegionGSStep<MatrixType, CorrectionType>* postsmoother = new TrustRegionGSStep<MatrixType, CorrectionType>;

    ContactMMGStep<MatrixType, CorrectionType>* mmgStep = new ContactMMGStep<MatrixType, CorrectionType>(numLevels);

    mmgStep->setMGType(mu_, nu1_, nu2_);
    mmgStep->dirichletNodes_    = &dirichletNodes_;
    mmgStep->basesolver_        = baseSolver;
    mmgStep->presmoother_       = presmoother;
    mmgStep->postsmoother_      = postsmoother;    
    mmgStep->hasObstacle_       = &hasObstacle_;
    mmgStep->obstacles_         = &trustRegionObstacles_;
    mmgStep->verbosity_         = Solver::FULL;



    EnergyNorm<MatrixType, CorrectionType>* energyNorm = new EnergyNorm<MatrixType, CorrectionType>(*mmgStep);

    mmgSolver_ = new IterativeSolver<MatrixType, CorrectionType>;
    mmgSolver_->iterationStep = mmgStep;
    mmgSolver_->numIt         = multigridIterations_;
    mmgSolver_->verbosity_    = Solver::FULL;
    mmgSolver_->errorNorm_    = energyNorm;
    mmgSolver_->tolerance_    = qpTolerance_;

    // ////////////////////////////////////////////////////////////
    //    Create Hessian matrix and its occupation structure
    // ////////////////////////////////////////////////////////////
    
    if (hessianMatrix_)
        delete hessianMatrix_;

    hessianMatrix_ = new MatrixType;
    MatrixIndexSet indices(grid_->size(1), grid_->size(1));
    rodAssembler_->getNeighborsPerVertex(indices);
    indices.exportIdx(*hessianMatrix_);
    
    // //////////////////////////////////////////////////////////
    //   Create obstacles
    // //////////////////////////////////////////////////////////
    
    hasObstacle_.resize(numLevels);
    for (int i=0; i<hasObstacle_.size(); i++)
        hasObstacle_[i].resize(grid_->size(i, 1),true);
    
    trustRegionObstacles_.resize(numLevels);
    
    for (int i=0; i<numLevels; i++)
        trustRegionObstacles_[i].resize(grid_->size(i,1));
    
    // ////////////////////////////////////
    //   Create the transfer operators
    // ////////////////////////////////////
    for (int k=0; k<mmgStep->mgTransfer_.size(); k++)
        delete(mmgStep->mgTransfer_[k]);
    
    mmgStep->mgTransfer_.resize(numLevels-1);
    
    for (int i=0; i<mmgStep->mgTransfer_.size(); i++){
        TruncatedMGTransfer<CorrectionType>* newTransferOp = new TruncatedMGTransfer<CorrectionType>;
        newTransferOp->setup(*grid_,i,i+1);
        mmgStep->mgTransfer_[i] = newTransferOp;
    }
    
}

template <class GridType>
void RodSolver<GridType>::solve()
{
    double trustRegionRadius = initialTrustRegionRadius_;

    
    // /////////////////////////////////////////////////////
    //   Trust-Region Solver
    // /////////////////////////////////////////////////////
    for (int i=0; i<maxTrustRegionSteps_; i++) {
        
        std::cout << "----------------------------------------------------" << std::endl;
        std::cout << "      Trust-Region Step Number: " << i << std::endl;
        std::cout << "----------------------------------------------------" << std::endl;
        
        std::cout << "### Trust-Region Radius: " << trustRegionRadius << " ###" << std::endl;
        
        CorrectionType rhs;
        CorrectionType corr(x_.size());
        corr = 0;

        //rodAssembler.setParameters(0,0,0,1,1,1);
        std::cout << "Rod energy: " <<rodAssembler_->computeEnergy(x_) << std::endl;
        rodAssembler_->assembleGradient(x_, rhs);
        rodAssembler_->assembleMatrix(x_, *hessianMatrix_);
        
        rhs *= -1;

        // Create trust-region obstacle on maxlevel
        setTrustRegionObstacles(trustRegionRadius,
                                trustRegionObstacles_[grid_->maxLevel()]);
        
        dynamic_cast<Dune::MultigridStep<MatrixType,CorrectionType>*>(mmgSolver_->iterationStep)->setProblem(*hessianMatrix_, corr, rhs, grid_->maxLevel()+1);
        
        mmgSolver_->preprocess();
        
        dynamic_cast<Dune::MultigridStep<MatrixType,CorrectionType>*>(mmgSolver_->iterationStep)->preprocess();
        
        
        // /////////////////////////////
        //    Solve !
        // /////////////////////////////
        mmgSolver_->solve();
        
        corr = dynamic_cast<Dune::MultigridStep<MatrixType,CorrectionType>*>(mmgSolver_->iterationStep)->getSol();
        
        std::cout << "Correction: " << std::endl << corr << std::endl;
        
        printf("infinity norm of the correction: %g\n", corr.infinity_norm());
        if (corr.infinity_norm() < 1e-10) {
            std::cout << "CORRECTION IS SMALL ENOUGH" << std::endl;
            break;
        }
        
        // ////////////////////////////////////////////////////
        //   Check whether trust-region step can be accepted
        // ////////////////////////////////////////////////////
        
        SolutionType newIterate = x_;
        for (int j=0; j<newIterate.size(); j++) {
            
            // Add translational correction
            for (int k=0; k<3; k++)
                newIterate[j].r[k] += corr[j][k];
            
            // Add rotational correction
            Quaternion<double> qCorr = Quaternion<double>::exp(corr[j][3], corr[j][4], corr[j][5]);
            newIterate[j].q = newIterate[j].q.mult(qCorr);
            //newIterate[j].q = qCorr.mult(newIterate[j].q);
            
        }
        
        
        std::cout << "newIterate: \n";
#if 1
        for (int j=0; j<newIterate.size(); j++)
            std::cout << newIterate[j] << std::endl;
#endif     
        
        /** \todo Don't always recompute oldEnergy */
        double oldEnergy = rodAssembler_->computeEnergy(x_);
        double energy    = rodAssembler_->computeEnergy(newIterate); 
        
        // compute the model decrease
        // It is $ m(x) - m(x+s) = -<g,s> - 0.5 <s, Hs>
        // Note that rhs = -g
        CorrectionType tmp(corr.size());
        tmp = 0;
        hessianMatrix_->mmv(corr, tmp);
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
            
            x_ = newIterate;
            trustRegionRadius *= 2;
            
        } else if ( (oldEnergy-energy) / modelDecrease > 0.01) {
            // successful iteration
            x_ = newIterate;
            
        } else {
            // unsuccessful iteration
            trustRegionRadius /= 2;
            std::cout << "Unsuccessful iteration!" << std::endl;
        }
        
        //  Write current energy
        std::cout << "--- Current energy: " << energy << " ---" << std::endl;
    }
    
}
