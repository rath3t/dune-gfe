#include <dune/common/bitfield.hh>

#include <dune/istl/io.hh>

#include <dune/disc/functions/p1function.hh>
#include <dune/disc/miscoperators/laplace.hh>
#include <dune/disc/operators/p1operator.hh>

#include <dune/ag-common/trustregiongsstep.hh>
#include <dune/ag-common/solvers/mmgstep.hh>
#include <dune/ag-common/contactobsrestrict.hh>
#include <dune/ag-common/iterativesolver.hh>

#include <dune/ag-common/norms/energynorm.hh>
#include <dune/ag-common/norms/h1seminorm.hh>

#include "configuration.hh"
#include "quaternion.hh"

// for debugging
#include "../test/fdcheck.hh"

// Number of degrees of freedom: 
// 7 (x, y, z, q_1, q_2, q_3, q_4) for a spatial rod
//const int blocksize = 6;

template <class GridType>
void RodSolver<GridType>::
setTrustRegionObstacles(double trustRegionRadius,
                        std::vector<BoxConstraint<field_type,blocksize> >& trustRegionObstacles)
{
    for (int j=0; j<trustRegionObstacles.size(); j++) {

        for (int k=0; k<blocksize; k++) {

            trustRegionObstacles[j].lower(k) = -trustRegionRadius;
            trustRegionObstacles[j].upper(k) =  trustRegionRadius;

        }
        
    }

    //std::cout << trustRegionObstacles << std::endl;
//     exit(0);
}


template <class GridType>
void RodSolver<GridType>::setup(const GridType& grid,
                                const RodAssembler<GridType>* rodAssembler,
                                const SolutionType& x,
                                const BlockBitField<blocksize>& dirichletNodes,
                                double tolerance,
                                int maxTrustRegionSteps,
                                double initialTrustRegionRadius,
                                int multigridIterations,
                                double mgTolerance,
                                int mu, 
                                int nu1,
                                int nu2,
                                int baseIterations,
                                double baseTolerance,
                                bool instrumented)
{
    using namespace Dune;

    grid_ = &grid;
    rodAssembler_             = rodAssembler;
    x_                        = x;
    tolerance_                = tolerance;
    maxTrustRegionSteps_      = maxTrustRegionSteps;
    initialTrustRegionRadius_ = initialTrustRegionRadius;
    multigridIterations_      = multigridIterations;
    qpTolerance_              = mgTolerance;
    mu_                       = mu;
    nu1_                      = nu1;
    nu2_                      = nu2;
    baseIt_                   = baseIterations;
    baseTolerance_            = baseTolerance;
    instrumented_             = instrumented;

    int numLevels = grid_->maxLevel()+1;

    // ////////////////////////////////
    //   Create a multigrid solver
    // ////////////////////////////////

    // First create a Gauss-seidel base solver
    TrustRegionGSStep<MatrixType, CorrectionType>* baseSolverStep = new TrustRegionGSStep<MatrixType, CorrectionType>;

    EnergyNorm<MatrixType, CorrectionType>* baseEnergyNorm = new EnergyNorm<MatrixType, CorrectionType>(*baseSolverStep);

    IterativeSolver<CorrectionType>* baseSolver 
        = new IterativeSolver<CorrectionType>(baseSolverStep,
                                                          baseIt_,
                                                          baseTolerance_,
                                                          baseEnergyNorm,
                                                          Solver::QUIET);

    // Make pre and postsmoothers
    TrustRegionGSStep<MatrixType, CorrectionType>* presmoother  = new TrustRegionGSStep<MatrixType, CorrectionType>;
    TrustRegionGSStep<MatrixType, CorrectionType>* postsmoother = new TrustRegionGSStep<MatrixType, CorrectionType>;

    MonotoneMGStep<MatrixType, CorrectionType>* mmgStep = new MonotoneMGStep<MatrixType, CorrectionType>(numLevels);

    mmgStep->setMGType(mu_, nu1_, nu2_);
    mmgStep->ignoreNodes        = &dirichletNodes;
    mmgStep->basesolver_        = baseSolver;
    mmgStep->presmoother_       = presmoother;
    mmgStep->postsmoother_      = postsmoother; 
    mmgStep->obstacleRestrictor_= new ContactObsRestriction<CorrectionType>();
    mmgStep->hasObstacle_       = &hasObstacle_;
    mmgStep->obstacles_         = &trustRegionObstacles_;
    mmgStep->verbosity_         = Solver::QUIET;

    // //////////////////////////////////////////////////////////////////////////////////////
    //   Assemble a Laplace matrix to create a norm that's equivalent to the H1-norm
    // //////////////////////////////////////////////////////////////////////////////////////
    LeafP1Function<GridType,double> u(grid),f(grid);
    LaplaceLocalStiffness<typename GridType::LeafGridView,double> laplaceStiffness;
    LeafP1OperatorAssembler<GridType,double,1>* A = new LeafP1OperatorAssembler<GridType,double,1>(grid);
    A->assemble(laplaceStiffness,u,f);

    typedef typename LeafP1OperatorAssembler<GridType,double,1>::RepresentationType LaplaceMatrixType;

    if (h1SemiNorm_)
        delete h1SemiNorm_;

    h1SemiNorm_ = new H1SemiNorm<CorrectionType>(**A);

    mmgSolver_ = new IterativeSolver<CorrectionType>(mmgStep,
                                                     multigridIterations_,
                                                     qpTolerance_,
                                                     h1SemiNorm_,
                                                     Solver::QUIET);

    // Write all intermediate solutions, if requested
    if (instrumented_)
        mmgSolver_->historyBuffer_ = "tmp/mgHistory";

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
    using namespace Dune;

    double trustRegionRadius = initialTrustRegionRadius_;

    // /////////////////////////////////////////////////////
    //   Set up the log file, if requested
    // /////////////////////////////////////////////////////
    FILE* fp;
    if (instrumented_) {

        fp = fopen("statistics", "w");
        if (!fp)
            DUNE_THROW(IOError, "Couldn't open statistics file for writing!");
    
    }

    // /////////////////////////////////////////////////////
    //   Trust-Region Solver
    // /////////////////////////////////////////////////////
    for (int i=0; i<maxTrustRegionSteps_; i++) {
        
        if (this->verbosity_ == FULL) {
            std::cout << "----------------------------------------------------" << std::endl;
            std::cout << "      Trust-Region Step Number: " << i 
                      << ",     radius: " << trustRegionRadius
                      << ",     energy: " << rodAssembler_->computeEnergy(x_) << std::endl;
            std::cout << "----------------------------------------------------" << std::endl;
        }

        CorrectionType rhs;
        CorrectionType corr(x_.size());
        corr = 0;

        rodAssembler_->assembleGradient(x_, rhs);
        //rodAssembler_->assembleMatrix(x_, *hessianMatrix_);
        rodAssembler_->assembleMatrixFD(x_, *hessianMatrix_);

        //gradientFDCheck(x_, rhs, *rodAssembler_);
        //hessianFDCheck(x_, *hessianMatrix_, *rodAssembler_);

        rhs *= -1;

        // Create trust-region obstacle on maxlevel
        setTrustRegionObstacles(trustRegionRadius,
                                trustRegionObstacles_[grid_->maxLevel()]);
        
        dynamic_cast<MultigridStep<MatrixType,CorrectionType>*>(mmgSolver_->iterationStep_)->setProblem(*hessianMatrix_, corr, rhs, grid_->maxLevel()+1);
        
        mmgSolver_->preprocess();
        
        dynamic_cast<MultigridStep<MatrixType,CorrectionType>*>(mmgSolver_->iterationStep_)->preprocess();
        
        
        // /////////////////////////////
        //    Solve !
        // /////////////////////////////
        mmgSolver_->solve();
        
        corr = dynamic_cast<MultigridStep<MatrixType,CorrectionType>*>(mmgSolver_->iterationStep_)->getSol();
        
        //std::cout << "Correction: " << std::endl << corr << std::endl;
        
        if (instrumented_) {

            fprintf(fp, "Trust-region step: %d, trust-region radius: %g\n",
                    i, trustRegionRadius);
                    
            // ///////////////////////////////////////////////////////////////
            //   Compute and measure progress against the exact solution
            //   for each trust region step
            // ///////////////////////////////////////////////////////////////
    
            CorrectionType exactSolution = corr;

            // Start from 0
            double oldError = 0;
            double totalConvRate = 1;
            double convRate = 1;
    
            // Write statistics of the initial solution
            // Compute the energy norm
            oldError = h1SemiNorm_->operator()(exactSolution);
    
            for (int j=0; j<multigridIterations_; j++) {
        
                // read iteration from file
                CorrectionType intermediateSol(grid_->size(1));
                intermediateSol = 0;
                char iSolFilename[100];
                sprintf(iSolFilename, "tmp/mgHistory/intermediatesolution_%04d", j);
                
                FILE* fpInt = fopen(iSolFilename, "rb");
                if (!fpInt)
                    DUNE_THROW(IOError, "Couldn't open intermediate solution");
                for (int k=0; k<intermediateSol.size(); k++)
                    for (int l=0; l<blocksize; l++)
                        fread(&intermediateSol[k][l], sizeof(double), 1, fpInt);
                
                fclose(fpInt);
                //std::cout << "intermediateSol\n" << intermediateSol << std::endl;

                // Compute errors
                intermediateSol -= exactSolution;
      
                //std::cout << "error\n" << intermediateSol << std::endl;

                // Compute the H1 norm
                double error = h1SemiNorm_->operator()(intermediateSol);

                convRate = error / oldError;
                totalConvRate *= convRate;
                
                if (error < 1e-12)
                    break;
                
                std::cout << "Iteration: " << j << "  ";
                std::cout << "Errors:  error " << error << ", convergence rate: " << convRate
                          << ",  total conv rate " << pow(totalConvRate, 1/((double)j+1)) << std::endl;
                

                fprintf(fp, "%d %g %g %g\n", j+1, error, convRate, pow(totalConvRate, 1/((double)j+1)));
                
                
                oldError = error;
                
            }
            
            
        }

        if (this->verbosity_ == FULL) {
            double translationMax = 0;
            double rotationMax    = 0;
            for (size_t j=0; j<corr.size(); j++) {
                for (int k=0; k<3; k++) {
                    translationMax = std::max(translationMax, corr[j][k]);
                    rotationMax    = std::max(rotationMax, corr[j][k+3]);
                }
            }
            printf("infinity norm of the correction: %g %g\n", translationMax, rotationMax);
        }

        if (corr.infinity_norm() < tolerance_) {
            if (this->verbosity_ == FULL)
                std::cout << "CORRECTION IS SMALL ENOUGH" << std::endl;

            if (this->verbosity_ != QUIET)
                std::cout << i+1 << " trust-region steps were taken." << std::endl;
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
            
        }
        
        /** \todo Don't always recompute oldEnergy */
        double oldEnergy = rodAssembler_->computeEnergy(x_);
        double energy    = rodAssembler_->computeEnergy(newIterate); 
        
        // compute the model decrease
        // It is $ m(x) - m(x+s) = -<g,s> - 0.5 <s, Hs>
        // Note that rhs = -g
        CorrectionType tmp(corr.size());
        tmp = 0;
        hessianMatrix_->umv(corr, tmp);
        double modelDecrease = (rhs*corr) - 0.5 * (corr*tmp);
        
        if (this->verbosity_ == FULL) {
            std::cout << "Absolute model decrease: " << modelDecrease 
                      << ",  functional decrease: " << oldEnergy - energy << std::endl;
            std::cout << "Relative model decrease: " << modelDecrease / energy
                      << ",  functional decrease: " << (oldEnergy - energy)/energy << std::endl;
        }            

        assert(modelDecrease >= 0);
        
        if (energy >= oldEnergy) {
            if (this->verbosity_ == FULL)
                printf("Richtung ist keine Abstiegsrichtung!\n");
        }

        if (energy >= oldEnergy &&
            (std::abs(oldEnergy-energy)/energy < 1e-9 || modelDecrease/energy < 1e-9)) {
            if (this->verbosity_ == FULL)
                std::cout << "Suspecting rounding problems" << std::endl;

            if (this->verbosity_ != QUIET)
                std::cout << i+1 << " trust-region steps were taken." << std::endl;

            x_ = newIterate;
            break;
        }

        // //////////////////////////////////////////////
        //   Check for acceptance of the step
        // //////////////////////////////////////////////
        if ( (oldEnergy-energy) / modelDecrease > 0.9) {
            // very successful iteration
            
            x_ = newIterate;
            trustRegionRadius *= 2;
            
        } else if ( (oldEnergy-energy) / modelDecrease > 0.01
                    || std::abs(oldEnergy-energy) < 1e-12) {
            // successful iteration
            x_ = newIterate;
            
        } else {
            // unsuccessful iteration
            trustRegionRadius /= 2;
            if (this->verbosity_ == FULL)
                std::cout << "Unsuccessful iteration!" << std::endl;
        }
        
        //  Write current energy
        if (this->verbosity_ == FULL)
            std::cout << "--- Current energy: " << energy << " ---" << std::endl;

        // /////////////////////////////////////////////////////////////////////
        //   Write the iterate to disk for later convergence rate measurement
        // /////////////////////////////////////////////////////////////////////

        if (instrumented_) {

            char iRodFilename[100];
            sprintf(iRodFilename, "tmp/intermediateSolution_%04d", i);

            FILE* fpRod = fopen(iRodFilename, "wb");
            if (!fpRod)
                DUNE_THROW(SolverError, "Couldn't open file " << iRodFilename << " for writing");
            
            for (int j=0; j<x_.size(); j++) {

                for (int k=0; k<3; k++)
                    fwrite(&x_[j].r[k], sizeof(double), 1, fpRod);

                for (int k=0; k<4; k++)  // 3d hardwired here!
                    fwrite(&x_[j].q[k], sizeof(double), 1, fpRod);

            }

            fclose(fpRod);

        }

    }

    // //////////////////////////////////////////////
    //   Close logfile
    // //////////////////////////////////////////////
    if (instrumented_)
        fclose(fp);
}
