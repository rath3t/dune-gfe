
// For using a monotone multigrid as the inner solver
#include <dune/solvers/iterationsteps/trustregiongsstep.hh>
#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include "maxnormtrustregion.hh"

#include <dune/gfe/gramschmidtsolver.hh>

template <class TargetSpace>
void TargetSpaceRiemannianTRSolver<TargetSpace>::
setup(const AverageDistanceAssembler<TargetSpace>* assembler,
      const TargetSpace& x,
      double tolerance,
      int maxTrustRegionSteps,
      double initialTrustRegionRadius,
      int innerIterations,
        double innerTolerance)
{
    assembler_                = assembler;
    x_                        = x;
    tolerance_                = tolerance;
    maxTrustRegionSteps_      = maxTrustRegionSteps;
    initialTrustRegionRadius_ = initialTrustRegionRadius;
    innerIterations_          = innerIterations;
    innerTolerance_           = innerTolerance;
    this->verbosity_          = NumProc::QUIET;
    minNumberOfIterations_    = 4;

#ifdef USE_GAUSS_SEIDEL_SOLVER
    // ////////////////////////////////
    //   Create a projected gauss-seidel solver
    // ////////////////////////////////

    // First create a Gauss-seidel base solver
    innerSolverStep_ = std::auto_ptr<TrustRegionGSStep<MatrixType, CorrectionType> >(new TrustRegionGSStep<MatrixType, CorrectionType>);

    energyNorm_ = std::auto_ptr<EnergyNorm<MatrixType, CorrectionType> >(new EnergyNorm<MatrixType, CorrectionType>(*innerSolverStep_.get()));

    innerSolver_ = std::auto_ptr< ::LoopSolver<CorrectionType> >(new ::LoopSolver<CorrectionType>(innerSolverStep_.get(),
                                                                                                  innerIterations,
                                                                                                  innerTolerance,
                                                                                                  energyNorm_.get(),
                                                                                                  Solver::QUIET));

    innerSolver_->useRelativeError_ = false;
#endif
}


template <class TargetSpace>
void TargetSpaceRiemannianTRSolver<TargetSpace>::solve()
{
    assert(minNumberOfIterations_ > 0);

    MaxNormTrustRegion<blocksize,field_type> trustRegion(1,   // we have only one block
                                              initialTrustRegionRadius_);

    field_type energy = assembler_->value(x_);

    // /////////////////////////////////////////////////////
    //   Trust-Region Solver
    // /////////////////////////////////////////////////////
    for (size_t i=0; i<maxTrustRegionSteps_; i++) {

        if (this->verbosity_ == Solver::FULL) {
            std::cout << "----------------------------------------------------" << std::endl;
            std::cout << "      Trust-Region Step Number: " << i
                      << ",     radius: " << trustRegion.radius()
                      << ",     energy: " << energy << std::endl;
            std::cout << "----------------------------------------------------" << std::endl;
        }

        CorrectionType rhs(1);   // length is 1 _block_
        CorrectionType corr(1);  // length is 1 _block_
#ifdef USE_GAUSS_SEIDEL_SOLVER
        corr = 0;
#endif

        MatrixType hesseMatrix(1,1);

#ifdef USE_GAUSS_SEIDEL_SOLVER
        assembler_->assembleGradient(x_, rhs[0]);
        assembler_->assembleHessian(x_, hesseMatrix[0][0]);
#else
        /** \todo Fix this sense copying */
        typename TargetSpace::EmbeddedTangentVector foo;
        assembler_->assembleEmbeddedGradient(x_, foo);
        rhs[0] = foo;
        assembler_->assembleEmbeddedHessian(x_, hesseMatrix[0][0]);
#endif

        // The right hand side is the _negative_ gradient
        rhs *= -1;

#ifdef USE_GAUSS_SEIDEL_SOLVER
        dynamic_cast<LinearIterationStep<MatrixType,CorrectionType>*>(innerSolver_->iterationStep_)->setProblem(hesseMatrix, corr, rhs);

        dynamic_cast<TrustRegionGSStep<MatrixType,CorrectionType>*>(innerSolver_->iterationStep_)->obstacles_ = &trustRegion.obstacles();

        innerSolver_->preprocess();
#endif
        // /////////////////////////////
        //    Solve !
        // /////////////////////////////
#ifdef USE_GAUSS_SEIDEL_SOLVER
        innerSolver_->solve();
#else
        Dune::FieldMatrix<field_type,blocksize,embeddedBlocksize> basis = x_.orthonormalFrame();
        GramSchmidtSolver<field_type, blocksize, embeddedBlocksize>::solve(hesseMatrix[0][0], corr[0], rhs[0], basis);
#endif

#ifdef USE_GAUSS_SEIDEL_SOLVER
        corr = innerSolver_->iterationStep_->getSol();
#endif

        //std::cout << "Corr: " << corr << std::endl;

        if (this->verbosity_ == NumProc::FULL)
            std::cout << "Infinity norm of the correction: " << corr.infinity_norm() << std::endl;

        if (corr.infinity_norm() < this->tolerance_ and i>=minNumberOfIterations_-1) {
            if (this->verbosity_ == NumProc::FULL)
                std::cout << "CORRECTION IS SMALL ENOUGH" << std::endl;

            if (this->verbosity_ != NumProc::QUIET)
                std::cout << i+1 << " trust-region steps were taken." << std::endl;
            break;
        }

        // ////////////////////////////////////////////////////
        //   Check whether trust-region step can be accepted
        // ////////////////////////////////////////////////////

        TargetSpace newIterate = x_;
        newIterate = TargetSpace::exp(newIterate, corr[0]);

        field_type oldEnergy = energy;
        field_type energy    = assembler_->value(newIterate);

        // compute the model decrease
        // It is $ m(x) - m(x+s) = -<g,s> - 0.5 <s, Hs>
        // Note that rhs = -g
        CorrectionType tmp(corr.size());
        tmp = 0;
        hesseMatrix.umv(corr, tmp);
        field_type modelDecrease = (rhs*corr) - 0.5 * (corr*tmp);

        if (this->verbosity_ == NumProc::FULL) {
            std::cout << "Absolute model decrease: " << modelDecrease
                      << ",  functional decrease: " << oldEnergy - energy << std::endl;
            std::cout << "Relative model decrease: " << modelDecrease / energy
                      << ",  functional decrease: " << (oldEnergy - energy)/energy << std::endl;
        }

        assert(modelDecrease >= -1e-15);

        if (energy >= oldEnergy) {
            if (this->verbosity_ == NumProc::FULL)
                printf("Richtung ist keine Abstiegsrichtung!\n");
        }

        if (energy >= oldEnergy &&
            i>minNumberOfIterations_-1 &&
            (std::abs(oldEnergy-energy)/energy < 1e-9 || modelDecrease/energy < 1e-9)) {
            if (this->verbosity_ == NumProc::FULL)
                std::cout << "Suspecting rounding problems" << std::endl;

            if (this->verbosity_ != NumProc::QUIET)
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
            trustRegion.scale(2);

        } else if ( (oldEnergy-energy) / modelDecrease > 0.01
                    || std::abs(oldEnergy-energy) < 1e-12) {
            // successful iteration
            x_ = newIterate;

        } else {
            // unsuccessful iteration
            trustRegion.scale(0.5);
            if (this->verbosity_ == NumProc::FULL)
                std::cout << "Unsuccessful iteration!" << std::endl;
        }

        //  Write current energy
        if (this->verbosity_ == NumProc::FULL)
            std::cout << "--- Current energy: " << energy << " ---" << std::endl;

    }

}
