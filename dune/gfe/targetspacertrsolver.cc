
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
      int maxNewtonSteps)
{
    assembler_                = assembler;
    x_                        = x;
    tolerance_                = tolerance;
    maxNewtonSteps_           = maxNewtonSteps;
    this->verbosity_          = NumProc::QUIET;
    minNumberOfIterations_    = 1;
}


template <class TargetSpace>
void TargetSpaceRiemannianTRSolver<TargetSpace>::solve()
{
    assert(minNumberOfIterations_ > 0);

    // /////////////////////////////////////////////////////
    //   Newton Solver
    // /////////////////////////////////////////////////////
    for (size_t i=0; i<maxNewtonSteps_; i++) {

        if (this->verbosity_ == Solver::FULL) {
            std::cout << "----------------------------------------------------" << std::endl;
            std::cout << "      Newton Step Number: " << i
                      << ",     energy: " << assembler_->value(x_) << std::endl;
            std::cout << "----------------------------------------------------" << std::endl;
        }

        CorrectionType corr;

        MatrixType hesseMatrix;

        typename TargetSpace::EmbeddedTangentVector rhs;
        assembler_->assembleEmbeddedGradient(x_, rhs);
        assembler_->assembleEmbeddedHessian(x_, hesseMatrix);

        // The right hand side is the _negative_ gradient
        rhs *= -1;

        // /////////////////////////////
        //    Solve !
        // /////////////////////////////
        Dune::FieldMatrix<field_type,blocksize,embeddedBlocksize> basis = x_.orthonormalFrame();
        GramSchmidtSolver<field_type, blocksize, embeddedBlocksize>::solve(hesseMatrix, corr, rhs, basis);

        if (this->verbosity_ == NumProc::FULL)
            std::cout << "Infinity norm of the correction: " << corr.infinity_norm() << std::endl;

        if (corr.infinity_norm() < this->tolerance_ and i>=minNumberOfIterations_-1) {
            if (this->verbosity_ == NumProc::FULL)
                std::cout << "CORRECTION IS SMALL ENOUGH" << std::endl;

            if (this->verbosity_ != NumProc::QUIET)
                std::cout << i+1 << " Newton steps were taken." << std::endl;
            break;
        }

        // ////////////////////////////////////////////////////
        //   Check whether trust-region step can be accepted
        // ////////////////////////////////////////////////////
#if 0
        TargetSpace newIterate = x_;
        newIterate = TargetSpace::exp(newIterate, corr);

        if (this->verbosity_ == NumProc::FULL)
        {
            field_type oldEnergy = assembler_->value(x_);
            field_type energy    = assembler_->value(newIterate);

            if (energy >= oldEnergy)
                printf("Richtung ist keine Abstiegsrichtung!\n");
        }

        //  Actually take the step
        x_ = newIterate;
#else
        x_ = TargetSpace::exp(x_, corr);
#endif
    }

}
