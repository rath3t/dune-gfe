#include <dune/common/bitsetvector.hh>
#include <dune/common/timer.hh>

#include <dune/istl/io.hh>

#include <dune/functions/functionspacebases/pq1nodalbasis.hh>
#include <dune/functions/functionspacebases/pqknodalbasis.hh>

#include <dune/fufem/functionspacebases/p1nodalbasis.hh>
#include <dune/fufem/assemblers/operatorassembler.hh>
#include <dune/fufem/assemblers/localassemblers/laplaceassembler.hh>
#include <dune/fufem/assemblers/localassemblers/massassembler.hh>
#include <dune/fufem/assemblers/basisinterpolationmatrixassembler.hh>

// Using a monotone multigrid as the inner solver
#include <dune/solvers/iterationsteps/trustregiongsstep.hh>
#include <dune/solvers/iterationsteps/mmgstep.hh>
#include <dune/solvers/transferoperators/truncatedcompressedmgtransfer.hh>
#if defined THIRD_ORDER || defined SECOND_ORDER
#include <dune/gfe/pktop1mgtransfer.hh>
#endif
#include <dune/solvers/transferoperators/mandelobsrestrictor.hh>
#include <dune/solvers/solvers/iterativesolver.hh>
#include "maxnormtrustregion.hh"

#include <dune/solvers/norms/twonorm.hh>
#include <dune/solvers/norms/h1seminorm.hh>

template <class BasisType, class VectorType>
void TrustRegionSolver<BasisType,VectorType>::
setup(const typename BasisType::GridView::Grid& grid,
      const FEAssembler<BasisType, VectorType>* assembler,
         const SolutionType& x,
         const Dune::BitSetVector<blocksize>& dirichletNodes,
         double tolerance,
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
    grid_                     = &grid;
    assembler_                = assembler;
    x_                        = x;
    this->tolerance_          = tolerance;
    maxTrustRegionSteps_      = maxTrustRegionSteps;
    initialTrustRegionRadius_ = initialTrustRegionRadius;
    innerIterations_          = multigridIterations;
    innerTolerance_           = mgTolerance;
    ignoreNodes_              = &dirichletNodes;

    int numLevels = grid_->maxLevel()+1;

    // ////////////////////////////////
    //   Create a multigrid solver
    // ////////////////////////////////

#ifdef HAVE_IPOPT
    // First create an IPOpt base solver
    QuadraticIPOptSolver<MatrixType, CorrectionType>* baseSolver = new QuadraticIPOptSolver<MatrixType,CorrectionType>;
    baseSolver->verbosity_ = NumProc::QUIET;
    baseSolver->tolerance_ = baseTolerance;
    baseSolver->linearSolverType_ = "mumps";
#else
    // First create a Gauss-seidel base solver
    TrustRegionGSStep<MatrixType, CorrectionType>* baseSolverStep = new TrustRegionGSStep<MatrixType, CorrectionType>;

    // Hack: the two-norm may not scale all that well, but it is fast!
    TwoNorm<CorrectionType>* baseNorm = new TwoNorm<CorrectionType>;

    ::LoopSolver<CorrectionType>* baseSolver = new ::LoopSolver<CorrectionType>(baseSolverStep,
                                                                            baseIterations,
                                                                            baseTolerance,
                                                                            baseNorm,
                                                                            Solver::QUIET);
#endif

    // Make pre and postsmoothers
    TrustRegionGSStep<MatrixType, CorrectionType>* presmoother  = new TrustRegionGSStep<MatrixType, CorrectionType>;
    TrustRegionGSStep<MatrixType, CorrectionType>* postsmoother = new TrustRegionGSStep<MatrixType, CorrectionType>;

    MonotoneMGStep<MatrixType, CorrectionType>* mmgStep = new MonotoneMGStep<MatrixType, CorrectionType>;

    mmgStep->setMGType(mu, nu1, nu2);
    mmgStep->ignoreNodes_ = &dirichletNodes;
    mmgStep->basesolver_        = baseSolver;
    mmgStep->setSmoother(presmoother, postsmoother);
    mmgStep->obstacleRestrictor_= new MandelObstacleRestrictor<CorrectionType>();
    mmgStep->verbosity_         = Solver::QUIET;

    // //////////////////////////////////////////////////////////////////////////////////////
    //   Assemble a Laplace matrix to create a norm that's equivalent to the H1-norm
    // //////////////////////////////////////////////////////////////////////////////////////

    BasisType basis(grid.leafGridView());
    OperatorAssembler<BasisType,BasisType> operatorAssembler(basis, basis);

    LaplaceAssembler<GridType, typename BasisType::LocalFiniteElement, typename BasisType::LocalFiniteElement> laplaceStiffness;
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> > ScalarMatrixType;
    ScalarMatrixType localA;

    operatorAssembler.assemble(laplaceStiffness, localA);

    if (h1SemiNorm_)
        delete h1SemiNorm_;

    ScalarMatrixType* A = new ScalarMatrixType(localA);

    h1SemiNorm_ = new H1SemiNorm<CorrectionType>(*A);

    innerSolver_ = std::shared_ptr<LoopSolver<CorrectionType> >(new ::LoopSolver<CorrectionType>(mmgStep,
                                                                                                   innerIterations_,
                                                                                                   innerTolerance_,
                                                                                                   h1SemiNorm_,
                                                                                                 Solver::REDUCED));

    // //////////////////////////////////////////////////////////////////////////////////////
    //   Assemble a mass matrix to create a norm that's equivalent to the L2-norm
    //   This will be used to monitor the gradient
    // //////////////////////////////////////////////////////////////////////////////////////

    MassAssembler<GridType, typename BasisType::LocalFiniteElement, typename BasisType::LocalFiniteElement> massStiffness;
    ScalarMatrixType localMassMatrix;

    operatorAssembler.assemble(massStiffness, localMassMatrix);

    ScalarMatrixType* massMatrix = new ScalarMatrixType(localMassMatrix);
    l2Norm_ = std::make_shared<H1SemiNorm<CorrectionType> >(*massMatrix);

    // ////////////////////////////////////////////////////////////
    //    Create Hessian matrix and its occupation structure
    // ////////////////////////////////////////////////////////////

    hessianMatrix_ = std::auto_ptr<MatrixType>(new MatrixType);
    Dune::MatrixIndexSet indices(grid_->size(1), grid_->size(1));
    assembler_->getNeighborsPerVertex(indices);
    indices.exportIdx(*hessianMatrix_);

    // ////////////////////////////////////
    //   Create the transfer operators
    // ////////////////////////////////////

    for (size_t k=0; k<mmgStep->mgTransfer_.size(); k++)
        delete(mmgStep->mgTransfer_[k]);

    ////////////////////////////////////////////////////////////////////////
    //  The P1 space (actually P1/Q1, depending on the grid) is special:
    //  If we work in such a space, then the multigrid hierarchy of spaces
    //  is constructed in the usual way.  For all other space, there is
    //  an additional restriction operator on the top of the hierarchy, which
    //  restricts the FE space to the P1/Q1 space on the same grid.
    //  On the lower grid levels a hierarchy of P1/Q1 spaces is used again.
    ////////////////////////////////////////////////////////////////////////

    typedef BasisType Basis;
    bool isP1Basis = std::is_same<Basis,DuneFunctionsBasis<Dune::Functions::PQ1NodalBasis<typename Basis::GridView> > >::value
                  || std::is_same<Basis,DuneFunctionsBasis<Dune::Functions::PQKNodalBasis<typename Basis::GridView, 1> > >::value;

    if (isP1Basis)
      mmgStep->mgTransfer_.resize(numLevels-1);
    else
      mmgStep->mgTransfer_.resize(numLevels);

    // Here we set up the restriction of the leaf grid space into the leaf grid P1/Q1 space
    if (not isP1Basis)
    {
        typedef typename TruncatedCompressedMGTransfer<CorrectionType>::TransferOperatorType TransferOperatorType;
        P1NodalBasis<typename GridType::LeafGridView,double> p1Basis(grid_->leafGridView());

        TransferOperatorType pkToP1TransferMatrix;
        assembleBasisInterpolationMatrix<TransferOperatorType,
                                         P1NodalBasis<typename GridType::LeafGridView,double>,
                                         BasisType>(pkToP1TransferMatrix,p1Basis,basis);
        mmgStep->mgTransfer_.back() = new TruncatedCompressedMGTransfer<CorrectionType>;
        Dune::shared_ptr<TransferOperatorType> topTransferOperator = Dune::make_shared<TransferOperatorType>(pkToP1TransferMatrix);
        dynamic_cast<TruncatedCompressedMGTransfer<CorrectionType>*>(mmgStep->mgTransfer_.back())->setMatrix(topTransferOperator);
    }

    // Now the P1/Q1 restriction operators for the remaining levels
    for (int i=0; i<numLevels-1; i++) {

        // Construct the local multigrid transfer matrix
        TruncatedCompressedMGTransfer<CorrectionType>* newTransferOp = new TruncatedCompressedMGTransfer<CorrectionType>;
        newTransferOp->setup(*grid_,i,i+1);

        typedef typename TruncatedCompressedMGTransfer<CorrectionType>::TransferOperatorType TransferOperatorType;
        mmgStep->mgTransfer_[i] = new TruncatedCompressedMGTransfer<CorrectionType>;
        std::shared_ptr<TransferOperatorType> transferOperatorMatrix = Dune::make_shared<TransferOperatorType>(newTransferOp->getMatrix());
        dynamic_cast<TruncatedCompressedMGTransfer<CorrectionType>*>(mmgStep->mgTransfer_[i])->setMatrix(transferOperatorMatrix);
    }

    // //////////////////////////////////////////////////////////
    //   Create obstacles
    // //////////////////////////////////////////////////////////

    hasObstacle_.resize(basis.size(), true);
    mmgStep->hasObstacle_ = &hasObstacle_;

}


template <class BasisType, class VectorType>
void TrustRegionSolver<BasisType,VectorType>::solve()
{
    MonotoneMGStep<MatrixType,CorrectionType>* mgStep = NULL;

    // if the inner solver is a monotone multigrid set up a max-norm trust-region
    if (dynamic_cast<LoopSolver<CorrectionType>*>(innerSolver_.get())) {
        mgStep = dynamic_cast<MonotoneMGStep<MatrixType,CorrectionType>*>(dynamic_cast<LoopSolver<CorrectionType>*>(innerSolver_.get())->iterationStep_);

    }

    BasisType basis(grid_->leafGridView());
    MaxNormTrustRegion<blocksize> trustRegion(basis.size(), initialTrustRegionRadius_);

    std::vector<BoxConstraint<field_type,blocksize> > trustRegionObstacles;

    // /////////////////////////////////////////////////////
    //   Trust-Region Solver
    // /////////////////////////////////////////////////////

    double oldEnergy = assembler_->computeEnergy(x_);

    bool recomputeGradientHessian = true;
    CorrectionType rhs;
    MatrixType stiffnessMatrix;

    for (int i=0; i<maxTrustRegionSteps_; i++) {

        Dune::Timer totalTimer;
        if (this->verbosity_ == Solver::FULL) {
            std::cout << "----------------------------------------------------" << std::endl;
            std::cout << "      Trust-Region Step Number: " << i
                      << ",     radius: " << trustRegion.radius()
                      << ",     energy: " << oldEnergy << std::endl;
            std::cout << "----------------------------------------------------" << std::endl;
        }

        Dune::Timer gradientTimer;

        if (recomputeGradientHessian) {

            assembler_->assembleGradientAndHessian(x_,
                                                   rhs,
                                                   *hessianMatrix_,
                                                   i==0    // assemble occupation pattern only for the first call
                                                   );

            rhs *= -1;        // The right hand side is the _negative_ gradient

            // Compute gradient norm to monitor convergence
            CorrectionType gradient = rhs;
            for (size_t j=0; j<gradient.size(); j++)
              for (size_t k=0; k<gradient[j].size(); k++)
                if ((*ignoreNodes_)[j][k])
                  gradient[j][k] = 0;

            if (this->verbosity_ == Solver::FULL)
              std::cout << "Gradient norm: " << l2Norm_->operator()(gradient) << std::endl;

            if (this->verbosity_ == Solver::FULL)
              std::cout << "Assembly took " << gradientTimer.elapsed() << " sec." << std::endl;

            // Transfer matrix data
            stiffnessMatrix = *hessianMatrix_;

            recomputeGradientHessian = false;

        }

        CorrectionType corr(rhs.size());
        corr = 0;

        mgStep->setProblem(stiffnessMatrix, corr, rhs);

        trustRegionObstacles = trustRegion.obstacles();
        mgStep->obstacles_ = &trustRegionObstacles;

        innerSolver_->preprocess();

        ///////////////////////////////
        //    Solve !
        ///////////////////////////////

        std::cout << "Solve quadratic problem..." << std::endl;

        Dune::Timer solutionTimer;
        innerSolver_->solve();
        std::cout << "Solving the quadratic problem took " << solutionTimer.elapsed() << " seconds." << std::endl;

        if (mgStep)
            corr = mgStep->getSol();

        //std::cout << "Correction: " << std::endl << corr << std::endl;

        if (this->verbosity_ == NumProc::FULL)
            std::cout << "Infinity norm of the correction: " << corr.infinity_norm() << std::endl;

        if (corr.infinity_norm() < this->tolerance_) {
            if (this->verbosity_ == NumProc::FULL)
                std::cout << "CORRECTION IS SMALL ENOUGH" << std::endl;

            if (this->verbosity_ != NumProc::QUIET)
                std::cout << i+1 << " trust-region steps were taken." << std::endl;
            break;
        }

        // ////////////////////////////////////////////////////
        //   Check whether trust-region step can be accepted
        // ////////////////////////////////////////////////////

        SolutionType newIterate = x_;
        for (size_t j=0; j<newIterate.size(); j++)
            newIterate[j] += corr[j];

        double energy    = assembler_->computeEnergy(newIterate);

        // compute the model decrease
        // It is $ m(x) - m(x+s) = -<g,s> - 0.5 <s, Hs>
        // Note that rhs = -g
        CorrectionType tmp(corr.size());
        tmp = 0;
        hessianMatrix_->umv(corr, tmp);
        double modelDecrease = (rhs*corr) - 0.5 * (corr*tmp);

        double relativeModelDecrease = modelDecrease / std::fabs(energy);

        if (this->verbosity_ == NumProc::FULL) {
            std::cout << "Absolute model decrease: " << modelDecrease
                      << ",  functional decrease: " << oldEnergy - energy << std::endl;
            std::cout << "Relative model decrease: " << relativeModelDecrease
                      << ",  functional decrease: " << (oldEnergy - energy)/energy << std::endl;
        }

        assert(modelDecrease >= 0);

        if (energy >= oldEnergy) {
            if (this->verbosity_ == NumProc::FULL)
                printf("Richtung ist keine Abstiegsrichtung!\n");
        }

        if (energy >= oldEnergy &&
            (std::abs((oldEnergy-energy)/energy) < 1e-9 || relativeModelDecrease < 1e-9)) {
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

            // current energy becomes 'oldEnergy' for the next iteration
            oldEnergy = energy;

            recomputeGradientHessian = true;

        } else if ( (oldEnergy-energy) / modelDecrease > 0.01
                    || std::abs(oldEnergy-energy) < 1e-12) {
            // successful iteration
            x_ = newIterate;

            // current energy becomes 'oldEnergy' for the next iteration
            oldEnergy = energy;

            recomputeGradientHessian = true;

        } else {

            // unsuccessful iteration

            // Decrease the trust-region radius
            trustRegion.scale(0.5);

            if (this->verbosity_ == NumProc::FULL)
                std::cout << "Unsuccessful iteration!" << std::endl;
        }

        std::cout << "iteration took " << totalTimer.elapsed() << " sec." << std::endl;
    }

}
