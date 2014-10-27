#include <dune/common/bitsetvector.hh>
#include <dune/common/timer.hh>

#include <dune/istl/io.hh>

#include <dune/fufem/functionspacebases/p1nodalbasis.hh>
#include <dune/fufem/assemblers/operatorassembler.hh>
#include <dune/fufem/assemblers/localassemblers/laplaceassembler.hh>
#include <dune/fufem/assemblers/localassemblers/massassembler.hh>

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

template <class GridType, class VectorType>
void TrustRegionSolver<GridType,VectorType>::
setup(const GridType& grid,
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
         double baseTolerance,
         const SolutionType& pointLoads)
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
    pointLoads_               = pointLoads;

    int numLevels = grid_->maxLevel()+1;

    // ////////////////////////////////
    //   Create a multigrid solver
    // ////////////////////////////////

#ifdef HAVE_IPOPT
    // First create an IPOpt base solver
    QuadraticIPOptSolver<MatrixType, CorrectionType>* baseSolver = new QuadraticIPOptSolver<MatrixType,CorrectionType>;
    baseSolver->verbosity_ = NumProc::QUIET;
    baseSolver->tolerance_ = baseTolerance;
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

    mmgStep->mgTransfer_.resize(numLevels-1);

#if defined THIRD_ORDER || defined SECOND_ORDER
    if (numLevels>1) {
        P1NodalBasis<typename GridType::LeafGridView,double> p1Basis(grid_->leafGridView());

        PKtoP1MGTransfer<CorrectionType>* topTransferOp = new PKtoP1MGTransfer<CorrectionType>;
        topTransferOp->setup(basis,p1Basis);

        mmgStep->mgTransfer_.back() = topTransferOp;

        for (int i=0; i<mmgStep->mgTransfer_.size()-1; i++){
          // Construct the local multigrid transfer matrix
          TruncatedCompressedMGTransfer<CorrectionType>* newTransferOp = new TruncatedCompressedMGTransfer<CorrectionType>;
          newTransferOp->setup(*grid_,i+1,i+2);

          mmgStep->mgTransfer_[i] = newTransferOp;
        }

    }

#else
    for (size_t i=0; i<mmgStep->mgTransfer_.size(); i++){

        // Construct the local multigrid transfer matrix
        TruncatedCompressedMGTransfer<CorrectionType>* newTransferOp = new TruncatedCompressedMGTransfer<CorrectionType>;
        newTransferOp->setup(*grid_,i,i+1);

        mmgStep->mgTransfer_[i] = newTransferOp;;
    }
#endif

    // //////////////////////////////////////////////////////////
    //   Create obstacles
    // //////////////////////////////////////////////////////////

    hasObstacle_.resize(basis.size(), true);
    mmgStep->hasObstacle_ = &hasObstacle_;

}


template <class GridType, class VectorType>
void TrustRegionSolver<GridType,VectorType>::solve()
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

    double oldEnergy = assembler_->computeEnergy(x_, pointLoads_);

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
                                                   pointLoads_,
                                                   rhs,
                                                   *hessianMatrix_,
                                                   i==0    // assemble occupation pattern only for the first call
                                                   );

            rhs *= -1;        // The right hand side is the _negative_ gradient

            // Compute gradient norm to monitor convergence
            CorrectionType gradient = rhs;
            for (size_t j=0; j<gradient.size(); j++)
              for (int k=0; k<gradient[j].size(); k++)
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

        //std::cout << "Correction: " << std::endl << corr_global << std::endl;

        // Output correction for debugging
        Dune::VTKWriter<typename GridType::LeafGridView> vtkWriter(grid_->leafGridView());

        Dune::BlockVector<Dune::FieldVector<double,3> > displacement(x_.size());
        for (size_t j=0; j<x_.size(); j++)
          displacement[j] = x_[j] - identity_[j];

        BasisType basis(grid_->leafGridView());
        Dune::shared_ptr<VTKBasisGridFunction<BasisType,Dune::BlockVector<Dune::FieldVector<double,3> > > > vtkDisplacement
               = Dune::make_shared<VTKBasisGridFunction<BasisType,Dune::BlockVector<Dune::FieldVector<double,3> > > >
                                  (basis, displacement, "Displacement");

        Dune::shared_ptr<VTKBasisGridFunction<BasisType,Dune::BlockVector<Dune::FieldVector<double,3> > > > vtkCorrection
               = Dune::make_shared<VTKBasisGridFunction<BasisType,Dune::BlockVector<Dune::FieldVector<double,3> > > >
                                  (basis, corr, "Correction");

        Dune::shared_ptr<VTKBasisGridFunction<BasisType,Dune::BlockVector<Dune::FieldVector<double,3> > > > vtkGradient
               = Dune::make_shared<VTKBasisGridFunction<BasisType,Dune::BlockVector<Dune::FieldVector<double,3> > > >
                                  (basis, rhs, "Gradient");

        vtkWriter.addVertexData(vtkDisplacement);
        vtkWriter.addVertexData(vtkCorrection);
        vtkWriter.addVertexData(vtkGradient);
        vtkWriter.write("hencky_correction_" + std::to_string(i+1));

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

        double energy    = assembler_->computeEnergy(newIterate, pointLoads_);

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
