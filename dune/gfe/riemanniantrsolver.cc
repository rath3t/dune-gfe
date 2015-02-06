#include <dune/common/bitsetvector.hh>
#include <dune/common/timer.hh>
#include <dune/common/parallel/mpihelper.hh>

#include <dune/istl/io.hh>

#include <dune/grid/common/mcmgmapper.hh>

#include <dune/fufem/functionspacebases/p1nodalbasis.hh>
#include <dune/fufem/assemblers/operatorassembler.hh>
#include <dune/fufem/assemblers/localassemblers/laplaceassembler.hh>
#include <dune/fufem/assemblers/localassemblers/massassembler.hh>
#include <dune/fufem/assemblers/basisinterpolationmatrixassembler.hh>

// Using a monotone multigrid as the inner solver
#include <dune/solvers/iterationsteps/trustregiongsstep.hh>
#include <dune/solvers/iterationsteps/mmgstep.hh>
#include <dune/solvers/transferoperators/truncatedcompressedmgtransfer.hh>
#include <dune/solvers/transferoperators/mandelobsrestrictor.hh>
#include <dune/solvers/solvers/iterativesolver.hh>
#include "maxnormtrustregion.hh"

#include <dune/solvers/norms/twonorm.hh>
#include <dune/solvers/norms/h1seminorm.hh>

#if HAVE_MPI
#include <dune/gfe/parallel/matrixcommunicator.hh>
#include <dune/gfe/parallel/vectorcommunicator.hh>
#endif

template <class Basis, class TargetSpace>
void RiemannianTrustRegionSolver<Basis,TargetSpace>::
setup(const GridType& grid,
      const GeodesicFEAssembler<Basis, TargetSpace>* assembler,
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
         bool instrumented)
{
    int rank = grid.comm().rank();

    grid_                     = &grid;
    assembler_                = assembler;
    x_                        = x;
    this->tolerance_          = tolerance;
    maxTrustRegionSteps_      = maxTrustRegionSteps;
    initialTrustRegionRadius_ = initialTrustRegionRadius;
    innerIterations_          = multigridIterations;
    innerTolerance_           = mgTolerance;
    instrumented_             = instrumented;
    ignoreNodes_              = &dirichletNodes;

    int numLevels = grid_->maxLevel()+1;

    //////////////////////////////////////////////////////////////////
    //  Create global numbering for matrix and vector transfer
    //////////////////////////////////////////////////////////////////

#if HAVE_MPI
    globalMapper_ = std::unique_ptr<GlobalMapper>(new GlobalMapper(grid_->leafGridView()));
#endif

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
#if HAVE_MPI
    // Transfer all Dirichlet data to the master processor
    VectorCommunicator<GlobalMapper, typename GridType::LeafGridView::CollectiveCommunication, Dune::BitSetVector<blocksize> > vectorComm(*globalMapper_,
                                                                                                                                     grid_->leafGridView().comm(),
                                                                                                                                     0);
    Dune::BitSetVector<blocksize>* globalDirichletNodes = NULL;
    globalDirichletNodes = new Dune::BitSetVector<blocksize>(vectorComm.reduceCopy(dirichletNodes));
#else
    Dune::BitSetVector<blocksize>* globalDirichletNodes = NULL;
    globalDirichletNodes = new Dune::BitSetVector<blocksize>(dirichletNodes);
#endif

    // Make pre and postsmoothers
    TrustRegionGSStep<MatrixType, CorrectionType>* presmoother  = new TrustRegionGSStep<MatrixType, CorrectionType>;
    TrustRegionGSStep<MatrixType, CorrectionType>* postsmoother = new TrustRegionGSStep<MatrixType, CorrectionType>;

    MonotoneMGStep<MatrixType, CorrectionType>* mmgStep = new MonotoneMGStep<MatrixType, CorrectionType>;

    mmgStep->setMGType(mu, nu1, nu2);
    mmgStep->ignoreNodes_ = globalDirichletNodes;
    mmgStep->basesolver_        = baseSolver;
    mmgStep->setSmoother(presmoother, postsmoother);
    mmgStep->obstacleRestrictor_= new MandelObstacleRestrictor<CorrectionType>();
    mmgStep->verbosity_         = Solver::QUIET;

    // //////////////////////////////////////////////////////////////////////////////////////
    //   Assemble a Laplace matrix to create a norm that's equivalent to the H1-norm
    // //////////////////////////////////////////////////////////////////////////////////////

    Basis basis(grid.leafGridView());
    OperatorAssembler<Basis,Basis> operatorAssembler(basis, basis);

    LaplaceAssembler<GridType, typename Basis::LocalFiniteElement, typename Basis::LocalFiniteElement> laplaceStiffness;
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> > ScalarMatrixType;
    ScalarMatrixType localA;

    operatorAssembler.assemble(laplaceStiffness, localA);

    if (h1SemiNorm_)
        delete h1SemiNorm_;
#if HAVE_MPI
    LocalMapper localMapper(grid_->leafGridView());

    MatrixCommunicator<GlobalMapper,
                       typename GridType::LeafGridView,
                       typename GridType::LeafGridView,
                       ScalarMatrixType,
                       LocalMapper,
                       LocalMapper> matrixComm(*globalMapper_, grid_->leafGridView(), localMapper, localMapper, 0);

    ScalarMatrixType* A = new ScalarMatrixType(matrixComm.reduceAdd(localA));
#else
    ScalarMatrixType* A = new ScalarMatrixType(localA);
#endif
    h1SemiNorm_ = new H1SemiNorm<CorrectionType>(*A);

    innerSolver_ = std::shared_ptr<LoopSolver<CorrectionType> >(new ::LoopSolver<CorrectionType>(mmgStep,
                                                                                                   innerIterations_,
                                                                                                   innerTolerance_,
                                                                                                   h1SemiNorm_,
                                                                                                 Solver::QUIET));

    // //////////////////////////////////////////////////////////////////////////////////////
    //   Assemble a mass matrix to create a norm that's equivalent to the L2-norm
    //   This will be used to monitor the gradient
    // //////////////////////////////////////////////////////////////////////////////////////

    MassAssembler<GridType, typename Basis::LocalFiniteElement, typename Basis::LocalFiniteElement> massStiffness;
    ScalarMatrixType localMassMatrix;

    operatorAssembler.assemble(massStiffness, localMassMatrix);

#if HAVE_MPI
    ScalarMatrixType* massMatrix = new ScalarMatrixType(matrixComm.reduceAdd(localMassMatrix));
#else
    ScalarMatrixType* massMatrix = new ScalarMatrixType(localMassMatrix);
#endif
    l2Norm_ = std::make_shared<H1SemiNorm<CorrectionType> >(*massMatrix);

    // Write all intermediate solutions, if requested
    if (instrumented_
        && dynamic_cast<IterativeSolver<CorrectionType>*>(innerSolver_.get()))
        dynamic_cast<IterativeSolver<CorrectionType>*>(innerSolver_.get())->historyBuffer_ = "tmp/mgHistory";

    // ////////////////////////////////////////////////////////////
    //    Create Hessian matrix and its occupation structure
    // ////////////////////////////////////////////////////////////

    hessianMatrix_ = std::unique_ptr<MatrixType>(new MatrixType);
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

    bool isP1Basis = std::is_same<Basis,P1NodalBasis<typename Basis::GridView> >::value;

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
                                         Basis>(pkToP1TransferMatrix,p1Basis,basis);
#if HAVE_MPI
        // If we are on more than 1 processors, join all local transfer matrices on rank 0,
        // and construct a single global transfer operator there.
        typedef Dune::GlobalP1Mapper<typename GridType::LeafGridView> GlobalLeafP1Mapper;
        GlobalLeafP1Mapper p1Index(grid_->leafGridView());

        typedef Dune::MultipleCodimMultipleGeomTypeMapper<typename GridType::LeafGridView, Dune::MCMGVertexLayout> LeafP1LocalMapper;
        LeafP1LocalMapper leafP1LocalMapper(grid_->leafGridView());

        MatrixCommunicator<GlobalMapper,
                           typename GridType::LeafGridView,
                           typename GridType::LeafGridView,
                           TransferOperatorType,
                           LocalMapper,
                           LeafP1LocalMapper,
                           GlobalLeafP1Mapper> matrixComm(*globalMapper_, p1Index, grid_->leafGridView(), grid_->leafGridView(), localMapper, leafP1LocalMapper, 0);

        mmgStep->mgTransfer_.back() = new TruncatedCompressedMGTransfer<CorrectionType>;
        Dune::shared_ptr<TransferOperatorType> topTransferOperator = Dune::make_shared<TransferOperatorType>(matrixComm.reduceCopy(pkToP1TransferMatrix));
#else
        mmgStep->mgTransfer_.back() = new TruncatedCompressedMGTransfer<CorrectionType>;
        Dune::shared_ptr<TransferOperatorType> topTransferOperator = Dune::make_shared<TransferOperatorType>(pkToP1TransferMatrix);
#endif
        dynamic_cast<TruncatedCompressedMGTransfer<CorrectionType>*>(mmgStep->mgTransfer_.back())->setMatrix(topTransferOperator);
    }

    // Now the P1/Q1 restriction operators for the remaining levels
    for (int i=0; i<numLevels-1; i++) {

        // Construct the local multigrid transfer matrix
        TruncatedCompressedMGTransfer<CorrectionType>* newTransferOp = new TruncatedCompressedMGTransfer<CorrectionType>;
        newTransferOp->setup(*grid_,i,i+1);
#if HAVE_MPI
        // If we are on more than 1 processors, join all local transfer matrices on rank 0,
        // and construct a single global transfer operator there.
        typedef Dune::GlobalP1Mapper<typename GridType::LevelGridView> GlobalLevelP1Mapper;
        GlobalLevelP1Mapper fineGUIndex(grid_->levelGridView(i+1));
        GlobalLevelP1Mapper coarseGUIndex(grid_->levelGridView(i));

        typedef Dune::MultipleCodimMultipleGeomTypeMapper<typename GridType::LevelGridView, Dune::MCMGVertexLayout> LevelLocalMapper;
        LevelLocalMapper fineLevelLocalMapper(grid_->levelGridView(i+1));
        LevelLocalMapper coarseLevelLocalMapper(grid_->levelGridView(i));
#endif
        typedef typename TruncatedCompressedMGTransfer<CorrectionType>::TransferOperatorType TransferOperatorType;
#if HAVE_MPI
        MatrixCommunicator<GlobalLevelP1Mapper,
                           typename GridType::LevelGridView,
                           typename GridType::LevelGridView,
                           TransferOperatorType,
                           LevelLocalMapper,
                           LevelLocalMapper> matrixComm(fineGUIndex, coarseGUIndex, grid_->levelGridView(i+1), grid_->levelGridView(i), fineLevelLocalMapper, coarseLevelLocalMapper, 0);

        mmgStep->mgTransfer_[i] = new TruncatedCompressedMGTransfer<CorrectionType>;
        std::shared_ptr<TransferOperatorType> transferOperatorMatrix = std::make_shared<TransferOperatorType>(matrixComm.reduceCopy(newTransferOp->getMatrix()));

#else
        mmgStep->mgTransfer_[i] = new TruncatedCompressedMGTransfer<CorrectionType>;
        std::shared_ptr<TransferOperatorType> transferOperatorMatrix = Dune::make_shared<TransferOperatorType>(newTransferOp->getMatrix());
#endif
        dynamic_cast<TruncatedCompressedMGTransfer<CorrectionType>*>(mmgStep->mgTransfer_[i])->setMatrix(transferOperatorMatrix);
    }

    // //////////////////////////////////////////////////////////
    //   Create obstacles
    // //////////////////////////////////////////////////////////

    if (rank==0)
    {
#if HAVE_MPI
        hasObstacle_.resize(globalMapper_->size(), true);
#else
        hasObstacle_.resize(basis.size(), true);
#endif
        mmgStep->hasObstacle_ = &hasObstacle_;
    }

}


template <class Basis, class TargetSpace>
void RiemannianTrustRegionSolver<Basis,TargetSpace>::solve()
{
    int argc = 0;
    char** argv;
    Dune::MPIHelper& mpiHelper = Dune::MPIHelper::instance(argc,argv);
    int rank = mpiHelper.rank();

    MonotoneMGStep<MatrixType,CorrectionType>* mgStep = NULL;

    // if the inner solver is a monotone multigrid set up a max-norm trust-region
    if (dynamic_cast<LoopSolver<CorrectionType>*>(innerSolver_.get())) {
        mgStep = dynamic_cast<MonotoneMGStep<MatrixType,CorrectionType>*>(dynamic_cast<LoopSolver<CorrectionType>*>(innerSolver_.get())->iterationStep_);

    }

#if HAVE_MPI
    MaxNormTrustRegion<blocksize> trustRegion(globalMapper_->size(), initialTrustRegionRadius_);
#else
    Basis basis(grid_->leafGridView());
    MaxNormTrustRegion<blocksize> trustRegion(basis.size(), initialTrustRegionRadius_);
#endif
    trustRegion.set(initialTrustRegionRadius_, scaling_);

    std::vector<BoxConstraint<field_type,blocksize> > trustRegionObstacles;

   // /////////////////////////////////////////////////////
    //   Set up the log file, if requested
    // /////////////////////////////////////////////////////
    FILE* fp = nullptr;
    if (instrumented_) {

        fp = fopen("statistics", "w");
        if (!fp)
            DUNE_THROW(Dune::IOError, "Couldn't open statistics file for writing!");

    }

    // /////////////////////////////////////////////////////
    //   Trust-Region Solver
    // /////////////////////////////////////////////////////

    double oldEnergy = assembler_->computeEnergy(x_);
    oldEnergy = mpiHelper.getCollectiveCommunication().sum(oldEnergy);

    bool recomputeGradientHessian = true;
    CorrectionType rhs;
    MatrixType stiffnessMatrix;
    CorrectionType rhs_global;
#if HAVE_MPI
    VectorCommunicator<GlobalMapper, typename GridType::LeafGridView::CollectiveCommunication, CorrectionType> vectorComm(*globalMapper_,
                                                                                                                     grid_->leafGridView().comm(),
                                                                                                                     0);

    LocalMapper localMapper(grid_->leafGridView());
    MatrixCommunicator<GlobalMapper,
                       typename GridType::LeafGridView,
                       typename GridType::LeafGridView,
                       MatrixType,
                       LocalMapper,
                       LocalMapper> matrixComm(*globalMapper_,
                                               grid_->leafGridView(),
                                               localMapper,
                                               localMapper,
                                               0);
#endif
    for (int i=0; i<maxTrustRegionSteps_; i++) {

/*        std::cout << "current iterate:\n";
        for (size_t j=0; j<x_.size(); j++)
            std::cout << x_[j] << std::endl;*/

        Dune::Timer totalTimer;
        if (this->verbosity_ == Solver::FULL and rank==0) {
            std::cout << "----------------------------------------------------" << std::endl;
            std::cout << "      Trust-Region Step Number: " << i
                      << ",     radius: " << trustRegion.radius()
                      << ",     energy: " << oldEnergy << std::endl;
            std::cout << "----------------------------------------------------" << std::endl;
        }

        CorrectionType corr(x_.size());
        corr = 0;

        Dune::Timer gradientTimer;

        if (recomputeGradientHessian) {

            assembler_->assembleGradientAndHessian(x_,
                                                   rhs,
                                                   *hessianMatrix_,
                                                   i==0    // assemble occupation pattern only for the first call
                                                   );

            rhs *= -1;        // The right hand side is the _negative_ gradient

            // Transfer vector data
#if HAVE_MPI
            rhs_global = vectorComm.reduceAdd(rhs);
#else
            rhs_global = rhs;
#endif
            CorrectionType gradient = rhs_global;
            for (size_t j=0; j<gradient.size(); j++)
              for (size_t k=0; k<gradient[j].size(); k++)
                if ((*mgStep->ignoreNodes_)[j][k])  // global Dirichlet nodes set
                  gradient[j][k] = 0;

            if (this->verbosity_ == Solver::FULL and rank==0)
              std::cout << "Gradient norm: " << l2Norm_->operator()(gradient) << std::endl;

            if (this->verbosity_ == Solver::FULL)
              std::cout << "Assembly took " << gradientTimer.elapsed() << " sec." << std::endl;

            // Transfer matrix data
#if HAVE_MPI
            stiffnessMatrix = matrixComm.reduceAdd(*hessianMatrix_);
#else
            stiffnessMatrix = *hessianMatrix_;
#endif
            recomputeGradientHessian = false;

        }

        CorrectionType corr_global(rhs_global.size());
        corr_global = 0;

        if (rank==0)
        {
            mgStep->setProblem(stiffnessMatrix, corr_global, rhs_global);

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
                corr_global = mgStep->getSol();

            //std::cout << "Correction: " << std::endl << corr_global << std::endl;
        }

        // Distribute solution
        if (mpiHelper.size()>1 and rank==0)
            std::cout << "Transfer solution back to root process ..." << std::endl;

#if HAVE_MPI
        corr = vectorComm.scatter(corr_global);
#else
        corr = corr_global;
#endif

        // Make infinity norm of corr_global known on all processors
        double corrNorm = corr.infinity_norm();
        double corrGlobalInfinityNorm = mpiHelper.getCollectiveCommunication().max(corrNorm);

        if (instrumented_) {

            fprintf(fp, "Trust-region step: %d, trust-region radius: %g\n",
                    i, trustRegion.radius());

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

            for (int j=0; j<innerIterations_; j++) {

                // read iteration from file
                CorrectionType intermediateSol(grid_->size(gridDim));
                intermediateSol = 0;
                char iSolFilename[100];
                sprintf(iSolFilename, "tmp/mgHistory/intermediatesolution_%04d", j);

                FILE* fpInt = fopen(iSolFilename, "rb");
                if (!fpInt)
                    DUNE_THROW(Dune::IOError, "Couldn't open intermediate solution");
                for (size_t k=0; k<intermediateSol.size(); k++)
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

        if (this->verbosity_ == NumProc::FULL)
            std::cout << "Infinity norm of the correction: " << corr.infinity_norm() << std::endl;

        if (corrGlobalInfinityNorm < this->tolerance_) {
            if (this->verbosity_ == NumProc::FULL and rank==0)
                std::cout << "CORRECTION IS SMALL ENOUGH" << std::endl;

            if (this->verbosity_ != NumProc::QUIET and rank==0)
                std::cout << i+1 << " trust-region steps were taken." << std::endl;
            break;
        }

        // ////////////////////////////////////////////////////
        //   Check whether trust-region step can be accepted
        // ////////////////////////////////////////////////////

        SolutionType newIterate = x_;
        for (size_t j=0; j<newIterate.size(); j++)
            newIterate[j] = TargetSpace::exp(newIterate[j], corr[j]);

        double energy    = assembler_->computeEnergy(newIterate);
        energy = mpiHelper.getCollectiveCommunication().sum(energy);

        // compute the model decrease
        // It is $ m(x) - m(x+s) = -<g,s> - 0.5 <s, Hs>
        // Note that rhs = -g
        CorrectionType tmp(corr.size());
        tmp = 0;
        hessianMatrix_->umv(corr, tmp);
        double modelDecrease = (rhs*corr) - 0.5 * (corr*tmp);
        modelDecrease = mpiHelper.getCollectiveCommunication().sum(modelDecrease);

        double relativeModelDecrease = modelDecrease / std::fabs(energy);

        if (this->verbosity_ == NumProc::FULL and rank==0) {
            std::cout << "Absolute model decrease: " << modelDecrease
                      << ",  functional decrease: " << oldEnergy - energy << std::endl;
            std::cout << "Relative model decrease: " << relativeModelDecrease
                      << ",  functional decrease: " << (oldEnergy - energy)/energy << std::endl;
        }

        assert(modelDecrease >= 0);

        if (energy >= oldEnergy and rank==0) {
            if (this->verbosity_ == NumProc::FULL)
                printf("Richtung ist keine Abstiegsrichtung!\n");
        }

        if (energy >= oldEnergy &&
            (std::abs((oldEnergy-energy)/energy) < 1e-9 || relativeModelDecrease < 1e-9)) {
            if (this->verbosity_ == NumProc::FULL and rank==0)
                std::cout << "Suspecting rounding problems" << std::endl;

            if (this->verbosity_ != NumProc::QUIET and rank==0)
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

            if (this->verbosity_ == NumProc::FULL and rank==0)
                std::cout << "Unsuccessful iteration!" << std::endl;
        }

        // /////////////////////////////////////////////////////////////////////
        //   Write the iterate to disk for later convergence rate measurement
        // /////////////////////////////////////////////////////////////////////

        if (instrumented_) {

            char iFilename[100];
            sprintf(iFilename, "tmp/intermediateSolution_%04d", i);

            FILE* fpIterate = fopen(iFilename, "wb");
            if (!fpIterate)
                DUNE_THROW(SolverError, "Couldn't open file " << iFilename << " for writing");

            for (size_t j=0; j<x_.size(); j++)
                fwrite(&x_[j], sizeof(TargetSpace), 1, fpIterate);

            fclose(fpIterate);

        }

        if (rank==0)
            std::cout << "iteration took " << totalTimer.elapsed() << " sec." << std::endl;
    }

    // //////////////////////////////////////////////
    //   Close logfile
    // //////////////////////////////////////////////
    if (instrumented_)
        fclose(fp);
}
