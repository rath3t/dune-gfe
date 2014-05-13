#include "omp.h"

#include <dune/common/bitsetvector.hh>
#include <dune/common/timer.hh>
#include <dune/common/parallel/mpihelper.hh>

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

#include <dune/solvers/norms/energynorm.hh>
#include <dune/solvers/norms/h1seminorm.hh>


template <class GridType, class TargetSpace>
void RiemannianTrustRegionSolver<GridType,TargetSpace>::
setup(const GridType& grid,
      const GeodesicFEAssembler<BasisType, TargetSpace>* assembler,
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

    // ////////////////////////////////
    //   Create a multigrid solver
    // ////////////////////////////////

#ifdef HAVE_IPOPT
    // First create an IPOpt base solver
    QuadraticIPOptSolver<MatrixType, CorrectionType>* baseSolver = new QuadraticIPOptSolver<MatrixType,CorrectionType>;
    baseSolver->verbosity_ = NumProc::QUIET;
    baseSolver->tolerance_ = baseTolerance;
#else
#warning IPOpt not installed -- falling back onto a Gauss-Seidel base solver
    // First create a Gauss-seidel base solver
    TrustRegionGSStep<MatrixType, CorrectionType>* baseSolverStep = new TrustRegionGSStep<MatrixType, CorrectionType>;

    EnergyNorm<MatrixType, CorrectionType>* baseEnergyNorm = new EnergyNorm<MatrixType, CorrectionType>(*baseSolverStep);

    ::LoopSolver<CorrectionType>* baseSolver = new ::LoopSolver<CorrectionType>(baseSolverStep,
                                                                            baseIterations,
                                                                            baseTolerance,
                                                                            baseEnergyNorm,
                                                                            Solver::QUIET);
#endif

    // Make pre and postsmoothers
    TrustRegionGSStep<MatrixType, CorrectionType>* presmoother  = new TrustRegionGSStep<MatrixType, CorrectionType>;
    TrustRegionGSStep<MatrixType, CorrectionType>* postsmoother = new TrustRegionGSStep<MatrixType, CorrectionType>;

    MonotoneMGStep<MatrixType, CorrectionType>* mmgStep = new MonotoneMGStep<MatrixType, CorrectionType>;

    mmgStep->setMGType(mu, nu1, nu2);
    mmgStep->ignoreNodes_       = &dirichletNodes;
    mmgStep->basesolver_        = baseSolver;
    mmgStep->setSmoother(presmoother, postsmoother);
    mmgStep->obstacleRestrictor_= new MandelObstacleRestrictor<CorrectionType>();
    mmgStep->hasObstacle_       = &hasObstacle_;
    mmgStep->verbosity_         = Solver::QUIET;

    // //////////////////////////////////////////////////////////////////////////////////////
    //   Assemble a Laplace matrix to create a norm that's equivalent to the H1-norm
    // //////////////////////////////////////////////////////////////////////////////////////
    typedef P1NodalBasis<typename GridType::LeafGridView,double> P1Basis;
    P1Basis p1Basis(grid.leafGridView());
    OperatorAssembler<P1Basis,P1Basis> operatorAssembler(p1Basis, p1Basis);

    LaplaceAssembler<GridType, typename P1Basis::LocalFiniteElement, typename P1Basis::LocalFiniteElement> laplaceStiffness;
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> > ScalarMatrixType;
    ScalarMatrixType* A = new ScalarMatrixType;

    operatorAssembler.assemble(laplaceStiffness, *A);

    if (h1SemiNorm_)
        delete h1SemiNorm_;

    h1SemiNorm_ = new H1SemiNorm<CorrectionType>(*A);

    innerSolver_ = std::shared_ptr<LoopSolver<CorrectionType> >(new ::LoopSolver<CorrectionType>(mmgStep,
                                                                                                   innerIterations_,
                                                                                                   innerTolerance_,
                                                                                                   h1SemiNorm_,
                                                                                                 Solver::QUIET));

    // Write all intermediate solutions, if requested
    if (instrumented_
        && dynamic_cast<IterativeSolver<CorrectionType>*>(innerSolver_.get()))
        dynamic_cast<IterativeSolver<CorrectionType>*>(innerSolver_.get())->historyBuffer_ = "tmp/mgHistory";

    // ////////////////////////////////////////////////////////////
    //    Create Hessian matrix and its occupation structure
    // ////////////////////////////////////////////////////////////

    hessianMatrix_ = std::auto_ptr<MatrixType>(new MatrixType);
    Dune::MatrixIndexSet indices(grid_->size(1), grid_->size(1));
    assembler_->getNeighborsPerVertex(indices);
    indices.exportIdx(*hessianMatrix_);

    // //////////////////////////////////////////////////////////
    //   Create obstacles
    // //////////////////////////////////////////////////////////

    hasObstacle_.resize(numLevels);
#if defined THIRD_ORDER || defined SECOND_ORDER
    BasisType basis(grid_->leafGridView());
    hasObstacle_.back().resize(basis.size(), true);

    for (int i=0; i<hasObstacle_.size()-1; i++)
        hasObstacle_[i].resize(grid_->size(i+1, gridDim),true);
#else
    for (size_t i=0; i<hasObstacle_.size(); i++)
        hasObstacle_[i].resize(grid_->size(i, gridDim),true);
#endif

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
            TruncatedCompressedMGTransfer<CorrectionType>* newTransferOp = new TruncatedCompressedMGTransfer<CorrectionType>;
            newTransferOp->setup(*grid_,i+1,i+2);
            mmgStep->mgTransfer_[i] = newTransferOp;
        }

    }

#else
    for (size_t i=0; i<mmgStep->mgTransfer_.size(); i++){
        TruncatedCompressedMGTransfer<CorrectionType>* newTransferOp = new TruncatedCompressedMGTransfer<CorrectionType>;
        newTransferOp->setup(*grid_,i,i+1);
        mmgStep->mgTransfer_[i] = newTransferOp;
    }
#endif
}


template <class GridType, class TargetSpace>
void RiemannianTrustRegionSolver<GridType,TargetSpace>::solve()
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

    MaxNormTrustRegion<blocksize> trustRegion(x_.size(), initialTrustRegionRadius_);

    std::vector<std::vector<BoxConstraint<field_type,blocksize> > > trustRegionObstacles((mgStep)
                                                                                         ? mgStep->numLevels()
                                                                                         : 0);

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

            if (this->verbosity_ == Solver::FULL)
              std::cout << "Assembly took " << gradientTimer.elapsed() << " sec." << std::endl;

            recomputeGradientHessian = false;
            
        }

/*        std::cout << "rhs:\n" << rhs << std::endl;
        std::cout << "matrix[0][0]:\n" << (*hessianMatrix_)[0][0] << std::endl;*/

        // //////////////////////////////////////////////////////////////////////
        //   Modify matrix and right-hand side to account for Dirichlet values
        // //////////////////////////////////////////////////////////////////////

        typedef typename MatrixType::row_type::Iterator ColumnIterator;

        for (size_t j=0; j<ignoreNodes_->size(); j++) {

            if (ignoreNodes_->operator[](j).count() > 0) {

                // make matrix row an identity row
                ColumnIterator cIt    = (*hessianMatrix_)[j].begin();
                ColumnIterator cEndIt = (*hessianMatrix_)[j].end();

                for (; cIt!=cEndIt; ++cIt) {
                    for (int k=0; k<blocksize; k++) {
                        if (ignoreNodes_->operator[](j)[k]) {
                            (*cIt)[k] = 0;
                            if (j==cIt.index())
                                (*cIt)[k][k] = 1;
                        }
                    }
                }

                // Dirichlet value.  Zero, because we are solving defect problems
                for (int k=0; k<blocksize; k++)
                    if (ignoreNodes_->operator[](j)[k])
                        rhs[j][k] = 0;
            }

        }

        mgStep->setProblem(*hessianMatrix_, corr, rhs);

        trustRegionObstacles.back() = trustRegion.obstacles();
        mgStep->obstacles_ = &trustRegionObstacles;

        innerSolver_->preprocess();

        // /////////////////////////////
        //    Solve !
        // /////////////////////////////

        innerSolver_->solve();

        if (mgStep)
            corr = mgStep->getSol();

        //std::cout << "Correction: " << std::endl << corr << std::endl;

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
            newIterate[j] = TargetSpace::exp(newIterate[j], corr[j]);

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
            (std::abs(oldEnergy-energy)/energy < 1e-9 || relativeModelDecrease < 1e-9)) {
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
        std::cout << "iteration took " << totalTimer.elapsed() << " sec." << std::endl;
    }

    // //////////////////////////////////////////////
    //   Close logfile
    // //////////////////////////////////////////////
    if (instrumented_)
        fclose(fp);
}
