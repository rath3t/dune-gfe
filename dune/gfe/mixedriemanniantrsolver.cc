#include "omp.h"

#include <dune/common/bitsetvector.hh>
#include <dune/common/timer.hh>
#include <dune/common/parallel/mpihelper.hh>

#include <dune/istl/io.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/fufem/functionspacebases/dunefunctionsbasis.hh>
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

#include <dune/gfe/parallel/matrixcommunicator.hh>
#include <dune/gfe/parallel/vectorcommunicator.hh>

#include <dune/gfe/cosseratvtkwriter.hh>

template <class GridType,
          class Basis,
          class Basis0, class TargetSpace0,
          class Basis1, class TargetSpace1>
void MixedRiemannianTrustRegionSolver<GridType,Basis,Basis0,TargetSpace0,Basis1,TargetSpace1>::
setup(const GridType& grid,
      const MixedGFEAssembler<Basis, TargetSpace0, TargetSpace1>* assembler,
      const Basis0& tmpBasis0,
      const Basis1& tmpBasis1,
         const SolutionType& x,
         const Dune::BitSetVector<blocksize0>& dirichletNodes0,
         const Dune::BitSetVector<blocksize1>& dirichletNodes1,
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
    basis0_                   = std::unique_ptr<Basis0>(new Basis0(tmpBasis0));
    basis1_                   = std::unique_ptr<Basis1>(new Basis1(tmpBasis1));
    x_                        = x;
    tolerance_                = tolerance;
    maxTrustRegionSteps_      = maxTrustRegionSteps;
    initialTrustRegionRadius_ = initialTrustRegionRadius;
    innerIterations_          = multigridIterations;
    innerTolerance_           = mgTolerance;
    ignoreNodes0_             = &dirichletNodes0;
    ignoreNodes1_             = &dirichletNodes1;

    int numLevels = grid_->maxLevel()+1;

    //////////////////////////////////////////////////////////////////
    //  Create global numbering for matrix and vector transfer
    //////////////////////////////////////////////////////////////////
#if 0
    guIndex_ = std::unique_ptr<GUIndex>(new GUIndex(grid_->leafGridView()));
#endif
    // ////////////////////////////////
    //   Create a multigrid solver
    // ////////////////////////////////
#ifdef HAVE_IPOPT
    // First create an IPOpt base solver
    auto baseSolver0 = std::make_shared<QuadraticIPOptSolver<MatrixType00,CorrectionType0>>();
    baseSolver0->setVerbosity(NumProc::QUIET);
    baseSolver0->setTolerance(baseTolerance);
    auto baseSolver1 = std::make_shared<QuadraticIPOptSolver<MatrixType11,CorrectionType1>>();
    baseSolver1->setVerbosity(NumProc::QUIET);
    baseSolver1->setTolerance(baseTolerance);
#else
    // First create a Gauss-seidel base solver
    TrustRegionGSStep<MatrixType00, CorrectionType0>* baseSolverStep0 = new TrustRegionGSStep<MatrixType00, CorrectionType0>;
    TrustRegionGSStep<MatrixType11, CorrectionType1>* baseSolverStep1 = new TrustRegionGSStep<MatrixType11, CorrectionType1>;

    // Hack: the two-norm may not scale all that well, but it is fast!
    TwoNorm<CorrectionType0>* baseNorm0 = new TwoNorm<CorrectionType0>;
    TwoNorm<CorrectionType1>* baseNorm1 = new TwoNorm<CorrectionType1>;

    ::LoopSolver<CorrectionType0>* baseSolver0 = new ::LoopSolver<CorrectionType0>(baseSolverStep0,
                                                                                   baseIterations,
                                                                                   baseTolerance,
                                                                                   baseNorm0,
                                                                                   Solver::QUIET);

    ::LoopSolver<CorrectionType0>* baseSolver1 = new ::LoopSolver<CorrectionType1>(baseSolverStep1,
                                                                                   baseIterations,
                                                                                   baseTolerance,
                                                                                   baseNorm1,
                                                                                   Solver::QUIET);
#endif

    // Transfer all Dirichlet data to the master processor
#if 0
    VectorCommunicator<GUIndex, Dune::BitSetVector<blocksize> > vectorComm(*guIndex_, 0);
    Dune::BitSetVector<blocksize>* globalDirichletNodes = NULL;
    globalDirichletNodes = new Dune::BitSetVector<blocksize>(vectorComm.reduceCopy(dirichletNodes));
#endif
    Dune::BitSetVector<blocksize0>* globalDirichletNodes0 = NULL;
    globalDirichletNodes0 = new Dune::BitSetVector<blocksize0>(dirichletNodes0);
    Dune::BitSetVector<blocksize1>* globalDirichletNodes1 = NULL;
    globalDirichletNodes1 = new Dune::BitSetVector<blocksize1>(dirichletNodes1);


    // Make pre and postsmoothers
    TrustRegionGSStep<MatrixType00, CorrectionType0>* presmoother0  = new TrustRegionGSStep<MatrixType00, CorrectionType0>;
    TrustRegionGSStep<MatrixType00, CorrectionType0>* postsmoother0 = new TrustRegionGSStep<MatrixType00, CorrectionType0>;

    mmgStep0 = new MonotoneMGStep<MatrixType00, CorrectionType0>;

    mmgStep0->setMGType(mu, nu1, nu2);
    mmgStep0->ignoreNodes_ = globalDirichletNodes0;
    mmgStep0->setBaseSolver(baseSolver0);
    mmgStep0->setSmoother(presmoother0, postsmoother0);
    mmgStep0->setObstacleRestrictor(std::make_shared<MandelObstacleRestrictor<CorrectionType0>>());
    mmgStep0->setVerbosity(Solver::QUIET);

    TrustRegionGSStep<MatrixType11, CorrectionType1>* presmoother1  = new TrustRegionGSStep<MatrixType11, CorrectionType1>;
    TrustRegionGSStep<MatrixType11, CorrectionType1>* postsmoother1 = new TrustRegionGSStep<MatrixType11, CorrectionType1>;

    mmgStep1 = new MonotoneMGStep<MatrixType11, CorrectionType1>;

    mmgStep1->setMGType(mu, nu1, nu2);
    mmgStep1->ignoreNodes_ = globalDirichletNodes1;
    mmgStep1->setBaseSolver(baseSolver1);
    mmgStep1->setSmoother(presmoother1, postsmoother1);
    mmgStep1->setObstacleRestrictor(std::make_shared<MandelObstacleRestrictor<CorrectionType1>>());
    mmgStep1->setVerbosity(Solver::QUIET);

    // //////////////////////////////////////////////////////////////////////////////////////
    //   Assemble a Laplace matrix to create a norm that's equivalent to the H1-norm
    // //////////////////////////////////////////////////////////////////////////////////////
    typedef DuneFunctionsBasis<Basis0> FufemBasis0;
    FufemBasis0 basis0(grid.leafGridView());

    typedef DuneFunctionsBasis<Basis1> FufemBasis1;
    FufemBasis1 basis1(grid.leafGridView());

#if 0
    BasisType basis(grid.leafGridView());
    OperatorAssembler<BasisType,BasisType> operatorAssembler(basis, basis);

    LaplaceAssembler<GridType, typename BasisType::LocalFiniteElement, typename BasisType::LocalFiniteElement> laplaceStiffness;
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> > ScalarMatrixType;
    ScalarMatrixType localA;

    operatorAssembler.assemble(laplaceStiffness, localA);

    if (h1SemiNorm_)
        delete h1SemiNorm_;


    MatrixCommunicator<GUIndex, ScalarMatrixType> matrixComm(*guIndex_, 0);
    ScalarMatrixType* A = new ScalarMatrixType(matrixComm.reduceAdd(localA));

    h1SemiNorm_ = new H1SemiNorm<CorrectionType>(*A);

    innerSolver_ = std::shared_ptr<LoopSolver<CorrectionType> >(new ::LoopSolver<CorrectionType>(mmgStep,
                                                                                                   innerIterations_,
                                                                                                   innerTolerance_,
                                                                                                   h1SemiNorm_,
                                                                                                 Solver::FULL));
#endif

    // ////////////////////////////////////////////////////////////
    //    Create Hessian matrix and its occupation structure
    // ////////////////////////////////////////////////////////////

    // \todo Why are the hessianMatrix objects class members at all, and not local to 'solve'?

    hessianMatrix_ = std::make_unique<MatrixType>();

    // ////////////////////////////////////
    //   Create the transfer operators
    // ////////////////////////////////////

    mmgStep0->mgTransfer_.resize(numLevels-1);
    mmgStep1->mgTransfer_.resize(numLevels-1);

    if (basis0.getLocalFiniteElement(*grid.leafGridView().template begin<0>()).localBasis().order() > 1)
    {
      if (numLevels>1) {
        typedef typename TruncatedCompressedMGTransfer<CorrectionType0>::TransferOperatorType TransferOperatorType;
        DuneFunctionsBasis<Dune::Functions::LagrangeBasis<typename GridType::LeafGridView,1> > p1Basis(grid_->leafGridView());

        TransferOperatorType pkToP1TransferMatrix;
        assembleBasisInterpolationMatrix<TransferOperatorType,
                                         DuneFunctionsBasis<Dune::Functions::LagrangeBasis<typename GridType::LeafGridView,1> >,
                                         FufemBasis0>(pkToP1TransferMatrix,p1Basis,*basis0_);

        mmgStep0->mgTransfer_.back() = std::make_shared<TruncatedCompressedMGTransfer<CorrectionType0>>();
        Dune::shared_ptr<TransferOperatorType> topTransferOperator = Dune::make_shared<TransferOperatorType>(pkToP1TransferMatrix);
        std::dynamic_pointer_cast<TruncatedCompressedMGTransfer<CorrectionType0>>(mmgStep0->mgTransfer_.back())->setMatrix(topTransferOperator);

        for (size_t i=0; i<mmgStep0->mgTransfer_.size()-1; i++){
          // Construct the local multigrid transfer matrix
          auto newTransferOp0 = std::make_shared<TruncatedCompressedMGTransfer<CorrectionType0>>();
          newTransferOp0->setup(*grid_,i+1,i+2);

          mmgStep0->mgTransfer_[i] = newTransferOp0;
        }

      }

    }
    else  // order == 1
    {

      for (size_t i=0; i<mmgStep0->mgTransfer_.size(); i++){

        // Construct the local multigrid transfer matrix
        auto newTransferOp0 = std::make_shared<TruncatedCompressedMGTransfer<CorrectionType0>>();
        newTransferOp0->setup(*grid_,i,i+1);

        mmgStep0->mgTransfer_[i] = newTransferOp0;
      }

    }

    if (basis1.getLocalFiniteElement(*grid.leafGridView().template begin<0>()).localBasis().order() > 1)
    {
      if (numLevels>1) {
        typedef typename TruncatedCompressedMGTransfer<CorrectionType1>::TransferOperatorType TransferOperatorType;
        DuneFunctionsBasis<Dune::Functions::LagrangeBasis<typename GridType::LeafGridView,1> > p1Basis(grid_->leafGridView());

        TransferOperatorType pkToP1TransferMatrix;
        assembleBasisInterpolationMatrix<TransferOperatorType,
                                         DuneFunctionsBasis<Dune::Functions::LagrangeBasis<typename GridType::LeafGridView,1> >,
                                         FufemBasis1>(pkToP1TransferMatrix,p1Basis,*basis1_);

        mmgStep1->mgTransfer_.back() = std::make_shared<TruncatedCompressedMGTransfer<CorrectionType1>>();
        std::shared_ptr<TransferOperatorType> topTransferOperator = std::make_shared<TransferOperatorType>(pkToP1TransferMatrix);
        std::dynamic_pointer_cast<TruncatedCompressedMGTransfer<CorrectionType1>>(mmgStep1->mgTransfer_.back())->setMatrix(topTransferOperator);

        for (size_t i=0; i<mmgStep1->mgTransfer_.size()-1; i++){
          // Construct the local multigrid transfer matrix
          auto newTransferOp1 = std::make_shared<TruncatedCompressedMGTransfer<CorrectionType1>>();
          newTransferOp1->setup(*grid_,i+1,i+2);
          mmgStep1->mgTransfer_[i] = newTransferOp1;
        }

      }

    }
    else  // order == 1
    {

      for (size_t i=0; i<mmgStep1->mgTransfer_.size(); i++){

        // Construct the local multigrid transfer matrix
        auto newTransferOp = std::make_shared<TruncatedCompressedMGTransfer<CorrectionType1>>();
        newTransferOp->setup(*grid_,i,i+1);
        mmgStep1->mgTransfer_[i] = newTransferOp;
      }
    }

    // //////////////////////////////////////////////////////////
    //   Create obstacles
    // //////////////////////////////////////////////////////////

    if (rank==0)
    {
      #if 0
      hasObstacle0_.resize(guIndex_->nGlobalEntity(), true);
      #else
      hasObstacle0_.resize(assembler->basis_.size({0}), true);
      hasObstacle1_.resize(assembler->basis_.size({1}), true);
      #endif
      mmgStep0->setHasObstacles(hasObstacle0_);
      mmgStep1->setHasObstacles(hasObstacle1_);
    }

  }


template <class GridType,
          class Basis,
          class Basis0, class TargetSpace0,
          class Basis1, class TargetSpace1>
void MixedRiemannianTrustRegionSolver<GridType,Basis,Basis0,TargetSpace0,Basis1,TargetSpace1>::solve()
{
    int argc = 0;
    char** argv;
    Dune::MPIHelper& mpiHelper = Dune::MPIHelper::instance(argc,argv);
    int rank = grid_->comm().rank();

    // \todo Use global index set instead of basis for parallel computations
    MaxNormTrustRegion<blocksize0> trustRegion0(assembler_->basis_.size({0}), initialTrustRegionRadius_);
    MaxNormTrustRegion<blocksize1> trustRegion1(assembler_->basis_.size({1}), initialTrustRegionRadius_);
    trustRegion0.set(initialTrustRegionRadius_, std::get<0>(scaling_));
    trustRegion1.set(initialTrustRegionRadius_, std::get<1>(scaling_));

    std::vector<BoxConstraint<field_type,blocksize0> > trustRegionObstacles0;
    std::vector<BoxConstraint<field_type,blocksize1> > trustRegionObstacles1;

    // /////////////////////////////////////////////////////
    //   Trust-Region Solver
    // /////////////////////////////////////////////////////

    using namespace Dune::TypeTree::Indices;

    double oldEnergy = assembler_->computeEnergy(x_[_0], x_[_1]);
    oldEnergy = mpiHelper.getCollectiveCommunication().sum(oldEnergy);

    bool recomputeGradientHessian = true;
    CorrectionType rhs;
    MatrixType stiffnessMatrix;
    CorrectionType rhs_global;
#if 0
    VectorCommunicator<GUIndex, CorrectionType> vectorComm(*guIndex_, 0);
    MatrixCommunicator<GUIndex, MatrixType> matrixComm(*guIndex_, 0);
#endif

    for (int i=0; i<maxTrustRegionSteps_; i++) {

        Dune::Timer totalTimer;
        if (this->verbosity_ == Solver::FULL and rank==0) {
            std::cout << "----------------------------------------------------" << std::endl;
            std::cout << "      Trust-Region Step Number: " << i
                      << ",     radius: " << trustRegion0.radius()
                      << ",     energy: " << oldEnergy << std::endl;
            std::cout << "----------------------------------------------------" << std::endl;
        }

        CorrectionType corr;
        corr[_0].resize(x_[_0].size());
        corr[_1].resize(x_[_1].size());
        corr = 0;

        Dune::Timer assemblyTimer;

        if (recomputeGradientHessian) {

            assembler_->assembleGradientAndHessian(x_[_0],
                                                   x_[_1],
                                                   rhs[_0],
                                                   rhs[_1],
                                                   *hessianMatrix_,
                                                   i==0    // assemble occupation pattern only for the first call
                                                   );

            rhs *= -1;        // The right hand side is the _negative_ gradient

            if (this->verbosity_ == Solver::FULL)
              std::cout << "Assembly took " << assemblyTimer.elapsed() << " sec." << std::endl;

            // Transfer matrix data
#if 0
            stiffnessMatrix00 = matrixComm.reduceAdd(*hessianMatrix00_);
            stiffnessMatrix01 = matrixComm.reduceAdd(*hessianMatrix01_);
            stiffnessMatrix10 = matrixComm.reduceAdd(*hessianMatrix10_);
            stiffnessMatrix11 = matrixComm.reduceAdd(*hessianMatrix11_);
#endif
            stiffnessMatrix = *hessianMatrix_;

            // Transfer vector data
#if 0
            rhs_global0 = vectorComm.reduceAdd(rhs0);
            rhs_global1 = vectorComm.reduceAdd(rhs1);
#endif
            rhs_global = rhs;

            recomputeGradientHessian = false;

        }

        CorrectionType corr_global;
        corr_global[_0].resize(rhs_global[_0].size());
        corr_global[_1].resize(rhs_global[_1].size());
        corr_global = 0;

        if (rank==0)
        {
            ///////////////////////////////
            //    Solve !
            ///////////////////////////////
            CorrectionType residual = rhs_global;

            mmgStep0->setProblem(stiffnessMatrix[_0][_0], corr_global[_0], residual[_0]);
            trustRegionObstacles0 = trustRegion0.obstacles();
            mmgStep0->setObstacles(trustRegionObstacles0);

            mmgStep0->preprocess();

            mmgStep1->setProblem(stiffnessMatrix[_1][_1], corr_global[_1], residual[_1]);
            trustRegionObstacles1 = trustRegion1.obstacles();
            mmgStep1->setObstacles(trustRegionObstacles1);

            mmgStep1->preprocess();


            std::cout << "Solve quadratic problem..." << std::endl;
            double oldEnergy = 0;
            Dune::Timer solutionTimer;
            for (int ii=0; ii<innerIterations_; ii++)
            {
              residual[_0] = rhs_global[_0];
              stiffnessMatrix[_0][_1].mmv(corr_global[_1], residual[_0]);
              mmgStep0->setRhs(residual[_0]);
              mmgStep0->iterate();

              residual[_1] = rhs_global[_1];
              stiffnessMatrix[_1][_0].mmv(corr_global[_0], residual[_1]);
              mmgStep1->setRhs(residual[_1]);
              mmgStep1->iterate();

              // Compute energy
              CorrectionType tmp(corr_global);
              stiffnessMatrix.mv(corr_global,tmp);
              double energy = 0.5 * (tmp*corr_global) - (rhs_global*corr_global);

              if (energy > oldEnergy)
                //DUNE_THROW(Dune::Exception, "energy increase!");
                std::cout << "Warning: energy increase!" << std::endl;

              oldEnergy = energy;
            }

            std::cout << "Solving the quadratic problem took " << solutionTimer.elapsed() << " seconds." << std::endl;

            //std::cout << "Correction: " << std::endl << corr_global << std::endl;

        }

        // Distribute solution
        if (mpiHelper.size()>1 and rank==0)
            std::cout << "Transfer solution back to root process ..." << std::endl;

        //corr = vectorComm.scatter(corr_global);
        corr = corr_global;

        if (this->verbosity_ == NumProc::FULL)
            std::cout << "Infinity norm of the correction: " << corr.infinity_norm() << std::endl;

        if (corr_global.infinity_norm() < tolerance_) {
            if (verbosity_ == NumProc::FULL and rank==0)
                std::cout << "CORRECTION IS SMALL ENOUGH" << std::endl;

            if (verbosity_ != NumProc::QUIET and rank==0)
                std::cout << i+1 << " trust-region steps were taken." << std::endl;
            break;
        }

        // ////////////////////////////////////////////////////
        //   Check whether trust-region step can be accepted
        // ////////////////////////////////////////////////////

        SolutionType newIterate = x_;
        for (size_t j=0; j<newIterate[_0].size(); j++)
            newIterate[_0][j] = TargetSpace0::exp(newIterate[_0][j], corr[_0][j]);

        for (size_t j=0; j<newIterate[_1].size(); j++)
            newIterate[_1][j] = TargetSpace1::exp(newIterate[_1][j], corr[_1][j]);

        double energy    = assembler_->computeEnergy(newIterate[_0],newIterate[_1]);
        energy = mpiHelper.getCollectiveCommunication().sum(energy);

        // compute the model decrease
        // It is $ m(x) - m(x+s) = -<g,s> - 0.5 <s, Hs>
        // Note that rhs = -g
        CorrectionType tmp(corr);
        hessianMatrix_->mv(corr,tmp);
        double modelDecrease = rhs*corr - 0.5 * (corr*tmp);
        modelDecrease = mpiHelper.getCollectiveCommunication().sum(modelDecrease);

        double relativeModelDecrease = modelDecrease / std::fabs(energy);

        if (verbosity_ == NumProc::FULL and rank==0) {
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
            trustRegion0.scale(2);
            trustRegion1.scale(2);

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
            trustRegion0.scale(0.5);
            trustRegion1.scale(0.5);

            if (this->verbosity_ == NumProc::FULL and rank==0)
                std::cout << "Unsuccessful iteration!" << std::endl;
        }
#if 0
        // Output each iterate, to better understand what the algorithm does
        DuneFunctionsBasis<Basis0> fufemBasis0(assembler_->basis0_);
        DuneFunctionsBasis<Basis1> fufemBasis1(assembler_->basis1_);
        std::stringstream iAsAscii;
        iAsAscii << i+1;
        CosseratVTKWriter<GridType>::template writeMixed<DuneFunctionsBasis<Basis0>, DuneFunctionsBasis<Basis1> >(fufemBasis0,x_[_0],
                                                                        fufemBasis1,x_[_1],
                                                                        "mixed-cosserat_iterate_" + iAsAscii.str());
#endif
        if (rank==0)
            std::cout << "iteration took " << totalTimer.elapsed() << " sec." << std::endl;

    }

}
