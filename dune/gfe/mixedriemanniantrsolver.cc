#include "omp.h"

#include <dune/common/bitsetvector.hh>
#include <dune/common/timer.hh>
#include <dune/common/parallel/mpihelper.hh>

#include <dune/istl/io.hh>

#include <dune/fufem/functionspacebases/p1nodalbasis.hh>
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
          class Basis0, class TargetSpace0,
          class Basis1, class TargetSpace1>
void MixedRiemannianTrustRegionSolver<GridType,Basis0,TargetSpace0,Basis1,TargetSpace1>::
setup(const GridType& grid,
      const MixedGFEAssembler<Basis0, TargetSpace0, Basis1, TargetSpace1>* assembler,
         const SolutionType0& x0,
         const SolutionType1& x1,
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
    x0_                       = x0;
    x1_                       = x1;
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
    mmgStep0->basesolver_        = baseSolver0;
    mmgStep0->setSmoother(presmoother0, postsmoother0);
    mmgStep0->obstacleRestrictor_= new MandelObstacleRestrictor<CorrectionType0>();
    mmgStep0->verbosity_         = Solver::FULL;

    TrustRegionGSStep<MatrixType11, CorrectionType1>* presmoother1  = new TrustRegionGSStep<MatrixType11, CorrectionType1>;
    TrustRegionGSStep<MatrixType11, CorrectionType1>* postsmoother1 = new TrustRegionGSStep<MatrixType11, CorrectionType1>;

    mmgStep1 = new MonotoneMGStep<MatrixType11, CorrectionType1>;

    mmgStep1->setMGType(mu, nu1, nu2);
    mmgStep1->ignoreNodes_ = globalDirichletNodes1;
    mmgStep1->basesolver_        = baseSolver1;
    mmgStep1->setSmoother(presmoother1, postsmoother1);
    mmgStep1->obstacleRestrictor_= new MandelObstacleRestrictor<CorrectionType1>();
    mmgStep1->verbosity_         = Solver::FULL;

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

    hessianMatrix00_ = std::unique_ptr<MatrixType00>(new MatrixType00);
    hessianMatrix01_ = std::unique_ptr<MatrixType01>(new MatrixType01);
    hessianMatrix10_ = std::unique_ptr<MatrixType10>(new MatrixType10);
    hessianMatrix11_ = std::unique_ptr<MatrixType11>(new MatrixType11);

    // ////////////////////////////////////
    //   Create the transfer operators
    // ////////////////////////////////////

    for (size_t k=0; k<mmgStep0->mgTransfer_.size(); k++)
        delete(mmgStep0->mgTransfer_[k]);

    for (size_t k=0; k<mmgStep1->mgTransfer_.size(); k++)
        delete(mmgStep1->mgTransfer_[k]);

    mmgStep0->mgTransfer_.resize(numLevels-1);
    mmgStep1->mgTransfer_.resize(numLevels-1);

    if (basis0.getLocalFiniteElement(*grid.leafGridView().template begin<0>()).localBasis().order() > 1)
    {
      if (numLevels>1) {
        typedef typename TruncatedCompressedMGTransfer<CorrectionType0>::TransferOperatorType TransferOperatorType;
        P1NodalBasis<typename GridType::LeafGridView,double> p1Basis(grid_->leafGridView());

        TransferOperatorType pkToP1TransferMatrix;
        assembleBasisInterpolationMatrix<TransferOperatorType,
                                         P1NodalBasis<typename GridType::LeafGridView,double>,
                                         FufemBasis0>(pkToP1TransferMatrix,p1Basis,assembler->basis0_);

        mmgStep0->mgTransfer_.back() = new TruncatedCompressedMGTransfer<CorrectionType0>;
        Dune::shared_ptr<TransferOperatorType> topTransferOperator = Dune::make_shared<TransferOperatorType>(pkToP1TransferMatrix);
        dynamic_cast<TruncatedCompressedMGTransfer<CorrectionType0>*>(mmgStep0->mgTransfer_.back())->setMatrix(topTransferOperator);

        for (size_t i=0; i<mmgStep0->mgTransfer_.size()-1; i++){
          // Construct the local multigrid transfer matrix
          TruncatedCompressedMGTransfer<CorrectionType0>* newTransferOp0 = new TruncatedCompressedMGTransfer<CorrectionType0>;
          newTransferOp0->setup(*grid_,i+1,i+2);

          mmgStep0->mgTransfer_[i] = newTransferOp0;
        }

      }

    }
    else  // order == 1
    {

      for (size_t i=0; i<mmgStep0->mgTransfer_.size(); i++){

        // Construct the local multigrid transfer matrix
        TruncatedCompressedMGTransfer<CorrectionType0>* newTransferOp0 = new TruncatedCompressedMGTransfer<CorrectionType0>;
        newTransferOp0->setup(*grid_,i,i+1);

        mmgStep0->mgTransfer_[i] = newTransferOp0;
      }

    }

    if (basis1.getLocalFiniteElement(*grid.leafGridView().template begin<0>()).localBasis().order() > 1)
    {
      if (numLevels>1) {
        typedef typename TruncatedCompressedMGTransfer<CorrectionType1>::TransferOperatorType TransferOperatorType;
        P1NodalBasis<typename GridType::LeafGridView,double> p1Basis(grid_->leafGridView());

        TransferOperatorType pkToP1TransferMatrix;
        assembleBasisInterpolationMatrix<TransferOperatorType,
                                         P1NodalBasis<typename GridType::LeafGridView,double>,
                                         FufemBasis1>(pkToP1TransferMatrix,p1Basis,assembler->basis1_);

        mmgStep0->mgTransfer_.back() = new TruncatedCompressedMGTransfer<CorrectionType1>;
        Dune::shared_ptr<TransferOperatorType> topTransferOperator = Dune::make_shared<TransferOperatorType>(pkToP1TransferMatrix);
        dynamic_cast<TruncatedCompressedMGTransfer<CorrectionType1>*>(mmgStep1->mgTransfer_.back())->setMatrix(topTransferOperator);

        for (size_t i=0; i<mmgStep1->mgTransfer_.size()-1; i++){
          // Construct the local multigrid transfer matrix
          TruncatedCompressedMGTransfer<CorrectionType1>* newTransferOp1 = new TruncatedCompressedMGTransfer<CorrectionType1>;
          newTransferOp1->setup(*grid_,i+1,i+2);
          mmgStep1->mgTransfer_[i] = newTransferOp1;
        }

      }

    }
    else  // order == 1
    {

      for (size_t i=0; i<mmgStep1->mgTransfer_.size(); i++){

        // Construct the local multigrid transfer matrix
        TruncatedCompressedMGTransfer<CorrectionType1>* newTransferOp = new TruncatedCompressedMGTransfer<CorrectionType1>;
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
      hasObstacle0_.resize(assembler->basis0_.indexSet().size(), true);
      hasObstacle1_.resize(assembler->basis1_.indexSet().size(), true);
      #endif
      mmgStep0->hasObstacle_ = &hasObstacle0_;
      mmgStep1->hasObstacle_ = &hasObstacle1_;
    }

  }


template <class GridType,
          class Basis0, class TargetSpace0,
          class Basis1, class TargetSpace1>
void MixedRiemannianTrustRegionSolver<GridType,Basis0,TargetSpace0,Basis1,TargetSpace1>::solve()
{
    int argc = 0;
    char** argv;
    Dune::MPIHelper& mpiHelper = Dune::MPIHelper::instance(argc,argv);
    int rank = grid_->comm().rank();

    // \todo Use global index set instead of basis for parallel computations
    MaxNormTrustRegion<blocksize0> trustRegion0(assembler_->basis0_.indexSet().size(), initialTrustRegionRadius_);
    MaxNormTrustRegion<blocksize1> trustRegion1(assembler_->basis1_.indexSet().size(), initialTrustRegionRadius_);

    std::vector<BoxConstraint<field_type,blocksize0> > trustRegionObstacles0;
    std::vector<BoxConstraint<field_type,blocksize1> > trustRegionObstacles1;

    // /////////////////////////////////////////////////////
    //   Trust-Region Solver
    // /////////////////////////////////////////////////////

    double oldEnergy = assembler_->computeEnergy(x0_, x1_);
    oldEnergy = mpiHelper.getCollectiveCommunication().sum(oldEnergy);

    bool recomputeGradientHessian = true;
    CorrectionType0 rhs0;
    CorrectionType1 rhs1;
    MatrixType00 stiffnessMatrix00;
    MatrixType01 stiffnessMatrix01;
    MatrixType10 stiffnessMatrix10;
    MatrixType11 stiffnessMatrix11;
    CorrectionType0 rhs_global0;
    CorrectionType1 rhs_global1;
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

        CorrectionType0 corr0(x0_.size());
        corr0 = 0;
        CorrectionType1 corr1(x1_.size());
        corr1 = 0;

        Dune::Timer assemblyTimer;

        if (recomputeGradientHessian) {

            assembler_->assembleGradientAndHessian(x0_,
                                                   x1_,
                                                   rhs0,
                                                   rhs1,
                                                   *hessianMatrix00_,
                                                   *hessianMatrix01_,
                                                   *hessianMatrix10_,
                                                   *hessianMatrix11_,
                                                   i==0    // assemble occupation pattern only for the first call
                                                   );

            rhs0 *= -1;        // The right hand side is the _negative_ gradient
            rhs1 *= -1;

            if (this->verbosity_ == Solver::FULL)
              std::cout << "Assembly took " << assemblyTimer.elapsed() << " sec." << std::endl;

            // Transfer matrix data
#if 0
            stiffnessMatrix00 = matrixComm.reduceAdd(*hessianMatrix00_);
            stiffnessMatrix01 = matrixComm.reduceAdd(*hessianMatrix01_);
            stiffnessMatrix10 = matrixComm.reduceAdd(*hessianMatrix10_);
            stiffnessMatrix11 = matrixComm.reduceAdd(*hessianMatrix11_);
#endif
            stiffnessMatrix00 = *hessianMatrix00_;
            stiffnessMatrix01 = *hessianMatrix01_;
            stiffnessMatrix10 = *hessianMatrix10_;
            stiffnessMatrix11 = *hessianMatrix11_;

            // Transfer vector data
#if 0
            rhs_global0 = vectorComm.reduceAdd(rhs0);
            rhs_global1 = vectorComm.reduceAdd(rhs1);
#endif
            rhs_global0 = rhs0;
            rhs_global1 = rhs1;

            recomputeGradientHessian = false;

        }

        CorrectionType0 corr_global0(rhs_global0.size());
        corr_global0 = 0;
        CorrectionType1 corr_global1(rhs_global1.size());
        corr_global1 = 0;

        if (rank==0)
        {
            ///////////////////////////////
            //    Solve !
            ///////////////////////////////
            CorrectionType0 residual0 = rhs_global0;
            CorrectionType1 residual1 = rhs_global1;

            mmgStep0->setProblem(stiffnessMatrix00, corr_global0, residual0);
            trustRegionObstacles0 = trustRegion0.obstacles();
            mmgStep0->obstacles_ = &trustRegionObstacles0;

            mmgStep0->preprocess();

            mmgStep1->setProblem(stiffnessMatrix11, corr_global1, residual1);
            trustRegionObstacles1 = trustRegion1.obstacles();
            mmgStep1->obstacles_ = &trustRegionObstacles1;

            mmgStep1->preprocess();


            std::cout << "Solve quadratic problem..." << std::endl;
            double oldEnergy = 0;
            Dune::Timer solutionTimer;
            for (int ii=0; ii<200; ii++)
            {
              std::cout << "Iteration " << ii << std::endl;
              residual0 = rhs_global0;
              stiffnessMatrix01.mmv(corr_global1, residual0);
              mmgStep0->setRhs(residual0);
              mmgStep0->iterate();

              residual1 = rhs_global1;
              stiffnessMatrix10.mmv(corr_global0, residual1);
              mmgStep1->setRhs(residual1);
              mmgStep1->iterate();

              // Compute energy
              CorrectionType0 tmp0(corr_global0);
              stiffnessMatrix00.mv(corr_global0,tmp0);
              stiffnessMatrix01.umv(corr_global1,tmp0);

              CorrectionType1 tmp1(corr_global1);
              stiffnessMatrix10.mv(corr_global0,tmp1);
              stiffnessMatrix11.umv(corr_global1,tmp1);

              double energy = 0.5 * (tmp0*corr_global0 + tmp1*corr_global1) - (rhs_global0*corr_global0 + rhs_global1*corr_global1);


              std::cout << "Energy: " << energy << std::endl;

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
        corr0 = corr_global0;
        corr1 = corr_global1;

        if (this->verbosity_ == NumProc::FULL)
            std::cout << "Infinity norm of the correction: " << std::max(corr0.infinity_norm(), corr1.infinity_norm()) << std::endl;

        if (corr_global0.infinity_norm() < tolerance_ and corr_global1.infinity_norm() < tolerance_) {
            if (verbosity_ == NumProc::FULL and rank==0)
                std::cout << "CORRECTION IS SMALL ENOUGH" << std::endl;

            if (verbosity_ != NumProc::QUIET and rank==0)
                std::cout << i+1 << " trust-region steps were taken." << std::endl;
            break;
        }

        // ////////////////////////////////////////////////////
        //   Check whether trust-region step can be accepted
        // ////////////////////////////////////////////////////

        SolutionType0 newIterate0 = x0_;
        for (size_t j=0; j<newIterate0.size(); j++)
            newIterate0[j] = TargetSpace0::exp(newIterate0[j], corr0[j]);

        SolutionType1 newIterate1 = x1_;
        for (size_t j=0; j<newIterate1.size(); j++)
            newIterate1[j] = TargetSpace1::exp(newIterate1[j], corr1[j]);

        double energy    = assembler_->computeEnergy(newIterate0,newIterate1);
        energy = mpiHelper.getCollectiveCommunication().sum(energy);

        // compute the model decrease
        // It is $ m(x) - m(x+s) = -<g,s> - 0.5 <s, Hs>
        // Note that rhs = -g
        CorrectionType0 tmp0(corr0.size());
        tmp0 = 0;
        hessianMatrix00_->umv(corr0, tmp0);
        hessianMatrix01_->umv(corr1, tmp0);

        CorrectionType1 tmp1(corr1.size());
        tmp1 = 0;
        hessianMatrix10_->umv(corr0, tmp1);
        hessianMatrix11_->umv(corr1, tmp1);

        double modelDecrease = (rhs0*corr0+rhs1*corr1) - 0.5 * (corr0*tmp0+corr1*tmp1);
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

            x0_ = newIterate0;
            x1_ = newIterate1;
            break;
        }

        // //////////////////////////////////////////////
        //   Check for acceptance of the step
        // //////////////////////////////////////////////
        if ( (oldEnergy-energy) / modelDecrease > 0.9) {
            // very successful iteration

            x0_ = newIterate0;
            x1_ = newIterate1;
            trustRegion0.scale(2);
            trustRegion1.scale(2);

            // current energy becomes 'oldEnergy' for the next iteration
            oldEnergy = energy;

            recomputeGradientHessian = true;

        } else if ( (oldEnergy-energy) / modelDecrease > 0.01
                    || std::abs(oldEnergy-energy) < 1e-12) {
            // successful iteration
            x0_ = newIterate0;
            x1_ = newIterate1;

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

        // Output each iterate, to better understand what the algorithm does
        DuneFunctionsBasis<Basis0> fufemBasis0(assembler_->basis0_);
        DuneFunctionsBasis<Basis1> fufemBasis1(assembler_->basis1_);
        std::stringstream iAsAscii;
        iAsAscii << i+1;
        CosseratVTKWriter<GridType>::template writeMixed<DuneFunctionsBasis<Basis0>, DuneFunctionsBasis<Basis1> >(fufemBasis0,x0_,
                                                                        fufemBasis1,x1_,
                                                                        "mixed-cosserat_iterate_" + iAsAscii.str());

        if (rank==0)
            std::cout << "iteration took " << totalTimer.elapsed() << " sec." << std::endl;

    }

}
