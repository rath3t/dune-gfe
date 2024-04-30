#include <dune/common/bitsetvector.hh>
#include <dune/common/timer.hh>

#include <dune/gfe/parallel/matrixcommunicator.hh>
#include <dune/gfe/parallel/vectorcommunicator.hh>

template <class MixedBasis,
    class Basis0,
    class TargetSpace0,
    class Basis1,
    class TargetSpace1,
    class BitVector>
void Dune::GFE::MixedRiemannianProximalNewtonSolver<MixedBasis,Basis0,TargetSpace0,Basis1,TargetSpace1,BitVector>::
setup(const GridType& grid,
      const MixedGFEAssembler<MixedBasis, TargetSpace>* assembler,
      const SolutionType& x,
      const BitVector& dirichletNodes,
      double tolerance,
      int maxProximalNewtonSteps,
      double initialRegularization,
      bool instrumented)
{
  grid_                     = &grid;
  assembler_                = assembler;
  x_                        = x;
  tolerance_                = tolerance;
  maxProximalNewtonSteps_   = maxProximalNewtonSteps;
  initialRegularization_    = initialRegularization;

  //////////////////////////////////////////////////////////////////
  //  Create global numbering for matrix and vector transfer
  //////////////////////////////////////////////////////////////////
#if HAVE_MPI
  globalMapper0_ = std::make_unique<GlobalMapper0>(grid_->leafGridView());
  globalMapper1_ = std::make_unique<GlobalMapper1>(grid_->leafGridView());

  // Transfer all Dirichlet data to the master processor
  using VectorCommunicator0 = VectorCommunicator<GlobalMapper0, typename GridType::LeafGridView::Communication, Dune::BitSetVector<blocksize0> >;
  VectorCommunicator0 vectorComm0(*globalMapper0_,
                                  grid_->leafGridView().comm(),
                                  0);
  using VectorCommunicator1 = VectorCommunicator<GlobalMapper1, typename GridType::LeafGridView::Communication, Dune::BitSetVector<blocksize1> >;
  VectorCommunicator1 vectorComm1(*globalMapper1_,
                                  grid_->leafGridView().comm(),
                                  0);

  using namespace Dune::Indices;
  Dune::BitSetVector<blocksize0> dirichletNodes0(dirichletNodes[_0].size(),false);
  Dune::BitSetVector<blocksize1> dirichletNodes1(dirichletNodes[_1].size(),false);
  for (std::size_t i = 0; i < dirichletNodes[_0].size(); i++)
    for (int j = 0; j < blocksize0; j++)
      dirichletNodes0[i][j] = dirichletNodes[_0][i][j];
  for (std::size_t i = 0; i < dirichletNodes[_1].size(); i++)
    for (int j = 0; j < blocksize1; j++)
      dirichletNodes1[i][j] = dirichletNodes[_1][i][j];

  auto globalDirichletNodes0 = vectorComm0.reduceCopy(dirichletNodes0);
  auto globalDirichletNodes1 = vectorComm1.reduceCopy(dirichletNodes1);

  BitVector globalDirichletNodes01;
  globalDirichletNodes01[_0].resize(globalDirichletNodes0.size());
  globalDirichletNodes01[_1].resize(globalDirichletNodes1.size());
  for (std::size_t i = 0; i < globalDirichletNodes0.size(); i++)
    for (int j = 0; j < blocksize0; j++)
      globalDirichletNodes01[_0][i][j] = globalDirichletNodes0[i][j];
  for (std::size_t i = 0; i < globalDirichletNodes1.size(); i++)
    for (int j = 0; j < blocksize1; j++)
      globalDirichletNodes01[_1][i][j] = globalDirichletNodes1[i][j];

  auto globalDirichletNodes = new BitVector(globalDirichletNodes01);
#else
  auto globalDirichletNodes = new BitVector(dirichletNodes);
#endif

  //////////////////////////////////////////////////////////////////
  //   Create the inner solver using a direct solver
  //////////////////////////////////////////////////////////////////

  innerSolver_ = std::make_shared<Solvers::CholmodSolver<MatrixType,CorrectionType,BitVector> >();

  innerSolver_->setIgnore(*globalDirichletNodes);
  hessianMatrix_ = std::make_unique<MatrixType>();
}


template <class MixedBasis,
    class Basis0,
    class TargetSpace0,
    class Basis1,
    class TargetSpace1,
    class BitVector>
void Dune::GFE::MixedRiemannianProximalNewtonSolver<MixedBasis,Basis0,TargetSpace0,Basis1,TargetSpace1,BitVector>::solve()
{
  int rank = grid_->comm().rank();

  // /////////////////////////////////////////////////////
  //   Proximal Newton Solver
  // /////////////////////////////////////////////////////

  using namespace Dune::TypeTree::Indices;

  Dune::Timer energyTimer;
  double oldEnergy = assembler_->computeEnergy(x_[_0], x_[_1]);
  if (this->verbosity_ == Solver::FULL)
    std::cout << "Energy computation took " << energyTimer.elapsed() << " sec." << std::endl;

  oldEnergy = grid_->comm().sum(oldEnergy);

  bool recomputeGradientHessian = true;
  CorrectionType rhs, rhs_global;
  MatrixType stiffnessMatrix;
#if HAVE_MPI
  using VectorCommunicator0 = VectorCommunicator<GlobalMapper0, typename GridType::LeafGridView::Communication, CorrectionType0>;
  VectorCommunicator0 vectorComm0(*globalMapper0_,
                                  grid_->leafGridView().comm(),
                                  0);
  using VectorCommunicator1 = VectorCommunicator<GlobalMapper1, typename GridType::LeafGridView::Communication, CorrectionType1>;
  VectorCommunicator1 vectorComm1(*globalMapper1_,
                                  grid_->leafGridView().comm(),
                                  0);

  LocalMapper0 localMapper0 = MapperFactory<Basis0>::createLocalMapper(grid_->leafGridView());
  LocalMapper1 localMapper1 = MapperFactory<Basis1>::createLocalMapper(grid_->leafGridView());

  MatrixCommunicator<GlobalMapper0,
      typename GridType::LeafGridView,
      typename GridType::LeafGridView,
      MatrixType00,
      LocalMapper0,
      LocalMapper0> matrixComm00(*globalMapper0_,
                                 grid_->leafGridView(),
                                 localMapper0,
                                 localMapper0,
                                 0);
  MatrixCommunicator<GlobalMapper1,
      typename GridType::LeafGridView,
      typename GridType::LeafGridView,
      MatrixType11,
      LocalMapper1,
      LocalMapper1> matrixComm11(*globalMapper1_,
                                 grid_->leafGridView(),
                                 localMapper1,
                                 localMapper1,
                                 0);
  MatrixCommunicator<GlobalMapper0,
      typename GridType::LeafGridView,
      typename GridType::LeafGridView,
      MatrixType01,
      LocalMapper0,
      LocalMapper1,
      GlobalMapper1> matrixComm01(*globalMapper0_,
                                  *globalMapper1_,
                                  grid_->leafGridView(),
                                  grid_->leafGridView(),
                                  localMapper0,
                                  localMapper1,
                                  0);
  MatrixCommunicator<GlobalMapper1,
      typename GridType::LeafGridView,
      typename GridType::LeafGridView,
      MatrixType10,
      LocalMapper1,
      LocalMapper0,
      GlobalMapper0> matrixComm10(*globalMapper1_,
                                  *globalMapper0_,
                                  grid_->leafGridView(),
                                  grid_->leafGridView(),
                                  localMapper1,
                                  localMapper0,
                                  0);
#endif
  double totalAssemblyTime = 0.0;
  double totalSolverTime = 0.0;
  double regularization = initialRegularization_;
  for (std::size_t i=0; i<maxProximalNewtonSteps_; i++)
  {

    Dune::Timer totalTimer;
    if (this->verbosity_ == Solver::FULL and rank==0) {
      std::cout << "----------------------------------------------------" << std::endl;
      std::cout << "      Mixed Proximal Newton Step Number: " << i
                << ",     regularization parameter: " << regularization
                << ",     energy: " << oldEnergy << std::endl;
      std::cout << "----------------------------------------------------" << std::endl;
    }

    CorrectionType corr;
    corr[_0].resize(x_[_0].size());
    corr[_1].resize(x_[_1].size());
    corr = 0;

    if (recomputeGradientHessian) {
      Dune::Timer assemblyTimer;
      assembler_->assembleGradientAndHessian(x_[_0],
                                             x_[_1],
                                             rhs[_0],
                                             rhs[_1],
                                             *hessianMatrix_,
                                             i==0          // assemble occupation pattern only for the first call
                                             );

      rhs *= -1;              // The right hand side is the _negative_ gradient

      if (this->verbosity_ == Solver::FULL)
        std::cout << "Assembly took " << assemblyTimer.elapsed() << " sec." << std::endl;
      totalAssemblyTime += assemblyTimer.elapsed();

#if HAVE_MPI
      // Transfer matrix data
      stiffnessMatrix[_0][_0] = matrixComm00.reduceAdd((*hessianMatrix_)[_0][_0]);
      stiffnessMatrix[_0][_1] = matrixComm01.reduceAdd((*hessianMatrix_)[_0][_1]);
      stiffnessMatrix[_1][_0] = matrixComm10.reduceAdd((*hessianMatrix_)[_1][_0]);
      stiffnessMatrix[_1][_1] = matrixComm11.reduceAdd((*hessianMatrix_)[_1][_1]);

      // Transfer vector data
      rhs_global[_0] = vectorComm0.reduceAdd(rhs[_0]);
      rhs_global[_1] = vectorComm1.reduceAdd(rhs[_1]);
#else
      stiffnessMatrix = *hessianMatrix_;
      rhs_global = rhs;
#endif
      recomputeGradientHessian = false;
    }

    CorrectionType corr_global;
    corr_global[_0].resize(rhs_global[_0].size());
    corr_global[_1].resize(rhs_global[_1].size());
    corr_global = 0;
    bool solvedByInnerSolver = true;

    if (rank==0)
    {
      // Add the regularization - Identity Matrix for now
      for (std::size_t i=0; i<stiffnessMatrix[_0][_0].N(); i++)
        for (int j=0; j<blocksize0; j++)
          stiffnessMatrix[_0][_0][i][i][j][j] += regularization;
      for (std::size_t i=0; i<stiffnessMatrix[_1][_1].N(); i++)
        for (int j=0; j<blocksize1; j++)
          stiffnessMatrix[_1][_1][i][i][j][j] += regularization;

      innerSolver_->setProblem(stiffnessMatrix,corr_global,rhs_global);
      innerSolver_->preprocess();

      ///////////////////////////////
      //    Solve !
      ///////////////////////////////

      std::cout << "Solve quadratic problem using cholmod solver..." << std::endl;
      Dune::Timer solutionTimer;
      try {
        innerSolver_->solve();
      } catch (Dune::Exception &e) {
        std::cerr << "Error while solving: " << e << std::endl;
        solvedByInnerSolver = false;
        corr_global = 0;
      }
      std::cout << "Solving the quadratic problem took " << solutionTimer.elapsed() << " seconds." << std::endl;
      totalSolverTime += solutionTimer.elapsed();
    }
#if HAVE_MPI
    // Distribute solution
    if (grid_->comm().size()>1 and rank==0)
      std::cout << "Transfer solution back to root process ..." << std::endl;

    corr[_0] = vectorComm0.scatter(corr_global[_0]);
    corr[_1] = vectorComm1.scatter(corr_global[_1]);
#else
    corr = corr_global;
#endif
    double corrNorm = corr.infinity_norm();
    double corrGlobalInfinityNorm = grid_->comm().max(corrNorm);
    if (std::isnan(corrGlobalInfinityNorm))
      solvedByInnerSolver = false;

    double energy = 0;
    double modelDecrease = 0;
    SolutionType newIterate = x_;
    if (i == maxProximalNewtonSteps_ - 1)
      std::cout << i+1 << " proximal newton steps were taken, the maximum was reached." << std::endl << "Total solver time: " << totalSolverTime << " sec., total assembly time: " << totalAssemblyTime << " sec." << std::endl;

    if (solvedByInnerSolver) {
      if (this->verbosity_ == NumProc::FULL && rank==0)
        std::cout << "Infinity norm of the correction: " << corrGlobalInfinityNorm << std::endl;

      if (corrGlobalInfinityNorm < tolerance_) {
        if (verbosity_ == NumProc::FULL and rank==0)
          std::cout << "CORRECTION IS SMALL ENOUGH" << std::endl;

        if (verbosity_ != NumProc::QUIET and rank==0)
          std::cout << i+1 << " proximal newton steps were taken" << std::endl << "Total solver time: " << totalSolverTime << " sec., total assembly time: " << totalAssemblyTime << " sec." << std::endl;
        break;
      }

      // ////////////////////////////////////////////////////
      //   Check whether proximal newton step can be accepted
      // ////////////////////////////////////////////////////

      for (size_t j=0; j<newIterate[_0].size(); j++)
        newIterate[_0][j] = TargetSpace0::exp(newIterate[_0][j], corr[_0][j]);

      for (size_t j=0; j<newIterate[_1].size(); j++)
        newIterate[_1][j] = TargetSpace1::exp(newIterate[_1][j], corr[_1][j]);
      try {
        energy = assembler_->computeEnergy(newIterate[_0],newIterate[_1]);
      } catch (Dune::Exception &e) {
        std::cerr << "Error while computing the energy of the new Iterate: " << e << std::endl;
        std::cerr << "Redoing proximal newton step with higher regularization parameter ..." << std::endl;
        solvedByInnerSolver = false;
      }
      solvedByInnerSolver = grid_->comm().min(solvedByInnerSolver);
      if (!solvedByInnerSolver) {
        newIterate = x_;
        energy = oldEnergy;
      } else {
        energy = grid_->comm().sum(energy);

        /**
         * Compute the model decrease.
         * The lifted quadratic model function defined on the Tangent space of the manifold at the
         * iterate $x_k$ is given by
         *
         *      m(s) := J(x_k) + <Grad J(x_k),s> + 0.5 <(Hess J(x_k)s, s> + 0.5 * \mu_k <s,s>   [QM]
         *
         * where
         * J      : energy functional
         * Grad J : Riemannian Gradient
         * Hess J : Riemannian Hessian
         * \mu_k  : regularization parameter
         *
         * We compute the correction 'corr' as minimizer of the quadratic model function [QM] by
         * solving the assoc. Newton-System:
         *
         * (Hess J(x_k) + \mu_k * I) = - Grad J(x_k)    [NS]
         *
         * With the identity matrix $I$.
         * So far we only use the scalar-product <*,*> inherited from the ambient Euclidean space of
         * the (embedded) manifold.
         *
         * The model decrease is given by the difference
         *
         * m(0) - m(corr) = - <Grad J(x_k),corr> - 0.5 <(Hess J(x_k)corr, corr> - 0.5 * \mu_k <corr,corr>
         *                = - 0.5 * <Grad J(x_k),corr>
         *
         * The last line is due to the fact that 'corr' actually solves the Newton-system [NS].
         *
         * Note that rhs = -g corresponds to the (negative) Riemannian Gradient '- Grad J(x_k)'.
         */
        modelDecrease = 0.5 * (rhs*corr);
        modelDecrease = grid_->comm().sum(modelDecrease);

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
            std::cout << "Direction is not a descent direction!" << std::endl;
        }
      }
    }
    // //////////////////////////////////////////////
    //   Check for acceptance of the step
    // //////////////////////////////////////////////
    if ( solvedByInnerSolver && oldEnergy >= energy && (oldEnergy-energy) / modelDecrease > 0.9)
    {
      // very successful iteration

      x_ = newIterate;
      regularization *= 0.5;

      // current energy becomes 'oldEnergy' for the next iteration
      oldEnergy = energy;

      recomputeGradientHessian = true;

    } else if ((solvedByInnerSolver && oldEnergy >= energy && (oldEnergy-energy) / modelDecrease > 0.01)
               || std::abs(oldEnergy-energy) < 1e-12) {
      // successful iteration
      x_ = newIterate;

      // current energy becomes 'oldEnergy' for the next iteration
      oldEnergy = energy;

      recomputeGradientHessian = true;

    } else {

      // unsuccessful iteration

      // Increase the regularization parameter
      regularization *= 2;

      if (this->verbosity_ == NumProc::FULL and rank==0)
        std::cout << "Unsuccessful iteration!" << std::endl;
    }

    if (rank==0)
      std::cout << "iteration took " << totalTimer.elapsed() << " sec." << std::endl;
  }
}
