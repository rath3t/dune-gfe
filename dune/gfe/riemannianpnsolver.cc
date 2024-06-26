#include <filesystem>
#include <dune/common/bitsetvector.hh>
#include <dune/common/timer.hh>

#include <dune/istl/io.hh>

#include <dune/grid/common/mcmgmapper.hh>

#include <dune/fufem/assemblers/dunefunctionsoperatorassembler.hh>
#include <dune/fufem/assemblers/localassemblers/laplaceassembler.hh>
#include <dune/fufem/assemblers/localassemblers/massassembler.hh>
#include <dune/fufem/assemblers/basisinterpolationmatrixassembler.hh>

#include <dune/solvers/norms/twonorm.hh>
#include <dune/solvers/norms/h1seminorm.hh>
#include <dune/solvers/solvers/cholmodsolver.hh>

#include <dune/gfe/parallel/matrixcommunicator.hh>
#include <dune/gfe/parallel/vectorcommunicator.hh>

template <class Basis, class TargetSpace, class Assembler>
void RiemannianProximalNewtonSolver<Basis, TargetSpace, Assembler>::
setup(const GridType& grid,
      const Assembler* assembler,
      const SolutionType& x,
      const Dune::BitSetVector<blocksize>& dirichletNodes,
      const Dune::ParameterTree& parameterSet)
{
  if(parameterSet.get("norm", "infinity") == "infinity")
    normType_ = ErrorNormType::infinity;
  else if(parameterSet.get("norm", "infinity") == "H1semi")
    normType_ = ErrorNormType::H1semi;
  else
    DUNE_THROW(Dune::Exception, "Unknown norm type for stopping criterion!");


  if(parameterSet.get("regularizationNorm", "Euclidean") == "Euclidean")
    regNormType_ = RegularizationNormType::Euclidean;
  else if(parameterSet.get("regularizationNorm", "Euclidean") == "H1")
    regNormType_ = RegularizationNormType::H1;
  else if(parameterSet.get("regularizationNorm", "Euclidean") == "H1semi")
    regNormType_ = RegularizationNormType::H1semi;
  else if(parameterSet.get("regularizationNorm", "Euclidean") == "L2")
    regNormType_ = RegularizationNormType::L2;
  else
    DUNE_THROW(Dune::Exception, "Unknown norm type for regularization!");

  instrumentedPath_ = parameterSet.get("instrumentedPath", "/tmp");

  // create 'intrumented' folder and 'mgHistory' subfolder if it does not exist.
  if (!(std::filesystem::exists(instrumentedPath_)))
  {
    std::filesystem::create_directory(instrumentedPath_);
    std::filesystem::create_directory(instrumentedPath_ + "/mgHistory");
  }

  setup(grid,
        assembler,
        x,
        dirichletNodes,
        parameterSet.get<double>("tolerance"),
        parameterSet.get<int>("maxProximalNewtonSteps"),
        parameterSet.get<double>("initialRegularization"),
        parameterSet.get<bool>("instrumented", 0));
}

template <class Basis, class TargetSpace, class Assembler>
void RiemannianProximalNewtonSolver<Basis,TargetSpace,Assembler>::
setup(const GridType& grid,
      const Assembler* assembler,
      const SolutionType& x,
      const Dune::BitSetVector<blocksize>& dirichletNodes,
      double tolerance,
      int maxProximalNewtonSteps,
      double initialRegularization,
      bool instrumented)
{
  grid_                     = &grid;
  assembler_                = assembler;
  x_                        = x;
  this->tolerance_          = tolerance;
  maxProximalNewtonSteps_   = maxProximalNewtonSteps;
  initialRegularization_    = initialRegularization;
  instrumented_             = instrumented;
  ignoreNodes_              = &dirichletNodes;

#if HAVE_MPI
  //////////////////////////////////////////////////////////////////
  //  Create global numbering for matrix and vector transfer
  //////////////////////////////////////////////////////////////////

  globalMapper_ = std::make_unique<GlobalMapper>(grid_->leafGridView());
  // Transfer all Dirichlet data to the master processor
  VectorCommunicator<GlobalMapper, typename GridType::LeafGridView::Communication, Dune::BitSetVector<blocksize> > vectorComm(*globalMapper_,
                                                                                                                              grid_->leafGridView().comm(),
                                                                                                                              0);
  auto globalDirichletNodes = new Dune::BitSetVector<blocksize>(vectorComm.reduceCopy(dirichletNodes));
#else
  auto globalDirichletNodes = new Dune::BitSetVector<blocksize>(dirichletNodes);
#endif

  // //////////////////////////////////////////////////////////////////////////////////////
  //   Assemble a Laplace matrix to create a norm that's equivalent to the H1-norm
  // //////////////////////////////////////////////////////////////////////////////////////

  const Basis& basis = assembler_->getBasis();
  Dune::Fufem::DuneFunctionsOperatorAssembler<Basis,Basis> operatorAssembler(basis, basis);

  Dune::Fufem::LaplaceAssembler laplaceStiffness;
  typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> > ScalarMatrixType;
  ScalarMatrixType localA;

  operatorAssembler.assembleBulk(Dune::Fufem::istlMatrixBackend(localA), laplaceStiffness);

#if HAVE_MPI
  LocalMapper localMapper = Dune::GFE::MapperFactory<Basis>::createLocalMapper(grid_->leafGridView());

  MatrixCommunicator<GlobalMapper,
      typename GridType::LeafGridView,
      typename GridType::LeafGridView,
      ScalarMatrixType,
      LocalMapper,
      LocalMapper> matrixComm(*globalMapper_, grid_->leafGridView(), localMapper, localMapper, 0);

  auto A = std::make_shared<ScalarMatrixType>(matrixComm.reduceAdd(localA));
#else
  auto A = std::make_shared<ScalarMatrixType>(localA);
#endif
  h1SemiNorm_ = std::make_shared<H1SemiNorm<CorrectionType> >(A);
  //////////////////////////////////////////////////////////////////
  //   Create the inner solver using a cholmod solver
  //////////////////////////////////////////////////////////////////

  innerSolver_ = std::make_shared<Dune::Solvers::CholmodSolver<MatrixType,CorrectionType> >();
  innerSolver_->setIgnore(*globalDirichletNodes);

  // //////////////////////////////////////////////////////////////////////////////////////
  //   Assemble a mass matrix to create a norm that's equivalent to the L2-norm
  //   This will be used to monitor the gradient
  // //////////////////////////////////////////////////////////////////////////////////////

  Dune::Fufem::MassAssembler massStiffness;
  ScalarMatrixType localMassMatrix;

  operatorAssembler.assembleBulk(Dune::Fufem::istlMatrixBackend(localMassMatrix), massStiffness);

#if HAVE_MPI
  auto massMatrix = std::make_shared<ScalarMatrixType>(matrixComm.reduceAdd(localMassMatrix));
#else
  auto massMatrix = std::make_shared<ScalarMatrixType>(localMassMatrix);
#endif
  l2Norm_ = std::make_shared<H1SemiNorm<CorrectionType> >(massMatrix);

  // Write all intermediate solutions, if requested
  if (instrumented_
      && dynamic_cast<IterativeSolver<CorrectionType>*>(innerSolver_.get()))
    dynamic_cast<IterativeSolver<CorrectionType>*>(innerSolver_.get())->historyBuffer_ = instrumentedPath_ + "/mgHistory";

  // ////////////////////////////////////////////////////////////
  //    Create Hessian matrix and its occupation structure
  // ////////////////////////////////////////////////////////////

  hessianMatrix_ = std::make_unique<MatrixType>();
  Dune::MatrixIndexSet indices(grid_->size(1), grid_->size(1));
  assembler_->getNeighborsPerVertex(indices);
  indices.exportIdx(*hessianMatrix_);
}


template <class Basis, class TargetSpace, class Assembler>
void RiemannianProximalNewtonSolver<Basis,TargetSpace,Assembler>::solve()
{
  int rank = grid_->comm().rank();

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
  //   Proximal Newton Solver
  // /////////////////////////////////////////////////////

  Dune::Timer energyTimer;
  double oldEnergy = assembler_->computeEnergy(x_);
  if (this->verbosity_ == Solver::FULL)
    std::cout << "Energy computation took " << energyTimer.elapsed() << " sec." << std::endl;


  oldEnergy = grid_->comm().sum(oldEnergy);

  bool recomputeGradientHessian = true;
  CorrectionType rhs, rhs_global;
  MatrixType stiffnessMatrix;

#if HAVE_MPI
  VectorCommunicator<GlobalMapper, typename GridType::LeafGridView::Communication, CorrectionType> vectorComm(*globalMapper_,
                                                                                                              grid_->leafGridView().comm(),
                                                                                                              0);
  LocalMapper localMapper = Dune::GFE::MapperFactory<Basis>::createLocalMapper(grid_->leafGridView());
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
  auto& i = statistics_.finalIteration;
  double totalAssemblyTime = 0.0;
  double totalSolverTime = 0.0;
  double regularization = initialRegularization_;
  for (i=0; i<maxProximalNewtonSteps_; i++) {

    Dune::Timer totalTimer;
    if (this->verbosity_ == Solver::FULL and rank==0) {
      std::cout << "----------------------------------------------------" << std::endl;
      std::cout << "      Proximal Newton Step Number: " << i
                << ",     regularization parameter: " << regularization
                << ",     energy: " << oldEnergy << std::endl;
      std::cout << "----------------------------------------------------" << std::endl;
    }

    CorrectionType corr(x_.size());
    corr = 0;

    if (recomputeGradientHessian) {
      Dune::Timer gradientTimer;
      assembler_->assembleGradientAndHessian(x_,
                                             rhs,
                                             *hessianMatrix_,
                                             i==0          // assemble occupation pattern only for the first call
                                             );
      std::cout << "Assembly took " << gradientTimer.elapsed() << " sec." << std::endl;

      rhs *= -1;              // The right hand side is the _negative_ gradient

      // Transfer vector data
#if HAVE_MPI
      rhs_global = vectorComm.reduceAdd(rhs);
#else
      rhs_global = rhs;
#endif
      CorrectionType gradient = rhs_global;
      for (size_t j=0; j<gradient.size(); j++)
        for (size_t k=0; k<gradient[j].size(); k++)
          if ((innerSolver_->ignore())[j][k])        // global Dirichlet nodes set
            gradient[j][k] = 0;

      if (this->verbosity_ == Solver::FULL and rank==0)
        std::cout << "Gradient norm: " << l2Norm_->operator()(gradient) << std::endl;

      if (this->verbosity_ == Solver::FULL and rank==0)
        std::cout << "Overall assembly took " << gradientTimer.elapsed() << " sec." << std::endl;
      totalAssemblyTime += gradientTimer.elapsed();

#if HAVE_MPI
      stiffnessMatrix = matrixComm.reduceAdd(*hessianMatrix_);
#else
      stiffnessMatrix = *hessianMatrix_;
#endif
      recomputeGradientHessian = false;

    }

    CorrectionType corr_global(rhs_global.size());
    corr_global = 0;
    bool solved = true;

    if (rank==0)
    {

      if (regNormType_ == RegularizationNormType::Euclidean)
      {
        if (this->verbosity_ == NumProc::FULL && rank==0)
          std::cout << "use Euclidean Norm regularization" << std::endl;
        for (std::size_t i=0; i<stiffnessMatrix.N(); i++)
          for(int j=0; j<blocksize; j++)
            stiffnessMatrix[i][i][j][j] += regularization/scaling_[j];
      }
      else if (regNormType_ == RegularizationNormType::H1semi)
      {
        if (this->verbosity_ == NumProc::FULL && rank==0)
          std::cout << "use H1-Semi Norm regularization" << std::endl;
        for (std::size_t i=0; i<stiffnessMatrix.N(); i++)
          for (auto && [v,index] : sparseRange(stiffnessMatrix[i]))
            for(int j=0; j<blocksize; j++)
              v[j][j] += (*(h1SemiNorm_->matrix_))[i][index][0][0] * regularization/scaling_[j];
      }
      else if (regNormType_ == RegularizationNormType::H1)
      {
        if (this->verbosity_ == NumProc::FULL && rank==0)
          std::cout << "use H1 Norm regularization" << std::endl;
        for (std::size_t i=0; i<stiffnessMatrix.N(); i++)
          for (auto && [v,index] : sparseRange(stiffnessMatrix[i]))
            for(int j=0; j<blocksize; j++)
              v[j][j] += ((*(h1SemiNorm_->matrix_))[i][index][0][0] + (*(l2Norm_->matrix_))[i][index][0][0]  ) * regularization/scaling_[j];
      }
      else if (regNormType_ == RegularizationNormType::L2)
      {
        if (this->verbosity_ == NumProc::FULL && rank==0)
          std::cout << "use L2 Norm regularization" << std::endl;
        for (std::size_t i=0; i<stiffnessMatrix.N(); i++)
          for (auto && [v,index] : sparseRange(stiffnessMatrix[i]))
            for(int j=0; j<blocksize; j++)
              v[j][j] += (*(l2Norm_->matrix_))[i][index][0][0] * regularization/scaling_[j];
      }
      else
        DUNE_THROW(Dune::Exception, "Unknown norm type for regularization!");

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
        solved = false;
        corr_global = 0;
      }
      std::cout << "Solving the quadratic problem took " << solutionTimer.elapsed() << " seconds." << std::endl;
      totalSolverTime += solutionTimer.elapsed();
    }

    // Distribute solution
    if (grid_->comm().size()>1 and rank==0)
      std::cout << "Transfer solution back to root process ..." << std::endl;

#if HAVE_MPI
    solved = grid_->comm().min(solved);
    if (solved) {
      corr = vectorComm.scatter(corr_global);
    } else  {
      corr_global = 0;
      corr = 0;
    }
#else
    corr = corr_global;
#endif

    double corrNorm;
    if (normType_ == ErrorNormType::infinity)
      corrNorm = corr.infinity_norm();
    else if (normType_ == ErrorNormType::H1semi)
      corrNorm = h1SemiNorm_->operator()(corr);
    else
      DUNE_THROW(Dune::Exception, "Unknown norm type for stopping criterion!");

    // Make norm of corr_global known on all processors
    double corrGlobalNorm = grid_->comm().max(corrNorm);

    if (std::isnan(corrGlobalNorm))
      solved = false;

    if (instrumented_) {
#if 0
      fprintf(fp, "Proximal newton step: %ld, regularization parameter: %g\n",
              i, regularization);

      // ///////////////////////////////////////////////////////////////
      //   Compute and measure progress against the exact solution
      //   for each proximal newton step
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
#endif
    }
    double energy = 0;
    double modelDecrease = 0;
    SolutionType newIterate = x_;
    if (i == maxProximalNewtonSteps_ - 1)
      std::cout << i+1 << " proximal newton steps were taken, the maximum was reached." << std::endl << "Total solver time: " << totalSolverTime << " sec., total assembly time: " << totalAssemblyTime << " sec." << std::endl;

    if (solved) {
      if (this->verbosity_ == NumProc::FULL && rank==0)
        switch (normType_)
        {
        case ErrorNormType::infinity :
          std::cout << "infinity norm of the correction: " << corrGlobalNorm << std::endl;
          break;

        case ErrorNormType::H1semi :
          std::cout << "H1-semi norm of the correction: " << corrGlobalNorm << std::endl;
          break;

        default :
          DUNE_THROW(Dune::Exception, "Unknown norm type for stopping criterion!");
        }

      if (corrGlobalNorm < this->tolerance_ && corrGlobalNorm < 1/regularization) {
        if (this->verbosity_ == NumProc::FULL and rank==0)
          std::cout << "CORRECTION IS SMALL ENOUGH" << std::endl;

        if (this->verbosity_ != NumProc::QUIET and rank==0)
          std::cout << i+1 << " proximal newton steps were taken" << std::endl << "Total solver time: " << totalSolverTime << " sec., total assembly time: " << totalAssemblyTime << " sec." << std::endl;
        break;
      }

      // ////////////////////////////////////////////////////
      //   Check whether proximal newton step can be accepted
      // ////////////////////////////////////////////////////

      for (size_t j=0; j<newIterate.size(); j++)
        newIterate[j] = TargetSpace::exp(newIterate[j], corr[j]);
      try {
        energy  = assembler_->computeEnergy(newIterate);
      } catch (Dune::Exception &e) {
        std::cerr << "Error while computing the energy of the new Iterate: " << e << std::endl;
        std::cerr << "Redoing proximal newton step with higher regularization parameter ..." << std::endl;
        solved = false;
      }
      solved = grid_->comm().min(solved);

      if (!solved) {
        energy = oldEnergy;
        newIterate = x_;
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

        if (this->verbosity_ == NumProc::FULL and rank==0) {
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

        if (energy >= oldEnergy &&
            (std::abs((oldEnergy-energy)/energy) < 1e-9 || relativeModelDecrease < 1e-9)) {
          if (this->verbosity_ == NumProc::FULL and rank==0)
            std::cout << "Suspecting rounding problems" << std::endl;

          if (this->verbosity_ != NumProc::QUIET and rank==0)
            std::cout << i+1 << " proximal newton steps were taken." << std::endl;

          x_ = newIterate;
          break;
        }
      }
    }

    // //////////////////////////////////////////////
    //   Check for acceptance of the step
    // //////////////////////////////////////////////
    if (solved && (oldEnergy-energy) / modelDecrease > 0.9) {
      // very successful iteration

      x_ = newIterate;
      regularization *= 0.5;

      // current energy becomes 'oldEnergy' for the next iteration
      oldEnergy = energy;

      recomputeGradientHessian = true;

    } else if (solved && ((oldEnergy-energy) / modelDecrease > 0.01
                          || std::abs(oldEnergy-energy) < 1e-12)) {
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

    // /////////////////////////////////////////////////////////////////////
    //   Write the iterate to disk for later convergence rate measurement
    // /////////////////////////////////////////////////////////////////////

    if (instrumented_) {

      char iFilename[200];
      sprintf(iFilename, (instrumentedPath_ + "/mgHistory/intermediatesolution_%04d").c_str(), i);


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

  statistics_.finalEnergy = oldEnergy;
}
