#include <dune/common/bitsetvector.hh>
#include <dune/common/timer.hh>

#include <dune/istl/io.hh>

#include <dune/grid/common/mcmgmapper.hh>

#include <dune/fufem/assemblers/dunefunctionsoperatorassembler.hh>
#include <dune/fufem/assemblers/localassemblers/laplaceassembler.hh>
#include <dune/fufem/assemblers/localassemblers/massassembler.hh>
#include <dune/fufem/assemblers/basisinterpolationmatrixassembler.hh>

#if DUNE_VERSION_GTE(DUNE_SOLVERS, 2, 8)
// Using a cholmod solver as the inner solver, available only since 2.8
#include <dune/solvers/solvers/cholmodsolver.hh>
#else
// Using a umfpack solver as the inner solver
#include <dune/solvers/solvers/umfpacksolver.hh>
#endif

#include <dune/solvers/norms/twonorm.hh>
#include <dune/solvers/norms/h1seminorm.hh>

#if HAVE_MPI
#include <dune/gfe/parallel/matrixcommunicator.hh>
#include <dune/gfe/parallel/vectorcommunicator.hh>
#endif

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

    //////////////////////////////////////////////////////////////////
    //  Create global numbering for matrix and vector transfer
    //////////////////////////////////////////////////////////////////

#if HAVE_MPI
    globalMapper_ = std::make_unique<GlobalMapper>(grid_->leafGridView());
#endif
    
#if HAVE_MPI
    // Transfer all Dirichlet data to the master processor
    VectorCommunicator<GlobalMapper, typename GridType::LeafGridView::CollectiveCommunication, Dune::BitSetVector<blocksize> > vectorComm(*globalMapper_,
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
    LocalMapper localMapper = MapperFactory<Basis>::createLocalMapper(grid_->leafGridView());

    MatrixCommunicator<GlobalMapper,
                       typename GridType::LeafGridView,
                       typename GridType::LeafGridView,
                       ScalarMatrixType,
                       LocalMapper,
                       LocalMapper> matrixComm(*globalMapper_, grid_->leafGridView(), localMapper, localMapper, 0);
    if (instrumented_) {
        auto A = std::make_shared<ScalarMatrixType>(matrixComm.reduceAdd(localA));
#else
    if (instrumented_) {
        auto A = std::make_shared<ScalarMatrixType>(localA);
#endif
        h1SemiNorm_ = std::make_shared<H1SemiNorm<CorrectionType> >(A);
    }
    //////////////////////////////////////////////////////////////////
    //   Create the inner solver using a cholmod solver
    //////////////////////////////////////////////////////////////////
#if DUNE_VERSION_GTE(DUNE_SOLVERS, 2, 8)
    innerSolver_ = std::make_shared<Dune::Solvers::CholmodSolver<MatrixType,CorrectionType> >();
#else
    std::cout << "using umfpacksolver" << std::endl;
    innerSolver_ = std::make_shared<Dune::Solvers::UMFPackSolver<MatrixType,CorrectionType> >();
#endif
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
        dynamic_cast<IterativeSolver<CorrectionType>*>(innerSolver_.get())->historyBuffer_ = "tmp/mgHistory";

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
    VectorCommunicator<GlobalMapper, typename GridType::LeafGridView::CollectiveCommunication, CorrectionType> vectorComm(*globalMapper_,
                                                                                                                     grid_->leafGridView().comm(),
                                                                                                                     0);
    LocalMapper localMapper = MapperFactory<Basis>::createLocalMapper(grid_->leafGridView());
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
                                                   i==0    // assemble occupation pattern only for the first call
                                                   );
            std::cout << "Assembly took " << gradientTimer.elapsed() << " sec." << std::endl;

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
                if ((innerSolver_->ignore())[j][k])  // global Dirichlet nodes set
                    gradient[j][k] = 0; 

            if (this->verbosity_ == Solver::FULL and rank==0)
              std::cout << "Gradient norm: " << l2Norm_->operator()(gradient) << std::endl;

            if (this->verbosity_ == Solver::FULL and rank==0)
              std::cout << "Overall assembly took " << gradientTimer.elapsed() << " sec." << std::endl;
            totalAssemblyTime += gradientTimer.elapsed();

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
        bool solved = true;

        if (rank==0)
        {
            // Add the regularization - Identity Matrix for now
            for (std::size_t i=0; i<stiffnessMatrix.N(); i++)
              for(int j=0; j<blocksize; j++)
                stiffnessMatrix[i][i][j][j] += regularization;

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

        // Make infinity norm of corr_global known on all processors
        double corrNorm = corr.infinity_norm();
        double corrGlobalInfinityNorm = grid_->comm().max(corrNorm);

        if (std::isnan(corrGlobalInfinityNorm))
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
                std::cout << "Infinity norm of the correction: " << corrGlobalInfinityNorm << std::endl;

            if (corrGlobalInfinityNorm < this->tolerance_ && corrGlobalInfinityNorm < 1/regularization) {
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

                // compute the model decrease
                // It is $ m(x) - m(x+s) = -<g,s> - 0.5 <s, Hs>
                // Note that rhs = -g
                CorrectionType tmp(corr.size());
                tmp = 0;
                hessianMatrix_->umv(corr, tmp);
                modelDecrease = (rhs*corr) - 0.5 * (corr*tmp);
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

            char iFilename[100];
            sprintf(iFilename, "tmp/intermediateSolution_%04ld", i);

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
