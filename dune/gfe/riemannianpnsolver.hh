#ifndef RIEMANNIAN_PROXIMAL_NEWTON_SOLVER_HH
#define RIEMANNIAN_PROXIMAL_NEWTON_SOLVER_HH

#include <vector>

#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>

#include <dune/solvers/common/boxconstraint.hh>
#include <dune/solvers/norms/h1seminorm.hh>
#include <dune/solvers/solvers/iterativesolver.hh>
#if DUNE_VERSION_GTE(DUNE_SOLVERS, 2, 8)
#include <dune/solvers/solvers/cholmodsolver.hh>
#else
#include <dune/solvers/solvers/umfpacksolver.hh>
#endif

#include "riemanniantrsolver.hh"
#include <dune/grid/utility/globalindexset.hh>
#include <dune/gfe/parallel/globalmapper.hh>
#include <dune/gfe/parallel/globalp1mapper.hh>
#include <dune/gfe/parallel/globalp2mapper.hh>
#include <dune/gfe/parallel/p2mapper.hh>

/** \brief Riemannian proximal-newton solver for geodesic finite-element problems */
template <class Basis, class TargetSpace, class Assembler = GeodesicFEAssembler<Basis,TargetSpace>>
class RiemannianProximalNewtonSolver
    : public IterativeSolver<std::vector<TargetSpace>,
                             Dune::BitSetVector<TargetSpace::TangentVector::dimension> >
{
    typedef typename Basis::GridView::Grid GridType;

    const static int blocksize = TargetSpace::TangentVector::dimension;

    const static int gridDim = GridType::dimension;

    // Centralize the field type here
    typedef double field_type;

    // Some types that I need
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<field_type, blocksize, blocksize> > MatrixType;
    typedef Dune::BlockVector<Dune::FieldVector<field_type, blocksize> >           CorrectionType;
    typedef std::vector<TargetSpace>                                               SolutionType;

#if HAVE_MPI
    typedef typename MapperFactory<Basis>::GlobalMapper GlobalMapper;
    typedef typename MapperFactory<Basis>::LocalMapper LocalMapper;
#endif

    /** \brief Records information about the last run of the RiemannianProximalNewtonSolver
     *
     * This is used primarily for unit testing.
     */
    struct Statistics
    {
      std::size_t finalIteration;

      field_type finalEnergy;
    };

public:

    RiemannianProximalNewtonSolver()
        : IterativeSolver<std::vector<TargetSpace>, Dune::BitSetVector<blocksize> >(0,100,NumProc::FULL),
          hessianMatrix_(nullptr), h1SemiNorm_(NULL)
    {
        std::fill(scaling_.begin(), scaling_.end(), 1.0);
    }

    /** \brief Set up the solver using a choldmod or umfpack solver as the inner solver */
    void setup(const GridType& grid,
               const Assembler* assembler,
               const SolutionType& x,
               const Dune::BitSetVector<blocksize>& dirichletNodes,
               double tolerance,
               int maxProximalNewtonSteps,
               double initialRegularization,
               bool instrumented);

    /** \brief Set up the solver using a monotone multigrid method as the inner solver */
    void setup(const GridType& grid,
               const Assembler* assembler,
               const SolutionType& x,
               const Dune::BitSetVector<blocksize>& dirichletNodes,
               const Dune::ParameterTree& parameterSet);

    void setScaling(const Dune::FieldVector<double,blocksize>& scaling)
    {
      scaling_ = scaling;
    }

    void setIgnoreNodes(const Dune::BitSetVector<blocksize>& ignoreNodes)
    {
        ignoreNodes_ = &ignoreNodes;
        innerSolver_->ignoreNodes_ = ignoreNodes_;
    }

    void solve();

    [[deprecated]] 
    void setInitialSolution(const SolutionType& x) {
        x_ = x;
    }

    void setInitialIterate(const SolutionType& x) {
        x_ = x;
    }

    SolutionType getSol() const {return x_;}

    const Statistics& getStatistics() const {return statistics_;}

protected:

#if HAVE_MPI
    std::unique_ptr<GlobalMapper> globalMapper_;
#endif

    /** \brief The grid */
    const GridType* grid_;

    /** \brief The solution vector */
    SolutionType x_;

    /** \brief The initial regularization parameter for the proximal newton step */
    double initialRegularization_;
    double tolerance_;

    /** \brief Regularization scaling */
    Dune::FieldVector<double,blocksize> scaling_;

    /** \brief Maximum number of proximal-newton steps */
    std::size_t maxProximalNewtonSteps_;

    /** \brief Hessian matrix */
    std::unique_ptr<MatrixType> hessianMatrix_;

    /** \brief The assembler for the material law */
    const Assembler* assembler_;

    /** \brief The solver for the quadratic inner problems */
#if DUNE_VERSION_GTE(DUNE_SOLVERS, 2, 8)
    std::shared_ptr<typename Dune::Solvers::CholmodSolver<MatrixType,CorrectionType>> innerSolver_;
#else
    std::shared_ptr<typename Dune::Solvers::UMFPackSolver<MatrixType,CorrectionType>> innerSolver_;
#endif
    /** \brief The Dirichlet nodes */
    const Dune::BitSetVector<blocksize>* ignoreNodes_;

    /** \brief The norm used to measure convergence for statistics*/
    std::shared_ptr<H1SemiNorm<CorrectionType> > h1SemiNorm_;

    /** \brief An L2-norm, really. The H1SemiNorm class is badly named */
    std::shared_ptr<H1SemiNorm<CorrectionType> > l2Norm_;

    /** \brief If set to true we log convergence speed and other stuff */
    bool instrumented_;

    /** \brief Norm type used for stopping criterion (default is infinity norm) */
    enum class ErrorNormType {infinity, H1semi} normType_ = ErrorNormType::infinity;

    /** \brief Store information about solver runs for unit testing */
    Statistics statistics_;

};

#include "riemannianpnsolver.cc"

#endif
