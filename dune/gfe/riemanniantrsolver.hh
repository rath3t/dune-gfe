#ifndef RIEMANNIAN_TRUST_REGION_SOLVER_HH
#define RIEMANNIAN_TRUST_REGION_SOLVER_HH

#include <vector>

#include <dune/common/bitsetvector.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>

#include <dune/solvers/common/boxconstraint.hh>
#include <dune/solvers/norms/h1seminorm.hh>
#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/solvers/loopsolver.hh>

#include <dune/fufem/functionspacebases/p1nodalbasis.hh>
#include <dune/fufem/functionspacebases/p2nodalbasis.hh>
#include <dune/fufem/functionspacebases/p3nodalbasis.hh>

#include "geodesicfeassembler.hh"

/** \brief Riemannian trust-region solver for geodesic finite-element problems */
template <class GridType, class TargetSpace>
class RiemannianTrustRegionSolver
    : public IterativeSolver<std::vector<TargetSpace>,
                             Dune::BitSetVector<TargetSpace::TangentVector::dimension> >
{
    const static int blocksize = TargetSpace::TangentVector::dimension;

    const static int gridDim = GridType::dimension;

    // Centralize the field type here
    typedef double field_type;

    // Some types that I need
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<field_type, blocksize, blocksize> > MatrixType;
    typedef Dune::BlockVector<Dune::FieldVector<field_type, blocksize> >           CorrectionType;
    typedef std::vector<TargetSpace>                                               SolutionType;

#ifdef THIRD_ORDER
    typedef P3NodalBasis<typename GridType::LeafGridView,double> BasisType;
#elif defined SECOND_ORDER
    typedef P2NodalBasis<typename GridType::LeafGridView,double> BasisType;
#else
    typedef P1NodalBasis<typename GridType::LeafGridView,double> BasisType;
#endif

public:

    RiemannianTrustRegionSolver()
        : IterativeSolver<std::vector<TargetSpace>, Dune::BitSetVector<blocksize> >(0,100,NumProc::FULL),
          hessianMatrix_(std::auto_ptr<MatrixType>(NULL)), h1SemiNorm_(NULL)
    {}

    /** \brief Set up the solver using a monotone multigrid method as the inner solver */
    void setup(const GridType& grid,
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
               bool instrumented);

    void setIgnoreNodes(const Dune::BitSetVector<blocksize>& ignoreNodes)
    {
        ignoreNodes_ = &ignoreNodes;
        Dune::shared_ptr<LoopSolver<CorrectionType> > loopSolver = std::dynamic_pointer_cast<LoopSolver<CorrectionType> >(innerSolver_);
        assert(loopSolver);
        loopSolver->iterationStep_->ignoreNodes_ = ignoreNodes_;
    }

    void solve();

    void setInitialSolution(const SolutionType& x) DUNE_DEPRECATED {
        x_ = x;
    }

    void setInitialIterate(const SolutionType& x) {
        x_ = x;
    }

    SolutionType getSol() const {return x_;}

protected:


    /** \brief The grid */
    const GridType* grid_;

    /** \brief The solution vector */
    SolutionType x_;

    /** \brief The initial trust-region radius in the maximum-norm */
    double initialTrustRegionRadius_;

    /** \brief Maximum number of trust-region steps */
    int maxTrustRegionSteps_;

    /** \brief Maximum number of multigrid iterations */
    int innerIterations_;

    /** \brief Error tolerance of the multigrid QP solver */
    double innerTolerance_;

    /** \brief Hessian matrix */
    std::auto_ptr<MatrixType> hessianMatrix_;

    /** \brief The assembler for the material law */
    const GeodesicFEAssembler<BasisType, TargetSpace>* assembler_;

    /** \brief The solver for the quadratic inner problems */
    std::shared_ptr<Solver> innerSolver_;

    /** \brief Dummy fields containing 'true' everywhere.  The multigrid step
        expects them :-( */
    std::vector<Dune::BitSetVector<1> > hasObstacle_;

    /** \brief The Dirichlet nodes */
    const Dune::BitSetVector<blocksize>* ignoreNodes_;

    /** \brief The norm used to measure multigrid convergence */
    H1SemiNorm<CorrectionType>* h1SemiNorm_;

    /** \brief If set to true we log convergence speed and other stuff */
    bool instrumented_;

};

#include "riemanniantrsolver.cc"

#endif
