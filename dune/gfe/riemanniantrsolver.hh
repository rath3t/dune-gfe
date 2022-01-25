#ifndef RIEMANNIAN_TRUST_REGION_SOLVER_HH
#define RIEMANNIAN_TRUST_REGION_SOLVER_HH

#include <vector>

#include <dune/common/bitsetvector.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>

#include <dune/solvers/common/boxconstraint.hh>
#include <dune/solvers/norms/h1seminorm.hh>
#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/solvers/loopsolver.hh>

#include <dune/gfe/periodic1dpq1nodalbasis.hh>

#include "geodesicfeassembler.hh"
#include <dune/grid/utility/globalindexset.hh>
#include <dune/gfe/parallel/globalmapper.hh>
#include <dune/gfe/parallel/globalp1mapper.hh>
#include <dune/gfe/parallel/globalp2mapper.hh>
#include <dune/gfe/parallel/p2mapper.hh>

/** \brief Assign GlobalMapper and LocalMapper types to a dune-fufem FunctionSpaceBasis */
template <typename Basis>
struct MapperFactory
{};

/** \brief Specialization for LagrangeBasis<1> */
template <typename GridView>
struct MapperFactory<Dune::Functions::LagrangeBasis<GridView,1> >
{
    typedef Dune::GlobalP1Mapper<Dune::Functions::LagrangeBasis<GridView,1>> GlobalMapper;
    typedef Dune::MultipleCodimMultipleGeomTypeMapper<GridView> LocalMapper;
    static LocalMapper createLocalMapper(const GridView& gridView)
    {
      return LocalMapper(gridView, Dune::mcmgVertexLayout());
    }
};

// This case is not going to actually work, but I need the specialization to make
// the sequential code compile.
template <typename GridView>
struct MapperFactory<Dune::Functions::Periodic1DPQ1NodalBasis<GridView> >
{
    typedef Dune::GlobalP1Mapper<Dune::Functions::Periodic1DPQ1NodalBasis<GridView>> GlobalMapper;
    typedef Dune::MultipleCodimMultipleGeomTypeMapper<GridView> LocalMapper;
};

template <typename GridView>
struct MapperFactory<Dune::Functions::LagrangeBasis<GridView,2> >
{
    typedef Dune::GlobalP2Mapper<Dune::Functions::LagrangeBasis<GridView,2>> GlobalMapper;
    typedef P2BasisMapper<GridView> LocalMapper;
    static LocalMapper createLocalMapper(const GridView& gridView)
    {
      return LocalMapper(gridView);
    }
};

/** \brief Specialization for LagrangeBasis<3> */
template <typename GridView>
struct MapperFactory<Dune::Functions::LagrangeBasis<GridView,3> >
{
    // Error: we don't currently have a global P3 mapper
};

/** \brief Riemannian trust-region solver for geodesic finite-element problems */
template <class Basis, class TargetSpace, class Assembler = GeodesicFEAssembler<Basis,TargetSpace>>
class RiemannianTrustRegionSolver
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

    /** \brief Records information about the last run of the RiemannianTrustRegionSolver
     *
     * This is used primarily for unit testing.
     */
    struct Statistics
    {
      std::size_t finalIteration;

      field_type finalEnergy;
    };

public:

    RiemannianTrustRegionSolver()
        : IterativeSolver<std::vector<TargetSpace>, Dune::BitSetVector<blocksize> >(0,100,NumProc::FULL),
          hessianMatrix_(nullptr), h1SemiNorm_(NULL)
    {
      std::fill(scaling_.begin(), scaling_.end(), 1.0);
    }

    /** \brief Set up the solver using a monotone multigrid method as the inner solver */
    void setup(const GridType& grid,
               const Assembler* assembler,
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

    void setScaling(const Dune::FieldVector<double,blocksize>& scaling)
    {
      scaling_ = scaling;
    }

    void setIgnoreNodes(const Dune::BitSetVector<blocksize>& ignoreNodes)
    {
        ignoreNodes_ = &ignoreNodes;
        std::shared_ptr<LoopSolver<CorrectionType> > loopSolver = std::dynamic_pointer_cast<LoopSolver<CorrectionType> >(innerSolver_);
        assert(loopSolver);
        loopSolver->iterationStep_->ignoreNodes_ = ignoreNodes_;
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

    /** \brief The initial trust-region radius in the maximum-norm */
    double initialTrustRegionRadius_;

    /** \brief Trust-region norm scaling */
    Dune::FieldVector<double,blocksize> scaling_;

    /** \brief Maximum number of trust-region steps */
    std::size_t maxTrustRegionSteps_;

    /** \brief Maximum number of multigrid iterations */
    int innerIterations_;

    /** \brief Error tolerance of the multigrid QP solver */
    double innerTolerance_;

    /** \brief Hessian matrix */
    std::unique_ptr<MatrixType> hessianMatrix_;

    /** \brief The assembler for the material law */
    const Assembler* assembler_;

    /** \brief The solver for the quadratic inner problems */
    std::shared_ptr<Solver> innerSolver_;

    /** \brief Contains 'true' everywhere -- the trust-region is bounded */
    Dune::BitSetVector<blocksize> hasObstacle_;

    /** \brief The Dirichlet nodes */
    const Dune::BitSetVector<blocksize>* ignoreNodes_;

    /** \brief The norm used to measure multigrid convergence */
    std::shared_ptr<H1SemiNorm<CorrectionType> > h1SemiNorm_;

    /** \brief An L2-norm, really.  The H1SemiNorm class is badly named */
    std::shared_ptr<H1SemiNorm<CorrectionType> > l2Norm_;

    /** \brief If set to true we log convergence speed and other stuff */
    bool instrumented_;

    /** \brief Store information about solver runs for unit testing */
    Statistics statistics_;

};

#include "riemanniantrsolver.cc"

#endif
