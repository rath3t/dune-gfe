#ifndef DUNE_GFE_MIXED_RIEMANNIAN_TRUST_REGION_SOLVER_HH
#define DUNE_GFE_MIXED_RIEMANNIAN_TRUST_REGION_SOLVER_HH

#include <vector>

#include <dune/common/bitsetvector.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/multitypeblockmatrix.hh>

#include <dune/grid/utility/globalindexset.hh>

#include <dune/common/tuplevector.hh>

#include <dune/solvers/common/boxconstraint.hh>
#include <dune/solvers/norms/h1seminorm.hh>
#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/solvers/loopsolver.hh>
#include <dune/solvers/iterationsteps/mmgstep.hh>

#include <dune/gfe/assemblers/mixedgfeassembler.hh>

/** \brief Riemannian trust-region solver for geodesic finite-element problems */
template <class GridType,
          class Basis,
          class Basis0, class TargetSpace0,
          class Basis1, class TargetSpace1>
class MixedRiemannianTrustRegionSolver
    : public NumProc
{
    const static int blocksize0 = TargetSpace0::TangentVector::dimension;
    const static int blocksize1 = TargetSpace1::TangentVector::dimension;

    const static int gridDim = GridType::dimension;

    // Centralize the field type here
    typedef double field_type;

    // Some types that I need
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<field_type, blocksize0, blocksize0> > MatrixType00;
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<field_type, blocksize0, blocksize1> > MatrixType01;
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<field_type, blocksize1, blocksize0> > MatrixType10;
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<field_type, blocksize1, blocksize1> > MatrixType11;
    typedef Dune::MultiTypeBlockMatrix<Dune::MultiTypeBlockVector<MatrixType00,MatrixType01>,
                                       Dune::MultiTypeBlockVector<MatrixType10,MatrixType11> > MatrixType;
    typedef Dune::BlockVector<Dune::FieldVector<field_type, blocksize0> >             CorrectionType0;
    typedef Dune::BlockVector<Dune::FieldVector<field_type, blocksize1> >             CorrectionType1;
    typedef Dune::MultiTypeBlockVector<CorrectionType0, CorrectionType1> CorrectionType;
    typedef Dune::TupleVector<std::vector<TargetSpace0>, std::vector<TargetSpace1> > SolutionType;

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

    MixedRiemannianTrustRegionSolver()
        : NumProc(NumProc::FULL),

          h1SemiNorm0_(nullptr), h1SemiNorm1_(nullptr)
    {
        std::fill(std::get<0>(scaling_).begin(), std::get<0>(scaling_).end(), 1.0);
        std::fill(std::get<1>(scaling_).begin(), std::get<1>(scaling_).end(), 1.0);
    }

    /** \brief Set up the solver using a monotone multigrid method as the inner solver */
    void setup(const GridType& grid,
               const MixedGFEAssembler<Basis, TargetSpace0, TargetSpace1>* assembler,
               const Basis0& basis0,
               const Basis1& basis1,
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
               bool instrumented);

    void setScaling(const Dune::FieldVector<double,blocksize0+blocksize1>& scaling)
    {
      for (int i=0; i<blocksize0; i++)
        std::get<0>(scaling_)[i] = scaling[i];

      for (int i=0; i<blocksize1; i++)
        std::get<1>(scaling_)[i] = scaling[i+blocksize0];
    }

#if 0
    void setIgnoreNodes(const Dune::BitSetVector<blocksize0>& ignoreNodes)
    {
        ignoreNodes_ = &ignoreNodes;
        Dune::shared_ptr<LoopSolver<CorrectionType> > loopSolver = std::dynamic_pointer_cast<LoopSolver<CorrectionType> >(innerSolver_);
        assert(loopSolver);
        loopSolver->iterationStep_->ignoreNodes_ = ignoreNodes_;
    }
#endif
    void solve();

    void setInitialIterate(const SolutionType& x)
    {
        x_ = x;
    }

    SolutionType getSol() const
    {
      return x_;
    }

    const Statistics& getStatistics() const {return statistics_;}

protected:
#if 0
    std::unique_ptr<GUIndex> guIndex_;
#endif
    /** \brief The grid */
    const GridType* grid_;

    /** \brief The solution vectors */
    SolutionType x_;

    /** \brief The initial trust-region radius in the maximum-norm */
    double initialTrustRegionRadius_;

    /** \brief Trust-region norm scaling */
    std::tuple<Dune::FieldVector<double,blocksize0>, Dune::FieldVector<double,blocksize1> > scaling_;

    /** \brief Maximum number of trust-region steps */
    int maxTrustRegionSteps_;

    /** \brief Maximum number of multigrid iterations */
    int innerIterations_;

    /** \brief Error tolerance of the multigrid QP solver */
    double innerTolerance_;

    /** \brief Hessian matrix */
    std::unique_ptr<MatrixType> hessianMatrix_;

    /** \brief The assembler for the material law */
    const MixedGFEAssembler<Basis, TargetSpace0, TargetSpace1>* assembler_;

    /** \brief TEMPORARY: The two separate matrices */
    std::unique_ptr<Basis0> basis0_;
    std::unique_ptr<Basis1> basis1_;

    /** \brief The solver for the quadratic inner problems */
    std::shared_ptr<Solver> innerSolver_;

    MonotoneMGStep<MatrixType00, CorrectionType0>* mmgStep0;
    MonotoneMGStep<MatrixType11, CorrectionType1>* mmgStep1;

    double tolerance_;

    /** \brief Contains 'true' everywhere -- the trust-region is bounded */
    Dune::BitSetVector<blocksize0> hasObstacle0_;
    Dune::BitSetVector<blocksize1> hasObstacle1_;

    /** \brief The Dirichlet nodes */
    const Dune::BitSetVector<blocksize0>* ignoreNodes0_;
    const Dune::BitSetVector<blocksize1>* ignoreNodes1_;

    /** \brief The norm used to measure multigrid convergence */
    H1SemiNorm<CorrectionType0>* h1SemiNorm0_;
    H1SemiNorm<CorrectionType1>* h1SemiNorm1_;

    /** \brief Store information about solver runs for testing */
    Statistics statistics_;
};

#include "mixedriemanniantrsolver.cc"

#endif
