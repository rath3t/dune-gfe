#ifndef DUNE_GFE_MIXED_RIEMANNIAN_TRUST_REGION_SOLVER_HH
#define DUNE_GFE_MIXED_RIEMANNIAN_TRUST_REGION_SOLVER_HH

#include <vector>

#include <dune/common/bitsetvector.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/multitypeblockmatrix.hh>

#include <dune/grid/utility/globalindexset.hh>

#include <dune/solvers/common/boxconstraint.hh>
#include <dune/solvers/norms/h1seminorm.hh>
#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/solvers/loopsolver.hh>
#include <dune/solvers/iterationsteps/mmgstep.hh>

#include <dune/gfe/mixedgfeassembler.hh>

/** \brief Riemannian trust-region solver for geodesic finite-element problems */
template <class GridType,
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
    typedef std::vector<TargetSpace0>                                                SolutionType0;
    typedef std::vector<TargetSpace1>                                                SolutionType1;

#if 0
#ifdef SECOND_ORDER
    typedef Dune::GlobalP2Mapper<typename GridType::LeafGridView> GUIndex;
#else
    typedef GlobalUniqueIndex<typename GridType::LeafGridView, gridDim> GUIndex;
#endif
#endif

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
               bool instrumented);

    void setScaling(const Dune::FieldVector<double,blocksize0+blocksize1>& scaling)
    {
      for (int i=0; i<3; i++)
      {
        std::get<0>(scaling_)[i] = scaling[i];
        std::get<1>(scaling_)[i] = scaling[i+3];
      }
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

    void setInitialIterate(const SolutionType0& x0,
                           const SolutionType1& x1)
    {
        x0_ = x0;
        x1_ = x1;
    }

    std::tuple<SolutionType0,SolutionType1> getSol() const
    {
      return std::make_tuple(x0_,x1_);
    }

protected:
#if 0
    std::unique_ptr<GUIndex> guIndex_;
#endif
    /** \brief The grid */
    const GridType* grid_;

    /** \brief The solution vectors */
    SolutionType0 x0_;
    SolutionType1 x1_;

    /** \brief The initial trust-region radius in the maximum-norm */
    double initialTrustRegionRadius_;

    /** \brief Trust-region norm scaling */
    std::tuple<Dune::FieldVector<double,3>, Dune::FieldVector<double,3> > scaling_;

    /** \brief Maximum number of trust-region steps */
    int maxTrustRegionSteps_;

    /** \brief Maximum number of multigrid iterations */
    int innerIterations_;

    /** \brief Error tolerance of the multigrid QP solver */
    double innerTolerance_;

    /** \brief Hessian matrix */
    std::unique_ptr<MatrixType> hessianMatrix_;

    /** \brief The assembler for the material law */
    const MixedGFEAssembler<Basis0, TargetSpace0, Basis1, TargetSpace1>* assembler_;

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
};

#include "mixedriemanniantrsolver.cc"

#endif
