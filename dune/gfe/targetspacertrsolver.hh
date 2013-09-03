#ifndef TARGET_SPACE_RIEMANNIAN_TRUST_REGION_SOLVER_HH
#define TARGET_SPACE_RIEMANNIAN_TRUST_REGION_SOLVER_HH

#include <dune/istl/matrix.hh>

#include <dune/solvers/common/boxconstraint.hh>
#include <dune/solvers/solvers/loopsolver.hh>
#include <dune/solvers/iterationsteps/trustregiongsstep.hh>
#include <dune/solvers/norms/energynorm.hh>

/** \brief Riemannian trust-region solver for geodesic finite-element problems
 \tparam TargetSpace The manifold that our functions take values in
 */
template <class TargetSpace>
class TargetSpaceRiemannianTRSolver
    : public NumProc
{
    const static int blocksize = TargetSpace::TangentVector::dimension;

    // Centralize the field type here
    typedef typename TargetSpace::ctype field_type;

    // Some types that I need
    // The types have the dynamic outer type because the dune-solvers solvers expect
    // this sort of type.
    typedef Dune::Matrix<Dune::FieldMatrix<field_type, blocksize, blocksize> > MatrixType;
    typedef Dune::BlockVector<Dune::FieldVector<field_type, blocksize> >       CorrectionType;

public:

    TargetSpaceRiemannianTRSolver()
    //        : IterativeSolver<std::vector<TargetSpace>, Dune::BitSetVector<blocksize> >(0,100,NumProc::FULL)
    {}

    /** \brief Set up the solver using a monotone multigrid method as the inner solver */
    void setup(const AverageDistanceAssembler<TargetSpace>* assembler,
               const TargetSpace& x,
               double tolerance,
               int maxTrustRegionSteps,
               double initialTrustRegionRadius,
               int innerIterations,
               double innerTolerance);

    void solve();

    void setInitialSolution(const TargetSpace& x) {
        x_ = x;
    }

    TargetSpace getSol() const {return x_;}

protected:

    /** \brief The solution vector */
    TargetSpace x_;

    /** \brief Tolerance of the solver */
    double tolerance_;

    /** \brief The initial trust-region radius in the maximum-norm */
    double initialTrustRegionRadius_;

    /** \brief Maximum number of trust-region steps */
    int maxTrustRegionSteps_;

    /** \brief Maximum number of multigrid iterations */
    int innerIterations_;

    /** \brief Error tolerance of the multigrid QP solver */
    double innerTolerance_;

    /** \brief The assembler for the average-distance functional */
    const AverageDistanceAssembler<TargetSpace>* assembler_;

    /** \brief The solver for the quadratic inner problems */
    std::auto_ptr< ::LoopSolver<CorrectionType> > innerSolver_;

    /** \brief The iteration step for the quadratic inner problems */
    std::auto_ptr<TrustRegionGSStep<MatrixType, CorrectionType> > innerSolverStep_;

    /** \brief Norm for the quadratic inner problems */
    std::auto_ptr<EnergyNorm<MatrixType, CorrectionType> > energyNorm_;

    /** \brief Specify a minimal number of iterations the trust-region solver has to do
     *
     * This is needed when working with automatic differentiation.    While a very low
     * number of iterations may be enough to precisely compute the value of a
     * geodesic finite element function, a higher number may be needed to make an AD
     * system compute a derivative with sufficient precision.
     */
    size_t minNumberOfIterations_;

};

#include "targetspacertrsolver.cc"

#endif
