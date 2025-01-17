#ifndef TARGET_SPACE_RIEMANNIAN_TRUST_REGION_SOLVER_HH
#define TARGET_SPACE_RIEMANNIAN_TRUST_REGION_SOLVER_HH

#include <dune/solvers/common/numproc.hh>

#include <dune/gfe/symmetricmatrix.hh>


namespace Dune::GFE
{

  /** \brief Riemannian trust-region solver for geodesic finite-element problems
     \tparam TargetSpace The manifold that our functions take values in
   */
  template <class TargetSpace>
  class TargetSpaceRiemannianTRSolver
    : public NumProc
  {
    const static int blocksize = TargetSpace::TangentVector::dimension;
    const static int embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension;

    // Centralize the field type here
    typedef typename TargetSpace::ctype field_type;

    typedef SymmetricMatrix<field_type, embeddedBlocksize> MatrixType;
    typedef Dune::FieldVector<field_type, embeddedBlocksize>     CorrectionType;

  public:

    /** \brief Set up the solver using a monotone multigrid method as the inner solver */
    void setup(const AverageDistanceAssembler<TargetSpace>* assembler,
               const TargetSpace& x,
               double tolerance,
               int maxNewtonSteps);

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

    /** \brief Maximum number of Newton steps */
    size_t maxNewtonSteps_;

    /** \brief The assembler for the average-distance functional */
    const AverageDistanceAssembler<TargetSpace>* assembler_;

    /** \brief Specify a minimal number of iterations the Newton solver has to do
     *
     * This is needed when working with automatic differentiation.    While a very low
     * number of iterations may be enough to precisely compute the value of a
     * geodesic finite element function, a higher number may be needed to make an AD
     * system compute a derivative with sufficient precision.
     */
    size_t minNumberOfIterations_;

  };

}  // namespace Dune::GFE

#include "targetspacertrsolver.cc"

#endif
