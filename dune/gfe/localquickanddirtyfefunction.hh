#ifndef DUNE_GFE_LOCALQUICKANDDIRTYFEFUNCTION_HH
#define DUNE_GFE_LOCALQUICKANDDIRTYFEFUNCTION_HH

#include <vector>

#include <dune/common/fvector.hh>

#include <dune/geometry/type.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/linearalgebra.hh>

namespace Dune {

  namespace GFE {

    /** \brief Interpolate on a manifold, as fast as we can
     *
     * This class implements interpolation of values on a manifold in a 'quick-and-dirty' way.
     * No particular interpolation rule is used consistenly for all manifolds.  Rather, it is
     * used whatever is quickest and 'reasonable' for any given space.  The reason to have this
     * is to provide initial iterates for the iterative solvers used to solve the local
     * geodesic-FE minimization problems.
     *
     * \note In particular, we currently do not implement the derivative of the interpolation,
     *   because it is not needed for producing initial iterates!
     *
     * \tparam dim Dimension of the reference element
     * \tparam ctype Type used for coordinates on the reference element
     * \tparam LocalFiniteElement A Lagrangian finite element whose shape functions define the interpolation weights
     * \tparam TargetSpace The manifold that the function takes its values in
     */
    template <int dim, class ctype, class LocalFiniteElement, class TS>
    class LocalQuickAndDirtyFEFunction
    {
    public:
      using TargetSpace=TS;
    private:
      typedef typename TargetSpace::ctype RT;

      typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;
      static const int embeddedDim = EmbeddedTangentVector::dimension;

      static const int spaceDim = TargetSpace::TangentVector::dimension;

    public:

      /** \brief The type used for derivatives */
      typedef Dune::FieldMatrix<RT, embeddedDim, dim> DerivativeType;

      /** \brief Constructor
       * \param localFiniteElement A Lagrangian finite element that provides the interpolation points
       * \param coefficients Values of the function at the Lagrange points
       */
      LocalQuickAndDirtyFEFunction(const LocalFiniteElement& localFiniteElement,
                               const std::vector<TargetSpace>& coefficients)
      : localFiniteElement_(localFiniteElement),
      coefficients_(coefficients)
      {
        assert(localFiniteElement_.localBasis().size() == coefficients_.size());
      }

      /** \brief The number of Lagrange points */
      unsigned int size() const
      {
        return localFiniteElement_.localBasis().size();
      }

      /** \brief The type of the reference element */
      Dune::GeometryType type() const
      {
        return localFiniteElement_.type();
      }

      /** \brief Evaluate the function */
      TargetSpace evaluate(const Dune::FieldVector<ctype, dim>& local) const;

      /** \brief Get the i'th base coefficient. */
      TargetSpace coefficient(int i) const
      {
        return coefficients_[i];
      }
    private:

      /** \brief The scalar local finite element, which provides the weighting factors
       *        \todo We really only need the local basis
       */
      const LocalFiniteElement& localFiniteElement_;

      /** \brief The coefficient vector */
      std::vector<TargetSpace> coefficients_;

    };

    template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
    TargetSpace LocalQuickAndDirtyFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::
    evaluate(const Dune::FieldVector<ctype, dim>& local) const
    {
      // Evaluate the weighting factors---these are the Lagrangian shape function values at 'local'
      std::vector<Dune::FieldVector<ctype,1> > w;
      localFiniteElement_.localBasis().evaluateFunction(local,w);

      typename TargetSpace::CoordinateType c(0);
      for (size_t i=0; i<coefficients_.size(); i++)
        c.axpy(w[i][0], coefficients_[i].globalCoordinates());

      return TargetSpace(c);
    }
  }   // namespace GFE

}   // namespace Dune
#endif
