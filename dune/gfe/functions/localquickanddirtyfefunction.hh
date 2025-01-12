#ifndef DUNE_GFE_FUNCTIONS_LOCALQUICKANDDIRTYFEFUNCTION_HH
#define DUNE_GFE_FUNCTIONS_LOCALQUICKANDDIRTYFEFUNCTION_HH

#include <vector>

#include <dune/common/fvector.hh>

#include <dune/geometry/type.hh>

#include <dune/gfe/linearalgebra.hh>
#include <dune/gfe/spaces/rotation.hh>

namespace Dune {

  namespace GFE {

    /** \brief Interpolate on a manifold, as fast as we can
     *
     * This class implements interpolation of values on a manifold in a 'quick-and-dirty' way.
     * No particular interpolation rule is used consistently for all manifolds.  Rather, it is
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

      // Special-casing for the Rotations.  This is quite a challenge:
      //
      // Projection-based interpolation in the
      // unit quaternions is not a good idea, here, because you may end up on the
      // wrong sheet of the covering of SO(3) by the unit quaternions.  The minimization
      // solver for the weighted distance functional may then actually converge
      // to the wrong local minimizer -- the interpolating functions wraps "the wrong way"
      // around SO(3).
      //
      // Projection-based interpolation in SO(3) is not a good idea, because it is about as
      // expensive as the geodesic interpolation itself.  That's not good if all we want
      // is an initial iterate.
      //
      // Averaging in R^{3x3} followed by QR decomposition is not a good idea either:
      // The averaging can give you singular matrices quite quickly (try two matrices that
      // differ by a rotation of pi/2).  Then, the QR factorization of the identity may
      // not be the identity (but differ in sign)!  I tried that with the implementation
      // from Numerical Recipes.
      //
      // What do we do instead?  We start from the coefficient that produces the lowest
      // value of the objective functional.
      if constexpr (std::is_same<TargetSpace,Rotation<double,3> >::value)
      {
        AverageDistanceAssembler averageDistance(coefficients_,w);

        auto minDistance = averageDistance.value(coefficients_[0]);
        std::size_t minCoefficient = 0;

        for (std::size_t i=1; i<coefficients_.size(); i++)
        {
          auto distance = averageDistance.value(coefficients_[i]);
          if (distance < minDistance)
          {
            minDistance = distance;
            minCoefficient = i;
          }
        }

        return coefficients_[minCoefficient];
      }
      else
      {
        typename TargetSpace::CoordinateType c(0);
        for (size_t i=0; i<coefficients_.size(); i++)
          c.axpy(w[i][0], coefficients_[i].globalCoordinates());

        return TargetSpace(c);
      }
    }
  }   // namespace GFE

}   // namespace Dune
#endif
