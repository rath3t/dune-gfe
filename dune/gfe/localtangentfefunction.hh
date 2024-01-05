#ifndef DUNE_GFE_LOCALTANGENTFEFUNCTION_HH
#define DUNE_GFE_LOCALTANGENTFEFUNCTION_HH

#include <vector>

#include <dune/common/fvector.hh>

#include <dune/geometry/type.hh>

namespace Dune {

  namespace GFE {

    /** \brief Interpolate on a tangent space
     *
     * \tparam dim Dimension of the reference element
     * \tparam ctype Type used for coordinates on the reference element
     * \tparam LocalFiniteElement A Lagrangian finite element whose shape functions define the interpolation weights
     * \tparam TargetSpace The manifold that the function takes its values in
     */
    template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
    class LocalTangentFEFunction
    {
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
      LocalTangentFEFunction(const LocalFiniteElement& localFiniteElement,
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

      /** \brief Evaluate the derivative of the function */
      DerivativeType evaluateDerivative(const Dune::FieldVector<ctype, dim>& local) const;

      /** \brief Evaluate the derivative of the function, if you happen to know the function value (much faster!)
       *        \param local Local coordinates in the reference element where to evaluate the derivative
       *        \param q Value of the local gfe function at 'local'.  If you provide something wrong here the result will be wrong, too!
       */
      DerivativeType evaluateDerivative(const Dune::FieldVector<ctype, dim>& local,
                                        const TargetSpace& q) const;

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
    TargetSpace LocalTangentFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::
    evaluate(const Dune::FieldVector<ctype, dim>& local) const
    {
      // Evaluate the weighting factors---these are the Lagrangian shape function values at 'local'
      std::vector<Dune::FieldVector<ctype,1> > w;
      localFiniteElement_.localBasis().evaluateFunction(local,w);

      // This is the base point of the tangent space that we are interpolating on
      // HACK: This point is reasonable only for spheres
      typename TargetSpace::CoordinateType baseEmb(0);
      baseEmb[0] = 1.0;
      TargetSpace base(baseEmb);

      // Map all points onto the tangent space at 'base'
      std::vector<typename TargetSpace::EmbeddedTangentVector> tangentCoefficients(coefficients_.size());
      for (size_t i=0; i<coefficients_.size(); i++)
        tangentCoefficients[i] = TargetSpace::log(base,coefficients_[i]);

      // Interpolate on the tangent space
      typename TargetSpace::EmbeddedTangentVector c(0);
      for (size_t i=0; i<tangentCoefficients.size(); i++)
        c.axpy(w[i][0], tangentCoefficients[i]);

      return TargetSpace::exp(base,c);
    }

    template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
    typename LocalTangentFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::DerivativeType
    LocalTangentFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::
    evaluateDerivative(const Dune::FieldVector<ctype, dim>& local) const
    {
      // the function value at the point where we are evaluating the derivative
      TargetSpace q = evaluate(local);

      // Actually compute the derivative
      return evaluateDerivative(local,q);
    }

    template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
    typename LocalTangentFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::DerivativeType
    LocalTangentFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::
    evaluateDerivative(const Dune::FieldVector<ctype, dim>& local, const TargetSpace& q) const
    {
      DUNE_THROW(NotImplemented, "evaluateDerivative not implemented yet!");
    }

  }

}
#endif
