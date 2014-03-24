#ifndef DUNE_GFE_LOCALPROJECTEDFEFUNCTION_HH
#define DUNE_GFE_LOCALPROJECTEDFEFUNCTION_HH

#include <vector>

#include <dune/common/fvector.hh>

#include <dune/geometry/type.hh>

namespace Dune {

  namespace GFE {

    /** \brief Interpolate in an embedding Euclidean space, and project back onto the Riemannian manifold
     *
     * \tparam dim Dimension of the reference element
     * \tparam ctype Type used for coordinates on the reference element
     * \tparam LocalFiniteElement A Lagrangian finite element whose shape functions define the interpolation weights
     * \tparam TargetSpace The manifold that the function takes its values in
     */
    template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
    class LocalProjectedFEFunction
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
      LocalProjectedFEFunction(const LocalFiniteElement& localFiniteElement,
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
    TargetSpace LocalProjectedFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::
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

    template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
    typename LocalProjectedFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::DerivativeType
    LocalProjectedFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::
    evaluateDerivative(const Dune::FieldVector<ctype, dim>& local) const
    {
      // the function value at the point where we are evaluating the derivative
      TargetSpace q = evaluate(local);

      // Actually compute the derivative
      return evaluateDerivative(local,q);
    }

    template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
    typename LocalProjectedFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::DerivativeType
    LocalProjectedFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::
    evaluateDerivative(const Dune::FieldVector<ctype, dim>& local, const TargetSpace& q) const
    {
      DUNE_THROW(NotImplemented, "Not implemented yet!");
    }

  }

}
#endif
