#ifndef LOCAL_GFE_TEST_FUNCTION_HH
#define LOCAL_GFE_TEST_FUNCTION_HH

#include <vector>

#include <dune/common/fvector.hh>

#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/tensor3.hh>
#include <dune/gfe/linearalgebra.hh>

/** \brief A function defined by simplicial geodesic interpolation 
           from the reference element to a Riemannian manifold.
    
\tparam dim Dimension of the reference element
\tparam ctype Type used for coordinates on the reference element
\tparam LocalFiniteElement A Lagrangian finite element whose shape functions define the interpolation weights
\tparam TargetSpace The manifold that the function takes its values in
*/
template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
class LocalGFETestFunction
{
    
    typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;
    static const int embeddedDim = EmbeddedTangentVector::dimension;
    
    static const int spaceDim = TargetSpace::TangentVector::dimension;

public:

    /** \brief Constructor 
     */
    LocalGFETestFunction(const LocalFiniteElement& localFiniteElement,
                         const std::vector<TargetSpace>& baseCoefficients,
                         const std::vector<typename TargetSpace::TangentVector>& tangentCoefficients)
        : localGFEFunction_(localFiniteElement, baseCoefficients),
        tangentCoefficients_(tangentCoefficients)
    {}

    /** \brief Evaluate the function */
    typename TargetSpace::EmbeddedTangentVector evaluate(const Dune::FieldVector<ctype, dim>& local) const;

    /** \brief Evaluate the derivative of the function */
    Dune::FieldMatrix<ctype, EmbeddedTangentVector::dimension, dim> evaluateDerivative(const Dune::FieldVector<ctype, dim>& local) const;

private:

    /** \brief The scalar local finite element, which provides the weighting factors 
     */
    const LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,TargetSpace> localGFEFunction_;

    std::vector<typename TargetSpace::TangentVector> tangentCoefficients_;

};

template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
typename TargetSpace::EmbeddedTangentVector LocalGFETestFunction<dim,ctype,LocalFiniteElement,TargetSpace>::
evaluate(const Dune::FieldVector<ctype, dim>& local) const
{
    typename TargetSpace::EmbeddedTangentVector result(0);

    for (size_t i=0; i<tangentCoefficients_.size(); i++) {
        
        Dune::FieldMatrix< double, embeddedDim, embeddedDim > derivative;
        
        localGFEFunction_.evaluateDerivativeOfValueWRTCoefficient (local,
                                                                   i,
                                                                   derivative);
        
        derivative.umv(tangentCoefficients_, result);
        
    }
    
    return result;
}

template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
Dune::FieldMatrix<ctype, TargetSpace::EmbeddedTangentVector::dimension, dim> LocalGFETestFunction<dim,ctype,LocalFiniteElement,TargetSpace>::
evaluateDerivative(const Dune::FieldVector<ctype, dim>& local) const
{
    typename Dune::FieldMatrix<ctype, TargetSpace::EmbeddedTangentVector::dimension, dim> result(0);

    for (size_t i=0; i<tangentCoefficients_.size(); i++) {
        
        Tensor3< double, embeddedDim, embeddedDim, dim > derivative;
        localGFEFunction_.evaluateDerivativeOfGradientWRTCoefficient (local,
                                                                   i,
                                                                   derivative);
        
        // Contract the second index of the derivative with the tangent vector at the i-th Lagrange point.
        // Add that to the result.
        for (size_t j=0; j<embeddedDim; j++)
            for (size_t k=0; k<embeddedDim; k++)
                for (size_t l=0; l<dim; l++)
                    result[j][l] += derivative[j][k][l] * tangentCoefficients_[i][k];
        
    }
    
    return result;

}


#endif
