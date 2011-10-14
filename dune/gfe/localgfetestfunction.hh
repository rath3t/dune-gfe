#ifndef LOCAL_GFE_TEST_FUNCTION_HH
#define LOCAL_GFE_TEST_FUNCTION_HH

#include <vector>

#include <dune/common/fvector.hh>
#include <dune/common/array.hh>

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
                         const std::vector<TargetSpace>& baseCoefficients)
        : localGFEFunction_(localFiniteElement, baseCoefficients)
    {}
    
    /** \brief The number of Lagrange points, NOT the number of basis functions */
    unsigned int size() const
    {
        return localGFEFunction_.size();
    }

    /** \brief Evaluate all shape functions at the given point */
    void evaluateFunction(const Dune::FieldVector<ctype, dim>& local,
                          std::vector<Dune::array<typename TargetSpace::EmbeddedTangentVector,spaceDim> >& out) const;

    /** \brief Evaluate the derivatives of all shape functions function */
    void evaluateJacobian(const Dune::FieldVector<ctype, dim>& local,
                          std::vector<Dune::array<Dune::FieldMatrix<ctype, EmbeddedTangentVector::dimension, dim>,spaceDim> >& out) const;
                          
    /** \brief Polynomial order */
    unsigned int order() const
    {
        return localGFEFunction_.localFiniteElement_.order();
    }

private:

    /** \brief The scalar local finite element, which provides the weighting factors 
     */
    const LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,TargetSpace> localGFEFunction_;

};

template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
void LocalGFETestFunction<dim,ctype,LocalFiniteElement,TargetSpace>::evaluateFunction(const Dune::FieldVector<ctype, dim>& local,
                      std::vector<Dune::array<typename TargetSpace::EmbeddedTangentVector, spaceDim> >& out) const
{
    out.resize(size());
    
    for (size_t i=0; i<size(); i++) {
        
        Dune::FieldMatrix< double, embeddedDim, embeddedDim > derivative;
        
        /** \todo This call internally keeps computing the value of the gfe function at 'local'.
         * This is expensive.  Eventually we should precompute it once and reused the result. */
        localGFEFunction_.evaluateDerivativeOfValueWRTCoefficient (local,
                                                                   i,
                                                                   derivative);

        Dune::FieldMatrix<ctype,spaceDim,embeddedDim> basisVectors = localGFEFunction_.coefficients_[i].orthonormalFrame();
        
        for (int j=0; j<spaceDim; j++)
            derivative.mv(basisVectors[j], out[i][j]);
        
    }
    
}

template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
void LocalGFETestFunction<dim,ctype,LocalFiniteElement,TargetSpace>::evaluateJacobian(const Dune::FieldVector<ctype, dim>& local,
                      std::vector<Dune::array<Dune::FieldMatrix<ctype, EmbeddedTangentVector::dimension, dim>,spaceDim> >& out) const
{
    out.resize(size());
    
    for (size_t i=0; i<size(); i++) {
        
        /** \todo This call internally keeps computing the value of the gfe function at 'local'.
         * This is expensive.  Eventually we should precompute it once and reused the result. */
        Tensor3< double, embeddedDim, embeddedDim, dim > derivative;
        localGFEFunction_.evaluateDerivativeOfGradientWRTCoefficient (local,
                                                                   i,
                                                                   derivative);
        
        Dune::FieldMatrix<ctype,spaceDim,embeddedDim> basisVectors = localGFEFunction_.coefficients_[i].orthonormalFrame();
        
        for (int j=0; j<spaceDim; j++) {
            
            out[i][j] = 0;
        
            // Contract the second index of the derivative with the tangent vector at the i-th Lagrange point.
            // Add that to the result.
            for (int k=0; k<embeddedDim; k++)
                for (int l=0; l<embeddedDim; l++)
                    for (size_t m=0; m<dim; m++)
                        out[i][j][k][m] += derivative[k][l][m] * basisVectors[j][l];
        
        }
    }
    
}


#endif
