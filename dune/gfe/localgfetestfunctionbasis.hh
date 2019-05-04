#ifndef LOCAL_GFE_TEST_FUNCTION_HH
#define LOCAL_GFE_TEST_FUNCTION_HH

#include <array>
#include <vector>

#include <dune/common/fvector.hh>
#include <dune/geometry/type.hh>

#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/tensor3.hh>
#include <dune/gfe/linearalgebra.hh>

#include <dune/localfunctions/common/localfiniteelementtraits.hh>
#include <dune/localfunctions/common/localbasis.hh>

// forward declaration
template <class LocalFiniteElement, class TargetSpace>
class LocalGfeTestFunctionBasis;

template <class LocalFiniteELement, class TargetSpace>
class LocalGfeTestFunctionInterpolation;

/** \brief The local gfe test function finite element
 *
 *  \tparam LagrangeLfe - A Lagrangian finite element whose shape functions define the interpolation weights
 *  \tparam TargetSpace - The manifold the tangent spaces this basis for from belong to.
 */
template <class LagrangeLfe, class TargetSpace>
class LocalGfeTestFunctionFiniteElement
{
    typedef typename LagrangeLfe::Traits::LocalBasisType::Traits LagrangeBasisTraits;
    typedef LocalGfeTestFunctionBasis<LagrangeLfe, TargetSpace> LocalBasis;
    typedef LocalGfeTestFunctionInterpolation<LagrangeLfe, TargetSpace> LocalInterpolation;

public:
    //! Traits 
    typedef Dune::LocalFiniteElementTraits<LocalBasis,typename LagrangeLfe::Traits::LocalCoefficientsType, LocalInterpolation> Traits;

    /** Construct local finite element from the base coefficients and Lagrange local finite element.
     * 
     *  \param lfe - The Lagrange local finite element.
     *  \param baseCoeff - The coefficients of the base points the tangent spaces live at.
     */
    LocalGfeTestFunctionFiniteElement(const LagrangeLfe& lfe, const std::vector<TargetSpace> baseCoeff) :
        baseCoeff_(baseCoeff),
        basis_(lfe, baseCoeff_),
        coefficients_(lfe.clone()->localCoefficients()),
        gt_(lfe.type())
    {}

    /** \brief Get reference to the local basis.*/
    const typename Traits::LocalBasisType& localBasis () const
    {
        return basis_;
    }

    /** \brief Get reference to the local coefficients. */
    const typename Traits::LocalCoefficientsType& localCoefficients () const
    {
        return coefficients_;
    }

    /** \brief Get reference to the local interpolation handler. */
    const typename Traits::LocalInterpolationType& localInterpolation () const
    {
        return interpolation_;
    }

    /** \brief Get the element type this finite element lives on. */
    Dune::GeometryType type () const
    {
        return gt_;
    }

    /** \brief Get base coefficients. */
    const std::vector<TargetSpace>& getBaseCoefficients() const {return baseCoeff_;}

private:
    const std::vector<TargetSpace> baseCoeff_;
    LocalBasis basis_;
    const typename Traits::LocalCoefficientsType& coefficients_;
    LocalInterpolation  interpolation_;
    Dune::GeometryType gt_;
};



/** \brief A local basis of the first variations of a given geodesic finite element function. 
 *
 *  \tparam LocalFiniteElement A Lagrangian finite element whose shape functions define the interpolation weights
 *  \tparam TargetSpace The manifold that the function takes its values in
 *
 *  Note that the shapefunctions of this local basis are given blockwise. Each dof corresponds to a local basis of
 *  the tangent space at that dof. Thus the methods return a vector of arrays.
 */
template <class LocalFiniteElement, class TargetSpace>
class LocalGfeTestFunctionBasis
{
    typedef typename LocalFiniteElement::Traits::LocalBasisType::Traits LagrangeBasisTraits;
    static const int dim = LagrangeBasisTraits::dimDomain;
    typedef typename LagrangeBasisTraits::DomainFieldType ctype;
    typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;
    static const int embeddedDim = EmbeddedTangentVector::dimension;
    
    static const int spaceDim = TargetSpace::TangentVector::dimension;

public :
    //! The local basis traits
    typedef Dune::LocalBasisTraits<ctype, dim, Dune::FieldVector<ctype,dim>, 
        typename EmbeddedTangentVector::value_type, embeddedDim, std::array<EmbeddedTangentVector,spaceDim>, 
        std::array<Dune::FieldMatrix<ctype, embeddedDim, dim>,spaceDim>> Traits;
       
    /** \brief Constructor 
     */
    LocalGfeTestFunctionBasis(const LocalFiniteElement& localFiniteElement,
            const std::vector<TargetSpace>& baseCoefficients)
        : localGFEFunction_(localFiniteElement, baseCoefficients)
    {}
    
    /** \brief The number of Lagrange points, NOT the number of basis functions */
    unsigned int size() const
    {
        return localGFEFunction_.size();
    }

    /** \brief Evaluate all shape functions at the given point */
    void evaluateFunction(const typename Traits::DomainType& local,
                                  std::vector<typename Traits::RangeType>& out) const;

    /** \brief Evaluate the derivatives of all shape functions function */
    void evaluateJacobian(const typename Traits::DomainType& local,
                      std::vector<typename Traits::JacobianType>& out) const;  

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

template <class LocalFiniteElement, class TargetSpace>
void LocalGfeTestFunctionBasis<LocalFiniteElement,TargetSpace>::evaluateFunction(const typename Traits::DomainType& local,
                                  std::vector<typename Traits::RangeType>& out) const
{
    out.resize(size());
    
    for (size_t i=0; i<size(); i++) {
        
        Dune::FieldMatrix< double, embeddedDim, embeddedDim > derivative;
        
        /** \todo This call internally keeps computing the value of the gfe function at 'local'.
         * This is expensive.  Eventually we should precompute it once and reused the result. */
        localGFEFunction_.evaluateDerivativeOfValueWRTCoefficient (local, i, derivative);
        Dune::FieldMatrix<ctype,spaceDim,embeddedDim> basisVectors = localGFEFunction_.coefficient(i).orthonormalFrame();
        
        for (int j=0; j<spaceDim; j++)
            derivative.mv(basisVectors[j], out[i][j]);
        
    }
    
}

template <class LocalFiniteElement, class TargetSpace>
void LocalGfeTestFunctionBasis<LocalFiniteElement,TargetSpace>::evaluateJacobian(const typename Traits::DomainType& local,
                      std::vector<typename Traits::JacobianType>& out) const
{
    out.resize(size());
    
    for (size_t i=0; i<size(); i++) {
        
        /** \todo This call internally keeps computing the value of the gfe function at 'local'.
         * This is expensive.  Eventually we should precompute it once and reused the result. */
        Tensor3< double, embeddedDim, embeddedDim, dim > derivative;
        localGFEFunction_.evaluateDerivativeOfGradientWRTCoefficient (local, i, derivative);

        Dune::FieldMatrix<ctype,spaceDim,embeddedDim> basisVectors = localGFEFunction_.coefficient(i).orthonormalFrame();
        
        for (int j=0; j<spaceDim; j++) {
            
            out[i][j] = 0;
            
            // Contract the second index of the derivative with the tangent vector at the i-th Lagrange point.
            // Add that to the result.
            for (int k=0; k<embeddedDim; k++)
                for (int l=0; l<embeddedDim; l++)
                    for (int m=0; m<dim; m++)
                        out[i][j][k][m] += derivative[k][l][m] * basisVectors[j][l];
        }
    }
    
}

template <class LocalFiniteElement, class TargetSpace>
class LocalGfeTestFunctionInterpolation 
{};

#endif
