#ifndef GLOBAL_GFE_TEST_FUNCTION_BASIS_HH
#define GLOBAL_GFE_TEST_FUNCTION_BASIS_HH

#include <dune/fufem/functionspacebases/functionspacebasis.hh>

#include <dune/gfe/localgfetestfunction.hh>

// forward declaration
//template <class Basis, class TargetSpace
//class GlobalGFETestFunctionBasis;
// not sure if this works
//template <class Basis, class TargetSpace>
//class GlobalGFETestFunctionBasis<Basis, TargetSpace>::LocalFiniteElement{};

template <class LocalFE, class TargetSpace>
class LocalTestFunctionWrapper;

/** \brief A global basis for the gfe test functions. 
 *
 *  The local finite elements are constructed by wrapping the LocalGFETestFunction and using them whenever localBasis() is called.
 *  The other LocalFiniteElement methods are not wrapped/implemented by now but it shouldn't be too difficult to do so when they are
 *  needed.
 */
template <class Basis, class TargetSpace, class CoefficientType>
class GlobalGFETestFunctionBasis : public FunctionSpaceBasis<typename Basis::GridView, typename TargetSpace::EmbeddedTangentVector, LocalTestFunctionWrapper<typename Basis::LocalFiniteElement, TargetSpace> > {

public:
    typedef typename Basis::GridView GridView;
    typedef LocalTestFunctionWrapper<typename Basis::LocalFiniteElement, TargetSpace> LocalFiniteElement; 
private:
    typedef typename GridView::template Codim<0>::Entity Element;
    typedef typename GridView::Grid::ctype  ctype;

    typedef GlobalGFETestFunctionBasis<Basis,TargetSpace> This;
    typedef FunctionSpaceBasis<GridView, typename TargetSpace::EmbeddedTangentVector, typename This::LocalFiniteElement> Base;     

public:
    GlobalGFETestFunctionBasis(const Basis& basis, const CoefficientType& baseCoefficients) :
        Base(basis.getGridView()),
        basis_(basis),
        baseCoefficients_(baseCoefficients),
        lfe_(NULL)
    {}

    /** \brief Get the local gfe test function wrapped as a LocalFiniteElement. */
    const LocalFiniteElement& getLocalFiniteElement(const Element& element) const
    {
        if (lfe_)
            delete(lfe_);

        int numOfBasisFct = basis_.getLocalFiniteElement(element).localBasis().size(); 

        // Extract local base and test coefficients 
        std::vector<TargetSpace> localBaseCoeff(numOfBasisFct);

        for (int i=0; i<numOfBasisFct; i++)
            localBaseCoeff[i] = baseCoefficients_[basis_.index(element,i)];

        // create local gfe test function
        lfe_ = new LocalFiniteElement(basis_.getLocalFiniteElement(element),localBaseCoeff);
    
        return *lfe_;
    }

    ~GlobalGFETestFunctionBasis() {
        if (lfe_)
            delete(lfe_);
    }

    /** \brief Get the total number of global (block) basis functions.
     *  Only return the number of Lagrange points. For each Lagrange point there are (tangentDim) local shape functions
     */
    int size() const
    { 
        return basis_.size();
    }

    /** \brief Get the global index of the (block) basis function. */
    int index(const Element& e, const int i) const
    {
        return basis_.index(e,i);   
    }


private:
    //! The global basis determining the weights
    const Basis& basis_;
    //! The coefficients of the configuration the tangent spaces are from. */
    const CoefficientType& baseCoefficients_; 
    //! Save the last local finite element - do I need to do this?
    mutable LocalFiniteElement* lfe_;
};

/** \brief Struct to wrap the LocalGeodesicTestFunction into a localBasis of a LocalFiniteElement. */
template <class LocalFE, class TargetSpace>
class LocalTestFunctionWrapper {

    public:
        typedef typename LocalFE::Traits::LocalBasisType::Traits BasisTraits;
        typedef LocalGFETestFunction<BasisTraits::dimDomain,typename BasisTraits::DomainFieldType,LocalFE,TargetSpace> LocalBasisType;

        LocalTestFunctionWrapper(const LocalFE& localFiniteElement, 
                const std::vector<TargetSpace>& localCoefficients) :
            localGfeTestFunction_(localFiniteElement,localCoefficients)
    {}
        
        /** \brief Get the local Basis. */
        const LocalBasisType& localBasis() const {return localGfeTestFunction_;}           

    private:
        LocalBasisType localGfeTestFunction_;     
};





#endif
