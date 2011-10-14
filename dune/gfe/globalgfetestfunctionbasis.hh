#ifndef GLOBAL_GFE_TEST_FUNCTION_BASIS_HH
#define GLOBAL_GFE_TEST_FUNCTION_BASIS_HH

#include <dune/fufem/functionspacebases/functionspacebasis.hh>

#include <dune/gfe/localgfetestfunction.hh>

// forward declaration
template <class GridView, class Basis, class TargetSpace>
class GlobalGFETestFunctionBasis;
// not sure if this works
template <class GridView, class Basis, class TargetSpace>
class GlobalGFETestFunctionBasis<GridView, Basis, TargetSpace>::LocalFiniteElement;

/** \brief A global basis for the gfe test functions. 
 *
 *  The local finite elements are constructed by wrapping the LocalGFETestFunction and using them whenever localBasis() is called.
 *  The other LocalFiniteElement methods are not wrapped/implemented by now but it shouldn't be too difficult to do so when they are
 *  needed.
 */
template <class GridView, class Basis, class TargetSpace>
class GlobalGFETestFunctionBasis : public FunctionSpaceBasis<typename Basis::GridView, typename TargetSpace::EmbeddedTangentVector, typename GlobalGFETestFunctionBasis<GridView,Basis,TargetSpace>::LocalFiniteElement> {

public:
    typedef typename Basis::GridView GridView;
private:
    typedef typename Basis::LocalFiniteElement Lfe;
    typedef typename GridView::template Codim<0>::Entity Element;
    typedef typename GridView::Grid::ctype  ctype;

    typedef GlobalGFETestFunctionBasis<GridView,Basis,TargetSpace> This;
    typedef FunctionSpaceBasis<GridView, typename TargetSpace::EmbeddedTangentVector, This> Base;     

    typedef LocalGFETestFunction<GridView::dimension,ctype,LocalFiniteElement,TargetSpace> LocalGFETestFunction;
    
    const static int tangentDim = TargetSpace::TangentVector::dimension;

    /** \brief Struct to wrap the LocalGeodesicTestFunction into a localBasis of a LocalFiniteElement. */
    class LocalFiniteElement {

        public:
            typedef LocalGFETestFunction LocalBasisType;    
    
            LocalFiniteElement(const Lfe& localFiniteElement, 
                               const std::vector<TargetSpace>& localCoefficients) :
                localGfeTestFunction_(localFiniteElement,localCoefficients)
            {}
            
            const LocalBasisType& localBasis() const {return localGfeTestFunction_;}           
 
        private:
            const LocalGFETestFunction& localGfeTestFunction_;     
    };

public:
    GlobalGFETestFunctionBasis(const Basis& basis, std::vector<TargetSpace>& baseCoefficients) :
        Base(basis.getGridView()),
        basis_(basis),
        baseCoefficients_(baseCoefficients)
    {}

    /** \brief Get the local gfe test function wrapped as a LocalFiniteElement. */
    const LocalFiniteElement& getLocalFiniteElement(const Element& element) const
    {
        if (lfe_)
            delete(lfe_);

        int numOfBasisFct = basis_.getLocalFiniteElement(element).localBasis().size(); 

        // Extract local base and test coefficients 
        std::vector<TargetSpace> localBaseCoeff(numOfBaseFct);

        for (int i=0; i<numOfBaseFct; i++)
            localBaseCoeff[i] = baseCoefficients_[basis_.index(element,i)];

        // create local gfe test function
        lfe_ = new LocalFiniteElement(basis_.getLocalFiniteElement(element),localBaseCoeff);
    
        return *lfe_;
    }

    ~GlobalGFETestFunctionBasis() {
        if (lfe_)
            delete(lfe_);
    }

    /** \brief Get the total number of global basis functions.
     *
     *  For each Lagrange point there are (dim TangentSpace) basis functions
     */
    int size() const
    { 
        return basis_.size()*tangentDim;
    }

    /** \brief Get the global index of the basis function. */
    int index(const Element& e, const int i) const
    {
        return basis_.index(e,i/tangentDim)*tangentDim + (i%tangentDim);   
    }


private:
    //! The global basis determining the weights
    const Basis& basis_;
    //! The coefficients of the configuration the tangent spaces are from. */
    const std::vector<TargetSpace>& baseCoefficients_; 
    //! Save the last local finite element - do I need to do this?
    mutable LocalFiniteElement* lfe_;
};

#endif
