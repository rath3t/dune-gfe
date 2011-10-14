#ifndef GLOBAL_GEODESIC_FINITE_ELEMENT_FUNCTION_HH
#define GLOBAL_GEODESIC_FINITE_ELEMENT_FUNCTION_HH

#include <vector>

#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>

#include <dune/gfe/localgeodesicfefunction.hh>


/** \brief Global geodesic finite element function. 
 *
 *  \tparam Basis  - The global basis type.
 *  \tparam TargetSpace - The manifold that this functions takes its values in.
 */
template<class Basis, class TargetSpace>
GlobalGeodesicFEFunction {

    typedef typename Basis::LocalFiniteElement LocalFiniteElement;
    typedef typename Basis::GridView GridView;
    typedef typename GridView::template Codim<0>::Entity Element;
    typedef typename GridView::Grid::ctype  ctype;

    typedef LocalGeodesicFEFunction<GridView::dimension, ctype, LocalFiniteElement, TargetSpace> LocalGFEFunction;
    typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;

    //! Dimension of the grid.
    enum { gridDim = GridView::dimension };
    
    //! Dimension of the embedded tanget space
    enum { embeddedDim = EmbeddedTangentVector::dimension };

    //! Create global function by a global basis and the corresponding coefficient vector
    GlobalGeodesicFEFunction(const Basis& basis, const std::vector<TargetSpace>& coefficients) :
        basis_(basis),
        coefficients_(coefficients)
    {}

    /** \brief Evaluate the function at local coordinates. */
    void evaluateLocal(const Element& element, const Dune::FieldVector<gridDim,ctype>& local, TargetSpace& out) 
    {
        int numOfBasisFct = basis_.getLocalFiniteElement(element).size(); 

        // Extract local solution
        std::vector<TargetSpace> localSolution(numOfBaseFct);
        
        for (int i=0; i<numOfBaseFct; i++)
            localCoeff[i] = coefficients_[basis_.index(element,i)];

        // create local gfe function
        LocalGFEFunction localGFE(basis_.getLocalFiniteElement(element),localCoeff);
        out = localGFE.evaluate(local);
    }

    /** \brief Evaluate the derivative of the function at local coordinates. */
    void evaluateDerivativeLocal(const Element& element, const Dune::FieldVector<gridDim,ctype>& local, 
                                 Dune::FieldMatrix<ctype, embeddedDim, gridDim>& out)
    {
        int numOfBasisFct = basis_.getLocalFiniteElement(element).size(); 

        // Extract local solution
        std::vector<TargetSpace> localSolution(numOfBaseFct);
        
        for (int i=0; i<numOfBaseFct; i++)
            localCoeff[i] = coefficients_[basis_.index(element,i)];

        // create local gfe function
        LocalGFEFunction localGFE(basis_.getLocalFiniteElement(element),localCoeff);
    
        // use it to evaluate the derivative
        out = localGFE.evaluateDerivative(local);
    }

private:
    //! The global basis
    const Basis& basis_;
    //! The coefficient vector
    const std::vector<TargetSpace>& coefficients_;
};
#endif
