#ifndef GLOBAL_GEODESIC_FINITE_ELEMENT_TEST_FUNCTION_HH
#define GLOBAL_GEODESIC_FINITE_ELEMENT_TEST_FUNCTION_HH

#include <vector>

#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/array.hh>

#include <dune/gfe/localgfetestfunction.hh>


/** \brief Global geodesic finite element test function. 
 *
 *  \tparam Basis  - The global basis type.
 *  \tparam TargetSpace - The manifold that this functions takes its values in.
 */
template<class Basis, class TargetSpace>
GlobalGFETestFunction {

    typedef typename Basis::LocalFiniteElement LocalFiniteElement;
    typedef typename Basis::GridView GridView;
    typedef typename GridView::template Codim<0>::Entity Element;
    typedef typename GridView::Grid::ctype  ctype;

    typedef LocalGFETestFunction<GridView::dimension, ctype, LocalFiniteElement, TargetSpace> LocalGFETestFunction;
    typedef typename TargetSpace::TangentVector TangentVector;
    typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;

    //! Dimension of the grid.
    enum { gridDim = GridView::dimension };
    
    //! Dimension of the embedded tanget space
    enum { embeddedDim = EmbeddedTangentVector::dimension };
    
    enum { tangentDim = TargetSpace::TangentVector::dimension };

public:

    //! Create global function by a global basis and the corresponding coefficient vectors
    GlobalGFETestFunction(const Basis& basis, const std::vector<TargetSpace>& baseCoefficients,
                             const std::vector<TangentVector>& testCoefficients) :
        basis_(basis),
        baseCoefficients_(baseCoefficients),
        testCoefficients_(testCoefficients)
    {}

    /** \brief Evaluate the function at local coordinates. */
    void evaluateLocal(const Element& element, const Dune::FieldVector<gridDim,ctype>& local, EmbeddedTangentVector& out) const;

    /** \brief Evaluate the derivative of the function at local coordinates. */
    void evaluateDerivativeLocal(const Element& element, const Dune::FieldVector<gridDim,ctype>& local, 
                                 Dune::FieldMatrix<ctype, embeddedDim, gridDim>& out) const;

private:

    //! The global basis
    const Basis& basis_;
    //! The coefficient vector of the configuration whose tangent spaces contain the test function
    const std::vector<TargetSpace>& baseCoefficients_;
    //! The coefficient vector for the global test function
    const std::vector<TangentVector>& testCoefficients_;
};

template<class Basis, class TargetSpace>
void GlobalGFETestFunction<Basis,TargetSpace>::evaluateLocal(const Element& element, const Dune::FieldVector<gridDim,ctype>& local, 
                                                             EmbeddedTangentVector& out) const
{
    int numOfBasisFct = basis_.getLocalFiniteElement(element).size(); 

    // Extract local base and test coefficients 
    std::vector<TargetSpace> localBaseCoeff(numOfBaseFct);
    std::vector<TangentVector> localTestCoeff(numOfBaseFct);

    for (int i=0; i<numOfBaseFct; i++) {
        localBaseCoeff[i] = baseCoefficients_[basis_.index(element,i)];
        localTestCoeff[i] = testCoefficients_[basis_.index(element,i)];
    }

    // values of the test basis functions 
    std::vector<Dune::array<EmbeddedTangentVector, tangentDim> > values;

    // create local gfe test function
    LocalGFETestFunction localGFE(basis_.getLocalFiniteElement(element),localBaseCoeff);
    localGFE.evaluateFunction(local, values);

    // multiply values with the corresponding test coefficients
    out = 0;

    for (int i=0; i<values.size(); i++)
        for (int j=0; j<tangentDim; j++) {
            values[i][j] *= localTestCoeff[i][j];
            out += values[i][j];
        }
} 
template<class Basis, class TargetSpace>
void GlobalGFETestFunction<Basis,TargetSpace>::evaluateDerivativeLocal(const Element& element, 
                                            const Dune::FieldVector<gridDim,ctype>& local, 
                                            Dune::FieldMatrix<ctype, embeddedDim, gridDim>& out) const
{
    int numOfBasisFct = basis_.getLocalFiniteElement(element).size(); 

    // Extract local base and test coefficients 
    std::vector<TargetSpace> localBaseCoeff(numOfBaseFct);
    std::vector<TangentVector> localTestCoeff(numOfBaseFct);

    for (int i=0; i<numOfBaseFct; i++) {
        localBaseCoeff[i] = baseCoefficients_[basis_.index(element,i)];
        localTestCoeff[i] = testCoefficients_[basis_.index(element,i)];
    }

    // jacobians of the test basis function  - a lot of dims here...
    std::vector<Dune::array<Dune::FieldMatrix<ctype, embeddedDim, gridDim>, tangentDim> > jacobians; 

    // create local gfe test function
    LocalGFETestFunction localGFE(basis_.getLocalFiniteElement(element),localBaseCoeff);
    localGFE.evaluateJacobian(local, jacobians);

    // multiply values with the corresponding test coefficients
    out = 0;

    for (int i=0; i<jacobians.size(); i++)
        for (int j=0; j<tangentDim; j++) {
            jacobians[i][j] *= localTestCoeff[i][j];
            out += values[i][j];
        }
}
#endif
