#ifndef GLOBAL_GEODESIC_FINITE_ELEMENT_TEST_FUNCTION_HH
#define GLOBAL_GEODESIC_FINITE_ELEMENT_TEST_FUNCTION_HH

#include <vector>
#include <dune/istl/bvector.hh>

#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/array.hh>


/** \brief Global geodesic finite element test function. 
 *
 *  \tparam B  - The global basis type.
 *  \tparam TargetSpace - The manifold that this functions takes its values in.
 */
template<class B, class TargetSpace, class CoefficientType>
class GlobalGFETestFunction 
{
public:
    typedef B Basis;

private:
    typedef typename Basis::LocalFiniteElement LocalFiniteElement;
    typedef typename Basis::GridView GridView;
    typedef typename GridView::template Codim<0>::Entity Element;
    typedef typename GridView::Grid::ctype  ctype;

    typedef typename TargetSpace::TangentVector TangentVector;
    typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;

    //! Dimension of the grid.
    enum { gridDim = GridView::dimension };
    
    //! Dimension of the embedded tanget space
    enum { embeddedDim = EmbeddedTangentVector::dimension };
    
    enum { tangentDim = TargetSpace::TangentVector::dimension };

public:

    //! Create global function by a global gfe test basis and the corresponding coefficient vectors
    GlobalGFETestFunction(const Basis& basis, const CoefficientType& coefficients) :
        basis_(basis),
        coefficients_(coefficients)
    {}

    /** \brief Evaluate the function at local coordinates. */
    void evaluateLocal(const Element& element, const Dune::FieldVector<ctype,gridDim>& local, EmbeddedTangentVector& out) const;

    /** \brief Evaluate the derivative of the function at local coordinates. */
    void evaluateDerivativeLocal(const Element& element, const Dune::FieldVector<ctype,gridDim>& local, 
                                 Dune::FieldMatrix<ctype, embeddedDim, gridDim>& out) const;

    /** \brief Export the basis. */
    const Basis& basis() const {return basis_;}

    /** \brief Export the coefficient vector. */
    const CoefficientType& coefficientVector() const {return coefficients_;}

private:
    const Basis& basis_;
    const CoefficientType& coefficients_;

};

template<class Basis, class TargetSpace, class CoefficientType>
void GlobalGFETestFunction<Basis,TargetSpace,CoefficientType>::evaluateLocal(const Element& element, const Dune::FieldVector<ctype,gridDim>& local, 
                                                             EmbeddedTangentVector& out) const
{
    // values of the test basis functions 
    std::vector<Dune::array<EmbeddedTangentVector, tangentDim> > values;

    // create local gfe test function
    basis_.getLocalFiniteElement(element).localBasis().evaluateFunction(local, values);

    // multiply values with the corresponding test coefficients and sum them up
    out = 0.0;

    for (size_t i=0; i<values.size(); i++) {
        int index = basis_.index(element,i);
        for (int j=0; j<tangentDim; j++) {
            values[i][j] *= coefficients_[index][j];
            out += values[i][j];
        }
    }
} 
template<class Basis, class TargetSpace , class CoefficientType>
void GlobalGFETestFunction<Basis,TargetSpace,CoefficientType>::evaluateDerivativeLocal(const Element& element, 
                                            const Dune::FieldVector<ctype,gridDim>& local, 
                                            Dune::FieldMatrix<ctype, embeddedDim, gridDim>& out) const
{
    // jacobians of the test basis function  - a lot of dims here...
    std::vector<Dune::array<Dune::FieldMatrix<ctype, embeddedDim, gridDim>, tangentDim> > refJacobians,jacobians; 

    // evaluate local gfe test function basis
    basis_.getLocalFiniteElement(element).localBasis().evaluateJacobian(local, refJacobians);

    const Dune::FieldMatrix<double,gridDim,gridDim>& jacInvTrans = element.geometry().jacobianInverseTransposed(local); 

    //transform the gradients
    jacobians.resize(refJacobians.size());
    for (size_t i = 0; i<jacobians.size(); i++)
        for (int j=0; j<tangentDim; j++) {// the local basis has a block structure
            jacobians[i][j] = 0;
            for (size_t comp=0; comp<jacobians[i][j].N(); comp++)
                jacInvTrans.umv(refJacobians[i][j][comp], jacobians[i][j][comp]); 
        } 

    // multiply values with the corresponding test coefficients and sum them up
    out = 0.0;

    for (int i=0; i<jacobians.size(); i++) {
        int index = basis_.index(element,i);
        for (int j=0; j<tangentDim; j++) {
            jacobians[i][j] *= coefficients_[index][j];
            out += jacobians[i][j];
        }
    }
}
#endif
