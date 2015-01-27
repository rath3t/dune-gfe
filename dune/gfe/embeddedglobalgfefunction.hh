#ifndef DUNE_GFE_EMBEDDEDGLOBALGFEFUNCTION_HH
#define DUNE_GFE_EMBEDDEDGLOBALGFEFUNCTION_HH

#include <vector>

#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>

#include <dune/gfe/localgeodesicfefunction.hh>

namespace Dune {

  namespace GFE {

/** \brief Global geodesic finite element function isometrically embedded in a Euclidean space
 *
 *  \tparam B  - The global basis type.
 *  \tparam TargetSpace - The manifold that this functions takes its values in.
 */
template<class B, class TargetSpace>
class EmbeddedGlobalGFEFunction
: public VirtualGridViewFunction<typename B::GridView, typename TargetSpace::CoordinateType>
{
public:
    typedef B Basis;

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
    EmbeddedGlobalGFEFunction(const Basis& basis, const std::vector<TargetSpace>& coefficients) :
        VirtualGridViewFunction<typename B::GridView, typename TargetSpace::CoordinateType>(basis.getGridView()),
        basis_(basis),
        coefficients_(coefficients)
    {}


    /** \brief Evaluate the function at local coordinates. */
    void evaluateLocal(const Element& element, const Dune::FieldVector<ctype,gridDim>& local, typename TargetSpace::CoordinateType& out) const
    {
        int numOfBaseFct = basis_.getLocalFiniteElement(element).localBasis().size();

        // Extract local coefficients
        std::vector<TargetSpace> localCoeff(numOfBaseFct);

        for (int i=0; i<numOfBaseFct; i++)
            localCoeff[i] = coefficients_[basis_.index(element,i)];

        // create local gfe function
        LocalGFEFunction localGFE(basis_.getLocalFiniteElement(element),localCoeff);
        out = localGFE.evaluate(local).globalCoordinates();
    }

    /** \brief Evaluate the derivative of the function at local coordinates. */
    void evaluateDerivativeLocal(const Element& element, const Dune::FieldVector<ctype,gridDim>& local,
                                 Dune::FieldMatrix<ctype, embeddedDim, gridDim>& out) const
    {
        int numOfBaseFct = basis_.getLocalFiniteElement(element).localBasis().size();

        // Extract local coefficients
        std::vector<TargetSpace> localCoeff(numOfBaseFct);

        for (int i=0; i<numOfBaseFct; i++)
            localCoeff[i] = coefficients_[basis_.index(element,i)];

        // create local gfe function
        LocalGFEFunction localGFE(basis_.getLocalFiniteElement(element),localCoeff);

        // use it to evaluate the derivative
        Dune::FieldMatrix<ctype, embeddedDim, gridDim> refJac = localGFE.evaluateDerivative(local);

        out =0.0;
        //transform the gradient
        const Dune::FieldMatrix<double,gridDim,gridDim>& jacInvTrans = element.geometry().jacobianInverseTransposed(local);
        for (size_t k=0; k< refJac.N(); k++)
            jacInvTrans.umv(refJac[k],out[k]);
    }

    /** \brief Export basis */
    const Basis& basis() const
    {
        return basis_;
    }

    /** \brief Export coefficients. */
    const std::vector<TargetSpace>& coefficients() const
    {
        return coefficients_;
    }

private:
    //! The global basis
    const Basis& basis_;
    //! The coefficient vector
    const std::vector<TargetSpace>& coefficients_;
};

  }   // namespace GFE

}   // namespace Dune
#endif
