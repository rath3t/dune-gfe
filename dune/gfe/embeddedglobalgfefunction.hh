#ifndef DUNE_GFE_EMBEDDEDGLOBALGFEFUNCTION_HH
#define DUNE_GFE_EMBEDDEDGLOBALGFEFUNCTION_HH

#include <vector>

#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>

namespace Dune {

  namespace GFE {

/** \brief Global geodesic finite element function isometrically embedded in a Euclidean space
 *
 *  \tparam B  - The global basis type.
 *  \tparam TargetSpace - The manifold that this functions takes its values in.
 */
template<class B, class LocalInterpolationRule, class TargetSpace>
class EmbeddedGlobalGFEFunction
: public VirtualGridViewFunction<typename B::GridView, typename TargetSpace::CoordinateType>
{
public:
    typedef B Basis;

    typedef typename Basis::LocalView::Tree::FiniteElement LocalFiniteElement;
    typedef typename Basis::GridView GridView;
    typedef typename GridView::template Codim<0>::Entity Element;
    typedef typename GridView::Grid::ctype  ctype;

    typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;

    //! Dimension of the grid.
    enum { gridDim = GridView::dimension };

    //! Dimension of the embedded tanget space
    enum { embeddedDim = EmbeddedTangentVector::dimension };


    //! Create global function by a global basis and the corresponding coefficient vector
    EmbeddedGlobalGFEFunction(const Basis& basis, const std::vector<TargetSpace>& coefficients) :
        VirtualGridViewFunction<typename B::GridView, typename TargetSpace::CoordinateType>(basis.gridView()),
        basis_(basis),
        coefficients_(coefficients)
    {}


    /** \brief Evaluate the function at local coordinates. */
    void evaluateLocal(const Element& element, const Dune::FieldVector<ctype,gridDim>& local, typename TargetSpace::CoordinateType& out) const
    {
        out = this->operator()(element,local);
    }

    /** \brief Global evaluation */
    typename TargetSpace::CoordinateType operator()(const Dune::FieldVector<ctype,gridDim>& global) const
    {
      HierarchicSearch<typename GridView::Grid,typename GridView::IndexSet> hierarchicSearch(basis_.gridView().grid(),
                                                                                             basis_.gridView().indexSet());

      auto element = hierarchicSearch.findEntity(global);
      auto localPos = element.geometry().local(global);

      return this->operator()(element,localPos);
    }

    /** \brief Evaluate the function at local coordinates. */
    typename TargetSpace::CoordinateType operator()(const Element& element, const Dune::FieldVector<ctype,gridDim>& local) const
    {
        auto localView = basis_.localView();
        auto localIndexSet = basis_.localIndexSet();
        localView.bind(element);
        localIndexSet.bind(localView);

        auto numOfBaseFct = localIndexSet.size();

        // Extract local coefficients
        std::vector<TargetSpace> localCoeff(numOfBaseFct);

        for (size_t i=0; i<numOfBaseFct; i++)
            localCoeff[i] = coefficients_[localIndexSet.index(i)];

        // create local gfe function
        LocalInterpolationRule localInterpolationRule(localView.tree().finiteElement(),localCoeff);
        return localInterpolationRule.evaluate(local).globalCoordinates();
    }

    /** \brief Evaluate the derivative of the function at local coordinates. */
    void evaluateDerivativeLocal(const Element& element, const Dune::FieldVector<ctype,gridDim>& local,
                                 Dune::FieldMatrix<ctype, embeddedDim, gridDim>& out) const
    {
        out = derivative(element,local);
    }

    /** \brief Evaluate the derivative of the function at local coordinates. */
    Dune::FieldMatrix<ctype, embeddedDim, gridDim> derivative(const Element& element, const Dune::FieldVector<ctype,gridDim>& local) const
    {
        auto localView = basis_.localView();
        auto localIndexSet = basis_.localIndexSet();
        localView.bind(element);
        localIndexSet.bind(localView);

        int numOfBaseFct = localIndexSet.size();

        // Extract local coefficients
        std::vector<TargetSpace> localCoeff(numOfBaseFct);

        for (int i=0; i<numOfBaseFct; i++)
            localCoeff[i] = coefficients_[localIndexSet.index(i)];

        // create local gfe function
        LocalInterpolationRule localInterpolationRule(localView.tree().finiteElement(),localCoeff);

        // use it to evaluate the derivative
        auto refJac = localInterpolationRule.evaluateDerivative(local);

        decltype(refJac) out =0.0;
        //transform the gradient
        const auto jacInvTrans = element.geometry().jacobianInverseTransposed(local);
        for (size_t k=0; k< refJac.N(); k++)
            jacInvTrans.umv(refJac[k],out[k]);

        return out;
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
