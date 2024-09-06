#ifndef DUNE_GFE_ASSEMBLERS_LOCALINTEGRALENERGY_HH
#define DUNE_GFE_ASSEMBLERS_LOCALINTEGRALENERGY_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/version.hh>

#include <dune/geometry/quadraturerules.hh>

#include <dune/gfe/assemblers/localenergy.hh>
#include <dune/gfe/densities/localdensity.hh>


namespace Dune::GFE {

#if ! DUNE_VERSION_GTE(DUNE_LOCALFUNCTIONS, 2, 10)
  namespace Impl
  {
    template <class Basis>
    class LocalFiniteElementFactory
    {
    public:
      static auto get(const typename Basis::LocalView& localView)
      -> decltype(localView.tree().child(0).finiteElement())
      {
        return localView.tree().child(0).finiteElement();
      }
    };

    /** \brief Specialize for scalar bases, here we cannot call tree().child() */
    template <class GridView, int order>
    class LocalFiniteElementFactory<Dune::Functions::LagrangeBasis<GridView,order> >
    {
    public:
      static auto get(const typename Dune::Functions::LagrangeBasis<GridView,order>::LocalView& localView)
      -> decltype(localView.tree().finiteElement())
      {
        return localView.tree().finiteElement();
      }
    };
  }
#endif

  /** \brief An energy given as an integral over a density
   *
   * \tparam Basis The scalar finite element basis used to construct the interpolation rule
   * \tparam LocalInterpolationRule The rule that turns coefficients into functions
   * \tparam TargetSpace The space that the geometric finite element function maps into
   */
  template<class Basis, class LocalInterpolationRule, class TargetSpace>
  class LocalIntegralEnergy
    : public Dune::GFE::LocalEnergy<Basis,TargetSpace>
  {
    using LocalView = typename Basis::LocalView;
    using GridView = typename LocalView::GridView;
    using DT = typename GridView::Grid::ctype;
    using RT = typename GFE::LocalEnergy<Basis,TargetSpace>::RT;

    constexpr static int gridDim = GridView::dimension;

  public:

    /** \brief Constructor with a Dune::GFE::LocalDensity
     */
    LocalIntegralEnergy(const std::shared_ptr<GFE::LocalDensity<FieldVector<DT,gridDim>,TargetSpace> >& ld)
      : localDensity_(ld)
    {}

  private:

    /** \brief Assemble the energy for a single element */
    RT energy(const typename Basis::LocalView& localView,
              const std::vector<TargetSpace>& localConfiguration) const override
    {
      RT energy = 0;

      if constexpr (Basis::LocalView::Tree::isLeaf || Basis::LocalView::Tree::isPower)
      {
#if DUNE_VERSION_GTE(DUNE_LOCALFUNCTIONS, 2, 10)
        // Get an appropriate scalar local finite element, to construct the interpolation rule with
        // TODO: This is not a good design, for several reasons:
        // * The interpolation rule could want to have state beyond what we know here.
        //   It should therefore be constructed outside of the LocalIntegralEnergy class
        // * I don't really see why Basis should be allowed to be scalar-valued
        //   to begin with, but a lot of code still currently does that.
        auto lfeGetter = [&localView]()
                         {
                           if constexpr (Basis::LocalView::Tree::isPower)
                             return localView.tree().child(0).finiteElement();
                           else
                             return localView.tree().finiteElement();
                         };

        const auto& localFiniteElement = lfeGetter();
#else
        const auto& localFiniteElement = Impl::LocalFiniteElementFactory<Basis>::get(localView);
#endif
        LocalInterpolationRule localInterpolationRule(localFiniteElement,localConfiguration);

        int quadOrder = (localFiniteElement.type().isSimplex())
           ? (localFiniteElement.localBasis().order()-1) * 2
           : (localFiniteElement.localBasis().order() * gridDim - 1) * 2;

        const auto element = localView.element();

        const auto& quad = QuadratureRules<double, gridDim>::rule(localFiniteElement.type(), quadOrder);

        for (auto&& qp : quad)
        {
          // Local position of the quadrature point
          const auto& quadPos = qp.position();

          const auto integrationElement = element.geometry().integrationElement(quadPos);

          const auto geometryJacobianInverse = element.geometry().jacobianInverse(quadPos);

          if (localDensity_->dependsOnValue())
          {
            if (localDensity_->dependsOnDerivative())
            {
              auto [value, derivative] = localInterpolationRule.evaluateValueAndDerivative(quadPos);
              derivative = derivative * geometryJacobianInverse;
              energy += qp.weight() * integrationElement * (*localDensity_)(quadPos,value.globalCoordinates(),derivative);
            }
            else
            {
              auto value = localInterpolationRule.evaluate(quadPos);
              typename LocalInterpolationRule::DerivativeType dummyDerivative;
              energy += qp.weight() * integrationElement * (*localDensity_)(quadPos,value.globalCoordinates(),dummyDerivative);
            }
          }
          else
          {
            if (localDensity_->dependsOnDerivative())
            {
              typename TargetSpace::CoordinateType dummyValue;
              auto derivative = localInterpolationRule.evaluateDerivative(quadPos);
              derivative = derivative * geometryJacobianInverse;
              energy += qp.weight() * integrationElement * (*localDensity_)(quadPos,dummyValue,derivative);
            }
            else
            {
              // Density does not depend on anything.  That's rather pointless, but not an error.
            }
          }
        }
      }
      else
      {
        // You need a scalar basis or a power basis when calling this method.
        std::abort();
      }

      return energy;
    }

    RT energy (const typename Basis::LocalView& localView,
               const typename Impl::LocalEnergyTypes<TargetSpace>::CompositeCoefficients& coefficients) const override
    {
      RT energy = 0;

      if constexpr (Impl::LocalEnergyTypes<TargetSpace>::isProductManifold
                    && Basis::LocalView::Tree::isComposite
                    && gridDim==GridView::dimensionworld) // TODO: Implement the case gridDim!=dimworld
      {
        static_assert(TargetSpace::size() == 2,
                      "LocalIntegralEnergy only implemented for product spaces with two factors!");

        const auto& element = localView.element();

        using namespace Indices;

        // composite Basis: grab the finite element of the first child
        const auto& localFiniteElement0 = localView.tree().child(_0,0).finiteElement();
        const auto& localFiniteElement1 = localView.tree().child(_1,0).finiteElement();

        using LocalGFEFunctionType0 = typename std::tuple_element<0, LocalInterpolationRule>::type;
        using LocalGFEFunctionType1 = typename std::tuple_element<1, LocalInterpolationRule>::type;

        LocalGFEFunctionType0 localGFEFunction0(localFiniteElement0, coefficients[_0]);
        LocalGFEFunctionType1 localGFEFunction1(localFiniteElement1, coefficients[_1]);

        int quadOrder = (element.type().isSimplex()) ? localFiniteElement0.localBasis().order()
                                                 : localFiniteElement0.localBasis().order() * gridDim;

        const auto& quad = QuadratureRules<DT, gridDim>::rule(element.type(), quadOrder);

        for (size_t pt=0; pt<quad.size(); pt++)
        {
          // Local position of the quadrature point
          const auto& quadPos = quad[pt].position();

          auto x = element.geometry().global(quadPos);

          const DT integrationElement = element.geometry().integrationElement(quadPos);

          const auto geometryJacobianInverse = element.geometry().jacobianInverse(quadPos);

          DT weightWithIntegrationElement = quad[pt].weight() * integrationElement;

          // Compute the required values of the interpolation function
          // A value is needed either directly, or for computing the derivative.
          TargetSpace value;
          if (localDensity_->dependsOnValue(0) || localDensity_->dependsOnDerivative(0))
            value[_0] = localGFEFunction0.evaluate(quadPos);
          if (localDensity_->dependsOnValue(1) || localDensity_->dependsOnDerivative(1))
            value[_1] = localGFEFunction1.evaluate(quadPos);

          // Compute the derivatives of the interpolation function factors
          typename LocalGFEFunctionType0::DerivativeType derivative0;
          typename LocalGFEFunctionType1::DerivativeType derivative1;

          if (localDensity_->dependsOnDerivative(0))
            derivative0 = localGFEFunction0.evaluateDerivative(quadPos,value[_0]) * geometryJacobianInverse;

          if (localDensity_->dependsOnDerivative(1))
            derivative1 = localGFEFunction1.evaluateDerivative(quadPos,value[_1]) * geometryJacobianInverse;

          // Copy the two derivatives into a joint matrix object
          // TODO: I am not sure about this.  May the densities should get the
          // separate derivatives.
          FieldMatrix<RT,derivative0.rows+derivative1.rows, derivative0.cols> derivative;

          for (int i=0; i<derivative0.rows; i++)
            derivative[i] = derivative0[i];

          for (int i=0; i<derivative1.rows; i++)
            derivative[i+derivative0.rows] = derivative1[i];

          energy += weightWithIntegrationElement * (*localDensity_)(x,
                                                                    value.globalCoordinates(),
                                                                    derivative);
        }
      }
      else
        DUNE_THROW(Dune::NotImplemented, "Non-product manifold or non-composite basis or gridDim!=dimworld");

      return energy;
    }

  protected:
    const std::shared_ptr<GFE::LocalDensity<FieldVector<DT,gridDim>,TargetSpace> > localDensity_;
  };

}  // namespace Dune::GFE

#endif   //#ifndef DUNE_GFE_ASSEMBLERS_LOCALINTEGRALENERGY_HH
