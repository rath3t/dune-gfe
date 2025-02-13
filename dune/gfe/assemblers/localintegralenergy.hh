#ifndef DUNE_GFE_ASSEMBLERS_LOCALINTEGRALENERGY_HH
#define DUNE_GFE_ASSEMBLERS_LOCALINTEGRALENERGY_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/version.hh>

#include <dune/geometry/quadraturerules.hh>

#include <dune/gfe/assemblers/localenergy.hh>
#include <dune/gfe/densities/localdensity.hh>


namespace Dune::GFE
{
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
    using Element = typename GridView::template Codim<0>::Entity;
    using DT = typename GridView::Grid::ctype;
    using RT = typename GFE::LocalEnergy<Basis,TargetSpace>::RT;

    constexpr static int gridDim = GridView::dimension;

  public:

    /** \brief Constructor from a GFE function as a shared pointer
     *
     * \param localGFEFunction The geometric finite element function that
     * the density will be evaluated on
     * \param ld The density that will be integrated
     */
    LocalIntegralEnergy(std::shared_ptr<LocalInterpolationRule> localGFEFunction,
                        const std::shared_ptr<GFE::LocalDensity<Element,TargetSpace> >& ld)
      : localGFEFunction_(localGFEFunction),
      localDensity_(ld)
    {}

    /** \brief Constructor from a GFE function r-value reference
     *
     * \param localGFEFunction The geometric finite element function that
     * the density will be evaluated on
     * \param ld The density that will be integrated
     */
    LocalIntegralEnergy(LocalInterpolationRule&& localGFEFunction,
                        const std::shared_ptr<GFE::LocalDensity<Element,TargetSpace> >& ld)
      : localGFEFunction_(std::make_shared<LocalInterpolationRule>(std::move(localGFEFunction))),
      localDensity_(ld)
    {}

  private:

    /** \brief Assemble the energy for a single element */
    RT energy(const typename Basis::LocalView& localView,
              const std::vector<TargetSpace>& localConfiguration) const override
    {
      RT energy = 0;

      if constexpr (Basis::LocalView::Tree::isLeaf || Basis::LocalView::Tree::isPower)
      {
        // Bind GFE function to the current element
        const auto& element = localView.element();
        localGFEFunction_->bind(element,localConfiguration);

        // Bind density to the element
        localDensity_->bind(element);

        // Get a suitable quadrature rule
        const auto localGFEOrder = localGFEFunction_->localFiniteElement().localBasis().order();
        int quadOrder = (element.type().isSimplex())
           ? (localGFEOrder-1) * 2
           : (localGFEOrder * gridDim - 1) * 2;

        const auto& quad = QuadratureRules<double, gridDim>::rule(element.type(), quadOrder);

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
              auto [value, derivative] = localGFEFunction_->evaluateValueAndDerivative(quadPos);
              derivative = derivative * geometryJacobianInverse;
              energy += qp.weight() * integrationElement * (*localDensity_)(quadPos,value.globalCoordinates(),derivative);
            }
            else
            {
              const auto value = localGFEFunction_->evaluate(quadPos);
              typename LocalInterpolationRule::DerivativeType dummyDerivative;
              energy += qp.weight() * integrationElement * (*localDensity_)(quadPos,value.globalCoordinates(),dummyDerivative);
            }
          }
          else
          {
            if (localDensity_->dependsOnDerivative())
            {
              typename TargetSpace::CoordinateType dummyValue;
              auto derivative = localGFEFunction_->evaluateDerivative(quadPos);
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

        using namespace Indices;

        // Bind GFE functions to the current element
        const auto& element = localView.element();
        std::get<0>(*localGFEFunction_).bind(element, coefficients[_0]);
        std::get<1>(*localGFEFunction_).bind(element, coefficients[_1]);

        // Bind density to the element
        localDensity_->bind(element);

        // Get a suitable quadrature rule
        const auto feOrder = localView.tree().child(_0,0).finiteElement().localBasis().order();
        int quadOrder = (element.type().isSimplex()) ? feOrder : feOrder * gridDim;

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
            value[_0] = std::get<0>(*localGFEFunction_).evaluate(quadPos);
          if (localDensity_->dependsOnValue(1) || localDensity_->dependsOnDerivative(1))
            value[_1] = std::get<1>(*localGFEFunction_).evaluate(quadPos);

          // Compute the derivatives of the interpolation function factors
          typename std::tuple_element_t<0, LocalInterpolationRule>::DerivativeType derivative0;
          typename std::tuple_element_t<1, LocalInterpolationRule>::DerivativeType derivative1;

          if (localDensity_->dependsOnDerivative(0))
            derivative0 = std::get<0>(*localGFEFunction_).evaluateDerivative(quadPos,value[_0]) * geometryJacobianInverse;

          if (localDensity_->dependsOnDerivative(1))
            derivative1 = std::get<1>(*localGFEFunction_).evaluateDerivative(quadPos,value[_1]) * geometryJacobianInverse;

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

    // The value and derivative of this function are evaluated at the quadrature points,
    // and given to the density.
    const std::shared_ptr<LocalInterpolationRule> localGFEFunction_;

    const std::shared_ptr<GFE::LocalDensity<Element,TargetSpace> > localDensity_;
  };

}  // namespace Dune::GFE

#endif   //#ifndef DUNE_GFE_ASSEMBLERS_LOCALINTEGRALENERGY_HH
