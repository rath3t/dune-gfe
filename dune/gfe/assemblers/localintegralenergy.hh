#ifndef DUNE_GFE_ASSEMBLERS_LOCALINTEGRALENERGY_HH
#define DUNE_GFE_ASSEMBLERS_LOCALINTEGRALENERGY_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/fmatrixev.hh>

#include <dune/geometry/quadraturerules.hh>

#include <dune/gfe/assemblers/localenergy.hh>
#include <dune/gfe/cosseratstrain.hh>
#include <dune/gfe/densities/localdensity.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>

#include <dune/elasticity/materials/localdensity.hh>

namespace Dune::GFE {

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

    /** \brief Constructor with a Dune::Elasticity::LocalDensity
     */
    LocalIntegralEnergy(const std::shared_ptr<Elasticity::LocalDensity<gridDim,RT,DT> >& ld)
      : localDensityElasticity_(ld)
    {}

    /** \brief Constructor with a Dune::GFE::LocalDensity
     */
    LocalIntegralEnergy(const std::shared_ptr<GFE::LocalDensity<FieldVector<DT,gridDim>,TargetSpace> >& ld)
      : localDensityGFE_(ld)
    {}

  private:

    /** \brief Assemble the energy for a single element */
    RT energy(const typename Basis::LocalView& localView,
              const std::vector<TargetSpace>& localConfiguration) const override
    {
      RT energy = 0;

      if constexpr (not Impl::LocalEnergyTypes<TargetSpace>::isProductManifold)
      {
        const auto& localFiniteElement = localView.tree().finiteElement();
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

          const auto jacobianInverse = element.geometry().jacobianInverse(quadPos);

          // Function value at the quadrature point
          TargetSpace q = localInterpolationRule.evaluate(quadPos);

          // The derivative of the finite element function at the quadrature point
          auto referenceDerivative = localInterpolationRule.evaluateDerivative(quadPos, q);
          auto derivative = referenceDerivative * jacobianInverse;

          energy += qp.weight() * integrationElement * (*localDensityGFE_)(quadPos,q,derivative);
        }
      }

      return energy;
    }

    RT energy (const typename Basis::LocalView& localView,
               const typename Impl::LocalEnergyTypes<TargetSpace>::CompositeCoefficients& coefficients) const override
    {
      RT energy = 0;

      if constexpr (Impl::LocalEnergyTypes<TargetSpace>::isProductManifold)
      {
        // TODO: Cosserat materials are hard-wired here for historical reasons.
        static_assert(TargetSpace::size() == 2, "LocalGeodesicIntegralEnergy needs two TargetSpaces!");
        using TargetSpaceDeformation = typename std::tuple_element<0, TargetSpace>::type;
        using TargetSpaceRotation = typename std::tuple_element<1, TargetSpace>::type;

        const auto& element = localView.element();

        using namespace Indices;
        const std::vector<TargetSpaceDeformation>& localDeformationConfiguration = coefficients[_0];
        const std::vector<TargetSpaceRotation>& localOrientationConfiguration = coefficients[_1];

        // composite Basis: grab the finite element of the first child
        const auto& deformationLocalFiniteElement = localView.tree().child(_0,0).finiteElement();
        const auto& orientationLocalFiniteElement = localView.tree().child(_1,0).finiteElement();

        using LocalDeformationGFEFunctionType = typename std::tuple_element<0, LocalInterpolationRule>::type;
        using LocalOrientationGFEFunctionType = typename std::tuple_element<1, LocalInterpolationRule>::type;

        LocalDeformationGFEFunctionType localDeformationGFEFunction(deformationLocalFiniteElement,localDeformationConfiguration);
        LocalOrientationGFEFunctionType localOrientationGFEFunction(orientationLocalFiniteElement,localOrientationConfiguration);


        int quadOrder = (element.type().isSimplex()) ? deformationLocalFiniteElement.localBasis().order()
                                                 : deformationLocalFiniteElement.localBasis().order() * gridDim;

        const auto& quad = QuadratureRules<DT, gridDim>::rule(element.type(), quadOrder);

        for (size_t pt=0; pt<quad.size(); pt++)
        {
          // Local position of the quadrature point
          const FieldVector<DT,gridDim>& quadPos = quad[pt].position();

          auto x = element.geometry().global(quadPos);

          const DT integrationElement = element.geometry().integrationElement(quadPos);

          const auto jacobianInverseTransposed = element.geometry().jacobianInverseTransposed(quadPos);

          DT weightWithintegrationElement = quad[pt].weight() * integrationElement;

          // The value of the local deformation
          RealTuple<RT,gridDim> deformationValue = localDeformationGFEFunction.evaluate(quadPos);

          // The derivative of the deformation defined on the reference element
          typename LocalDeformationGFEFunctionType::DerivativeType deformationReferenceDerivative = localDeformationGFEFunction.evaluateDerivative(quadPos,deformationValue);

          // The derivative of the deformation defined on the actual element
          typename LocalDeformationGFEFunctionType::DerivativeType deformationDerivative;
          for (size_t comp=0; comp<deformationReferenceDerivative.N(); comp++)
            jacobianInverseTransposed.mv(deformationReferenceDerivative[comp], deformationDerivative[comp]);

          // Integrate the energy density
          if (localDensityElasticity_)
            energy += weightWithintegrationElement * (*localDensityElasticity_)(x, deformationDerivative);
          else if (localDensityGFE_) {
            // The value of the local rotation
            Rotation<RT,gridDim>  orientationValue = localOrientationGFEFunction.evaluate(quadPos);

            ProductManifold<RealTuple<RT,gridDim>,Rotation<RT,gridDim> > value;
            value[_0] = deformationValue;
            value[_1] = orientationValue;

            // The derivative of the rotation defined on the reference element
            typename LocalOrientationGFEFunctionType::DerivativeType orientationReferenceDerivative = localOrientationGFEFunction.evaluateDerivative(quadPos,orientationValue);

            // The derivative of the rotation defined on the actual element
            typename LocalOrientationGFEFunctionType::DerivativeType orientationDerivative;
            for (size_t comp=0; comp<orientationReferenceDerivative.N(); comp++)
              jacobianInverseTransposed.mv(orientationReferenceDerivative[comp], orientationDerivative[comp]);

            // Copy the two derivatives into a joint matrix object
            // TODO: I am not sure about this.  May the densities should get the
            // separate derivatives.
            FieldMatrix<RT,deformationDerivative.rows+orientationDerivative.rows, deformationDerivative.cols> derivative;

            for (int i=0; i<deformationDerivative.rows; i++)
              derivative[i] = deformationDerivative[i];

            for (int i=0; i<orientationDerivative.rows; i++)
              derivative[i+deformationDerivative.rows] = orientationDerivative[i];

            energy += weightWithintegrationElement * (*localDensityGFE_)(x,
                                                                         value,
                                                                         derivative);
          }
        }
      }

      return energy;
    }

  protected:
    const std::shared_ptr<Elasticity::LocalDensity<gridDim,RT,DT> > localDensityElasticity_ = nullptr;
    const std::shared_ptr<GFE::LocalDensity<FieldVector<DT,gridDim>,TargetSpace> > localDensityGFE_ = nullptr;
  };

}  // namespace Dune::GFE

#endif   //#ifndef DUNE_GFE_ASSEMBLERS_LOCALINTEGRALENERGY_HH
