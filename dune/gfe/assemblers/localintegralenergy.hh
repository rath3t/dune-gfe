#ifndef DUNE_GFE_ASSEMBLERS_LOCALINTEGRALENERGY_HH
#define DUNE_GFE_ASSEMBLERS_LOCALINTEGRALENERGY_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/fmatrixev.hh>

#include <dune/geometry/quadraturerules.hh>

#include <dune/gfe/assemblers/localenergy.hh>
#include <dune/gfe/cosseratstrain.hh>
#include <dune/gfe/densities/localdensity.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rigidbodymotion.hh>
#ifdef PROJECTED_INTERPOLATION
#include <dune/gfe/localprojectedfefunction.hh>
#else
#include <dune/gfe/localgeodesicfefunction.hh>
#endif

#include <dune/elasticity/materials/localdensity.hh>

namespace Dune::GFE {

  /**
    \brief Assembles the elastic energy for a single element integrating the localdensity over one element.

           This class works similarly to the class Dune::Elasticity::LocalIntegralEnergy, where Dune::Elasticity::LocalIntegralEnergy extends
           Dune::Elasticity::LocalEnergy and Dune::GFE::LocalIntegralEnergy extends Dune::GFE::LocalEnergy.
  */
template<class Basis, class... TargetSpaces>
class LocalIntegralEnergy
  : public Dune::GFE::LocalEnergy<Basis,TargetSpaces...>
{
  using LocalView = typename Basis::LocalView;
  using GridView = typename LocalView::GridView;
  using DT = typename GridView::Grid::ctype;
  using RT = typename GFE::LocalEnergy<Basis,TargetSpaces...>::RT;

  constexpr static int gridDim = GridView::dimension;

public:

  /** \brief Constructor with a Dune::Elasticity::LocalDensity
    */
  LocalIntegralEnergy(const std::shared_ptr<Elasticity::LocalDensity<gridDim,RT,DT>>& ld)
  : localDensityElasticity_(ld)
  {}

  /** \brief Constructor with a Dune::GFE::LocalDensity
    */
  LocalIntegralEnergy(const std::shared_ptr<GFE::LocalDensity<gridDim,RT,DT>>& ld)
  : localDensityGFE_(ld)
  {}

private:

    /** \brief Assemble the energy for a single element */
  RT energy(const typename Basis::LocalView& localView,
            const std::vector<TargetSpaces>&... localSolutions) const
  { 
    const auto& element = localView.element();

    static_assert(sizeof...(TargetSpaces) > 1, "LocalGeodesicIntegralEnergy needs at least two TargetSpace!");

    using TargetSpaceDeformation = typename std::tuple_element<0, std::tuple<TargetSpaces...> >::type;
    using TargetSpaceRotation = typename std::tuple_element<1, std::tuple<TargetSpaces...> >::type;

    static_assert( (std::is_same<TargetSpaceDeformation, RigidBodyMotion<RT,gridDim>>::value)
      or (std::is_same<TargetSpaceDeformation, RealTuple<RT,gridDim>>::value), "The first TargetSpace of LocalGeodesicIntegralEnergy needs to be RigidBodyMotion or RealTuple!" );
    
    const std::vector<TargetSpaceDeformation>& localDeformationConfiguration = std::get<0>(std::forward_as_tuple(localSolutions...));
    const std::vector<TargetSpaceRotation>& localOrientationConfiguration = std::get<1>(std::forward_as_tuple(localSolutions...));

    using namespace Indices;
    // composite Basis: grab the finite element of the first child
    const auto& deformationLocalFiniteElement = localView.tree().child(_0,0).finiteElement();
    const auto& orientationLocalFiniteElement = localView.tree().child(_1,0).finiteElement();

#ifdef PROJECTED_INTERPOLATION
    using LocalDeformationGFEFunctionType = GFE::LocalProjectedFEFunction<gridDim, DT, decltype(deformationLocalFiniteElement), RealTuple<RT,gridDim> >;
    using LocalOrientationGFEFunctionType = GFE::LocalProjectedFEFunction<gridDim, DT, decltype(orientationLocalFiniteElement), Rotation<RT,gridDim> >;
#else
    using LocalDeformationGFEFunctionType = LocalGeodesicFEFunction<gridDim, DT, decltype(deformationLocalFiniteElement), RealTuple<RT,gridDim> >;
    using LocalOrientationGFEFunctionType = LocalGeodesicFEFunction<gridDim, DT, decltype(orientationLocalFiniteElement), Rotation<RT,gridDim> >;
#endif
    LocalDeformationGFEFunctionType localDeformationGFEFunction(deformationLocalFiniteElement,localDeformationConfiguration);
    LocalOrientationGFEFunctionType localOrientationGFEFunction(orientationLocalFiniteElement,localOrientationConfiguration);

    RT energy = 0;

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

          // The derivative of the rotation defined on the reference element
          typename LocalOrientationGFEFunctionType::DerivativeType orientationReferenceDerivative = localOrientationGFEFunction.evaluateDerivative(quadPos,orientationValue);

          // The derivative of the rotation defined on the actual element
          typename LocalOrientationGFEFunctionType::DerivativeType orientationDerivative;
          for (size_t comp=0; comp<orientationReferenceDerivative.N(); comp++)
              jacobianInverseTransposed.mv(orientationReferenceDerivative[comp], orientationDerivative[comp]);

          energy += weightWithintegrationElement * (*localDensityGFE_)(x,
                                                                       deformationValue,
                                                                       deformationDerivative,
                                                                       orientationValue,
                                                                       orientationDerivative);
        }
    }

    return energy;
  }

protected:
  const std::shared_ptr<Elasticity::LocalDensity<gridDim,RT,DT>> localDensityElasticity_ = nullptr;
  const std::shared_ptr<GFE::LocalDensity<gridDim,RT,DT>> localDensityGFE_ = nullptr;
};

}  // namespace Dune::GFE

#endif   //#ifndef DUNE_GFE_ASSEMBLERS_LOCALINTEGRALENERGY_HH
