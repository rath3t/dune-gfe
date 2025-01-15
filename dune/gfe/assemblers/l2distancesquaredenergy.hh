// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_GFE_L2_DISTANCE_SQUARED_ENERGY_HH
#define DUNE_GFE_L2_DISTANCE_SQUARED_ENERGY_HH

#include <dune/geometry/quadraturerules.hh>

#include <dune/gfe/assemblers/localenergy.hh>
#include <dune/gfe/functions/globalgfefunction.hh>
#include <dune/gfe/functions/localgeodesicfefunction.hh>


namespace Dune::GFE
{

template<class Basis, class TargetSpace>
class L2DistanceSquaredEnergy
  : public Dune::GFE::LocalEnergy<Basis,TargetSpace>
{
  // grid types
  typedef typename Basis::GridView GridView;
  typedef typename GridView::ctype DT;
  typedef typename TargetSpace::ctype RT;

  // some other sizes
  constexpr static int gridDim = GridView::dimension;

  using LocalInterpolationRule = LocalGeodesicFEFunction<gridDim, DT, typename Basis::LocalView::Tree::FiniteElement, typename TargetSpace::template rebind<double>::other>;

public:

  // This is the function that we are computing the L2-distance to
  std::shared_ptr<Dune::GFE::GlobalGFEFunction<Basis, LocalInterpolationRule, typename TargetSpace::template rebind<double>::other > > origin_;

  /** \brief Assemble the energy for a single element */
  RT energy (const typename Basis::LocalView& localView,
             const std::vector<TargetSpace>& localSolution) const override

  {
    RT energy = 0;

    const auto& localFiniteElement = localView.tree().finiteElement();
    typedef LocalGeodesicFEFunction<gridDim, double, decltype(localFiniteElement), TargetSpace> LocalGFEFunctionType;
    LocalGFEFunctionType localGeodesicFEFunction(localFiniteElement,localSolution);

    const auto element = localView.element();
    auto localOrigin = localFunction(*origin_);
    localOrigin.bind(element);

    // Just guessing an appropriate quadrature order
    auto quadOrder = localFiniteElement.localBasis().order() * 2 * gridDim;

    const auto& quad = Dune::QuadratureRules<double, gridDim>::rule(localFiniteElement.type(), quadOrder);

    for (size_t pt=0; pt<quad.size(); pt++)
    {
      // Local position of the quadrature point
      const Dune::FieldVector<double,gridDim>& quadPos = quad[pt].position();

      const double integrationElement = element.geometry().integrationElement(quadPos);

      auto weight = quad[pt].weight() * integrationElement;

      // The function value
      auto value = localGeodesicFEFunction.evaluate(quadPos);
      auto originValue = localOrigin(quadPos);

      // The derivative of the 'origin' function
      // First: as function defined on the reference element
      auto originReferenceDerivative = derivative(localOrigin)(quadPos);

      // The derivative of the function defined on the actual element
      auto originDerivative = originReferenceDerivative * element.geometry().jacobianInverse(quadPos);

      double weightFactor = originDerivative.frobenius_norm();
      // un-comment the following line to switch off the weight factor
      //weightFactor = 1.0;

      // Add the local energy density
      energy += weight * weightFactor * TargetSpace::distanceSquared(originValue, value);

    }

    return energy;
  }

  virtual RT energy (const typename Basis::LocalView& localView,
                     const typename Dune::GFE::Impl::LocalEnergyTypes<TargetSpace>::CompositeCoefficients& coefficients) const override
  {
    DUNE_THROW(Dune::NotImplemented, "!");
  }
};

}  // namespace Dune::GFE

#endif
