#ifndef DUNE_GFE_HARMONICENERGY_HH
#define DUNE_GFE_HARMONICENERGY_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/gfe/assemblers/localenergy.hh>
#include <dune/gfe/densities/harmonicdensity.hh>

template<class Basis, class LocalInterpolationRule, class TargetSpace>
class HarmonicEnergy
  : public Dune::GFE::LocalEnergy<Basis,TargetSpace>
{
  // grid types
  typedef typename Basis::GridView GridView;
  typedef typename GridView::ctype DT;
  typedef typename TargetSpace::ctype RT;

  // some other sizes
  constexpr static int gridDim = GridView::dimension;

public:

  /** \brief Assemble the energy for a single element */
  RT energy (const typename Basis::LocalView& localView,
             const std::vector<TargetSpace>& localSolution) const override;

};

template <class Basis, class LocalInterpolationRule, class TargetSpace>
typename HarmonicEnergy<Basis, LocalInterpolationRule, TargetSpace>::RT
HarmonicEnergy<Basis, LocalInterpolationRule, TargetSpace>::
energy(const typename Basis::LocalView& localView,
       const std::vector<TargetSpace>& localSolution) const
{
  Dune::GFE::HarmonicDensity<Dune::FieldVector<DT,gridDim>,TargetSpace> density;

  RT energy = 0;

  const auto& localFiniteElement = localView.tree().finiteElement();
  LocalInterpolationRule localInterpolationRule(localFiniteElement,localSolution);

  int quadOrder = (localFiniteElement.type().isSimplex()) ? (localFiniteElement.localBasis().order()-1) * 2
                                                 : (localFiniteElement.localBasis().order() * gridDim - 1) * 2;

  const auto element = localView.element();

  const auto& quad = Dune::QuadratureRules<double, gridDim>::rule(localFiniteElement.type(), quadOrder);

  for (size_t pt=0; pt<quad.size(); pt++) {

    // Local position of the quadrature point
    const Dune::FieldVector<double,gridDim>& quadPos = quad[pt].position();

    const auto integrationElement = element.geometry().integrationElement(quadPos);

    const auto jacobianInverse = element.geometry().jacobianInverse(quadPos);

    auto weight = quad[pt].weight() * integrationElement;

    // The derivative of the local function defined on the reference element

    // Function value at the point where we are evaluating the derivative
    // (not needed by the energy, but needed to compute the derivative)
    TargetSpace q = localInterpolationRule.evaluate(quadPos);

    // Compute the derivative
    auto referenceDerivative = localInterpolationRule.evaluateDerivative(quadPos, q);
    auto derivative = referenceDerivative * jacobianInverse;

    energy += weight * density(quadPos,q,derivative);
  }

  return energy;
}

#endif
