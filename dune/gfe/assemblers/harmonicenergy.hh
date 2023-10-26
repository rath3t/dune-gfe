#ifndef DUNE_GFE_HARMONICENERGY_HH
#define DUNE_GFE_HARMONICENERGY_HH

#include <dune/common/fmatrix.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/gfe/assemblers/localenergy.hh>

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

        const auto jacobianInverseTransposed = element.geometry().jacobianInverseTransposed(quadPos);

        auto weight = quad[pt].weight() * integrationElement;

        // The derivative of the local function defined on the reference element
        auto referenceDerivative = localInterpolationRule.evaluateDerivative(quadPos);

        // Compute the Frobenius norm squared of the derivative.  This is the correct term
        // if both domain and target space use the metric inherited from an embedding.
        for (size_t i=0; i<jacobianInverseTransposed.N(); i++)
          for (int j=0; j<TargetSpace::embeddedDim; j++)
          {
            RT entry = 0;
            for (size_t k=0; k<jacobianInverseTransposed.M(); k++)
              entry += jacobianInverseTransposed[i][k] * referenceDerivative[j][k];
            energy += weight * entry * entry;
          }

    }

    return 0.5 * energy;
}

#endif

