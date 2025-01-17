#ifndef DUNE_GFE_ASSEMBLERS_DISCRETEKIRCHHOFFBENDINGENERGY_HH
#define DUNE_GFE_ASSEMBLERS_DISCRETEKIRCHHOFFBENDINGENERGY_HH

#include <dune/common/fmatrix.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/gfe/assemblers/localenergy.hh>
#include <dune/gfe/bendingisometryhelper.hh>
#include <dune/gfe/tensor3.hh>

#include <dune/localfunctions/lagrange/lagrangesimplex.hh>

/**
 * \brief Assemble the discrete Kirchhoff bending energy for a single element.
 *
 *  The energy is essentially the harmonic energy of a
 *  discrete Jacobian operator which is represented as a
 *  linear combination in a P2 finite element space.
 *  The gradient of that linear combination serves as an
 *  discrete Hessian in the energy.
 */
namespace Dune::GFE
{
  template<class Basis, class DiscreteKirchhoffFunction, class LocalForce, class TargetSpace>
  class DiscreteKirchhoffBendingEnergy
    : public Dune::GFE::LocalEnergy<Basis,TargetSpace>
  {
    // grid types
    typedef typename Basis::GridView GridView;
    typedef typename GridView::ctype DT;
    typedef typename TargetSpace::ctype RT;
    typedef typename DiscreteKirchhoffFunction::DerivativeType DerivativeType;

    // Grid dimension
    constexpr static int gridDim = GridView::dimension;

    // P2-LocalFiniteElement used to represent the discrete Jacobian on the current element.
    typedef typename Dune::LagrangeSimplexLocalFiniteElement<DT, double, gridDim, 2> P2LocalFiniteElement;

  public:
    DiscreteKirchhoffBendingEnergy(DiscreteKirchhoffFunction &localFunction)
      : localFunction_(localFunction)
    {}

    /** \brief Assemble the energy for a single element */
    virtual RT energy (const typename Basis::LocalView& localView,
                       const typename Impl::LocalEnergyTypes<TargetSpace>::CompositeCoefficients& coefficients) const override
    {
      DUNE_THROW(NotImplemented, "!");
    }


    /** \brief Assemble the energy for a single element */
    virtual RT energy (const typename Basis::LocalView& localView,
                       const std::vector<TargetSpace>& localConfiguration) const override
    {

      const auto element = localView.element();

      int quadOrder = 3;
      const auto &quad = QuadratureRules<double, gridDim>::rule(element.type(), quadOrder);

      /**
          bind the underlying Hermite basis to the current element.
          Note: The DiscreteKirchhoffFunction object stores a vector of global
          and local basis coefficients that are accessed for evaluation
          methods (linear combinations of hermite basis function and local coefficients).
          Therefore the local configuration is passed to the bind method as well since
          these coefficient vectors need to be updated with the new local configuration
          previous to evaluation.
       */
      localFunction_.bind(element,localConfiguration);

      RT bendingEnergy = 0;

      for (auto&& quadPoint : quad)
      {
        auto discreteHessian= localFunction_.evaluateDiscreteHessian(quadPoint.position());

        auto weight = quadPoint.weight() * element.geometry().integrationElement(quadPoint.position());
        bendingEnergy += weight * discreteHessian.frobenius_norm2();
      }

      return 0.5 * bendingEnergy;
    }

  private:
    DiscreteKirchhoffFunction& localFunction_;

    P2LocalFiniteElement lagrangeLFE_;
  };

}  // namespace Dune::GFE
#endif
