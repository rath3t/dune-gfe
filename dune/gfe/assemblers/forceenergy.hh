#ifndef DUNE_GFE_FORCEENERGY_HH
#define DUNE_GFE_FORCEENERGY_HH

#include <dune/common/fmatrix.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/gfe/assemblers/localenergy.hh>
#include <dune/gfe/bendingisometryhelper.hh>


/**
 * \brief Assemble the energy of a force term (force times deformation) for a single element.
 */
namespace Dune::GFE
{
  template<class Basis, class LocalDiscreteKirchhoffFunction, class LocalForce, class TargetSpace>
  class ForceEnergy
    : public Dune::GFE::LocalEnergy<Basis,TargetSpace>
  {
    // grid types
    typedef typename Basis::GridView GridView;
    typedef typename GridView::ctype DT;
    typedef typename TargetSpace::ctype RT;

    // Grid dimension
    constexpr static int gridDim = GridView::dimension;

  public:
    ForceEnergy(LocalDiscreteKirchhoffFunction &localFunction,
                LocalForce &localForce)
      : localFunction_(localFunction),
      localForce_(localForce)
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
      RT energy = 0;

      int quadOrder = 3;

      const auto &quad = QuadratureRules<double, gridDim>::rule(GeometryTypes::simplex(gridDim), quadOrder);

      const auto element = localView.element();

      /** bind the local force to the current element */
      localForce_.bind(element);

      /**
          bind the underlying Hermite basis to the current element.
          Note: The DiscreteKirchhoffFunction object stores a vector of global
          and local basis coefficients that are accessed for evaluation
          methods (linear comibnations of hermite basis function and local coefficients).
          Therefore the local configuration is passed to the bind method as well since
          these coefficient vectors need to be updated with the new local configuration
          previous to evaluation.
       */
      localFunction_.bind(element,localConfiguration);

      /**
       * @brief Compute contribution of the force Term
       *
       * Integrate the scalar product of the force
       * with the local deformation function 'localFunction_'
       * over the current element.
       */
      RT forceTermEnergy = 0;

      for (auto&& quadPoint : quad)
      {
        auto deformationValue = localFunction_.evaluate(quadPoint.position());
        auto forceValue = localForce_(quadPoint.position());
        auto weight = quadPoint.weight() * element.geometry().integrationElement(quadPoint.position());
        forceTermEnergy += deformationValue.globalCoordinates() * forceValue * weight;
      }

      return forceTermEnergy;
    }

  private:
    LocalDiscreteKirchhoffFunction& localFunction_;

    LocalForce& localForce_;

  };

}  // namespace Dune::GFE
#endif
