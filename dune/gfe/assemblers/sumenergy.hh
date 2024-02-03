#ifndef DUNE_GFE_SUMENERGY_HH
#define DUNE_GFE_SUMENERGY_HH

#include <vector>

#include <dune/gfe/assemblers/localenergy.hh>
#include <dune/gfe/assemblers/mixedlocalgeodesicfestiffness.hh>

namespace Dune::GFE {

  /**
     \brief Assembles the a sum of energies for a single element by summing up the energies of each GFE::LocalEnergy.

           This class works similarly to the class Dune::Elasticity::SumEnergy, where Dune::Elasticity::SumEnergy extends
           Dune::Elasticity::LocalEnergy and Dune::GFE::SumEnergy extends Dune::GFE::LocalEnergy.
   */

  template<class Basis, class ... TargetSpaces>
  class SumEnergy
    : public Dune::GFE::LocalEnergy<Basis, ProductManifold<TargetSpaces...> >,
      public MixedLocalGeodesicFEStiffness<Basis, ProductManifold<TargetSpaces...> >
      //Inheriting from MixedLocalGeodesicFEStiffness is hack, and will be replaced eventually; once MixedLocalGFEADOLCStiffness
      //will be removed and its functionality will be included in LocalGeodesicFEADOLCStiffness this is not needed anymore!

  {
    using TargetSpace = ProductManifold<TargetSpaces...>;

    using LocalView = typename Basis::LocalView;
    using GridView = typename LocalView::GridView;
    using DT = typename GridView::Grid::ctype;
    using RT = typename Dune::GFE::LocalEnergy<Basis,TargetSpace>::RT;

  public:

    /** \brief Default Constructor*/
    SumEnergy()
    {}

    void addLocalEnergy(std::shared_ptr<GFE::LocalEnergy<Basis,TargetSpace> > localEnergy)
    {
      localEnergies_.push_back(localEnergy);
    }

    RT energy(const typename Basis::LocalView& localView,
              const std::vector<TargetSpace>& localSolution) const override
    {
      RT sum = 0.;
      for ( const auto& localEnergy : localEnergies_ )
        sum += localEnergy->energy(localView, localSolution);

      return sum;
    }

    RT energy(const typename Basis::LocalView& localView,
              const typename Impl::LocalEnergyTypes<TargetSpace>::CompositeCoefficients& coefficients) const override
    {
      RT sum = 0.;
      for ( const auto& localEnergy : localEnergies_ )
        sum += localEnergy->energy(localView, coefficients);

      return sum;
    }

    RT energy(const typename Basis::LocalView& localView,
              const std::vector<TargetSpaces>& ... localSolution) const override
    {
      RT sum = 0.;
      for ( const auto& localEnergy : localEnergies_ )
        sum += localEnergy->energy(localView, TupleVector<std::vector<TargetSpaces>...>(localSolution ...));

      return sum;
    }

  private:
    // vector containing pointers to the local energies
    std::vector<std::shared_ptr<GFE::LocalEnergy<Basis, TargetSpace> > > localEnergies_;
  };

}  // namespace Dune::GFE

#endif   //#ifndef DUNE_GFE_SUMENERGY_HH
