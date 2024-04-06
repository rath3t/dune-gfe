#ifndef DUNE_GFE_ASSEMBLERS_LOCALFIRSTORDERMODEL_HH
#define DUNE_GFE_ASSEMBLERS_LOCALFIRSTORDERMODEL_HH

#include <dune/gfe/assemblers/localenergy.hh>

namespace Dune::GFE
{
  /** \brief Base class for problems that have an energy and a first derivative
   */
  template<class Basis, class TargetSpace>
  class LocalFirstOrderModel
    : public Dune::GFE::LocalEnergy<Basis,TargetSpace>
  {
  public:

    /** \brief Assemble the element gradient of the energy functional */
    virtual void assembleGradient(const typename Basis::LocalView& localView,
                                  const typename Impl::LocalEnergyTypes<TargetSpace>::Coefficients& coefficients,
                                  std::vector<double>& gradient) const = 0;

  };

}  // namespace Dune

#endif   // DUNE_GFE_LOCALFIRSTORDERMODEL_HH
