#ifndef DUNE_GFE_LOCALFIRSTORDERMODEL_HH
#define DUNE_GFE_LOCALFIRSTORDERMODEL_HH

#include <dune/gfe/localenergy.hh>

namespace Dune {

namespace GFE {

template<class Basis, class TargetSpace>
class LocalFirstOrderModel
: public Dune::GFE::LocalEnergy<Basis,TargetSpace>
{
public:

    /** \brief Assemble the element gradient of the energy functional

    The default implementation in this class uses a finite difference approximation */
    virtual void assembleGradient(const typename Basis::LocalView& localView,
                                  const std::vector<TargetSpace>& solution,
                                  std::vector<typename TargetSpace::TangentVector>& gradient) const = 0;

};

}  // namespace GFE

}  // namespace Dune

#endif   // DUNE_GFE_LOCALFIRSTORDERMODEL_HH

