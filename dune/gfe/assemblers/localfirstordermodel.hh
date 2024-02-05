#ifndef DUNE_GFE_ASSEMBLERS_LOCALFIRSTORDERMODEL_HH
#define DUNE_GFE_ASSEMBLERS_LOCALFIRSTORDERMODEL_HH

#include <dune/gfe/assemblers/localenergy.hh>

namespace Dune::GFE
{
  namespace Impl
  {
    /** \brief A class exporting container types for sets of tangent vectors
     *
     * This generic template handles TargetSpaces that are not product manifolds.
     */
    template <class TargetSpace>
    struct LocalFirstOrderModelTypes
      : public LocalEnergyTypes<TargetSpace>
    {
      using Gradient = std::vector<typename TargetSpace::TangentVector>;
    };

    /** \brief A class exporting container types for sets of tangent vectors -- specialization for product manifolds
     */
    template <class ... Factors>
    struct LocalFirstOrderModelTypes<ProductManifold<Factors...> >
      : public LocalEnergyTypes<ProductManifold<Factors...> >
    {
      using Gradient = std::vector<typename ProductManifold<Factors...>::TangentVector>;
    };

  }  // namespace Impl


  /** \brief Base class for problems that have an energy and a first derivative
   */
  template<class Basis, class TargetSpace>
  class LocalFirstOrderModel
    : public Dune::GFE::LocalEnergy<Basis,TargetSpace>
  {
  public:

    /** \brief Assemble the element gradient of the energy functional */
    virtual void assembleGradient(const typename Basis::LocalView& localView,
                                  const typename Impl::LocalFirstOrderModelTypes<TargetSpace>::Coefficients& coefficients,
                                  typename Impl::LocalFirstOrderModelTypes<TargetSpace>::Gradient& gradient) const = 0;

  };

}  // namespace Dune

#endif   // DUNE_GFE_LOCALFIRSTORDERMODEL_HH
