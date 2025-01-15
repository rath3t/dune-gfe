#ifndef DUNE_GFE_LOCALENERGY_HH
#define DUNE_GFE_LOCALENERGY_HH

#include <vector>

#include <dune/common/tuplevector.hh>

#include <dune/gfe/spaces/productmanifold.hh>

namespace Dune::GFE
{
  namespace Impl
  {
    /** \brief A class exporting container types for coefficient sets
     *
     * This generic template handles TargetSpaces that are not product manifolds.
     */
    template <class TargetSpace>
    struct LocalEnergyTypes
    {
      constexpr static bool isProductManifold = false;

      using Coefficients = std::vector<TargetSpace>;
      using CompositeCoefficients = TupleVector<std::vector<TargetSpace> >;
    };

    /** \brief A class exporting container types for coefficient sets -- specialization for product manifolds
     */
    template <class ... Factors>
    struct LocalEnergyTypes<ProductManifold<Factors...> >
    {
      constexpr static bool isProductManifold = true;

      using Coefficients = std::vector<ProductManifold<Factors...> >;
      using CompositeCoefficients = TupleVector<std::vector<Factors>... >;
    };
  }  // namespace Impl


    /** \brief Base class for energies defined by integrating over one grid element */
    template<class Basis, class TargetSpace>
    class LocalEnergy
    {
    public:
      using RT = typename TargetSpace::ctype;

      /** \brief Compute the energy
       *
       * \param localView Local view specifying the current element and the FE space there
       * \param coefficients The coefficients of a FE function on the current element
       */
      virtual RT
      energy (const typename Basis::LocalView& localView,
              const typename Impl::LocalEnergyTypes<TargetSpace>::Coefficients& coefficients) const = 0;

      /** \brief ProductManifolds: Compute the energy from coefficients in separate containers
       * for each factor
       */
      virtual RT
      energy (const typename Basis::LocalView& localView,
              const typename Impl::LocalEnergyTypes<TargetSpace>::CompositeCoefficients& coefficients) const = 0;

      /** Empty virtual default destructor
       *
       * To allow proper destruction of derived classes through a base class pointer
       */
      virtual ~LocalEnergy() = default;

    };

}  // namespace Dune::GFE

#endif  // DUNE_GFE_LOCALENERGY_HH
