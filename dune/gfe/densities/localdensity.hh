#ifndef DUNE_GFE_DENSITIES_LOCALDENSITY_HH
#define DUNE_GFE_DENSITIES_LOCALDENSITY_HH

#include <dune/common/fmatrix.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>

namespace Dune::GFE {

  /** \brief A base class for energy densities to be evaluated in an integral energy
   *
   * \tparam Position The evaluation point in the integration domain
   * \tparam TargetSpace Type for the function value
   */
  template<class Position, class TargetSpace>
  class LocalDensity
  {
    using field_type = typename TargetSpace::field_type;
    using DerivativeType = FieldMatrix<field_type,TargetSpace::EmbeddedTangentVector::dimension,Position::size()>;

  public:

    /** \brief Evaluate the density for a given value and first derivative
     *
     * \param x The current position
     * \param value The value of the integrand at x
     * \param derivative The derivative of the integrand at x
     */
    virtual field_type operator() (const Position& x,
                                   const TargetSpace& value,
                                   const DerivativeType& derivative) const = 0;

    /** \brief Whether the density depends on the 'value' parameter
     *
     * If TargetSpace is a ProductManifold, then this method returns the information
     * for one factor space only.
     *
     * \param factor The factor space that is being asked about.
     *   The default value -1 means: Does any of the factors depend on the value?
     */
    virtual bool dependsOnValue(int factor=-1) const = 0;

    /** \brief Whether the density depends on the 'derivative' parameter
     *
     * If TargetSpace is a ProductManifold, then this method returns the information
     * for one factor space only.
     *
     * \param factor The factor space that is being asked about
     *   The default value -1 means: Does any of the factors depend on the derivative?
     */
    virtual bool dependsOnDerivative(int factor=-1) const = 0;
  };

}  // namespace Dune::GFE

#endif  // DUNE_GFE_DENSITIES_LOCALDENSITY_HH
