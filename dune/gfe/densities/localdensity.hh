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

    /** \brief Evaluation with the current position, the deformation function, the deformation gradient, the rotation and the rotation gradient
     *
     * \param x The current position
     * \param value The deformation at the current position
     * \param derivative The derivative of the deformation at the current position
     */
    virtual field_type operator() (const Position& x,
                                   const TargetSpace& value,
                                   const DerivativeType& derivative) const = 0;

  };

}  // namespace Dune::GFE

#endif  // DUNE_GFE_DENSITIES_LOCALDENSITY_HH
