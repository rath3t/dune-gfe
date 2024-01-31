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
  public:

    /** \brief Evaluation with the current position, the deformation function, the deformation gradient, the rotation and the rotation gradient
     *
     * \param x The current position
     * \param deformationValue The deformation at the current position
     * \param deformationDerivative The derivative of the deformation at the current position
     * \param orientationValue The orientation at the current position
     * \param orientationDerivative The derivative of the orientation at the current position
     */
    virtual field_type operator() (const Position& x,
                                   const RealTuple<field_type,3>& deformation,
                                   const FieldMatrix<field_type,3,3>& gradient,
                                   const Rotation<field_type,3>& rotation,
                                   const FieldMatrix<field_type, 4, 3>& rotationGradient) const = 0;

  };

}  // namespace Dune::GFE

#endif  // DUNE_GFE_DENSITIES_LOCALDENSITY_HH
