#ifndef DUNE_GFE_DENSITIES_LOCALDENSITY_HH
#define DUNE_GFE_DENSITIES_LOCALDENSITY_HH

#include <dune/common/fmatrix.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>

namespace Dune::GFE {

/** \brief A base class for energy densities to be evaluated in an integral energy
 *
 * \tparam field_type type of the gradient entries
 * \tparam ctype type of the coordinates
 */
template<int dim, class field_type = double, class ctype = double>
class LocalDensity
{
private:
  static const int embeddedDim = Rotation<field_type,dim>::embeddedDim;
public:

  /** \brief Evaluation with the current position, the deformation function, the deformation gradient, the rotation and the rotation gradient
   *
   * \param x The current position
   * \param deformationValue The deformation at the current position
   * \param deformationDerivative The derivative of the deformation at the current position
   * \param orientationValue The orientation at the current position
   * \param orientationDerivative The derivative of the orientation at the current position
   */
  virtual field_type operator() (const FieldVector<ctype,dim>& x,
                                 const RealTuple<field_type,dim>& deformation,
                                 const FieldMatrix<field_type,dim,dim>& gradient,
                                 const Rotation<field_type,dim>& rotation,
                                 const FieldMatrix<field_type, embeddedDim, dim>& rotationGradient) const = 0;

};

}  // namespace Dune::GFE

#endif  // DUNE_GFE_DENSITIES_LOCALDENSITY_HH
