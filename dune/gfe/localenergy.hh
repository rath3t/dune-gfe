#ifndef DUNE_GFE_LOCALENERGY_HH
#define DUNE_GFE_LOCALENERGY_HH

#include <vector>

namespace Dune {

namespace GFE {

/** \brief Base class for energies defined by integrating over one grid element */
template<class Basis, class TargetSpace>
class LocalEnergy
{
public:

  /** \brief Compute the energy
   *
   * \param localView Local view specifying the current element and the FE space there
   * \param coefficients The coefficients of a FE function on the current element
   */
  virtual typename TargetSpace::ctype
  energy (const typename Basis::LocalView& localView,
          const std::vector<TargetSpace>& localSolution) const = 0;

  /** Empty virtual default destructor
   *
   * To allow proper destruction of derived classes through a base class pointer
   */
  virtual ~LocalEnergy() = default;

};

}  // namespace GFE

}  // namespace Dune

#endif  // DUNE_GFE_LOCALENERGY_HH

