#ifndef DUNE_GFE_DENSITIES_HARMONICDENSITY_HH
#define DUNE_GFE_DENSITIES_HARMONICDENSITY_HH

#include <dune/common/fmatrix.hh>

#include <dune/gfe/densities/localdensity.hh>

namespace Dune::GFE
{
  template<class Position, class TargetSpace>
  class HarmonicDensity final
    : public LocalDensity<Position,TargetSpace>
  {
    using field_type = typename TargetSpace::field_type;

    constexpr static auto dim = Position::size();
    constexpr static auto embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension;

  public:

    /** \brief Evaluate the density
     *
     * \param x The current position
     * \param value The deformation at the current position
     * \param derivative The derivative of the deformation at the current position
     */
    virtual field_type operator() (const Position& x,
                                   const TargetSpace& value,
                                   const FieldMatrix<field_type,embeddedBlocksize,dim>& derivative) const override
    {
      return 0.5 * derivative.frobenius_norm2();
    }
  };

}  // namespace Dune:GFE

#endif
