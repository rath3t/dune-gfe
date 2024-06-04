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

    using ATargetSpace = typename TargetSpace::template rebind<adouble>::other;

  public:

    /** \brief Evaluate the density
     *
     * \param x The current position
     * \param value The deformation at the current position
     * \param derivative The derivative of the deformation at the current position
     */
    virtual field_type operator() (const Position& x,
                                   const typename TargetSpace::CoordinateType& value,
                                   const FieldMatrix<field_type,embeddedBlocksize,dim>& derivative) const override
    {
      return 0.5 * derivative.frobenius_norm2();
    }

    // Construct a copy of this density but using 'adouble' as the number type
    virtual std::unique_ptr<LocalDensity<Position,ATargetSpace> > makeActiveDensity() const
    {
      return std::make_unique<HarmonicDensity<Position,ATargetSpace> >();
    }

    /** \brief The density does not depend on the value */
    virtual bool dependsOnValue([[maybe_unused]] int factor=-1) const override
    {
      return false;
    }

    /** \brief The density depends on the derivative */
    virtual bool dependsOnDerivative([[maybe_unused]] int factor=-1) const override
    {
      return true;
    }

  };

}  // namespace Dune:GFE

#endif
