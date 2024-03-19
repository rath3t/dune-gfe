#ifndef DUNE_GFE_DENSITIES_CHIRALSKYRMIONDENSITY_HH
#define DUNE_GFE_DENSITIES_CHIRALSKYRMIONDENSITY_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/parametertree.hh>

#include <dune/gfe/densities/localdensity.hh>

namespace Dune::GFE
{

  /** \brief Energy density of certain chiral Skyrmion model
   *
   * The energy is discussed in:
   * - Christof Melcher, "Chiral skyrmions in the plane", Proc. of the Royal Society, online DOI DOI: 10.1098/rspa.2014.0394
   */
  template<class Position, class field_type>
  class ChiralSkyrmionDensity
    : public GFE::LocalDensity<Position,UnitVector<field_type,3> >
  {
    // various useful types
    using TargetSpace = UnitVector<field_type,3>;
    using Derivative = FieldMatrix<field_type, TargetSpace::embeddedDim, Position::size()>;

  public:

    ChiralSkyrmionDensity(const Dune::ParameterTree& parameters)
    {
      h_     = parameters.template get<double>("h");
      kappa_ = parameters.template get<double>("kappa");
    }

    //! Dimension of a tangent space
    constexpr static int blocksize = TargetSpace::TangentVector::dimension;

    /** \brief Evaluation with the current position, the deformation function, the deformation gradient, the rotation and the rotation gradient
     *
     * \param x The current position
     * \param value The deformation at the current position
     * \param derivative The derivative of the deformation at the current position
     */
    virtual field_type operator() (const Position& x,
                                   const TargetSpace& value,
                                   const Derivative& derivative) const override
    {
      //////////////////////////////////////////////////////////////
      //  Exchange energy (aka harmonic energy)
      //////////////////////////////////////////////////////////////

      field_type density = 0.5 * derivative.frobenius_norm2();

      //////////////////////////////////////////////////////////////
      //  Dzyaloshinskii-Moriya interaction term
      //////////////////////////////////////////////////////////////

      // derivative[a][b] contains the partial derivative of m_a in the direction x_b
      FieldVector<field_type, 3> curl = {derivative[2][1], -derivative[2][0], derivative[1][0]-derivative[0][1]};

      FieldVector<field_type, 3> v = value.globalCoordinates();

      density += kappa_ * (v * curl);

      //////////////////////////////////////////////////////////////
      //  Zeeman interaction term
      //////////////////////////////////////////////////////////////
      v[2] -= 1;   // subtract e_3
      density += 0.5 * h_ * v.two_norm2();

      return density;
    }

  private:
    field_type h_;
    field_type kappa_;
  };

}  // namespace Dune::GFE
#endif
