#ifndef DUNE_GFE_DENSITIES_CHIRALSKYRMIONDENSITY_HH
#define DUNE_GFE_DENSITIES_CHIRALSKYRMIONDENSITY_HH

#include <adolc/adouble.h>

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
  template<class ElementOrIntersection, class field_type>
  class ChiralSkyrmionDensity
    : public GFE::LocalDensity<ElementOrIntersection,UnitVector<field_type,3> >
  {
    // various useful types
    using LocalCoordinate = typename ElementOrIntersection::Geometry::LocalCoordinate;
    using TargetSpace = UnitVector<field_type,3>;
    using Derivative = FieldMatrix<field_type, TargetSpace::embeddedDim, LocalCoordinate::size()>;

    using ATargetSpace = typename TargetSpace::template rebind<adouble>::other;

  public:

    ChiralSkyrmionDensity(const Dune::ParameterTree& parameters)
    {
      h_     = parameters.template get<double>("h");
      kappa_ = parameters.template get<double>("kappa");
    }

    ChiralSkyrmionDensity(double h, double kappa)
      : h_(h), kappa_(kappa)
    {}

    //! Dimension of a tangent space
    constexpr static int blocksize = TargetSpace::TangentVector::dimension;

    /** \brief Evaluation with the current position, the deformation function, the deformation gradient, the rotation and the rotation gradient
     *
     * \param x The current position
     * \param value The deformation at the current position
     * \param derivative The derivative of the deformation at the current position
     */
    virtual field_type operator() (const LocalCoordinate& x,
                                   const typename TargetSpace::CoordinateType& value,
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

      density += kappa_ * (value * curl);

      //////////////////////////////////////////////////////////////
      //  Zeeman interaction term
      //////////////////////////////////////////////////////////////
      FieldVector<field_type, 3> v = value;
      v[2] -= 1;   // subtract e_3
      density += 0.5 * h_ * v.two_norm2();

      return density;
    }

    // Construct a copy of this density but using 'adouble' as the number type
    virtual std::unique_ptr<LocalDensity<ElementOrIntersection,ATargetSpace> > makeActiveDensity() const
    {
      return std::make_unique<ChiralSkyrmionDensity<ElementOrIntersection,adouble> >(h_, kappa_);
    }

    /** \brief The density depends on the value */
    virtual bool dependsOnValue([[maybe_unused]] int factor=-1) const override
    {
      return true;
    }

    /** \brief The density depends on the derivative */
    virtual bool dependsOnDerivative([[maybe_unused]] int factor=-1) const override
    {
      return true;
    }

  private:
    double h_;
    double kappa_;
  };

}  // namespace Dune::GFE
#endif
