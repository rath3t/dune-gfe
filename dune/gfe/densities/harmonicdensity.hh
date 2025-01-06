#ifndef DUNE_GFE_DENSITIES_HARMONICDENSITY_HH
#define DUNE_GFE_DENSITIES_HARMONICDENSITY_HH

#include <dune/common/fmatrix.hh>

#include <dune/gfe/densities/localdensity.hh>

namespace Dune::GFE
{
  template<class ElementOrIntersection, class TargetSpace>
  class HarmonicDensity final
    : public LocalDensity<ElementOrIntersection,TargetSpace>
  {
    using LocalCoordinate = typename ElementOrIntersection::Geometry::LocalCoordinate;
    using field_type = typename TargetSpace::field_type;

    constexpr static auto dim = LocalCoordinate::size();
    constexpr static auto embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension;

    using ATargetSpace = typename TargetSpace::template rebind<adouble>::other;

  public:

    /** \brief Evaluate the density
     *
     * \param x The current position
     * \param value The deformation at the current position
     * \param derivative The derivative of the deformation at the current position
     */
    virtual field_type operator() (const LocalCoordinate& x,
                                   const typename TargetSpace::CoordinateType& value,
                                   const FieldMatrix<field_type,embeddedBlocksize,dim>& derivative) const override
    {
      return 0.5 * derivative.frobenius_norm2();
    }

    /** \brief Compute value, first and second derivatives of the density
     */
    virtual void derivatives(const LocalCoordinate& x,
                             const TargetSpace& value,
                             const FieldMatrix<field_type,embeddedBlocksize,dim>& derivative,
                             field_type& densityValue,
                             std::vector<field_type>& densityGradient,
                             Matrix<field_type>& densityHessian) const
    {
      // Compute value of the density
      densityValue = 0.5 * derivative.frobenius_norm2();

      // Compute gradient of the density
      std::size_t count = 0;
      for (int i=0; i<embeddedBlocksize; i++)
        densityGradient[count++] = 0;

      for (int i=0; i<derivative.rows; i++)
        for (std::size_t j=0; j<derivative.cols; j++)
          densityGradient[count++] = derivative[i][j];

      // Compute Hessian of the density
      densityHessian = 0;

      for (std::size_t i=embeddedBlocksize; i<densityHessian.N(); i++)
        densityHessian[i][i] = 1.0;
    }

    // Construct a copy of this density but using 'adouble' as the number type
    virtual std::unique_ptr<LocalDensity<ElementOrIntersection,ATargetSpace> > makeActiveDensity() const
    {
      auto result = std::make_unique<HarmonicDensity<ElementOrIntersection,ATargetSpace> >();
      if (this->elementOrIntersection_)
        result->bind(*this->elementOrIntersection_);
      return result;
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
