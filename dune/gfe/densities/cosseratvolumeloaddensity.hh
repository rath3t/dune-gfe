#ifndef DUNE_GFE_DENSITIES_COSSERATVOLUMELOADDENSITY_HH
#define DUNE_GFE_DENSITIES_COSSERATVOLUMELOADDENSITY_HH

#include <dune/common/fmatrix.hh>

#include <dune/gfe/densities/localdensity.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>

namespace Dune::GFE
{
  template<class ElementOrIntersection, class field_type>
  class CosseratVolumeLoadDensity final
    : public GFE::LocalDensity<ElementOrIntersection, GFE::ProductManifold<RealTuple<field_type,3>,Rotation<field_type,3> > >
  {
    using LocalCoordinate = typename ElementOrIntersection::Geometry::LocalCoordinate;

    static constexpr int gridDim = LocalCoordinate::size();
    static constexpr auto dimworld = ElementOrIntersection::Geometry::coorddimension;

    // The target space with 'adouble' as the number type
    using ATargetSpace = GFE::ProductManifold<RealTuple<adouble,3>,Rotation<adouble,3> >;

    /** \brief The function implementing a volume load */
    const std::function<FieldVector<double,3>(FieldVector<double,dimworld>)> volumeLoad_;

  public:

    /** \brief Constructor with a given load density
     */
    CosseratVolumeLoadDensity(const std::function<FieldVector<double,3>(FieldVector<double,dimworld>)>& volumeLoad)
      : volumeLoad_(volumeLoad)
    {}

    /** \brief Evaluate the density
     */
    virtual field_type operator() (const LocalCoordinate& x,
                                   const typename GFE::ProductManifold<RealTuple<field_type,3>,Rotation<field_type,3> >::CoordinateType& value,
                                   const FieldMatrix<field_type,7,gridDim>& derivative) const override
    {
      field_type density = 0;

      // Value of the volume load density at the current position
      auto loadVector = volumeLoad_(this->elementOrIntersection_->geometry().global(x));

      // In this implementation, only translational dofs are affected by the volume load
      for (size_t i=0; i<loadVector.size(); i++)
        density += loadVector[i] * value[i];

      return density;
    }

    // Construct a copy of this density but using 'adouble' as the number type
    virtual std::unique_ptr<LocalDensity<ElementOrIntersection,ATargetSpace> > makeActiveDensity() const override
    {
      auto result = std::make_unique<CosseratVolumeLoadDensity<ElementOrIntersection,adouble> >(volumeLoad_);

      if (this->elementOrIntersection_)
        result->bind(*this->elementOrIntersection_);
      return result;
    }

    /** \brief The density depends on the deformation value, but not on the microrotation value
     */
    virtual bool dependsOnValue(int factor=-1) const override
    {
      return factor==-1 || factor==0;
    }

    /** \brief The density neither depends on the deformation gradient nor on the microrotation gradient
     */
    virtual bool dependsOnDerivative([[maybe_unused]] int factor=-1) const override
    {
      return false;
    }
  };

} // namespace GFE

#endif   //#ifndef DUNE_GFE_DENSITIES_COSSERATVOLUMELOADDENSITY_HH
