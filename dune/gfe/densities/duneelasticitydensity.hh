#ifndef DUNE_GFE_DENSITIES_DUNEELASTICITYDENSITY_HH
#define DUNE_GFE_DENSITIES_DUNEELASTICITYDENSITY_HH

#include <dune/common/fmatrix.hh>

#include <dune/gfe/densities/localdensity.hh>

#if DUNE_VERSION_GTE(DUNE_ELASTICITY, 2, 11)
#include <dune/elasticity/densities/localdensity.hh>
#else
#include <dune/elasticity/materials/localdensity.hh>
#endif

namespace Dune::GFE
{

  /** \brief Adapter for densities from the dune-elasticity module
   *
   * \tparam ElementOrIntersection The domain of the density.
   *   Can be either a grid element (i.e., a codimension-0 Entity)
   *   or an Intersection.
   * \tparam index The TargetSpace is typically a product with one factor being
   *   RealTuple. The 'index' argument selects which factor to assign the
   *   dune-elasticity density to.
   */
  template<class ElementOrIntersection, class TargetSpace, std::size_t index=0>
  class DuneElasticityDensity final
    : public LocalDensity<ElementOrIntersection,TargetSpace>
  {
    using LocalCoordinate = typename ElementOrIntersection::Geometry::LocalCoordinate;
    using ctype = typename ElementOrIntersection::Geometry::ctype;
    using field_type = typename TargetSpace::field_type;

    constexpr static auto dim = LocalCoordinate::size();
    constexpr static auto embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension;

    using ATargetSpace = typename TargetSpace::template rebind<adouble>::other;

  public:

    /** \brief Constructor with a Dune::Elasticity::LocalDensity
     */
    DuneElasticityDensity(const std::shared_ptr<Elasticity::LocalDensity<dim,field_type,ctype> >& elasticityDensity)
      : elasticityDensity_(elasticityDensity)
    {}

    /** \brief Constructor with a Dune::Elasticity::LocalDensity
     */
    DuneElasticityDensity(std::unique_ptr<Elasticity::LocalDensity<dim,field_type,ctype> >&& elasticityDensity)
      : elasticityDensity_(std::shared_ptr<Elasticity::LocalDensity<dim,field_type,ctype> >(elasticityDensity.release()))
    {}

    /** \brief Evaluate the density
     *
     * \param x The current position
     * \param value The deformation at the current position
     * \param derivative The derivative at the current position
     */
    virtual field_type operator() (const LocalCoordinate& x,
                                   const typename TargetSpace::CoordinateType& value,
                                   const FieldMatrix<field_type,embeddedBlocksize,dim>& derivative) const override
    {
      // 'derivative' is the derivative of the entire TargetSpace point x.
      // But (assuming that TargetSpace is a ProductManifold) we only want
      // the parts that belongs to the factor given by the 'index' template parameter.
      FieldMatrix<field_type,dim,dim> factorDerivative;

      static_assert(index==0, "index!=0 is not implemented");
      for (std::size_t i=0; i<dim; i++)
        factorDerivative[i] = derivative[i];

      return (*elasticityDensity_)(x, factorDerivative);
    }

    // Construct a copy of this density but using 'adouble' as the number type
    virtual std::unique_ptr<LocalDensity<ElementOrIntersection,ATargetSpace> > makeActiveDensity() const
    {
      // The active dune-elasticity density
      auto activeDensity = elasticityDensity_->makeActiveDensity();

      // Wrap it as a dune-gfe density
      return std::make_unique<DuneElasticityDensity<ElementOrIntersection,ATargetSpace,index> >(std::move(activeDensity));
    }

    /** \brief The density does not depend on the value */
    virtual bool dependsOnValue([[maybe_unused]] int factor=-1) const override
    {
      return false;
    }

    /** \brief The density does depend on the derivative */
    virtual bool dependsOnDerivative([[maybe_unused]] int factor=-1) const override
    {
      return factor==-1 || factor==index;
    }

  private:
    const std::shared_ptr<Elasticity::LocalDensity<dim,field_type,ctype> > elasticityDensity_;
  };

}  // namespace Dune::GFE

#endif    //  DUNE_GFE_DENSITIES_DUNEELASTICITYDENSITY_HH
