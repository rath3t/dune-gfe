#ifndef DUNE_GFE_DENSITIES_DUNEELASTICITYDENSITY_HH
#define DUNE_GFE_DENSITIES_DUNEELASTICITYDENSITY_HH

#include <dune/common/fmatrix.hh>

#include <dune/gfe/densities/localdensity.hh>

#include <dune/elasticity/materials/localdensity.hh>

namespace Dune::GFE
{

  /** \brief Adapter for densities from the dune-elasticity module
   *
   * \tparam index The TargetSpace is typically a product with one factor being
   *   RealTuple. The 'index' argument selects which factor to assign the
   *   dune-elasticity density to.
   */
  template<class Position, class TargetSpace, std::size_t index=0>
  class DuneElasticityDensity final
    : public LocalDensity<Position,TargetSpace>
  {
    using ctype = typename Position::value_type;
    using field_type = typename TargetSpace::field_type;

    constexpr static auto dim = Position::size();
    constexpr static auto embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension;

  public:

    /** \brief Constructor with a Dune::Elasticity::LocalDensity
     */
    DuneElasticityDensity(const std::shared_ptr<Elasticity::LocalDensity<dim,field_type,ctype> >& elasticityDensity)
      : elasticityDensity_(elasticityDensity)
    {}

    /** \brief Evaluate the density
     *
     * \param x The current position
     * \param value The deformation at the current position
     * \param derivative The derivative at the current position
     */
    virtual field_type operator() (const Position& x,
                                   const TargetSpace& value,
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

    /** \brief The density depends on the value */
    virtual bool dependsOnValue([[maybe_unused]] int factor=-1) const override
    {
      return false;
    }

    /** \brief The density does not depend on the derivative */
    virtual bool dependsOnDerivative([[maybe_unused]] int factor=-1) const override
    {
      return factor==-1 || factor==index;
    }

  private:
    const std::shared_ptr<Elasticity::LocalDensity<dim,field_type,ctype> > elasticityDensity_;
  };

}  // namespace Dune::GFE

#endif    //  DUNE_GFE_DENSITIES_DUNEELASTICITYDENSITY_HH
