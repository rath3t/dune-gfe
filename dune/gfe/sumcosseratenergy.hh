#ifndef DUNE_GFE_SUMCOSSERATENERGY_HH
#define DUNE_GFE_SUMCOSSERATENERGY_HH

#include <dune/elasticity/assemblers/localfestiffness.hh>

#include <dune/gfe/localenergy.hh>

namespace Dune {

namespace GFE {
template<class Basis, class TargetSpace, class field_type=double>
class SumCosseratEnergy
: public GFE::LocalEnergy<Basis, TargetSpace>
{
 // grid types
  typedef typename Basis::GridView GridView;
  typedef typename GridView::ctype ctype;
  typedef typename Basis::LocalView::Tree::FiniteElement LocalFiniteElement;
  typedef typename GridView::ctype DT;
  typedef typename TargetSpace::ctype RT;

  enum {dim=GridView::dimension};

public:

  /** \brief Constructor with elastic energy and cosserat energy
   * \param elasticEnergy The elastic energy
   * \param cosseratEnergy The cosserat energy
   */
  SumCosseratEnergy(std::shared_ptr<Elasticity::LocalEnergy<GridView,LocalFiniteElement,std::vector<Dune::FieldVector<field_type,dim> > > > elasticEnergy,
            std::shared_ptr<GFE::LocalEnergy<Basis, TargetSpace>> cosseratEnergy)

  : elasticEnergy_(elasticEnergy),
    cosseratEnergy_(cosseratEnergy)
  {}

  /** \brief Assemble the energy for a single element */
  RT energy (const typename Basis::LocalView& localView,
               const std::vector<TargetSpace>& localSolution) const
  { 
    auto&& element = localView.element();
    auto&& localFiniteElement = localView.tree().finiteElement();
    std::vector<Dune::FieldVector<field_type,dim>> localConfiguration(localSolution.size());
    for (int i = 0; i < localSolution.size(); i++) {
      localConfiguration[i] = localSolution[i].r;
    }
    return elasticEnergy_->energy(element, localFiniteElement, localConfiguration) + cosseratEnergy_->energy(localView, localSolution);
  }

private:

  std::shared_ptr<Elasticity::LocalEnergy<GridView,LocalFiniteElement,std::vector<Dune::FieldVector<field_type,dim> > > > elasticEnergy_;

  std::shared_ptr<GFE::LocalEnergy<Basis, TargetSpace> > cosseratEnergy_;
};

}  // namespace GFE

}  // namespace Dune

#endif   //#ifndef DUNE_GFE_SUMCOSSERATENERGY_HH
