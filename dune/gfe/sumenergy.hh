#ifndef DUNE_GFE_SUMENERGY_HH
#define DUNE_GFE_SUMENERGY_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/fmatrixev.hh>

#include <dune/geometry/quadraturerules.hh>

#include <dune/fufem/functions/virtualgridfunction.hh>
#include <dune/fufem/boundarypatch.hh>

#include <dune/gfe/localgeodesicfestiffness.hh>
#include <dune/gfe/localfestiffness.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/realtuple.hh>

namespace Dune {

template<class GridView, class LocalFiniteElement, class field_type=double>
class SumEnergy
: public LocalFEStiffness<GridView,LocalFiniteElement,std::vector<Dune::FieldVector<field_type,3> > >
{
 // grid types
  typedef typename GridView::ctype ctype;
  typedef typename GridView::template Codim<0>::Entity Entity;

  enum {dim=GridView::dimension};

public:

  /** \brief Constructor with a set of material parameters
   * \param parameters The material parameters
   */
  SumEnergy(std::shared_ptr<LocalFEStiffness<GridView,LocalFiniteElement,std::vector<Dune::FieldVector<field_type,3> > > > a,
            std::shared_ptr<LocalFEStiffness<GridView,LocalFiniteElement,std::vector<Dune::FieldVector<field_type,3> > > > b)
  : a_(a),
    b_(b)
  {}

  /** \brief Assemble the energy for a single element */
  field_type energy (const Entity& element,
                     const LocalFiniteElement& localFiniteElement,
                     const std::vector<Dune::FieldVector<field_type,dim> >& localConfiguration) const
  {
    return a_->energy(element, localFiniteElement, localConfiguration)
         + b_->energy(element, localFiniteElement, localConfiguration);
  }

private:

  std::shared_ptr<LocalFEStiffness<GridView,LocalFiniteElement,std::vector<Dune::FieldVector<field_type,3> > > > a_;

  std::shared_ptr<LocalFEStiffness<GridView,LocalFiniteElement,std::vector<Dune::FieldVector<field_type,3> > > > b_;
};

}

#endif   //#ifndef DUNE_GFE_SUMENERGY_HH


