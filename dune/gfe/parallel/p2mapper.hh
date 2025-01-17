// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_GFE_PARALLEL_P2MAPPER_HH
#define DUNE_GFE_PARALLEL_P2MAPPER_HH

#include <dune/geometry/type.hh>
#include <dune/common/typetraits.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>


namespace Dune::GFE
{

  /** \brief Mimic a dune-grid mapper for a P2 space, using the dune-functions dof ordering of such a space
   */
  template<class GridView>
  class P2BasisMapper
  {
    typedef typename GridView::Grid::template Codim<0>::Entity Element;
  public:

    typedef typename Dune::Functions::LagrangeBasis<GridView,2>::MultiIndex::value_type Index;

    P2BasisMapper(const GridView& gridView)
      : p2Basis_(gridView)
    {}

    std::size_t size() const
    {
      return p2Basis_.size();
    }

    template <class Entity>
    bool contains(const Entity& entity, uint subEntity, uint codim, Index& result) const
    {
      auto localView = p2Basis_.localView();
      localView.bind(entity);

      for (size_t i=0; i<localView.size(); i++)
      {
        if (localView.tree().finiteElement().localCoefficients().localKey(i).subEntity() == subEntity
            and localView.tree().finiteElement().localCoefficients().localKey(i).codim() == codim)
        {
          result = localView.index(i);
          return true;
        }
      }

      return false;
    }

    Dune::Functions::LagrangeBasis<GridView,2> p2Basis_;
  };

}  // namespace Dune::GFE

#endif
