#ifndef DUNE_GFE_PARALLEL_GLOBALP1MAPPER_HH
#define DUNE_GFE_PARALLEL_GLOBALP1MAPPER_HH

/** include base class functionality for the communication interface */
#include <dune/grid/common/mcmgmapper.hh>

#include <dune/gfe/parallel/globalmapper.hh>

namespace Dune::GFE
{

  template <class Basis>
  class GlobalP1Mapper : public GlobalMapper<Basis>
  {
    using GridView = typename Basis::GridView;
    using P1BasisMapper = MultipleCodimMultipleGeomTypeMapper<GridView>;

  public:
    /** \brief The integer number type used for indices */
    using typename GlobalMapper<Basis>::Index;

    GlobalP1Mapper(const typename Basis::GridView& gridView)
      : GlobalMapper<Basis>(gridView),
      p1Mapper_(gridView,mcmgVertexLayout())
    {
#if !HAVE_DUNE_PARMG
      const int dim = GridView::dimension;

      GlobalIndexSet<GridView> globalVertexIndexSet(gridView,dim);

      // total number of degrees of freedom
      this->size_ = globalVertexIndexSet.size(dim);

      // Determine
      for (auto it = gridView.template begin<0>(); it != gridView.template end<0>(); ++it)
      {
        // Loop over all vertices
        for (size_t i=0; i<it->subEntities(dim); i++)
        {
          int localIndex  = p1Mapper_.subIndex(*it, i, dim);
          int globalIndex = globalVertexIndexSet.subIndex(*it, i, dim);

          localGlobalMap_[localIndex]  = globalIndex;
        }
      }
#endif
    }

    template <class Entity>
    Index subIndex(const Entity& entity, uint i, uint codim) const
    {
      int localIndex = p1Mapper_.subIndex(entity, i, codim);
#if HAVE_DUNE_PARMG
      return this->index(localIndex);
#else
      return localGlobalMap_.find(localIndex)->second;
#endif
    }

    template <class Entity>
    bool contains(const Entity& entity, uint i, uint codim, Index& result) const
    {
      if (codim != GridView::dimension)
        return false;
      result = subIndex(entity,i,codim);
      return true;
    }

    P1BasisMapper p1Mapper_;

#if !HAVE_DUNE_PARMG
    typedef std::map<Index,Index>    IndexMap;

    /** \brief Given a local index, retrieve its index globally unique over all processes. */
    Index index(const int& localIndex) const {
      return localGlobalMap_.find(localIndex)->second;
    }

    IndexMap localGlobalMap_;
#endif
  };

}  // namespace Dune::GFE

#endif /* DUNE_GFE_PARALLEL_GLOBALP1MAPPER_HH */
