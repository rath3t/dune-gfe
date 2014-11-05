#ifndef DUNE_GFE_PARALLEL_GLOBALP1MAPPER_HH
#define DUNE_GFE_PARALLEL_GLOBALP1MAPPER_HH

/** \brief Include standard header files. */
#include <vector>
#include <iostream>
#include <map>
#include <utility>

/** include base class functionality for the communication interface */
#include <dune/grid/common/datahandleif.hh>
#include <dune/grid/common/mcmgmapper.hh>

// Include Dune header files
#include <dune/common/version.hh>

/** include parallel capability */
#if HAVE_MPI
  #include <dune/common/parallel/mpihelper.hh>
#endif

namespace Dune {

  template <class GridView>
  class GlobalP1Mapper
  {
    typedef MultipleCodimMultipleGeomTypeMapper<GridView, MCMGVertexLayout> P1BasisMapper;

  public:

    /** \brief The integer number type used for indices */
    typedef typename GridView::IndexSet::IndexType Index;

    typedef std::map<Index,Index>    IndexMap;

    GlobalP1Mapper(const GridView& gridView)
    : p1Mapper_(gridView)
    {
      const int dim = GridView::dimension;

      GlobalIndexSet<GridView> globalVertexIndexSet(gridView,dim);

      // total number of degrees of freedom
      size_ = globalVertexIndexSet.size(dim);

      // Determine
      for (auto it = gridView.template begin<0>(); it != gridView.template end<0>(); ++it)
      {
        // Loop over all vertices
#if DUNE_VERSION_NEWER(DUNE_GRID,2,4)
        for (size_t i=0; i<it->subEntities(dim); i++)
#else
        for (size_t i=0; i<it->template count<dim>(); i++)
#endif
        {
          int localIndex  = p1Mapper_.map(*it, i, dim);
          int globalIndex = globalVertexIndexSet.subIndex(*it, i, dim);

          localGlobalMap_[localIndex]  = globalIndex;
          globalLocalMap_[globalIndex] = localIndex;
        }

      }

    }

    /** \brief Given a local index, retrieve its index globally unique over all processes. */
    Index index(const int& localIndex) const {
      return localGlobalMap_.find(localIndex)->second;
    }

    template <class Entity>
    Index subIndex(const Entity& entity, uint i, uint codim) const
    {
      int localIndex = p1Mapper_.map(entity, i, codim);
      return localGlobalMap_.find(localIndex)->second;
    }

    Index localIndex(const int& globalIndex) const {
      return globalLocalMap_.find(globalIndex)->second;
    }

    unsigned int size() const
    {
      return size_;
    }

    P1BasisMapper p1Mapper_;

    IndexMap localGlobalMap_;
    IndexMap globalLocalMap_;

    size_t size_;

  };

}
#endif /* GLOBALUNIQUEINDEX_HH_ */
