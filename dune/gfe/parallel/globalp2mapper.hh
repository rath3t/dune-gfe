#ifndef DUNE_GFE_PARALLEL_GLOBALP2MAPPER_HH
#define DUNE_GFE_PARALLEL_GLOBALP2MAPPER_HH

/** \brief Include standard header files. */
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <utility>

/** include base class functionality for the communication interface */
#include <dune/grid/common/datahandleif.hh>

// Include Dune header files
#include <dune/common/version.hh>

/** include parallel capability */
#if HAVE_MPI
  #include <dune/common/parallel/mpihelper.hh>
#endif

namespace Dune {

  template <class GridView>
  class GlobalP2Mapper
  {
  public:

    /** \brief The integer number type used for indices */
    typedef typename GridView::IndexSet::IndexType Index;

    typedef std::map<Index,Index>    IndexMap;

    GlobalP2Mapper(const GridView& gridView)
    : p2Mapper_(gridView)
    {
      static_assert(GridView::dimension==2, "Only implemented for two-dimensional grids");

      GlobalIndexSet<GridView> globalVertexIndex(gridView,2);
      GlobalIndexSet<GridView> globalEdgeIndex(gridView,1);
      GlobalIndexSet<GridView> globalElementIndex(gridView,0);

      // total number of degrees of freedom
      size_ = globalVertexIndex.size(2) + globalEdgeIndex.size(1) + globalElementIndex.size(0);

      // Determine
      for (auto it = gridView.template begin<0>(); it != gridView.template end<0>(); ++it)
      {
        // Loop over all vertices
#if DUNE_VERSION_NEWER(DUNE_GRID,2,4)
        for (size_t i=0; i<it->subEntities(2); i++)
#else
        for (size_t i=0; i<it->template count<2>(); i++)
#endif
        {
          //int localIndex  = globalVertexIndex.localIndex (*it->template subEntity<2>(i));
          int localIndex  = p2Mapper_.map(*it, i, 2);
          int globalIndex = globalVertexIndex.index(*it->template subEntity<2>(i));

          localGlobalMap_[localIndex]  = globalIndex;
          globalLocalMap_[globalIndex] = localIndex;
        }

        // Loop over all edges
#if DUNE_VERSION_NEWER(DUNE_GRID,2,4)
        for (size_t i=0; i<it->subEntities(1); i++)
#else
        for (size_t i=0; i<it->template count<1>(); i++)
#endif
        {
          //int localIndex  = globalEdgeIndex.localIndex (*it->template subEntity<1>(i)) + gridView.size(2);
          int localIndex  = p2Mapper_.map(*it, i, 1);
          int globalIndex = globalEdgeIndex.index(*it->template subEntity<1>(i)) + globalVertexIndex.size(2);

          localGlobalMap_[localIndex]  = globalIndex;
          globalLocalMap_[globalIndex] = localIndex;
        }

        // One element degree of freedom for quadrilaterals
        if (not it->type().isQuadrilateral())
          DUNE_THROW(Dune::NotImplemented, "for non-quadrilaterals");

        if (it->type().isQuadrilateral())
        {
          //int localIndex  = globalEdgeIndex.localIndex (*it->template subEntity<1>(i)) + gridView.size(2);
          int localIndex  = p2Mapper_.map(*it, 0, 0);
          int globalIndex = globalElementIndex.index(*it->template subEntity<0>(0))
                            + globalEdgeIndex.size(1)
                            + globalVertexIndex.size(2);

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
      int localIndex = p2Mapper_.map(entity, i, codim);
      return localGlobalMap_.find(localIndex)->second;
    }

    template <class Entity>
    bool contains(const Entity& entity, uint i, uint codim, Index& result) const
    {
      Index localIndex;
      if (not p2Mapper_.contains(entity, i, codim,localIndex))
        return false;
      result = localGlobalMap_.find(localIndex)->second;
      return true;
    }

    Index localIndex(const int& globalIndex) const {
      return globalLocalMap_.find(globalIndex)->second;
    }

    unsigned int size() const
    {
      return size_;
    }

    P2BasisMapper<GridView> p2Mapper_;

    IndexMap localGlobalMap_;
    IndexMap globalLocalMap_;

    size_t nOwnedLocalEntity_;
    size_t size_;

  };

}
#endif /* GLOBALUNIQUEINDEX_HH_ */
