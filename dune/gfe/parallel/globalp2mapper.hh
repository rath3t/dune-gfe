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

#include <dune/functions/functionspacebases/pqknodalbasis.hh>

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

      auto localView = p2Mapper_.localView();
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
      auto localIndexSet = p2Mapper_.localIndexSet();
#endif

      // Determine
      for (const auto& element : elements(gridView))
      {
        localView.bind(element);
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
        localIndexSet.bind(localView);
#endif

        // Loop over all local degrees of freedom
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
        for (size_t i=0; i<localIndexSet.size(); i++)
#else
        for (size_t i=0; i<localView.size(); i++)
#endif
        {
          int codim = localView.tree().finiteElement().localCoefficients().localKey(i).codim();
          int entity   = localView.tree().finiteElement().localCoefficients().localKey(i).subEntity();

#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
          int localIndex  = localIndexSet.index(i);
#else
          auto localIndex  = localView.index(i);
#endif
          int globalIndex;
          switch (codim)
          {
            case 2:  // vertex dofs
              globalIndex = globalVertexIndex.index(element.template subEntity<2>(entity));
              break;

            case 1:  // edge dofs
              globalIndex = globalEdgeIndex.index(element.template subEntity<1>(entity)) + globalVertexIndex.size(2);
              break;

            case 0:  // element dofs
              globalIndex = globalElementIndex.index(element.template subEntity<0>(entity))
                            + globalEdgeIndex.size(1)
                            + globalVertexIndex.size(2);
              break;

            default:
              DUNE_THROW(Dune::Exception, "Impossible codimension!");
          }

          localGlobalMap_[localIndex]  = globalIndex;
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
    bool contains(const Entity& entity, uint subEntity, uint codim, Index& result) const
    {
      auto localView = p2Mapper_.localView();
      localView.bind(entity);
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
      auto localIndexSet = p2Mapper_.localIndexSet();
      localIndexSet.bind(localView);
#endif

      Index localIndex;
      bool dofFound = false;
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
      for (size_t i=0; i<localIndexSet.size(); i++)
#else
      for (size_t i=0; i<localView.size(); i++)
#endif
      {
        if (localView.tree().finiteElement().localCoefficients().localKey(i).subEntity() == subEntity
          and localView.tree().finiteElement().localCoefficients().localKey(i).codim() == codim)
        {
          dofFound = true;
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
          localIndex = localIndexSet.index(i);
#else
          localIndex = localView.index(i);
#endif
          break;
        }
      }

      if (not dofFound)
        return false;

      result = localGlobalMap_.find(localIndex)->second;
      return true;
    }

    unsigned int size() const
    {
      return size_;
    }

    Functions::PQkNodalBasis<GridView,2> p2Mapper_;

    IndexMap localGlobalMap_;

    size_t nOwnedLocalEntity_;
    size_t size_;

  };

}
#endif   // DUNE_GFE_PARALLEL_GLOBALP2MAPPER_HH
