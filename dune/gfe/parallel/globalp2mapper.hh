#ifndef DUNE_GFE_PARALLEL_GLOBALP2MAPPER_HH
#define DUNE_GFE_PARALLEL_GLOBALP2MAPPER_HH

#include <dune/functions/functionspacebases/lagrangebasis.hh>

#include <dune/gfe/parallel/globalmapper.hh>

namespace Dune {

  template <class Basis>
  class GlobalP2Mapper : public Dune::GlobalMapper<Basis>
  {
    using GridView = typename Basis::GridView;
    using P2BasisMapper = Functions::LagrangeBasis<GridView,2>;

  public:
    /** \brief The integer number type used for indices */
    using typename GlobalMapper<Basis>::Index;

    GlobalP2Mapper(const typename Basis::GridView& gridView) :
      GlobalMapper<Basis>(gridView),
      p2Mapper_(gridView)
    {
#if !HAVE_DUNE_PARMG
      static_assert(GridView::dimension<=2, "Only implemented for one- and two-dimensional grids");

      if (gridView.size(GeometryTypes::triangle)>1)
        DUNE_THROW(NotImplemented, "GlobalP2Mapper only works for quad grids!");

      GlobalIndexSet<GridView> globalVertexIndex(gridView,GridView::dimension);
      GlobalIndexSet<GridView> globalElementIndex(gridView,0);

      auto localView = p2Mapper_.localView();

      if (GridView::dimension==1)
      {
        // total number of degrees of freedom
        this->size_ = globalVertexIndex.size(GridView::dimension) + globalElementIndex.size(0);

        // Determine
        for (const auto& element : elements(gridView))
        {
          localView.bind(element);

          // Loop over all local degrees of freedom
          for (size_t i=0; i<localView.size(); i++)
          {
            int codim = localView.tree().finiteElement().localCoefficients().localKey(i).codim();
            int entity   = localView.tree().finiteElement().localCoefficients().localKey(i).subEntity();

            auto localIndex  = localView.index(i);
            int globalIndex;
            switch (codim)
            {
            case 1 :   // vertex dofs
              globalIndex = globalVertexIndex.index(element.template subEntity<1>(entity));
              break;

            case 0 :   // element dofs
              globalIndex = globalElementIndex.index(element.template subEntity<0>(entity))
                            + globalVertexIndex.size(1);
              break;

            default :
              DUNE_THROW(Dune::Exception, "Impossible codimension!");
            }

            localGlobalMap_[localIndex]  = globalIndex;
          }
        }

      }
      else
      {
        GlobalIndexSet<GridView> globalEdgeIndex(gridView,1);

        // total number of degrees of freedom
        this->size_ = globalVertexIndex.size(2) + globalEdgeIndex.size(1) + globalElementIndex.size(0);

        // Determine
        for (const auto& element : elements(gridView))
        {
          localView.bind(element);

          // Loop over all local degrees of freedom
          for (size_t i=0; i<localView.size(); i++)
          {
            int codim = localView.tree().finiteElement().localCoefficients().localKey(i).codim();
            int entity   = localView.tree().finiteElement().localCoefficients().localKey(i).subEntity();

            auto localIndex  = localView.index(i);
            int globalIndex;
            switch (codim)
            {
            case 2 :   // vertex dofs
              globalIndex = globalVertexIndex.index(element.template subEntity<GridView::dimension>(entity));
              break;

            case 1 :   // edge dofs
              globalIndex = globalEdgeIndex.index(element.template subEntity<1>(entity)) + globalVertexIndex.size(2);
              break;

            case 0 :   // element dofs
              globalIndex = globalElementIndex.index(element.template subEntity<0>(entity))
                            + globalEdgeIndex.size(1)
                            + globalVertexIndex.size(2);
              break;

            default :
              DUNE_THROW(Dune::Exception, "Impossible codimension!");
            }

            localGlobalMap_[localIndex]  = globalIndex;
          }
        }
      }
#endif
    }

    template <class Entity>
    Index subIndex(const Entity& entity, uint i, uint codim) const
    {
      int localIndex = p2Mapper_.map(entity, i, codim);
#if HAVE_DUNE_PARMG
      return this->index(localIndex);
#else
      return localGlobalMap_.find(localIndex)->second;
#endif
    }

    template <class Entity>
    bool contains(const Entity& entity, uint subEntity, uint codim, Index& result) const
    {
      auto localView = p2Mapper_.localView();
      localView.bind(entity);

      Index localIndex;
      bool dofFound = false;
      for (size_t i=0; i<localView.size(); i++)
      {
        if (localView.tree().finiteElement().localCoefficients().localKey(i).subEntity() == subEntity
            and localView.tree().finiteElement().localCoefficients().localKey(i).codim() == codim)
        {
          dofFound = true;
          localIndex = localView.index(i);
          break;
        }
      }

      if (not dofFound)
        return false;

#if HAVE_DUNE_PARMG
      result = this->index(localIndex);
#else
      result = localGlobalMap_.find(localIndex)->second;
#endif
      return true;
    }

#if !HAVE_DUNE_PARMG
    /** \brief Given a local index, retrieve its index globally unique over all processes. */
    Index index(const int& localIndex) const {
      return localGlobalMap_.find(localIndex)->second;
    }

    std::map<Index,Index> localGlobalMap_;
#endif

    P2BasisMapper p2Mapper_;
  };
}
#endif   // DUNE_GFE_PARALLEL_GLOBALP2MAPPER_HH
