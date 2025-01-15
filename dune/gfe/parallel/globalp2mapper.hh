#ifndef DUNE_GFE_PARALLEL_GLOBALP2MAPPER_HH
#define DUNE_GFE_PARALLEL_GLOBALP2MAPPER_HH

#include <memory>

#include <dune/functions/functionspacebases/lagrangebasis.hh>

#include <dune/gfe/parallel/globalmapper.hh>

namespace Dune::GFE
{

  template <class Basis>
  class GlobalP2Mapper : public GlobalMapper<Basis>
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
      if (gridView.size(GeometryTypes::simplex(GridView::dimension))>=1)
        DUNE_THROW(NotImplemented, "GlobalP2Mapper only works for grids without simplex elements!");

      // Indexed by *co*dimension!
      std::array<std::unique_ptr<GlobalIndexSet<GridView> >, GridView::dimension+1> globalIndexSets;

      for (int i=0; i<=GridView::dimension; ++i)
        globalIndexSets[i] = std::make_unique<GlobalIndexSet<GridView> >(gridView, i);

      // total number of degrees of freedom
      this->size_ = 0;
      for (int i=0; i<=GridView::dimension; ++i)
        this->size_ += globalIndexSets[i]->size(i);

      auto localView = p2Mapper_.localView();

      // Loop over all elements, as a means to loop over all degrees of freedom
      for (const auto& element : elements(gridView))
      {
        localView.bind(element);

        // Loop over all local degrees of freedom
        for (size_t i=0; i<localView.size(); i++)
        {
          auto codim = localView.tree().finiteElement().localCoefficients().localKey(i).codim();
          auto entity = localView.tree().finiteElement().localCoefficients().localKey(i).subEntity();

          auto localIndex  = localView.index(i);

          // Compute a global index for this degree of freedom
          int globalIndex = 0;

          for (int j=codim+1; j<=GridView::dimension; ++j)
            globalIndex += globalIndexSets[j]->size(j);

          globalIndex += globalIndexSets[codim]->subIndex(element, entity, codim);

          localGlobalMap_[localIndex] = globalIndex;
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

}  // namespace Dune::GFE

#endif   // DUNE_GFE_PARALLEL_GLOBALP2MAPPER_HH
