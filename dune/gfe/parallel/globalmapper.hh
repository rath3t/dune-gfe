#ifndef DUNE_GFE_PARALLEL_GLOBALMAPPER_HH
#define DUNE_GFE_PARALLEL_GLOBALMAPPER_HH

#if HAVE_DUNE_PARMG
#include <dune/parmg/parallel/dofmap.hh>
#include <dune/parmg/parallel/globaldofindex.hh>
#include <dune/parmg/parallel/datahandle.hh>
#endif

namespace Dune {

  template <class Basis>
  class GlobalMapper
  {
  public:
    using GridView = typename Basis::GridView;

    /** \brief The integer number type used for indices */
    using Index = typename GridView::IndexSet::IndexType;
#if HAVE_DUNE_PARMG
    typedef Dune::ParMG::EntitiesToDofsMap<Basis> DofMap;
    typedef Dune::ParMG::EntityMasterRank<typename Basis::GridView> Master;
    typedef std::vector<char> DofMaster;

    GlobalMapper(const Basis& basis)
    {
      DofMap dofmap = Dune::ParMG::entitiesToDofsMap(&basis);
      Master m = Dune::ParMG::entityMasterRank(basis.gridView(), [&](int, int codim) -> bool {
              return dofmap.codimSet().test(codim);
            });

      globalDof_ = globalDof(basis,m, dofmap);

      // total number of degrees of freedom
      size_ = globalDof_.size;
    }
#else
    GlobalMapper(const Basis& basis)
    {
      // Total number of degrees of freedom
      size_ = basis.size();
    }
#endif

    /** \brief Given a local index, retrieve its index globally unique over all processes. */
    Index index(const int& localIndex) const {
#if HAVE_DUNE_PARMG
      return globalDof_.globalDof[localIndex];
#else
      return localIndex;
#endif
    }

    std::size_t size() const
    {
      return size_;
    }

#if HAVE_DUNE_PARMG
    ParMG::GlobalDof globalDof_;
#endif

    std::size_t size_;
  };

}
#endif   // DUNE_GFE_PARALLEL_GLOBALMAPPER_HH
