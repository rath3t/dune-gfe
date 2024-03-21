#ifndef DUNE_GFE_PARALLEL_MAPPERFACTORY_HH
#define DUNE_GFE_PARALLEL_MAPPERFACTORY_HH

#include <dune/grid/utility/globalindexset.hh>
#include <dune/gfe/parallel/globalmapper.hh>
#include <dune/gfe/parallel/globalp1mapper.hh>
#include <dune/gfe/parallel/globalp2mapper.hh>
#include <dune/gfe/parallel/p2mapper.hh>

namespace Dune::GFE
{

  /** \brief Assign GlobalMapper and LocalMapper types to a dune-functions function space basis */
  template <typename Basis>
  struct MapperFactory
  {};

  /** \brief Specialization for LagrangeBasis<1> */
  template <typename GridView>
  struct MapperFactory<Functions::LagrangeBasis<GridView,1> >
  {
    typedef Dune::GlobalP1Mapper<Functions::LagrangeBasis<GridView,1> > GlobalMapper;
    typedef Dune::MultipleCodimMultipleGeomTypeMapper<GridView> LocalMapper;
    static LocalMapper createLocalMapper(const GridView& gridView)
    {
      return LocalMapper(gridView, Dune::mcmgVertexLayout());
    }
  };

  template <typename GridView>
  struct MapperFactory<Dune::Functions::LagrangeBasis<GridView,2> >
  {
    typedef Dune::GlobalP2Mapper<Functions::LagrangeBasis<GridView,2> > GlobalMapper;
    typedef P2BasisMapper<GridView> LocalMapper;
    static LocalMapper createLocalMapper(const GridView& gridView)
    {
      return LocalMapper(gridView);
    }
  };

  /** \brief Specialization for LagrangeBasis<3> */
  template <typename GridView>
  struct MapperFactory<Dune::Functions::LagrangeBasis<GridView,3> >
  {
    // Error: we don't currently have a global P3 mapper
  };

}  // namespace Dune::GFE

#endif
