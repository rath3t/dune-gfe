#ifndef ROD_REFINE_HH
#define ROD_REFINE_HH

#include "configuration.hh"

template <class GridType>
void globalRodRefine(GridType& grid, std::vector<Configuration>& x)
{
    // Should be a static assertion
    assert(GridType::dimension == 1);

    typedef typename GridType::template Codim<0>::LeafIterator ElementIterator;
    typedef typename GridType::template Codim<1>::LeafIterator VertexIterator;

    // /////////////////////////////////////////////////////
    //   Save leaf p1 data in a map
    // /////////////////////////////////////////////////////

    const typename GridType::Traits::LocalIdSet&    idSet    = grid.localIdSet();
    const typename GridType::Traits::LeafIndexSet& indexSet = grid.leafIndexSet();

    std::map<typename GridType::Traits::LocalIdSet::IdType, Configuration> dofMap;

    VertexIterator vIt    = grid.template leafbegin<1>();
    VertexIterator vEndIt = grid.template leafend<1>();

    for (; vIt!=vEndIt; ++vIt)
        dofMap.insert(std::make_pair(idSet.id(*vIt), x[indexSet.index(*vIt)]));



    // /////////////////////////////////////////////////////
    //   Globally refine the grid
    // /////////////////////////////////////////////////////

    grid.globalRefine(1);


    // /////////////////////////////////////////////////////
    //   Restore and interpolate the data
    // /////////////////////////////////////////////////////

    x.resize(grid.size(1));

    ElementIterator eIt    = grid.template leafbegin<0>();
    ElementIterator eEndIt = grid.template leafend<0>();

    for (; eIt!=eEndIt; ++eIt) {

        for (int i=0; i<2; i++) {

            if (dofMap.find(idSet.template subId<1>(*eIt,i)) != dofMap.end()) {

                x[indexSet.template subIndex<1>(*eIt,i)] = dofMap[idSet.template subId<1>(*eIt,i)];

            } else {
                // Interpolate
                Configuration p0 = dofMap[idSet.template subId<1>(*eIt->father(),0)];
                Configuration p1 = dofMap[idSet.template subId<1>(*eIt->father(),1)];

                x[indexSet.template subIndex<1>(*eIt,i)].r = (p0.r + p1.r);
                x[indexSet.template subIndex<1>(*eIt,i)].r *= 0.5;
                x[indexSet.template subIndex<1>(*eIt,i)].q
                    = Rotation<3,double>::interpolate(p0.q, p1.q, 0.5);

            }

        }

    }


}

#endif
