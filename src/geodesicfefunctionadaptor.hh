#ifndef GEODESIC_FE_FUNCTION_ADAPTOR_HH
#define GEODESIC_FE_FUNCTION_ADAPTOR_HH

#include <vector>
#include <map>

#include "localgeodesicfefunction.hh"

template <class GridType, class TargetSpace>
void geodesicFEFunctionAdaptor(GridType& grid, std::vector<TargetSpace>& x)
{
    const int dim = GridType::dimension;

    typedef typename GridType::template Codim<0>::LeafIterator ElementIterator;
    typedef typename GridType::template Codim<dim>::LeafIterator VertexIterator;

    // /////////////////////////////////////////////////////
    //   Save leaf p1 data in a map
    // /////////////////////////////////////////////////////

    const typename GridType::Traits::LocalIdSet&   idSet    = grid.localIdSet();
    const typename GridType::Traits::LeafIndexSet& indexSet = grid.leafIndexSet();

    std::map<typename GridType::Traits::LocalIdSet::IdType, TargetSpace> dofMap;

    VertexIterator vIt    = grid.template leafbegin<dim>();
    VertexIterator vEndIt = grid.template leafend<dim>();

    for (; vIt!=vEndIt; ++vIt)
        dofMap.insert(std::make_pair(idSet.id(*vIt), x[indexSet.index(*vIt)]));



    // /////////////////////////////////////////////////////
    //   Globally refine the grid
    // /////////////////////////////////////////////////////

    grid.globalRefine(1);


    // /////////////////////////////////////////////////////
    //   Restore and interpolate the data
    // /////////////////////////////////////////////////////

    x.resize(grid.size(dim));

    ElementIterator eIt    = grid.template leafbegin<0>();
    ElementIterator eEndIt = grid.template leafend<0>();

    for (; eIt!=eEndIt; ++eIt) {

        // Set up a local gfe function on the father element
        std::vector<TargetSpace> coefficients(eIt->father()->template count<dim>());

        for (int i=0; i<eIt->father()->template count<dim>(); i++)
            coefficients[i] = dofMap.find(idSet.subId(*eIt->father(),i,dim))->second;

        LocalGeodesicFEFunction<dim,double,TargetSpace> fatherFunction(coefficients);

        // The embedding of this element into the father geometry
        const typename GridType::template Codim<0>::LocalGeometry& geometryInFather = eIt->geometryInFather();

        for (int i=0; i<eIt->template count<dim>(); i++) {

            if (dofMap.find(idSet.subId(*eIt,i,dim)) != dofMap.end()) {

                // If the vertex exists on the coarser level we take the value from there.
                // That should be faster and more accurate than interpolating
                x[indexSet.subIndex(*eIt,i,dim)] = dofMap[idSet.subId(*eIt,i,dim)];

            } else {

                // Interpolate
                x[indexSet.subIndex(*eIt,i,dim)] = fatherFunction.evaluate(geometryInFather.corner(i));

            }

        }

    }


}

#endif
