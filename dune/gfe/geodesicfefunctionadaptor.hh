#ifndef GEODESIC_FE_FUNCTION_ADAPTOR_HH
#define GEODESIC_FE_FUNCTION_ADAPTOR_HH

#include <vector>
#include <map>

#include<dune/geometry/referenceelements.hh>

#include <dune/fufem/functionspacebases/p2nodalbasis.hh>

#include "localgeodesicfefunction.hh"

/** \brief Refine a grid globally and prolong a given geodesic finite element function
 */
template <class GridType, class TargetSpace>
void geodesicFEFunctionAdaptor(GridType& grid, std::vector<TargetSpace>& x)
{
    const int dim = GridType::dimension;

    assert(x.size() == grid.size(dim));

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

    P1NodalBasis<typename GridType::LeafGridView> p1Basis(grid.leafView());
    x.resize(grid.size(dim));

    ElementIterator eIt    = grid.template leafbegin<0>();
    ElementIterator eEndIt = grid.template leafend<0>();

    for (; eIt!=eEndIt; ++eIt) {

        // Set up a local gfe function on the father element
        std::vector<TargetSpace> coefficients(dim+1);

        for (int i=0; i<eIt->father()->template count<dim>(); i++)
            coefficients[i] = dofMap.find(idSet.subId(*eIt->father(),i,dim))->second;

        typedef typename P1NodalBasis<typename GridType::LeafGridView>::LocalFiniteElement LocalFiniteElement;
        LocalGeodesicFEFunction<dim,double,LocalFiniteElement,TargetSpace> fatherFunction(p1Basis.getLocalFiniteElement(*eIt),
                                                                                          coefficients);

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



/** \brief Refine a grid globally and prolong a given geodesic finite element function
 */
template <class GridType, class TargetSpace>
void higherOrderGFEFunctionAdaptor(GridType& grid, std::vector<TargetSpace>& x)
{
    const int dim = GridType::dimension;

    typedef typename GridType::template Codim<0>::LeafIterator ElementIterator;

    // /////////////////////////////////////////////////////
    //   Save leaf p1 data in a map
    // /////////////////////////////////////////////////////

    const typename GridType::Traits::LocalIdSet&   idSet    = grid.localIdSet();

    // DUNE ids are not unique across all codimensions, hence the following hack.   Sigh...
    typedef std::pair<typename GridType::Traits::LocalIdSet::IdType, unsigned int> IdType;
    std::map<IdType, TargetSpace> dofMap;

    typedef P2NodalBasis<typename GridType::LeafGridView,double> P2Basis;
    P2Basis p2Basis(grid.leafView());
    
    assert(x.size() == p2Basis.size());
    
    ElementIterator eIt    = grid.template leafbegin<0>();
    ElementIterator eEndIt = grid.template leafend<0>();

    for (; eIt!=eEndIt; ++eIt) {

        const typename P2Basis::LocalFiniteElement& lfe = p2Basis.getLocalFiniteElement(*eIt);
        //localCoefficients = p2Basis.getLocalFiniteElement(*eIt).localCoefficients();
        
        for (size_t i=0; i<lfe.localCoefficients().size(); i++) {
            
            IdType id = std::make_pair(idSet.subId(*eIt,
                                                   lfe.localCoefficients().localKey(i).subEntity(),
                                                   lfe.localCoefficients().localKey(i).codim()),
                                       lfe.localCoefficients().localKey(i).codim());

            unsigned int idx = p2Basis.index(*eIt, i);
            
            //std::cout << "id: (" << id.first << ", " << id.second << ")" << std::endl;
            dofMap.insert(std::make_pair(id, x[idx]));

        }
        
    }


    // /////////////////////////////////////////////////////
    //   Globally refine the grid
    // /////////////////////////////////////////////////////

    grid.globalRefine(1);


    // /////////////////////////////////////////////////////
    //   Restore and interpolate the data
    // /////////////////////////////////////////////////////

    p2Basis.update(grid.leafView());
    
    x.resize(p2Basis.size());

    for (eIt=grid.template leafbegin<0>(); eIt!=eEndIt; ++eIt) {

        const typename P2Basis::LocalFiniteElement& lfe = p2Basis.getLocalFiniteElement(*eIt);

        std::auto_ptr<typename Dune::PQkLocalFiniteElementFactory<double,double,dim,2>::FiniteElementType> fatherLFE 
            = std::auto_ptr<typename Dune::PQkLocalFiniteElementFactory<double,double,dim,2>::FiniteElementType>(Dune::PQkLocalFiniteElementFactory<double,double,dim,2>::create(eIt->type()));
        
        // Set up a local gfe function on the father element
        std::vector<TargetSpace> coefficients(fatherLFE->localCoefficients().size());

        for (int i=0; i<fatherLFE->localCoefficients().size(); i++) {

            IdType id = std::make_pair(idSet.subId(*eIt->father(),
                                                   fatherLFE->localCoefficients().localKey(i).subEntity(),
                                                   fatherLFE->localCoefficients().localKey(i).codim()),
                                       fatherLFE->localCoefficients().localKey(i).codim());

            coefficients[i] = dofMap.find(id)->second;
            
        }

        LocalGeodesicFEFunction<dim,double,typename P2Basis::LocalFiniteElement,TargetSpace> fatherFunction(*fatherLFE, coefficients);

        // The embedding of this element into the father geometry
        const typename GridType::template Codim<0>::LocalGeometry& geometryInFather = eIt->geometryInFather();

        for (int i=0; i<lfe.localCoefficients().size(); i++) {
            
            IdType id = std::make_pair(idSet.subId(*eIt,
                                                   lfe.localCoefficients().localKey(i).subEntity(),
                                                   lfe.localCoefficients().localKey(i).codim()),
                                       lfe.localCoefficients().localKey(i).codim());

            unsigned int idx = p2Basis.index(*eIt, i);
            
            if (dofMap.find(id) != dofMap.end()) {

                // If the vertex exists on the coarser level we take the value from there.
                // That should be faster and more accurate than interpolating
                x[idx] = dofMap[id];

            } else {

                // Interpolate
                const Dune::GenericReferenceElement<double,dim>& refElement = Dune::GenericReferenceElements<double,dim>::general(eIt->type());
                const Dune::FieldVector<double,dim>& pos = refElement.position(lfe.localCoefficients().localKey(i).subEntity(),
                                                                         lfe.localCoefficients().localKey(i).codim());
                x[idx] = fatherFunction.evaluate(geometryInFather.global(pos));

            }

        }

    }

}

#endif
