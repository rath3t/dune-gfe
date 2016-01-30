#ifndef GEODESIC_FE_FUNCTION_ADAPTOR_HH
#define GEODESIC_FE_FUNCTION_ADAPTOR_HH

#include <vector>
#include <map>

#include "localgeodesicfefunction.hh"

template <class Basis, class TargetSpace>
class GeodesicFEFunctionAdaptor
{
public:
/** \brief Refine a grid globally and prolong a given geodesic finite element function
 */
template <class GridType>
static void geodesicFEFunctionAdaptor(GridType& grid, std::vector<TargetSpace>& x)
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

    P1NodalBasis<typename GridType::LeafGridView> p1Basis(grid.leafGridView());
    x.resize(grid.size(dim));

    ElementIterator eIt    = grid.template leafbegin<0>();
    ElementIterator eEndIt = grid.template leafend<0>();

    for (; eIt!=eEndIt; ++eIt) {

        // Set up a local gfe function on the father element
#if DUNE_VERSION_NEWER(DUNE_GRID,2,4)
        size_t nFatherDofs = eIt->father()->subEntities(dim);
#else
        size_t nFatherDofs = eIt->father()->template count<dim>();
#endif
        std::vector<TargetSpace> coefficients(nFatherDofs);

        for (int i=0; i<nFatherDofs; i++)
            coefficients[i] = dofMap.find(idSet.subId(*eIt->father(),i,dim))->second;

        typedef typename P1NodalBasis<typename GridType::LeafGridView>::LocalFiniteElement LocalFiniteElement;
        LocalGeodesicFEFunction<dim,double,LocalFiniteElement,TargetSpace> fatherFunction(p1Basis.getLocalFiniteElement(*eIt),
                                                                                          coefficients);

        // The embedding of this element into the father geometry
        const typename GridType::template Codim<0>::LocalGeometry& geometryInFather = eIt->geometryInFather();

#if DUNE_VERSION_NEWER(DUNE_GRID,2,4)
        size_t nDofs = eIt->subEntities(dim);
#else
        size_t nDofs = eIt->template count<dim>();
#endif
        for (int i=0; i<nDofs; i++) {

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


/** \brief Coordinate function in one variable, constant in the others

    This is used to extract the positions of the Lagrange nodes.
 */
template <int dim>
struct CoordinateFunction
    : public Dune::VirtualFunction<Dune::FieldVector<double,dim>, Dune::FieldVector<double,1> >
{
    CoordinateFunction(int d)
    : d_(d)
    {}

    void evaluate(const Dune::FieldVector<double, dim>& x, Dune::FieldVector<double,1>& out) const {
        out[0] = x[d_];
    }

    //
    int d_;
};


/** \brief Refine a grid globally and prolong a given geodesic finite element function
 *
 * \tparam order Interpolation order of the function space.  Kinda stupid that I
 *  have to provide this by hand.  Will change...
 */
template <int order>
static void higherOrderGFEFunctionAdaptor(Basis& basis,
                                          typename Basis::GridView::Grid& grid, std::vector<TargetSpace>& x)
{
    typedef typename Basis::GridView::Grid GridType;
    const int dim = GridType::dimension;

    typedef typename GridType::template Codim<0>::LeafIterator ElementIterator;

    // /////////////////////////////////////////////////////
    //   Save leaf p1 data in a map
    // /////////////////////////////////////////////////////

    const typename GridType::Traits::LocalIdSet&   idSet    = grid.localIdSet();

    typedef typename GridType::Traits::LocalIdSet::IdType IdType;
    std::map<IdType, std::vector<TargetSpace> > dofMap;

    assert(x.size() == basis.size());

    ElementIterator eIt    = grid.template leafbegin<0>();
    ElementIterator eEndIt = grid.template leafend<0>();

    for (; eIt!=eEndIt; ++eIt) {

        const typename Basis::LocalFiniteElement& lfe = basis.getLocalFiniteElement(*eIt);
        std::vector<TargetSpace> coefficients(lfe.localCoefficients().size());

        for (size_t i=0; i<lfe.localCoefficients().size(); i++)
            coefficients[i] = x[basis.index(*eIt, i)];

        IdType id = idSet.id(*eIt);
        dofMap.insert(std::make_pair(id, coefficients));

    }


    // /////////////////////////////////////////////////////
    //   Globally refine the grid
    // /////////////////////////////////////////////////////

    grid.globalRefine(1);


    // /////////////////////////////////////////////////////
    //   Restore and interpolate the data
    // /////////////////////////////////////////////////////

    basis.update(grid.leafGridView());

    x.resize(basis.size());

    for (eIt=grid.template leafbegin<0>(); eIt!=eEndIt; ++eIt) {

        const typename Basis::LocalFiniteElement& lfe = basis.getLocalFiniteElement(*eIt);

        typedef typename Dune::PQkLocalFiniteElementFactory<double,double,dim,order>::FiniteElementType FatherFiniteElementType;

        auto fatherLFE = std::unique_ptr<FatherFiniteElementType>(Dune::PQkLocalFiniteElementFactory<double,double,dim,order>::create(eIt->father()->type()));

        // Set up a local gfe function on the father element
        std::vector<TargetSpace> coefficients = dofMap[idSet.id(*eIt->father())];

        LocalGeodesicFEFunction<dim,double,typename Basis::LocalFiniteElement,TargetSpace> fatherFunction(*fatherLFE, coefficients);

        // The embedding of this element into the father geometry
        const typename GridType::template Codim<0>::LocalGeometry& geometryInFather = eIt->geometryInFather();

        // Generate position of the Lagrange nodes
        std::vector<Dune::FieldVector<double,dim> > lagrangeNodes(lfe.localBasis().size());

        for (int i=0; i<dim; i++) {
            CoordinateFunction<dim> lFunction(i);
            std::vector<Dune::FieldVector<double,1> > coordinates;
            lfe.localInterpolation().interpolate(lFunction, coordinates);

            for (size_t j=0; j<coordinates.size(); j++)
                lagrangeNodes[j][i] = coordinates[j];

        }

        for (int i=0; i<lfe.localCoefficients().size(); i++) {

            unsigned int idx = basis.index(*eIt, i);

            x[idx] = fatherFunction.evaluate(geometryInFather.global(lagrangeNodes[i]));

        }

    }

}

};

#endif
