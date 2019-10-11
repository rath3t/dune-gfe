#include "config.h"

#include <dune/grid/uggrid.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>

#include <dune/gfe/unitvector.hh>
#include <dune/gfe/harmonicenergy.hh>
#include <dune/gfe/localgeodesicfefunction.hh>

#include "multiindex.hh"
#include "valuefactory.hh"

const int dim = 2;

typedef UnitVector<double,3> TargetSpace;

using namespace Dune;

/** \brief Computes the diameter of a set */
template <class TargetSpace>
double diameter(const std::vector<TargetSpace>& v)
{
    double d = 0;
    for (size_t i=0; i<v.size(); i++)
        for (size_t j=0; j<v.size(); j++)
            d = std::max(d, TargetSpace::distance(v[i],v[j]));
    return d;
}

template <class Basis>
void testEnergy(const Basis& basis, const std::vector<TargetSpace>& coefficients)
{
    using GridView = typename Basis::GridView;
    GridView gridView = basis.gridView();

    using GeodesicInterpolationRule  = LocalGeodesicFEFunction<dim, double, typename Basis::LocalView::Tree::FiniteElement, TargetSpace>;

    HarmonicEnergy<Basis,GeodesicInterpolationRule,TargetSpace> assembler;
    std::vector<TargetSpace> rotatedCoefficients(coefficients.size());

    auto localView = basis.localView();
    localView.bind(*gridView.template begin<0>());

    for (int i=0; i<10; i++) {

        Rotation<double,3> rotation(FieldVector<double,3>(1), double(i));

        FieldMatrix<double,3,3> matrix;
        rotation.matrix(matrix);
        for (size_t j=0; j<coefficients.size(); j++) {
            FieldVector<double,3> tmp;
            matrix.mv(coefficients[j].globalCoordinates(), tmp);
            rotatedCoefficients[j] = tmp;
        }

        std::cout << "energy: " << assembler.energy(localView,
                                                    rotatedCoefficients) << std::endl;

    }

}


template <int domainDim>
void testUnitVector3d()
{
    // ////////////////////////////////////////////////////////
    //   Make a test grid consisting of a single simplex
    // ////////////////////////////////////////////////////////

    typedef UGGrid<domainDim> GridType;

    GridFactory<GridType> factory;

    FieldVector<double,dim> pos(0);
    factory.insertVertex(pos);

    for (int i=0; i<domainDim+1; i++) {
        pos = 0;
        pos[i] = 1;
        factory.insertVertex(pos);
    }

    std::vector<unsigned int> v(domainDim+1);
    for (int i=0; i<domainDim+1; i++)
        v[i] = i;
    factory.insertElement(GeometryTypes::simplex(dim), v);

    auto grid = std::unique_ptr<const GridType>(factory.createGrid());
    using GridView = typename GridType::LeafGridView;
    auto gridView = grid->leafGridView();

    // A function space basis
    using Basis = Functions::LagrangeBasis<GridView,1>;
    Basis basis(gridView);

    // //////////////////////////////////////////////////////////
    //  Test whether the energy is invariant under isometries
    // //////////////////////////////////////////////////////////

    std::vector<TargetSpace> testPoints;
    ValueFactory<TargetSpace>::get(testPoints);

    // Set up elements of S^2
    std::vector<TargetSpace> coefficients(dim+1);

    ::MultiIndex index(dim+1, testPoints.size());
    int numIndices = index.cycle();

    for (int i=0; i<numIndices; i++, ++index) {
        
        for (int j=0; j<dim+1; j++)
            coefficients[j] = testPoints[index[j]];

        // This may be overly restrictive, but see the TODO in targetspacetrsolver.cc
        if (diameter(coefficients) > TargetSpace::convexityRadius)
            continue;

        testEnergy<Basis>(basis, coefficients);
        
    }

}


int main(int argc, char** argv)
{
    MPIHelper::instance(argc, argv);

    testUnitVector3d<2>();
}
