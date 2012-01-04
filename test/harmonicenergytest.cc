#include "config.h"

#include <dune/grid/uggrid.hh>

#include <dune/localfunctions/lagrange/pqkfactory.hh>

#include <dune/gfe/unitvector.hh>
#include <dune/gfe/harmonicenergystiffness.hh>

#include "multiindex.hh"
#include "valuefactory.hh"

const int dim = 2;

typedef UnitVector<double,3> TargetSpace;

using namespace Dune;

template <class GridType>
void testEnergy(const GridType* grid, const std::vector<TargetSpace>& coefficients) {

    PQkLocalFiniteElementCache<double,double,GridType::dimension,1> feCache;
    typedef typename PQkLocalFiniteElementCache<double,double,GridType::dimension,1>::FiniteElementType LocalFiniteElement;
    
    //LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement,TargetSpace> f(feCache.get(element),corners);

    
    HarmonicEnergyLocalStiffness<typename GridType::LeafGridView,LocalFiniteElement,TargetSpace> assembler;
    std::vector<TargetSpace> rotatedCoefficients(coefficients.size());

    for (int i=0; i<10; i++) {

        Rotation<double,3> rotation(FieldVector<double,3>(1), double(i));

        FieldMatrix<double,3,3> matrix;
        rotation.matrix(matrix);
        for (size_t j=0; j<coefficients.size(); j++) {
            FieldVector<double,3> tmp;
            matrix.mv(coefficients[j].globalCoordinates(), tmp);
            rotatedCoefficients[j] = tmp;
        }

        std::cout << "energy: " << assembler.energy(*grid->template leafbegin<0>(), 
                                                    feCache.get(grid->template leafbegin<0>()->type()),
                                                    rotatedCoefficients) << std::endl;

        std::vector<typename TargetSpace::EmbeddedTangentVector> rotatedGradient;
        assembler.assembleEmbeddedGradient(*grid->template leafbegin<0>(),
                                           feCache.get(grid->template leafbegin<0>()->type()),
                                   rotatedCoefficients,
                                   rotatedGradient);

        for (size_t j=0; j<coefficients.size(); j++) {
            FieldVector<double,3> tmp;
            matrix.mtv(rotatedGradient[j], tmp);
            std::cout << "gradient: " << tmp << std::endl;
        }

    }

}

template <class GridType>
void testGradientOfEnergy(const GridType* grid, const std::vector<TargetSpace>& coefficients)
{
    PQkLocalFiniteElementCache<double,double,GridType::dimension,1> feCache;
    typedef typename PQkLocalFiniteElementCache<double,double,GridType::dimension,1>::FiniteElementType LocalFiniteElement;
    
    //LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement,TargetSpace> f(feCache.get(element),corners);
    HarmonicEnergyLocalStiffness<typename GridType::LeafGridView,LocalFiniteElement,TargetSpace> assembler;

    std::vector<typename TargetSpace::EmbeddedTangentVector> gradient;
    assembler.assembleEmbeddedGradient(*grid->template leafbegin<0>(),
                                       feCache.get(grid->template leafbegin<0>()->type()),
                                       coefficients,
                                       gradient);

    ////////////////////////////////////////////////////////////////////////////////////////
    //  Assemble finite difference approximation of the gradient
    ////////////////////////////////////////////////////////////////////////////////////////

    std::vector<typename TargetSpace::EmbeddedTangentVector> fdGradient;
    assembler.assembleEmbeddedFDGradient(*grid->template leafbegin<0>(),
                                         feCache.get(grid->template leafbegin<0>()->type()),
                                       coefficients,
                                       fdGradient);

    double diff = 0;
    for (size_t i=0; i<fdGradient.size(); i++)
        diff = std::max(diff, (gradient[i] - fdGradient[i]).infinity_norm());
    
    if (diff > 1e-6) {
        
        std::cout << "gradient: " << std::endl;
        for (size_t i=0; i<gradient.size(); i++)
            std::cout << gradient[i] << std::endl;
    
        std::cout << "fd gradient: " << std::endl;
        for (size_t i=0; i<fdGradient.size(); i++)
            std::cout << fdGradient[i] << std::endl;

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
    factory.insertElement(GeometryType(GeometryType::simplex,dim), v);

    const GridType* grid = factory.createGrid();
    

    // //////////////////////////////////////////////////////////
    //  Test whether the energy is invariant under isometries
    // //////////////////////////////////////////////////////////

    std::vector<TargetSpace> testPoints;
    ValueFactory<TargetSpace>::get(testPoints);

    // Set up elements of S^2
    std::vector<TargetSpace> coefficients(dim+1);

    MultiIndex index(dim+1, testPoints.size());
    int numIndices = index.cycle();

    for (int i=0; i<numIndices; i++, ++index) {
        
        for (int j=0; j<dim+1; j++)
            coefficients[j] = testPoints[index[j]];

        testEnergy<GridType>(grid, coefficients);
        testGradientOfEnergy(grid, coefficients);
        
    }

}


int main(int argc, char** argv)
{
    testUnitVector3d<2>();
}
