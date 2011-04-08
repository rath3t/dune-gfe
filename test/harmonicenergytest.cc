#include "config.h"

#include <dune/grid/uggrid.hh>

#include <dune/gfe/unitvector.hh>
#include <dune/gfe/harmonicenergystiffness.hh>

#include "multiindex.hh"

const int dim = 2;

typedef UnitVector<3> TargetSpace;

using namespace Dune;

template <class GridType>
void testEnergy(const GridType* grid, const std::vector<TargetSpace>& coefficients) {

    HarmonicEnergyLocalStiffness<typename GridType::LeafGridView,TargetSpace> assembler;
    std::vector<TargetSpace> rotatedCoefficients(coefficients.size());

    for (int i=0; i<10; i++) {

        Rotation<3,double> rotation(FieldVector<double,3>(1), double(i));

        FieldMatrix<double,3,3> matrix;
        rotation.matrix(matrix);
        for (size_t j=0; j<coefficients.size(); j++) {
            FieldVector<double,3> tmp;
            matrix.mv(coefficients[j].globalCoordinates(), tmp);
            rotatedCoefficients[j] = tmp;
        }

        std::cout << "energy: " << assembler.energy(*grid->template leafbegin<0>(), 
                                                    rotatedCoefficients) << std::endl;

        std::vector<typename TargetSpace::EmbeddedTangentVector> rotatedGradient;
        assembler.assembleEmbeddedGradient(*grid->template leafbegin<0>(),
                                   rotatedCoefficients,
                                   rotatedGradient);

        for (size_t j=0; j<coefficients.size(); j++) {
            FieldVector<double,3> tmp;
            matrix.mtv(rotatedGradient[j], tmp);
            std::cout << "gradient: " << tmp << std::endl;
        }

    }

};

int main(int argc, char** argv)
{
    // ////////////////////////////////////////////////////////
    //   Make a test grid consisting of a single simplex
    // ////////////////////////////////////////////////////////

    typedef UGGrid<dim> GridType;

    GridFactory<GridType> factory;

    FieldVector<double,dim> pos(0);
    factory.insertVertex(pos);
    pos[0] = 1;  pos[1] = 0;
    factory.insertVertex(pos);
    pos[0] = 0;  pos[1] = 1;
    factory.insertVertex(pos);

    std::vector<unsigned int> v(dim+1);
    v[0] = 0;  v[1] = 1;  v[2] = 2;
    factory.insertElement(GeometryType(GeometryType::simplex,dim), v);

    const GridType* grid = factory.createGrid();
    

    // //////////////////////////////////////////////////////////
    //  Test whether the energy is invariant under isometries
    // //////////////////////////////////////////////////////////

    int nTestPoints = 10;
    double testPoints[10][3] = {{1,0,0}, {0,1,0}, {-0.838114,0.356751,-0.412667},
                               {-0.490946,-0.306456,0.81551},{-0.944506,0.123687,-0.304319},
                               {-0.6,0.1,-0.2},{0.45,0.12,0.517},
                               {-0.1,0.3,-0.1},{-0.444506,0.123687,0.104319},{-0.7,-0.123687,-0.304319}};
    
    // Set up elements of S^2
    std::vector<TargetSpace> coefficients(dim+1);

    MultiIndex<dim+1> index(nTestPoints);
    int numIndices = index.cycle();

    for (int i=0; i<numIndices; i++, ++index) {
        
        for (int j=0; j<dim+1; j++) {
            Dune::array<double,3> w = {testPoints[index[j]][0], testPoints[index[j]][1], testPoints[index[j]][2]};
            coefficients[j] = UnitVector<3>(w);
        }

        testEnergy<GridType>(grid, coefficients);
                
    }

}
