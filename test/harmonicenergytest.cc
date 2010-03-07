
#include <dune/grid/uggrid.hh>

#include <dune/src/unitvector.hh>
#include <dune/src/harmonicenergystiffness.hh>

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

        std::vector<Dune::FieldVector<double,3> > rotatedGradient;
        assembler.assembleGradient(*grid->template leafbegin<0>(),
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

    FieldVector<double,3> uv;
    std::vector<TargetSpace> coefficients(dim+1);

    uv[0] = 1;  uv[1] = 0;  uv[2] = 0;
    coefficients[0] = uv;
    uv[0] = 0;  uv[1] = 1;  uv[2] = 0;
    coefficients[1] = uv;
    uv[0] = 0;  uv[1] = 0;  uv[2] = 1;
    coefficients[2] = uv;
    
    testEnergy<GridType>(grid, coefficients);
}
