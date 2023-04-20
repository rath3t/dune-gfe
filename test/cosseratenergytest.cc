#include "config.h"

#include <dune/grid/uggrid.hh>

#include <dune/geometry/type.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>

#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/cosseratenergystiffness.hh>

#include "multiindex.hh"
#include "valuefactory.hh"

typedef RigidBodyMotion<double,3> TargetSpace;

using namespace Dune;


// ////////////////////////////////////////////////////////
//   Make a test grid consisting of a single simplex
// ////////////////////////////////////////////////////////

template <class GridType>
std::unique_ptr<GridType> makeSingleSimplexGrid()
{
    static const int domainDim = GridType::dimension;
    GridFactory<GridType> factory;

    FieldVector<double,domainDim> pos(0);
    factory.insertVertex(pos);

    for (int i=0; i<domainDim; i++) {
        pos = 0;
        pos[i] = 1;
        factory.insertVertex(pos);
    }

    std::vector<unsigned int> v(domainDim+1);
    for (int i=0; i<domainDim+1; i++)
        v[i] = i;
    factory.insertElement(GeometryTypes::simplex(domainDim), v);

    return factory.createGrid();
}


//////////////////////////////////////////////////////////////////////////////////////
//   Test invariance of the energy functional under rotations
//////////////////////////////////////////////////////////////////////////////////////

template <class GridType>
void testEnergy(const GridType* grid, const std::vector<TargetSpace>& coefficients)
{
    ParameterTree materialParameters;
    materialParameters["thickness"] = "0.1";
    materialParameters["mu"] = "3.8462e+05";
    materialParameters["lambda"] = "2.7149e+05";
    materialParameters["mu_c"] = "3.8462e+05";
    materialParameters["L_c"] = "0.1";
    materialParameters["q"] = "2.5";
    materialParameters["kappa"] = "0.1";
    materialParameters["b1"] = "1";
    materialParameters["b2"] = "1";
    materialParameters["b3"] = "1";

    typedef Dune::Functions::LagrangeBasis<typename GridType::LeafGridView,1> FEBasis;
    FEBasis feBasis(grid->leafGridView());

    CosseratEnergyLocalStiffness<FEBasis,3> assembler(materialParameters,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr);
    
    // compute reference energy
    auto localView = feBasis.localView();
    localView.bind(*grid->leafGridView().template begin<0>());

    double referenceEnergy = assembler.energy(localView,
                                              coefficients);
    
    // rotate the entire configuration
    std::vector<TargetSpace> rotatedCoefficients(coefficients.size());
    
    std::vector<Rotation<double,3> > testRotations;
    ValueFactory<Rotation<double,3> >::get(testRotations);

    for (size_t i=0; i<testRotations.size(); i++) {

        /////////////////////////////////////////////////////////////////////////
        //  Multiply the given configuration by the test rotation.
        //  The energy should remain unchanged.
        /////////////////////////////////////////////////////////////////////////
        FieldMatrix<double,3,3> matrix;
        testRotations[i].matrix(matrix);
        
        for (size_t j=0; j<coefficients.size(); j++) {
            FieldVector<double,3> tmp;
            matrix.mv(coefficients[j].r, tmp);
            rotatedCoefficients[j].r = tmp;
            
            rotatedCoefficients[j].q = testRotations[i].mult(coefficients[j].q);
        }
        
        double energy = assembler.energy(localView,
                                         rotatedCoefficients);
        assert(std::fabs(energy-referenceEnergy)/std::fabs(energy) < 1e-4);

    }

}


template <int domainDim>
void testFrameInvariance()
{
    std::cout << " --- Testing frame invariance of the Cosserat energy, domain dimension: " << domainDim << " ---" << std::endl;

    // ////////////////////////////////////////////////////////
    //   Make a test grid consisting of a single simplex
    // ////////////////////////////////////////////////////////

    typedef UGGrid<domainDim> GridType;
    const std::unique_ptr<GridType> grid = makeSingleSimplexGrid<GridType>();
    
    // //////////////////////////////////////////////////////////
    //  Test whether the energy is invariant under isometries
    // //////////////////////////////////////////////////////////

    std::vector<TargetSpace> testPoints;
    ValueFactory<TargetSpace>::get(testPoints);

    // Set up elements of SE(3)
    std::vector<TargetSpace> coefficients(domainDim+1);

    ::MultiIndex index(domainDim+1, testPoints.size());
    int numIndices = index.cycle();

    for (int i=0; i<numIndices; i++, ++index) {
        
        // Discard all configurations that deform the element to zero area
        bool identicalPoints = false;
        for (int j=0; j<domainDim+1; j++)
          for (int k=0; k<domainDim+1; k++)
            if (j!=k and (testPoints[index[j]].r - testPoints[index[k]].r).two_norm() < 1e-5)
              identicalPoints = true;

        if (identicalPoints)
          continue;

        for (int j=0; j<domainDim+1; j++)
            coefficients[j] = testPoints[index[j]];

        testEnergy<GridType>(grid.get(), coefficients);
        
    }
    
}

int main(int argc, char** argv)
{
    MPIHelper::instance(argc, argv);

    const int domainDim = 2;

    //////////////////////////////////////////////////////////////////////////////////////
    //   Test invariance of the energy functional under rotations
    //////////////////////////////////////////////////////////////////////////////////////
    
    testFrameInvariance<domainDim>();

}
