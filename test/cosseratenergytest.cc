#include "config.h"

#include <dune/grid/uggrid.hh>
#include <dune/grid/onedgrid.hh>

#include <dune/geometry/type.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>

#include <dune/fufem/functions/constantfunction.hh>

#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/cosseratenergystiffness.hh>

#include "multiindex.hh"
#include "valuefactory.hh"

const double eps = 1e-4;

typedef RigidBodyMotion<double,3> TargetSpace;

using namespace Dune;


/** \brief Computes the diameter of a set */
template <class TargetSpace>
double diameter(const std::vector<TargetSpace>& v)
{
    double d = 0;
    for (size_t i=0; i<v.size(); i++)
        for (size_t j=0; j<v.size(); j++)
            d = std::max(d, Rotation<double,3>::distance(v[i].q,v[j].q));
    return d;
}



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

    return std::unique_ptr<GridType>(factory.createGrid());
}


template <int domainDim,class LocalFiniteElement>
Tensor3<double,3,3,domainDim> evaluateDerivativeFD(const LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement,TargetSpace>& f,
                                           const Dune::FieldVector<double,domainDim>& local)
{
    Tensor3<double,3,3,domainDim> result(0);

    for (int i=0; i<domainDim; i++) {
        
        Dune::FieldVector<double, domainDim> forward  = local;
        Dune::FieldVector<double, domainDim> backward = local;
        
        forward[i]  += eps;
        backward[i] -= eps;

        TargetSpace forwardValue = f.evaluate(forward);
        TargetSpace backwardValue = f.evaluate(backward);
        
        FieldMatrix<double,3,3> forwardMatrix, backwardMatrix;
        forwardValue.q.matrix(forwardMatrix);
        backwardValue.q.matrix(backwardMatrix);
        
        FieldMatrix<double,3,3> fdDer = (forwardMatrix - backwardMatrix) / (2*eps);
        
        for (int j=0; j<3; j++)
            for (int k=0; k<3; k++)
                result[j][k][i] = fdDer[j][k];
        
    }

    return result;

}

template <int domainDim>
void testDerivativeOfRotationMatrix(const std::vector<TargetSpace>& corners)
{
    // Make local fe function to be tested
    PQkLocalFiniteElementCache<double,double,domainDim,1> feCache;
    typedef typename PQkLocalFiniteElementCache<double,double,domainDim,1>::FiniteElementType LocalFiniteElement;

    LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement, TargetSpace> f(feCache.get(GeometryTypes::simplex(domainDim)), corners);

    // A quadrature rule as a set of test points
    int quadOrder = 3;
    
    const auto& quad = Dune::QuadratureRules<double, domainDim>::rule(GeometryTypes::simplex(domainDim), quadOrder);
    
    for (size_t pt=0; pt<quad.size(); pt++) {
        
        const Dune::FieldVector<double,domainDim>& quadPos = quad[pt].position();

        // evaluate actual derivative
        Dune::FieldMatrix<double, TargetSpace::EmbeddedTangentVector::dimension, domainDim> derivative = f.evaluateDerivative(quadPos);

        Tensor3<double,3,3,domainDim> DR;
        typedef Dune::Functions::LagrangeBasis<typename UGGrid<domainDim>::LeafGridView,1> FEBasis;
        CosseratEnergyLocalStiffness<FEBasis,3>::computeDR(f.evaluate(quadPos),derivative, DR);

        // evaluate fd approximation of derivative
        Tensor3<double,3,3,domainDim> DR_fd = evaluateDerivativeFD(f,quadPos);

        double maxDiff = 0;
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                for (int k=0; k<domainDim; k++)
                    maxDiff = std::max(maxDiff, std::abs(DR[i][j][k] - DR_fd[i][j][k]));
        
        if ( maxDiff > 100*eps ) {
            std::cout << className(corners[0]) << ": Analytical gradient does not match fd approximation." << std::endl;
            std::cout << "Analytical:\n " << DR << std::endl;
            std::cout << "FD        :\n " << DR_fd << std::endl;
            assert(false);
        }

    }
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
        
        assert(std::fabs(energy-referenceEnergy) < 1e-4);
        
        //std::cout << "energy: " << energy << std::endl;

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
        
        for (int j=0; j<domainDim+1; j++)
            coefficients[j] = testPoints[index[j]];

        testEnergy<GridType>(grid.get(), coefficients);
        
    }
    
}

int main(int argc, char** argv)
{
    MPIHelper::instance(argc, argv);

    const int domainDim = 2;

    // ////////////////////////////////////////////////////////
    //   Make a test grid consisting of a single simplex
    // ////////////////////////////////////////////////////////

    typedef UGGrid<domainDim> GridType;
    const std::unique_ptr<GridType> grid = makeSingleSimplexGrid<GridType>();

    ////////////////////////////////////////////////////////////////////////////
    //  Create a local assembler object
    ////////////////////////////////////////////////////////////////////////////

    typedef Dune::Functions::LagrangeBasis<typename GridType::LeafGridView,1> Basis;
    Basis basis(grid->leafGridView());

    std::cout << " --- Testing derivative of rotation matrix, domain dimension: " << domainDim << " ---" << std::endl;
    
    std::vector<Rotation<double,3> > testPoints;
    
    ValueFactory<Rotation<double,3> >::get(testPoints);
    
    int nTestPoints = testPoints.size();

    // Set up elements of SO(3)
    std::vector<TargetSpace> corners(domainDim+1);

    ::MultiIndex index(domainDim+1, nTestPoints);
    int numIndices = index.cycle();

    for (int i=0; i<numIndices; i++, ++index)
    {
        for (int j=0; j<domainDim+1; j++)
            corners[j].q = testPoints[index[j]];

        testDerivativeOfRotationMatrix<domainDim>(corners);
    }
    
    //////////////////////////////////////////////////////////////////////////////////////
    //   Test invariance of the energy functional under rotations
    //////////////////////////////////////////////////////////////////////////////////////
    
    testFrameInvariance<domainDim>();

}
