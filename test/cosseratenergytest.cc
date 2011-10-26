#include "config.h"

#include <dune/grid/uggrid.hh>

#include <dune/localfunctions/lagrange/pqkfactory.hh>

#include <dune/fufem/functions/constantfunction.hh>

#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/cosseratenergystiffness.hh>

#include "multiindex.hh"
#include "valuefactory.hh"

const int dim = 2;

const double eps = 1e-4;

typedef RigidBodyMotion<3> TargetSpace;

using namespace Dune;



template <int domainDim,class LocalFiniteElement>
Tensor3<double,3,3,3> evaluateDerivativeFD(const LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement,TargetSpace>& f,
                                           const Dune::FieldVector<double,domainDim>& local)
{
    Tensor3<double,3,3,3> result(0);

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

    GeometryType simplex;
    simplex.makeSimplex(domainDim);
    
    LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement, TargetSpace> f(feCache.get(simplex), corners);

    // A quadrature rule as a set of test points
    int quadOrder = 3;
    
    const Dune::QuadratureRule<double, domainDim>& quad 
        = Dune::QuadratureRules<double, domainDim>::rule(GeometryType(GeometryType::simplex,domainDim), quadOrder);
    
    for (size_t pt=0; pt<quad.size(); pt++) {
        
        const Dune::FieldVector<double,domainDim>& quadPos = quad[pt].position();

        // evaluate actual derivative
        Dune::FieldMatrix<double, TargetSpace::EmbeddedTangentVector::dimension, domainDim> derivative = f.evaluateDerivative(quadPos);

        Tensor3<double,3,3,3> DR;
        CosseratEnergyLocalStiffness<typename UGGrid<domainDim>::LeafGridView,LocalFiniteElement,3>::computeDR(f.evaluate(quadPos),derivative, DR);

        //std::cout << "DR:\n" << DR << std::endl;

        // evaluate fd approximation of derivative
        Tensor3<double,3,3,3> DR_fd = evaluateDerivativeFD(f,quadPos);

        double maxDiff = 0;
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                for (int k=0; k<3; k++)
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
void testEnergy(const GridType* grid, const std::vector<TargetSpace>& coefficients) {

    PQkLocalFiniteElementCache<double,double,GridType::dimension,1> feCache;
    typedef typename PQkLocalFiniteElementCache<double,double,GridType::dimension,1>::FiniteElementType LocalFiniteElement;
    
    //LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement,TargetSpace> f(feCache.get(element),corners);

    ParameterTree materialParameters;
    materialParameters["thickness"] = "0.1";
    materialParameters["mu"] = "3.8462e+05";
    materialParameters["lambda"] = "2.7149e+05";
    materialParameters["mu_c"] = "3.8462e+05";
    materialParameters["L_c"] = "0.1";
    materialParameters["q"] = "2.5";

    ConstantFunction<Dune::FieldVector<double,GridType::dimension>, Dune::FieldVector<double,3> > zeroFunction(Dune::FieldVector<double,3>(0));
    
    CosseratEnergyLocalStiffness<typename GridType::LeafGridView,LocalFiniteElement,3> assembler(materialParameters,
                                                                                                 NULL,
                                                                                                 &zeroFunction);
    std::vector<TargetSpace> rotatedCoefficients(coefficients.size());
    
    std::vector<Rotation<3> > testRotations;
    ValueFactory<Rotation<3> >::get(testRotations);

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
        
        std::cout << "energy: " << assembler.energy(*grid->template leafbegin<0>(), 
                                                    feCache.get(grid->template leafbegin<0>()->type()),
                                                    rotatedCoefficients) << std::endl;

    }

}


template <int domainDim>
void testFrameInvariance()
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

    // Set up elements of SE(3)
    std::vector<TargetSpace> coefficients(dim+1);

    MultiIndex index(dim+1, testPoints.size());
    int numIndices = index.cycle();

    for (int i=0; i<numIndices; i++, ++index) {
        
        for (int j=0; j<dim+1; j++)
            coefficients[j] = testPoints[index[j]];

        testEnergy<GridType>(grid, coefficients);
        
    }
    
}


int main(int argc, char** argv) try
{
    const int domainDim = 2;
    std::cout << " --- Testing Rotation<3>, domain dimension: " << domainDim << " ---" << std::endl;

    std::vector<Rotation<3,double> > testPoints;
    
    ValueFactory<Rotation<3,double> >::get(testPoints);
    
    int nTestPoints = testPoints.size();

    // Set up elements of SO(3)
    std::vector<TargetSpace> corners(domainDim+1);

    MultiIndex index(domainDim+1, nTestPoints);
    int numIndices = index.cycle();

    for (int i=0; i<numIndices; i++, ++index) {
        
        for (int j=0; j<domainDim+1; j++)
            corners[j].q = testPoints[index[j]];

        testDerivativeOfRotationMatrix<2>(corners);
                
    }
    
    //////////////////////////////////////////////////////////////////////////////////////
    //   Test invariance of the energy functional under rotations
    //////////////////////////////////////////////////////////////////////////////////////
    
    testFrameInvariance<2>();
    
} catch (Exception e) {

    std::cout << e << std::endl;

 }
