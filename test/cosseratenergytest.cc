#include "config.h"

#include <dune/grid/uggrid.hh>

#include <dune/localfunctions/lagrange/pqkfactory.hh>

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




int main(int argc, char** argv)
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
    
}
