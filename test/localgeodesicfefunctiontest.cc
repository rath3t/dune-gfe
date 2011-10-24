#include <config.h>

#include <fenv.h>
#include <iostream>

#include <dune/common/fvector.hh>
#include <dune/grid/common/quadraturerules.hh>

#include <dune/localfunctions/lagrange/pqkfactory.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/unitvector.hh>

#include <dune/gfe/localgeodesicfefunction.hh>
#include "multiindex.hh"
#include "valuefactory.hh"

const double eps = 1e-6;

using namespace Dune;

template <int domainDim>
void testDerivativeTangentiality(const RealTuple<1>& x,
                                 const FieldMatrix<double,1,domainDim>& derivative)
{
    // By construction, derivatives of RealTuples are always tangent
}

// the columns of the derivative must be tangential to the manifold
template <int domainDim, int vectorDim>
void testDerivativeTangentiality(const UnitVector<vectorDim>& x,
                                 const FieldMatrix<double,vectorDim,domainDim>& derivative)
{
    for (int i=0; i<domainDim; i++) {

        // The i-th column is a tangent vector if its scalar product with the global coordinates
        // of x vanishes.
        double sp = 0;
        for (int j=0; j<vectorDim; j++)
            sp += x.globalCoordinates()[j] * derivative[j][i];

        if (std::fabs(sp) > 1e-8)
            DUNE_THROW(Dune::Exception, "Derivative is not tangential: Column: " << i << ",  product: " << sp);

    }

}

// the columns of the derivative must be tangential to the manifold
template <int domainDim, int vectorDim>
void testDerivativeTangentiality(const Rotation<vectorDim-1,double>& x,
                                 const FieldMatrix<double,vectorDim,domainDim>& derivative)
{
}

// the columns of the derivative must be tangential to the manifold
template <int domainDim, int vectorDim>
void testDerivativeTangentiality(const RigidBodyMotion<3,double>& x,
                                 const FieldMatrix<double,vectorDim,domainDim>& derivative)
{
}

/** \brief Test whether interpolation is invariant under permutation of the simplex vertices
 */
template <int domainDim, class TargetSpace>
void testPermutationInvariance(const std::vector<TargetSpace>& corners)
{
    // works only for 2d domains
    assert(domainDim==2);

    PQkLocalFiniteElementCache<double,double,domainDim,1> feCache;
    typedef typename PQkLocalFiniteElementCache<double,double,domainDim,1>::FiniteElementType LocalFiniteElement;
    
    GeometryType simplex;
    simplex.makeSimplex(domainDim);

    //
    std::vector<TargetSpace> cornersRotated1(domainDim+1);
    std::vector<TargetSpace> cornersRotated2(domainDim+1);

    cornersRotated1[0] = cornersRotated2[2] = corners[1];
    cornersRotated1[1] = cornersRotated2[0] = corners[2];
    cornersRotated1[2] = cornersRotated2[1] = corners[0];
    
    LocalGeodesicFEFunction<2,double,LocalFiniteElement,TargetSpace> f0(feCache.get(simplex), corners);
    LocalGeodesicFEFunction<2,double,LocalFiniteElement,TargetSpace> f1(feCache.get(simplex), cornersRotated1);
    LocalGeodesicFEFunction<2,double,LocalFiniteElement,TargetSpace> f2(feCache.get(simplex), cornersRotated2);

    // A quadrature rule as a set of test points
    int quadOrder = 3;

    const Dune::QuadratureRule<double, domainDim>& quad 
        = Dune::QuadratureRules<double, domainDim>::rule(simplex, quadOrder);
    
    for (size_t pt=0; pt<quad.size(); pt++) {
        
        const Dune::FieldVector<double,domainDim>& quadPos = quad[pt].position();

        Dune::FieldVector<double,domainDim> l0 = quadPos;
        Dune::FieldVector<double,domainDim> l1, l2;
        
        l1[0] = quadPos[1];
        l1[1] = 1-quadPos[0]-quadPos[1];

        l2[0] = 1-quadPos[0]-quadPos[1];
        l2[1] = quadPos[0];

        // evaluate the three functions
        TargetSpace v0 = f0.evaluate(l0);
        TargetSpace v1 = f1.evaluate(l1);
        TargetSpace v2 = f2.evaluate(l2);
        
        // Check that they are all equal
        assert(TargetSpace::distance(v0,v1) < eps);
        assert(TargetSpace::distance(v0,v2) < eps);

    }

}

template <int domainDim, class TargetSpace>
void testDerivative(const std::vector<TargetSpace>& corners)
{
    // Make local fe function to be tested
    PQkLocalFiniteElementCache<double,double,domainDim,1> feCache;
    typedef typename PQkLocalFiniteElementCache<double,double,domainDim,1>::FiniteElementType LocalFiniteElement;
    
    GeometryType simplex;
    simplex.makeSimplex(domainDim);

    LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement,TargetSpace> f(feCache.get(simplex), corners);

    static const int embeddedDim = TargetSpace::EmbeddedTangentVector::dimension;
    
    // A quadrature rule as a set of test points
    int quadOrder = 3;
    
    const Dune::QuadratureRule<double, domainDim>& quad 
        = Dune::QuadratureRules<double, domainDim>::rule(GeometryType(GeometryType::simplex,domainDim), quadOrder);
    
    for (size_t pt=0; pt<quad.size(); pt++) {
        
        const Dune::FieldVector<double,domainDim>& quadPos = quad[pt].position();

        // evaluate actual derivative
        Dune::FieldMatrix<double, embeddedDim, domainDim> derivative = f.evaluateDerivative(quadPos);

        // evaluate fd approximation of derivative
        Dune::FieldMatrix<double, embeddedDim, domainDim> fdDerivative = f.evaluateDerivativeFD(quadPos);

        Dune::FieldMatrix<double, embeddedDim, domainDim> diff = derivative;
        diff -= fdDerivative;
        
        if ( diff.infinity_norm() > 100*eps ) {
            std::cout << className(corners[0]) << ": Analytical gradient does not match fd approximation." << std::endl;
            std::cout << "Analytical: " << derivative << std::endl;
            std::cout << "FD        : " << fdDerivative << std::endl;
        }

        testDerivativeTangentiality(f.evaluate(quadPos), derivative);

    }
}


template <int domainDim, class TargetSpace>
void testDerivativeOfValueWRTCoefficients(const std::vector<TargetSpace>& corners)
{
    // Make local fe function to be tested
    PQkLocalFiniteElementCache<double,double,domainDim,1> feCache;
    typedef typename PQkLocalFiniteElementCache<double,double,domainDim,1>::FiniteElementType LocalFiniteElement;
    
    GeometryType simplex;
    simplex.makeSimplex(domainDim);

    LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement,TargetSpace> f(feCache.get(simplex), corners);

    static const int embeddedDim = TargetSpace::EmbeddedTangentVector::dimension;
    
    // A quadrature rule as a set of test points
    int quadOrder = 3;
    
    const Dune::QuadratureRule<double, domainDim>& quad 
        = Dune::QuadratureRules<double, domainDim>::rule(simplex, quadOrder);
    
    for (size_t pt=0; pt<quad.size(); pt++) {
        
        Dune::FieldVector<double,domainDim> quadPos = quad[pt].position();
        
        // loop over the coefficients
        for (size_t i=0; i<corners.size(); i++) {

            // evaluate actual derivative
            FieldMatrix<double, embeddedDim, embeddedDim> derivative;
            f.evaluateDerivativeOfValueWRTCoefficient(quadPos, i, derivative);

            // evaluate fd approximation of derivative
            FieldMatrix<double, embeddedDim, embeddedDim> fdDerivative;
            f.evaluateFDDerivativeOfValueWRTCoefficient(quadPos, i, fdDerivative);

            if ( (derivative - fdDerivative).infinity_norm() > eps ) {
                std::cout << className<TargetSpace>() << ": Analytical derivative of value does not match fd approximation." << std::endl;
                std::cout << "coefficient: " << i << std::endl;
                std::cout << "quad pos: " << quadPos << std::endl;
                std::cout << "gfe: ";
                for (int j=0; j<domainDim+1; j++)
                    std::cout << ",   " << corners[j];
                std::cout << std::endl;
                std::cout << "Analytical:\n " << derivative << std::endl;
                std::cout << "FD        :\n " << fdDerivative << std::endl;
                assert(false);
            }

            //testDerivativeTangentiality(f.evaluate(quadPos), derivative);

        }
        
    }
}

template <int domainDim, class TargetSpace>
void testDerivativeOfGradientWRTCoefficients(const std::vector<TargetSpace>& corners)
{
    // Make local fe function to be tested
    PQkLocalFiniteElementCache<double,double,domainDim,1> feCache;
    typedef typename PQkLocalFiniteElementCache<double,double,domainDim,1>::FiniteElementType LocalFiniteElement;
    
    GeometryType simplex;
    simplex.makeSimplex(domainDim);

    LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement,TargetSpace> f(feCache.get(simplex),corners);

    static const int embeddedDim = TargetSpace::EmbeddedTangentVector::dimension;
    
    // A quadrature rule as a set of test points
    int quadOrder = 3;
    
    const Dune::QuadratureRule<double, domainDim>& quad 
        = Dune::QuadratureRules<double, domainDim>::rule(simplex, quadOrder);
    
    for (size_t pt=0; pt<quad.size(); pt++) {
        
        const Dune::FieldVector<double,domainDim>& quadPos = quad[pt].position();
        
        // loop over the coefficients
        for (size_t i=0; i<corners.size(); i++) {

            // evaluate actual derivative
            Tensor3<double, embeddedDim, embeddedDim, domainDim> derivative;
            f.evaluateDerivativeOfGradientWRTCoefficient(quadPos, i, derivative);

            // evaluate fd approximation of derivative
            Tensor3<double, embeddedDim, embeddedDim, domainDim> fdDerivative;
            f.evaluateFDDerivativeOfGradientWRTCoefficient(quadPos, i, fdDerivative);

            if ( (derivative - fdDerivative).infinity_norm() > eps ) {
                std::cout << className(corners[0]) << ": Analytical derivative of gradient does not match fd approximation." << std::endl;
                std::cout << "coefficient: " << i << std::endl;
                std::cout << "quad pos: " << quadPos << std::endl;
                std::cout << "gfe: ";
                for (int j=0; j<domainDim+1; j++)
                    std::cout << ",   " << corners[j];
                std::cout << std::endl;
                std::cout << "Analytical:\n " << derivative << std::endl;
                std::cout << "FD        :\n " << fdDerivative << std::endl;
                assert(false);
            }

            //testDerivativeTangentiality(f.evaluate(quadPos), derivative);

        }
        
    }
}


template <int domainDim>
void testRealTuples()
{
    std::cout << " --- Testing RealTuple<1>, domain dimension: " << domainDim << " ---" << std::endl;

    typedef RealTuple<1> TargetSpace;

    std::vector<TargetSpace> corners = {TargetSpace(1),
                                        TargetSpace(2),
                                        TargetSpace(3)};

    testPermutationInvariance<domainDim>(corners);
    testDerivative<domainDim>(corners);
}

template <int domainDim>
void testUnitVector2d()
{
    std::cout << " --- Testing UnitVector<2>, domain dimension: " << domainDim << " ---" << std::endl;

    typedef UnitVector<2> TargetSpace;

    std::vector<TargetSpace> testPoints;
    ValueFactory<TargetSpace>::get(testPoints);
    
    int nTestPoints = testPoints.size();
    
    // Set up elements of S^1
    std::vector<TargetSpace> corners(domainDim+1);
    
    MultiIndex<domainDim+1> index(nTestPoints);
    int numIndices = index.cycle();
    
    for (int i=0; i<numIndices; i++, ++index) {
        
        for (int j=0; j<domainDim+1; j++)
            corners[j] = testPoints[index[j]];

        bool spreadOut = false;
        for (int j=0; j<domainDim+1; j++)
            for (int k=0; k<domainDim+1; k++)
                if (UnitVector<2>::distance(corners[j],corners[k]) > M_PI*0.98)
                    spreadOut = true;
                    
        if (spreadOut)
            continue;
        
        //testPermutationInvariance(corners);
        testDerivative<domainDim>(corners);
        testDerivativeOfValueWRTCoefficients<domainDim>(corners);
        testDerivativeOfGradientWRTCoefficients<domainDim>(corners);

    }

}


template <class TargetSpace, int domainDim>
void test()
{
    std::cout << " --- Testing " << className<TargetSpace>() << ", domain dimension: " << domainDim << " ---" << std::endl;

    std::vector<TargetSpace> testPoints;
    ValueFactory<TargetSpace>::get(testPoints);
    
    int nTestPoints = testPoints.size();

    // Set up elements of SO(3)
    std::vector<TargetSpace> corners(domainDim+1);

    MultiIndex<domainDim+1> index(nTestPoints);
    int numIndices = index.cycle();

    for (int i=0; i<numIndices; i++, ++index) {
        
        for (int j=0; j<domainDim+1; j++)
            corners[j] = testPoints[index[j]];

        //testPermutationInvariance(corners);
        testDerivative<domainDim>(corners);
        testDerivativeOfValueWRTCoefficients<domainDim>(corners);
        testDerivativeOfGradientWRTCoefficients<domainDim>(corners);
                
    }

}


int main()
{
    // choke on NaN -- don't enable this by default, as there are
    // a few harmless NaN in the loopsolver
    //feenableexcept(FE_INVALID);

    std::cout << std::setw(15) << std::setprecision(12);
    
    testRealTuples<1>();
    testUnitVector2d<1>();
    test<UnitVector<3>,1>();
    testUnitVector2d<2>();
    test<UnitVector<3>,2>();
    
    test<Rotation<3,double>,1>();
    test<Rotation<3,double>,2>();
    
    test<RigidBodyMotion<3,double>,1>();
    test<RigidBodyMotion<3,double>,2>();
}
