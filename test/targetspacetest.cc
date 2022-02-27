#include <config.h>

#include <dune/gfe/unitvector.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/rotation.hh>
#include <dune/gfe/productmanifold.hh>
#include <dune/gfe/hyperbolichalfspacepoint.hh>

#include "valuefactory.hh"

using namespace Dune;


/** \file
    \brief Unit tests for classes that implement value manifolds for geodesic FE functions
*/

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

const double eps = 1e-4;

template <class TargetSpace>
double distanceSquared(const TargetSpace& a, const TargetSpace& b)
{
    return Dune::power(TargetSpace::distance(a,b), 2);
}

// Squared distance between two points slightly off the manifold.
// This is required for finite difference approximations.
template <class TargetSpace, int dim>
double distanceSquared(const FieldVector<double,dim>& a, const FieldVector<double,dim>& b)
{
    return Dune::power(TargetSpace::distance(TargetSpace(a),TargetSpace(b)), 2);
}

/** \brief Compute the Riemannian Hessian of the squared distance function in global coordinates

    The formula for the Riemannian Hessian has been taken from Absil, Mahony, Sepulchre:
    'Optimization algorithms on matrix manifolds', page 107
*/
template <class TargetSpace, int worldDim>
FieldMatrix<double,worldDim,worldDim> getSecondDerivativeOfSecondArgumentFD(const TargetSpace& a, const TargetSpace& b)
{
    
    const size_t spaceDim = TargetSpace::dim;

    // finite-difference approximation
    FieldMatrix<double,spaceDim,spaceDim> d2d2_fd(0);
    
    FieldMatrix<double,spaceDim,worldDim> B = b.orthonormalFrame();
    
    for (size_t i=0; i<spaceDim; i++) {
        for (size_t j=0; j<spaceDim; j++) {

            FieldVector<double,worldDim> epsXi  = eps * B[i];
            FieldVector<double,worldDim> epsEta = eps * B[j];
            
            FieldVector<double,worldDim> minusEpsXi  = -1 * epsXi;
            FieldVector<double,worldDim> minusEpsEta = -1 * epsEta;
            
            double forwardValue  = distanceSquared(a,TargetSpace::exp(b,epsXi+epsEta)) - distanceSquared(a, TargetSpace::exp(b,epsXi)) - distanceSquared(a,TargetSpace::exp(b,epsEta));
            double centerValue   = distanceSquared(a,b) - distanceSquared(a,b) - distanceSquared(a,b);
            double backwardValue = distanceSquared(a,TargetSpace::exp(b,minusEpsXi + minusEpsEta)) - distanceSquared(a, TargetSpace::exp(b,minusEpsXi)) - distanceSquared(a,TargetSpace::exp(b,minusEpsEta));
            
            d2d2_fd[i][j] = 0.5 * (forwardValue - 2*centerValue + backwardValue) / (eps*eps);
            
        }
    }
    
    //B.invert();
    FieldMatrix<double,worldDim,spaceDim> BT;
    for (int i=0; i<worldDim; i++)
        for (size_t j=0; j<spaceDim; j++)
            BT[i][j] = B[j][i];
    
    
    FieldMatrix<double,spaceDim,worldDim> ret1;
    FMatrixHelp::multMatrix(d2d2_fd,B,ret1);
    
    FieldMatrix<double,worldDim,worldDim> ret2;
    FMatrixHelp::multMatrix(BT,ret1,ret2);
    return ret2;
}

template <class TargetSpace>
void testOrthonormalFrame(const TargetSpace& a)
{
    const size_t spaceDim = TargetSpace::dim;
    const size_t embeddedDim = TargetSpace::embeddedDim;
    
    FieldMatrix<double,spaceDim,embeddedDim> B = a.orthonormalFrame();

    for (size_t i=0; i<spaceDim; i++)
        for (size_t j=0; j<spaceDim; j++)
            assert( std::fabs(a.metric(B[i],B[j]) - (i==j)) < 1e-10 );
}

template <class TargetSpace>
void testDerivativeOfDistanceSquared(const TargetSpace& a, const TargetSpace& b)
{
    static const size_t embeddedDim = TargetSpace::embeddedDim;
    
    ///////////////////////////////////////////////////////////////////
    //  Test derivative with respect to second argument
    ///////////////////////////////////////////////////////////////////
    typename TargetSpace::EmbeddedTangentVector d2 =  TargetSpace::derivativeOfDistanceSquaredWRTSecondArgument(a, b);    

    // finite-difference approximation
    Dune::FieldMatrix<double,TargetSpace::TangentVector::dimension,embeddedDim> B = b.orthonormalFrame();

    typename TargetSpace::TangentVector d2_fd;
    for (size_t i=0; i<TargetSpace::TangentVector::dimension; i++) {
        
        typename TargetSpace::EmbeddedTangentVector fwVariation =  eps * B[i];
        typename TargetSpace::EmbeddedTangentVector bwVariation = -eps * B[i];
        TargetSpace bPlus  = TargetSpace::exp(b,fwVariation);
        TargetSpace bMinus = TargetSpace::exp(b,bwVariation);

        d2_fd[i] = (distanceSquared(a,bPlus) - distanceSquared(a,bMinus)) / (2*eps);
    }

    // transform into embedded coordinates
    typename TargetSpace::EmbeddedTangentVector d2_fd_embedded;
    B.mtv(d2_fd,d2_fd_embedded);

    if ( (d2 - d2_fd_embedded).infinity_norm() > 200*eps ) {
        std::cout << className(a) << ": Analytical gradient does not match fd approximation." << std::endl;
        std::cout << "d2 Analytical: " << d2 << std::endl;
        std::cout << "d2 FD        : " << d2_fd << std::endl;
        assert(false);
    }
    
}

template <class TargetSpace>
void testHessianOfDistanceSquared(const TargetSpace& a, const TargetSpace& b)
{
    static const int embeddedDim = TargetSpace::embeddedDim;
    
    ///////////////////////////////////////////////////////////////////
    //  Test second derivative with respect to second argument
    ///////////////////////////////////////////////////////////////////
    FieldMatrix<double,embeddedDim,embeddedDim> d2d2 = (TargetSpace::secondDerivativeOfDistanceSquaredWRTSecondArgument(a, b)).matrix();

    // finite-difference approximation
    FieldMatrix<double,embeddedDim,embeddedDim> d2d2_fd = getSecondDerivativeOfSecondArgumentFD<TargetSpace,embeddedDim>(a,b);
    
    if ( (d2d2 - d2d2_fd).infinity_norm() > 200*eps) {
        std::cout << className(a) << ": Analytical second derivative does not match fd approximation." << std::endl;
        std::cout << "d2d2 Analytical:" << std::endl << d2d2 << std::endl;
        std::cout << "d2d2 FD        :" << std::endl << d2d2_fd << std::endl;
        assert(false);
    }
    
}

template <class TargetSpace>
void testMixedDerivativesOfDistanceSquared(const TargetSpace& a, const TargetSpace& b)
{
    static const size_t embeddedDim = TargetSpace::embeddedDim;
    
    //////////////////////////////////////////////////////////////////////////////
    //  Test mixed second derivative with respect to first and second argument
    //////////////////////////////////////////////////////////////////////////////

    FieldMatrix<double,embeddedDim,embeddedDim> d1d2 = TargetSpace::secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(a, b);
    
    // finite-difference approximation
    FieldMatrix<double,embeddedDim,embeddedDim> d1d2_fd;
    
    for (size_t i=0; i<embeddedDim; i++) {
        for (size_t j=0; j<embeddedDim; j++) {
            
            FieldVector<double,embeddedDim> aPlus  = a.globalCoordinates();
            FieldVector<double,embeddedDim> aMinus = a.globalCoordinates();
            aPlus[i]  += eps;
            aMinus[i] -= eps;

            FieldVector<double,embeddedDim> bPlus  = b.globalCoordinates();
            FieldVector<double,embeddedDim> bMinus = b.globalCoordinates();
            bPlus[j]  += eps;
            bMinus[j] -= eps;
                
            d1d2_fd[i][j] = (distanceSquared<TargetSpace>(aPlus,bPlus) + distanceSquared<TargetSpace>(aMinus,bMinus)
                            - distanceSquared<TargetSpace>(aPlus,bMinus) - distanceSquared<TargetSpace>(aMinus,bPlus)) / (4*eps*eps);

        }
    }
    
    if ( (d1d2 - d1d2_fd).infinity_norm() > 200*eps ) {
        std::cout << className(a) << ": Analytical mixed second derivative does not match fd approximation." << std::endl;
        std::cout << "d1d2 Analytical:" << std::endl << d1d2 << std::endl;
        std::cout << "d1d2 FD        :" << std::endl << d1d2_fd << std::endl;
        assert(false);
    }

}


template <class TargetSpace>
void testDerivativeOfHessianOfDistanceSquared(const TargetSpace& a, const TargetSpace& b)
{
    static const size_t embeddedDim = TargetSpace::embeddedDim;
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    //  Test mixed third derivative with respect to first (once) and second (twice) argument
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    Tensor3<double,embeddedDim,embeddedDim,embeddedDim> d2d2d2 = TargetSpace::thirdDerivativeOfDistanceSquaredWRTSecondArgument(a, b);
    
    Tensor3<double,embeddedDim,embeddedDim,embeddedDim> d2d2d2_fd;
    
    for (size_t i=0; i<embeddedDim; i++) {
        
        FieldVector<double,embeddedDim> bPlus  = b.globalCoordinates();
        FieldVector<double,embeddedDim> bMinus = b.globalCoordinates();
        bPlus[i]  += eps;
        bMinus[i] -= eps;

        FieldMatrix<double,embeddedDim,embeddedDim> hPlus  = getSecondDerivativeOfSecondArgumentFD<TargetSpace,embeddedDim>(a,TargetSpace(bPlus));
        FieldMatrix<double,embeddedDim,embeddedDim> hMinus = getSecondDerivativeOfSecondArgumentFD<TargetSpace,embeddedDim>(a,TargetSpace(bMinus));
        
        d2d2d2_fd[i] = hPlus;
        d2d2d2_fd[i] -= hMinus;
        d2d2d2_fd[i] /= 2*eps;
        
    }

    if ( (d2d2d2 - d2d2d2_fd).infinity_norm() > 200*eps) {
        std::cout << className(a) << ": Analytical third derivative does not match fd approximation." << std::endl;
        std::cout << "d2d2d2 Analytical:" << std::endl << d2d2d2 << std::endl;
        std::cout << "d2d2d2 FD        :" << std::endl << d2d2d2_fd << std::endl;
        assert(false);
    }

}


template <class TargetSpace>
void testMixedDerivativeOfHessianOfDistanceSquared(const TargetSpace& a, const TargetSpace& b)
{
    static const size_t embeddedDim = TargetSpace::embeddedDim;
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    //  Test mixed third derivative with respect to first (once) and second (twice) argument
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    Tensor3<double,embeddedDim,embeddedDim,embeddedDim> d1d2d2 = TargetSpace::thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(a, b);
    
    Tensor3<double,embeddedDim,embeddedDim,embeddedDim> d1d2d2_fd;
    
    for (size_t i=0; i<embeddedDim; i++) {
        
        FieldVector<double,embeddedDim> aPlus  = a.globalCoordinates();
        FieldVector<double,embeddedDim> aMinus = a.globalCoordinates();
        aPlus[i]  += eps;
        aMinus[i] -= eps;

        FieldMatrix<double,embeddedDim,embeddedDim> hPlus  = getSecondDerivativeOfSecondArgumentFD<TargetSpace,embeddedDim>(TargetSpace(aPlus),b);
        FieldMatrix<double,embeddedDim,embeddedDim> hMinus = getSecondDerivativeOfSecondArgumentFD<TargetSpace,embeddedDim>(TargetSpace(aMinus),b);
        
        d1d2d2_fd[i] = hPlus;
        d1d2d2_fd[i] -= hMinus;
        d1d2d2_fd[i] /= 2*eps;
        
    }

    if ( (d1d2d2 - d1d2d2_fd).infinity_norm() > 200*eps ) {
        std::cout << className(a) << ": Analytical mixed third derivative does not match fd approximation." << std::endl;
        std::cout << "d1d2d2 Analytical:" << std::endl << d1d2d2 << std::endl;
        std::cout << "d1d2d2 FD        :" << std::endl << d1d2d2_fd << std::endl;
        assert(false);
    }

}


template <class TargetSpace>
void testDerivativesOfDistanceSquared(const TargetSpace& a, const TargetSpace& b)
{
    
    ///////////////////////////////////////////////////////////////////
    //  Test derivative with respect to second argument
    ///////////////////////////////////////////////////////////////////
    
    testDerivativeOfDistanceSquared<TargetSpace>(a,b);
    
    ///////////////////////////////////////////////////////////////////
    //  Test second derivative with respect to second argument
    ///////////////////////////////////////////////////////////////////

    testHessianOfDistanceSquared<TargetSpace>(a,b);

    //////////////////////////////////////////////////////////////////////////////
    //  Test mixed second derivative with respect to first and second argument
    //////////////////////////////////////////////////////////////////////////////

    testMixedDerivativesOfDistanceSquared<TargetSpace>(a,b);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    //  Test third derivative with respect to second argument
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    testDerivativeOfHessianOfDistanceSquared<TargetSpace>(a,b);

    /////////////////////////////////////////////////////////////////////////////////////////////
    //  Test mixed third derivative with respect to first (once) and second (twice) argument
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    testMixedDerivativeOfHessianOfDistanceSquared<TargetSpace>(a,b);

}


template <class TargetSpace>
void test()
{
    std::cout << "Testing class " << className<TargetSpace>() << std::endl;
    
    std::vector<TargetSpace> testPoints;
    ValueFactory<TargetSpace>::get(testPoints);
    
    int nTestPoints = testPoints.size();
    
    // Test each element in the list
    for (int i=0; i<nTestPoints; i++) {
        
        //testOrthonormalFrame<TargetSpace>(testPoints[i]);
        
        for (int j=0; j<nTestPoints; j++) {
            
            std::vector<TargetSpace> testPointPair(2);
            testPointPair[0] = testPoints[i];
            testPointPair[1] = testPoints[j];
            if (diameter(testPointPair) > TargetSpace::convexityRadius)
                continue;

            testDerivativesOfDistanceSquared<TargetSpace>(testPoints[i], testPoints[j]);
            
        }
        
    }
    
}


int main() try
{
//     test<RealTuple<double,1> >();
//     test<RealTuple<double,3> >();
    
    test<UnitVector<double,2> >();
    test<UnitVector<double,3> >();
    test<UnitVector<double,4> >();

    test<Dune::GFE::ProductManifold<RealTuple<double,1>,Rotation<double,3>,UnitVector<double,2>>>();
    test<Dune::GFE::ProductManifold<Rotation<double,3>,UnitVector<double,5>>>();

//     test<Rotation<double,3> >();
//
    // Test the RigidBodyMotion class
    test<RigidBodyMotion<double,3> >();
//
//     test<HyperbolicHalfspacePoint<double,2> >();

} catch (Exception& e) {

    std::cout << e.what() << std::endl;

}

