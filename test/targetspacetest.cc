#include <config.h>

#include <dune/gfe/unitvector.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/rotation.hh>

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
double energy(const TargetSpace& a, const TargetSpace& b)
{
    return TargetSpace::distance(a,b) * TargetSpace::distance(a,b);
}

template <class TargetSpace, int dim>
double energy(const TargetSpace& a, const FieldVector<double,dim>& b)
{
#warning Cast where there should not be one
    return TargetSpace::distance(a,TargetSpace(b)) * TargetSpace::distance(a,TargetSpace(b));
}

template <class TargetSpace, int dim>
double energy(const FieldVector<double,dim>& a, const FieldVector<double,dim>& b)
{
#warning Cast where there should not be one
    return TargetSpace::distance(TargetSpace(a),TargetSpace(b)) * TargetSpace::distance(TargetSpace(a),TargetSpace(b));
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

            FieldVector<double,worldDim> epsXi = B[i];    epsXi *= eps;
            FieldVector<double,worldDim> epsEta = B[j];   epsEta *= eps;
            
            FieldVector<double,worldDim> minusEpsXi  = epsXi;   minusEpsXi  *= -1;
            FieldVector<double,worldDim> minusEpsEta = epsEta;  minusEpsEta *= -1;
            
            double forwardValue  = energy(a,TargetSpace::exp(b,epsXi+epsEta)) - energy(a, TargetSpace::exp(b,epsXi)) - energy(a,TargetSpace::exp(b,epsEta));
            double centerValue   = energy(a,b)                   - energy(a,b)              - energy(a,b);
            double backwardValue = energy(a,TargetSpace::exp(b,minusEpsXi + minusEpsEta)) - energy(a, TargetSpace::exp(b,minusEpsXi)) - energy(a,TargetSpace::exp(b,minusEpsEta));
            
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
            assert( std::fabs(B[i]*B[j] - (i==j)) < 1e-10 );
}

template <class TargetSpace>
void testDerivativeOfSquaredDistance(const TargetSpace& a, const TargetSpace& b)
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
        
        typename TargetSpace::EmbeddedTangentVector fwVariation = B[i];
        typename TargetSpace::EmbeddedTangentVector bwVariation = B[i];
        fwVariation *= eps;
        bwVariation *= -eps;
        TargetSpace bPlus  = TargetSpace::exp(b,fwVariation);
        TargetSpace bMinus = TargetSpace::exp(b,bwVariation);

        d2_fd[i] = (energy(a,bPlus) - energy(a,bMinus)) / (2*eps);
    }

    // transform into embedded coordinates
    typename TargetSpace::EmbeddedTangentVector d2_fd_embedded;
    B.mtv(d2_fd,d2_fd_embedded);
    
    if ( (d2 - d2_fd_embedded).infinity_norm() > 100*eps ) {
        std::cout << className(a) << ": Analytical gradient does not match fd approximation." << std::endl;
        std::cout << "d2 Analytical: " << d2 << std::endl;
        std::cout << "d2 FD        : " << d2_fd << std::endl;
        assert(false);
    }
    
}

template <class TargetSpace>
void testHessianOfSquaredDistance(const TargetSpace& a, const TargetSpace& b)
{
    static const int embeddedDim = TargetSpace::embeddedDim;
    
    ///////////////////////////////////////////////////////////////////
    //  Test second derivative with respect to second argument
    ///////////////////////////////////////////////////////////////////
    FieldMatrix<double,embeddedDim,embeddedDim> d2d2 = TargetSpace::secondDerivativeOfDistanceSquaredWRTSecondArgument(a, b);
    
    // finite-difference approximation
    FieldMatrix<double,embeddedDim,embeddedDim> d2d2_fd = getSecondDerivativeOfSecondArgumentFD<TargetSpace,embeddedDim>(a,b);
    
    FieldMatrix<double,embeddedDim,embeddedDim> d2d2_diff = d2d2;
    d2d2_diff -= d2d2_fd;
    if ( (d2d2_diff).infinity_norm() > 100*eps) {
        std::cout << className(a) << ": Analytical second derivative does not match fd approximation." << std::endl;
        std::cout << "d2d2 Analytical:" << std::endl << d2d2 << std::endl;
        std::cout << "d2d2 FD        :" << std::endl << d2d2_fd << std::endl;
        assert(false);
    }
    
}

template <class TargetSpace>
void testMixedDerivativesOfSquaredDistance(const TargetSpace& a, const TargetSpace& b)
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
                
            d1d2_fd[i][j] = (energy<TargetSpace>(aPlus,bPlus) + energy<TargetSpace>(aMinus,bMinus)
                            - energy<TargetSpace>(aPlus,bMinus) - energy<TargetSpace>(aMinus,bPlus)) / (4*eps*eps);

        }
    }
    
    FieldMatrix<double,embeddedDim,embeddedDim> d1d2_diff = d1d2;
    d1d2_diff -= d1d2_fd;
    if ( (d1d2_diff).infinity_norm() > 100*eps ) {
        std::cout << className(a) << ": Analytical mixed second derivative does not match fd approximation." << std::endl;
        std::cout << "d1d2 Analytical:" << std::endl << d1d2 << std::endl;
        std::cout << "d1d2 FD        :" << std::endl << d1d2_fd << std::endl;
        assert(false);
    }

}


template <class TargetSpace>
void testDerivativeOfHessianOfSquaredDistance(const TargetSpace& a, const TargetSpace& b)
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
    
    if ( (d2d2d2 - d2d2d2_fd).infinity_norm() > 100*eps) {
        std::cout << className(a) << ": Analytical third derivative does not match fd approximation." << std::endl;
        std::cout << "d2d2d2 Analytical:" << std::endl << d2d2d2 << std::endl;
        std::cout << "d2d2d2 FD        :" << std::endl << d2d2d2_fd << std::endl;
        assert(false);
    }

}


template <class TargetSpace>
void testMixedDerivativeOfHessianOfSquaredDistance(const TargetSpace& a, const TargetSpace& b)
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
    
    if ( (d1d2d2 - d1d2d2_fd).infinity_norm() > 100*eps ) {
        std::cout << className(a) << ": Analytical mixed third derivative does not match fd approximation." << std::endl;
        std::cout << "d1d2d2 Analytical:" << std::endl << d1d2d2 << std::endl;
        std::cout << "d1d2d2 FD        :" << std::endl << d1d2d2_fd << std::endl;
        assert(false);
    }

}


template <class TargetSpace>
void testDerivativesOfSquaredDistance(const TargetSpace& a, const TargetSpace& b)
{
    
    ///////////////////////////////////////////////////////////////////
    //  Test derivative with respect to second argument
    ///////////////////////////////////////////////////////////////////
    
    testDerivativeOfSquaredDistance<TargetSpace>(a,b);
    
    ///////////////////////////////////////////////////////////////////
    //  Test second derivative with respect to second argument
    ///////////////////////////////////////////////////////////////////

    testHessianOfSquaredDistance<TargetSpace>(a,b);

    //////////////////////////////////////////////////////////////////////////////
    //  Test mixed second derivative with respect to first and second argument
    //////////////////////////////////////////////////////////////////////////////

    testMixedDerivativesOfSquaredDistance<TargetSpace>(a,b);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    //  Test third derivative with respect to second argument
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    testDerivativeOfHessianOfSquaredDistance<TargetSpace>(a,b);

    /////////////////////////////////////////////////////////////////////////////////////////////
    //  Test mixed third derivative with respect to first (once) and second (twice) argument
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    testMixedDerivativeOfHessianOfSquaredDistance<TargetSpace>(a,b);

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
        
        testOrthonormalFrame<TargetSpace>(testPoints[i]);
        
        for (int j=0; j<nTestPoints; j++) {
            
            std::vector<TargetSpace> testPointPair(2);
            testPointPair[0] = testPoints[i];
            testPointPair[1] = testPoints[j];
            if (diameter(testPointPair) > 0.5*M_PI)
                continue;
            
            testDerivativesOfSquaredDistance<TargetSpace>(testPoints[i], testPoints[j]);
            
        }
        
    }
    
}


int main() try
{
    test<UnitVector<double,2> >();
    test<UnitVector<double,3> >();
    test<UnitVector<double,4> >();

    test<Rotation<double,3> >();
    
    test<RigidBodyMotion<double,3> >();
    
} catch (Exception e) {

    std::cout << e << std::endl;

}

