#include <config.h>

#include <dune/gfe/unitvector.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/rotation.hh>

#include "valuefactory.hh"

using Dune::FieldVector;


/** \file
    \brief Unit tests for the UnitVector class
*/

using namespace Dune;

const double eps = 1e-4;

template <class TargetSpace>
double energy(const TargetSpace& a, const TargetSpace& b)
{
    return TargetSpace::distance(a,b) * TargetSpace::distance(a,b);
}

template <class TargetSpace, int dim>
double energy(const TargetSpace& a, const FieldVector<double,dim>& b)
{
    return TargetSpace::distance(a,b) * TargetSpace::distance(a,b);
}

template <class TargetSpace, int dim>
double energy(const FieldVector<double,dim>& a, const FieldVector<double,dim>& b)
{
    return TargetSpace::distance(a,b) * TargetSpace::distance(a,b);
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

template <class TargetSpace, int worldDim>
void testOrthonormalFrame(const TargetSpace& a)
{
    const size_t spaceDim = TargetSpace::dim;
    FieldMatrix<double,spaceDim,worldDim> B = a.orthonormalFrame();

    for (size_t i=0; i<spaceDim; i++)
        for (size_t j=0; j<spaceDim; j++)
            assert( std::fabs(B[i]*B[j] - (i==j)) < 1e-10 );
}

template <class TargetSpace, int dim>
void testDerivativeOfSquaredDistance(const TargetSpace& a, const TargetSpace& b)
{
    
    ///////////////////////////////////////////////////////////////////
    //  Test derivative with respect to second argument
    ///////////////////////////////////////////////////////////////////
    typename TargetSpace::EmbeddedTangentVector d2 =  TargetSpace::derivativeOfDistanceSquaredWRTSecondArgument(a, b);    

    // finite-difference approximation
    typename TargetSpace::EmbeddedTangentVector d2_fd;
    for (size_t i=0; i<dim; i++) {
        FieldVector<double,dim> bPlus  = b.globalCoordinates();
        FieldVector<double,dim> bMinus = b.globalCoordinates();
        bPlus[i]  += eps;
        bMinus[i] -= eps;
        d2_fd[i] = (energy(a,bPlus) - energy(a,bMinus)) / (2*eps);
    }
    
    if ( (d2 - d2_fd).infinity_norm() > 100*eps ) {
        std::cout << className(a) << ": Analytical gradient does not match fd approximation." << std::endl;
        std::cout << "d2 Analytical: " << d2 << std::endl;
        std::cout << "d2 FD        : " << d2_fd << std::endl;
    }
    
}

template <class TargetSpace, int dim>
void testHessianOfSquaredDistance(const TargetSpace& a, const TargetSpace& b)
{
    
    ///////////////////////////////////////////////////////////////////
    //  Test second derivative with respect to second argument
    ///////////////////////////////////////////////////////////////////
    FieldMatrix<double,dim,dim> d2d2 = TargetSpace::secondDerivativeOfDistanceSquaredWRTSecondArgument(a, b);
    
    // finite-difference approximation
    FieldMatrix<double,dim,dim> d2d2_fd = getSecondDerivativeOfSecondArgumentFD<TargetSpace,dim>(a,b);
    
    FieldMatrix<double,dim,dim> d2d2_diff = d2d2;
    d2d2_diff -= d2d2_fd;
    if ( (d2d2_diff).infinity_norm() > 100*eps) {
        std::cout << className(a) << ": Analytical second derivative does not match fd approximation." << std::endl;
        std::cout << "d2d2 Analytical:" << std::endl << d2d2 << std::endl;
        std::cout << "d2d2 FD        :" << std::endl << d2d2_fd << std::endl;
    }
    
}

template <class TargetSpace, int dim>
void testMixedDerivativesOfSquaredDistance(const TargetSpace& a, const TargetSpace& b)
{
    //////////////////////////////////////////////////////////////////////////////
    //  Test mixed second derivative with respect to first and second argument
    //////////////////////////////////////////////////////////////////////////////

    FieldMatrix<double,dim,dim> d1d2 = TargetSpace::secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(a, b);
    
    // finite-difference approximation
    FieldMatrix<double,dim,dim> d1d2_fd;
    
    for (size_t i=0; i<dim; i++) {
        for (size_t j=0; j<dim; j++) {
            
            FieldVector<double,dim> aPlus  = a.globalCoordinates();
            FieldVector<double,dim> aMinus = a.globalCoordinates();
            aPlus[i]  += eps;
            aMinus[i] -= eps;

            FieldVector<double,dim> bPlus  = b.globalCoordinates();
            FieldVector<double,dim> bMinus = b.globalCoordinates();
            bPlus[j]  += eps;
            bMinus[j] -= eps;
                
            d1d2_fd[i][j] = (energy<TargetSpace>(aPlus,bPlus) + energy<TargetSpace>(aMinus,bMinus)
                            - energy<TargetSpace>(aPlus,bMinus) - energy<TargetSpace>(aMinus,bPlus)) / (4*eps*eps);

        }
    }
    
    FieldMatrix<double,dim,dim> d1d2_diff = d1d2;
    d1d2_diff -= d1d2_fd;
    if ( (d1d2_diff).infinity_norm() > 100*eps ) {
        std::cout << className(a) << ": Analytical mixed second derivative does not match fd approximation." << std::endl;
        std::cout << "d1d2 Analytical:" << std::endl << d1d2 << std::endl;
        std::cout << "d1d2 FD        :" << std::endl << d1d2_fd << std::endl;
    }

}


template <class TargetSpace, int dim>
void testDerivativeOfHessianOfSquaredDistance(const TargetSpace& a, const TargetSpace& b)
{
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    //  Test mixed third derivative with respect to first (once) and second (twice) argument
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    Tensor3<double,dim,dim,dim> d2d2d2 = TargetSpace::thirdDerivativeOfDistanceSquaredWRTSecondArgument(a, b);
    
    Tensor3<double,dim,dim,dim> d2d2d2_fd;
    
    for (size_t i=0; i<dim; i++) {
        
        FieldVector<double,dim> bPlus  = b.globalCoordinates();
        FieldVector<double,dim> bMinus = b.globalCoordinates();
        bPlus[i]  += eps;
        bMinus[i] -= eps;

        FieldMatrix<double,dim,dim> hPlus  = getSecondDerivativeOfSecondArgumentFD<TargetSpace,dim>(a,TargetSpace(bPlus));
        FieldMatrix<double,dim,dim> hMinus = getSecondDerivativeOfSecondArgumentFD<TargetSpace,dim>(a,TargetSpace(bMinus));
        
        d2d2d2_fd[i] = hPlus;
        d2d2d2_fd[i] -= hMinus;
        d2d2d2_fd[i] /= 2*eps;
        
    }
    
    if ( (d2d2d2 - d2d2d2_fd).infinity_norm() > 100*eps) {
        std::cout << className(a) << ": Analytical third derivative does not match fd approximation." << std::endl;
        std::cout << "d2d2d2 Analytical:" << std::endl << d2d2d2 << std::endl;
        std::cout << "d2d2d2 FD        :" << std::endl << d2d2d2_fd << std::endl;
    }

}


template <class TargetSpace, int dim>
void testMixedDerivativeOfHessianOfSquaredDistance(const TargetSpace& a, const TargetSpace& b)
{
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    //  Test mixed third derivative with respect to first (once) and second (twice) argument
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    Tensor3<double,dim,dim,dim> d1d2d2 = TargetSpace::thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(a, b);
    
    Tensor3<double,dim,dim,dim> d1d2d2_fd;
    
    for (size_t i=0; i<dim; i++) {
        
        FieldVector<double,dim> aPlus  = a.globalCoordinates();
        FieldVector<double,dim> aMinus = a.globalCoordinates();
        aPlus[i]  += eps;
        aMinus[i] -= eps;

        FieldMatrix<double,dim,dim> hPlus  = getSecondDerivativeOfSecondArgumentFD<TargetSpace,dim>(TargetSpace(aPlus),b);
        FieldMatrix<double,dim,dim> hMinus = getSecondDerivativeOfSecondArgumentFD<TargetSpace,dim>(TargetSpace(aMinus),b);
        
        d1d2d2_fd[i] = hPlus;
        d1d2d2_fd[i] -= hMinus;
        d1d2d2_fd[i] /= 2*eps;
        
    }
    
    if ( (d1d2d2 - d1d2d2_fd).infinity_norm() > 100*eps ) {
        std::cout << className(a) << ": Analytical mixed third derivative does not match fd approximation." << std::endl;
        std::cout << "d1d2d2 Analytical:" << std::endl << d1d2d2 << std::endl;
        std::cout << "d1d2d2 FD        :" << std::endl << d1d2d2_fd << std::endl;
    }

}


template <class TargetSpace, int dim>
void testDerivativesOfSquaredDistance(const TargetSpace& a, const TargetSpace& b)
{
    
    ///////////////////////////////////////////////////////////////////
    //  Test derivative with respect to second argument
    ///////////////////////////////////////////////////////////////////
    
    testDerivativeOfSquaredDistance<TargetSpace,dim>(a,b);
    
    ///////////////////////////////////////////////////////////////////
    //  Test second derivative with respect to second argument
    ///////////////////////////////////////////////////////////////////

    testHessianOfSquaredDistance<TargetSpace,dim>(a,b);

    //////////////////////////////////////////////////////////////////////////////
    //  Test mixed second derivative with respect to first and second argument
    //////////////////////////////////////////////////////////////////////////////

    testMixedDerivativesOfSquaredDistance<TargetSpace,dim>(a,b);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    //  Test third derivative with respect to second argument
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    testDerivativeOfHessianOfSquaredDistance<TargetSpace,dim>(a,b);

    /////////////////////////////////////////////////////////////////////////////////////////////
    //  Test mixed third derivative with respect to first (once) and second (twice) argument
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    testMixedDerivativeOfHessianOfSquaredDistance<TargetSpace,dim>(a,b);

}

void testUnitVector2d()
{
    std::vector<UnitVector<2> > testPoints;
    ValueFactory<UnitVector<2> >::get(testPoints);
    
    int nTestPoints = testPoints.size();
    
    // Set up elements of S^1
    for (int i=0; i<nTestPoints; i++) {
        
        testOrthonormalFrame<UnitVector<2>, 2>(testPoints[i]);
        
        for (int j=0; j<nTestPoints; j++) {
            
            if (UnitVector<2>::distance(testPoints[i],testPoints[j]) > M_PI*0.98)
                continue;

            testDerivativesOfSquaredDistance<UnitVector<2>, 2>(testPoints[i], testPoints[j]);
            
        }
        
    }
}

void testUnitVector3d()
{
    std::vector<UnitVector<3> > testPoints;
    ValueFactory<UnitVector<3> >::get(testPoints);
    
    int nTestPoints = testPoints.size();
                                  
    // Set up elements of S^2
    for (int i=0; i<nTestPoints; i++) {
        
        testOrthonormalFrame<UnitVector<3>, 3>(testPoints[i]);
        
        for (int j=0; j<nTestPoints; j++) {
            
            testDerivativesOfSquaredDistance<UnitVector<3>, 3>(testPoints[i], testPoints[j]);
            
        }
        
    }
    
}

void testRotation3d()
{
    int nTestPoints = 10;
    double testPoints[10][4] = {{1,0,0,0}, {0,1,0,0}, {-0.838114,0.356751,-0.412667,0.5},
                               {-0.490946,-0.306456,0.81551,0.23},{-0.944506,0.123687,-0.304319,-0.7},
                               {-0.6,0.1,-0.2,0.8},{0.45,0.12,0.517,0},
                               {-0.1,0.3,-0.1,0.73},{-0.444506,0.123687,0.104319,-0.23},{-0.7,-0.123687,-0.304319,0.72}};
                                  
    // Set up elements of SO(3)
    for (int i=0; i<nTestPoints; i++) {
        
        Dune::array<double,4> w0 = {{testPoints[i][0], testPoints[i][1], testPoints[i][2], testPoints[i][3]}};
        Rotation<3,double> uv0(w0);

        testOrthonormalFrame<Rotation<3,double>, 4>(uv0);

        for (int j=0; j<nTestPoints; j++) {
            
            Dune::array<double,4> w1 = {{testPoints[j][0], testPoints[j][1], testPoints[j][2], testPoints[i][3]}};
            Rotation<3,double> uv1(w1);
        
            testDerivativesOfSquaredDistance<Rotation<3,double>, 4>(uv0, uv1);
            
        }

    }
    
}

int main() try
{
    testUnitVector2d();
    testUnitVector3d();

    testRotation3d();
} catch (Exception e) {

    std::cout << e << std::endl;

}

