#include <config.h>

#include <dune/gfe/unitvector.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/rotation.hh>

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
template <class TargetSpace, int dim>
FieldMatrix<double,dim,dim> getSecondDerivativeOfSecondArgumentFD(const TargetSpace& a, const TargetSpace& b)
{
    // finite-difference approximation
    FieldMatrix<double,dim,dim> d2d2_fd;
    
    for (size_t i=0; i<dim; i++) {
        for (size_t j=0; j<dim; j++) {

            FieldVector<double,dim> epsXi(0);
            epsXi[i] = eps;
            FieldVector<double,dim> epsEta(0);
            epsEta[j] = eps;
            
            FieldVector<double,dim> minusEpsXi  = epsXi;   minusEpsXi  *= -1;
            FieldVector<double,dim> minusEpsEta = epsEta;  minusEpsEta *= -1;
            
            double forwardValue  = energy(a,TargetSpace::exp(b,epsXi+epsEta)) - energy(a, TargetSpace::exp(b,epsXi)) - energy(a,TargetSpace::exp(b,epsEta));
            double centerValue   = energy(a,b)                   - energy(a,b)              - energy(a,b);
            double backwardValue = energy(a,TargetSpace::exp(b,minusEpsXi + minusEpsEta)) - energy(a, TargetSpace::exp(b,minusEpsXi)) - energy(a,TargetSpace::exp(b,minusEpsEta));
            
            d2d2_fd[i][j] = 0.5 * (forwardValue - 2*centerValue + backwardValue) / (eps*eps);
            
        }
    }
    
    FieldMatrix<double,dim,dim> B = b.orthonormalFrame();
    B.invert();
    FieldMatrix<double,dim,dim> BT;
    for (int i=0; i<dim; i++)
        for (int j=0; j<dim; j++)
            BT[i][j] = B[j][i];
    
    
    FieldMatrix<double,dim,dim> ret1;
    FMatrixHelp::multMatrix(d2d2_fd,B,ret1);
    
    FieldMatrix<double,dim,dim> ret2;
    FMatrixHelp::multMatrix(BT,ret1,ret2);
    return ret2;
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
        std::cout << "Analytical gradient does not match fd approximation." << std::endl;
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
        std::cout << "Analytical second derivative does not match fd approximation." << std::endl;
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
        std::cout << "Analytical mixed second derivative does not match fd approximation." << std::endl;
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
        std::cout << "Analytical mixed third derivative does not match fd approximation." << std::endl;
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
    //  Test mixed third derivative with respect to first (once) and second (twice) argument
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    testDerivativeOfHessianOfSquaredDistance<TargetSpace,dim>(a,b);

}


int main()
{
#if 0
    // Set up elements of S^1
    Dune::array<double,2> w0 = {{1,0}};
    UnitVector<2> uv0(w0);
    Dune::array<double,2> w1 = {{0,1}};
    UnitVector<2> uv1(w1);
    testDerivativesOfSquaredDistance<UnitVector<2>, 2>(uv0, uv1);
#endif
    
    // Set up elements of R^1
    Dune::FieldVector<double,2> w0;  w0[0] = 0;  w0[1] = 1;
    RealTuple<2> uv0(w0);
    Dune::FieldVector<double,2> w1;  w1[0] = 1;  w1[1] = 0;
    RealTuple<2> uv1(w1);
    testDerivativesOfSquaredDistance<RealTuple<2>, 2>(uv0, uv1);
//     Dune::array<double,3> w3_0 = {{0,1,0}};
//     UnitVector<3> v3_0(w3_0);
//     Dune::array<double,3> w3_1 = {{1,1,0}};
//     UnitVector<3> v3_1(w3_1);
//     testDerivativesOfSquaredDistance<3>(v3_0, v3_1);
    
#if 0
    // Set up elements of S^1
    FieldVector<double,2> v;
    v[0] = 1;  v[1] = 1;
    UnitVector<2> uv1;  uv1 = v;
    v[0] = 0;  v[1] = 1;
    UnitVector<2> uv0;  uv0 = v;

    // Set up elements of SO(2)
    Rotation<2,double> ro1(M_PI/4);
    Rotation<2,double> ro0(M_PI/2);

    std::cout << UnitVector<2>::distance(uv0, uv1) << std::endl;
    std::cout << Rotation<2,double>::distance(ro0, ro1) << std::endl;

    std::cout << UnitVector<2>::derivativeOfDistanceSquaredWRTSecondArgument(uv0, uv1) << std::endl;
    std::cout << Rotation<2,double>::derivativeOfDistanceSquaredWRTSecondArgument(ro0, ro1) << std::endl;

    std::cout << UnitVector<2>::secondDerivativeOfDistanceSquaredWRTSecondArgument(uv0, uv1) << std::endl;
    std::cout << Rotation<2,double>::secondDerivativeOfDistanceSquaredWRTSecondArgument(ro0, ro1) << std::endl;
#endif
}

