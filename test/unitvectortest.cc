#include <config.h>

#include <dune/gfe/unitvector.hh>
#include <dune/gfe/rotation.hh>

/** \file
    \brief Unit tests for the UnitVector class
*/

using namespace Dune;

const double eps = 1e-6;

template <int dim>
double energy(const UnitVector<dim>& a, const UnitVector<dim>& b)
{
    return UnitVector<dim>::distance(a,b) * UnitVector<dim>::distance(a,b);
}

template <int dim>
double energy(const UnitVector<dim>& a, const FieldVector<double,dim>& b)
{
    return UnitVector<dim>::distance(a,b) * UnitVector<dim>::distance(a,b);
}

template <int dim>
void testDerivativesOfSquaredDistance(const UnitVector<dim>& a, const UnitVector<dim>& b)
{
    
    ///////////////////////////////////////////////////////////////////
    //  Test derivative with respect to second argument
    ///////////////////////////////////////////////////////////////////
    typename UnitVector<dim>::EmbeddedTangentVector d2 =  UnitVector<dim>::derivativeOfDistanceSquaredWRTSecondArgument(a, b);    

    // finite-difference approximation
    typename UnitVector<dim>::EmbeddedTangentVector d2_fd;
    for (size_t i=0; i<dim; i++) {
        FieldVector<double,dim> bPlus  = b.globalCoordinates();
        FieldVector<double,dim> bMinus = b.globalCoordinates();
        bPlus[i]  += eps;
        bMinus[i] -= eps;
        d2_fd[i] = (energy(a,bPlus) - energy(a,bMinus)) / (2*eps);
    }
    
    std::cout << "Analytical: " << d2 << std::endl;
    std::cout << "FD        : " << d2_fd << std::endl;
    
    
    ///////////////////////////////////////////////////////////////////
    //  Test second derivative with respect to second argument
    ///////////////////////////////////////////////////////////////////
    FieldMatrix<double,dim,dim> d2d2 = UnitVector<dim>::secondDerivativeOfDistanceSquaredWRTSecondArgument(a, b);
    
    // finite-difference approximation
    FieldMatrix<double,dim,dim> d2d2_fd;
    
    for (size_t i=0; i<dim; i++) {
        for (size_t j=0; j<dim; j++) {
            
            if (i==j) {
            
                FieldVector<double,dim> bPlus  = b.globalCoordinates();
                FieldVector<double,dim> bMinus = b.globalCoordinates();
                bPlus[i]  += eps;
                bMinus[i] -= eps;

                d2d2_fd[i][i] = (energy(a,bPlus) - 2*energy(a,b) + energy(a,bMinus)) / (eps*eps);
            } else {
                
                FieldVector<double,dim> bPlusPlus   = b.globalCoordinates();
                FieldVector<double,dim> bPlusMinus  = b.globalCoordinates();
                FieldVector<double,dim> bMinusPlus  = b.globalCoordinates();
                FieldVector<double,dim> bMinusMinus = b.globalCoordinates();
            
                bPlusPlus[i]   += eps;  bPlusPlus[j]   += eps;
                bPlusMinus[i]  += eps;  bPlusMinus[j]  -= eps;
                bMinusPlus[i]  -= eps;  bMinusPlus[j]  += eps;
                bMinusMinus[i] -= eps;  bMinusMinus[j] -= eps;
            
                d2d2_fd[i][j] = (energy(a,bPlusPlus) + energy(a,bMinusMinus)
                                        - energy(a,bPlusMinus) - energy(a,bMinusPlus)) / (4*eps*eps);
            }
        }
    }
    
    std::cout << "Analytical:\n" << d2d2 << std::endl;
    std::cout << "FD        :\n" << d2d2_fd << std::endl;
    
    //////////////////////////////////////////////////////////////////////////////
    //  Test mixed second derivative with respect to first and second argument
    //////////////////////////////////////////////////////////////////////////////

    FieldMatrix<double,dim,dim> d1d2 = UnitVector<dim>::secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(a, b);
    
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    //  Test mixed third derivative with respect to first (once) and second (twice) argument
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    Tensor3<double,dim,dim,dim> d1d2d2 = UnitVector<dim>::thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(a, b);
}

int main()
{
    // Set up elements of S^1
    Dune::array<double,2> w0 = {{0,1}};
    UnitVector<2> uv0(w0);
    Dune::array<double,2> w1 = {{1,1}};
    UnitVector<2> uv1(w1);
    testDerivativesOfSquaredDistance<2>(uv0, uv1);
    
    Dune::array<double,3> w3_0 = {{0,1,0}};
    UnitVector<3> v3_0(w3_0);
    Dune::array<double,3> w3_1 = {{1,1,0}};
    UnitVector<3> v3_1(w3_1);
    testDerivativesOfSquaredDistance<3>(v3_0, v3_1);
    
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

