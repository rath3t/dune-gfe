#include <config.h>

#include <fenv.h>
#include <iostream>

#include <dune/common/fvector.hh>
#include <dune/grid/common/quadraturerules.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/unitvector.hh>

#include <dune/gfe/localgeodesicfefunction.hh>

// Domain dimension
const int dim = 2;

const double eps = 1e-6;

using namespace Dune;

void testDerivativeTangentiality(const RealTuple<1>& x,
                                 const FieldMatrix<double,1,dim>& derivative)
{
    // By construction, derivatives of RealTuples are always tangent
}

// the columns of the derivative must be tangential to the manifold
template <int vectorDim>
void testDerivativeTangentiality(const UnitVector<vectorDim>& x,
                                 const FieldMatrix<double,vectorDim,dim>& derivative)
{
    for (int i=0; i<dim; i++) {

        // The i-th column is a tangent vector if its scalar product with the global coordinates
        // of x vanishes.
        double sp = 0;
        for (int j=0; j<vectorDim; j++)
            sp += x.globalCoordinates()[j] * derivative[j][i];

        if (std::fabs(sp) > 1e-8)
            DUNE_THROW(Dune::Exception, "Derivative is not tangential: Column: " << i << ",  product: " << sp);

    }

}

/** \brief Test whether interpolation is invariant under permutation of the simplex vertices
 */
template <class TargetSpace>
void testPermutationInvariance(const std::vector<TargetSpace>& corners)
{
    std::vector<TargetSpace> cornersRotated1(dim+1);
    std::vector<TargetSpace> cornersRotated2(dim+1);

    cornersRotated1[0] = cornersRotated2[2] = corners[1];
    cornersRotated1[1] = cornersRotated2[0] = corners[2];
    cornersRotated1[2] = cornersRotated2[1] = corners[0];
    
    LocalGeodesicFEFunction<2,double,TargetSpace> f0(corners);
    LocalGeodesicFEFunction<2,double,TargetSpace> f1(cornersRotated1);
    LocalGeodesicFEFunction<2,double,TargetSpace> f2(cornersRotated2);

    // A quadrature rule as a set of test points
    int quadOrder = 3;

    const Dune::QuadratureRule<double, dim>& quad 
        = Dune::QuadratureRules<double, dim>::rule(GeometryType(GeometryType::simplex,dim), quadOrder);
    
    for (size_t pt=0; pt<quad.size(); pt++) {
        
        const Dune::FieldVector<double,dim>& quadPos = quad[pt].position();

        Dune::FieldVector<double,dim> l0 = quadPos;
        Dune::FieldVector<double,dim> l1, l2;

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

template <class TargetSpace>
void testDerivative(const std::vector<TargetSpace>& corners)
{
    // Make local fe function to be tested
    LocalGeodesicFEFunction<2,double,TargetSpace> f(corners);

    // A quadrature rule as a set of test points
    int quadOrder = 3;
    
    const Dune::QuadratureRule<double, dim>& quad 
        = Dune::QuadratureRules<double, dim>::rule(GeometryType(GeometryType::simplex,dim), quadOrder);
    
    for (size_t pt=0; pt<quad.size(); pt++) {
        
        const Dune::FieldVector<double,dim>& quadPos = quad[pt].position();

        // evaluate actual derivative
        Dune::FieldMatrix<double, TargetSpace::EmbeddedTangentVector::size, dim> derivative = f.evaluateDerivative(quadPos);

        // evaluate fd approximation of derivative
        Dune::FieldMatrix<double, TargetSpace::EmbeddedTangentVector::size, dim> fdDerivative = f.evaluateDerivativeFD(quadPos);

        Dune::FieldMatrix<double, TargetSpace::EmbeddedTangentVector::size, dim> diff = derivative;
        diff -= fdDerivative;
        
        if ( diff.infinity_norm() > 100*eps ) {
            std::cout << className(corners[0]) << ": Analytical gradient does not match fd approximation." << std::endl;
            std::cout << "Analytical: " << derivative << std::endl;
            std::cout << "FD        : " << fdDerivative << std::endl;
        }

        testDerivativeTangentiality(f.evaluate(quadPos), derivative);

    }
}

void testRealTuples()
{
    std::cout << " --- Testing RealTuple<1> ---" << std::endl;

    typedef RealTuple<1> TargetSpace;

    std::vector<TargetSpace> corners = {TargetSpace(1),
                                        TargetSpace(2),
                                        TargetSpace(3)};

    testPermutationInvariance(corners);
    testDerivative(corners);
}

void testUnitVector2d()
{
    std::cout << " --- Testing UnitVector<2> ---" << std::endl;

    typedef UnitVector<2> TargetSpace;

    int nTestPoints = 10;
    double testPoints[10][2] = {{1,0}, {0.5,0.5}, {0,1}, {-0.5,0.5}, {-1,0}, {-0.5,-0.5}, {0,-1}, {0.5,-0.5}, {0.1,1}, {1,.1}};
    assert(dim==2);
    
    // Set up elements of S^1
    std::vector<TargetSpace> corners(dim+1);

    for (int i=0; i<nTestPoints; i++) {
        
        for (int j=0; j<nTestPoints; j++) {

            for (int k=0; k<nTestPoints; k++) {
                
                Dune::array<double,2> w0 = {testPoints[i][0], testPoints[i][1]};
                corners[0] = UnitVector<2>(w0);
                Dune::array<double,2> w1 = {testPoints[j][0], testPoints[j][1]};
                corners[1] = UnitVector<2>(w1);
                Dune::array<double,2> w2 = {testPoints[k][0], testPoints[k][1]};
                corners[2] = UnitVector<2>(w2);

                if (UnitVector<2>::distance(corners[0],corners[1]) > M_PI*0.98
                    or UnitVector<2>::distance(corners[1],corners[2]) > M_PI*0.98
                    or UnitVector<2>::distance(corners[2],corners[0]) > M_PI*0.98)
                    continue;

                testPermutationInvariance(corners);
                testDerivative(corners);
                
            }
        }
    }
}

void testUnitVector3d()
{
    std::cout << " --- Testing UnitVector<3> ---" << std::endl;

    typedef UnitVector<3> TargetSpace;

    int nTestPoints = 10;
    double testPoints[10][3] = {{1,0,0}, {0,1,0}, {-0.838114,0.356751,-0.412667},
                               {-0.490946,-0.306456,0.81551},{-0.944506,0.123687,-0.304319},
                               {-0.6,0.1,-0.2},{0.45,0.12,0.517},
                               {-0.1,0.3,-0.1},{-0.444506,0.123687,0.104319},{-0.7,-0.123687,-0.304319}};
    assert(dim==2);
    
    // Set up elements of S^1
    std::vector<TargetSpace> corners(dim+1);

    for (int i=0; i<nTestPoints; i++) {
        
        for (int j=0; j<nTestPoints; j++) {

            for (int k=0; k<nTestPoints; k++) {
                
                Dune::array<double,3> w0 = {testPoints[i][0], testPoints[i][1], testPoints[i][2]};
                corners[0] = UnitVector<3>(w0);
                Dune::array<double,3> w1 = {testPoints[j][0], testPoints[j][1], testPoints[j][2]};
                corners[1] = UnitVector<3>(w1);
                Dune::array<double,3> w2 = {testPoints[k][0], testPoints[k][1], testPoints[k][2]};
                corners[2] = UnitVector<3>(w2);

                testPermutationInvariance(corners);
                testDerivative(corners);
                
            }
        }
    }

}

void testRotations()
{
    std::cout << " --- Testing Rotation<3> ---" << std::endl;

    typedef Rotation<3,double> TargetSpace;

    FieldVector<double,3> xAxis(0);
    xAxis[0] = 1;
    FieldVector<double,3> yAxis(0);
    yAxis[1] = 1;
    FieldVector<double,3> zAxis(0);
    zAxis[2] = 1;


    std::vector<TargetSpace> corners(dim+1);
    corners[0] = Rotation<3,double>(xAxis,0.1);
    corners[1] = Rotation<3,double>(yAxis,0.1);
    corners[2] = Rotation<3,double>(zAxis,0.1);

    testPermutationInvariance(corners);
    //testDerivative(corners);
}


int main()
{
    // choke on NaN
    feenableexcept(FE_INVALID);

    //testRealTuples();
    testUnitVector2d();
    testUnitVector3d();
    //testRotations();
}
