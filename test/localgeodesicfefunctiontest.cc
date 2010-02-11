#include <config.h>

#include <iostream>

#include <dune/common/fvector.hh>
#include <dune/grid/common/quadraturerules.hh>

#include <dune/src/rotation.hh>
#include <dune/src/realtuple.hh>
#include <dune/src/unitvector.hh>

#include <dune/src/localgeodesicfefunction.hh>

// Domain dimension
const int dim = 2;

using namespace Dune;

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
        assert(TargetSpace::distance(v0,v1) < 1e-5);
        assert(TargetSpace::distance(v0,v2) < 1e-5);

        std::cout << v0 << std::endl;
        std::cout << v1 << std::endl;
        std::cout << v2 << std::endl;

    }

}

void testRealTuples()
{
    typedef RealTuple<1> TargetSpace;

    std::vector<TargetSpace> corners = {TargetSpace(1),
                                        TargetSpace(2),
                                        TargetSpace(3)};

    testPermutationInvariance(corners);
}

void testUnitVectors()
{
    typedef UnitVector<3> TargetSpace;

    std::vector<TargetSpace> corners(dim+1);

    FieldVector<double,3> input;
    input[0] = 1;  input[1] = 0;  input[2] = 0;
    corners[0] = input;
    input[0] = 0;  input[1] = 1;  input[2] = 0;
    corners[1] = input;
    input[0] = 0;  input[1] = 0;  input[2] = 1;
    corners[2] = input;

    testPermutationInvariance(corners);
}

void testRotations()
{
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
}


int main()
{
    //testRealTuples();
    testUnitVectors();
    //testRotations();
}
