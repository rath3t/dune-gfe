#include <config.h>

#include <iostream>

#include <dune/common/fvector.hh>
#include <dune/grid/common/quadraturerules.hh>

#include <dune/src/rotation.hh>
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
    int quadOrder = 1;

    const Dune::QuadratureRule<double, dim>& quad 
        = Dune::QuadratureRules<double, dim>::rule(GeometryType(GeometryType::simplex,dim), quadOrder);
    
    for (size_t pt=0; pt<quad.size(); pt++) {
        
        const Dune::FieldVector<double,dim>& quadPos = quad[pt].position();

        Dune::FieldVector<double,dim> l0 = quadPos;
        Dune::FieldVector<double,dim> l1, l2;

        l1[0] = 1-quadPos[0]-quadPos[1];
        l1[1] = quadPos[0];

        l2[0] = quadPos[1];
        l2[1] = 1-quadPos[0]-quadPos[1];

        
        std::cout << f0.evaluate(l0) << std::endl;
        std::cout << f1.evaluate(l1) << std::endl;
        std::cout << f2.evaluate(l2) << std::endl;

    }

}

int main()
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
