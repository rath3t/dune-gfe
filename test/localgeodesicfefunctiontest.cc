#include <config.h>

#include <iostream>

#include <dune/common/fvector.hh>
#include <dune/grid/common/quadraturerules.hh>

#include <dune/src/rotation.hh>
#include <dune/src/localgeodesicfefunction.hh>

using namespace Dune;

int main()
{
    typedef Rotation<3,double> TargetSpace;

    const int dim = 2;

    FieldVector<double,3> xAxis(0);
    xAxis[0] = 1;
    FieldVector<double,3> yAxis(0);
    yAxis[1] = 1;
    FieldVector<double,3> zAxis(0);
    zAxis[2] = 1;



    TargetSpace v0 = Rotation<3,double>(xAxis,1);
    TargetSpace v1 = Rotation<3,double>(yAxis,2);
    TargetSpace v2 = Rotation<3,double>(zAxis,3);

    std::vector<TargetSpace> coef0(3), coef1(3), coef2(3);
    coef0[0] = v0;  coef0[1] = v1;  coef0[2] = v2;
    coef1[0] = v2;  coef1[1] = v0;  coef1[2] = v1;
    coef2[0] = v1;  coef2[1] = v2;  coef2[2] = v0;

    GeometryType triangle;
    triangle.makeTriangle();

    LocalGeodesicFEFunction<2,double,TargetSpace> f0(triangle, coef0);
    LocalGeodesicFEFunction<2,double,TargetSpace> f1(triangle, coef1);
    LocalGeodesicFEFunction<2,double,TargetSpace> f2(triangle, coef2);

    int quadOrder = 1;

    const Dune::QuadratureRule<double, dim>& quad 
        = Dune::QuadratureRules<double, dim>::rule(triangle, quadOrder);
    
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
