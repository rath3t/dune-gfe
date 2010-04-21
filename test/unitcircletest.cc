#include <config.h>

#include <dune/src/unitvector.hh>
#include <dune/src/rotation.hh>


using namespace Dune;

int main()
{
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
}

