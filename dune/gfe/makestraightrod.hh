#ifndef MAKE_STRAIGHT_ROD_HH
#define MAKE_STRAIGHT_ROD_HH

#include <vector>
#include <dune/common/fvector.hh>
#include <dune/fufem/crossproduct.hh>

#include "rigidbodymotion.hh"

/** \brief Make a straight rod from two given endpoints

\param[out] rod The new rod
\param[in] n The number of vertices
*/
template <int dim>
void makeStraightRod(std::vector<RigidBodyMotion<dim> >& rod, int n,
                     const Dune::FieldVector<double,3>& beginning, const Dune::FieldVector<double,3>& end)
{
    // Compute the correct orientation
    Rotation<3,double> orientation = Rotation<3,double>::identity();

    Dune::FieldVector<double,3> zAxis(0);
    zAxis[2] = 1;
    Dune::FieldVector<double,3> axis = crossProduct(Dune::FieldVector<double,3>(end-beginning), zAxis);
    if (axis.two_norm() != 0)
        axis /= -axis.two_norm();

    Dune::FieldVector<double,3> d3 = end-beginning;
    d3 /= d3.two_norm();

    double angle = std::acos(zAxis * d3);

    if (angle != 0)
        orientation = Rotation<3,double>(axis, angle);

    // Set the values
    rod.resize(n);
    for (int i=0; i<n; i++) {

        rod[i].r = beginning;
        rod[i].r.axpy(double(i) / (n-1), end-beginning);
        rod[i].q = orientation;

    }


}

#endif
