#ifndef CONFIGURATION_HH
#define CONFIGURATION_HH

#include <dune/common/fvector.hh>
#include "quaternion.hh"

/** \brief Configuration of a nonlinear rod in 3d */
struct Configuration 
{
    // Translational part
    Dune::FieldVector<double,3> r;

    // Rotational part
    Quaternion<double> q;

};

#endif
