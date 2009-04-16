#ifndef CONFIGURATION_HH
#define CONFIGURATION_HH

#include <dune/common/fvector.hh>
#include "rotation.hh"

/** \brief Configuration of a nonlinear rod in 3d */
struct Configuration 
{
    // Translational part
    Dune::FieldVector<double,3> r;

    // Rotational part
    Rotation<3,double> q;

};

//! Send configuration to output stream
std::ostream& operator<< (std::ostream& s, const Configuration& c)
  {
      s << "(" << c.r << ")  (" << c.q << ")";
      return s;
  }

#endif
