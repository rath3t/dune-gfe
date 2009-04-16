#ifndef RIGID_BODY_MOTION_HH
#define RIGID_BODY_MOTION_HH

#include <dune/common/fvector.hh>
#include "rotation.hh"

/** \brief A rigid-body motion in, R^d, i.e., a member of SE(d) */
template <int dim, class ctype=double>
struct RigidBodyMotion
{
    // Translational part
    Dune::FieldVector<ctype, dim> r;

    // Rotational part
    Rotation<dim,ctype> q;

};

//! Send configuration to output stream
template <int dim, class ctype>
std::ostream& operator<< (std::ostream& s, const RigidBodyMotion<dim,ctype>& c)
  {
      s << "(" << c.r << ")  (" << c.q << ")";
      return s;
  }

#endif
