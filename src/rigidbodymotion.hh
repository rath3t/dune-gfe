#ifndef RIGID_BODY_MOTION_HH
#define RIGID_BODY_MOTION_HH

#include <dune/common/fvector.hh>
#include "rotation.hh"

/** \brief A rigid-body motion in, R^d, i.e., a member of SE(d) */
template <int dim, class ctype=double>
struct RigidBodyMotion
{
    /** \brief Type of an infinitesimal rigid body motion */
    typedef Dune::FieldVector<ctype, (dim==3) ? 6 : 3> TangentVector;

    /** \brief Compute difference vector from a to b on the tangent space of a */
    static TangentVector difference(const RigidBodyMotion<dim,ctype>& a,
                                    const RigidBodyMotion<dim,ctype>& b) {

        TangentVector result;

        // Usual linear difference
        for (int i=0; i<dim; i++)
            result[i] = a.r[i] - b.r[i];

        // Subtract orientations on the tangent space of 'a'
        typename Rotation<dim,ctype>::TangentVector v = Rotation<dim,ctype>::difference(a.q, b.q);

        // Compute difference on T_a SO(3)
        for (int i=0; i<Rotation<dim,ctype>::TangentVector::size; i++)
            result[i+dim] = v[i];

        return result;
    }

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
