#ifndef ROTATION_HH
#define ROTATION_HH

/** \file
    \brief Define rotations in Euclidean spaces
*/

#include <dune/common/fvector.hh>
#include <dune/common/fixedarray.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/exceptions.hh>

#include "quaternion.hh"


template <int dim, class T>
class Rotation
{

};

/** \brief Specialization for dim==3 

Uses unit quaternion coordinates.
*/
template <class T>
class Rotation<3,T> : public Quaternion<T>
{

    /** \brief Computes sin(x/2) / x without getting unstable for small x */
    static T sincHalf(const T& x) {
        return (x < 1e-4) ? 0.5 + (x*x/48) : std::sin(x/2)/x;
    }

public:

    /** \brief Default constructor creates the identity element */
    Rotation()
        : Quaternion<T>(0,0,0,1)
    {}

    Rotation<3,T>(Dune::FieldVector<T,3> axis, T angle) 
        : Quaternion<T>(axis, angle)
    {}

    /** \brief Assignment from a quaternion
        \deprecated Using this is bad design.
    */
    Rotation& operator=(const Quaternion<T>& other) {
        (*this)[0] = other[0];
        (*this)[1] = other[1];
        (*this)[2] = other[2];
        (*this)[3] = other[3];
        return *this;
    }

    /** \brief Return the identity element */
    static Rotation<3,T> identity() {
        // Default constructor creates an identity
        Rotation<3,T> id;
        return id;
    }

    /** \brief Right multiplication */
    Rotation<3,T> mult(const Rotation<3,T>& other) const {
        Rotation<3,T> q;
        q[0] =   (*this)[3]*other[0] - (*this)[2]*other[1] + (*this)[1]*other[2] + (*this)[0]*other[3];
        q[1] =   (*this)[2]*other[0] + (*this)[3]*other[1] - (*this)[0]*other[2] + (*this)[1]*other[3];
        q[2] = - (*this)[1]*other[0] + (*this)[0]*other[1] + (*this)[3]*other[2] + (*this)[2]*other[3];
        q[3] = - (*this)[0]*other[0] - (*this)[1]*other[1] - (*this)[2]*other[2] + (*this)[3]*other[3];

        return q;
    }

    /** \brief The exponential map from \f$ \mathfrak{so}(3) \f$ to \f$ SO(3) \f$
     */
    static Rotation<3,T> exp(const T& v0, const T& v1, const T& v2) {
        Rotation<3,T> q;

        T normV = std::sqrt(v0*v0 + v1*v1 + v2*v2);

        // Stabilization for small |v| due to Grassia
        T sin = sincHalf(normV);

        // if normV == 0 then q = (0,0,0,1)
        assert(!isnan(sin));
            
        q[0] = sin * v0;
        q[1] = sin * v1;
        q[2] = sin * v2;
        q[3] = std::cos(normV/2);

        return q;
    }
    
};

#endif
