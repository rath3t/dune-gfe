#ifndef QUATERNION_HH
#define QUATERNION_HH

#include <dune/common/fvector.hh>
#include <dune/common/exceptions.hh>

template <class T>
class Quaternion : public Dune::FieldVector<T,4>
{
public:

    Quaternion() {}
    Quaternion(const Dune::FieldVector<T,4>& other) : Dune::FieldVector<T,4>(other) {}

    /** \brief Return the identity element */
    static Quaternion<T> identity() {
        Quaternion<T> id;
        id[0] = 0;
        id[1] = 0;
        id[2] = 0;
        id[3] = 1;
        return id;
    }

    /** \brief The exponential map from \f$ \mathfrak{so}(3) \f$ to \f$ SO(3) \f$
     */
    static Quaternion<T> exp(const T& v0, const T& v1, const T& v2) {
        Quaternion<T> q;

        T normV = std::sqrt(v0*v0 + v1*v1 + v2*v2);
        T sin   = std::sin(normV/2)/normV;

        // if normV == 0 then q = (0,0,0,1)
        if (isnan(sin))
            sin = 0;
            
        q[0] = sin * v0;
        q[1] = sin * v1;
        q[2] = sin * v2;
        q[3] = std::cos(normV/2);

        return q;
    }

    /** \brief Right quaternion multiplication */
    Quaternion<T> mult(const Quaternion<T>& other) {
        Quaternion<T> q;
        q[0] =   (*this)[3]*other[0] - (*this)[2]*other[1] + (*this)[1]*other[2] + (*this)[0]*other[3];
        q[1] =   (*this)[2]*other[0] + (*this)[3]*other[1] - (*this)[0]*other[2] + (*this)[1]*other[3];
        q[2] = - (*this)[1]*other[0] + (*this)[0]*other[1] + (*this)[3]*other[2] + (*this)[2]*other[3];
        q[3] = - (*this)[0]*other[0] - (*this)[1]*other[1] - (*this)[2]*other[2] + (*this)[3]*other[3];

        return q;
    }
    /** \brief Return the tripel of director vectors represented by a unit quaternion

    The formulas are taken from Dichmann, Li, Maddocks, (2.6.4), (2.6.5), (2.6.6)
    */
    Dune::FieldVector<T,3> director(int i) const {
        
        Dune::FieldVector<T,3> d;
        const Dune::FieldVector<T,4>& q = *this;   // simpler notation

        if (i==0) {
            d[0] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];
            d[1] = 2 * (q[0]*q[1] + q[2]*q[3]);
            d[2] = 2 * (q[0]*q[2] - q[1]*q[3]);
        } else if (i==1) {
            d[0] = 2 * (q[0]*q[1] - q[2]*q[3]);
            d[1] = -q[0]*q[0] + q[1]*q[1] - q[2]*q[2] + q[3]*q[3];
            d[2] = 2 * (q[1]*q[2] + q[0]*q[3]);
        } else if (i==2) {
            d[0] = 2 * (q[0]*q[2] + q[1]*q[3]);
            d[1] = 2 * (q[1]*q[2] - q[0]*q[3]);
            d[2] = -q[0]*q[0] - q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
        } else
            DUNE_THROW(Dune::Exception, "Nonexisting director " << i << " requested!");

        return d;
    }
            
    /** \brief Turn quaternion into a unit quaternion by dividing by its Euclidean norm */
    void normalize() {
        (*this) /= this->two_norm();
    }

};

#endif
