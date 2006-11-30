#ifndef QUATERNION_HH
#define QUATERNION_HH

#include <dune/common/fvector.hh>
#include <dune/common/exceptions.hh>

template <class T>
class Quaternion : public Dune::FieldVector<T,4>
{
public:

    /** \brief Default constructor */
    Quaternion() {}

    /** \brief Constructor with the four components */
    Quaternion(const T& a, const T& b, const T& c, const T& d) {

        (*this)[0] = a;
        (*this)[1] = b;
        (*this)[2] = c;
        (*this)[3] = d;
    
    }

    /** \brief Copy constructor */
    Quaternion(const Dune::FieldVector<T,4>& other) : Dune::FieldVector<T,4>(other) {}

    /** \brief Constructor with rotation axis and angle */
    Quaternion(Dune::FieldVector<T,3> axis, T angle) {
        axis /= axis.two_norm();
        axis *= std::sin(angle/2);
        (*this)[0] = axis[0];
        (*this)[1] = axis[1];
        (*this)[2] = axis[2];
        (*this)[3] = std::cos(angle/2);
    }

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

        // Stabilization for small |v| due to Grassia
        T sin   = (normV < 1e-4) ? 0.5 * (normV*normV/48) : std::sin(normV/2)/normV;

        // if normV == 0 then q = (0,0,0,1)
        assert(!isnan(sin));
            
        q[0] = sin * v0;
        q[1] = sin * v1;
        q[2] = sin * v2;
        q[3] = std::cos(normV/2);

        return q;
    }

    /** \brief Right quaternion multiplication */
    Quaternion<T> mult(const Quaternion<T>& other) const {
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

    Dune::FieldVector<double,3> rotate(const Dune::FieldVector<double,3>& v) const {

        Dune::FieldVector<double,3> result;
        Dune::FieldVector<double,3> d0 = director(0);
        Dune::FieldVector<double,3> d1 = director(1);
        Dune::FieldVector<double,3> d2 = director(2);

        for (int i=0; i<3; i++)
            result[i] = v[0]*d0[i] + v[1]*d1[i] + v[2]*d2[i];

        return result;
    }

    /** \brief Interpolate between two rotations */
    static Quaternion<T> interpolate(const Quaternion<T>& a, const Quaternion<T>& b, double omega) {

        Quaternion<T> result;

        for (int i=0; i<4; i++)
            result[i] = a[i]*(1-omega) + b[i]*omega;

        result.normalize();

        return result;
    }

    /** \brief Return the corresponding orthogonal matrix */
    void matrix(Dune::FieldMatrix<T,3,3>& m) const {

        m[0][0] = (*this)[0]*(*this)[0] - (*this)[1]*(*this)[1] - (*this)[2]*(*this)[2] + (*this)[3]*(*this)[3];
        m[0][1] = 2 * ( (*this)[0]*(*this)[1] - (*this)[2]*(*this)[3] );
        m[0][2] = 2 * ( (*this)[0]*(*this)[2] + (*this)[1]*(*this)[3] );

        m[1][0] = 2 * ( (*this)[0]*(*this)[1] + (*this)[2]*(*this)[3] );
        m[1][1] = - (*this)[0]*(*this)[0] + (*this)[1]*(*this)[1] - (*this)[2]*(*this)[2] + (*this)[3]*(*this)[3];
        m[1][2] = 2 * ( -(*this)[0]*(*this)[3] + (*this)[1]*(*this)[2] );

        m[2][0] = 2 * ( (*this)[0]*(*this)[2] - (*this)[1]*(*this)[3] );
        m[2][1] = 2 * ( (*this)[0]*(*this)[3] + (*this)[1]*(*this)[2] );
        m[2][2] = - (*this)[0]*(*this)[0] - (*this)[1]*(*this)[1] + (*this)[2]*(*this)[2] + (*this)[3]*(*this)[3];

    }

    /** \brief Set unit quaternion from orthogonal matrix 

    We tacitly assume that the matrix really is orthogonal */
    void set(const Dune::FieldMatrix<T,3,3>& m) {

        // Easier writing
        Dune::FieldVector<T,4>& p = (*this);
        // The following equations for the derivation of a unit quaternion from a rotation
        // matrix comes from 'E. Salamin, Application of Quaternions to Computation with
        // Rotations, Technical Report, Stanford, 1974'

        p[0] = (1 + m[0][0] - m[1][1] - m[2][2]) / 4;
        p[1] = (1 - m[0][0] + m[1][1] - m[2][2]) / 4;
        p[2] = (1 - m[0][0] - m[1][1] + m[2][2]) / 4;
        p[3] = (1 + m[0][0] + m[1][1] + m[2][2]) / 4;

        // avoid rounding problems
        if (p[0] >= p[1] && p[0] >= p[2] && p[0] >= p[3]) {

            p[0] = std::sqrt(p[0]);

            // r_x r_y = (R_12 + R_21) / 4
            p[1] = (m[0][1] + m[1][0]) / 4 / p[0];

            // r_x r_z = (R_13 + R_31) / 4
            p[2] = (m[0][2] + m[2][0]) / 4 / p[0];

            // r_0 r_x = (R_32 - R_23) / 4
            p[3] = (m[2][1] - m[1][2]) / 4 / p[0]; 

        } else if (p[1] >= p[0] && p[1] >= p[2] && p[1] >= p[3]) {

            p[1] = std::sqrt(p[1]);

            // r_x r_y = (R_12 + R_21) / 4
            p[0] = (m[0][1] + m[1][0]) / 4 / p[1];

            // r_y r_z = (R_23 + R_32) / 4
            p[2] = (m[1][2] + m[2][1]) / 4 / p[1];

            // r_0 r_y = (R_13 - R_31) / 4
            p[3] = (m[0][2] - m[2][0]) / 4 / p[1]; 

        } else if (p[2] >= p[0] && p[2] >= p[1] && p[2] >= p[3]) {

            p[2] = std::sqrt(p[2]);

            // r_x r_z = (R_13 + R_31) / 4
            p[0] = (m[0][2] + m[2][0]) / 4 / p[2];

            // r_y r_z = (R_23 + R_32) / 4
            p[1] = (m[1][2] + m[2][1]) / 4 / p[2];

            // r_0 r_z = (R_21 - R_12) / 4
            p[3] = (m[1][0] - m[0][1]) / 4 / p[2]; 

        } else {

            p[3] = std::sqrt(p[3]);

            // r_0 r_x = (R_32 - R_23) / 4
            p[0] = (m[2][1] - m[1][2]) / 4 / p[3];

            // r_0 r_y = (R_13 - R_31) / 4
            p[1] = (m[0][2] - m[2][0]) / 4 / p[3];

            // r_0 r_z = (R_21 - R_12) / 4
            p[2] = (m[1][0] - m[0][1]) / 4 / p[3]; 

        }

    }
};

#endif
