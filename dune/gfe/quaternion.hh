#ifndef QUATERNION_HH
#define QUATERNION_HH

#include <dune/common/fvector.hh>
#include <dune/common/array.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/exceptions.hh>

#include <dune/gfe/tensor3.hh>

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

    /** \brief Constructor from a single scalar */
    Quaternion(const T& a) : Dune::FieldVector<T,4>(a) {}

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

    /** \brief Assignment from a scalar */
    Quaternion<T>& operator=(const T& v) {
        for (int i=0; i<4; i++)
            (*this)[i] = v;
        return (*this);
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

    /** \brief Conjugate the quaternion */
    void conjugate() {

        (*this)[0] *= -1;
        (*this)[1] *= -1;
        (*this)[2] *= -1;

    }

    /** \brief Invert the quaternion */
    void invert() {

        conjugate();
        (*this) /= this->two_norm2();

    }
    
    /** \brief Yield the inverse quaternion */
    Quaternion<T> inverse() const {
        
        Quaternion<T> result = *this;
        result.invert();
        return result;
    }

    /** \brief Create three vectors which form an orthonormal basis of \mathbb{H} together
        with this one.

        This is used to compute the strain in rod problems.  
        See: Dichmann, Li, Maddocks, 'Hamiltonian Formulations and Symmetries in
        Rod Mechanics', page 83 
    */
    Quaternion<T> B(int m) const {
        assert(m>=0 && m<3);
        Quaternion<T> r;
        if (m==0) {
            r[0] =  (*this)[3];
            r[1] =  (*this)[2];
            r[2] = -(*this)[1];
            r[3] = -(*this)[0];
        } else if (m==1) {
            r[0] = -(*this)[2];
            r[1] =  (*this)[3];
            r[2] =  (*this)[0];
            r[3] = -(*this)[1];
        } else {
            r[0] =  (*this)[1];
            r[1] = -(*this)[0];
            r[2] =  (*this)[3];
            r[3] = -(*this)[2];
        } 

        return r;
    }
    

};

#endif
