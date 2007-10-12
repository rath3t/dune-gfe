#ifndef QUATERNION_HH
#define QUATERNION_HH

#include <dune/common/fvector.hh>
#include <dune/common/fixedarray.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/exceptions.hh>

template <class T>
class Quaternion : public Dune::FieldVector<T,4>
{

    /** \brief Computes sin(x/2) / x without getting unstable for small x */
    static T sincHalf(const T& x) {
        return (x < 1e-4) ? 0.5 + (x*x/48) : std::sin(x/2)/x;
    }

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

    /** \brief The exponential map from \f$ \mathfrak{so}(3) \f$ to \f$ SO(3) \f$
     */
    static Quaternion<T> exp(const Dune::FieldVector<T,3>& v) {
        return exp(v[0], v[1], v[2]);
    }

    /** \brief The exponential map from \f$ \mathfrak{so}(3) \f$ to \f$ SO(3) \f$
     */
    static Quaternion<T> exp(const T& v0, const T& v1, const T& v2) {
        Quaternion<T> q;

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

    static Dune::FieldMatrix<T,4,3> Dexp(const Dune::FieldVector<T,3>& v) {

        Dune::FieldMatrix<T,4,3> result(0);
        T norm = v.two_norm();
        
        for (int i=0; i<3; i++) {

            for (int m=0; m<3; m++) {
                
                result[m][i] = (norm==0) 
                    /** \todo Isn't there a better way to implement this stably? */
                    ? 0.5 * (i==m) 
                    : 0.5 * std::cos(norm/2) * v[i] * v[m] / (norm*norm) + sincHalf(norm) * ( (i==m) - v[i]*v[m]/(norm*norm));

            }

            result[3][i] = - 0.5 * sincHalf(norm) * v[i];

        }
        return result;
    }

    /** \brief The inverse of the exponential map */
    static Dune::FieldVector<T,3> expInv(const Quaternion<T>& q) {
        // Compute v = exp^{-1} q
        // Due to numerical dirt, q[3] may be larger than 1. 
        // In that case, use 1 instead of q[3].
        Dune::FieldVector<T,3> v;
        if (q[3] > 1.0) {

            v = 0;

        } else {
            
            T invSinc = 1/sincHalf(2*std::acos(q[3]));
            
            v[0] = q[0] * invSinc;
            v[1] = q[1] * invSinc;
            v[2] = q[2] * invSinc;

        }
        return v;
    }

    /** \brief The derivative of the inverse of the exponential map, evaluated at q */
    static Dune::FieldMatrix<T,3,4> DexpInv(const Quaternion<T>& q) {
        
        // Compute v = exp^{-1} q
        Dune::FieldVector<T,3> v = expInv(q);

        // The derivative of exp at v
        Dune::FieldMatrix<T,4,3> A = Dexp(v);

        // Compute the Moore-Penrose pseudo inverse  A^+ = (A^T A)^{-1} A^T
        Dune::FieldMatrix<T,3,3> ATA;

        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++) {
                ATA[i][j] = 0;
                for (int k=0; k<4; k++)
                    ATA[i][j] += A[k][i] * A[k][j];
            }

        ATA.invert();

        Dune::FieldMatrix<T,3,4> APseudoInv;
        for (int i=0; i<3; i++)
            for (int j=0; j<4; j++) {
                APseudoInv[i][j] = 0;
                for (int k=0; k<3; k++)
                    APseudoInv[i][j] += ATA[i][k] * A[j][k];
            }

        return APseudoInv;
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
            
    void getFirstDerivativesOfDirectors(Dune::array<Dune::FieldMatrix<double,3 , 4>, 3>& dd_dq) const
    {
        const Quaternion<T>& q = (*this);

        dd_dq[0][0][0] =  2*q[0];  dd_dq[0][0][1] = -2*q[1];  dd_dq[0][0][2] = -2*q[2];  dd_dq[0][0][3] =  2*q[3];
        dd_dq[0][1][0] =  2*q[1];  dd_dq[0][1][1] =  2*q[0];  dd_dq[0][1][2] =  2*q[3];  dd_dq[0][1][3] =  2*q[2];
        dd_dq[0][2][0] =  2*q[2];  dd_dq[0][2][1] = -2*q[3];  dd_dq[0][2][2] =  2*q[0];  dd_dq[0][2][3] = -2*q[1];

        dd_dq[1][0][0] =  2*q[1];  dd_dq[1][0][1] =  2*q[0];  dd_dq[1][0][2] = -2*q[3];  dd_dq[1][0][3] = -2*q[2];
        dd_dq[1][1][0] = -2*q[0];  dd_dq[1][1][1] =  2*q[1];  dd_dq[1][1][2] = -2*q[2];  dd_dq[1][1][3] =  2*q[3];
        dd_dq[1][2][0] =  2*q[3];  dd_dq[1][2][1] =  2*q[2];  dd_dq[1][2][2] =  2*q[1];  dd_dq[1][2][3] =  2*q[0];

        dd_dq[2][0][0] =  2*q[2];  dd_dq[2][0][1] =  2*q[3];  dd_dq[2][0][2] =  2*q[0];  dd_dq[2][0][3] =  2*q[1];
        dd_dq[2][1][0] = -2*q[3];  dd_dq[2][1][1] =  2*q[2];  dd_dq[2][1][2] =  2*q[1];  dd_dq[2][1][3] = -2*q[0];
        dd_dq[2][2][0] = -2*q[0];  dd_dq[2][2][1] = -2*q[1];  dd_dq[2][2][2] =  2*q[2];  dd_dq[2][2][3] =  2*q[3];

    }

    void getFirstDerivativesOfDirectors(Dune::array<Dune::array<Dune::FieldVector<double,3>, 3>, 3>& dd_dvj) const
    {
        const Quaternion<T>& q = (*this);

        // Contains \partial q / \partial v^i_j  at v = 0
        Dune::array<Quaternion<double>,3> dq_dvj;
        for (int j=0; j<3; j++) {
            for (int m=0; m<4; m++)
                dq_dvj[j][m]    = (j==m) * 0.5;
        }
        
        // Contains \parder d \parder v_j
        
        for (int j=0; j<3; j++) {
            
            // d1
            dd_dvj[0][j][0] = q[0]*(q.mult(dq_dvj[j]))[0] - q[1]*(q.mult(dq_dvj[j]))[1] 
                - q[2]*(q.mult(dq_dvj[j]))[2] + q[3]*(q.mult(dq_dvj[j]))[3];
            
            dd_dvj[0][j][1] = (q.mult(dq_dvj[j]))[0]*q[1] + q[0]*(q.mult(dq_dvj[j]))[1]
                + (q.mult(dq_dvj[j]))[2]*q[3] + q[2]*(q.mult(dq_dvj[j]))[3];
            
            dd_dvj[0][j][2] = (q.mult(dq_dvj[j]))[0]*q[2] + q[0]*(q.mult(dq_dvj[j]))[2]
                - (q.mult(dq_dvj[j]))[1]*q[3] - q[1]*(q.mult(dq_dvj[j]))[3];
            
            // d2
            dd_dvj[1][j][0] = (q.mult(dq_dvj[j]))[0]*q[1] + q[0]*(q.mult(dq_dvj[j]))[1]
                - (q.mult(dq_dvj[j]))[2]*q[3] - q[2]*(q.mult(dq_dvj[j]))[3];
            
            dd_dvj[1][j][1] = - q[0]*(q.mult(dq_dvj[j]))[0] + q[1]*(q.mult(dq_dvj[j]))[1] 
                - q[2]*(q.mult(dq_dvj[j]))[2] + q[3]*(q.mult(dq_dvj[j]))[3];
            
            dd_dvj[1][j][2] = (q.mult(dq_dvj[j]))[1]*q[2] + q[1]*(q.mult(dq_dvj[j]))[2]
                + (q.mult(dq_dvj[j]))[0]*q[3] + q[0]*(q.mult(dq_dvj[j]))[3];
            
            // d3
            dd_dvj[2][j][0] = (q.mult(dq_dvj[j]))[0]*q[2] + q[0]*(q.mult(dq_dvj[j]))[2]
                + (q.mult(dq_dvj[j]))[1]*q[3] + q[1]*(q.mult(dq_dvj[j]))[3];
            
            dd_dvj[2][j][1] = (q.mult(dq_dvj[j]))[1]*q[2] + q[1]*(q.mult(dq_dvj[j]))[2]
                - (q.mult(dq_dvj[j]))[0]*q[3] - q[0]*(q.mult(dq_dvj[j]))[3];
            
            dd_dvj[2][j][2] = - q[0]*(q.mult(dq_dvj[j]))[0] - q[1]*(q.mult(dq_dvj[j]))[1] 
                + q[2]*(q.mult(dq_dvj[j]))[2] + q[3]*(q.mult(dq_dvj[j]))[3];
            
            
            dd_dvj[0][j] *= 2;
            dd_dvj[1][j] *= 2;
            dd_dvj[2][j] *= 2;
            
        }
    
    // Check: The derivatives of the directors must be orthogonal to the directors
    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
            assert (std::abs(q.director(i) * dd_dvj[i][j]) < 1e-7);

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

    static Dune::FieldVector<T,3> difference(const Quaternion<T>& a, const Quaternion<T>& b) {

        Quaternion<T> diff = a;
        diff.invert();
        diff = diff.mult(b);

        // Compute the geodesical distance between a and b on SO(3)
        // Due to numerical dirt, diff[3] may be larger than 1. 
        // In that case, use 1 instead of diff[3].
        Dune::FieldVector<T,3> v;
        if (diff[3] > 1.0) {

            v = 0;

        } else {
            
            T dist = 2*std::acos( std::min(diff[3],1.0) );
            
            T invSinc = 1/sincHalf(dist);
            
            // Compute difference on T_a SO(3)
            v[0] = diff[0] * invSinc;
            v[1] = diff[1] * invSinc;
            v[2] = diff[2] * invSinc;

        }

        return v;
    }

    /** \brief Interpolate between two rotations */
    /** \todo Use method 'difference' here */
    static Quaternion<T> interpolate(const Quaternion<T>& a, const Quaternion<T>& b, double omega) {

        // Compute difference on T_a SO(3)
        Dune::FieldVector<T,3> v = difference(a,b);

        v *= omega;

        return a.mult(exp(v[0], v[1], v[2]));
    }

    /** \brief Interpolate between two rotations */
    /** \todo Use method 'difference' here */
    static Quaternion<T> interpolateDerivative(const Quaternion<T>& a, const Quaternion<T>& b, 
                                               double omega, double intervallLength) {
        Quaternion<T> result;

        // Compute difference on T_a SO(3)
        Dune::FieldVector<double,3> v = difference(a,b);

        Dune::FieldVector<double,3> der = v;
        der /= intervallLength;
        
        // //////////////////////////////////////////////////////////////
        //   v now contains the derivative at 'a'.  The derivative at
        //   the requested site is v pushed forward by Dexp.
        // /////////////////////////////////////////////////////////////

        v *= omega;

        Dune::FieldMatrix<double,4,3> diffExp = Quaternion<double>::Dexp(v);

        result[0] = result[1] = result[2] = result[3] = 0;
        diffExp.umv(der,result);

        result = a.mult(result);

        // ////////////////////////////////////////////////////////////////////////////
        //   Check correctness by comparing with a finite difference approximation
        // ////////////////////////////////////////////////////////////////////////////
        double eps = 1e-6;
        Quaternion<T> fdResult = interpolate(a,b, omega+eps);
        fdResult              -= interpolate(a,b, omega-eps);
        fdResult /= 2*eps;
        fdResult /= intervallLength;

        if ((result-fdResult).two_norm() > 1e-4) {
            std::cout << "Wrong interpolation:\n";
            std::cout << "Analytical: " << result << "        fd: " << fdResult << std::endl;
            abort();
        }
            
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
