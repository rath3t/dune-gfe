#ifndef ROTATION_HH
#define ROTATION_HH

/** \file
    \brief Define rotations in Euclidean spaces
*/

#include <dune/common/fvector.hh>
#include <dune/common/array.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/exceptions.hh>

#include "quaternion.hh"
#include <dune/gfe/tensor3.hh>
#include <dune/gfe/unitvector.hh>


template <int dim, class T>
class Rotation
{

};

/** \brief Specialization for dim==2
    \tparam T The type used for coordinates
*/
template <class T>
class Rotation<2,T>
{
public:
    /** \brief The type used for coordinates */
    typedef T ctype;

    /** \brief Member of the corresponding Lie algebra.  This really is a skew-symmetric matrix */
    typedef Dune::FieldVector<T,1> TangentVector;

    /** \brief Member of the corresponding Lie algebra.  This really is a skew-symmetric matrix

    This vector is not really embedded in anything.  I have to make my notation more consistent! */
    typedef Dune::FieldVector<T,1> EmbeddedTangentVector;

    /** \brief Default constructor, create the identity rotation */
    Rotation() 
        : angle_(0)
    {}

    Rotation(const T& angle)
        : angle_(angle)
    {}

    /** \brief Return the identity element */
    static Rotation<2,T> identity() {
        // Default constructor creates an identity
        Rotation<2,T> id;
        return id;
    }

    static T distance(const Rotation<2,T>& a, const Rotation<2,T>& b) {
        T dist = a.angle_ - b.angle_;
        while (dist < 0)
            dist += 2*M_PI;
        while (dist > 2*M_PI)
            dist -= 2*M_PI;

        return (dist <= M_PI) ? dist : 2*M_PI - dist;
    }

    /** \brief The exponential map from a given point $p \in SO(3)$. */
    static Rotation<2,T> exp(const Rotation<2,T>& p, const TangentVector& v) {
        Rotation<2,T> result = p;
        result.angle_ += v;
        return result;
    }

    /** \brief The exponential map from \f$ \mathfrak{so}(2) \f$ to \f$ SO(2) \f$
     */
    static Rotation<2,T> exp(const Dune::FieldVector<T,1>& v) {
        Rotation<2,T> result;
        result.angle_ = v[0];
        return result;
    }

    static TangentVector derivativeOfDistanceSquaredWRTSecondArgument(const Rotation<2,T>& a, 
                                                                      const Rotation<2,T>& b) {
        // This assertion is here to remind me of the following laziness:
        // The difference has to be computed modulo 2\pi
        assert( std::fabs(a.angle_ - b.angle_) <= M_PI );
        return -2 * (a.angle_ - b.angle_);
    }

    static Dune::FieldMatrix<T,1,1> secondDerivativeOfDistanceSquaredWRTSecondArgument(const Rotation<2,T>& a, 
                                                                                            const Rotation<2,T>& b) {
        return 2;
    }

    /** \brief Right multiplication */
    Rotation<2,T> mult(const Rotation<2,T>& other) const {
        Rotation<2,T> q = *this;
        q.angle_ += other.angle_;
        return q;
    }

    /** \brief Compute an orthonormal basis of the tangent space of SO(3).

    This basis is of course not globally continuous.
    */
    Dune::FieldMatrix<T,1,1> orthonormalFrame() const {
        return Dune::FieldMatrix<T,1,1>(1);
    }
    
    //private:

    // We store the rotation as an angle
    T angle_;
};

//! Send configuration to output stream
template <class T>
std::ostream& operator<< (std::ostream& s, const Rotation<2,T>& c)
  {
      return s << "[" << c.angle_ << "  (" << std::sin(c.angle_) << " " << std::cos(c.angle_) << ") ]";
  }


/** \brief Specialization for dim==3 

Uses unit quaternion coordinates.
*/
template <class T>
class Rotation<3,T> : public Quaternion<T>
{

    /** \brief Computes sin(x/2) / x without getting unstable for small x */
    static T sincHalf(const T& x) {
        return (x < 1e-4) ? 0.5 - (x*x/48) : std::sin(x/2)/x;
    }

    /** \brief Compute the derivative of arccos^2 without getting unstable for x close to 1 */
    static T derivativeOfArcCosSquared(const T& x) {
        const T eps = 1e-12;
        if (x > 1-eps) {  // regular expression is unstable, use the series expansion instead
            return -2 + 2*(x-1)/3 - 4/15*(x-1)*(x-1) + 4/35*(x-1)*(x-1)*(x-1);
        } else if (x < -1+eps) {  // The function is not differentiable
            DUNE_THROW(Dune::Exception, "arccos^2 is not differentiable at x==-1!");
        } else
            return -2*std::acos(x) / std::sqrt(1-x*x);
    }

public:

    /** \brief The type used for coordinates */
    typedef T ctype;

    /** \brief The type used for global coordinates */
    typedef Dune::FieldVector<T,4> CoordinateType;
    
    /** \brief Dimension of the manifold formed by the 3d rotations */
    static const int dim = 3;
    
    /** \brief Member of the corresponding Lie algebra.  This really is a skew-symmetric matrix */
    typedef Dune::FieldVector<T,3> TangentVector;

    /** \brief A tangent vector as a vector in the surrounding coordinate space */
    typedef Quaternion<T> EmbeddedTangentVector;

    /** \brief Default constructor creates the identity element */
    Rotation()
        : Quaternion<T>(0,0,0,1)
    {}
    
    Rotation<3,T>(const Dune::array<T,4>& c)
    {
        for (int i=0; i<4; i++)
            (*this)[i] = c[i];

        *this /= this->two_norm();
    }
    
    Rotation<3,T>(const Dune::FieldVector<T,4>& c)
        : Quaternion<T>(c)
    {
        *this /= this->two_norm();
    }
    
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

    /** \brief Right multiplication with a quaternion
        \todo do we really need this?*/
    Rotation<3,T> mult(const Quaternion<T>& other) const {
        Rotation<3,T> q;
        q[0] =   (*this)[3]*other[0] - (*this)[2]*other[1] + (*this)[1]*other[2] + (*this)[0]*other[3];
        q[1] =   (*this)[2]*other[0] + (*this)[3]*other[1] - (*this)[0]*other[2] + (*this)[1]*other[3];
        q[2] = - (*this)[1]*other[0] + (*this)[0]*other[1] + (*this)[3]*other[2] + (*this)[2]*other[3];
        q[3] = - (*this)[0]*other[0] - (*this)[1]*other[1] - (*this)[2]*other[2] + (*this)[3]*other[3];

        return q;
    }

    /** \brief The exponential map from \f$ \mathfrak{so}(3) \f$ to \f$ SO(3) \f$
     */
    static Rotation<3,T> exp(const Dune::FieldVector<T,3>& v) {
        return exp(v[0], v[1], v[2]);
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

    /** \brief The exponential map from a given point $p \in SO(3)$. */
    static Rotation<3,T> exp(const Rotation<3,T>& p, const TangentVector& v) {
        Rotation<3,T> corr = exp(v);
        return p.mult(corr);
    }

    /** \brief The exponential map from a given point $p \in SO(3)$.
     
        There may be a more direct way to implement this
        
        \param v A tangent vector in quaternion coordinates
     */
    static Rotation<3,T> exp(const Rotation<3,T>& p, const EmbeddedTangentVector& v) {
        
        assert( std::fabs(p*v) < 1e-8 );
        
        // The vector v as a quaternion
        Quaternion<T> vQuat(v);
        
        // left multiplication by the inverse base point yields a tangent vector at the identity
        Quaternion<T> vAtIdentity = p.inverse().mult(vQuat);
        assert( std::fabs(vAtIdentity[3]) < 1e-8 );

        // vAtIdentity as a skew matrix
        TangentVector vMatrix;
        vMatrix[0] = 2*vAtIdentity[0];
        vMatrix[1] = 2*vAtIdentity[1];
        vMatrix[2] = 2*vAtIdentity[2];
        
        // The actual exponential map
        return exp(p, vMatrix);
    }

    static Rotation<3,T> exp(const Rotation<3,T>& p, const Dune::FieldVector<T,4>& v) {
        
        assert( std::fabs(p*v) < 1e-8 );
        
        // The vector v as a quaternion
        Quaternion<T> vQuat(v);
        
        // left multiplication by the inverse base point yields a tangent vector at the identity
        Quaternion<T> vAtIdentity = p.inverse().mult(vQuat);
        assert( std::fabs(vAtIdentity[3]) < 1e-8 );

        // vAtIdentity as a skew matrix
        TangentVector vMatrix;
        vMatrix[0] = 2*vAtIdentity[0];
        vMatrix[1] = 2*vAtIdentity[1];
        vMatrix[2] = 2*vAtIdentity[2];
        
        // The actual exponential map
        return exp(p, vMatrix);
    }

    static Dune::FieldMatrix<T,4,3> Dexp(const Dune::FieldVector<T,3>& v) {

        Dune::FieldMatrix<T,4,3> result(0);
        T norm = v.two_norm();
        
        for (int i=0; i<3; i++) {

            for (int m=0; m<3; m++) {
                
                result[m][i] = (norm<1e-10) 
                    /** \todo Isn't there a better way to implement this stably? */
                    ? 0.5 * (i==m) 
                    : 0.5 * std::cos(norm/2) * v[i] * v[m] / (norm*norm) + sincHalf(norm) * ( (i==m) - v[i]*v[m]/(norm*norm));

            }

            result[3][i] = - 0.5 * sincHalf(norm) * v[i];

        }
        return result;
    }

    static void DDexp(const Dune::FieldVector<T,3>& v,
                      Dune::array<Dune::FieldMatrix<T,3,3>, 4>& result) {

        T norm = v.two_norm();
        if (norm<=1e-10) {

            for (int m=0; m<4; m++)
                result[m] = 0;

            for (int i=0; i<3; i++)
                result[3][i][i] = -0.25;


        } else {

            for (int i=0; i<3; i++) {
                
                for (int j=0; j<3; j++) {
                    
                    for (int m=0; m<3; m++) {
                        
                        result[m][i][j] = -0.25*std::sin(norm/2)*v[i]*v[j]*v[m]/(norm*norm*norm)
                            + ((i==j)*v[m] + (j==m)*v[i] + (i==m)*v[j] - 3*v[i]*v[j]*v[m]/(norm*norm))
                            * (0.5*std::cos(norm/2) - sincHalf(norm)) / (norm*norm);
                        

                    }

                    result[3][i][j] = -0.5/(norm*norm)
                        * ( 0.5*std::cos(norm/2)*v[i]*v[j] + std::sin(norm/2) * ((i==j)*norm - v[i]*v[j]/norm));

                }

            }

        }

    }

    /** \brief The inverse of the exponential map */
    static Dune::FieldVector<T,3> expInv(const Rotation<3,T>& q) {
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
    static Dune::FieldMatrix<T,3,4> DexpInv(const Rotation<3,T>& q) {
        
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

    static T distance(const Rotation<3,T>& a, const Rotation<3,T>& b) {
        Quaternion<T> diff = a;

        diff.invert();
        diff = diff.mult(b);
        
        // Compute the geodesical distance between a and b on SO(3)
        // Due to numerical dirt, diff[3] may be larger than 1. 
        // In that case, use 1 instead of diff[3].
        return (diff[3] > 1.0)
            ? 0
            : 2*std::acos( std::min(diff[3],1.0) );
    }

    /** \brief Compute the vector in T_aSO(3) that is mapped by the exponential map
        to the geodesic from a to b
    */
    static Dune::FieldVector<T,3> difference(const Rotation<3,T>& a, const Rotation<3,T>& b) {

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

    static EmbeddedTangentVector derivativeOfDistanceSquaredWRTSecondArgument(const Rotation<3,T>& p, 
                                                                      const Rotation<3,T>& q) {
        
        Rotation<3,T> pInv = p;
        pInv.invert();
        
        // the forth component of pInv times q
        T pInvq_4 = - pInv[0]*q[0] - pInv[1]*q[1] - pInv[2]*q[2] + pInv[3]*q[3];
        
        T arccosSquaredDer_pInvq_4 = derivativeOfArcCosSquared(pInvq_4);
        
        EmbeddedTangentVector result;
        result[0] = -4 * arccosSquaredDer_pInvq_4 * pInv[0];
        result[1] = -4 * arccosSquaredDer_pInvq_4 * pInv[1];
        result[2] = -4 * arccosSquaredDer_pInvq_4 * pInv[2];
        result[3] =  4 * arccosSquaredDer_pInvq_4 * pInv[3];
        
        // project onto the tangent space at q
        EmbeddedTangentVector projectedResult = result;
        projectedResult.axpy(-1*(q*result), q);
        
        assert(std::fabs(projectedResult * q) < 1e-7);
        
        return projectedResult;
    }

    /** \brief Compute the Hessian of the squared distance function keeping the first argument fixed

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static Dune::FieldMatrix<T,4,4> secondDerivativeOfDistanceSquaredWRTSecondArgument(const Rotation<3,T>& p, const Rotation<3,T>& q) {
        // use the functionality from the unitvector class
        Dune::FieldMatrix<T,4,4> result = UnitVector<4>::secondDerivativeOfDistanceSquaredWRTSecondArgument(p.globalCoordinates(),
                                                                                                                 q.globalCoordinates());
        // for some reason that I don't really understand, the distance we have defined for the rotations (== Unit quaternions)
        // is twice the corresponding distance on the unit quaternions seen as a sphere.  Hence the derivative of the
        // squared distance needs to be multiplied by 4.
        result *= 4;
        return result;
    }
    
    /** \brief Compute the mixed second derivate \partial d^2 / \partial da db

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static Dune::FieldMatrix<T,4,4> secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(const Rotation<3,T>& p, const Rotation<3,T>& q) {
        // use the functionality from the unitvector class
        Dune::FieldMatrix<T,4,4> result = UnitVector<4>::secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(p.globalCoordinates(),
                                                                                                                         q.globalCoordinates());
        // for some reason that I don't really understand, the distance we have defined for the rotations (== Unit quaternions)
        // is twice the corresponding distance on the unit quaternions seen as a sphere.  Hence the derivative of the
        // squared distance needs to be multiplied by 4.
        result *= 4;
        return result;
    }
    
    /** \brief Compute the third derivative \partial d^3 / \partial dq^3

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static Tensor3<T,4,4,4> thirdDerivativeOfDistanceSquaredWRTSecondArgument(const Rotation<3,T>& p, const Rotation<3,T>& q) {
        // use the functionality from the unitvector class
        Tensor3<T,4,4,4> result = UnitVector<4>::thirdDerivativeOfDistanceSquaredWRTSecondArgument(p.globalCoordinates(),
                                                                                                        q.globalCoordinates());
        // for some reason that I don't really understand, the distance we have defined for the rotations (== Unit quaternions)
        // is twice the corresponding distance on the unit quaternions seen as a sphere.  Hence the derivative of the
        // squared distance needs to be multiplied by 4.
        result *= 4;
        return result;
    }
    
    /** \brief Compute the mixed third derivative \partial d^3 / \partial da db^2

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static Tensor3<T,4,4,4> thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(const Rotation<3,T>& p, const Rotation<3,T>& q) {
        // use the functionality from the unitvector class
        Tensor3<T,4,4,4> result = UnitVector<4>::thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(p.globalCoordinates(),
                                                                                                                  q.globalCoordinates());
        // for some reason that I don't really understand, the distance we have defined for the rotations (== Unit quaternions)
        // is twice the corresponding distance on the unit quaternions seen as a sphere.  Hence the derivative of the
        // squared distance needs to be multiplied by 4.
        result *= 4;
        return result;
    }



    
    /** \brief Interpolate between two rotations */
    static Rotation<3,T> interpolate(const Rotation<3,T>& a, const Rotation<3,T>& b, T omega) {

        // Compute difference on T_a SO(3)
        Dune::FieldVector<T,3> v = difference(a,b);

        v *= omega;

        return a.mult(exp(v[0], v[1], v[2]));
    }

    /** \brief Interpolate between two rotations 
        \param omega must be between 0 and 1
    */
    static Quaternion<T> interpolateDerivative(const Rotation<3,T>& a, const Rotation<3,T>& b, 
                                               T omega) {
        Quaternion<T> result(0);

        // Compute difference on T_a SO(3)
        Dune::FieldVector<T,3> xi = difference(a,b);

        Dune::FieldVector<T,3> v = xi;
        v *= omega;
        
        // //////////////////////////////////////////////////////////////
        //   v now contains the derivative at 'a'.  The derivative at
        //   the requested site is v pushed forward by Dexp.
        // /////////////////////////////////////////////////////////////

        Dune::FieldMatrix<T,4,3> diffExp = Dexp(v);

        diffExp.umv(xi,result);

        return a.Quaternion<T>::mult(result);
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

    /** \brief Set rotation from orthogonal matrix 

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
    
    /** \brief Project tangent vector of R^n onto the tangent space */
    EmbeddedTangentVector projectOntoTangentSpace(const EmbeddedTangentVector& v) const {
        EmbeddedTangentVector result = v;
        EmbeddedTangentVector data = *this;
        result.axpy(-1*(data*result), data);
        return result;
    }
    
    /** \brief The global coordinates, if you really want them */
    const CoordinateType& globalCoordinates() const {
        return *this;
    }

    /** \brief Compute an orthonormal basis of the tangent space of SO(3).

    This basis is of course not globally continuous.
    */
    Dune::FieldMatrix<T,3,4> orthonormalFrame() const {
        Dune::FieldMatrix<T,3,4> result;
        for (int i=0; i<3; i++)
            result[i] = B(i);
        return result;
    }
    
};



#endif
