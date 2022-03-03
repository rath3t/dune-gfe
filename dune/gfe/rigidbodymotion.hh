#ifndef RIGID_BODY_MOTION_HH
#define RIGID_BODY_MOTION_HH

#include <dune/common/fvector.hh>

#include <dune/gfe/realtuple.hh>
#include "rotation.hh"

/** \brief A rigid-body motion in R^N, i.e., a member of SE(N) */
template <class T, int N>
struct RigidBodyMotion
{
public:

    /** \brief Dimension of manifold */
    static const int dim = N + Rotation<T,N>::dim;

    /** \brief Dimension of the embedding space */
    static const int embeddedDim = N + Rotation<T,N>::embeddedDim;

    /** \brief Type of an infinitesimal rigid body motion */
    typedef Dune::FieldVector<T, dim> TangentVector;

    /** \brief Type of an infinitesimal rigid body motion */
    typedef Dune::FieldVector<T, embeddedDim> EmbeddedTangentVector;

    /** \brief The type used for coordinates */
    typedef T ctype;
    typedef T field_type;

    /** \brief The type used for global coordinates */
    typedef Dune::FieldVector<T,embeddedDim> CoordinateType;

    /** \brief The global convexity radius of the rigid body motions */
    static constexpr double convexityRadius = Rotation<T,N>::convexityRadius;

    /** \brief Default constructor */
    RigidBodyMotion()
    {}

    /** \brief Constructor from a translation and a rotation */
    RigidBodyMotion(const Dune::FieldVector<ctype, N>& translation,
                    const Rotation<ctype,N>& rotation)
    : r(translation), q(rotation)
    {}

    RigidBodyMotion(const CoordinateType& globalCoordinates)
    {
        for (int i=0; i<N; i++)
            r[i] = globalCoordinates[i];

        for (int i=N; i<embeddedDim; i++)
            q[i-N] = globalCoordinates[i];

        // Turn this into a unit quaternion if it isn't already
        q.normalize();
    }

    /** \brief Assigment from RigidBodyMotion with different type -- used for automatic differentiation with ADOL-C */
    template <class T2>
    RigidBodyMotion& operator <<= (const RigidBodyMotion<T2,N>& other) {
        for (int i=0; i<N; i++)
            r[i] <<= other.r[i];
        q <<= other.q;
        return *this;
    }

     /** \brief Rebind the RigidBodyMotion to another coordinate type */
    template<class U>
    struct rebind
    {
      typedef RigidBodyMotion<U,N> other;
    };

    /** \brief The exponential map from a given point $p \in SE(d)$.

     Why the template parameter?  Well, it should work with both TangentVector and EmbeddedTangentVector.
     In general these differ and we could just have two exp methods.  However in 2d they do _not_ differ,
     and then the compiler complains about having two methods with the same signature.
     */
    template <class TVector>
    static RigidBodyMotion<ctype,N> exp(const RigidBodyMotion<ctype,N>& p, const TVector& v) {

        RigidBodyMotion<ctype,N> result;

        // Add translational correction
        for (int i=0; i<N; i++)
            result.r[i] = p.r[i] + v[i];

        // Add rotational correction
        typedef typename std::conditional<std::is_same<TVector,TangentVector>::value,
                                          typename Rotation<ctype,N>::TangentVector,
                                          typename Rotation<ctype,N>::EmbeddedTangentVector>::type RotationTangentVector;
        RotationTangentVector qCorr;
        for (int i=0; i<RotationTangentVector::dimension; i++)
            qCorr[i] = v[N+i];

        result.q = Rotation<ctype,N>::exp(p.q, qCorr);
        return result;
    }

    /** \brief Compute difference vector from a to b on the tangent space of a */
    static EmbeddedTangentVector log(const RigidBodyMotion<ctype,N>& a,
                                     const RigidBodyMotion<ctype,N>& b)
    {
        EmbeddedTangentVector result;

        // Usual linear difference
        for (int i=0; i<N; i++)
            result[i] = b.r[i] - a.r[i];

        // Subtract orientations on the tangent space of 'a'
        typename Rotation<ctype,N>::EmbeddedTangentVector v = Rotation<ctype,N>::log(a.q, b.q);

        // Compute difference on T_a SO(3)
        for (int i=0; i<Rotation<ctype,N>::EmbeddedTangentVector::dimension; i++)
            result[i+N] = v[i];

        return result;
    }

    /** \brief Compute geodesic distance from a to b */
    static T distance(const RigidBodyMotion<ctype,N>& a, const RigidBodyMotion<ctype,N>& b) {

        T euclideanDistanceSquared = (a.r - b.r).two_norm2();

        T rotationDistance = Rotation<ctype,N>::distance(a.q, b.q);

        return std::sqrt(euclideanDistanceSquared + rotationDistance*rotationDistance);
    }

    /** \brief Compute difference vector from a to b on the tangent space of a

     * \warning The method is buggy!  See https://gitlab.mn.tu-dresden.de/osander/dune-gfe/-/merge_requests/2
     * \deprecated Use the log method instead!
     */
    [[deprecated("Use RigidBodyMotion::log instead of RigidBodyMotion::difference!")]]
    static TangentVector difference(const RigidBodyMotion<ctype,N>& a,
                                    const RigidBodyMotion<ctype,N>& b) {

        TangentVector result;

        // Usual linear difference
        for (int i=0; i<N; i++)
            result[i] = a.r[i] - b.r[i];

        // Subtract orientations on the tangent space of 'a'
        typename Rotation<ctype,N>::TangentVector v = Rotation<ctype,N>::difference(a.q, b.q).axial();

        // Compute difference on T_a SO(3)
        for (int i=0; i<Rotation<ctype,N>::TangentVector::dimension; i++)
            result[i+N] = v[i];

        return result;
    }

    static EmbeddedTangentVector derivativeOfDistanceSquaredWRTSecondArgument(const RigidBodyMotion<ctype,N>& a,
                                                                              const RigidBodyMotion<ctype,N>& b) {

        // linear part
        Dune::FieldVector<ctype,N> linearDerivative = a.r;
        linearDerivative -= b.r;
        linearDerivative *= -2;

        // rotation part
        typename Rotation<ctype,N>::EmbeddedTangentVector rotationDerivative
                = Rotation<ctype,N>::derivativeOfDistanceSquaredWRTSecondArgument(a.q, b.q);

        return concat(linearDerivative, rotationDerivative);
    }

    /** \brief Compute the Hessian of the squared distance function keeping the first argument fixed */
    static Dune::SymmetricMatrix<T,embeddedDim> secondDerivativeOfDistanceSquaredWRTSecondArgument(const RigidBodyMotion<ctype,N> & p, const RigidBodyMotion<ctype,N> & q)
    {
        Dune::SymmetricMatrix<T,embeddedDim> result;

        // The linear part
        Dune::SymmetricMatrix<T,N> linearPart = RealTuple<T,N>::secondDerivativeOfDistanceSquaredWRTSecondArgument(p.r,q.r);
        for (int i=0; i<N; i++)
            for (int j=0; j<=i; j++)
                result(i,j) = linearPart(i,j);

        // The rotation part
        Dune::SymmetricMatrix<T,Rotation<T,N>::embeddedDim> rotationPart
                = Rotation<ctype,N>::secondDerivativeOfDistanceSquaredWRTSecondArgument(p.q,q.q);
        for (int i=0; i<Rotation<T,N>::embeddedDim; i++)
        {
            for (int j=0; j<N; j++)
                result(N+i,j) = 0;
            for (int j=0; j<=i; j++)
                result(N+i,N+j) = rotationPart(i,j);
        }

        return result;
    }

    /** \brief Compute the mixed second derivate \partial d^2 / \partial da db

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static Dune::FieldMatrix<T,embeddedDim,embeddedDim> secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(const RigidBodyMotion<ctype,N> & p, const RigidBodyMotion<ctype,N> & q)
    {
        Dune::FieldMatrix<T,embeddedDim,embeddedDim> result(0);

        // The linear part
        Dune::FieldMatrix<T,N,N> linearPart = RealTuple<T,N>::secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(p.r,q.r);
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
                result[i][j] = linearPart[i][j];

        // The rotation part
        Dune::FieldMatrix<T,Rotation<T,N>::embeddedDim,Rotation<T,N>::embeddedDim> rotationPart
                = Rotation<ctype,N>::secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(p.q,q.q);
        for (int i=0; i<Rotation<T,N>::embeddedDim; i++)
            for (int j=0; j<Rotation<T,N>::embeddedDim; j++)
                result[N+i][N+j] = rotationPart[i][j];

        return result;
    }

    /** \brief Compute the third derivative \partial d^3 / \partial dq^3

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static Tensor3<T,embeddedDim,embeddedDim,embeddedDim> thirdDerivativeOfDistanceSquaredWRTSecondArgument(const RigidBodyMotion<ctype,N> & p, const RigidBodyMotion<ctype,N> & q)
    {
        Tensor3<T,embeddedDim,embeddedDim,embeddedDim> result(0);

        // The linear part
        Tensor3<T,N,N,N> linearPart = RealTuple<T,N>::thirdDerivativeOfDistanceSquaredWRTSecondArgument(p.r,q.r);
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
                for (int k=0; k<N; k++)
                    result[i][j][k] = linearPart[i][j][k];

        // The rotation part
        Tensor3<T,Rotation<T,N>::embeddedDim,Rotation<T,N>::embeddedDim,Rotation<T,N>::embeddedDim> rotationPart
                = Rotation<ctype,N>::thirdDerivativeOfDistanceSquaredWRTSecondArgument(p.q,q.q);

        for (int i=0; i<Rotation<T,N>::embeddedDim; i++)
            for (int j=0; j<Rotation<T,N>::embeddedDim; j++)
                for (int k=0; k<Rotation<T,N>::embeddedDim; k++)
                    result[N+i][N+j][N+k] = rotationPart[i][j][k];

        return result;
    }

    /** \brief Compute the mixed third derivative \partial d^3 / \partial da db^2

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static Tensor3<T,embeddedDim,embeddedDim,embeddedDim> thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(const RigidBodyMotion<ctype,N> & p, const RigidBodyMotion<ctype,N> & q)
    {
        Tensor3<T,embeddedDim,embeddedDim,embeddedDim> result(0);

        // The linear part
        Tensor3<T,N,N,N> linearPart = RealTuple<T,N>::thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(p.r,q.r);
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
                for (int k=0; k<N; k++)
                    result[i][j][k] = linearPart[i][j][k];

        // The rotation part
        Tensor3<T,Rotation<T,N>::embeddedDim,Rotation<T,N>::embeddedDim,Rotation<T,N>::embeddedDim> rotationPart = Rotation<ctype,N>::thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(p.q,q.q);
        for (int i=0; i<Rotation<T,N>::embeddedDim; i++)
            for (int j=0; j<Rotation<T,N>::embeddedDim; j++)
                for (int k=0; k<Rotation<T,N>::embeddedDim; k++)
                    result[N+i][N+j][N+k] = rotationPart[i][j][k];

        return result;
    }



    /** \brief Project tangent vector of R^n onto the tangent space */
    EmbeddedTangentVector projectOntoTangentSpace(const EmbeddedTangentVector& v) const {
        EmbeddedTangentVector result;

        // translation part
        for (int i=0; i<N; i++)
          result[i] = v[i];

        // rotation part
        typename Rotation<T,N>::EmbeddedTangentVector rotV;
        for (int i=0; i<Rotation<T,N>::embeddedDim; i++)
            rotV[i] = v[i+N];

        rotV = q.projectOntoTangentSpace(rotV);

        for (int i=0; i<Rotation<T,N>::embeddedDim; i++)
          result[i+N] = rotV[i];

        return result;
    }

    /** \brief Project tangent vector of R^n onto the normal space space */
    EmbeddedTangentVector projectOntoNormalSpace(const EmbeddedTangentVector& v) const {

        EmbeddedTangentVector result;

        // translation part
        for (int i=0; i<N; i++)
          result[i] = 0;

        // rotation part
        T sp = 0;
        for (int i=0; i<Rotation<T,N>::embeddedDim; i++)
          sp += v[i+N] * q[i];

        for (int i=0; i<Rotation<T,N>::embeddedDim; i++)
          result[i+N] = sp * q[i];

        return result;
    }

        /** \brief Transform tangent vectors from quaternion representation to matrix representation
     *
     * This class represents rotations as unit quaternions, and therefore variations of rotations
     * (i.e., tangent vector) are represented in quaternion space, too.  However, some applications
     * require the tangent vectors as matrices. To obtain matrix coordinates we use the
     * chain rule, which means that we have to multiply the given derivative with
     * the derivative of the embedding of the unit quaternion into the space of 3x3 matrices.
     * This second derivative is almost given by the method getFirstDerivativesOfDirectors.
     * However, since the directors of a given unit quaternion are the _columns_ of the
     * corresponding orthogonal matrix, we need to invert the i and j indices
     *
     * As a typical GFE assembler will require this for several tangent vectors at once,
     * the implementation here is a vector one: It allows to treat several tangent vectors
     * together, which have to come as the columns of the `derivative` parameter.
     *
     * So, if I am not mistaken, result[i][j][k] contains \partial R_ij / \partial k
     *
     * \param derivative Tangent vector in quaternion coordinates
     * \returns DR Tangent vector in matrix coordinates
     */
    template <int blocksize>
    Tensor3<T,3,3,blocksize> quaternionTangentToMatrixTangent(const Dune::FieldMatrix<T,7,blocksize>& derivative) const
    {
        Tensor3<T,3,3,blocksize> result = T(0);

        Tensor3<T,3 , 3, 4> dd_dq;
        q.getFirstDerivativesOfDirectors(dd_dq);

        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                for (int k=0; k<blocksize; k++)
                    for (int l=0; l<4; l++)
                        result[i][j][k] += dd_dq[j][i][l] * derivative[l+3][k];

        return result;
    }

    /** \brief The Weingarten map */
    EmbeddedTangentVector weingarten(const EmbeddedTangentVector& z, const EmbeddedTangentVector& v) const {

        EmbeddedTangentVector result;

        // translation part: nothing, the space is flat
        for (int i=0; i<N; i++)
          result[i] = 0;

        // rotation part
        T sp = 0;
        for (int i=0; i<Rotation<T,N>::embeddedDim; i++)
          sp += v[i+N] * q[i];

        for (int i=0; i<Rotation<T,N>::embeddedDim; i++)
          result[i+N] = -sp * z[i+N];

        return result;
    }

    /** \brief Compute an orthonormal basis of the tangent space of SE(3).

    This basis may not be globally continuous.
    */
    Dune::FieldMatrix<T,dim,embeddedDim> orthonormalFrame() const {
        Dune::FieldMatrix<T,dim,embeddedDim> result(0);

        // Get the R^d part
        for (int i=0; i<N; i++)
            result[i][i] = 1;

        Dune::FieldMatrix<T,Rotation<T,N>::dim,Rotation<T,N>::embeddedDim> SO3Part = q.orthonormalFrame();

        for (int i=0; i<Rotation<T,N>::dim; i++)
            for (int j=0; j<Rotation<T,N>::embeddedDim; j++)
                result[N+i][N+j] = SO3Part[i][j];

        return result;
    }

    /** \brief The global coordinates, if you really want them */
    CoordinateType globalCoordinates() const {
        return concat(r, q.globalCoordinates());
    }



    // Translational part
    Dune::FieldVector<ctype, N> r;

    // Rotational part
    Rotation<ctype,N> q;

private:

    /** \brief Concatenate two FieldVectors */
    template <int NN, int M>
    static Dune::FieldVector<ctype,NN+M> concat(const Dune::FieldVector<ctype,NN>& a,
                                               const Dune::FieldVector<ctype,M>& b)
    {
        Dune::FieldVector<ctype,NN+M> result;
        for (int i=0; i<NN; i++)
            result[i] = a[i];
        for (int i=0; i<M; i++)
            result[i+NN] = b[i];
        return result;
    }

};

//! Send configuration to output stream
template <int N, class ctype>
std::ostream& operator<< (std::ostream& s, const RigidBodyMotion<ctype,N>& c)
  {
      s << "(" << c.r << ")  (" << c.q << ")";
      return s;
  }

#endif
