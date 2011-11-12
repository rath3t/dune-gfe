#ifndef RIGID_BODY_MOTION_HH
#define RIGID_BODY_MOTION_HH

#include <dune/common/fvector.hh>

#include <dune/gfe/realtuple.hh>
#include "rotation.hh"

/** \brief A rigid-body motion in R^N, i.e., a member of SE(N) */
template <int N, class T=double>
struct RigidBodyMotion
{
public:
    
    /** \brief Dimension of manifold */
    static const int dim = N + Rotation<N,T>::dim;
    
    /** \brief Dimension of the embedding space */
    static const int embeddedDim = N + Rotation<N,T>::embeddedDim;
    
    /** \brief Type of an infinitesimal rigid body motion */
    typedef Dune::FieldVector<T, dim> TangentVector;

    /** \brief Type of an infinitesimal rigid body motion */
    typedef Dune::FieldVector<T, embeddedDim> EmbeddedTangentVector;

    /** \brief The type used for coordinates */
    typedef T ctype;
    
    /** \brief The type used for global coordinates */
    typedef Dune::FieldVector<T,embeddedDim> CoordinateType;

    /** \brief Default constructor */
    RigidBodyMotion()
    {}
    
    /** \brief Constructor from a translation and a rotation */
    RigidBodyMotion(const Dune::FieldVector<ctype, N>& translation,
                    const Rotation<N,ctype>& rotation)
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
    
    /** \brief The exponential map from a given point $p \in SE(d)$. 
     
     Why the template parameter?  Well, it should work with both TangentVector and EmbeddedTangentVector.
     In general these differ and we could just have two exp methods.  However in 2d they do _not_ differ,
     and then the compiler complains about having two methods with the same signature.
     */
    template <class TVector>
    static RigidBodyMotion<N,ctype> exp(const RigidBodyMotion<N,ctype>& p, const TVector& v) {

        RigidBodyMotion<N,ctype> result;

        // Add translational correction
        for (int i=0; i<N; i++)
            result.r[i] = p.r[i] + v[i];

        // Add rotational correction
        typedef typename Dune::SelectType<Dune::is_same<TVector,TangentVector>::value,
                                          typename Rotation<N,ctype>::TangentVector,
                                          typename Rotation<N,ctype>::EmbeddedTangentVector>::Type RotationTangentVector;
        RotationTangentVector qCorr;
        for (int i=0; i<RotationTangentVector::dimension; i++)
            qCorr[i] = v[N+i];

        result.q = Rotation<N,ctype>::exp(p.q, qCorr);
        return result;
    }

    /** \brief Compute geodesic distance from a to b */
    static T distance(const RigidBodyMotion<N,ctype>& a, const RigidBodyMotion<N,ctype>& b) {
        
        T euclideanDistanceSquared = (a.r - b.r).two_norm2();
        
        T rotationDistance = Rotation<N,ctype>::distance(a.q, b.q);
        
        return std::sqrt(euclideanDistanceSquared + rotationDistance*rotationDistance);
    }
    
    /** \brief Compute difference vector from a to b on the tangent space of a */
    static TangentVector difference(const RigidBodyMotion<N,ctype>& a,
                                    const RigidBodyMotion<N,ctype>& b) {

        TangentVector result;

        // Usual linear difference
        for (int i=0; i<N; i++)
            result[i] = a.r[i] - b.r[i];

        // Subtract orientations on the tangent space of 'a'
        typename Rotation<N,ctype>::TangentVector v = Rotation<N,ctype>::difference(a.q, b.q).axial();

        // Compute difference on T_a SO(3)
        for (int i=0; i<Rotation<N,ctype>::TangentVector::dimension; i++)
            result[i+N] = v[i];

        return result;
    }
    
    static EmbeddedTangentVector derivativeOfDistanceSquaredWRTSecondArgument(const RigidBodyMotion<N,ctype>& a,
                                                                              const RigidBodyMotion<N,ctype>& b) {

        // linear part
        Dune::FieldVector<ctype,N> linearDerivative = a.r;
        linearDerivative -= b.r;
        linearDerivative *= -2;

        // rotation part
        typename Rotation<N,ctype>::EmbeddedTangentVector rotationDerivative 
                = Rotation<N,ctype>::derivativeOfDistanceSquaredWRTSecondArgument(a.q, b.q);
        
        return concat(linearDerivative, rotationDerivative);
    }
    
    /** \brief Compute the Hessian of the squared distance function keeping the first argument fixed */
    static Dune::FieldMatrix<T,embeddedDim,embeddedDim> secondDerivativeOfDistanceSquaredWRTSecondArgument(const RigidBodyMotion<N,ctype> & p, const RigidBodyMotion<N,ctype> & q)
    {
        Dune::FieldMatrix<T,embeddedDim,embeddedDim> result(0);
        
        // The linear part
        Dune::FieldMatrix<T,N,N> linearPart = RealTuple<N>::secondDerivativeOfDistanceSquaredWRTSecondArgument(p.r,q.r);
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
                result[i][j] = linearPart[i][j];

        // The rotation part
        Dune::FieldMatrix<T,Rotation<N,T>::embeddedDim,Rotation<N,T>::embeddedDim> rotationPart 
                = Rotation<N,ctype>::secondDerivativeOfDistanceSquaredWRTSecondArgument(p.q,q.q);
        for (int i=0; i<Rotation<N,T>::embeddedDim; i++)
            for (int j=0; j<Rotation<N,T>::embeddedDim; j++)
                result[N+i][N+j] = rotationPart[i][j];

        return result;
    }
    
    /** \brief Compute the mixed second derivate \partial d^2 / \partial da db

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static Dune::FieldMatrix<T,embeddedDim,embeddedDim> secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(const RigidBodyMotion<N,ctype> & p, const RigidBodyMotion<N,ctype> & q)
    {
        Dune::FieldMatrix<T,embeddedDim,embeddedDim> result(0);
        
        // The linear part
        Dune::FieldMatrix<T,N,N> linearPart = RealTuple<N>::secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(p.r,q.r);
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
                result[i][j] = linearPart[i][j];

        // The rotation part
        Dune::FieldMatrix<T,Rotation<N,T>::embeddedDim,Rotation<N,T>::embeddedDim> rotationPart 
                = Rotation<N,ctype>::secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(p.q,q.q);
        for (int i=0; i<Rotation<N,T>::embeddedDim; i++)
            for (int j=0; j<Rotation<N,T>::embeddedDim; j++)
                result[N+i][N+j] = rotationPart[i][j];

        return result;
    }
    
    /** \brief Compute the third derivative \partial d^3 / \partial dq^3

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static Tensor3<T,embeddedDim,embeddedDim,embeddedDim> thirdDerivativeOfDistanceSquaredWRTSecondArgument(const RigidBodyMotion<N,ctype> & p, const RigidBodyMotion<N,ctype> & q)
    {
        Tensor3<T,embeddedDim,embeddedDim,embeddedDim> result(0);
        
        // The linear part
        Tensor3<T,N,N,N> linearPart = RealTuple<N>::thirdDerivativeOfDistanceSquaredWRTSecondArgument(p.r,q.r);
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
                for (int k=0; k<N; k++)
                    result[i][j][k] = linearPart[i][j][k];

        // The rotation part
        Tensor3<T,Rotation<N,T>::embeddedDim,Rotation<N,T>::embeddedDim,Rotation<N,T>::embeddedDim> rotationPart 
                = Rotation<N,ctype>::thirdDerivativeOfDistanceSquaredWRTSecondArgument(p.q,q.q);
                
        for (int i=0; i<Rotation<N,T>::embeddedDim; i++)
            for (int j=0; j<Rotation<N,T>::embeddedDim; j++)
                for (int k=0; k<Rotation<N,T>::embeddedDim; k++)
                    result[N+i][N+j][N+k] = rotationPart[i][j][k];

        return result;
    }
    
    /** \brief Compute the mixed third derivative \partial d^3 / \partial da db^2

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static Tensor3<T,embeddedDim,embeddedDim,embeddedDim> thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(const RigidBodyMotion<N,ctype> & p, const RigidBodyMotion<N,ctype> & q)
    {
        Tensor3<T,embeddedDim,embeddedDim,embeddedDim> result(0);
        
        // The linear part
        Tensor3<T,N,N,N> linearPart = RealTuple<N>::thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(p.r,q.r);
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
                for (int k=0; k<N; k++)
                    result[i][j][k] = linearPart[i][j][k];

        // The rotation part
        Tensor3<T,Rotation<N,T>::embeddedDim,Rotation<N,T>::embeddedDim,Rotation<N,T>::embeddedDim> rotationPart = Rotation<N,ctype>::thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(p.q,q.q);
        for (int i=0; i<Rotation<N,T>::embeddedDim; i++)
            for (int j=0; j<Rotation<N,T>::embeddedDim; j++)
                for (int k=0; k<Rotation<N,T>::embeddedDim; k++)
                    result[N+i][N+j][N+k] = rotationPart[i][j][k];

        return result;
    }

    
    
    /** \brief Project tangent vector of R^n onto the tangent space */
    EmbeddedTangentVector projectOntoTangentSpace(const EmbeddedTangentVector& v) const {
        DUNE_THROW(Dune::NotImplemented, "!");
    }

    
    /** \brief Compute an orthonormal basis of the tangent space of SE(3).

    This basis may not be globally continuous.
    */
    Dune::FieldMatrix<T,dim,embeddedDim> orthonormalFrame() const {
        Dune::FieldMatrix<T,dim,embeddedDim> result(0);

        // Get the R^d part
        for (int i=0; i<N; i++)
            result[i][i] = 1;
        
        Dune::FieldMatrix<T,Rotation<N>::dim,Rotation<N>::embeddedDim> SO3Part = q.orthonormalFrame();

        for (int i=0; i<Rotation<N>::dim; i++)
            for (int j=0; j<Rotation<N>::embeddedDim; j++)
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
    Rotation<N,ctype> q;
    
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
std::ostream& operator<< (std::ostream& s, const RigidBodyMotion<N,ctype>& c)
  {
      s << "(" << c.r << ")  (" << c.q << ")";
      return s;
  }

#endif
