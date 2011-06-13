#ifndef RIGID_BODY_MOTION_HH
#define RIGID_BODY_MOTION_HH

#include <dune/common/fvector.hh>

#include <dune/gfe/realtuple.hh>
#include "rotation.hh"

/** \brief A rigid-body motion in R^d, i.e., a member of SE(d) */
template <int dim, class T=double>
struct RigidBodyMotion
{
    /** \brief Type of an infinitesimal rigid body motion */
    typedef Dune::FieldVector<T, dim + Rotation<dim,T>::TangentVector::size> TangentVector;

    /** \brief Type of an infinitesimal rigid body motion */
    typedef Dune::FieldVector<T, dim + Rotation<dim,T>::EmbeddedTangentVector::size> EmbeddedTangentVector;

    /** \brief The type used for coordinates */
    typedef T ctype;
    
    /** \brief Default constructor */
    RigidBodyMotion()
    {}
    
    /** \brief Constructor from a translation and a rotation */
    RigidBodyMotion(const Dune::FieldVector<ctype, dim>& translation,
                    const Rotation<dim,ctype>& rotation)
    : r(translation), q(rotation)
    {}

    /** \brief The exponential map from a given point $p \in SE(d)$. 
     
     Why the template parameter?  Well, it should work with both TangentVector and EmbeddedTangentVector.
     In general these differ and we could just have two exp methods.  However in 2d they do _not_ differ,
     and then the compiler complains about having two methods with the same signature.
     */
    template <class TVector>
    static RigidBodyMotion<dim,ctype> exp(const RigidBodyMotion<dim,ctype>& p, const TVector& v) {

        RigidBodyMotion<dim,ctype> result;

        // Add translational correction
        for (int i=0; i<dim; i++)
            result.r[i] = p.r[i] + v[i];

        // Add rotational correction
        typedef typename Dune::SelectType<Dune::is_same<TVector,TangentVector>::value,
                                          typename Rotation<dim,ctype>::TangentVector,
                                          typename Rotation<dim,ctype>::EmbeddedTangentVector>::Type RotationTangentVector;
        RotationTangentVector qCorr;
        for (int i=0; i<RotationTangentVector::size; i++)
            qCorr[i] = v[dim+i];

        result.q = Rotation<dim,ctype>::exp(p.q, qCorr);
        return result;
    }

    /** \brief Compute geodesic distance from a to b */
    static T distance(const RigidBodyMotion<dim,ctype>& a, const RigidBodyMotion<dim,ctype>& b) {
        
        T euclideanDistanceSquared = (a.r - b.r).two_norm2();
        
        T rotationDistance = Rotation<dim,ctype>::distance(a.q, b.q);
        
        return std::sqrt(euclideanDistanceSquared + rotationDistance*rotationDistance);
    }
    
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
    
    static EmbeddedTangentVector derivativeOfDistanceSquaredWRTSecondArgument(const RigidBodyMotion<dim,ctype>& a,
                                                                              const RigidBodyMotion<dim,ctype>& b) {

        // linear part
        Dune::FieldVector<ctype,dim> linearDerivative = a.r;
        linearDerivative -= b.r;
        linearDerivative *= -2;

        // rotation part
        typename Rotation<dim,ctype>::EmbeddedTangentVector rotationDerivative 
                = Rotation<dim,ctype>::derivativeOfDistanceSquaredWRTSecondArgument(a.q, b.q);
        
        return concat(linearDerivative, rotationDerivative);
    }
    
    /** \brief Compute the Hessian of the squared distance function keeping the first argument fixed */
    static Dune::FieldMatrix<double,7,7> secondDerivativeOfDistanceSquaredWRTSecondArgument(const RigidBodyMotion<dim,ctype> & p, const RigidBodyMotion<dim,ctype> & q)
    {
        Dune::FieldMatrix<double,7,7> result(0);
        
        // The linear part
        Dune::FieldMatrix<double,3,3> linearPart = RealTuple<3>::secondDerivativeOfDistanceSquaredWRTSecondArgument(p.r,q.r);
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                result[i][j] = linearPart[i][j];

        // The rotation part
        Dune::FieldMatrix<double,4,4> rotationPart = Rotation<dim,ctype>::secondDerivativeOfDistanceSquaredWRTSecondArgument(p.q,q.q);
        for (int i=0; i<4; i++)
            for (int j=0; j<4; j++)
                result[3+i][3+j] = rotationPart[i][j];

        return result;
    }
    
    /** \brief Compute the mixed second derivate \partial d^2 / \partial da db

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static Dune::FieldMatrix<double,7,7> secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(const RigidBodyMotion<dim,ctype> & p, const RigidBodyMotion<dim,ctype> & q)
    {
        Dune::FieldMatrix<double,7,7> result(0);
        
        // The linear part
        Dune::FieldMatrix<double,3,3> linearPart = RealTuple<3>::secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(p.r,q.r);
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                result[i][j] = linearPart[i][j];

        // The rotation part
        Dune::FieldMatrix<double,4,4> rotationPart = Rotation<dim,ctype>::secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(p.q,q.q);
        for (int i=0; i<4; i++)
            for (int j=0; j<4; j++)
                result[3+i][3+j] = rotationPart[i][j];

        return result;
    }
    
    /** \brief Compute the third derivative \partial d^3 / \partial dq^3

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static Tensor3<double,7,7,7> thirdDerivativeOfDistanceSquaredWRTSecondArgument(const RigidBodyMotion<dim,ctype> & p, const RigidBodyMotion<dim,ctype> & q)
    {
        Tensor3<double,7,7,7> result(0);
        
        // The linear part
        Tensor3<double,3,3,3> linearPart = RealTuple<3>::thirdDerivativeOfDistanceSquaredWRTSecondArgument(p.r,q.r);
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                for (int k=0; k<3; k++)
                    result[i][j][k] = linearPart[i][j][k];

        // The rotation part
        Tensor3<double,4,4,4> rotationPart = Rotation<dim,ctype>::thirdDerivativeOfDistanceSquaredWRTSecondArgument(p.q,q.q);
        for (int i=0; i<4; i++)
            for (int j=0; j<4; j++)
                for (int k=0; k<4; k++)
                    result[3+i][3+j][3+j] = rotationPart[i][j][k];

        return result;
    }
    
    /** \brief Compute the mixed third derivative \partial d^3 / \partial da db^2

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static Tensor3<double,7,7,7> thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(const RigidBodyMotion<dim,ctype> & p, const RigidBodyMotion<dim,ctype> & q)
    {
        Tensor3<double,7,7,7> result(0);
        
        // The linear part
        Tensor3<double,3,3,3> linearPart = RealTuple<3>::thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(p.r,q.r);
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                for (int k=0; k<3; k++)
                    result[i][j][k] = linearPart[i][j][k];

        // The rotation part
        Tensor3<double,4,4,4> rotationPart = Rotation<dim,ctype>::thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(p.q,q.q);
        for (int i=0; i<4; i++)
            for (int j=0; j<4; j++)
                for (int k=0; k<4; k++)
                    result[3+i][3+j][3+j] = rotationPart[i][j][k];

        return result;
    }

    
    
    /** \brief Project tangent vector of R^n onto the tangent space */
    EmbeddedTangentVector projectOntoTangentSpace(const EmbeddedTangentVector& v) const {
        DUNE_THROW(Dune::NotImplemented, "!");
    }

    
    /** \brief Compute an orthonormal basis of the tangent space of SE(3).

    This basis is of course not globally continuous.
    */
    Dune::FieldMatrix<double,6,7> orthonormalFrame() const {
        Dune::FieldMatrix<double,6,7> result(0);

        // Get the R^d part
        for (int i=0; i<dim; i++)
            result[i][i] = 1;
        
        Dune::FieldMatrix<double,3,4> SO3Part = q.orthonormalFrame();

        for (int i=0; i<dim; i++)
            for (int j=0; j<4; j++)
                result[3+i][3+j] = SO3Part[i][j];

        return result;
    }
    


    // Translational part
    Dune::FieldVector<ctype, dim> r;

    // Rotational part
    Rotation<dim,ctype> q;
    
private:
    
    /** \brief Concatenate two FieldVectors */
    template <int N, int M>
    static Dune::FieldVector<ctype,N+M> concat(const Dune::FieldVector<ctype,N>& a,
                                               const Dune::FieldVector<ctype,M>& b)
    {
        Dune::FieldVector<ctype,N+M> result;
        for (int i=0; i<N; i++)
            result[i] = a[i];
        for (int i=0; i<M; i++)
            result[i+N] = b[i];
        return result;
    }

};

//! Send configuration to output stream
template <int dim, class ctype>
std::ostream& operator<< (std::ostream& s, const RigidBodyMotion<dim,ctype>& c)
  {
      s << "(" << c.r << ")  (" << c.q << ")";
      return s;
  }

#endif
