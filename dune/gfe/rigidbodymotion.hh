#ifndef RIGID_BODY_MOTION_HH
#define RIGID_BODY_MOTION_HH

#include <dune/common/fvector.hh>
#include "rotation.hh"

/** \brief A rigid-body motion in, R^d, i.e., a member of SE(d) */
template <int dim, class T=double>
struct RigidBodyMotion
{
    /** \brief Global coordinates wrt an isometric embedding function are available */
    static const bool globalIsometricCoordinates = Rotation<dim,T>::globalIsometricCoordinates;

    /** \brief Type of an infinitesimal rigid body motion */
    typedef Dune::FieldVector<T, dim + Rotation<dim,T>::TangentVector::size> TangentVector;

    /** \brief Type of an infinitesimal rigid body motion */
    typedef Dune::FieldVector<T, dim + Rotation<dim,T>::EmbeddedTangentVector::size> EmbeddedTangentVector;

    /** \brief The type used for coordinates */
    typedef T ctype;

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

    // Translational part
    Dune::FieldVector<ctype, dim> r;;

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
