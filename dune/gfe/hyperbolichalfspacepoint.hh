#ifndef HYPERBOLIC_HALF_SPACE_POINT_HH
#define HYPERBOLIC_HALF_SPACE_POINT_HH

#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/power.hh>

#include <dune/istl/scaledidmatrix.hh>

#include <dune/gfe/tensor3.hh>

/** \brief A point in the hyperbolic half-space H^N

    \tparam N Dimension of the hyperbolic space space
    \tparam T The type used for individual coordinates
*/
template <class T, int N>
class HyperbolicHalfspacePoint
{
    dune_static_assert(N>=2, "A hyperbolic half-space needs to be at least two-dimensional!");
    
    /** \brief Compute the derivative of arccosh^2 without getting unstable for x close to 1 */
    static T derivativeOfArcCosHSquared(const T& x) {
        const T eps = 1e-4;
        if (x < 1+eps) {  // regular expression is unstable, use the series expansion instead
            return 2 - 2*(x-1)/3 + 4/15*(x-1)*(x-1);
        } else
            return 2*std::acosh(x) / std::sqrt(x*x-1);
    }

    /** \brief Compute the second derivative of arccosh^2 without getting unstable for x close to 1 */
    static T secondDerivativeOfArcCosHSquared(const T& x) {
        const T eps = 1e-4;
        if (x < 1+eps) {  // regular expression is unstable, use the series expansion instead
            return -2.0/3 + 8*(x-1)/15;
        } else
            return 2/(x*x-1) - 2*x*std::acosh(x) / std::pow(x*x-1,1.5);
    }

    /** \brief Compute the third derivative of arccos^2 without getting unstable for x close to 1 */
    static T thirdDerivativeOfArcCosHSquared(const T& x) {
        const T eps = 1e-4;
        if (x < 1+eps) {  // regular expression is unstable, use the series expansion instead
            return 8.0/15 - 24*(x-1)/35;
        } else {
            T d = x*x-1;
            return -6*x/(d*d) + (4*x*x+2)*std::acosh(x)/(std::pow(d,5.2));
        }
    }

    /** \brief Compute derivative of $F(p,q) = 1 + ||p-q||^2 / 2p_nq_n with respect to p 
     \param[in] diffNormSquared Expected to be ||p-q||^2, taken from the caller for efficiency reasons
     */
    static Dune::FieldVector<T,N> computeDFdp(const HyperbolicHalfspacePoint& p, const HyperbolicHalfspacePoint& q, const T& diffNormSquared)
    {
        Dune::FieldVector<T,N> result;
        
        for (size_t i=0; i<N-1; i++)
            result[i] = ( p.data_[i] - q.data_[i] ) / (q.data_[N-1] * p.data_[N-1]);
        
        result[N-1] = - diffNormSquared / (2*p.data_[N-1]*p.data_[N-1]*q.data_[N-1]) + (p.data_[N-1] - q.data_[N-1]) / (p.data_[N-1]*q.data_[N-1]);
        
        return result;
    }
    
    /** \brief Compute derivative of $F(p,q) = 1 + ||p-q||^2 / 2p_nq_n with respect to q 
     \param[in] diffNormSquared Expected to be ||p-q||^2, taken from the caller for efficiency reasons
     */
    static Dune::FieldVector<T,N> computeDFdq(const HyperbolicHalfspacePoint& p, const HyperbolicHalfspacePoint& q, const T& diffNormSquared)
    {
        Dune::FieldVector<T,N> result;
        
        for (size_t i=0; i<N-1; i++)
            result[i] = ( q.data_[i] - p.data_[i] ) / (q.data_[N-1] * p.data_[N-1]);
        
        result[N-1] = - diffNormSquared / (2*p.data_[N-1]*q.data_[N-1]*q.data_[N-1]) - (p.data_[N-1] - q.data_[N-1]) / (p.data_[N-1]*q.data_[N-1]);
        
        return result;
    }
    
    /** \brief Compute second derivative of $F(p,q) = 1 + ||p-q||^2 / 2p_nq_n with respect to p and q 
     \param[in] diffNormSquared Expected to be ||p-q||^2, taken from the caller for efficiency reasons
     */
    static Dune::FieldMatrix<T,N,N> computeDFdpdq(const HyperbolicHalfspacePoint& a, const HyperbolicHalfspacePoint& b, const T& diffNormSquared)
    {
        // abbreviate notation
        const Dune::FieldVector<T,N>& p = a.data_;
        const Dune::FieldVector<T,N>& q = b.data_;
        
        Dune::FieldMatrix<T,N,N> dFdpdq;
                      
        for (size_t i=0; i<N; i++) {
            
            for (size_t j=0; j<N; j++) {

                if (i!=N-1 and j!=N-1) {
                    
                    dFdpdq[i][j] = -(i==j) / (p[N-1]*q[N-1]);
                    
                } else if (i!=N-1 and j==N-1) {
                    
                    dFdpdq[i][j] = -(p[i] - q[i]) / (p[N-1]*q[N-1]*q[N-1]);
                    
                } else if (i==N-1 and j!=N-1) {
                    
                    dFdpdq[i][j] = (p[j] - q[j]) / (p[N-1]*p[N-1]*q[N-1]);
                    
                } else if (i==N-1 and j==N-1) {
                    
                    dFdpdq[i][j] = -1/(p[N-1]*q[N-1]) 
                                   - (p[N-1]-q[N-1]) / (p[N-1]*q[N-1]*q[N-1]) 
                                   + (p[N-1]-q[N-1]) / (p[N-1]*p[N-1]*q[N-1]) + diffNormSquared / (2*p[N-1]*p[N-1]*q[N-1]*q[N-1]);
                
                }
                
            }
            
        }

        return dFdpdq;
    }
    
    /** \brief Compute second derivative of $F(p,q) = 1 + ||p-q||^2 / 2p_nq_n with respect to q 
     \param[in] diffNormSquared Expected to be ||p-q||^2, taken from the caller for efficiency reasons
     */
    static Dune::FieldMatrix<T,N,N> computeDFdqdq(const HyperbolicHalfspacePoint& a, const HyperbolicHalfspacePoint& b, const T& diffNormSquared)
    {
        // abbreviate notation
        const Dune::FieldVector<T,N>& p = a.data_;
        const Dune::FieldVector<T,N>& q = b.data_;
        
        Dune::FieldMatrix<T,N,N> dFdqdq;
               
        for (size_t i=0; i<N; i++) {
            
            for (size_t j=0; j<N; j++) {

                if (i!=N-1 and j!=N-1) {
                    
                    dFdqdq[i][j] = (i==j) / (p[N-1]*q[N-1]);
                    
                } else if (i!=N-1 and j==N-1) {
                    
                    dFdqdq[i][j] = (p[i] - q[i]) / (p[N-1]*q[N-1]*q[N-1]);
                    
                } else if (i==N-1 and j!=N-1) {
                    
                    dFdqdq[i][j] = (p[j] - q[j]) / (p[N-1]*q[N-1]*q[N-1]);
                    
                } else if (i==N-1 and j==N-1) {
                    
                    dFdqdq[i][j] = 1/(q[N-1]*q[N-1]) + (p[N-1]-q[N-1]) / (p[N-1]*q[N-1]*q[N-1]) + diffNormSquared / (p[N-1]*q[N-1]*q[N-1]*q[N-1]);
                
                }
                
            }
            
        }

        return dFdqdq;
    }
    
    
    
public:

    /** \brief The type used for coordinates */
    typedef T ctype;

    /** \brief The type used for global coordinates */
    typedef Dune::FieldVector<T,N> CoordinateType;
    
    /** \brief Dimension of the manifold */
    static const int dim = N;

    /** \brief Dimension of the Euclidean space the manifold is embedded in */
    static const int embeddedDim = N;

    /** \brief Type of a tangent vector in local coordinates */
    typedef Dune::FieldVector<T,N> TangentVector;

    /** \brief Type of a tangent vector in the embedding space */
    typedef Dune::FieldVector<T,N> EmbeddedTangentVector;
    
    /** \brief The global convexity radius of the hyberbolic plane */
    static constexpr T convexityRadius = std::numeric_limits<T>::infinity();
    
    /** \brief Default constructor */
    HyperbolicHalfspacePoint()
    {}
    
    /** \brief Constructor from a vector.  The vector gets normalized */
    HyperbolicHalfspacePoint(const Dune::FieldVector<T,N>& vector)
        : data_(vector)
    {
        assert(vector[N-1]>0);
    }
    
    /** \brief Constructor from an array.  The array gets normalized */
    HyperbolicHalfspacePoint(const Dune::array<T,N>& vector)
    {
        assert(vector.back()>0);
        for (int i=0; i<N; i++)
            data_[i] = vector[i];
    }

     /** \brief The exponential map */
    static HyperbolicHalfspacePoint exp(const HyperbolicHalfspacePoint& p, const TangentVector& v) {
        
        assert (N==2);
        
        T vNorm = v.two_norm();
        
        // we compute geodesics by applying an isometry to a fixed unit-speed geodesic.
        // Hence we need a unit velocity vector.
        if (vNorm <= 0)
            return p;

        TangentVector vUnit = v;
        vUnit /= vNorm;

        // Compute the coefficients a,b,c,d of the Moebius transform that transforms
        // the unit speed upward geodesic to the one through p with direction vUnit.
        // We expect the Moebius transform to be an isometry, i.e. ad-bc = 1.
        T cc = 1/(2*p.data_[N-1]) - vUnit[N-1] / (2*p.data_[N-1]*p.data_[N-1]);
        T dd = 1/(2*p.data_[N-1]) + vUnit[N-1] / (2*p.data_[N-1]*p.data_[N-1]);
        T ac = vUnit[0] / (2*p.data_[N-1]) + p.data_[0]*cc;
        T bd = p.data_[0] / p.data_[N-1] - ac;
        
        HyperbolicHalfspacePoint result;
        
        // vertical part
        result.data_[1] = std::exp(vNorm) / (cc*std::exp(2*vNorm) + dd);
        
        // horizontal part
        result.data_[0] = (ac*std::exp(2*vNorm) + bd) / (cc*std::exp(2*vNorm) + dd);
        
        return result;
    }

    /** \brief Hyperbolic distance between two points
     * 
     * \f dist(a,b) = arccosh ( 1 + ||a-b||^2 / (2a_n b_n) \f
     */
     static T distance(const HyperbolicHalfspacePoint& a, const HyperbolicHalfspacePoint& b) {

         T result(0);
         
         for (size_t i=0; i<N; i++)
             result += (a.data_[i]-b.data_[i])*(a.data_[i]-b.data_[i]);
         
         return std::acosh(1 + result / (2*a.data_[N-1]*b.data_[N-1]));
    }

    /** \brief Compute the gradient of the squared distance function keeping the first argument fixed

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static EmbeddedTangentVector derivativeOfDistanceSquaredWRTSecondArgument(const HyperbolicHalfspacePoint& a, const HyperbolicHalfspacePoint& b) {
        
        T diffNormSquared(0);
         
        for (size_t i=0; i<N; i++)
            diffNormSquared += (a.data_[i]-b.data_[i])*(a.data_[i]-b.data_[i]);

        TangentVector result = computeDFdq(a,b,diffNormSquared);
        
        T x = 1 + diffNormSquared/ (2*a.data_[N-1]*b.data_[N-1]);
        
        result *= derivativeOfArcCosHSquared(x);

        return result;
    }

    /** \brief Compute the Hessian of the squared distance function keeping the first argument fixed

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static Dune::FieldMatrix<T,N,N> secondDerivativeOfDistanceSquaredWRTSecondArgument(const HyperbolicHalfspacePoint& a, const HyperbolicHalfspacePoint& b) {

        T diffNormSquared = (a.data_-b.data_).two_norm2();

        // Compute first derivative of F
        Dune::FieldVector<T,N> dFdq = computeDFdq(a,b,diffNormSquared);

        // Compute second derivatives of F
        Dune::FieldMatrix<T,N,N> dFdqdq = computeDFdqdq(a,b,diffNormSquared);
        
        //
        T x = 1 + diffNormSquared/ (2*a.data_[N-1]*b.data_[N-1]);
        T alphaPrime      = derivativeOfArcCosHSquared(x);
        T alphaPrimePrime = secondDerivativeOfArcCosHSquared(x);

        // Sum it all together
        Dune::FieldMatrix<T,N,N> result;
        for (size_t i=0; i<N; i++)
            for (size_t j=0; j<N; j++)
                result[i][j] = alphaPrimePrime * dFdq[i] * dFdq[j] + alphaPrime * dFdqdq[i][j];
        
        return result;
    }

    /** \brief Compute the mixed second derivative \partial d^2 / \partial da db

     */
    static Dune::FieldMatrix<T,N,N> secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(const HyperbolicHalfspacePoint& a, const HyperbolicHalfspacePoint& b)
    {
        // abbreviate notation
        const Dune::FieldVector<T,N>& p = a.data_;
        const Dune::FieldVector<T,N>& q = b.data_;
        
        T diffNormSquared = (p-q).two_norm2();

        // Compute first derivatives of F with respect to p and q
        Dune::FieldVector<T,N> dFdp = computeDFdp(a,b,diffNormSquared);
        Dune::FieldVector<T,N> dFdq = computeDFdq(a,b,diffNormSquared);

        // Compute second derivatives of F
        Dune::FieldMatrix<T,N,N> dFdpdq = computeDFdpdq(a,b,diffNormSquared);
        
        //
        T x = 1 + diffNormSquared/ (2*p[N-1]*q[N-1]);
        T alphaPrime      = derivativeOfArcCosHSquared(x);
        T alphaPrimePrime = secondDerivativeOfArcCosHSquared(x);

        // Sum it all together
        Dune::FieldMatrix<T,N,N> result;
        for (size_t i=0; i<N; i++)
            for (size_t j=0; j<N; j++)
                result[i][j] = alphaPrimePrime * dFdp[i] * dFdq[j] + alphaPrime * dFdpdq[i][j];
        
        return result;
    }
    
    
    /** \brief Compute the third derivative \partial d^3 / \partial dq^3

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static Tensor3<T,N,N,N> thirdDerivativeOfDistanceSquaredWRTSecondArgument(const HyperbolicHalfspacePoint& a, const HyperbolicHalfspacePoint& b)
    {
        Tensor3<T,N,N,N> result;

        // abbreviate notation
        const Dune::FieldVector<T,N>& p = a.data_;
        const Dune::FieldVector<T,N>& q = b.data_;
        
        T diffNormSquared = (p-q).two_norm2();

        // Compute first derivative of F
        Dune::FieldVector<T,N> dFdq = computeDFdq(a,b,diffNormSquared);

        // Compute second derivatives of F
        Dune::FieldMatrix<T,N,N> dFdqdq = computeDFdqdq(a,b,diffNormSquared);

        // Compute third derivatives of F
        Tensor3<T,N,N,N> dFdqdqdq;
       
        for (size_t i=0; i<N; i++) {
            
            for (size_t j=0; j<N; j++) {

                for (size_t k=0; k<N; k++) {
                    
                    if (i!=N-1 and j!=N-1 and k!=N-1) {
                    
                        dFdqdqdq[i][j][k] = 0;
                    
                    } else if (i!=N-1 and j!=N-1 and k==N-1) {
                    
                        dFdqdqdq[i][j][k] = -(i==j) / (p[N-1]*q[N-1]*q[N-1]);
                    
                    } else if (i!=N-1 and j==N-1 and k!=N-1) {
                    
                        dFdqdqdq[i][j][k] = -(i==k) / (p[N-1]*q[N-1]*q[N-1]);
                    
                    } else if (i!=N-1 and j==N-1 and k==N-1) {
                    
                        dFdqdqdq[i][j][k] = -2*(p[i] - q[i]) / (p[N-1]*Dune::Power<3>::eval(q[N-1]));
                    
                    } else if (i==N-1 and j!=N-1 and k!=N-1) {
                    
                        dFdqdqdq[i][j][k] = - (j==k) / (p[N-1]*q[N-1]*q[N-1]);
                    
                    } else if (i==N-1 and j!=N-1 and k==N-1) {
                    
                        dFdqdqdq[i][j][k] = -2*(p[j] - q[j]) / (p[N-1]*Dune::Power<3>::eval(q[N-1]));
                    
                    } else if (i==N-1 and j==N-1 and k!=N-1) {
                    
                        dFdqdqdq[i][j][k] = -2*(p[k] - q[k]) / (p[N-1]*Dune::Power<3>::eval(q[N-1]));
                
                    } else if (i==N-1 and j==N-1 and k==N-1) {
                    
                        dFdqdqdq[i][j][k] = (-1 -6*p[N-1] + 4*q[N-1] -3*diffNormSquared)/(p[N-1]*Dune::Power<3>::eval(q[N-1])); 
                
                    }
                    
                }
                
            }
            
        }

        //
        T x = 1 + diffNormSquared/ (2*p[N-1]*q[N-1]);
        T alphaPrime           = derivativeOfArcCosHSquared(x);
        T alphaPrimePrime      = secondDerivativeOfArcCosHSquared(x);
        T alphaPrimePrimePrime = thirdDerivativeOfArcCosHSquared(x);

        // Sum it all together
        for (size_t i=0; i<N; i++)
            for (size_t j=0; j<N; j++)
                for (size_t k=0; k<N; k++)
                    result[i][j][k] = alphaPrimePrimePrime * dFdq[i] * dFdq[j] * dFdq[k]
                                    + alphaPrimePrime * (dFdqdq[i][j] * dFdq[k] + dFdqdq[i][k] * dFdq[j] + dFdqdq[j][k] * dFdq[i])
                                    + alphaPrime * dFdqdqdq[i][j][k];

        return result;
    }    
        
    /** \brief Compute the mixed third derivative \partial d^3 / \partial da db^2
     */
    static Tensor3<T,N,N,N> thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(const HyperbolicHalfspacePoint& a, const HyperbolicHalfspacePoint& b)
    {
        Tensor3<T,N,N,N> result;

        // abbreviate notation
        const Dune::FieldVector<T,N>& p = a.data_;
        const Dune::FieldVector<T,N>& q = b.data_;
        
        T diffNormSquared = (p-q).two_norm2();

        // Compute first derivatives of F with respect to p and q
        Dune::FieldVector<T,N> dFdp = computeDFdp(a,b,diffNormSquared);
        Dune::FieldVector<T,N> dFdq = computeDFdq(a,b,diffNormSquared);

        // Compute second derivatives of F
        Dune::FieldMatrix<T,N,N> dFdqdq = computeDFdqdq(a,b,diffNormSquared);

        Dune::FieldMatrix<T,N,N> dFdpdq = computeDFdpdq(a,b,diffNormSquared);

        // Compute third derivatives of F
        Tensor3<T,N,N,N> dFdpdqdq;
       
        for (size_t i=0; i<N; i++) {
            
            for (size_t j=0; j<N; j++) {

                for (size_t k=0; k<N; k++) {
                    
                    if (i!=N-1 and j!=N-1 and k!=N-1) {
                    
                        dFdpdqdq[i][j][k] = 0;
                    
                    } else if (i!=N-1 and j!=N-1 and k==N-1) {
                    
                        dFdpdqdq[i][j][k] = (i==j) / (p[N-1]*q[N-1]*q[N-1]);
                    
                    } else if (i!=N-1 and j==N-1 and k!=N-1) {
                    
                        dFdpdqdq[i][j][k] = (i==k) / (p[N-1]*q[N-1]*q[N-1]);
                    
                    } else if (i!=N-1 and j==N-1 and k==N-1) {
                    
                        dFdpdqdq[i][j][k] = 2*(p[i] - q[i]) / (p[N-1]*Dune::Power<3>::eval(q[N-1]));
                    
                    } else if (i==N-1 and j!=N-1 and k!=N-1) {
                    
                        dFdpdqdq[i][j][k] = (j==k) / (p[N-1]*q[N-1]*q[N-1]);
                    
                    } else if (i==N-1 and j!=N-1 and k==N-1) {
                    
                        dFdpdqdq[i][j][k] = 2*(p[j] - q[j]) / (p[N-1]*Dune::Power<3>::eval(q[N-1]));
                    
                    } else if (i==N-1 and j==N-1 and k!=N-1) {
                    
                        dFdpdqdq[i][j][k] = 2*(p[k] - q[k]) / (p[N-1]*Dune::Power<3>::eval(q[N-1]));
                
                    } else if (i==N-1 and j==N-1 and k==N-1) {
                    
                        dFdpdqdq[i][j][k] = 1.0/(p[N-1]*p[N-1]*q[N-1]*q[N-1]) + 1.0/(p[N-1]*q[N-1]*q[N-1])
                                          + 2*(p[N-1]-q[N-1])/(p[N-1]*Dune::Power<3>::eval(q[N-1]))
                                          + 2*(p[N-1]-q[N-1])/(p[N-1]*p[N-1]*q[N-1])
                                          - diffNormSquared / (p[N-1]*p[N-1]*q[N-1]*q[N-1]);
                
                    }
                    
                }
                
            }
            
        }

        //
        T x = 1 + diffNormSquared/ (2*p[N-1]*q[N-1]);
        T alphaPrime           = derivativeOfArcCosHSquared(x);
        T alphaPrimePrime      = secondDerivativeOfArcCosHSquared(x);
        T alphaPrimePrimePrime = thirdDerivativeOfArcCosHSquared(x);

        // Sum it all together
        for (size_t i=0; i<N; i++)
            for (size_t j=0; j<N; j++)
                for (size_t k=0; k<N; k++)
                    result[i][j][k] = alphaPrimePrimePrime * dFdp[i] * dFdq[j] * dFdq[k]
                                    + alphaPrimePrime * (dFdpdq[i][j] * dFdq[k] + dFdpdq[i][k] * dFdq[j] + dFdqdq[j][k] * dFdp[i])
                                    + alphaPrime * dFdpdqdq[i][j][k];

        return result;

    }
    
    
    /** \brief Project tangent vector of R^n onto the tangent space.  For H^m this is the identity */
    EmbeddedTangentVector projectOntoTangentSpace(const EmbeddedTangentVector& v) const {
        return v;
    }

    /** \brief The global coordinates, if you really want them */
    const CoordinateType& globalCoordinates() const {
        return data_;
    }

    /** \brief Compute an orthonormal basis of the tangent space of H^N.
    */
    Dune::FieldMatrix<T,N,N> orthonormalFrame() const {

        Dune::ScaledIdentityMatrix<T,N> result( data_[N-1] );

        return Dune::FieldMatrix<T,N,N>(result);
    }
    
    /** \brief Scalar product of two tangent vectors */
    T metric(const TangentVector& v, const TangentVector& w) const
    {
        return v*w/(data_[N-1]*data_[N-1]);
    }

    /** \brief Write unit vector object to output stream */
    friend std::ostream& operator<< (std::ostream& s, const HyperbolicHalfspacePoint& p)
    {
        return s << p.data_;
    }


private:

    Dune::FieldVector<T,N> data_;
};

#endif
