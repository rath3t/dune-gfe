#ifndef HYPERBOLIC_HALF_SPACE_POINT_HH
#define HYPERBOLIC_HALF_SPACE_POINT_HH

#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>

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
    
    /** \brief Computes sin(x) / x without getting unstable for small x */
    static T sinc(const T& x) {
        return (x < 1e-4) ? 1 - (x*x/6) : std::sin(x)/x;
    }

    /** \brief Compute the derivative of arccos^2 without getting unstable for x close to 1 */
    static T derivativeOfArcCosSquared(const T& x) {
        const T eps = 1e-4;
        if (x > 1-eps) {  // regular expression is unstable, use the series expansion instead
            return -2 + 2*(x-1)/3 - 4/15*(x-1)*(x-1);
        } else if (x < -1+eps) {  // The function is not differentiable
            DUNE_THROW(Dune::Exception, "arccos^2 is not differentiable at x==-1!");
        } else
            return -2*std::acos(x) / std::sqrt(1-x*x);
    }

    /** \brief Compute the derivative of arccosh^2 without getting unstable for x close to 1 */
    static T derivativeOfArcCosHSquared(const T& x) {
        const T eps = 1e-4;
        if (x > 1-eps) {  // regular expression is unstable, use the series expansion instead
            return 2 - 2*(x-1)/3 + 4/15*(x-1)*(x-1);
        } else
            return 2*std::acosh(x) / std::sqrt(x*x-1);
    }

    /** \brief Compute the second derivative of arccos^2 without getting unstable for x close to 1 */
    static T secondDerivativeOfArcCosSquared(const T& x) {
        const T eps = 1e-4;
        if (x > 1-eps) {  // regular expression is unstable, use the series expansion instead
            return 2.0/3 - 8*(x-1)/15;
        } else if (x < -1+eps) {  // The function is not differentiable
            DUNE_THROW(Dune::Exception, "arccos^2 is not differentiable at x==-1!");
        } else
            return 2/(1-x*x) - 2*x*std::acos(x) / std::pow(1-x*x,1.5);
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
    static T thirdDerivativeOfArcCosSquared(const T& x) {
        const T eps = 1e-4;
        if (x > 1-eps) {  // regular expression is unstable, use the series expansion instead
            return -8.0/15 + 24*(x-1)/35;
        } else if (x < -1+eps) {  // The function is not differentiable
            DUNE_THROW(Dune::Exception, "arccos^2 is not differentiable at x==-1!");
        } else {
            T d = 1-x*x;
            return 6*x/(d*d) - 6*x*x*std::acos(x)/(d*d*std::sqrt(d)) - 2*std::acos(x)/(d*std::sqrt(d));
        }
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
        
        TangentVector result;
        
        T diffNormSquared(0);
         
        for (size_t i=0; i<N; i++)
            diffNormSquared += (a.data_[i]-b.data_[i])*(a.data_[i]-b.data_[i]);

        for (size_t i=0; i<N-1; i++)
            result[i] = ( b.data_[i] - a.data_[i] ) / (a.data_[N-1] * b.data_[N-1]);
        
        result[N-1] = - diffNormSquared / (2*a.data_[N-1]*b.data_[N-1]*b.data_[N-1]) - (a.data_[N-1] - b.data_[N-1]) / (a.data_[N-1]*b.data_[N-1]);
        
        T x = 1 + diffNormSquared/ (2*a.data_[N-1]*b.data_[N-1]);
        
        result *= derivativeOfArcCosHSquared(x);

        return result;
    }

    /** \brief Compute the Hessian of the squared distance function keeping the first argument fixed

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static Dune::FieldMatrix<T,N,N> secondDerivativeOfDistanceSquaredWRTSecondArgument(const HyperbolicHalfspacePoint& a, const HyperbolicHalfspacePoint& b) {

        // abbreviate notation
        const Dune::FieldVector<T,N>& p = a.data_;
        const Dune::FieldVector<T,N>& q = b.data_;
        
        T diffNormSquared = (p-q).two_norm2();

        // Compute first derivative of F
        Dune::FieldVector<T,N> dFdq;
        for (size_t i=0; i<N-1; i++)
            dFdq[i] = ( b.data_[i] - a.data_[i] ) / (a.data_[N-1] * b.data_[N-1]);
        
        dFdq[N-1] = - diffNormSquared / (2*a.data_[N-1]*b.data_[N-1]*b.data_[N-1]) - (a.data_[N-1] - b.data_[N-1]) / (a.data_[N-1]*b.data_[N-1]);

        // Compute second derivatives of F
        Dune::FieldMatrix<T,N,N> dFdqdq;
       
        for (size_t i=0; i<N; i++) {
            
            for (size_t j=0; j<N; j++) {

                if (i!=N-1 and j!=N-1) {
                    
                    dFdqdq[i][j] = (i==j) / (p[N-1]*q[N-1]);
                    
                } else if (i!=N-1 and j==N-1) {
                    
                    dFdqdq[i][j] = (p[i] - q[i]) / (p[N-1]*q[N-1]*q[N-1]);
                    
                } else if (i!=N-1 and j==N-1) {
                    
                    dFdqdq[i][j] = (p[j] - q[j]) / (p[N-1]*q[N-1]*q[N-1]);
                    
                } else if (i==N-1 and j==N-1) {
                    
                    dFdqdq[i][j] = 1/(q[N-1]*q[N-1]) + (p[N-1]*q[N-1]) / (p[N-1]*q[N-1]*q[N-1]) + diffNormSquared / (p[N-1]*q[N-1]*q[N-1]*q[N-1]);
                
                }
                
            }
            
        }
        
        //
        T x = 1 + diffNormSquared/ (2*p[N-1]*q[N-1]);
        T alphaPrime      = derivativeOfArcCosHSquared(x);
        T alphaPrimePrime = secondDerivativeOfArcCosHSquared(x);

        // Sum it all together
        Dune::FieldMatrix<T,N,N> result;
        for (size_t i=0; i<N; i++)
            for (size_t j=0; j<N; j++)
                result[i][j] = alphaPrimePrime * dFdq[i] * dFdq[j] + alphaPrime * dFdqdq[i][j];
        
        return result;
    }

    /** \brief Compute the mixed second derivate \partial d^2 / \partial da db

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static Dune::FieldMatrix<T,N,N> secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(const HyperbolicHalfspacePoint& a, const HyperbolicHalfspacePoint& b) {

        T sp = a.data_ * b.data_;

        // Compute vector A (see notes)
        Dune::FieldMatrix<T,1,N> row;
        row[0] = b.projectOntoTangentSpace(a.globalCoordinates());

        Dune::FieldVector<T,N> tmp = a.projectOntoTangentSpace(b.globalCoordinates());
        Dune::FieldMatrix<T,N,1> column;
        for (int i=0; i<N; i++)  // turn row vector into column vector
            column[i] = tmp[i];

        Dune::FieldMatrix<T,N,N> A;
        // A = row * column
        Dune::FMatrixHelp::multMatrix(column,row,A);
        A *= secondDerivativeOfArcCosSquared(sp);

        // Compute matrix B (see notes)
        Dune::FieldMatrix<T,N,N> Pp, Pq;
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++) {
                Pp[i][j] = (i==j) - a.data_[i]*a.data_[j];
                Pq[i][j] = (i==j) - b.data_[i]*b.data_[j];
            }
            
        Dune::FieldMatrix<T,N,N> B;
        Dune::FMatrixHelp::multMatrix(Pp,Pq,B);

        // Bring it all together
        A.axpy(derivativeOfArcCosSquared(sp), B);

        return A;
    }
    
    
    /** \brief Compute the third derivative \partial d^3 / \partial dq^3

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static Tensor3<T,N,N,N> thirdDerivativeOfDistanceSquaredWRTSecondArgument(const HyperbolicHalfspacePoint& p, const HyperbolicHalfspacePoint& q) {

        Tensor3<T,N,N,N> result;

        T sp = p.data_ * q.data_;
        
        // The projection matrix onto the tangent space at p and q
        Dune::FieldMatrix<T,N,N> Pq;
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
                Pq[i][j] = (i==j) - q.globalCoordinates()[i]*q.globalCoordinates()[j];
            
        Dune::FieldVector<T,N> pProjected = q.projectOntoTangentSpace(p.globalCoordinates());

        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
                for (int k=0; k<N; k++) {

                    result[i][j][k] = thirdDerivativeOfArcCosSquared(sp) * pProjected[i] * pProjected[j] * pProjected[k]
                                    - secondDerivativeOfArcCosSquared(sp) * ((i==j)*sp + p.globalCoordinates()[i]*q.globalCoordinates()[j])*pProjected[k]
                                    - secondDerivativeOfArcCosSquared(sp) * ((i==k)*sp + p.globalCoordinates()[i]*q.globalCoordinates()[k])*pProjected[j]
                                    - secondDerivativeOfArcCosSquared(sp) * pProjected[i] * Pq[j][k] * sp
                                    + derivativeOfArcCosSquared(sp) * ((i==j)*q.globalCoordinates()[k] + (i==k)*q.globalCoordinates()[j]) * sp
                                    - derivativeOfArcCosSquared(sp) * p.globalCoordinates()[i] * Pq[j][k];
                }
                
        result = Pq * result;
                
        return result;
    }    
        
    /** \brief Compute the mixed third derivative \partial d^3 / \partial da db^2

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static Tensor3<T,N,N,N> thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(const HyperbolicHalfspacePoint& p, const HyperbolicHalfspacePoint& q) {

        Tensor3<T,N,N,N> result;

        T sp = p.data_ * q.data_;
        
        // The projection matrix onto the tangent space at p and q
        Dune::FieldMatrix<T,N,N> Pp, Pq;
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++) {
                Pp[i][j] = (i==j) - p.globalCoordinates()[i]*p.globalCoordinates()[j];
                Pq[i][j] = (i==j) - q.globalCoordinates()[i]*q.globalCoordinates()[j];
            }
            
        Dune::FieldVector<T,N> pProjected = q.projectOntoTangentSpace(p.globalCoordinates());
        Dune::FieldVector<T,N> qProjected = p.projectOntoTangentSpace(q.globalCoordinates());
        
        Tensor3<T,N,N,N> derivativeOfPqOTimesPq;
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
                for (int k=0; k<N; k++) {
                    derivativeOfPqOTimesPq[i][j][k] = 0;
                    for (int l=0; l<N; l++)
                        derivativeOfPqOTimesPq[i][j][k] += Pp[i][l] * (Pq[j][l]*pProjected[k] + pProjected[j]*Pq[k][l]);
                }
                
        result = thirdDerivativeOfArcCosSquared(sp)         * Tensor3<T,N,N,N>::product(qProjected,pProjected,pProjected)
                 + secondDerivativeOfArcCosSquared(sp)      * derivativeOfPqOTimesPq
                 - secondDerivativeOfArcCosSquared(sp) * sp * Tensor3<T,N,N,N>::product(qProjected,Pq)
                 - derivativeOfArcCosSquared(sp)            * Tensor3<T,N,N,N>::product(qProjected,Pq);
               
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

    /** \brief Compute an orthonormal basis of the tangent space of S^n.

    This basis is of course not globally continuous.
    */
    Dune::FieldMatrix<T,N,N> orthonormalFrame() const {

        Dune::ScaledIdentityMatrix<T,N> result( data_[N-1]*data_[N-1] );

        return Dune::FieldMatrix<T,N,N>(result);
    }

    /** \brief Write unit vector object to output stream */
    friend std::ostream& operator<< (std::ostream& s, const HyperbolicHalfspacePoint& unitVector)
    {
        return s << unitVector.data_;
    }


private:

    Dune::FieldVector<T,N> data_;
};

#endif
