#ifndef REAL_TUPLE_HH
#define REAL_TUPLE_HH

#include <dune/common/fvector.hh>

#include <dune/istl/scaledidmatrix.hh>

#include <dune/gfe/tensor3.hh>
#include <dune/gfe/symmetricmatrix.hh>


/** \brief Implement a tuple of real numbers as a Riemannian manifold

Currently this class only exists for testing purposes.
*/

template <class T, int N>
class RealTuple
{
public:

    typedef T ctype;
    typedef T field_type;

    /** \brief The type used for global coordinates */
    typedef Dune::FieldVector<T,N> CoordinateType;

    /** \brief Dimension of the manifold formed by unit vectors */
    static const int dim = N;

    /** \brief Dimension of the Euclidean space the manifold is embedded in */
    static const int embeddedDim = N;

    typedef Dune::FieldVector<T,N> EmbeddedTangentVector;

    typedef Dune::FieldVector<T,N> TangentVector;

    /** \brief The global convexity radius of the Euclidean space */
    static constexpr double convexityRadius = std::numeric_limits<double>::infinity();

    /** \brief The return type of the derivativeOfProjection method */
    typedef Dune::ScaledIdentityMatrix<T, N> DerivativeOfProjection;

    /** \brief Default constructor */
    RealTuple()
    {}

    /** \brief Construction from a scalar */
    RealTuple(T v)
    {
        data_ = v;
    }

    /** \brief Copy constructor */
    RealTuple(const RealTuple<T,N>& other)
        : data_(other.data_)
    {}

    /** \brief Constructor from FieldVector*/
    RealTuple(const Dune::FieldVector<T,N>& other)
        : data_(other)
    {}

    RealTuple& operator=(const Dune::FieldVector<T,N>& other) {
        data_ = other;
        return *this;
    }

    RealTuple& operator-=(const Dune::FieldVector<T,N>& other) {
        data_ -= other;
        return *this;
    }

    template <class T2>
    RealTuple& operator-=(const RealTuple<T2,N>& other) {
        data_ -= other.data_;
        return *this;
    }

    /** \brief Assigment from RealTuple with different type -- used for automatic differentiation with ADOL-C */
    template <class T2>
    RealTuple& operator <<= (const RealTuple<T2,N>& other) {
        for (size_t i=0; i<N; i++)
            data_[i] <<= other.data_[i];
        return *this;
    }

    /** \brief Const random-access operator*/
    T operator[] (const size_t indexVariable ) const {
        return data_[indexVariable];
    }

     /** \brief Rebind the RealTuple to another coordinate type */
    template<class U>
    struct rebind
    {
      typedef RealTuple<U,N> other;
    };

    /** \brief The exponention map */
    static RealTuple exp(const RealTuple& p, const TangentVector& v) {
        return RealTuple(p.data_+v);
    }

    /** \brief Geodesic distance between two points

    Simply the Euclidean distance */
    static T distance(const RealTuple& a, const RealTuple& b) {
        return log(a.data_,b.data_).two_norm();
    }

    /** \brief The logarithmic map
     *
     * \result A vector in the tangent space of a, viz: b-a
     * */
    static auto log(const RealTuple& a, const RealTuple& b) {
        return b.data_ - a.data_;
    }

#if ADOLC_ADOUBLE_H
    /** \brief Geodesic distance squared between two points

    Simply the Euclidean distance squared */
    static adouble distanceSquared(const RealTuple<double,N>& a, const RealTuple<adouble,N>& b) {
      adouble result(0.0);
      for (int i=0; i<N; i++)
        result += (a.globalCoordinates()[i] - b.globalCoordinates()[i]) * (a.globalCoordinates()[i] - b.globalCoordinates()[i]);
      return result;
    }
#endif


    /** \brief Compute the gradient of the squared distance function keeping the first argument fixed

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static EmbeddedTangentVector derivativeOfDistanceSquaredWRTSecondArgument(const RealTuple& a, const RealTuple& b) {
        EmbeddedTangentVector result = a.data_;
        result -= b.data_;
        result *= -2;
        return result;
    }

        /** \brief Compute the Hessian of the squared distance function keeping the first argument fixed

    Unlike the distance itself the squared distance is differentiable at zero
        */
    static Dune::SymmetricMatrix<T,N> secondDerivativeOfDistanceSquaredWRTSecondArgument(const RealTuple& a, const RealTuple& b) {

        Dune::SymmetricMatrix<T,N> result;
        for (int i=0; i<N; i++)
            for (int j=0; j<=i; j++)
                result(i,j) = 2*(i==j);

        return result;
    }

    /** \brief Compute the mixed second derivate \partial d^2 / \partial da db

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static Dune::FieldMatrix<T,N,N> secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(const RealTuple& a, const RealTuple& b) {

        Dune::FieldMatrix<T,N,N> result;
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
                result[i][j] = -2*(i==j);

        return result;
    }

    /** \brief Compute the mixed third derivative \partial d^3 / \partial db^3

        The result is the constant zero-tensor.
     */
    static Tensor3<T,N,N,N> thirdDerivativeOfDistanceSquaredWRTSecondArgument(const RealTuple& a, const RealTuple& b) {
        return Tensor3<T,N,N,N>(0);
    }

    /** \brief Compute the mixed third derivative \partial d^3 / \partial da db^2

        The result is the constant zero-tensor.
     */
    static Tensor3<T,N,N,N> thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(const RealTuple& a, const RealTuple& b) {
        return Tensor3<T,N,N,N>(0);
    }

    /** \brief Project tangent vector of R^n onto the tangent space */
    EmbeddedTangentVector projectOntoTangentSpace(const EmbeddedTangentVector& v) const {
        return v;
    }

        /** \brief Project tangent vector of R^n onto the normal space space */
    EmbeddedTangentVector projectOntoNormalSpace(const EmbeddedTangentVector& v) const {
        return EmbeddedTangentVector(0);
    }

    /** \brief The Weingarten map */
    EmbeddedTangentVector weingarten(const EmbeddedTangentVector& z, const EmbeddedTangentVector& v) const {
        return EmbeddedTangentVector(0);
    }

    /** \brief Projection from the embedding space onto the manifold
     *
     * For RealTuples this is simply the identity map
     */
    static RealTuple<T,N> projectOnto(const CoordinateType& p)
    {
      return RealTuple<T,N>(p);
    }

    /** \brief Derivative of the projection from the embedding space onto the manifold
     *
     * For RealTuples this is simply the identity
     */
    static DerivativeOfProjection derivativeOfProjection(const Dune::FieldVector<T,N>& p)
    {
      return Dune::ScaledIdentityMatrix<T,N>(1.0);
    }

    /** \brief The global coordinates, if you really want them */
    const Dune::FieldVector<T,N>& globalCoordinates() const {
        return data_;
    }

     Dune::FieldVector<T,N>& globalCoordinates()  {
        return data_;
    }


    /** \brief Compute an orthonormal basis of the tangent space of R^n.

    In general this frame field, may of course not be continuous, but for RealTuples it is.
    */
    Dune::FieldMatrix<T,N,N> orthonormalFrame() const {

        Dune::FieldMatrix<T,N,N> result;

        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
                result[i][j] = (i==j);
        return result;
    }

    /** \brief Write LocalKey object to output stream */
    friend std::ostream& operator<< (std::ostream& s, const RealTuple& realTuple)
    {
        return s << realTuple.data_;
    }

private:

    Dune::FieldVector<T,N> data_;

};

#endif
