#ifndef DUNE_GFE_SPACES_UNITVECTOR_HH
#define DUNE_GFE_SPACES_UNITVECTOR_HH

#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/version.hh>
#include <dune/common/math.hh>

#include <dune/gfe/tensor3.hh>
#include <dune/gfe/symmetricmatrix.hh>

template <class T, int N>
class Rotation;

/** \brief A unit vector in R^N

    \tparam N Dimension of the embedding space
    \tparam T The type used for individual coordinates
 */
template <class T, int N>
class UnitVector
{
  // Rotation<T,3> is friend, because it needs the various derivatives of the arccos
  friend class Rotation<T,3>;

  /** \brief Computes sin(x) / x without getting unstable for small x */
  static T sinc(const T& x) {
    using std::sin;
    return (x < 1e-2) ? 1.0-x*x/6.0+ Dune::power(x,4)/120.0-Dune::power(x,6)/5040.0+Dune::power(x,8)/362880.0 : sin(x)/x;
  }

  /** \brief Compute arccos^2 without using the (at 1) nondifferentiable function acos x close to 1 */
  static T arcCosSquared(const T& x) {
    using std::acos;
    const T eps = 1e-2;
    if (x > 1-eps) {      // acos is not differentiable, use the series expansion instead,
      // we need here lots of terms to be sure that the numerical derivatives are also within maschine precision
      //return -2 * (x-1) + 1.0/3 * (x-1)*(x-1) - 4.0/45 * (x-1)*(x-1)*(x-1);
      return 11665028.0/4729725.0
             -141088.0/45045.0*x
             +   413.0/429.0*x*x
             -  5344.0/12285.0*Dune::power(x,3)
             +    245.0/1287.0*Dune::power(x,4)
             -  1632.0/25025.0*Dune::power(x,5)
             +     56.0/3861.0*Dune::power(x,6)
             -    32.0/21021.0*Dune::power(x,7);
    } else {
      return Dune::power(acos(x),2);
    }
  }

  /** \brief Compute the derivative of arccos^2 without getting unstable for x close to 1 */
  static T derivativeOfArcCosSquared(const T& x) {
    using std::acos;
    using std::sqrt;
    const T eps = 1e-2;
    if (x > 1-eps) {      // regular expression is unstable, use the series expansion instead
      // we need here lots of terms to be sure that the numerical derivatives are also within maschine precision
      //return -2 + 2*(x-1)/3 - 4/15*(x-1)*(x-1);
      return -47104.0/15015.0
             +12614.0/6435.0*x
             -63488.0/45045.0*x*x
             + 1204.0/1287.0*Dune::power(x,3)
             - 2048.0/4095.0*Dune::power(x,4)
             +   112.0/585.0*Dune::power(x,5)
             -2048.0/45045.0*Dune::power(x,6)
             +   32.0/6435.0*Dune::power(x,7);

    } else if (x < -1+eps) {      // The function is not differentiable
      DUNE_THROW(Dune::Exception, "arccos^2 is not differentiable at x==-1!");
    } else
      return -2*acos(x) / sqrt(1-x*x);
  }

  /** \brief Compute the second derivative of arccos^2 without getting unstable for x close to 1 */
  static T secondDerivativeOfArcCosSquared(const T& x) {
    using std::acos;
    using std::pow;
    const T eps = 1e-2;
    if (x > 1-eps) {      // regular expression is unstable, use the series expansion instead
      // we need here lots of terms to be sure that the numerical derivatives are also within maschine precision
      //return 2.0/3 - 8*(x-1)/15;
      return 1350030.0/676039.0+5632.0/2028117.0*Dune::power(x,10)
             -1039056896.0/334639305.0*x
             +150876.0/39767.0*x*x
             -445186048.0/111546435.0*Dune::power(x,3)
             +       343728.0/96577.0*Dune::power(x,4)
             -  57769984.0/22309287.0*Dune::power(x,5)
             +      710688.0/482885.0*Dune::power(x,6)
             -  41615360.0/66927861.0*Dune::power(x,7)
             +     616704.0/3380195.0*Dune::power(x,8)
             -     245760.0/7436429.0*Dune::power(x,9);
    } else if (x < -1+eps) {      // The function is not differentiable
      DUNE_THROW(Dune::Exception, "arccos^2 is not differentiable at x==-1!");
    } else
      return 2/(1-x*x) - 2*x*acos(x) / pow(1-x*x,1.5);
  }

  /** \brief Compute the third derivative of arccos^2 without getting unstable for x close to 1 */
  static T thirdDerivativeOfArcCosSquared(const T& x) {
    using std::acos;
    using std::sqrt;
    const T eps = 1e-2;
    if (x > 1-eps) {      // regular expression is unstable, use the series expansion instead
      // we need here lots of terms to be sure that the numerical derivatives are also within maschine precision
      //return -8.0/15 + 24*(x-1)/35;
      return -1039056896.0/334639305.0
             +301752.0/39767.0*x
             -445186048.0/37182145.0*x*x
             +1374912.0/96577.0*Dune::power(x,3)
             -288849920.0/22309287.0*Dune::power(x,4)
             +4264128.0/482885.0*Dune::power(x,5)
             -41615360.0/9561123.0*Dune::power(x,6)
             +4933632.0/3380195.0*Dune::power(x,7)
             -2211840.0/7436429.0*Dune::power(x,8)
             +56320.0/2028117.0*Dune::power(x,9);
    } else if (x < -1+eps) {      // The function is not differentiable

      DUNE_THROW(Dune::Exception, "arccos^2 is not differentiable at x==-1!");
    } else {
      T d = 1-x*x;
      return 6*x/(d*d) - 6*x*x*acos(x)/(d*d*sqrt(d)) - 2*acos(x)/(d*sqrt(d));
    }
  }

  template <class T2, int N2>
  friend class UnitVector;

public:

  /** \brief The type used for coordinates */
  typedef T ctype;
  typedef T field_type;

  /** \brief The type used for global coordinates */
  typedef Dune::FieldVector<T,N> CoordinateType;

  /** \brief Dimension of the manifold formed by unit vectors */
  static const int dim = N-1;

  /** \brief Dimension of the Euclidean space the manifold is embedded in */
  static const int embeddedDim = N;

  typedef Dune::FieldVector<T,N-1> TangentVector;

  typedef Dune::FieldVector<T,N> EmbeddedTangentVector;

  /** \brief The global convexity radius of the unit sphere */
  static constexpr double convexityRadius = 0.5*M_PI;

  /** \brief The return type of the derivativeOfProjection method */
  typedef Dune::FieldMatrix<T, N, N> DerivativeOfProjection;

  /** \brief Default constructor */
  UnitVector()
  {}

  /** \brief Constructor from a vector.  The vector gets normalized! */
  UnitVector(const Dune::FieldVector<T,N>& vector)
    : data_(vector)
  {
    data_ /= data_.two_norm();
  }

  /** \brief Constructor from an array.  The array gets normalized! */
  UnitVector(const std::array<T,N>& vector)
  {
    for (int i=0; i<N; i++)
      data_[i] = vector[i];
    data_ /= data_.two_norm();
  }

  /** \brief Assignment from UnitVector with different type -- used for automatic differentiation with ADOL-C */
  template <class T2>
  UnitVector& operator <<= (const UnitVector<T2,N>& other) {
    for (int i=0; i<N; i++)
      data_[i] <<= other.data_[i];
    return *this;
  }

  /** \brief Rebind the UnitVector to another coordinate type */
  template<class U>
  struct rebind
  {
    typedef UnitVector<U,N> other;
  };



  UnitVector<T,N>& operator=(const Dune::FieldVector<T,N>& vector)
  {
    data_ = vector;
    data_ /= data_.two_norm();
    return *this;
  }

  /** \brief The exponential map */
  static UnitVector exp(const UnitVector& p, const TangentVector& v) {

    Dune::FieldMatrix<T,N-1,N> frame = p.orthonormalFrame();

    EmbeddedTangentVector ev;
    frame.mtv(v,ev);

    return exp(p,ev);
  }

  /** \brief The exponential map */
  static UnitVector exp(const UnitVector& p, const EmbeddedTangentVector& v) {

    using std::abs;
    using std::cos;
    assert( abs(p.data_*v) < 1e-5 );

    const T norm = v.two_norm();
    UnitVector result = p;
    result.data_ *= cos(norm);
    result.data_.axpy(sinc(norm), v);
    return result;
  }

  /** \brief The inverse of the exponential map
   *
   * \results A vector in the tangent space of p
   */
  static EmbeddedTangentVector log(const UnitVector& p, const UnitVector& q)
  {
    EmbeddedTangentVector result = p.projectOntoTangentSpace(q.data_-p.data_);
    if (result.two_norm() > 1e-10)
      result *= distance(p,q) / result.two_norm();
    return result;
  }

  /** \brief Length of the great arc connecting the two points */
  static T distance(const UnitVector& a, const UnitVector& b) {

    using std::acos;
    using std::min;

    // Not nice: we are in a class for unit vectors, but the class is actually
    // supposed to handle perturbations of unit vectors as well.  Therefore
    // we normalize here.
    T x = a.data_ * b.data_/a.data_.two_norm()/b.data_.two_norm();

    // paranoia:  if the argument is just eps larger than 1 acos returns NaN
    x = min(x,1.0);

    return acos(x);
  }

#if ADOLC_ADOUBLE_H
  /** \brief Squared length of the great arc connecting the two points */
  static adouble distanceSquared(const UnitVector<double,N>& a, const UnitVector<adouble,N>& b)
  {
    // Not nice: we are in a class for unit vectors, but the class is actually
    // supposed to handle perturbations of unit vectors as well.  Therefore
    // we normalize here.
    adouble x = a.data_ * b.data_ / (a.data_.two_norm()*b.data_.two_norm());

    // paranoia:  if the argument is just eps larger than 1 acos returns NaN
    using std::min;
    x = min(x,1.0);

    // Special implementation that remains AD-differentiable near x==1
    return arcCosSquared(x);
  }
#endif

  /** \brief Compute the gradient of the squared distance function keeping the first argument fixed

     Unlike the distance itself the squared distance is differentiable at zero
   */
  static EmbeddedTangentVector derivativeOfDistanceSquaredWRTSecondArgument(const UnitVector& a, const UnitVector& b) {
    T x = a.data_ * b.data_;

    EmbeddedTangentVector result = a.data_;

    result *= derivativeOfArcCosSquared(x);

    // Project gradient onto the tangent plane at b in order to obtain the surface gradient
    result = b.projectOntoTangentSpace(result);

    // Gradient must be a tangent vector at b, in other words, orthogonal to it
    using std::abs;
    assert(abs(b.data_ * result) < 1e-5);

    return result;
  }

  /** \brief Compute the Hessian of the squared distance function keeping the first argument fixed

     Unlike the distance itself the squared distance is differentiable at zero
   */
  static Dune::SymmetricMatrix<T,N> secondDerivativeOfDistanceSquaredWRTSecondArgument(const UnitVector& p, const UnitVector& q) {

    T sp = p.data_ * q.data_;

    Dune::FieldVector<T,N> pProjected = q.projectOntoTangentSpace(p.globalCoordinates());

    Dune::SymmetricMatrix<T,N> A;
    for (int i=0; i<N; i++)
      for (int j=0; j<=i; j++)
        A(i,j) = pProjected[i]*pProjected[j];

    A *= secondDerivativeOfArcCosSquared(sp);

    // Compute matrix B (see notes)
    Dune::SymmetricMatrix<T,N> Pq;
    for (int i=0; i<N; i++)
      for (int j=0; j<=i; j++)
        Pq(i,j) = (i==j) - q.data_[i]*q.data_[j];

    // Bring it all together
    A.axpy(-1*derivativeOfArcCosSquared(sp)*sp, Pq);

    return A;
  }

  /** \brief Compute the mixed second derivate \partial d^2 / \partial da db

     Unlike the distance itself the squared distance is differentiable at zero
   */
  static Dune::FieldMatrix<T,N,N> secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(const UnitVector& a, const UnitVector& b) {

    T sp = a.data_ * b.data_;

    // Compute vector A (see notes)
    Dune::FieldMatrix<T,1,N> row;
    row[0] = b.projectOntoTangentSpace(a.globalCoordinates());

    Dune::FieldVector<T,N> tmp = a.projectOntoTangentSpace(b.globalCoordinates());
    Dune::FieldMatrix<T,N,1> column;
    for (int i=0; i<N; i++)      // turn row vector into column vector
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
  static Tensor3<T,N,N,N> thirdDerivativeOfDistanceSquaredWRTSecondArgument(const UnitVector& p, const UnitVector& q) {

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
  static Tensor3<T,N,N,N> thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(const UnitVector& p, const UnitVector& q) {

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


  /** \brief Project tangent vector of R^n onto the tangent space */
  EmbeddedTangentVector projectOntoTangentSpace(const EmbeddedTangentVector& v) const {
    EmbeddedTangentVector result = v;
    result.axpy(-1*(data_*result), data_);
    return result;
  }

  /** \brief Project tangent vector of R^n onto the normal space space */
  EmbeddedTangentVector projectOntoNormalSpace(const EmbeddedTangentVector& v) const {

    EmbeddedTangentVector result;

    T sp = 0;
    for (int i=0; i<N; i++)
      sp += v[i] * data_[i];

    for (int i=0; i<N; i++)
      result[i] = sp * data_[i];

    return result;
  }

  /** \brief The Weingarten map */
  EmbeddedTangentVector weingarten(const EmbeddedTangentVector& z, const EmbeddedTangentVector& v) const {

    EmbeddedTangentVector result;

    T sp = 0;
    for (int i=0; i<N; i++)
      sp += v[i] * data_[i];

    for (int i=0; i<N; i++)
      result[i] = -sp * z[i];

    return result;
  }

  static UnitVector<T,N> projectOnto(const CoordinateType& p)
  {
    UnitVector<T,N> result(p);
    result.data_ /= result.data_.two_norm();
    return result;
  }

  static DerivativeOfProjection derivativeOfProjection(const Dune::FieldVector<T,N>& p)
  {
    auto normSquared = p.two_norm2();
    auto norm = std::sqrt(normSquared);

    Dune::FieldMatrix<T,N,N> result;
    for (int i=0; i<N; i++)
      for (int j=0; j<N; j++)
        result[i][j] = ( (i==j) - p[i]*p[j] / normSquared ) / norm;
    return result;
  }

  /** \brief The global coordinates, if you really want them */
  const CoordinateType& globalCoordinates() const {
    return data_;
  }

  /** \brief Compute an orthonormal basis of the tangent space of S^n.

     This basis is of course not globally continuous.
   */
  Dune::FieldMatrix<T,N-1,N> orthonormalFrame() const {

    Dune::FieldMatrix<T,N-1,N> result;

    // Coordinates of the stereographic projection
    Dune::FieldVector<T,N-1> X;

    if (data_[N-1] <= 0) {

      // Stereographic projection from the north pole onto R^{N-1}
      for (size_t i=0; i<N-1; i++)
        X[i] = data_[i] / (1-data_[N-1]);

    } else {

      // Stereographic projection from the south pole onto R^{N-1}
      for (size_t i=0; i<N-1; i++)
        X[i] = data_[i] / (1+data_[N-1]);

    }

    T RSquared = X.two_norm2();

    for (size_t i=0; i<N-1; i++)
      for (size_t j=0; j<N-1; j++)
        // Note: the matrix is the transpose of the one in the paper
        result[j][i] = 2*(i==j)*(1+RSquared) - 4*X[i]*X[j];

    for (size_t j=0; j<N-1; j++)
      result[j][N-1] = 4*X[j];

    // Upper hemisphere: adapt formulas so it is the stereographic projection from the south pole
    if (data_[N-1] > 0)
      for (size_t j=0; j<N-1; j++)
        result[j][N-1] *= -1;

    // normalize the rows to make the orthogonal basis orthonormal
    for (size_t i=0; i<N-1; i++)
      result[i] /= result[i].two_norm();

    return result;
  }

  /** \brief Write unit vector object to output stream */
  friend std::ostream& operator<< (std::ostream& s, const UnitVector& unitVector)
  {
    return s << unitVector.data_;
  }


private:

  Dune::FieldVector<T,N> data_;
};

#endif
