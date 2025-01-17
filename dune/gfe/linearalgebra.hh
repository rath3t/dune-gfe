#ifndef DUNE_GFE_LINEARALGEBRA_HH
#define DUNE_GFE_LINEARALGEBRA_HH

#include <random>
#include <type_traits>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>

#include <dune/fufem/utilities/adolcnamespaceinjections.hh>  // For PromotedType


///////////////////////////////////////////////////////////////////////////////////////////
//  Various matrix methods
///////////////////////////////////////////////////////////////////////////////////////////

namespace Dune::GFE
{

#if ADOLC_ADOUBLE_H
  /** \brief Calculates ret = s*A, where A has as field_type of adouble.
   *
   * The function template is disabled if s isn't a scalar or adolc type.
   */
  template<typename T1,int m, int n,class = typename std::enable_if< std::is_scalar_v<T1> || std::is_base_of_v<badouble,T1> >::type >
  auto operator* ( const T1& s, const Dune::FieldMatrix<adouble, m, n> &A)
  {
    typedef typename Dune::FieldMatrix<adouble,m,n> :: size_type size_type;
    Dune::FieldMatrix<adouble,m,n> ret;

    for( size_type i = 0; i < m; ++i )
      for( size_type j = 0; j < n; ++j )
        ret[i][j] = s * A[i][j];

    return ret;
  }

  /** \brief Calculates ret = A*v, where A has as field_type of adouble.
   *
   * The function template is disabled if the field_type of v is no an adolc type
   */
  template<typename T1,int m, int n,class = typename std::enable_if<  std::is_base_of_v<badouble,T1> >::type >
  auto operator* (const Dune::FieldMatrix<double, m, n> &A,  const Dune::FieldVector<T1,n>& v )
  {
    typedef typename Dune::FieldMatrix<adouble,m,n> :: size_type size_type;
    Dune::FieldVector<adouble,m> ret(0.0);

    for( size_type i = 0; i < m; ++i )
      for( size_type j = 0; j < n; ++j )
        ret[i] +=  A[i][j]*v[j];

    return ret;
  }


  /** \brief Calculates ret = A*s, where A has as field_type of adouble.
   *
   * The function template is disabled if s isn't a scalar or adolc type.
   */
  template<typename T1,int m, int n,class = typename std::enable_if< std::is_scalar_v<T1> || std::is_base_of_v<badouble,T1> >::type >
  auto operator* (const Dune::FieldMatrix<adouble, m, n> &A, const T1& s )
  {
    return s*A;
  }
#endif

  /** \brief Return the trace of a matrix */
  template <class T, int n>
  static T trace(const FieldMatrix<T,n,n>& A)
  {
    T trace = 0;
    for (int i=0; i<n; i++)
      trace += A[i][i];
    return trace;
  }

  /** \brief Return the square of the trace of a matrix */
  template <class T, int n>
  static T traceSquared(const FieldMatrix<T,n,n>& A)
  {
    T trace = 0;
    for (int i=0; i<n; i++)
      trace += A[i][i];
    return trace*trace;
  }

  /** \brief Compute the symmetric part of a matrix A, i.e. \f$ \frac 12 (A + A^T) \f$ */
  template <class T, int n>
  static FieldMatrix<T,n,n> sym(const FieldMatrix<T,n,n>& A)
  {
    FieldMatrix<T,n,n> result;
    for (int i=0; i<n; i++)
      for (int j=0; j<n; j++)
        result[i][j] = 0.5 * (A[i][j] + A[j][i]);
    return result;
  }

  /** \brief Compute the antisymmetric part of a matrix A, i.e. \f$ \frac 12 (A - A^T) \f$ */
  template <class T, int n>
  static FieldMatrix<T,n,n> skew(const FieldMatrix<T,n,n>& A)
  {
    FieldMatrix<T,n,n> result;
    for (int i=0; i<n; i++)
      for (int j=0; j<n; j++)
        result[i][j] = 0.5 * (A[i][j] - A[j][i]);
    return result;
  }

  /** \brief Compute the deviator of a matrix A */
  template <class T, int n>
  static FieldMatrix<T,n,n> dev(const FieldMatrix<T,n,n>& A)
  {
    FieldMatrix<T,n,n> result = A;
    auto t = trace(A);
    for (int i=0; i<n; i++)
      result[i][i] -= t / n;
    return result;
  }

  /** \brief Return the transposed matrix */
  template <class T, int n, int m>
  static FieldMatrix<T,m,n> transpose(const FieldMatrix<T,n,m>& A)
  {
    FieldMatrix<T,m,n> result;

    for (int i=0; i<m; i++)
      for (int j=0; j<n; j++)
        result[i][j] = A[j][i];

    return result;
  }

  /** \brief The Frobenius (i.e., componentwise) product of two matrices */
  template <class T, int n>
  static T frobeniusProduct(const FieldMatrix<T,n,n>& A, const FieldMatrix<T,n,n>& B)
  {
    T result(0.0);

    for (int i=0; i<n; i++)
      for (int j=0; j<n; j++)
        result += A[i][j] * B[i][j];

    return result;
  }


  /** \brief Return a*b^T */
  template <class T1,class T2, int n, int m>
  static auto dyadicProduct(const FieldVector<T1,n>& a, const FieldVector<T2,m>& b)
  {
    using ScalarResultType = typename Dune::PromotionTraits<T1,T2>::PromotedType;
    FieldMatrix<ScalarResultType,n,m> result;
    for (int i=0; i<n; i++)
      for (int j=0; j<m; j++)
        result[i][j] = a[i]*b[j];

    return result;
  }


  /** \brief Get the requested column of fieldmatrix */
  template<typename field_type, int cols, int rows>
  auto col(const Dune::FieldMatrix<field_type, rows, cols> &mat, const int requestedCol)
  {
    Dune::FieldVector<field_type, rows> col;

    for (int i = 0; i < rows; ++i)
      col[i] = mat[i][requestedCol];

    return col;
  }

  /** \brief Return a segment of a FieldVector from lower up to lower+size-1 */
  template< int lower, int size,typename field_type,int n>
  static FieldVector<field_type,size> segment(const FieldVector<field_type,n>& v)
  {
    FieldVector<field_type,size> res;
    std::copy(v.begin()+lower,v.begin()+lower+size,res.begin());
    return res;
  }

  /** \brief Return a segment of a FieldVector from lower up to lower+size-1
   * lower is unknown at compile time*/
  template< int size,typename field_type,int n>
  static FieldVector<field_type,size> segmentAt(const FieldVector<field_type,n>& v,const size_t lower)
  {
    FieldVector<field_type,size> res;
    std::copy(v.begin()+lower,v.begin()+lower+size,res.begin());
    return res;
  }

  /** \brief Return a block of a FieldMatrix  (lower1...lower1+size1-1,lower2...lower2+size2-1 */
  template< int lower1, int size1, int lower2,int size2,typename field_type,int n,int m>
  static auto block(const FieldMatrix<field_type,n,m>& v)
  {
    static_assert(lower1+size1<=n && lower2+size2<=m, "Size mismatch for Block!");
    FieldMatrix<field_type,size1,size2> res;

    for(int i=lower1; i<lower1+size1; ++i)
      for(int j=lower2; j<lower2+size2; ++j)
        res[i-lower1][j-lower2] = v[i][j];
    return res;
  }

  /** \brief Return a block of a FieldMatrix  (lower1...lower1+size1-1,lower2...lower2+size2-1
   * * lower1 and lower2 are unknown at compile time*/
  template< int size1,int size2,typename field_type,int n,int m>
  static auto blockAt(const FieldMatrix<field_type,n,m>& v, const size_t& lower1, const size_t& lower2)
  {
    assert(lower1+size1<=n && lower2+size2<=m);
    FieldMatrix<field_type,size1,size2> res;

    for(size_t i=lower1; i<lower1+size1; ++i)
      for(size_t j=lower2; j<lower2+size2; ++j)
        res[i-lower1][j-lower2] = v[i][j];
    return res;
  }

  /** \brief Generates FieldVector with random entries in the range -1..1 */
  template<typename field_type,int n>
  auto randomFieldVector(field_type lower=-1, field_type upper=1)
  {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<field_type> dist(lower, upper);
    auto rand = [&dist,&mt](){
                  return dist(mt);
                };
    FieldVector<field_type,n> vec;
    std::generate(vec.begin(), vec.end(), rand);
    return vec;
  }

}  // namespace Dune::GFE

#endif
