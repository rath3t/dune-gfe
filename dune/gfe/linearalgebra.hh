#ifndef DUNE_GFE_LINEARALGEBRA_HH
#define DUNE_GFE_LINEARALGEBRA_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/version.hh>

///////////////////////////////////////////////////////////////////////////////////////////
//  Various vector-space matrix methods
///////////////////////////////////////////////////////////////////////////////////////////

#if ADOLC_ADOUBLE_H
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
template< int m, int n, int p >
auto operator* ( const Dune::FieldMatrix< adouble, m, n > &A, const Dune::FieldMatrix< adouble, n, p > &B)
  -> Dune::FieldMatrix<adouble, m, p>
{
    typedef typename Dune::FieldMatrix< adouble, m, p > :: size_type size_type;
    Dune::FieldMatrix< adouble, m, p > ret;

    for( size_type i = 0; i < m; ++i ) {

        for( size_type j = 0; j < p; ++j ) {
            ret[ i ][ j ] = 0.0;
            for( size_type k = 0; k < n; ++k )
                ret[ i ][ j ] += A[ i ][ k ] * B[ k ][ j ];
        }
    }
    return ret;
}

template< int m, int n, int p >
auto operator* ( const Dune::FieldMatrix< adouble, m, n > &A, const Dune::FieldMatrix< double, n, p > &B)
  -> Dune::FieldMatrix<adouble, m, p>
{
    typedef typename Dune::FieldMatrix< adouble, m, p > :: size_type size_type;
    Dune::FieldMatrix< adouble, m, p > ret;

    for( size_type i = 0; i < m; ++i ) {

        for( size_type j = 0; j < p; ++j ) {
            ret[ i ][ j ] = 0.0;
            for( size_type k = 0; k < n; ++k )
                ret[ i ][ j ] += A[ i ][ k ] * B[ k ][ j ];
        }
    }
    return ret;
}

template< int m, int n, int p >
auto operator* ( const Dune::FieldMatrix< double, m, n > &A, const Dune::FieldMatrix< adouble, n, p > &B)
  -> Dune::FieldMatrix<adouble, m, p>
{
    typedef typename Dune::FieldMatrix< adouble, m, p > :: size_type size_type;
    Dune::FieldMatrix< adouble, m, p > ret;

    for( size_type i = 0; i < m; ++i ) {

        for( size_type j = 0; j < p; ++j ) {
            ret[ i ][ j ] = 0.0;
            for( size_type k = 0; k < n; ++k )
                ret[ i ][ j ] += A[ i ][ k ] * B[ k ][ j ];
        }
    }
    return ret;
}
#endif
#endif

#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
//! calculates ret = A + B
template< class K, int m, int n>
Dune::FieldMatrix<K,m,n> operator+ ( const Dune::FieldMatrix<K, m, n> &A, const Dune::FieldMatrix<K,m,n> &B)
{
    typedef typename Dune::FieldMatrix<K,m,n> :: size_type size_type;
    Dune::FieldMatrix<K,m,n> ret;

    for( size_type i = 0; i < m; ++i )
        for( size_type j = 0; j < n; ++j )
            ret[i][j] = A[i][j] + B[i][j];

    return ret;
}

//! calculates ret = A - B
template <class T, int m, int n>
auto operator- ( const Dune::FieldMatrix< T, m, n > &A, const Dune::FieldMatrix< T, m, n > &B)
  -> Dune::FieldMatrix<T, m, n>
{
    Dune::FieldMatrix<T,m,n> result;
    typedef typename decltype(result)::size_type size_type;

    for( size_type i = 0; i < m; ++i )
        for( size_type j = 0; j < n; ++j )
            result[i][j] = A[i][j] - B[i][j];

    return result;
}
#endif

#if ADOLC_ADOUBLE_H
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
//! calculates ret = A - B
template <int m, int n>
auto operator- ( const Dune::FieldMatrix< adouble, m, n > &A, const Dune::FieldMatrix< double, m, n > &B)
  -> Dune::FieldMatrix<adouble, m, n>
{
    Dune::FieldMatrix<adouble,m,n> result;
    typedef typename decltype(result)::size_type size_type;

    for( size_type i = 0; i < m; ++i )
        for( size_type j = 0; j < n; ++j )
            result[i][j] = A[i][j] - B[i][j];

    return result;
}
#endif

//! calculates ret = s*A
template< int m, int n>
auto operator* ( const double& s, const Dune::FieldMatrix<adouble, m, n> &A)
  -> Dune::FieldMatrix<adouble,m,n>
{
    typedef typename Dune::FieldMatrix<adouble,m,n> :: size_type size_type;
    Dune::FieldMatrix<adouble,m,n> ret;

    for( size_type i = 0; i < m; ++i )
        for( size_type j = 0; j < n; ++j )
            ret[i][j] = s * A[i][j];

    return ret;
}
#endif

#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
//! calculates ret = s*A
template< int m, int n>
auto operator* ( const double& s, const Dune::FieldMatrix<double, m, n> &A)
  -> Dune::FieldMatrix<double,m,n>
{
    typedef typename Dune::FieldMatrix<double,m,n> :: size_type size_type;
    Dune::FieldMatrix<double,m,n> ret;

    for( size_type i = 0; i < m; ++i )
        for( size_type j = 0; j < n; ++j )
            ret[i][j] = s * A[i][j];

    return ret;
}

//! calculates ret = A/s
template< class K, int m, int n>
Dune::FieldMatrix<K,m,n> operator/ ( const Dune::FieldMatrix<K, m, n> &A, const K& s)
{
    typedef typename Dune::FieldMatrix<K,m,n> :: size_type size_type;
    Dune::FieldMatrix<K,m,n> ret;

    for( size_type i = 0; i < m; ++i )
        for( size_type j = 0; j < n; ++j )
            ret[i][j] = A[i][j] / s;

    return ret;
}

//! calculates ret = A/s
template< class K, int m>
Dune::FieldVector<K,m> operator/ ( const Dune::FieldVector<K, m> &A, const K& s)
{
    typedef typename Dune::FieldVector<K,m> :: size_type size_type;
    Dune::FieldVector<K,m> result;

    for( size_type i = 0; i < m; ++i )
        result[i] = A[i] / s;

    return result;
}
#endif


///////////////////////////////////////////////////////////////////////////////////////////
//  Various other matrix methods
///////////////////////////////////////////////////////////////////////////////////////////

namespace Dune {

  namespace GFE {

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
  template <class T, int n>
  static FieldMatrix<T,n,n> transpose(const FieldMatrix<T,n,n>& A)
  {
    FieldMatrix<T,n,n> result;

    for (int i=0; i<n; i++)
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

  template <class T, int n>
  static auto dyadicProduct(const FieldVector<T,n>& A, const FieldVector<T,n>& B)
  {
    FieldMatrix<T,n,n> result;

    for (int i=0; i<n; i++)
      for (int j=0; j<n; j++)
        result[i][j] = A[i]*B[j];

    return result;
  }

#if ADOLC_ADOUBLE_H
  template <int n>
  static auto dyadicProduct(const FieldVector<adouble,n>& A, const FieldVector<double,n>& B)
    -> FieldMatrix<adouble,n,n>
  {
    FieldMatrix<adouble,n,n> result;

    for (int i=0; i<n; i++)
      for (int j=0; j<n; j++)
        result[i][j] = A[i]*B[j];

    return result;
  }
#endif

  }
}


#endif
