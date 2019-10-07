#ifndef LINEAR_ALGEBRA_HH
#define LINEAR_ALGEBRA_HH

#include <dune/common/fmatrix.hh>

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
#endif

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

#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
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

#endif
