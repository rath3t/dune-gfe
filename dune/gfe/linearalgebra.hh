#ifndef LINEAR_ALGEBRA_HH
#define LINEAR_ALGEBRA_HH

#include <dune/common/fmatrix.hh>

//! calculates ret = A * B
template< class K, int m, int n, int p >
Dune::FieldMatrix< K, m, p > operator* ( const Dune::FieldMatrix< K, m, n > &A, const Dune::FieldMatrix< K, n, p > &B)
{
    typedef typename Dune::FieldMatrix< K, m, p > :: size_type size_type;
    Dune::FieldMatrix< K, m, p > ret;

    for( size_type i = 0; i < m; ++i ) {

        for( size_type j = 0; j < p; ++j ) {
            ret[ i ][ j ] = K( 0 );
            for( size_type k = 0; k < n; ++k )
                ret[ i ][ j ] += A[ i ][ k ] * B[ k ][ j ];
        }
    }
    return ret;
}

//! calculates ret = A - B
template< class K, int m, int n>
Dune::FieldMatrix<K,m,n> operator- ( const Dune::FieldMatrix<K, m, n> &A, const Dune::FieldMatrix<K,m,n> &B)
{
    typedef typename Dune::FieldMatrix<K,m,n> :: size_type size_type;
    Dune::FieldMatrix<K,m,n> ret;

    for( size_type i = 0; i < m; ++i )
        for( size_type j = 0; j < n; ++j )
            ret[i][j] = A[i][j] - B[i][j];

    return ret;
}


//! calculates ret = A - B
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

#endif
