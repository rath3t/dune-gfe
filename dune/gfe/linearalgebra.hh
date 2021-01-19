#ifndef DUNE_GFE_LINEARALGEBRA_HH
#define DUNE_GFE_LINEARALGEBRA_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/version.hh>


///////////////////////////////////////////////////////////////////////////////////////////
//  Various matrix methods
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
