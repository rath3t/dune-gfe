#ifndef DUNE_GFE_LINEARALGEBRA_HH
#define DUNE_GFE_LINEARALGEBRA_HH

#include <random>

#include <dune/common/fmatrix.hh>
#include <dune/common/version.hh>
#include <dune/istl/scaledidmatrix.hh>


///////////////////////////////////////////////////////////////////////////////////////////
//  Various matrix methods
///////////////////////////////////////////////////////////////////////////////////////////

namespace Dune {

  namespace GFE {

  /** \brief Multiplication of a ScalecIdentityMatrix with another FieldMatrix */
  template <class T, int N, int otherCols>
  Dune::FieldMatrix<T,N,otherCols> operator* ( const Dune::ScaledIdentityMatrix<T, N>& diagonalMatrix,
                        const Dune::FieldMatrix<T, N, otherCols>& matrix)
  {
      Dune::FieldMatrix<T,N,otherCols> result(0);

      for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < otherCols; ++j)
          result[i][j] = diagonalMatrix[i][i]*matrix[i][j];

      return result;
  }

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

    /** \brief Return a segment of a FieldVector from lower up to lower+size-1 */
    template< int lower, int size,typename field_type,int n>
    static FieldVector<field_type,size> segment(const FieldVector<field_type,n>& v)
    {
      FieldVector<field_type,size> res;
      std::copy(v.begin()+lower,v.begin()+lower+size,res.begin());
      return res;
    }

    /** \brief Return a segment of a FieldVector from lower up to lower+size-1
     * lower is unkown at compile time*/
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
         * * lower1 and lower2 is unkown at compile time*/
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
  }
}


#endif
