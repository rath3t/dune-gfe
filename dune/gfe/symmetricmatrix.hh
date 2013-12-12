#ifndef DUNE_GFE_SYMMETRICMATRIX_HH
#define DUNE_GFE_SYMMETRICMATRIX_HH

#include <dune/common/fvector.hh>

namespace Dune {

/** \brief  A class implementing a symmetric matrix with compile-time size
 *
 *  A \f$ dim\times dim \f$ matrix is stored internally as a <tt>Dune::FieldVector<double, dim*(dim+1)/2></tt>
 *  The components are assumed to be \f$ [ E(1,1),\  E(2,1),\ E(2,2),\ E(3,1),\ E(3,2),\ E(3,3) ]\f$
 *  and analogous for other dimensions
 *  \tparam  dim number of lines/columns of the matrix
 */
template <class T, int N>
class SymmetricMatrix
{
public:

  /** \brief The type used for scalars
   */
  typedef T field_type;

  enum {blocklevel = 0};

    /** \brief Default constructor
     *
     *  Tensor is initialized containing zeros if no argument is given.
     *  \param eye if true tensor is initialized as identity
     */
    SymmetricMatrix()
    {}

    SymmetricMatrix<T,N>& operator=(const T& s)
    {
      data_ = s;
      return *this;
    }

    /** \brief Matrix style random read/write access to components 
     *  \param i line index
     *  \param j column index
     * \note You need to know what you are doing:  You can only access the lower
     * left triangular submatrix using this methods.  It requires i<=j.
     */
    T& operator() (int i, int j)
    {
      assert(i>=j);
      return data_[i*(i+1)/2+j];
    }

    /** \brief Matrix style random read access to components
     *  \param i line index
     *  \param j column index
     * \note You need to know what you are doing:  You can only access the lower
     * left triangular submatrix using this methods.  It requires i<=j.
     */
    const T& operator() (int i, int j) const
    {
      assert(i>=j);
      return data_[i*(i+1)/2+j];
    }

    T energyScalarProduct (const FieldVector<T,N>& v1, const FieldVector<T,N>& v2) const
    {
      T result = 0;
      for (size_t i=0; i<N; i++)
        for (size_t j=0; j<=i; j++)
          result += (1+(i!=j)) * operator()(i,j) * v1[i] * v2[j];

      return result;
    }


private:
    Dune::FieldVector<T,N*(N+1)/2> data_;
};

}
#endif
