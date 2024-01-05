#ifndef DUNE_SKEW_MATRIX_HH
#define DUNE_SKEW_MATRIX_HH

/** \brief Static dense skew-symmetric matrix */
template <class T, int N>
class SkewMatrix
{};


/** \brief Static dense skew-symmetric 3x3 matrix */
template <class T>
class SkewMatrix<T,3>
{

public:

  /** \brief Default constructor -- does nothing */
  SkewMatrix()
  {}

  /** \brief Constructor from an axial vector */
  explicit SkewMatrix(const Dune::FieldVector<T,3>& v)
    : data_(v)
  {}

  /** \brief Constructor from a general matrix */
  explicit SkewMatrix(const Dune::FieldMatrix<T,3,3>& m)
  {
    data_ = {m[2][1], m[0][2], m[1][0]};
  }

  SkewMatrix(T v)
    : data_(v)
  {}

  SkewMatrix<T,3>& operator*=(const T& a)
  {
    data_ *= a;
    return *this;
  }

  Dune::FieldVector<T,3>& axial()
  {
    return data_;
  }

  const Dune::FieldVector<T,3>& axial() const
  {
    return data_;
  }

  typedef typename Dune::FieldVector<T,3>::Iterator Iterator;

  //! begin iterator
  Iterator begin ()
  {
    return Iterator(data_,0);
  }

  //! end iterator
  Iterator end ()
  {
    return Iterator(data_,3);
  }

  typedef typename Dune::FieldVector<T,3>::ConstIterator ConstIterator;
  //! begin iterator
  ConstIterator begin () const
  {
    return ConstIterator(data_,0);
  }

  //! end iterator
  ConstIterator end () const
  {
    return ConstIterator(data_,3);
  }

  /** \brief Embed the skew-symmetric matrix in R^3x3 */
  Dune::FieldMatrix<T,3,3> toMatrix() const
  {
    Dune::FieldMatrix<T,3,3> mat;
    mat = 0;

    mat[0][1] = -data_[2];
    mat[1][0] =  data_[2];
    mat[0][2] =  data_[1];
    mat[2][0] = -data_[1];
    mat[1][2] = -data_[0];
    mat[2][1] =  data_[0];

    return mat;
  }

private:
  // we store the axial vector
  Dune::FieldVector<T,3> data_;

};

#endif
