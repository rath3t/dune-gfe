#ifndef ORTHOGONAL_MATRIX_HH
#define ORTHOGONAL_MATRIX_HH

#include <dune/common/fmatrix.hh>

#include <dune/gfe/skewmatrix.hh>

/** \brief An orthogonal \f$ n \times n \f$ matrix
 * \tparam T Type of the matrix entries
 * \tparam N Space dimension
 */
template <class T, int N>
class OrthogonalMatrix
{
public:
  /** \brief The type used for coordinates */
  typedef T ctype;

  /** \brief Dimension of the manifold formed by the orthogonal matrices */
  static const int dim = N*(N-1)/2;

  /** \brief Coordinates are embedded into a Euclidean space of N x N - matrices */
  static const int embeddedDim = N*N;

  /** \brief The type used for global coordinates */
  typedef Dune::FieldVector<T,embeddedDim> CoordinateType;

  /** \brief Member of the corresponding Lie algebra.  This really is a skew-symmetric matrix */
  typedef Dune::FieldVector<T,dim> TangentVector;

  /** \brief A tangent vector as a vector in the surrounding coordinate space */
  typedef Dune::FieldVector<T,embeddedDim> EmbeddedTangentVector;

    /** \brief Default constructor -- leaves the matrix uninitialized */
    OrthogonalMatrix()
    {}
    
    /** \brief Constructor from a general matrix
     * 
     * It is not checked whether the matrix is really orthogonal
     */
    explicit OrthogonalMatrix(const Dune::FieldMatrix<T,N,N>& matrix)
    : data_(matrix)
    {}

    /** \brief Copy constructor
     *
     * It is not checked whether the matrix is really orthogonal
     */
    explicit OrthogonalMatrix(const CoordinateType& other)
    {
      DUNE_THROW(Dune::NotImplemented, "constructor");
    }

    /** \brief Const access to the matrix entries */
    const Dune::FieldMatrix<T,N,N>& data() const
    {
        return data_;
    }
    
    /** \brief Project the input matrix onto the tangent space at "this"
     * 
     * See Absil, Mahoney, Sepulche, Example 5.3.2 for the formula
     * \f[ P_X \xi = X \operatorname{skew} (X^T \xi)  \f]
     *
     */
    Dune::FieldMatrix<T,N,N> projectOntoTangentSpace(const Dune::FieldMatrix<T,N,N>& matrix) const
    {
        // rename to get the same notation as Absil & Co
        const Dune::FieldMatrix<T,N,N>& X = data_;
        
        // X^T \xi
        Dune::FieldMatrix<T,N,N> XTxi(0);
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
                for (int k=0; k<N; k++)
                    XTxi[i][j] += X[k][i] * matrix[k][j];
                
        // skew (X^T \xi)
        Dune::FieldMatrix<T,N,N> skewXTxi(0);
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
                skewXTxi[i][j] = 0.5 * ( XTxi[i][j] - XTxi[j][i] );
                
        // X skew (X^T \xi)
        Dune::FieldMatrix<T,N,N> result(0);
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
                for (int k=0; k<N; k++)
                    result[i][j] += X[i][k] * skewXTxi[k][j];
                
        return result;
    }

    /** \brief Rebind the OrthogonalMatrix to another coordinate type */
    template<class U>
    struct rebind
    {
      typedef OrthogonalMatrix<U,N> other;
    };

    OrthogonalMatrix& operator= (const CoordinateType& other)
    {
      std::size_t idx = 0;
      for (int i=0; i<N; i++)
        for (int j=0; j<N; j++)
          data_[i][j] = other[idx++];
      return *this;
    }

  /** \brief The exponential map from a given point $p \in SO(n)$.

    \param v A tangent vector.
   */
  static OrthogonalMatrix<T,3> exp(const OrthogonalMatrix<T,N>& p, const TangentVector& v)
  {
    DUNE_THROW(Dune::NotImplemented, "exp");
  }

    /** \brief Return the actual matrix */
    void matrix(Dune::FieldMatrix<T,N,N>& m) const
  {
      m = data_;
  }

  /** \brief Project tangent vector of R^n onto the normal space space */
  EmbeddedTangentVector projectOntoNormalSpace(const EmbeddedTangentVector& v) const
  {
    DUNE_THROW(Dune::NotImplemented, "projectOntoNormalSpace");
  }

  /** \brief The Weingarten map */
  EmbeddedTangentVector weingarten(const EmbeddedTangentVector& z, const EmbeddedTangentVector& v) const
  {
    EmbeddedTangentVector result;
    DUNE_THROW(Dune::NotImplemented, "weingarten");
    return result;
  }

  static OrthogonalMatrix<T,N> projectOnto(const CoordinateType& p)
  {
    DUNE_THROW(Dune::NotImplemented, "projectOnto");
  }

  // TODO return type is wrong
  static Dune::FieldMatrix<T,1,1> derivativeOfProjection(const CoordinateType& p)
  {
    Dune::FieldMatrix<T,1,1> result;
    return result;
  }

  /** \brief The global coordinates, if you really want them */
  CoordinateType globalCoordinates() const
  {
    CoordinateType result;
    std::size_t idx = 0;

    for (int i=0; i<N; i++)
      for (int j=0; j<N; j++)
        result[idx++] = data_[i][j];

    return result;
  }

  /** \brief Compute an orthonormal basis of the tangent space of SO(n). */
  Dune::FieldMatrix<T,dim,embeddedDim> orthonormalFrame() const
  {
    Dune::FieldMatrix<T,dim,embeddedDim> result;
    DUNE_THROW(Dune::NotImplemented, "orthonormalFrame");
    return result;
  }

private:
    
    /** \brief The actual data */
    Dune::FieldMatrix<T,N,N> data_;
    
};

#endif // ORTHOGONAL_MATRIX_HH
