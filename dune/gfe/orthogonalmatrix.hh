#ifndef ORTHOGONAL_MATRIX_HH
#define ORTHOGONAL_MATRIX_HH

#include <dune/common/fmatrix.hh>

/** \brief An orthogonal \f$ n \times n \f$ matrix
 * \tparam T Type of the matrix entries
 * \tparam N Space dimension
 */
template <class T, int N>
class OrthogonalMatrix
{
public:
    
    /** \brief Constructor from a general matrix
     * 
     * The input matrix gets normalized to det = 1.  Since computing
     * the determinant is expensive you may not want to use this method
     * if you know your input matrix to be orthogonal anyways.
     */
    explicit OrthogonalMatrix(const Dune::FieldMatrix<T,N,N>& matrix)
    : data_(matrix)
    {
        data_ /= matrix.determinant();
    }

    /** \brief Const access to the matrix entries */
    const Dune::FieldMatrix<T,N,N>& data() const
    {
        return data_;
    }
    
    /** \brief Project the input matrix onto the tangent space at "this"
     * 
     * See Absil, Mahoney, Sepulche, Example 5.3.2 for the formula
     * \f[ P_X \xi = (I - XX^T) \xi + X \operatorname{skew} (X^T \xi)  \f]
     */
    Dune::FieldMatrix<T,N,N> projectOntoTangentSpace(const Dune::FieldMatrix<T,N,N>& matrix) const
    {
        // rename to get the same notation as Absil & Co
        const Dune::FieldMatrix<T,N,N>& X = data_;
        
        // I - XX^T
        Dune::FieldMatrix<T,N,N> IdMinusXXT;
        
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++) {
                IdMinusXXT = (i==j);
                for (int k=0; k<N; k++)
                    IdMinusXXT[i][j] -= X[i][k] * X[j][k];
            }
            
        // (I - XX^T) \xi
        Dune::FieldMatrix<T,N,N> result(0);
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
                for (int k=0; k<N; k++)
                    result[i][j] += IdMinusXXT[i][k] * matrix[k][j];
                
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
        Dune::FieldMatrix<T,N,N> XskewXTxi(0);
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
                for (int k=0; k<N; k++)
                    XskewXTxi[i][j] += X[i][k] * skewXTxi[k][j];
                
        //
        result += XskewXTxi;
        
        return result;
    }
    
private:
    
    /** \brief The actual data */
    Dune::FieldMatrix<T,N,N> data_;
    
};

#endif // ORTHOGONAL_MATRIX_HH