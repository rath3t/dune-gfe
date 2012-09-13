#ifndef DUNE_TENSOR_SSD_HH
#define DUNE_TENSOR_SSD_HH

/** \file
    \brief A third-rank tensor with two static (SS) and one dynamic (D) dimension
    */

#include <dune/common/array.hh>
#include <dune/common/fmatrix.hh>
    
/** \brief A third-rank tensor with two static (SS) and one dynamic (D) dimension
 * 
 * \tparam T Type of the entries
 * \tparam N1 Size of the first dimension
 * \tparam N2 Size of the second dimension
*/
template <class T, int N1, int N2>
class TensorSSD
{
public:

    /** \brief Constructor with the third dimension */
    explicit TensorSSD(size_t N3)
    : N3_(N3)
    {
        for (int i=0; i<N1; i++)
            for (int j=0; j<N2; j++)
                data_[i][j].resize(N3_);
    }
        
    size_t dim(int index) const
    {
        switch (index) {
            case 0:
                return N1;
            case 1:
                return N2;
            case 2:
                return N3_;
            default:
                assert(false);
        }
        // Make compiler happy even if NDEBUG is set
        return 0;
    }

    /** \brief Direct access to individual entries */
    T& operator()(size_t i, size_t j, size_t k)
    {
        assert(i<N1 && j<N2 && k<N3_);
        return data_[i][j][k];
    }
        
    /** \brief Direct const access to individual entries */
    const T& operator()(size_t i, size_t j, size_t k) const
    {
        assert(i<N1 && j<N2 && k<N3_);
        return data_[i][j][k];
    }
        
    /** \brief Assignment from scalar */
    TensorSSD<T,N1,N2>& operator=(const T& scalar)
    {
        for (int i=0; i<N1; i++)
            for (int j=0; j<N2; j++)
                for (size_t k=0; k<dim(2); k++)
                    data_[i][j][k] = scalar;
                    
        return *this;
    }

    friend TensorSSD<T,N1,N2> operator*(const TensorSSD<T,N1,N2>& a, const Dune::Matrix<Dune::FieldMatrix<T,1,1> >& b)
    {
        TensorSSD<T,N1,N2> result(b.M());
            
        assert(a.dim(2)==b.N());
        size_t N4 = a.dim(2);  // third dimension of a
            
        for (int i=0; i<N1; i++)
            for (int j=0; j<N2; j++)
                for (size_t k=0; k<b.M(); k++) {
                    result.data_[i][j][k] = 0;
                    for (size_t l=0; l<N4; l++)
                        result.data_[i][j][k] += a.data_[i][j][l]*b[l][k][0][0];
                }
                    
        return result;
    }

    friend TensorSSD<T,N1,N2> operator+(const TensorSSD<T,N1,N2>& a, const TensorSSD<T,N1,N2>& b)
    {
        assert(a.dim(2)==b.dim(2));
        size_t N3 = a.dim(2);
        TensorSSD<T,N1,N2> result(N3);
            
        for (int i=0; i<N1; i++)
            for (int j=0; j<N2; j++)
                for (size_t k=0; k<N3; k++)
                    result.data_[i][j][k] = a.data_[i][j][k] + b.data_[i][j][k];
                    
        return result;
    }
    
private:

    // having the dynamic data type on the inside is kind of a stupid data layout
    Dune::array<Dune::array<std::vector<T>, N2>, N1> data_;
    
    // size of the third dimension
    size_t N3_;
};

//! Output operator for TensorSSD
template <class T, int N1, int N2>
inline std::ostream& operator<< (std::ostream& s, const TensorSSD<T,N1,N2>& tensor)
{
    for (int i=0; i<N1; i++) {
        for (int j=0; j<N2; j++) {
            for (size_t k=0; k<tensor.dim(2); k++)
                s << tensor(i,j,k) << "  ";
            s << std::endl;
        }
        s << std::endl;
    }
    return s;
}


#endif