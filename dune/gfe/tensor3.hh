#ifndef DUNE_TENSOR_3_HH
#define DUNE_TENSOR_3_HH

/** \file
    \brief A third-rank tensor
    */

#include <dune/common/array.hh>
#include <dune/common/fmatrix.hh>
    
template <class T, int N1, int N2, int N3>
class Tensor3
    : public Dune::array<Dune::FieldMatrix<T,N2,N3>,N1>
{
    public:

        T infinity_norm() const
        {
            dune_static_assert(N1>0, "infinity_norm not implemented for empty tensors");
            T norm = (*this)[0].infinity_norm();
            for (int i=1; i<N1; i++)
                norm = std::max(norm, (*this)[i].infinity_norm());
            return norm;
        }
        
        static Tensor3<T,N1,N2,N3> product(const Dune::FieldVector<T,N1>& a, const Dune::FieldVector<T,N2>& b, const Dune::FieldVector<T,N3>& c)
        {
            Tensor3<T,N1,N2,N3> result;
            
            for (int i=0; i<N1; i++)
                for (int j=0; j<N2; j++)
                    for (int k=0; k<N3; k++)
                        result[i][j][k] = a[i]*b[j]*c[k];
                    
            return result;
        }
    
        static Tensor3<T,N1,N2,N3> product(const Dune::FieldMatrix<T,N1,N2>& ab, const Dune::FieldVector<T,N3>& c)
        {
            Tensor3<T,N1,N2,N3> result;
            
            for (int i=0; i<N1; i++)
                for (int j=0; j<N2; j++)
                    for (int k=0; k<N3; k++)
                        result[i][j][k] = ab[i][j]*c[k];
                    
            return result;
        }
    
        static Tensor3<T,N1,N2,N3> product(const Dune::FieldVector<T,N3>& a, const Dune::FieldMatrix<T,N1,N2>& bc)
        {
            Tensor3<T,N1,N2,N3> result;
            
            for (int i=0; i<N1; i++)
                for (int j=0; j<N2; j++)
                    for (int k=0; k<N3; k++)
                        result[i][j][k] = a[i]*bc[j][k];
                    
            return result;
        }
    
    friend Tensor3<T,N1,N2,N3> operator+(const Tensor3<T,N1,N2,N3>& a, const Tensor3<T,N1,N2,N3>& b)
    {
            Tensor3<T,N1,N2,N3> result;
            
            for (int i=0; i<N1; i++)
                for (int j=0; j<N2; j++)
                    for (int k=0; k<N3; k++)
                        result[i][j][k] = a[i][j][k] + b[i][j][k];
                    
            return result;
    }

    friend Tensor3<T,N1,N2,N3> operator-(const Tensor3<T,N1,N2,N3>& a, const Tensor3<T,N1,N2,N3>& b)
    {
            Tensor3<T,N1,N2,N3> result;
            
            for (int i=0; i<N1; i++)
                for (int j=0; j<N2; j++)
                    for (int k=0; k<N3; k++)
                        result[i][j][k] = a[i][j][k] - b[i][j][k];
                    
            return result;
    }

    friend Tensor3<T,N1,N2,N3> operator*(const T& scalar, const Tensor3<T,N1,N2,N3>& tensor)
    {
        Tensor3<T,N1,N2,N3> result;
            
        for (int i=0; i<N1; i++)
            for (int j=0; j<N2; j++)
                for (int k=0; k<N3; k++)
                    result[i][j][k] = scalar * tensor[i][j][k];
                    
        return result;
        
    }

};

//! Output operator for array
template <class T, int N1, int N2, int N3>
inline std::ostream& operator<< (std::ostream& s, const Tensor3<T,N1,N2,N3>& tensor)
{
    for (int i=0; i<N1; i++) 
        s << tensor[i];
    return s;
}



#endif