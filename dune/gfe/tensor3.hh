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

#endif