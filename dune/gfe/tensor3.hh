#ifndef DUNE_TENSOR_3_HH
#define DUNE_TENSOR_3_HH

/** \file
    \brief A third-rank tensor
    */
    
template <class T, int N1, int N2, int N3>
class Tensor3
    : public Dune::array<Dune::FieldMatrix<T,N2,N3>,N1>
{


    
};

#endif