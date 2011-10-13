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
    
private:
    // we store the axial vector
    Dune::FieldVector<T,3> data_;
    
};

#endif