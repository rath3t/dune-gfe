#ifndef UNIT_VECTOR_HH
#define UNIT_VECTOR_HH

#include <dune/common/fvector.hh>

template <int dim>
class UnitVector
{
public:

    typedef Dune::FieldVector<double,dim> TangentVector;
    typedef Dune::FieldVector<double,dim> EmbeddedTangentVector;

    UnitVector<dim>& operator=(const Dune::FieldVector<double,dim>& vector)
    {
        data_ = vector;
        data_ /= data_.two_norm();
        return *this;
    }

    /** \brief Write LocalKey object to output stream */
    friend std::ostream& operator<< (std::ostream& s, const UnitVector& unitVector)
    {
        return s << unitVector.data_;
    }


private:

    Dune::FieldVector<double,dim> data_;
};

#endif
