#ifndef REAL_TUPLE_HH
#define REAL_TUPLE_HH

#include <dune/common/array.hh>
#include <dune/common/fvector.hh>


/** \brief Implement a tuple of real numbers as a Riemannian manifold

Currently this class only exists for testing purposes.
*/

template <int N>
class RealTuple
{
public:

    typedef Dune::FieldVector<double,N> EmbeddedTangentVector;

    /** \brief Default constructor */
    RealTuple()
    {}

    /** \brief Construction from a scalar */
    RealTuple(double v)
    {
        data_.assign(v);
    }
    
    /** \brief Write LocalKey object to output stream */
    friend std::ostream& operator<< (std::ostream& s, const RealTuple& realTuple)
    {
        return s << realTuple.data_;
    }

private:
    
    Dune::array<double,N> data_;

};

#endif
