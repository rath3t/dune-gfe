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

    typedef Dune::FieldVector<double,N> TangentVector;

    /** \brief Default constructor */
    RealTuple()
    {}

    /** \brief Construction from a scalar */
    RealTuple(double v)
    {
        data_ = v;
    }

    /** \brief The exponention map */
    static RealTuple exp(const RealTuple& p, const TangentVector& v) {
        return RealTuple(p.data_+v);
    }

    /** \brief Geodesic distance between two points 

    Simply the Euclidean distance */
    static double distance(const RealTuple& a, const RealTuple& b) {
        return (a.data_ - b.data_).two_norm();
    }

    
    /** \brief Write LocalKey object to output stream */
    friend std::ostream& operator<< (std::ostream& s, const RealTuple& realTuple)
    {
        return s << realTuple.data_;
    }

private:
    
    Dune::FieldVector<double,N> data_;

};

#endif
