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
        data_.assign(v);
    }

    /** \brief Geodesic distance between two points 

    Simply the Euclidean distance */
    static double distance(const RealTuple& a, const RealTuple& b) {
        double result = 0;
        for (int i=0; i<N; i++)
            result += (a.data_[0] - b.data_[0]) * (a.data_[0] - b.data_[0]);
        return std::sqrt(result);
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
