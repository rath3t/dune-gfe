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

    typedef double ctype;

    /** \brief Global coordinates wrt an isometric embedding function are available */
    static const bool globalIsometricCoordinates = true;

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

    /** \brief Copy constructor */
    RealTuple(const RealTuple<N>& other)
        : data_(other.data_)
    {}

    /** \brief Constructor from FieldVector*/
    RealTuple(const Dune::FieldVector<double,N>& other)
        : data_(other)
    {}

    RealTuple& operator=(const Dune::FieldVector<double,N>& other) {
        data_ = other;
        return *this;
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

    /** \brief Compute the gradient of the squared distance function keeping the first argument fixed

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static EmbeddedTangentVector derivativeOfDistanceSquaredWRTSecondArgument(const RealTuple& a, const RealTuple& b) {
        EmbeddedTangentVector result = a.data_;
        result -= b.data_;
        result *= -2;
        return result;
    }

        /** \brief Compute the Hessian of the squared distance function keeping the first argument fixed

    Unlike the distance itself the squared distance is differentiable at zero
        */
    static Dune::FieldMatrix<double,N,N> secondDerivativeOfDistanceSquaredWRTSecondArgument(const RealTuple& a, const RealTuple& b) {

        Dune::FieldMatrix<double,N,N> result;
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
                result[i][j] = 2*(i==j);

        return result;
    }

    /** \brief Project tangent vector of R^n onto the tangent space */
    EmbeddedTangentVector projectOntoTangentSpace(const EmbeddedTangentVector& v) const {
        return v;
    }

    /** \brief The global coordinates, if you really want them */
    const Dune::FieldVector<double,N>& globalCoordinates() const {
        return data_;
    }

    /** \brief Compute an orthonormal basis of the tangent space of S^n.

    This basis is of course not globally continuous.
    */
    Dune::FieldMatrix<double,N,N> orthonormalFrame() const {

        Dune::FieldMatrix<double,N,N> result;
        
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
                result[i][j] = (i==j);
        return result;
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
