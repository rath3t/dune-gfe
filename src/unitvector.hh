#ifndef UNIT_VECTOR_HH
#define UNIT_VECTOR_HH

#include <dune/common/fvector.hh>

template <int dim>
class UnitVector
{
public:

    /** \brief The type used for coordinates */
    typedef double ctype;

    typedef Dune::FieldVector<double,dim> TangentVector;
    typedef Dune::FieldVector<double,dim> EmbeddedTangentVector;

    UnitVector<dim>& operator=(const Dune::FieldVector<double,dim>& vector)
    {
        data_ = vector;
        data_ /= data_.two_norm();
        return *this;
    }

     /** \brief The exponential map */
    static UnitVector exp(const UnitVector& p, const TangentVector& v) {

        assert( std::abs(p.data_*v) < 1e-7 );

        const double norm = v.two_norm();
        UnitVector result = p;
        result.data_ *= std::cos(norm);
        result.data_.axpy(std::sin(norm)/norm, v);
        return result;
    }

    /** \brief Length of the great arc connecting the two points */
     static double distance(const UnitVector& a, const UnitVector& b) {
        return std::acos(a.data_ * b.data_);
    }

    /** \brief Compute the gradient of the squared distance function keeping the first argument fixed

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static EmbeddedTangentVector derivativeOfDistanceSquaredWRTSecondArgument(const UnitVector& a, const UnitVector& b) {
        double x = a.data_ * b.data_;
        const double eps = 1e-12;

        EmbeddedTangentVector result = a.data_;

        if (x > 1-eps) {  // regular expression is unstable, use the series expansion instead
            result *= -2 + 2*(x-1)/3 - 4/15*(x-1)*(x-1) + 4/35*(x-1)*(x-1)*(x-1);
        } else {
            result *= -2*std::acos(x) / std::sqrt(1-x*x);
        }

        // Project gradient onto the tangent plane at b in order to obtain the surface gradient
        result = b.projectOntoTangentSpace(result);

        // Gradient must be a tangent vector at b, in other words, orthogonal to it
        assert( std::abs(b.data_ * result) < 1e-7);

        return result;
    }

    /** \brief Project tangent vector of R^n onto the tangent space */
    EmbeddedTangentVector projectOntoTangentSpace(const EmbeddedTangentVector& v) const {
        EmbeddedTangentVector result = v;
        result.axpy(-1*(data_*result), data_);
        return result;
    }

    /** \brief The global coordinates, if you really want them */
    const Dune::FieldVector<double,dim>& globalCoordinates() const {
        return data_;
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
