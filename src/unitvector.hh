#ifndef UNIT_VECTOR_HH
#define UNIT_VECTOR_HH

#include <dune/common/fvector.hh>

template <int dim>
class UnitVector
{
    /** \brief Computes sin(x/2) / x without getting unstable for small x */
    static double sinc(const double& x) {
        return (x < 1e-4) ? 1 + (x*x/6) : std::sin(x)/x;
    }

    /** \brief Compute the derivative of arccos^2 without getting unstable for x close to 1 */
    static double derivativeOfArcCosSquared(const double& x) {
        const double eps = 1e-12;
        if (x > 1-eps) {  // regular expression is unstable, use the series expansion instead
            return -2 + 2*(x-1)/3 - 4/15*(x-1)*(x-1) + 4/35*(x-1)*(x-1)*(x-1);
        } else if (x < -1+eps) {  // The function is not differentiable
            DUNE_THROW(Dune::Exception, "Distance is not differentiable for conjugate points!");
        } else
            return -2*std::acos(x) / std::sqrt(1-x*x);
    }

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

        assert( std::abs(p.data_*v) < 1e-5 );

        const double norm = v.two_norm();
        UnitVector result = p;
        result.data_ *= std::cos(norm);
        result.data_.axpy(sinc(norm), v);
        return result;
    }

    /** \brief Length of the great arc connecting the two points */
     static double distance(const UnitVector& a, const UnitVector& b) {

         // Not nice: we are in a class for unit vectors, but the class is actually
         // supposed to handle perturbations of unit vectors as well.  Therefore
         // we normalize here.
         double x = a.data_ * b.data_/a.data_.two_norm()/b.data_.two_norm();
         
         // paranoia:  if the argument is just eps larger than 1 acos returns NaN
         x = std::min(x,1.0);
         
         return std::acos(x);
    }

    /** \brief Compute the gradient of the squared distance function keeping the first argument fixed

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static EmbeddedTangentVector derivativeOfDistanceSquaredWRTSecondArgument(const UnitVector& a, const UnitVector& b) {
        double x = a.data_ * b.data_;

        EmbeddedTangentVector result = a.data_;

        result *= derivativeOfArcCosSquared(x);

        // Project gradient onto the tangent plane at b in order to obtain the surface gradient
        result = b.projectOntoTangentSpace(result);

        // Gradient must be a tangent vector at b, in other words, orthogonal to it
        assert( std::abs(b.data_ * result) < 1e-5);

        return result;
    }

    /** \brief Compute the Hessian of the squared distance function keeping the first argument fixed

    Unlike the distance itself the squared distance is differentiable at zero
     */
    static Dune::FieldMatrix<double,dim,dim> secondDerivativeOfDistanceSquaredWRTSecondArgument(const UnitVector& a, const UnitVector& b) {

        Dune::FieldMatrix<double,dim,dim> result;

        double sp = a.data_ * b.data_;

        // Compute vector A (see notes)
        Dune::FieldMatrix<double,1,dim> row;
        row[0] = a.globalCoordinates();
        row *= -1 / (1-sp*sp) * (1 + std::acos(sp)/std::sqrt(1-sp*sp) * sp);

        Dune::FieldMatrix<double,dim,1> column;
        for (int i=0; i<dim; i++)
            column[i] = a.globalCoordinates()[i] - b.globalCoordinates()[i]*sp;

        Dune::FieldMatrix<double,dim,dim> A;
        // A = row * column
        Dune::FMatrixHelp::multMatrix(column,row,A);

        // Compute matrix B (see notes)
        Dune::FieldMatrix<double,dim,dim> B;
        for (int i=0; i<dim; i++)
            for (int j=0; j<dim; j++)
                B[i][j] = (i==j)*sp + a.data_[j]*b.data_[i];

        // Bring it all together
        result = A;
        result *= -2;
        result.axpy(derivativeOfArcCosSquared(sp), B);

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


    //private:

    Dune::FieldVector<double,dim> data_;
};

#endif
