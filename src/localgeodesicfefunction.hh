#ifndef LOCAL_GEODESIC_FE_FUNCTION
#define LOCAL_GEODESIC_FE_FUNCTION

#include <vector>

#include <dune/common/fvector.hh>
#include <dune/common/geometrytype.hh>


/** \brief A geodesic function from the reference element to a manifold 
    
\tparam dim Dimension of the reference element
\tparam ctype Type used for coordinates on the reference element
\tparam TargetSpace The manifold that the function takes its values in
*/
template <int dim, class ctype, class TargetSpace>
class LocalGeodesicFEFunction
{
    
    typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;

public:

    /** \brief Constructor */
    LocalGeodesicFEFunction(const Dune::GeometryType& type,
                            const std::vector<TargetSpace>& coefficients)
        : type_(type), coefficients_(coefficients)
    {
        // currently only simplices are implemented
        assert(type.isSimplex());
    }

    /** \brief Evaluate the function */
    TargetSpace evaluate(const Dune::FieldVector<ctype, dim>& local);

    /** \brief Evaluate the derivative of the function */
    Dune::FieldMatrix<ctype, EmbeddedTangentVector::size, dim> evaluateDerivative(const Dune::FieldVector<ctype, dim>& local);

private:

    /** \brief Compute the derivate of geodesic interpolation wrt to the initial point
        and return it as an _embedded_ tangent vector

        \todo This is group-specific and should not really be here
    */
    static Dune::FieldMatrix<ctype, EmbeddedTangentVector::size, EmbeddedTangentVector::size>
    derivativeWRTFirstPoint(const Rotation<3,ctype>& a, const Rotation<3,ctype>& b, double s);

    /** \brief The type of the reference element */
    Dune::GeometryType type_;

    /** \brief The coefficient vector */
    std::vector<TargetSpace> coefficients_;

};

template <int dim, class ctype, class TargetSpace>
Dune::FieldMatrix<ctype, TargetSpace::EmbeddedTangentVector::size, TargetSpace::EmbeddedTangentVector::size>
LocalGeodesicFEFunction<dim,ctype,TargetSpace>::
derivativeWRTFirstPoint(const Rotation<3,ctype>& a, const Rotation<3,ctype>& b, double s)
{
    DUNE_THROW(Dune::NotImplemented, "derivativeWRTFirstPoint");
}


template <int dim, class ctype, class TargetSpace>
TargetSpace LocalGeodesicFEFunction<dim,ctype,TargetSpace>::
evaluate(const Dune::FieldVector<ctype, dim>& local)
{
    TargetSpace result = TargetSpace::interpolate(coefficients_[0], coefficients_[1], local[0]);

    for (int i=1; i<dim; i++)
        result = TargetSpace::interpolate(result, coefficients_[i+1], local[i]);

    return result;
}

template <int dim, class ctype, class TargetSpace>
Dune::FieldMatrix<ctype, TargetSpace::EmbeddedTangentVector::size, dim> LocalGeodesicFEFunction<dim,ctype,TargetSpace>::
evaluateDerivative(const Dune::FieldVector<ctype, dim>& local)
{
    Dune::FieldMatrix<ctype, EmbeddedTangentVector::size, dim> result;

    if (dim==1) {

        EmbeddedTangentVector tmp = TargetSpace::interpolateDerivative(coefficients_[0], coefficients_[1], local[0]);

        for (int i=0; i<EmbeddedTangentVector::size; i++)
            result[i][0] = tmp[i];

    }

    if (dim==2) {

        TargetSpace firstPoint = TargetSpace::interpolate(coefficients_[0], coefficients_[1], local[0]);

        // Derivative with respect to \xi_0

        // -- Outer derivative
        Dune::FieldMatrix<ctype, EmbeddedTangentVector::size, EmbeddedTangentVector::size> outerDerivative
            = derivativeWRTFirstPoint(firstPoint, coefficients_[2], local[1]);

        // -- Inner derivative
        EmbeddedTangentVector innerDerivative = TargetSpace::interpolateDerivative(coefficients_[0], coefficients_[1], local[0]);

        EmbeddedTangentVector der;
        outerDerivative.mv(innerDerivative, der);
        
        for (int i=0; i<EmbeddedTangentVector::size; i++)
            result[i][0] = der[i];


        // Derivative with respect to \xi_1
        EmbeddedTangentVector tmp = TargetSpace::interpolateDerivative(firstPoint, coefficients_[2], local[1]);
        
        for (int i=0; i<EmbeddedTangentVector::size; i++)
            result[i][1] = tmp[i];

    }

    assert(dim==1 || dim==2);

}

#endif
