#ifndef LOCAL_GEODESIC_FE_FUNCTION
#define LOCAL_GEODESIC_FE_FUNCTION

#include <vector>

#include <dune/common/fvector.hh>
#include <dune/common/geometrytype.hh>

#include <dune/src/averagedistanceassembler.hh>
#include <dune/src/targetspacertrsolver.hh>

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
    LocalGeodesicFEFunction(const std::vector<TargetSpace>& coefficients)
        : coefficients_(coefficients)
    {}

    /** \brief Evaluate the function */
    TargetSpace evaluate(const Dune::FieldVector<ctype, dim>& local);

    /** \brief Evaluate the derivative of the function */
    Dune::FieldMatrix<ctype, EmbeddedTangentVector::size, dim> evaluateDerivative(const Dune::FieldVector<ctype, dim>& local);

    /** \brief Evaluate the derivative of the function using a finite-difference approximation*/
    Dune::FieldMatrix<ctype, EmbeddedTangentVector::size, dim> evaluateDerivativeFD(const Dune::FieldVector<ctype, dim>& local);

private:

    /** \brief The coefficient vector */
    std::vector<TargetSpace> coefficients_;

};

template <int dim, class ctype, class TargetSpace>
TargetSpace LocalGeodesicFEFunction<dim,ctype,TargetSpace>::
evaluate(const Dune::FieldVector<ctype, dim>& local)
{
#if 0   // Interpolation using recursive geodesic cones, doesn't work :-(
    ctype extraCoord = 1;
    for (int i=0; i<dim; i++)
        extraCoord -= local[i];

    ctype normalizingFactor = extraCoord;

    TargetSpace result = coefficients_[0];

    for (int i=0; i<dim; i++) {
        assert(local[i]>=0);
        normalizingFactor += local[i];
        result = TargetSpace::interpolate(result, coefficients_[i+1], 
                                          (std::abs(normalizingFactor) > 1e-10)
                                          ? local[i] / normalizingFactor
                                          : 0);
    }

    return result;
#endif

#warning Some code out-commented
#if 0
    Dune::FieldVector<ctype, dim+1> barycentricCoordinates;

    barycentricCoordinates[0] = 1;
    for (int i=0; i<dim; i++) {
        barycentricCoordinates[0] -= local[i];
         barycentricCoordinates[i+1] = local[i];
    }

    AverageDistanceAssembler<TargetSpace> assembler(coefficients_, barycentricCoordinates);

    TargetSpaceRiemannianTRSolver<TargetSpace> solver;

    solver.setup(&assembler,
                 coefficients_[0],   // initial iterate
                 20,      // maxTrustRegionSteps
                 1,       // initial trust region radius
                 20,      // inner iterations
                 1e-8     // inner tolerance
                 );

    solver.solve();

    return solver.getSol();
#endif
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

        DUNE_THROW(Dune::NotImplemented, "evaluateDerivative");

    }

    assert(dim==1 || dim==2);

    return result;
}

template <int dim, class ctype, class TargetSpace>
Dune::FieldMatrix<ctype, TargetSpace::EmbeddedTangentVector::size, dim> LocalGeodesicFEFunction<dim,ctype,TargetSpace>::
evaluateDerivativeFD(const Dune::FieldVector<ctype, dim>& local)
{
    double eps = 1e-6;

    Dune::FieldMatrix<ctype, EmbeddedTangentVector::size, dim> result;

    for (int i=0; i<dim; i++) {
        
        Dune::FieldVector<ctype, dim> forward  = local;
        Dune::FieldVector<ctype, dim> backward = local;
        
        forward[i]  += eps;
        backward[i] -= eps;
        
        EmbeddedTangentVector fdDer = evaluate(forward) - evaluate(backward);
        fdDer /= 2*eps;
        
        for (int j=0; j<EmbeddedTangentVector::size; j++)
            result[j][i] = fdDer[j];
        
    }

    return result;
}

#endif
