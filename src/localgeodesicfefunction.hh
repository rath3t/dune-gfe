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

    static Dune::FieldVector<ctype,dim+1> barycentricCoordinates(const Dune::FieldVector<ctype,dim>& local) {
        Dune::FieldVector<ctype,dim+1> result;
        result[0] = 1;
        for (int i=0; i<dim; i++) {
            result[0]  -= local[i];
            result[i+1] = local[i];
        }
    }
        

    /** \brief The coefficient vector */
    std::vector<TargetSpace> coefficients_;

};

template <int dim, class ctype, class TargetSpace>
TargetSpace LocalGeodesicFEFunction<dim,ctype,TargetSpace>::
evaluate(const Dune::FieldVector<ctype, dim>& local)
{
    // First compute the coordinates on the standard simplex (in R^{n+1})
    std::vector<ctype> barycentricCoordinates(dim+1);

    barycentricCoordinates[0] = 1;
    for (int i=0; i<dim; i++) {
        barycentricCoordinates[0] -= local[i];
         barycentricCoordinates[i+1] = local[i];
    }

    AverageDistanceAssembler<TargetSpace> assembler(coefficients_, barycentricCoordinates);

    TargetSpaceRiemannianTRSolver<TargetSpace> solver;

    solver.setup(&assembler,
                 coefficients_[0],   // initial iterate
                 1e-8,    // tolerance
                 20,      // maxTrustRegionSteps
                 1,       // initial trust region radius
                 20,      // inner iterations
                 1e-8     // inner tolerance
                 );

    solver.solve();

    return solver.getSol();
}

template <int dim, class ctype, class TargetSpace>
Dune::FieldMatrix<ctype, TargetSpace::EmbeddedTangentVector::size, dim> LocalGeodesicFEFunction<dim,ctype,TargetSpace>::
evaluateDerivative(const Dune::FieldVector<ctype, dim>& local)
{
    Dune::FieldMatrix<ctype, EmbeddedTangentVector::size, dim> result;

#if 0  // this is probably faster than the general implementation, but we leave it out for testing purposes
    if (dim==1) {

        EmbeddedTangentVector tmp = TargetSpace::interpolateDerivative(coefficients_[0], coefficients_[1], local[0]);

        for (int i=0; i<EmbeddedTangentVector::size; i++)
            result[i][0] = tmp[i];

    }
#endif

    // ////////////////////////////////////////////////////////////////////////
    //  The derivative is evaluated using the implicit function theorem.
    //  Hence we need to solve a small system of linear equations.
    // ////////////////////////////////////////////////////////////////////////

    // the function value at the point where we are evaluation the derivative
    TargetSpace q = evaluate(local);

    // the matrix that turns coordinates on the reference simplex into coordinates on the standard simplex
    Dune::FieldMatrix<ctype,dim+1,dim> B;
    B[0] = -1;
    for (int i=0; i<dim; i++)
        for (int j=0; j<dim; j++)
            B[i+1][j] = (i==j);

    // compute negative derivate of F(w,q) (the derivative of the weighted distance fctl) wrt to w
    Dune::FieldMatrix<ctype,dim+1,dim+1> dFdw;
    for (int i=0; i<dim+1; i++)
        dFdw[i] = TargetSpace::derivativeOfDistanceSquaredWRTSecondArgument(coefficients_[i], q);

    dFdw *= -1;

    // multiply the two previous matrices: the result is the right hand side
    Dune::FieldMatrix<ctype,dim+1,dim> RHS;
    Dune::FMatrixHelp::multMatrix(dFdw,B, RHS);

    // the actual system matrix
    Dune::FieldVector<ctype,dim+1> w = barycentricCoordinates(local);

    Dune::FieldMatrix<ctype,dim+1,dim+1> dFdq(0);
    for (int i=0; i<dim+1; i++)
        dFdq.axpy(w[i], TargetSpace::secondDerivativeOfDistanceSquaredWRTSecondArgument(coefficients_[i], q));

    // ////////////////////////////////////
    //   solve the system
    // ////////////////////////////////////

    for (int i=0; i<dim; i++) {

        Dune::FieldVector<ctype,dim+1> rhs, x;
        for (int j=0; j<dim+1; j++)
            rhs[j] = RHS[j][i];

        dFdq.solve(x, rhs);

        for (int j=0; j<dim+1; j++)
            result[j][i] = x[j];

    }

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
        
        EmbeddedTangentVector fdDer = evaluate(forward).globalCoordinates() - evaluate(backward).globalCoordinates();
        fdDer /= 2*eps;
        
        for (int j=0; j<EmbeddedTangentVector::size; j++)
            result[j][i] = fdDer[j];
        
    }

    return result;
}

#endif
