#ifndef LOCAL_GEODESIC_FE_FUNCTION
#define LOCAL_GEODESIC_FE_FUNCTION

#include <vector>

#include <dune/common/fvector.hh>
#include <dune/common/geometrytype.hh>

#include <dune/gfe/averagedistanceassembler.hh>
#include <dune/gfe/targetspacertrsolver.hh>

#include <dune/gfe/svd.hh>

/** \brief A geodesic function from the reference element to a manifold 
    
\tparam dim Dimension of the reference element
\tparam ctype Type used for coordinates on the reference element
\tparam TargetSpace The manifold that the function takes its values in
*/
template <int dim, class ctype, class TargetSpace>
class LocalGeodesicFEFunction
{
    
    typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;
    static const int targetDim = EmbeddedTangentVector::size;

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

    static std::vector<ctype> barycentricCoordinates(const Dune::FieldVector<ctype,dim>& local) {
        std::vector<ctype> result(dim+1);
        result[0] = 1;
        for (int i=0; i<dim; i++) {
            result[0]  -= local[i];
            result[i+1] = local[i];
        }
        return result;
    }

    static Dune::FieldMatrix<double,targetDim,targetDim> pseudoInverse(const Dune::FieldMatrix<double,targetDim,targetDim>& A)
    {
        Dune::FieldMatrix<double,targetDim,targetDim> U = A;
        Dune::FieldVector<double,targetDim> W;
        Dune::FieldMatrix<double,targetDim,targetDim> V;

        svdcmp(U,W,V);

        // pseudoInv = V W^{-1} U^T
        Dune::FieldMatrix<double,targetDim,targetDim> UT;

        for (int i=0; i<targetDim; i++)
            for (int j=0; j<targetDim; j++)
                UT[i][j] = U[j][i];

        for (int i=0; i<targetDim; i++) {
            if (std::abs(W[i]) > 1e-12)  // Diagonal may be zero, that's why we're using the pseudo inverse
                UT[i] /= W[i];
            else
                UT[i] = 0;
        }

        Dune::FieldMatrix<double,targetDim,targetDim> pseudoInv;
        Dune::FMatrixHelp::multMatrix(V,UT,pseudoInv);
        
        return pseudoInv;
    }

    /** \brief The coefficient vector */
    std::vector<TargetSpace> coefficients_;

};

template <int dim, class ctype, class TargetSpace>
TargetSpace LocalGeodesicFEFunction<dim,ctype,TargetSpace>::
evaluate(const Dune::FieldVector<ctype, dim>& local)
{
    // First compute the coordinates on the standard simplex (in R^{n+1})
    std::vector<ctype> w = barycentricCoordinates(local);

    AverageDistanceAssembler<TargetSpace> assembler(coefficients_, w);

    TargetSpaceRiemannianTRSolver<TargetSpace> solver;

    solver.setup(&assembler,
                 coefficients_[0],   // initial iterate
                 1e-8,    // tolerance
                 20,      // maxTrustRegionSteps
                 2,       // initial trust region radius
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
    const int targetDim = EmbeddedTangentVector::size;

    Dune::FieldMatrix<ctype, targetDim, dim> result;

#if 0  // this is probably faster than the general implementation, but we leave it out for testing purposes
    if (dim==1) {

        EmbeddedTangentVector tmp = TargetSpace::interpolateDerivative(coefficients_[0], coefficients_[1], local[0]);

        for (int i=0; i<targetDim; i++)
            result[i][0] = tmp[i];

    }
#endif

    // ////////////////////////////////////////////////////////////////////////
    //  The derivative is evaluated using the implicit function theorem.
    //  Hence we need to solve a small system of linear equations.
    // ////////////////////////////////////////////////////////////////////////

    // the function value at the point where we are evaluating the derivative
    TargetSpace q = evaluate(local);

    // the matrix that turns coordinates on the reference simplex into coordinates on the standard simplex
    Dune::FieldMatrix<ctype,dim+1,dim> B;
    B[0] = -1;
    for (int i=0; i<dim; i++)
        for (int j=0; j<dim; j++)
            B[i+1][j] = (i==j);

    // compute negative derivate of F(w,q) (the derivative of the weighted distance fctl) wrt to w
    Dune::FieldMatrix<ctype,targetDim,dim+1> dFdw;
    for (int i=0; i<dim+1; i++) {
        Dune::FieldVector<ctype,targetDim> tmp = TargetSpace::derivativeOfDistanceSquaredWRTSecondArgument(coefficients_[i], q);
        for (int j=0; j<targetDim; j++)
            dFdw[j][i] = tmp[j];
    }

    dFdw *= -1;

    // multiply the two previous matrices: the result is the right hand side
    Dune::FieldMatrix<ctype,targetDim,dim> RHS;
    Dune::FMatrixHelp::multMatrix(dFdw,B, RHS);

    // the actual system matrix
    std::vector<ctype> w = barycentricCoordinates(local);
    AverageDistanceAssembler<TargetSpace> assembler(coefficients_, w);
    
    Dune::FieldMatrix<ctype,targetDim,targetDim> dFdq(0);
    assembler.assembleHessian(q,dFdq);

    // ////////////////////////////////////
    //   solve the system
    // ////////////////////////////////////

    // dFDq is not invertible, if the target space is embedded into a higher-dimensional
    // Euclidean space.  Therefore we use its pseudo inverse.  I don't think that is the
    // best way, though.
    Dune::FieldMatrix<ctype,targetDim,targetDim> dFdqPseudoInv = pseudoInverse(dFdq);

    for (int i=0; i<dim; i++) {

        Dune::FieldVector<ctype,targetDim> rhs, x;
        for (int j=0; j<targetDim; j++)
            rhs[j] = RHS[j][i];

        //dFdq.solve(x, rhs);
        dFdqPseudoInv.mv(rhs,x);

        for (int j=0; j<targetDim; j++)
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
