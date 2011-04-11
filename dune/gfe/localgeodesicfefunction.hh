#ifndef LOCAL_GEODESIC_FE_FUNCTION
#define LOCAL_GEODESIC_FE_FUNCTION

#include <vector>

#include <dune/common/fvector.hh>
#include <dune/common/geometrytype.hh>

#include <dune/gfe/averagedistanceassembler.hh>
#include <dune/gfe/targetspacertrsolver.hh>

#include <dune/gfe/svd.hh>
#include <dune/gfe/tensor3.hh>

//! calculates ret = A * B
template< class K, int m, int n, int p >
Dune::FieldMatrix< K, m, p > operator* ( const Dune::FieldMatrix< K, m, n > &A, const Dune::FieldMatrix< K, n, p > &B)
{
    typedef typename Dune::FieldMatrix< K, m, p > :: size_type size_type;
    Dune::FieldMatrix< K, m, p > ret;
        
    for( size_type i = 0; i < m; ++i ) {
        
        for( size_type j = 0; j < p; ++j ) {
            ret[ i ][ j ] = K( 0 );
            for( size_type k = 0; k < n; ++k )
                ret[ i ][ j ] += A[ i ][ k ] * B[ k ][ j ];
        }
    }
    return ret;
}

//! calculates ret = A - B
template< class K, int m, int n>
Dune::FieldMatrix<K,m,n> operator- ( const Dune::FieldMatrix<K, m, n> &A, const Dune::FieldMatrix<K,m,n> &B)
{
    typedef typename Dune::FieldMatrix<K,m,n> :: size_type size_type;
    Dune::FieldMatrix<K,m,n> ret;
        
    for( size_type i = 0; i < m; ++i )
        for( size_type j = 0; j < n; ++j )
            ret[i][j] = A[i][j] - B[i][j];

    return ret;
}

#if 0
template< class K, int m, int n>
void transpose(Dune::FieldMatrix<K, m, n> &A)
{
   for( size_type i = 0; i < m; ++i )
        for( size_type j = 0; j < i; ++j )
			std::swap(A[i][j], A[j][i]);
}
#endif


/** \brief A function defined by simplicial geodesic interpolation 
           from the reference element to a Riemannian manifold.
    
\tparam dim Dimension of the reference element
\tparam ctype Type used for coordinates on the reference element
\tparam TargetSpace The manifold that the function takes its values in
*/
template <int dim, class ctype, class TargetSpace>
class LocalGeodesicFEFunction
{
    
    typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;
    static const int embeddedDim = EmbeddedTangentVector::size;

public:

    /** \brief Constructor */
    LocalGeodesicFEFunction(const std::vector<TargetSpace>& coefficients)
        : coefficients_(coefficients)
    {
        assert(coefficients_.size() == dim+1);
    }

    /** \brief Evaluate the function */
    TargetSpace evaluate(const Dune::FieldVector<ctype, dim>& local) const;

    /** \brief Evaluate the derivative of the function */
    Dune::FieldMatrix<ctype, EmbeddedTangentVector::size, dim> evaluateDerivative(const Dune::FieldVector<ctype, dim>& local) const;

    /** \brief For debugging: Evaluate the derivative of the function using a finite-difference approximation*/
    Dune::FieldMatrix<ctype, EmbeddedTangentVector::size, dim> evaluateDerivativeFD(const Dune::FieldVector<ctype, dim>& local) const;
    
    /** \brief Evaluate the derivative of the gradient of the function with respect to a coefficient */
    void evaluateDerivativeOfGradientWRTCoefficient(const Dune::FieldVector<ctype, dim>& local,
                                                    int coefficient,
                                                    Tensor3<double, TargetSpace::EmbeddedTangentVector::size,TargetSpace::EmbeddedTangentVector::size,dim>& result) const;

    /** \brief Evaluate the derivative of the gradient of the function with respect to a coefficient */
    void evaluateFDDerivativeOfGradientWRTCoefficient(const Dune::FieldVector<ctype, dim>& local,
                                                    int coefficient,
                                                    Tensor3<double, TargetSpace::EmbeddedTangentVector::size,TargetSpace::EmbeddedTangentVector::size,dim>& result) const;

private:

    /** \brief The linear part of the map that turns coordinates on the reference simplex into coordinates on the standard simplex 
        \todo A special-purpose implementation of this matrix may lead to some speed-up */
    static Dune::FieldMatrix<ctype,dim+1,dim> referenceToBarycentricLinearPart()
    {
        Dune::FieldMatrix<ctype,dim+1,dim> B;
        B[0] = -1;
        for (int i=0; i<dim; i++)
            for (int j=0; j<dim; j++)
                B[i+1][j] = (i==j);
        return B;
    }
        
    static std::vector<ctype> barycentricCoordinates(const Dune::FieldVector<ctype,dim>& local) {
        std::vector<ctype> result(dim+1);
        result[0] = 1;
        for (int i=0; i<dim; i++) {
            result[0]  -= local[i];
            result[i+1] = local[i];
        }
        return result;
    }

    static Dune::FieldMatrix<double,embeddedDim,embeddedDim> pseudoInverse(const Dune::FieldMatrix<double,embeddedDim,embeddedDim>& A)
    {
        Dune::FieldMatrix<double,embeddedDim,embeddedDim> U = A;
        Dune::FieldVector<double,embeddedDim> W;
        Dune::FieldMatrix<double,embeddedDim,embeddedDim> V;

        svdcmp(U,W,V);

        // pseudoInv = V W^{-1} U^T
        Dune::FieldMatrix<double,embeddedDim,embeddedDim> UT;

        for (int i=0; i<embeddedDim; i++)
            for (int j=0; j<embeddedDim; j++)
                UT[i][j] = U[j][i];

        for (int i=0; i<embeddedDim; i++) {
            if (std::abs(W[i]) > 1e-12)  // Diagonal may be zero, that's why we're using the pseudo inverse
                UT[i] /= W[i];
            else
                UT[i] = 0;
        }

        return V*UT;
    }
    
    /** \brief Compute derivate of F(w,q) (the derivative of the weighted distance fctl) wrt to w */
    Dune::FieldMatrix<ctype,embeddedDim,dim+1> computeDFdw(const TargetSpace& q) const
    {
        Dune::FieldMatrix<ctype,embeddedDim,dim+1> dFdw;
        for (int i=0; i<dim+1; i++) {
            Dune::FieldVector<ctype,embeddedDim> tmp = TargetSpace::derivativeOfDistanceSquaredWRTSecondArgument(coefficients_[i], q);
            for (int j=0; j<embeddedDim; j++)
                dFdw[j][i] = tmp[j];
        }
        return dFdw;
    }
    
    /** \brief The coefficient vector */
    std::vector<TargetSpace> coefficients_;

};

template <int dim, class ctype, class TargetSpace>
TargetSpace LocalGeodesicFEFunction<dim,ctype,TargetSpace>::
evaluate(const Dune::FieldVector<ctype, dim>& local) const
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
evaluateDerivative(const Dune::FieldVector<ctype, dim>& local) const
{
    Dune::FieldMatrix<ctype, embeddedDim, dim> result;

#if 0  // this is probably faster than the general implementation, but we leave it out for testing purposes
    if (dim==1) {

        EmbeddedTangentVector tmp = TargetSpace::interpolateDerivative(coefficients_[0], coefficients_[1], local[0]);

        for (int i=0; i<embeddedDim; i++)
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
    Dune::FieldMatrix<ctype,dim+1,dim> B = referenceToBarycentricLinearPart();

    // compute negative derivative of F(w,q) (the derivative of the weighted distance fctl) wrt to w
    Dune::FieldMatrix<ctype,embeddedDim,dim+1> dFdw = computeDFdw(q);
    dFdw *= -1;

    // multiply the two previous matrices: the result is the right hand side
    Dune::FieldMatrix<ctype,embeddedDim,dim> RHS = dFdw * B;

    // the actual system matrix
    std::vector<ctype> w = barycentricCoordinates(local);
    AverageDistanceAssembler<TargetSpace> assembler(coefficients_, w);
    
    Dune::FieldMatrix<ctype,embeddedDim,embeddedDim> dFdq(0);
    assembler.assembleHessian(q,dFdq);

    // ////////////////////////////////////
    //   solve the system
    // ////////////////////////////////////

    // dFDq is not invertible, if the target space is embedded into a higher-dimensional
    // Euclidean space.  Therefore we use its pseudo inverse.  I don't think that is the
    // best way, though.
    Dune::FieldMatrix<ctype,embeddedDim,embeddedDim> dFdqPseudoInv = pseudoInverse(dFdq);

    for (int i=0; i<dim; i++) {

        Dune::FieldVector<ctype,embeddedDim> rhs, x;
        for (int j=0; j<embeddedDim; j++)
            rhs[j] = RHS[j][i];

        //dFdq.solve(x, rhs);
        dFdqPseudoInv.mv(rhs,x);

        for (int j=0; j<embeddedDim; j++)
            result[j][i] = x[j];

    }

    return result;
}

template <int dim, class ctype, class TargetSpace>
Dune::FieldMatrix<ctype, TargetSpace::EmbeddedTangentVector::size, dim> LocalGeodesicFEFunction<dim,ctype,TargetSpace>::
evaluateDerivativeFD(const Dune::FieldVector<ctype, dim>& local) const
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

template <int dim, class ctype, class TargetSpace>
void LocalGeodesicFEFunction<dim,ctype,TargetSpace>::
evaluateDerivativeOfGradientWRTCoefficient(const Dune::FieldVector<ctype, dim>& local,
                                           int coefficient,
                                           Tensor3<double, TargetSpace::EmbeddedTangentVector::size,TargetSpace::EmbeddedTangentVector::size,dim>& result) const
{
    const int embeddedDim = EmbeddedTangentVector::size;
    
    // the function value at the point where we are evaluating the derivative
    TargetSpace q = evaluate(local);

    // the matrix that turns coordinates on the reference simplex into coordinates on the standard simplex
    Dune::FieldMatrix<ctype,dim+1,dim> B = referenceToBarycentricLinearPart();
    
    // compute derivate of F(w,q) (the derivative of the weighted distance fctl) wrt to w
    Dune::FieldMatrix<ctype,embeddedDim,dim+1> dFdw = computeDFdw(q);
    
    // the actual system matrix
    std::vector<ctype> w = barycentricCoordinates(local);
    AverageDistanceAssembler<TargetSpace> assembler(coefficients_, w);
    
    Dune::FieldMatrix<ctype,embeddedDim,embeddedDim> dFdq(0);
    assembler.assembleHessian(q,dFdq);
    
   
    Dune::FieldMatrix<ctype,embeddedDim,embeddedDim> mixedDerivative = TargetSpace::secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(coefficients_[coefficient], q);
    Tensor3<double,embeddedDim,embeddedDim,dim+1> dvDwF(0);
    for (int i=0; i<embeddedDim; i++)
        for (int j=0; j<embeddedDim; j++)
            dvDwF[i][j][coefficient] = mixedDerivative[i][j];
    
    
    // dFDq is not invertible, if the target space is embedded into a higher-dimensional
    // Euclidean space.  Therefore we use its pseudo inverse.  I don't think that is the
    // best way, though.
    Dune::FieldMatrix<ctype,embeddedDim,embeddedDim> dFdqPseudoInv = pseudoInverse(dFdq);
    
    //
    Tensor3<double,embeddedDim,embeddedDim,embeddedDim> dvDqF
       =  TargetSpace::thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(coefficients_[coefficient], q);
    
    dvDqF = w[coefficient] * dvDqF;
       
    // Put it all together
#if 0
    for (size_t i=0; i<result.size(); i++)
        result[i] = dFdqPseudoInv * ( dvDqF[i] * dFdqPseudoInv * dFdw - dvDwF[i]) * B;   
#else
    Dune::FieldMatrix<ctype, TargetSpace::EmbeddedTangentVector::size, dim> derivative = evaluateDerivative(local);
    Tensor3<double, TargetSpace::EmbeddedTangentVector::size,TargetSpace::EmbeddedTangentVector::size,dim> foo;
    foo = -1 * dvDwF * B - dvDqF*derivative;
    result = 0;
    for (int i=0; i<embeddedDim; i++)
        for (int j=0; j<embeddedDim; j++)
            for (int k=0; k<dim; k++)
                for (int l=0; l<embeddedDim; l++)
                    result[i][j][k] += dFdqPseudoInv[j][l] * foo[i][l][k];
#endif
}


template <int dim, class ctype, class TargetSpace>
void LocalGeodesicFEFunction<dim,ctype,TargetSpace>::
evaluateFDDerivativeOfGradientWRTCoefficient(const Dune::FieldVector<ctype, dim>& local,
                                           int coefficient,
                                           Tensor3<double, TargetSpace::EmbeddedTangentVector::size,TargetSpace::EmbeddedTangentVector::size,dim>& result) const
{
    double eps = 1e-6;
    for (int j=0; j<TargetSpace::EmbeddedTangentVector::size; j++) {
                
        std::vector<TargetSpace> cornersPlus  = coefficients_;
        std::vector<TargetSpace> cornersMinus = coefficients_;
        typename TargetSpace::CoordinateType aPlus  = coefficients_[coefficient].globalCoordinates();
        typename TargetSpace::CoordinateType aMinus = coefficients_[coefficient].globalCoordinates();
        aPlus[j]  += eps;
        aMinus[j] -= eps;
        cornersPlus[coefficient]  = TargetSpace(aPlus);
        cornersMinus[coefficient] = TargetSpace(aMinus);
        LocalGeodesicFEFunction<dim,double,TargetSpace> fPlus(cornersPlus);
        LocalGeodesicFEFunction<dim,double,TargetSpace> fMinus(cornersMinus);
                
        Dune::FieldMatrix<double,TargetSpace::EmbeddedTangentVector::size,dim> hPlus  = fPlus.evaluateDerivative(local);
        Dune::FieldMatrix<double,TargetSpace::EmbeddedTangentVector::size,dim> hMinus = fMinus.evaluateDerivative(local);
                
        result[j]  = hPlus;
        result[j] -= hMinus;
        result[j] /= 2*eps;
                
        TargetSpace q = evaluate(local);
        Dune::FieldVector<double,TargetSpace::EmbeddedTangentVector::size> foo;
        for (int l=0; l<dim; l++) {
                    
            for (int k=0; k<TargetSpace::EmbeddedTangentVector::size; k++)
                foo[k] = result[j][k][l];

            foo = q.projectOntoTangentSpace(foo);

            for (int k=0; k<TargetSpace::EmbeddedTangentVector::size; k++)
                result[j][k][l] = foo[k];
                    
        }
        
    }
    
}

#endif
