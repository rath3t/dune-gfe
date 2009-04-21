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

    /** \brief Evaluate the derivative of the function using a finite-difference approximation*/
    Dune::FieldMatrix<ctype, EmbeddedTangentVector::size, dim> evaluateDerivativeFD(const Dune::FieldVector<ctype, dim>& local);

private:

    /** \brief Compute the derivate of geodesic interpolation wrt to the initial point
        and return it as a linear map on quaternion space

        \todo This is group-specific and should not really be here
    */
    static Dune::FieldMatrix<ctype, EmbeddedTangentVector::size, EmbeddedTangentVector::size>
    derivativeWRTFirstPoint(const Rotation<3,ctype>& a, const Rotation<3,ctype>& b, double s);

    /** \brief Compute the derivate of geodesic interpolation wrt to the initial point

        \todo This is group-specific and should not really be here
    */
    static void interpolationDerivative(const Rotation<3,ctype>& q0, const Rotation<3,ctype>& q1, double s,
                                        Dune::array<Quaternion<double>,6>& grad);
    

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
    // Get the derivative of [a,b](s) wrt 'a'
    /** \brief The method actually compute the derivatives wrt to 'a' and 'b'.  This is a waste! */
    Dune::array<Quaternion<double>,6> grad;
    interpolationDerivative(a,b,s,grad);

    // We are really only interested in the first three entries

    Quaternion<ctype> aInv = a;
    aInv.invert();

    Dune::array<Quaternion<double>,3> derAlpha;
    derAlpha[0] = aInv.mult(grad[0]);
    derAlpha[1] = aInv.mult(grad[1]);
    derAlpha[2] = aInv.mult(grad[2]);

    // Copy the thing into a matrix
    Dune::FieldMatrix<ctype,3,4> derAlphaMatrix;
    for (int i=0; i<3; i++)
        derAlphaMatrix[i] = derAlpha[i];

    // Get derivative of the exponential map at the identity.  Incidentally, the implementation
    // maps skew-symmetric matrices (== three-vectors) to quaternions

    Dune::FieldMatrix<ctype,4,3> Dexp = Rotation<3,ctype>::Dexp(Dune::FieldVector<ctype,3>(0));

    Dune::FieldMatrix<ctype, TargetSpace::EmbeddedTangentVector::size, TargetSpace::EmbeddedTangentVector::size> result;

    Dune::FMatrixHelp::multMatrix(Dexp, derAlphaMatrix, result);

    return result;
}

template <int dim, class ctype, class TargetSpace>
void LocalGeodesicFEFunction<dim,ctype,TargetSpace>::
interpolationDerivative(const Rotation<3,ctype>& q0, const Rotation<3,ctype>& q1, double s,
                        Dune::array<Quaternion<double>,6>& grad)
{
    // Clear output array
    for (int i=0; i<6; i++)
        grad[i] = 0;
    
    // The derivatives with respect to w^1
    
    // Compute q_1^{-1}q_0
    Rotation<3,ctype> q1InvQ0 = q1;
    q1InvQ0.invert();
    q1InvQ0 = q1InvQ0.mult(q0);
    
    {
        // Compute v = (1-s) \exp^{-1} ( q_1^{-1} q_0)
        Dune::FieldVector<ctype,3> v = Rotation<3,ctype>::expInv(q1InvQ0);
        v *= (1-s);
        
        Dune::FieldMatrix<ctype,4,3> dExp_v = Rotation<3,ctype>::Dexp(v);
        
        Dune::FieldMatrix<ctype,3,4> dExpInv = Rotation<3,ctype>::DexpInv(q1InvQ0);
        
        Dune::FieldMatrix<ctype,4,4> mat(0);
        for (int i=0; i<4; i++)
            for (int j=0; j<4; j++)
                for (int k=0; k<3; k++)
                    mat[i][j] += (1-s) * dExp_v[i][k] * dExpInv[k][j];
        
        for (int i=0; i<3; i++) {
            
            Quaternion<ctype> dw;
            for (int j=0; j<4; j++)
                dw[j] = 0.5 * (i==j);  // dExp[j][i] at v=0
            
            dw = q1InvQ0.mult(dw);
            
            mat.umv(dw,grad[i]);
            grad[i] = q1.mult(grad[i]);
            
        }
    }
    
    
    // The derivatives with respect to w^1
    
    // Compute q_0^{-1}
    Rotation<3,ctype> q0InvQ1 = q0;
    q0InvQ1.invert();
    q0InvQ1 = q0InvQ1.mult(q1);
    
    {
        // Compute v = s \exp^{-1} ( q_0^{-1} q_1)
        Dune::FieldVector<ctype,3> v = Rotation<3,ctype>::expInv(q0InvQ1);
        v *= s;
        
        Dune::FieldMatrix<ctype,4,3> dExp_v = Rotation<3,ctype>::Dexp(v);
        
        Dune::FieldMatrix<ctype,3,4> dExpInv = Rotation<3,ctype>::DexpInv(q0InvQ1);
        
        Dune::FieldMatrix<ctype,4,4> mat(0);
        for (int i=0; i<4; i++)
            for (int j=0; j<4; j++)
                for (int k=0; k<3; k++)
                    mat[i][j] += s * dExp_v[i][k] * dExpInv[k][j];
        
        for (int i=3; i<6; i++) {
            
            Quaternion<ctype> dw;
            for (int j=0; j<4; j++)
                dw[j] = 0.5 * ((i-3)==j);  // dExp[j][i-3] at v=0
            
            dw = q0InvQ1.mult(dw);
            
            mat.umv(dw,grad[i]);
            grad[i] = q0.mult(grad[i]);
            
        }
    }

}


template <int dim, class ctype, class TargetSpace>
TargetSpace LocalGeodesicFEFunction<dim,ctype,TargetSpace>::
evaluate(const Dune::FieldVector<ctype, dim>& local)
{
    ctype extraCoord = 1;
    for (int i=0; i<dim; i++)
        extraCoord -= local[i];

    ctype normalizingFactor = extraCoord+local[0];

    TargetSpace result = TargetSpace::interpolate(coefficients_[0], coefficients_[1], local[0]/normalizingFactor);

    for (int i=1; i<dim; i++) {
        normalizingFactor += local[i];
        result = TargetSpace::interpolate(result, coefficients_[i+1], local[i] / normalizingFactor);
    }

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

    return result;
}

template <int dim, class ctype, class TargetSpace>
Dune::FieldMatrix<ctype, TargetSpace::EmbeddedTangentVector::size, dim> LocalGeodesicFEFunction<dim,ctype,TargetSpace>::
evaluateDerivativeFD(const Dune::FieldVector<ctype, dim>& local)
{
    Dune::FieldMatrix<ctype, EmbeddedTangentVector::size, dim> result;

    if (dim==1) {

        EmbeddedTangentVector tmp = TargetSpace::interpolateDerivative(coefficients_[0], coefficients_[1], local[0]);

        for (int i=0; i<EmbeddedTangentVector::size; i++)
            result[i][0] = tmp[i];

    }

    if (dim==2) {

        double eps = 1e-6;

        Dune::FieldVector<ctype, dim> fX = local;
        Dune::FieldVector<ctype, dim> bX = local;
        Dune::FieldVector<ctype, dim> fY = local;
        Dune::FieldVector<ctype, dim> bY = local;

        fX[0] += eps;
        bX[0] -= eps;
        fY[1] += eps;
        bY[1] -= eps;

        Quaternion<double> dX = (evaluate(fX) - evaluate(bX));
        Quaternion<double> dY = (evaluate(fY) - evaluate(bY));

        dX /= 2*eps;
        dY /= 2*eps;

        for (int i=0; i<EmbeddedTangentVector::size; i++){
            result[i][0] = dX[i];
            result[i][1] = dY[i];
        }

    }

    assert(dim==1 || dim==2);

    return result;
}

#endif
