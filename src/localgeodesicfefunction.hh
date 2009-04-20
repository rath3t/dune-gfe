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
        and return it as a linear map on quaternion space

        \todo This is group-specific and should not really be here
    */
    static Dune::FieldMatrix<ctype, EmbeddedTangentVector::size, EmbeddedTangentVector::size>
    derivativeWRTFirstPoint(const Rotation<3,ctype>& a, const Rotation<3,ctype>& b, double s);

    /** \brief Compute the derivate of geodesic interpolation wrt to the initial point

        \todo This is group-specific and should not really be here
    */
    static void interpolationVelocityDerivative(const Rotation<3,ctype>& q0, const Rotation<3,ctype>& q1, double s,
                                         double intervalLength, Dune::array<Quaternion<double>,6>& grad);
    

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
    interpolationVelocityDerivative(a,b,s,1,grad);

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
interpolationVelocityDerivative(const Rotation<3,ctype>& q0, const Rotation<3,ctype>& q1, double s,
                                double intervalLength, Dune::array<Quaternion<double>,6>& grad)
{
    // Clear output array
    for (int i=0; i<6; i++)
        grad[i] = 0;

    // Compute q_0^{-1}
    Rotation<3,ctype> q0Inv = q0;
    q0Inv.invert();


    // Compute v = s \exp^{-1} ( q_0^{-1} q_1)
    Dune::FieldVector<ctype,3> v = Rotation<3,ctype>::expInv(q0Inv.mult(q1));
    v *= s/intervalLength;

    Dune::FieldMatrix<ctype,4,3> dExp_v = Rotation<3,ctype>::Dexp(v);

    Dune::array<Dune::FieldMatrix<ctype,3,3>, 4> ddExp;
    Rotation<3,ctype>::DDexp(v, ddExp);

    Dune::FieldMatrix<ctype,3,4> dExpInv = Rotation<3,ctype>::DexpInv(q0Inv.mult(q1));

    Dune::FieldMatrix<ctype,4,4> mat(0);
    for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
            for (int k=0; k<3; k++)
                mat[i][j] += 1/intervalLength * dExp_v[i][k] * dExpInv[k][j];

    
    // /////////////////////////////////////////////////
    // The derivatives with respect to w^0
    // /////////////////////////////////////////////////
    for (int i=0; i<3; i++) {

        // \partial exp \partial w^1_j at 0
        Quaternion<ctype> dw;
        for (int j=0; j<4; j++)
            dw[j] = 0.5*(i==j);  // dExp_v_0[j][i];

        // \xi = \exp^{-1} q_0^{-1} q_1
        Dune::FieldVector<ctype,3> xi = Rotation<3,ctype>::expInv(q0Inv.mult(q1));

        Quaternion<ctype> addend0;
        addend0 = 0;
        dExp_v.umv(xi,addend0);
        addend0 = dw.mult(addend0);
        addend0 /= intervalLength;

        //  \parder{\xi}{w^1_j} = ...
        Quaternion<ctype> dwConj = dw;
        dwConj.conjugate();
        //dwConj[3] -= 2 * dExp_v_0[3][i];   the last row of dExp_v_0 is zero
        dwConj = dwConj.mult(q0Inv.mult(q1));

        Dune::FieldVector<ctype,3> dxi(0);
        Rotation<3,ctype>::DexpInv(q0Inv.mult(q1)).umv(dwConj, dxi);

        Quaternion<ctype> vHv;
        for (int j=0; j<4; j++) {
            vHv[j] = 0;
            // vHv[j] = dxi * DDexp * xi
            for (int k=0; k<3; k++)
                for (int l=0; l<3; l++)
                    vHv[j] += ddExp[j][k][l]*dxi[k]*xi[l];
        }

        vHv *= s/intervalLength/intervalLength;

        // Third addend
        mat.umv(dwConj,grad[i]);

        // add up
        grad[i] += addend0;
        grad[i] += vHv;

        grad[i] = q0.mult(grad[i]);
    }


    // /////////////////////////////////////////////////
    // The derivatives with respect to w^1
    // /////////////////////////////////////////////////
    for (int i=3; i<6; i++) {

        // \partial exp \partial w^1_j at 0
        Quaternion<ctype> dw;
        for (int j=0; j<4; j++)
            dw[j] = 0.5 * ((i-3)==j);  // dw[j] = dExp_v_0[j][i-3];

        // \xi = \exp^{-1} q_0^{-1} q_1
        Dune::FieldVector<ctype,3> xi = Rotation<3,ctype>::expInv(q0Inv.mult(q1));

        //  \parder{\xi}{w^1_j} = ...
        Dune::FieldVector<ctype,3> dxi(0);
        dExpInv.umv(q0Inv.mult(q1.mult(dw)), dxi);

        Quaternion<ctype> vHv;
        for (int j=0; j<4; j++) {
            // vHv[j] = dxi * DDexp * xi
            vHv[j] = 0;
            for (int k=0; k<3; k++)
                for (int l=0; l<3; l++)
                    vHv[j] += ddExp[j][k][l]*dxi[k]*xi[l];
        }

        vHv *= s/intervalLength/intervalLength;

        // ///////////////////////////////////
        //   second addend
        // ///////////////////////////////////
            
        
        dw = q0Inv.mult(q1.mult(dw));
        
        mat.umv(dw,grad[i]);
        grad[i] += vHv;

        grad[i] = q0.mult(grad[i]);

    }

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
