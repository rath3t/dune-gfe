#ifndef COSSERAT_ENERGY_LOCAL_STIFFNESS_HH
#define COSSERAT_ENERGY_LOCAL_STIFFNESS_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/parametertree.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/fufem/functions/virtualgridfunction.hh>
#include <dune/fufem/boundarypatch.hh>

#include "localgeodesicfestiffness.hh"
#include "localgeodesicfefunction.hh"
#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/tensor3.hh>
#include <dune/gfe/orthogonalmatrix.hh>

#define DONT_USE_CURL

//#define QUADRATIC_MEMBRANE_ENERGY


template<class GridView, class LocalFiniteElement, int dim>
class CosseratEnergyLocalStiffness 
    : public LocalGeodesicFEStiffness<GridView,LocalFiniteElement,RigidBodyMotion<double,dim> >
{
    // grid types
    typedef typename GridView::Grid::ctype DT;
    typedef RigidBodyMotion<double,dim> TargetSpace;
    typedef typename TargetSpace::ctype RT;
    typedef typename GridView::template Codim<0>::Entity Entity;
    
    // some other sizes
    enum {gridDim=GridView::dimension};
    
    
    /** \brief Compute the symmetric part of a matrix A, i.e. \f$ \frac 12 (A + A^T) \f$ */
    static Dune::FieldMatrix<double,dim,dim> sym(const Dune::FieldMatrix<double,dim,dim>& A)
    {
        Dune::FieldMatrix<double,dim,dim> result;
        for (int i=0; i<dim; i++)
            for (int j=0; j<dim; j++)
                result[i][j] = 0.5 * (A[i][j] + A[j][i]);
        return result;
    }

    /** \brief Compute the antisymmetric part of a matrix A, i.e. \f$ \frac 12 (A - A^T) \f$ */
    static Dune::FieldMatrix<double,dim,dim> skew(const Dune::FieldMatrix<double,dim,dim>& A)
    {
        Dune::FieldMatrix<double,dim,dim> result;
        for (int i=0; i<dim; i++)
            for (int j=0; j<dim; j++)
                result[i][j] = 0.5 * (A[i][j] - A[j][i]);
        return result;
    }
    
    /** \brief Return the square of the trace of a matrix */
    template <int N>
    static double traceSquared(const Dune::FieldMatrix<double,N,N>& A)
    {
        double trace = 0;
        for (int i=0; i<N; i++)
            trace += A[i][i];
        return trace*trace;
    }

    /** \brief Compute the (row-wise) curl of a matrix R \f$ 
        \param DR The partial derivatives of the matrix R
     */
    static Dune::FieldMatrix<double,dim,dim> curl(const Tensor3<double,dim,dim,dim>& DR)
    {
        Dune::FieldMatrix<double,dim,dim> result;
        
        for (int i=0; i<dim; i++) {
            result[i][0] = DR[i][2][1] - DR[i][1][2];
            result[i][1] = DR[i][0][2] - DR[i][2][0];
            result[i][2] = DR[i][1][0] - DR[i][0][1];
        }
            
        return result;
    }

    /** \brief Return the adjugate of the strain matrix U
     * 
     * Optimized for U, as we know a bit about its structure
     */
    static Dune::FieldMatrix<double,dim,dim> adjugateU(const Dune::FieldMatrix<double,dim,dim>& U)
    {
        Dune::FieldMatrix<double,dim,dim> result;
        
        result[0][0] =  U[1][1];
        result[0][1] = -U[0][1];
        result[0][2] =  0;

        result[1][0] = -U[1][0];
        result[1][1] =  U[0][0];
        result[1][2] =  0;

        result[2][0] =   U[1][0]*U[2][1] - U[2][0]*U[1][1];
        result[2][1] = - U[0][0]*U[2][1] + U[2][0]*U[0][1];
        result[2][2] =   U[0][0]*U[1][1] - U[1][0]*U[0][1];

        return result;
    }
    
    
public:  // for testing
    /** \brief Compute the derivative of the rotation (with respect to x), but wrt matrix coordinates
        \param value Value of the gfe function at a certain point
        \param derivative First derivative of the gfe function wrt x at that point, in quaternion coordinates
        \param DR First derivative of the gfe function wrt x at that point, in matrix coordinates
     */
    static void computeDR(const RigidBodyMotion<double,3>& value, 
                          const Dune::FieldMatrix<double,7,gridDim>& derivative,
                          Tensor3<double,3,3,3>& DR)
    {
        // The LocalGFEFunction class gives us the derivatives of the orientation variable,
        // but as a map into quaternion space.  To obtain matrix coordinates we use the
        // chain rule, which means that we have to multiply the given derivative with
        // the derivative of the embedding of the unit quaternion into the space of 3x3 matrices.
        // This second derivative is almost given by the method getFirstDerivativesOfDirectors.
        // However, since the directors of a given unit quaternion are the _columns_ of the
        // corresponding orthogonal matrix, we need to invert the i and j indices
        //
        // So, if I am not mistaken, DR[i][j][k] contains \partial R_ij / \partial k
        Tensor3<double,3 , 3, 4> dd_dq;
        value.q.getFirstDerivativesOfDirectors(dd_dq);
        
        DR = 0;
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                for (int k=0; k<gridDim; k++)
                    for (int l=0; l<4; l++)
                        DR[i][j][k] += dd_dq[j][i][l] * derivative[l+3][k];

    }

    /** \brief Compute the derivative of the rotation (with respect to the corner coefficients), but wrt matrix coordinates
        \param value Value of the gfe function at a certain point
        \param derivative First derivative of the gfe function wrt the coefficients at that point, in quaternion coordinates
     */
    static void computeDR_dv(const RigidBodyMotion<double,3>& value, 
                          const Dune::FieldMatrix<double,7,7>& derivative,
                          Tensor3<double,3,3,4>& dR_dv)
    {
        // The LocalGFEFunction class gives us the derivatives of the orientation variable,
        // but as a map into quaternion space.  To obtain matrix coordinates we use the
        // chain rule, which means that we have to multiply the given derivative with
        // the derivative of the embedding of the unit quaternion into the space of 3x3 matrices.
        // This second derivative is almost given by the method getFirstDerivativesOfDirectors.
        // However, since the directors of a given unit quaternion are the _columns_ of the
        // corresponding orthogonal matrix, we need to invert the i and j indices
        //
        // So, if I am not mistaken, DR[i][j][k] contains \partial R_ij / \partial k
        Tensor3<double,3 , 3, 4> dd_dq;
        value.q.getFirstDerivativesOfDirectors(dd_dq);
        
        dR_dv = 0;
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                for (int k=0; k<4; k++)
                    for (int l=0; l<4; l++)
                        dR_dv[i][j][k] += dd_dq[j][i][l] * derivative[k+3][l+3];

    }

        /** \brief Compute the derivative of the gradient of the rotation with respect to the corner coefficients,
         *   but in matrix coordinates
         * 
        \param value Value of the gfe function at a certain point
        \param derivative First derivative of the gfe function wrt the coefficients at that point, in quaternion coordinates
     */
    static void compute_dDR_dv(const RigidBodyMotion<double,3>& value, 
                               const Dune::FieldMatrix<double,7,gridDim>& derOfValueWRTx,
                               const Dune::FieldMatrix<double,7,7>& derOfValueWRTCoefficient,
                               const Tensor3<double,7,7,gridDim>& derOfGradientWRTCoefficient,
                               Dune::array<Tensor3<double,3,3,4>, 3>& dDR_dv,
                               Tensor3<double,3,gridDim,4>& dDR3_dv)
    {
        // The LocalGFEFunction class gives us the derivatives of the orientation variable,
        // but as a map into quaternion space.  To obtain matrix coordinates we use the
        // chain rule, which means that we have to multiply the given derivative with
        // the derivative of the embedding of the unit quaternion into the space of 3x3 matrices.
        // This second derivative is almost given by the method getFirstDerivativesOfDirectors.
        // However, since the directors of a given unit quaternion are the _columns_ of the
        // corresponding orthogonal matrix, we need to invert the i and j indices
        //
        // So, if I am not mistaken, DR[i][j][k] contains \partial R_ij / \partial k
        Tensor3<double,3 , 3, 4> dd_dq;
        value.q.getFirstDerivativesOfDirectors(dd_dq);
        
        /** \todo This is a constant -- don't recompute it every time */
        Dune::array<Tensor3<double,3,4,4>, 3> dd_dq_dq;
        Rotation<double,3>::getSecondDerivativesOfDirectors(dd_dq_dq);
        
        for (size_t i=0; i<dDR_dv.size(); i++)
            dDR_dv[i] = 0;
        
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                for (int k=0; k<gridDim; k++)
                    for (int v_i=0; v_i<4; v_i++)
                        for (int l=0; l<4; l++)
                            for (int m=0; m<4; m++)
                                dDR_dv[i][j][k][v_i] += dd_dq_dq[j][i][l][m]  * derOfValueWRTx[l+3][k] * derOfValueWRTCoefficient[v_i+3][m+3];

        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                for (int k=0; k<gridDim; k++)
                    for (int v_i=0; v_i<4; v_i++)
                        for (int l=0; l<4; l++)
                            dDR_dv[i][j][k][v_i] += dd_dq[j][i][l] * derOfGradientWRTCoefficient[v_i+3][l+3][k];

        ///////////////////////////////////////////////////////////////////////////////////////////
        // Compute covariant derivative for the third director
        // by projecting onto the tangent space at S^2.  Yes, that is something different!
        ///////////////////////////////////////////////////////////////////////////////////////////
        
        Dune::FieldMatrix<double,3,3> Mtmp;
        value.q.matrix(Mtmp);
        OrthogonalMatrix<double,3> M(Mtmp);                   

        Dune::FieldVector<double,3> tmpDirector;
        for (int i=0; i<3; i++)
            tmpDirector[i] = M.data()[i][2];
        UnitVector<double,3> director(tmpDirector);

        for (int k=0; k<gridDim; k++)
            for (int v_i=0; v_i<4; v_i++) {
                
                Dune::FieldVector<double,3> unprojected;
                for (int i=0; i<3; i++)
                    unprojected[i] = dDR_dv[i][2][k][v_i];
                
                Dune::FieldVector<double,3> projected = director.projectOntoTangentSpace(unprojected);
                
                for (int i=0; i<3; i++)
                    dDR3_dv[i][k][v_i] = projected[i];
            }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // Compute covariant derivative on SO(3) by projecting onto the tangent space at M(q).
        // This has to come second, as it overwrites the dDR_dv variable.
        ///////////////////////////////////////////////////////////////////////////////////////////

        for (int k=0; k<gridDim; k++)
            for (int v_i=0; v_i<4; v_i++) {
                
                Dune::FieldMatrix<double,3,3> unprojected;
                for (int i=0; i<3; i++)
                    for (int j=0; j<3; j++)
                        unprojected[i][j] = dDR_dv[i][j][k][v_i];
                    
                Dune::FieldMatrix<double,3,3> projected = M.projectOntoTangentSpace(unprojected);
                
                for (int i=0; i<3; i++)
                    for (int j=0; j<3; j++)
                        dDR_dv[i][j][k][v_i] = projected[i][j];
            }
            
    }

public:
    
    /** \brief Constructor with a set of material parameters
     * \param parameters The material parameters
     */
    CosseratEnergyLocalStiffness(const Dune::ParameterTree& parameters,
                                 const BoundaryPatch<GridView>* neumannBoundary,
                                 const Dune::VirtualFunction<Dune::FieldVector<double,gridDim>, Dune::FieldVector<double,3> >* neumannFunction)
    : neumannBoundary_(neumannBoundary),
      neumannFunction_(neumannFunction)
    {
        // The shell thickness
        thickness_ = parameters.template get<double>("thickness");
    
        // Lame constants
        mu_ = parameters.template get<double>("mu");
        lambda_ = parameters.template get<double>("lambda");

        // Cosserat couple modulus, preferably 0
        mu_c_ = parameters.template get<double>("mu_c");
    
        // Length scale parameter
        L_c_ = parameters.template get<double>("L_c");
    
        // Curvature exponent
        q_ = parameters.template get<double>("q");
        
        // Shear correction factor
        kappa_ = parameters.template get<double>("kappa");
    }

    /** \brief Assemble the energy for a single element */
    RT energy (const Entity& e,
               const LocalFiniteElement& localFiniteElement,
               const std::vector<TargetSpace>& localSolution) const;

    /** \brief Assemble the element gradient of the energy functional */
    virtual void assembleGradient(const Entity& element,
                                  const LocalFiniteElement& localFiniteElement,
                                  const std::vector<TargetSpace>& localSolution,
                                  std::vector<typename TargetSpace::TangentVector>& localGradient) const;

    /** \brief The energy \f$ W_{mp}(\overline{U}) \f$, as written in
     * the first equation of (4.4) in Neff's paper
     */
    RT quadraticMembraneEnergy(const Dune::FieldMatrix<double,3,3>& U) const
    {
        Dune::FieldMatrix<double,3,3> UMinus1 = U;
        for (int i=0; i<dim; i++)
            UMinus1[i][i] -= 1;
        
        return mu_ * sym(UMinus1).frobenius_norm2()
                + mu_c_ * skew(UMinus1).frobenius_norm2()
                + (mu_*lambda_)/(2*mu_ + lambda_) * traceSquared(sym(UMinus1));
    }

    /** \brief The energy \f$ W_{mp}(\overline{U}) \f$, as written in
     * the second equation of (4.4) in Neff's paper
     */
    RT longQuadraticMembraneEnergy(const Dune::FieldMatrix<double,3,3>& U) const
    {
        RT result = 0;
        
        // shear-stretch energy
        Dune::FieldMatrix<double,dim-1,dim-1> sym2x2;
        for (int i=0; i<dim-1; i++)
            for (int j=0; j<dim-1; j++)
                sym2x2[i][j] = 0.5 * (U[i][j] + U[j][i]) - (i==j);

        result += mu_ * sym2x2.frobenius_norm2();
        
        // first order drill energy
        Dune::FieldMatrix<double,dim-1,dim-1> skew2x2;
        for (int i=0; i<dim-1; i++)
            for (int j=0; j<dim-1; j++)
                skew2x2[i][j] = 0.5 * (U[i][j] - U[j][i]);

        result += mu_c_ * skew2x2.frobenius_norm2();
        
        
        // classical transverse shear energy
        result += kappa_ * (mu_ + mu_c_)/2 * (U[2][0]*U[2][0] + U[2][1]*U[2][1]);
        
        // elongational stretch energy
        result += mu_*lambda_ / (2*mu_ + lambda_) * traceSquared(sym2x2);
        
        return result;
    }
    
    /** \brief Energy for large-deformation problems (private communication by Patrizio Neff)
     */
    RT nonquadraticMembraneEnergy(const Dune::FieldMatrix<double,3,3>& U) const
    {
        Dune::FieldMatrix<double,3,3> UMinus1 = U;
        for (int i=0; i<dim; i++)
            UMinus1[i][i] -= 1;
       
        RT detU = U.determinant();
        
        return mu_ * sym(UMinus1).frobenius_norm2()
                + (mu_*lambda_)/(2*mu_ + lambda_) * 0.5 * ((detU-1)*(detU-1) + (1.0/detU -1)*(1.0/detU -1));
    }

    RT curvatureEnergy(const Tensor3<double,3,3,3>& DR) const
    {
#ifdef DONT_USE_CURL
        return mu_ * std::pow(L_c_ * DR.frobenius_norm(),q_);
#else
        return mu_ * std::pow(L_c_ * curl(DR).frobenius_norm(),q_);
#endif
    }

    RT bendingEnergy(const Dune::FieldMatrix<double,dim,dim>& R, const Tensor3<double,3,3,3>& DR) const
    {
        // left-multiply the derivative of the third director (in DR[][2][]) with R^T
        Dune::FieldMatrix<double,3,3> RT_DR3;
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++) {
                RT_DR3[i][j] = 0;
                for (int k=0; k<3; k++)
                    RT_DR3[i][j] += R[k][i] * DR[k][2][j];
            }
                
            
            
        return mu_ * sym(RT_DR3).frobenius_norm2()
               + mu_c_ * skew(RT_DR3).frobenius_norm2()
               + mu_*lambda_/(2*mu_+lambda_) * traceSquared(RT_DR3);
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    //  Methods to compute the energy gradient
    ///////////////////////////////////////////////////////////////////////////////////////
    void longQuadraticMembraneEnergyGradient(typename TargetSpace::EmbeddedTangentVector& localGradient,
                                    const Dune::FieldMatrix<double,3,3>& R,
                                    const Tensor3<double,3,3,4>& dR_dv,
                                    const Dune::FieldMatrix<double,7,gridDim>& derivative,
                                    const Tensor3<double,7,7,gridDim>& derOfGradientWRTCoefficient,
                                    const Dune::FieldMatrix<double,3,3>& U) const;

    void nonquadraticMembraneEnergyGradient(typename TargetSpace::EmbeddedTangentVector& localGradient,
                                    const Dune::FieldMatrix<double,3,3>& R,
                                    const Tensor3<double,3,3,4>& dR_dv,
                                    const Dune::FieldMatrix<double,7,gridDim>& derivative,
                                    const Tensor3<double,7,7,gridDim>& derOfGradientWRTCoefficient,
                                    const Dune::FieldMatrix<double,3,3>& U) const;

    void curvatureEnergyGradient(typename TargetSpace::EmbeddedTangentVector& embeddedLocalGradient,
                                 const Dune::FieldMatrix<double,3,3>& R,
                                 const Tensor3<double,3,3,3>& DR,
                                 const Dune::array<Tensor3<double,3,3,4>, 3>& dDR_dv) const;
    
    void bendingEnergyGradient(typename TargetSpace::EmbeddedTangentVector& embeddedLocalGradient,
                               const Dune::FieldMatrix<double,3,3>& R,
                               const Tensor3<double,3,3,4>& dR_dv,
                               const Tensor3<double,3,3,3>& DR,
                               const Tensor3<double,3,gridDim,4>& dDR3_dv) const;

                                 
    /** \brief The shell thickness */
    double thickness_;
    
    /** \brief Lame constants */
    double mu_, lambda_;

    /** \brief Cosserat couple modulus, preferably 0 */
    double mu_c_;
    
    /** \brief Length scale parameter */
    double L_c_;
    
    /** \brief Curvature exponent */
    double q_;

    /** \brief Shear correction factor */
    double kappa_;
    
    /** \brief The Neumann boundary */
    const BoundaryPatch<GridView>* neumannBoundary_;
    
    /** \brief The function implementing the Neumann data */
    const Dune::VirtualFunction<Dune::FieldVector<double,gridDim>, Dune::FieldVector<double,3> >* neumannFunction_;
};

template <class GridView, class LocalFiniteElement, int dim>
typename CosseratEnergyLocalStiffness<GridView,LocalFiniteElement,dim>::RT
CosseratEnergyLocalStiffness<GridView,LocalFiniteElement,dim>::
energy(const Entity& element,
       const LocalFiniteElement& localFiniteElement,
       const std::vector<RigidBodyMotion<double,dim> >& localSolution) const
{
    RT energy = 0;

    LocalGeodesicFEFunction<gridDim, double, LocalFiniteElement, TargetSpace> localGeodesicFEFunction(localFiniteElement,
                                                                                                      localSolution);

    int quadOrder = (element.type().isSimplex()) ? localFiniteElement.localBasis().order()
                                                 : localFiniteElement.localBasis().order() * gridDim;

    const Dune::QuadratureRule<double, gridDim>& quad 
        = Dune::QuadratureRules<double, gridDim>::rule(element.type(), quadOrder);
    
    for (size_t pt=0; pt<quad.size(); pt++) {
        
        // Local position of the quadrature point
        const Dune::FieldVector<double,gridDim>& quadPos = quad[pt].position();
        
        const double integrationElement = element.geometry().integrationElement(quadPos);

        const Dune::FieldMatrix<double,gridDim,gridDim>& jacobianInverseTransposed = element.geometry().jacobianInverseTransposed(quadPos);
        
        double weight = quad[pt].weight() * integrationElement;
        
        // The value of the local function
        RigidBodyMotion<double,dim> value = localGeodesicFEFunction.evaluate(quadPos);

        // The derivative of the local function defined on the reference element
        Dune::FieldMatrix<double, TargetSpace::EmbeddedTangentVector::dimension, gridDim> referenceDerivative = localGeodesicFEFunction.evaluateDerivative(quadPos,value);

        // The derivative of the function defined on the actual element
        Dune::FieldMatrix<double, TargetSpace::EmbeddedTangentVector::dimension, gridDim> derivative(0);

        for (size_t comp=0; comp<referenceDerivative.N(); comp++)
            jacobianInverseTransposed.umv(referenceDerivative[comp], derivative[comp]);
        
        /////////////////////////////////////////////////////////
        // compute U, the Cosserat strain
        /////////////////////////////////////////////////////////
        dune_static_assert(dim>=gridDim, "Codim of the grid must be nonnegative");
        
        //
        Dune::FieldMatrix<double,dim,dim> R;
        value.q.matrix(R);
        
        /* Compute F, the deformation gradient.
           In the continuum case this is
           \f$ \hat{F} = \nabla m  \f$
           In the case of a shell it is
          \f$ \hat{F} = (\nabla m | \overline{R}_3) \f$
        */
        Dune::FieldMatrix<double,dim,dim> F;
        for (int i=0; i<dim; i++)
            for (int j=0; j<gridDim; j++)
                F[i][j] = derivative[i][j];
            
        for (int i=0; i<dim; i++)
            for (int j=gridDim; j<dim; j++)
                F[i][j] = R[i][j];
        
        // U = R^T F
        Dune::FieldMatrix<double,dim,dim> U;
        for (int i=0; i<dim; i++)
            for (int j=0; j<dim; j++) {
                U[i][j] = 0;
                for (int k=0; k<dim; k++)
                    U[i][j] += R[k][i] * F[k][j];
            }
            
        //////////////////////////////////////////////////////////
        //  Compute the derivative of the rotation
        //  Note: we need it in matrix coordinates
        //////////////////////////////////////////////////////////
                
        Tensor3<double,3,3,3> DR;
        computeDR(value, derivative, DR);
        
        // Add the local energy density
        if (gridDim==2) {
#ifdef QUADRATIC_MEMBRANE_ENERGY
            //energy += weight * thickness_ * quadraticMembraneEnergy(U);
            energy += weight * thickness_ * longQuadraticMembraneEnergy(U);
#else
            energy += weight * thickness_ * nonquadraticMembraneEnergy(U);
#endif
            energy += weight * thickness_ * curvatureEnergy(DR);
            energy += weight * std::pow(thickness_,3) / 12.0 * bendingEnergy(R,DR);
        } else if (gridDim==3) {
            energy += weight * quadraticMembraneEnergy(U);
            energy += weight * curvatureEnergy(DR);
        } else
            DUNE_THROW(Dune::NotImplemented, "CosseratEnergyStiffness for 1d grids");

    }
    
    
    //////////////////////////////////////////////////////////////////////////////
    //   Assemble boundary contributions
    //////////////////////////////////////////////////////////////////////////////
    
    assert(neumannFunction_);

    for (typename Entity::LeafIntersectionIterator it = element.ileafbegin(); it != element.ileafend(); ++it) {
        
        if (not neumannBoundary_ or not neumannBoundary_->contains(*it))
            continue;
        
        const Dune::QuadratureRule<double, gridDim-1>& quad 
            = Dune::QuadratureRules<double, gridDim-1>::rule(it->type(), quadOrder);
    
        for (size_t pt=0; pt<quad.size(); pt++) {
        
            // Local position of the quadrature point
            const Dune::FieldVector<double,gridDim>& quadPos = it->geometryInInside().global(quad[pt].position());
        
            const double integrationElement = it->geometry().integrationElement(quad[pt].position());

            // The value of the local function
            RigidBodyMotion<double,dim> value = localGeodesicFEFunction.evaluate(quadPos);

            // Value of the Neumann data at the current position
            Dune::FieldVector<double,3> neumannValue;
            
            if (dynamic_cast<const VirtualGridViewFunction<GridView,Dune::FieldVector<double,3> >*>(neumannFunction_))
                dynamic_cast<const VirtualGridViewFunction<GridView,Dune::FieldVector<double,3> >*>(neumannFunction_)->evaluateLocal(element, quadPos, neumannValue);
            else
                neumannFunction_->evaluate(it->geometry().global(quad[pt].position()), neumannValue);

            // Only translational dofs are affected by the Neumann force
            energy += thickness_ * (neumannValue * value.r) * quad[pt].weight() * integrationElement;
            
        }
        
    }
    
    return energy;
}

template <class GridView, class LocalFiniteElement, int dim>
void CosseratEnergyLocalStiffness<GridView, LocalFiniteElement, dim>::
nonquadraticMembraneEnergyGradient(typename TargetSpace::EmbeddedTangentVector& embeddedLocalGradient,
                                    const Dune::FieldMatrix<double,3,3>& R,
                                    const Tensor3<double,3,3,4>& dR_dv,
                                    const Dune::FieldMatrix<double,7,gridDim>& derivative,
                                    const Tensor3<double,7,7,gridDim>& derOfGradientWRTCoefficient,
                                    const Dune::FieldMatrix<double,3,3>& U
                                   ) const
{
    // Compute partial/partial v ((R_1|R_2)^T\nablam)
    Tensor3<double,3,3,7> dU_dv(0);
        
    for (size_t i=0; i<3; i++) {

        for (size_t j=0; j<2; j++) {
                
            for (size_t k=0; k<3; k++) {
                
                // Translational dofs of the coefficient
                for (size_t v_i=0; v_i<3; v_i++)
                    dU_dv[i][j][v_i] += R[k][i] * derOfGradientWRTCoefficient[k][v_i][j];
                
                // Rotational dofs of the coefficient
                for (size_t v_i=0; v_i<4; v_i++)
                    dU_dv[i][j][v_i+3] += dR_dv[k][i][v_i] * derivative[k][j];
                    
            }
        
        }
        
        dU_dv[i][2] = 0;
        
    }
    
    
    ////////////////////////////////////////////////////////////////////////////////////
    //  Quadratic part
    ////////////////////////////////////////////////////////////////////////////////////

    for (size_t v_i=0; v_i<7; v_i++) {
     
        for (size_t i=0; i<3; i++)
            for (size_t j=0; j<3; j++) {
                double symUMinusI = 0.5*U[i][j] + 0.5*U[j][i] - (i==j);
                embeddedLocalGradient[v_i] += mu_ * symUMinusI * (dU_dv[i][j][v_i] + dU_dv[j][i][v_i]);
            }

    }


    /////////////////////////////////////////////////////////////////////////////////////
    //  Nonlinear part
    /////////////////////////////////////////////////////////////////////////////////////
    RT detU = U.determinant();  // todo: use structure of U
    Dune::FieldMatrix<RT,3,3> adjU = adjugateU(U);  // the transpose of the derivative of detU
    
    for (size_t v_i=0; v_i<7; v_i++)
        for (size_t i=0; i<3; i++)
            for (size_t j=0; j<3; j++)
                embeddedLocalGradient[v_i] += 2 * (detU - 1 - (1.0/detU -1)/(detU*detU)) * adjU[j][i] * dU_dv[i][j][v_i];

}


template <class GridView, class LocalFiniteElement, int dim>
void CosseratEnergyLocalStiffness<GridView, LocalFiniteElement, dim>::
longQuadraticMembraneEnergyGradient(typename TargetSpace::EmbeddedTangentVector& embeddedLocalGradient,
                                    const Dune::FieldMatrix<double,3,3>& R,
                                    const Tensor3<double,3,3,4>& dR_dv,
                                    const Dune::FieldMatrix<double,7,gridDim>& derivative,
                                    const Tensor3<double,7,7,gridDim>& derOfGradientWRTCoefficient,
                                    const Dune::FieldMatrix<double,3,3>& U
                                   ) const
{
    // Compute partial/partial v ((R_1|R_2)^T\nablam)
    Tensor3<double,2,2,7> dR1R2Tnablam_dv(0);
        
    for (size_t i=0; i<2; i++) {

        for (size_t j=0; j<2; j++) {
                
            for (size_t k=0; k<3; k++) {
                
                // Translational dofs of the coefficient
                for (size_t v_i=0; v_i<3; v_i++)
                    dR1R2Tnablam_dv[i][j][v_i] += R[k][i] * derOfGradientWRTCoefficient[k][v_i][j];
                
                // Rotational dofs of the coefficient
                for (size_t v_i=0; v_i<4; v_i++)
                    dR1R2Tnablam_dv[i][j][v_i+3] += dR_dv[k][i][v_i] * derivative[k][j];
                    
            }
        
        }
        
    }
    
    
    ////////////////////////////////////////////////////////////////////////////////////
    //  Shear--Stretch Energy
    ////////////////////////////////////////////////////////////////////////////////////

    for (size_t v_i=0; v_i<7; v_i++) {
     
        for (size_t i=0; i<2; i++)
            for (size_t j=0; j<2; j++) {
                double symUMinusI = 0.5*U[i][j] + 0.5*U[j][i] - (i==j);
                embeddedLocalGradient[v_i] += mu_ * symUMinusI * (dR1R2Tnablam_dv[i][j][v_i] + dR1R2Tnablam_dv[j][i][v_i]);
            }

    }


    ////////////////////////////////////////////////////////////////////////////////////
    //  First-order Drill Energy
    ////////////////////////////////////////////////////////////////////////////////////

    for (size_t v_i=0; v_i<7; v_i++) {
     
        for (size_t i=0; i<2; i++)
            for (size_t j=0; j<2; j++)
                embeddedLocalGradient[v_i] += mu_c_ * 0.5* (U[i][j]-U[j][i]) * (dR1R2Tnablam_dv[i][j][v_i] - dR1R2Tnablam_dv[j][i][v_i]);
    
    }


    ////////////////////////////////////////////////////////////////////////////////////
    //  Classical Transverse Shear Energy
    ////////////////////////////////////////////////////////////////////////////////////
    
    for (size_t v_i=0; v_i<7; v_i++) {
     
        for (size_t j=0; j<2; j++) {
            double sp = 0;
            for (size_t k=0; k<3; k++)
                sp += R[k][2]*derivative[k][j];
            double spDer = 0;
            
            if (v_i < 3)
                for (size_t k=0; k<3; k++)
                    spDer += R[k][2] * derOfGradientWRTCoefficient[k][v_i][j];
            else
                for (size_t k=0; k<3; k++)
                    spDer += dR_dv[k][2][v_i-3] * derivative[k][j];
                
            embeddedLocalGradient[v_i] += 0.5 * kappa_ * (mu_ + mu_c_) * 2 * sp * spDer;
        }

    }


    ////////////////////////////////////////////////////////////////////////////////////
    //  Elongational Stretch Energy
    ////////////////////////////////////////////////////////////////////////////////////
    
    double trace = U[0][0] + U[1][1] - 2;
    
    for (size_t v_i=0; v_i<7; v_i++)
        for (size_t i=0; i<2; i++)
            embeddedLocalGradient[v_i] += 2 * mu_*lambda_ / (2*mu_+lambda_)* trace * dR1R2Tnablam_dv[i][i][v_i];

}


template <class GridView, class LocalFiniteElement, int dim>
void CosseratEnergyLocalStiffness<GridView, LocalFiniteElement, dim>::
curvatureEnergyGradient(typename TargetSpace::EmbeddedTangentVector& embeddedLocalGradient,
                        const Dune::FieldMatrix<double,3,3>& R,
                        const Tensor3<double,3,3,3>& DR,
                        const Dune::array<Tensor3<double,3,3,4>, 3>& dDR_dv) const
{
#ifndef DONT_USE_CURL
#error curvatureEnergyGradient not implemented for the curl curvature energy
#endif
    embeddedLocalGradient = 0;
    
    for (size_t v_i=0; v_i<4; v_i++) {
        
        for (size_t i=0; i<3; i++)
            for (size_t j=0; j<3; j++)
                for (size_t k=0; k<3; k++)
                    embeddedLocalGradient[v_i+3] += DR[i][j][k] * dDR_dv[i][j][k][v_i];
        
    }

    // multiply with constant pre-factor
    embeddedLocalGradient *= mu_ * q_ * std::pow(L_c_, q_) * std::pow(DR.frobenius_norm(), q_ - 2);
}


template <class GridView, class LocalFiniteElement, int dim>
void CosseratEnergyLocalStiffness<GridView, LocalFiniteElement, dim>::
bendingEnergyGradient(typename TargetSpace::EmbeddedTangentVector& embeddedLocalGradient,
                      const Dune::FieldMatrix<double,3,3>& R,
                      const Tensor3<double,3,3,4>& dR_dv,
                      const Tensor3<double,3,3,3>& DR,
                      const Tensor3<double,3,gridDim,4>& dDR3_dv) const
{
    embeddedLocalGradient = 0;
    
    // left-multiply the derivative of the third director (in DR[][2][]) with R^T
    Dune::FieldMatrix<double,3,3> RT_DR3(0);
    for (int i=0; i<3; i++)
        for (int j=0; j<gridDim; j++)
            for (int k=0; k<3; k++)
                RT_DR3[i][j] += R[k][i] * DR[k][2][j];

    for (int i=0; i<gridDim; i++)
        assert(std::fabs(RT_DR3[2][i]) < 1e-7);
        
    // -----------------------------------------------------------------------------
    
    Tensor3<double,3,3,4> d_RT_DR3(0);
    for (size_t v_i=0; v_i<4; v_i++)
        for (int i=0; i<3; i++)
            for (int j=0; j<gridDim; j++)
                for (int k=0; k<3; k++)
                    d_RT_DR3[i][j][v_i] += dR_dv[k][i][v_i] * DR[k][2][j] + R[k][i] * dDR3_dv[k][j][v_i];

    // Project onto the tangent space at (0,0,1).  I don't really understand why this is needed,
    // but it does make the test pass in all cases.  I suppose that the chain rule demands it
    // in some way or the other.
    for (int i=0; i<gridDim; i++)
        for (size_t v_i=0; v_i<4; v_i++)
            d_RT_DR3[2][i][v_i] = 0;
            
     ////////////////////////////////////////////////////////////////////////////////
     //  "Shear--Stretch Energy"
     ////////////////////////////////////////////////////////////////////////////////

    for (size_t v_i=0; v_i<4; v_i++)
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                embeddedLocalGradient[v_i+3] += mu_ * 0.5 * (RT_DR3[i][j]+RT_DR3[j][i]) * (d_RT_DR3[i][j][v_i] + d_RT_DR3[j][i][v_i]);

    ////////////////////////////////////////////////////////////////////////////////
    //  "First-order Drill Energy"
    ////////////////////////////////////////////////////////////////////////////////

    for (size_t v_i=0; v_i<4; v_i++)
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                embeddedLocalGradient[v_i+3] += mu_c_ * 0.5 * (RT_DR3[i][j]-RT_DR3[j][i]) * (d_RT_DR3[i][j][v_i] - d_RT_DR3[j][i][v_i]);

    ////////////////////////////////////////////////////////////////////////////////
    //  "Elongational Stretch Energy"
    ////////////////////////////////////////////////////////////////////////////////

    double factor = 2 * mu_*lambda_ / (2*mu_ + lambda_) * (RT_DR3[0][0] + RT_DR3[1][1] + RT_DR3[2][2]);
    for (size_t v_i=0; v_i<4; v_i++)
        for (int i=0; i<3; i++)
            embeddedLocalGradient[v_i+3] += factor * d_RT_DR3[i][i][v_i];
        
}

template <class GridView, class LocalFiniteElement, int dim>
void CosseratEnergyLocalStiffness<GridView, LocalFiniteElement, dim>::
assembleGradient(const Entity& element,
                 const LocalFiniteElement& localFiniteElement,
                 const std::vector<TargetSpace>& localSolution,
                 std::vector<typename TargetSpace::TangentVector>& localGradient) const
{
    std::vector<typename TargetSpace::EmbeddedTangentVector> embeddedLocalGradient(localSolution.size());
    std::fill(embeddedLocalGradient.begin(), embeddedLocalGradient.end(), 0);

    
    LocalGeodesicFEFunction<gridDim, double, LocalFiniteElement, TargetSpace> localGeodesicFEFunction(localFiniteElement,
                                                                                                      localSolution);

    /** \todo Is this larger than necessary? */
    int quadOrder = (element.type().isSimplex()) ? localFiniteElement.localBasis().order()
                                                 : localFiniteElement.localBasis().order() * gridDim;

    const Dune::QuadratureRule<double, gridDim>& quad 
        = Dune::QuadratureRules<double, gridDim>::rule(element.type(), quadOrder);
    
    for (size_t pt=0; pt<quad.size(); pt++) {
        
        // Local position of the quadrature point
        const Dune::FieldVector<double,gridDim>& quadPos = quad[pt].position();
        
        const double integrationElement = element.geometry().integrationElement(quadPos);

        const Dune::FieldMatrix<double,gridDim,gridDim>& jacobianInverseTransposed = element.geometry().jacobianInverseTransposed(quadPos);
        
        double weight = quad[pt].weight() * integrationElement;
        
        // The value of the local function
        RigidBodyMotion<double,dim> value = localGeodesicFEFunction.evaluate(quadPos);

        // The derivative of the local function defined on the reference element
        Dune::FieldMatrix<double, TargetSpace::EmbeddedTangentVector::dimension, gridDim> referenceDerivative = localGeodesicFEFunction.evaluateDerivative(quadPos,value);

        // The derivative of the function defined on the actual element
        Dune::FieldMatrix<double, TargetSpace::EmbeddedTangentVector::dimension, gridDim> derivative(0);

        for (size_t comp=0; comp<referenceDerivative.N(); comp++)
            jacobianInverseTransposed.umv(referenceDerivative[comp], derivative[comp]);
        
        /////////////////////////////////////////////////////////
        // compute U, the Cosserat strain
        /////////////////////////////////////////////////////////
        dune_static_assert(dim>=gridDim, "Codim of the grid must be nonnegative");
        
        //
        Dune::FieldMatrix<double,dim,dim> R;
        value.q.matrix(R);
        
        /* Compute F, the deformation gradient.
           In the continuum case this is
           \f$ \hat{F} = \nabla m  \f$
           In the case of a shell it is
          \f$ \hat{F} = (\nabla m | \overline{R}_3) \f$
        */
        Dune::FieldMatrix<double,dim,dim> F;
        for (int i=0; i<dim; i++)
            for (int j=0; j<gridDim; j++)
                F[i][j] = derivative[i][j];
            
        for (int i=0; i<dim; i++)
            for (int j=gridDim; j<dim; j++)
                F[i][j] = R[i][j];
        
        // U = R^T F
        Dune::FieldMatrix<double,dim,dim> U;
        for (int i=0; i<dim; i++)
            for (int j=0; j<dim; j++) {
                U[i][j] = 0;
                for (int k=0; k<dim; k++)
                    U[i][j] += R[k][i] * F[k][j];
            }
            
        //////////////////////////////////////////////////////////
        //  Compute the derivative of the rotation
        //  Note: we need it in matrix coordinates
        //////////////////////////////////////////////////////////
                
        Tensor3<double,3,3,3> DR;
        computeDR(value, derivative, DR);

        // loop over all the element's degrees of freedom and compute the gradient wrt it
        for (size_t i=0; i<localSolution.size(); i++) {
         
            // --------------------------------------------------------------------
            Dune::FieldMatrix<double,7,7> derOfValueWRTCoefficient;
            localGeodesicFEFunction.evaluateDerivativeOfValueWRTCoefficient(quadPos, i, derOfValueWRTCoefficient);

            Tensor3<double,3,3,4> dR_dv;
            computeDR_dv(value, derOfValueWRTCoefficient, dR_dv);

            // --------------------------------------------------------------------
            Tensor3<double, TargetSpace::EmbeddedTangentVector::dimension,TargetSpace::EmbeddedTangentVector::dimension,gridDim> referenceDerivativeDerivative;
            localGeodesicFEFunction.evaluateDerivativeOfGradientWRTCoefficient(quadPos, i, referenceDerivativeDerivative);
            
            // multiply the transformation from the reference element to the actual element
            Tensor3<double, TargetSpace::EmbeddedTangentVector::dimension,TargetSpace::EmbeddedTangentVector::dimension,gridDim> derOfGradientWRTCoefficient;
            for (int ii=0; ii<TargetSpace::EmbeddedTangentVector::dimension; ii++)
                for (int jj=0; jj<TargetSpace::EmbeddedTangentVector::dimension; jj++)
                    for (int kk=0; kk<gridDim; kk++) {
                        derOfGradientWRTCoefficient[ii][jj][kk] = 0;
                        for (int ll=0; ll<gridDim; ll++)
                            derOfGradientWRTCoefficient[ii][jj][kk] += referenceDerivativeDerivative[ii][jj][ll] * jacobianInverseTransposed[kk][ll];
                    }

            // ---------------------------------------------------------------------
            Dune::array<Tensor3<double,3,3,4>, 3> dDR_dv;
            Tensor3<double,3,gridDim,4> dDR3_dv;
            compute_dDR_dv(value, derivative, derOfValueWRTCoefficient, derOfGradientWRTCoefficient, dDR_dv, dDR3_dv);
            
            // Add the local energy density
            if (gridDim==2) {
                typename TargetSpace::EmbeddedTangentVector tmp(0);
#ifdef QUADRATIC_MEMBRANE_ENERGY
                longQuadraticMembraneEnergyGradient(tmp,R,dR_dv,derivative,derOfGradientWRTCoefficient,U);
#else
                nonquadraticMembraneEnergyGradient(tmp,R,dR_dv,derivative,derOfGradientWRTCoefficient,U);
#endif
                embeddedLocalGradient[i].axpy(weight * thickness_, tmp);
                
                tmp = 0;
                curvatureEnergyGradient(tmp,R,DR,dDR_dv);
                embeddedLocalGradient[i].axpy(weight * thickness_, tmp);
                
                tmp = 0;
                bendingEnergyGradient(tmp,R,dR_dv,DR,dDR3_dv);
                embeddedLocalGradient[i].axpy(weight * std::pow(thickness_,3) / 12.0, tmp);
            } else if (gridDim==3) {
                assert(gridDim==2);  // 3d not implemented yet
//             energy += weight * quadraticMembraneEnergyGradient(U);
//             energy += weight * curvatureEnergyGradient(DR);
            } else
                DUNE_THROW(Dune::NotImplemented, "CosseratEnergyStiffness for 1d grids");

        }
        
    }
    
    //////////////////////////////////////////////////////////////////////////////
    //   Assemble boundary contributions
    //////////////////////////////////////////////////////////////////////////////
    
    assert(neumannFunction_);

    for (typename Entity::LeafIntersectionIterator it = element.ileafbegin(); it != element.ileafend(); ++it) {
        
        if (not neumannBoundary_ or not neumannBoundary_->contains(*it))
            continue;
        
        const Dune::QuadratureRule<double, gridDim-1>& quad 
            = Dune::QuadratureRules<double, gridDim-1>::rule(it->type(), quadOrder);
    
        for (size_t pt=0; pt<quad.size(); pt++) {
        
            // Local position of the quadrature point
            const Dune::FieldVector<double,gridDim>& quadPos = it->geometryInInside().global(quad[pt].position());
        
            const double integrationElement = it->geometry().integrationElement(quad[pt].position());

            // Value of the Neumann data at the current position
            Dune::FieldVector<double,3> neumannValue;
            
            if (dynamic_cast<const VirtualGridViewFunction<GridView,Dune::FieldVector<double,3> >*>(neumannFunction_))
                dynamic_cast<const VirtualGridViewFunction<GridView,Dune::FieldVector<double,3> >*>(neumannFunction_)->evaluateLocal(element, quadPos, neumannValue);
            else
                neumannFunction_->evaluate(it->geometry().global(quad[pt].position()), neumannValue);

            // loop over all the element's degrees of freedom and compute the gradient wrt it
            for (size_t i=0; i<localSolution.size(); i++) {
                
                // --------------------------------------------------------------------
                Dune::FieldMatrix<double,7,7> derOfValueWRTCoefficient;
                localGeodesicFEFunction.evaluateDerivativeOfValueWRTCoefficient(quadPos, i, derOfValueWRTCoefficient);

                // Only translational dofs are affected by the Neumann force
                for (size_t v_i=0; v_i<3; v_i++)
                    for (size_t j=0; j<3; j++)
#warning Try whether the arguments of derOfValueWRTCoefficient are swapped
                        embeddedLocalGradient[i][v_i] += thickness_ * (neumannValue[j] * derOfValueWRTCoefficient[j][v_i]) * quad[pt].weight() * integrationElement;
            
            }
        }
        
    }


    // transform to coordinates on the tangent space
    localGradient.resize(embeddedLocalGradient.size());

    for (size_t i=0; i<localGradient.size(); i++)
        localSolution[i].orthonormalFrame().mv(embeddedLocalGradient[i], localGradient[i]);
}

#endif   //#ifndef COSSERAT_ENERGY_LOCAL_STIFFNESS_HH

