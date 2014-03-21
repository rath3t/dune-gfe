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


template<class GridView, class LocalFiniteElement, int dim, class field_type=double>
class CosseratEnergyLocalStiffness
    : public LocalGeodesicFEStiffness<GridView,LocalFiniteElement,RigidBodyMotion<field_type,dim> >
{
    // grid types
    typedef typename GridView::Grid::ctype DT;
    typedef RigidBodyMotion<field_type,dim> TargetSpace;
    typedef typename TargetSpace::ctype RT;
    typedef typename GridView::template Codim<0>::Entity Entity;

    // some other sizes
    enum {gridDim=GridView::dimension};


    /** \brief Compute the symmetric part of a matrix A, i.e. \f$ \frac 12 (A + A^T) \f$ */
    static Dune::FieldMatrix<field_type,dim,dim> sym(const Dune::FieldMatrix<field_type,dim,dim>& A)
    {
        Dune::FieldMatrix<field_type,dim,dim> result;
        for (int i=0; i<dim; i++)
            for (int j=0; j<dim; j++)
                result[i][j] = 0.5 * (A[i][j] + A[j][i]);
        return result;
    }

    /** \brief Compute the antisymmetric part of a matrix A, i.e. \f$ \frac 12 (A - A^T) \f$ */
    static Dune::FieldMatrix<field_type,dim,dim> skew(const Dune::FieldMatrix<field_type,dim,dim>& A)
    {
        Dune::FieldMatrix<field_type,dim,dim> result;
        for (int i=0; i<dim; i++)
            for (int j=0; j<dim; j++)
                result[i][j] = 0.5 * (A[i][j] - A[j][i]);
        return result;
    }

    /** \brief Return the square of the trace of a matrix */
    template <int N>
    static field_type traceSquared(const Dune::FieldMatrix<field_type,N,N>& A)
    {
        field_type trace = 0;
        for (int i=0; i<N; i++)
            trace += A[i][i];
        return trace*trace;
    }

    /** \brief Compute the (row-wise) curl of a matrix R \f$
        \param DR The partial derivatives of the matrix R
     */
    static Dune::FieldMatrix<field_type,dim,dim> curl(const Tensor3<field_type,dim,dim,dim>& DR)
    {
        Dune::FieldMatrix<field_type,dim,dim> result;

        for (int i=0; i<dim; i++) {
            result[i][0] = DR[i][2][1] - DR[i][1][2];
            result[i][1] = DR[i][0][2] - DR[i][2][0];
            result[i][2] = DR[i][1][0] - DR[i][0][1];
        }

        return result;
    }

public:  // for testing
    /** \brief Compute the derivative of the rotation (with respect to x), but wrt matrix coordinates
        \param value Value of the gfe function at a certain point
        \param derivative First derivative of the gfe function wrt x at that point, in quaternion coordinates
        \param DR First derivative of the gfe function wrt x at that point, in matrix coordinates
     */
    static void computeDR(const RigidBodyMotion<field_type,3>& value,
                          const Dune::FieldMatrix<field_type,7,gridDim>& derivative,
                          Tensor3<field_type,3,3,3>& DR)
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
        Tensor3<field_type,3 , 3, 4> dd_dq;
        value.q.getFirstDerivativesOfDirectors(dd_dq);

        DR = field_type(0);
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                for (int k=0; k<gridDim; k++)
                    for (int l=0; l<4; l++)
                        DR[i][j][k] += dd_dq[j][i][l] * derivative[l+3][k];

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

    /** \brief The energy \f$ W_{mp}(\overline{U}) \f$, as written in
     * the first equation of (4.4) in Neff's paper
     */
    RT quadraticMembraneEnergy(const Dune::FieldMatrix<field_type,3,3>& U) const
    {
        Dune::FieldMatrix<field_type,3,3> UMinus1 = U;
        for (int i=0; i<dim; i++)
            UMinus1[i][i] -= 1;

        return mu_ * sym(UMinus1).frobenius_norm2()
                + mu_c_ * skew(UMinus1).frobenius_norm2()
                + (mu_*lambda_)/(2*mu_ + lambda_) * traceSquared(sym(UMinus1));
    }

    /** \brief The energy \f$ W_{mp}(\overline{U}) \f$, as written in
     * the second equation of (4.4) in Neff's paper
     */
    RT longQuadraticMembraneEnergy(const Dune::FieldMatrix<field_type,3,3>& U) const
    {
        RT result = 0;

        // shear-stretch energy
        Dune::FieldMatrix<field_type,dim-1,dim-1> sym2x2;
        for (int i=0; i<dim-1; i++)
            for (int j=0; j<dim-1; j++)
                sym2x2[i][j] = 0.5 * (U[i][j] + U[j][i]) - (i==j);

        result += mu_ * sym2x2.frobenius_norm2();

        // first order drill energy
        Dune::FieldMatrix<field_type,dim-1,dim-1> skew2x2;
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
    RT nonquadraticMembraneEnergy(const Dune::FieldMatrix<field_type,3,3>& U) const
    {
        Dune::FieldMatrix<field_type,3,3> UMinus1 = U;
        for (int i=0; i<dim; i++)
            UMinus1[i][i] -= 1;

        RT detU = U.determinant();

        return mu_ * sym(UMinus1).frobenius_norm2()
                + (mu_*lambda_)/(2*mu_ + lambda_) * 0.5 * ((detU-1)*(detU-1) + (1.0/detU -1)*(1.0/detU -1));
    }

    RT curvatureEnergy(const Tensor3<field_type,3,3,3>& DR) const
    {
#ifdef DONT_USE_CURL
        return mu_ * std::pow(L_c_ * L_c_ * DR.frobenius_norm2(),q_/2.0);
#else
        return mu_ * std::pow(L_c_ * L_c_ * curl(DR).frobenius_norm2(),q_/2.0);
#endif
    }

    RT bendingEnergy(const Dune::FieldMatrix<field_type,dim,dim>& R, const Tensor3<field_type,3,3,3>& DR) const
    {
        // left-multiply the derivative of the third director (in DR[][2][]) with R^T
        Dune::FieldMatrix<field_type,3,3> RT_DR3;
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

template <class GridView, class LocalFiniteElement, int dim, class field_type>
typename CosseratEnergyLocalStiffness<GridView,LocalFiniteElement,dim,field_type>::RT
CosseratEnergyLocalStiffness<GridView,LocalFiniteElement,dim,field_type>::
energy(const Entity& element,
       const LocalFiniteElement& localFiniteElement,
       const std::vector<RigidBodyMotion<field_type,dim> >& localSolution) const
{
    assert(element.type() == localFiniteElement.type());
    typedef typename GridView::template Codim<0>::Entity::Geometry Geometry;

    RT energy = 0;

    typedef LocalGeodesicFEFunction<gridDim, DT, LocalFiniteElement, TargetSpace> LocalGFEFunctionType;
    LocalGFEFunctionType localGeodesicFEFunction(localFiniteElement,localSolution);

    int quadOrder = (element.type().isSimplex()) ? localFiniteElement.localBasis().order()
                                                 : localFiniteElement.localBasis().order() * gridDim;

    const Dune::QuadratureRule<DT, gridDim>& quad
        = Dune::QuadratureRules<DT, gridDim>::rule(element.type(), quadOrder);

    for (size_t pt=0; pt<quad.size(); pt++) {

        // Local position of the quadrature point
        const Dune::FieldVector<DT,gridDim>& quadPos = quad[pt].position();

        const DT integrationElement = element.geometry().integrationElement(quadPos);

        const typename Geometry::JacobianInverseTransposed& jacobianInverseTransposed = element.geometry().jacobianInverseTransposed(quadPos);

        DT weight = quad[pt].weight() * integrationElement;

        // The value of the local function
        RigidBodyMotion<field_type,dim> value = localGeodesicFEFunction.evaluate(quadPos);

        // The derivative of the local function defined on the reference element
        typename LocalGFEFunctionType::DerivativeType referenceDerivative = localGeodesicFEFunction.evaluateDerivative(quadPos,value);

        // The derivative of the function defined on the actual element
        typename LocalGFEFunctionType::DerivativeType derivative(0);

        for (size_t comp=0; comp<referenceDerivative.N(); comp++)
            jacobianInverseTransposed.umv(referenceDerivative[comp], derivative[comp]);

        /////////////////////////////////////////////////////////
        // compute U, the Cosserat strain
        /////////////////////////////////////////////////////////
        dune_static_assert(dim>=gridDim, "Codim of the grid must be nonnegative");

        //
        Dune::FieldMatrix<field_type,dim,dim> R;
        value.q.matrix(R);

        /* Compute F, the deformation gradient.
           In the continuum case this is
           \f$ \hat{F} = \nabla m  \f$
           In the case of a shell it is
          \f$ \hat{F} = (\nabla m | \overline{R}_3) \f$
        */
        Dune::FieldMatrix<field_type,dim,dim> F;
        for (int i=0; i<dim; i++)
            for (int j=0; j<gridDim; j++)
                F[i][j] = derivative[i][j];

        for (int i=0; i<dim; i++)
            for (int j=gridDim; j<dim; j++)
                F[i][j] = R[i][j];

        // U = R^T F
        Dune::FieldMatrix<field_type,dim,dim> U;
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

        Tensor3<field_type,3,3,3> DR;
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

    if (not neumannFunction_)
        return energy;

    for (typename Entity::LeafIntersectionIterator it = element.ileafbegin(); it != element.ileafend(); ++it) {

        if (not neumannBoundary_ or not neumannBoundary_->contains(*it))
            continue;

        const Dune::QuadratureRule<DT, gridDim-1>& quad
            = Dune::QuadratureRules<DT, gridDim-1>::rule(it->type(), quadOrder);

        for (size_t pt=0; pt<quad.size(); pt++) {

            // Local position of the quadrature point
            const Dune::FieldVector<DT,gridDim>& quadPos = it->geometryInInside().global(quad[pt].position());

            const DT integrationElement = it->geometry().integrationElement(quad[pt].position());

            // The value of the local function
            RigidBodyMotion<field_type,dim> value = localGeodesicFEFunction.evaluate(quadPos);

            // Value of the Neumann data at the current position
            Dune::FieldVector<double,3> neumannValue;

            if (dynamic_cast<const VirtualGridViewFunction<GridView,Dune::FieldVector<double,3> >*>(neumannFunction_))
                dynamic_cast<const VirtualGridViewFunction<GridView,Dune::FieldVector<double,3> >*>(neumannFunction_)->evaluateLocal(element, quadPos, neumannValue);
            else
                neumannFunction_->evaluate(it->geometry().global(quad[pt].position()), neumannValue);

            // Only translational dofs are affected by the Neumann force
            for (size_t i=0; i<neumannValue.size(); i++)
                energy += thickness_ * (neumannValue[i] * value.r[i]) * quad[pt].weight() * integrationElement;

        }

    }

    return energy;
}

#endif   //#ifndef COSSERAT_ENERGY_LOCAL_STIFFNESS_HH

