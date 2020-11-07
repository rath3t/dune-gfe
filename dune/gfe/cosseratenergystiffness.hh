#ifndef COSSERAT_ENERGY_LOCAL_STIFFNESS_HH
#define COSSERAT_ENERGY_LOCAL_STIFFNESS_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/parametertree.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/fufem/boundarypatch.hh>

#include <dune/gfe/localenergy.hh>
#include <dune/gfe/mixedlocalgeodesicfestiffness.hh>
#ifdef PROJECTED_INTERPOLATION
#include <dune/gfe/localprojectedfefunction.hh>
#else
#include "localgeodesicfefunction.hh"
#endif
#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/tensor3.hh>
#include <dune/gfe/orthogonalmatrix.hh>
#include <dune/gfe/cosseratstrain.hh>

#define DONT_USE_CURL

//#define QUADRATIC_MEMBRANE_ENERGY

/** \brief Get LocalFiniteElements from a localView, for different tree depths of the local view
 *
 * We instantiate the CosseratEnergyLocalStiffness class with two different kinds of Basis:
 * A scalar one and a composite one that combines two scalar ones.  But code for accessing the
 * finite elements in the basis tree only work for one kind of basis, not for the other.
 * To allow both kinds of basis in a single class we need this trickery below.
 */
template <class Basis, std::size_t i>
class LocalFiniteElementFactory
{
public:
  static auto get(const typename Basis::LocalView& localView,
           std::integral_constant<std::size_t, i> iType)
    -> decltype(localView.tree().child(iType,0).finiteElement())
  {
    return localView.tree().child(iType,0).finiteElement();
  }
};

/** \brief Specialize for scalar bases, here we cannot call tree().child() */
template <class GridView, int order, std::size_t i>
class LocalFiniteElementFactory<Dune::Functions::LagrangeBasis<GridView,order>,i>
{
public:
  static auto get(const typename Dune::Functions::LagrangeBasis<GridView,order>::LocalView& localView,
           std::integral_constant<std::size_t, i> iType)
    -> decltype(localView.tree().finiteElement())
  {
    return localView.tree().finiteElement();
  }
};

template<class Basis, int dim, class field_type=double>
class CosseratEnergyLocalStiffness
    : public Dune::GFE::LocalEnergy<Basis,RigidBodyMotion<field_type,dim> >,
      public MixedLocalGeodesicFEStiffness<Basis,
                                           RealTuple<field_type,dim>,
                                           Rotation<field_type,dim> >
{
    // grid types
    typedef typename Basis::GridView GridView;
    typedef typename GridView::ctype DT;
    typedef RigidBodyMotion<field_type,dim> TargetSpace;
    typedef typename TargetSpace::ctype RT;
    typedef typename GridView::template Codim<0>::Entity Entity;

    // some other sizes
    enum {gridDim=GridView::dimension};
    enum {dimworld=GridView::dimensionworld};

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

public:

    /** \brief Constructor with a set of material parameters
     * \param parameters The material parameters
     */
    CosseratEnergyLocalStiffness(const Dune::ParameterTree& parameters,
                                 const BoundaryPatch<GridView>* neumannBoundary,
                                 const std::function<Dune::FieldVector<double,3>(Dune::FieldVector<double,dimworld>)> neumannFunction,
                                 const std::function<Dune::FieldVector<double,3>(Dune::FieldVector<double,dimworld>)> volumeLoad)
    : neumannBoundary_(neumannBoundary),
      neumannFunction_(neumannFunction)
    {
        // The shell thickness
        thickness_ = parameters.template get<double>("thickness");

        // Lame constants
        mu_ = parameters.template get<double>("mu");
        lambda_ = parameters.template get<double>("lambda");

        // Cosserat couple modulus
        mu_c_ = parameters.template get<double>("mu_c");

        // Length scale parameter
        L_c_ = parameters.template get<double>("L_c");

        // Curvature exponent
        q_ = parameters.template get<double>("q");

        // Shear correction factor
        kappa_ = parameters.template get<double>("kappa");
    }

    /** \brief Assemble the energy for a single element */
    RT energy (const typename Basis::LocalView& localView,
               const std::vector<TargetSpace>& localSolution) const override;

    /** \brief Assemble the energy for a single element */
    RT energy (const typename Basis::LocalView& localView,
               const std::vector<RealTuple<field_type,dim> >& localDisplacementConfiguration,
               const std::vector<Rotation<field_type,dim> >& localOrientationConfiguration) const override;

    /** \brief The energy \f$ W_{mp}(\overline{U}) \f$, as written in
     * the first equation of (4.4) in Neff's paper
     */
    RT quadraticMembraneEnergy(const Dune::GFE::CosseratStrain<field_type,3,gridDim>& U) const
    {
        Dune::FieldMatrix<field_type,3,3> UMinus1 = U.matrix();
        for (int i=0; i<dim; i++)
            UMinus1[i][i] -= 1;

        return mu_ * Dune::GFE::sym(UMinus1).frobenius_norm2()
                + mu_c_ * Dune::GFE::skew(UMinus1).frobenius_norm2()
                + (mu_*lambda_)/(2*mu_ + lambda_) * Dune::GFE::traceSquared(Dune::GFE::sym(UMinus1));
    }

    /** \brief The energy \f$ W_{mp}(\overline{U}) \f$, as written in
     * the second equation of (4.4) in Neff's paper
     */
    RT longQuadraticMembraneEnergy(const Dune::GFE::CosseratStrain<field_type,3,gridDim>& U) const
    {
        RT result = 0;

        // shear-stretch energy
        Dune::FieldMatrix<field_type,dim-1,dim-1> sym2x2;
        for (int i=0; i<dim-1; i++)
            for (int j=0; j<dim-1; j++)
                sym2x2[i][j] = 0.5 * (U.matrix()[i][j] + U.matrix()[j][i]) - (i==j);

        result += mu_ * sym2x2.frobenius_norm2();

        // first order drill energy
        Dune::FieldMatrix<field_type,dim-1,dim-1> skew2x2;
        for (int i=0; i<dim-1; i++)
            for (int j=0; j<dim-1; j++)
                skew2x2[i][j] = 0.5 * (U.matrix()[i][j] - U.matrix()[j][i]);

        result += mu_c_ * skew2x2.frobenius_norm2();


        // classical transverse shear energy
        result += kappa_ * (mu_ + mu_c_)/2 * (U.matrix()[2][0]*U.matrix()[2][0] + U.matrix()[2][1]*U.matrix()[2][1]);

        // elongational stretch energy
        result += mu_*lambda_ / (2*mu_ + lambda_) * traceSquared(sym2x2);

        return result;
    }

    /** \brief Energy for large-deformation problems (private communication by Patrizio Neff)
     */
    RT nonquadraticMembraneEnergy(const Dune::GFE::CosseratStrain<field_type,3,gridDim>& U) const
    {
        Dune::FieldMatrix<field_type,3,3> UMinus1 = U.matrix();
        for (int i=0; i<dim; i++)
            UMinus1[i][i] -= 1;

        RT detU = U.determinant();

        return mu_ * Dune::GFE::sym(UMinus1).frobenius_norm2() + mu_c_ * Dune::GFE::skew(UMinus1).frobenius_norm2()
                + (mu_*lambda_)/(2*mu_ + lambda_) * 0.5 * ((detU-1)*(detU-1) + (1.0/detU -1)*(1.0/detU -1));
    }

    /** \brief The energy \f$ W_{mp}(\overline{U}) \f$, as written in
     * the second equation of (4.4) in Neff's paper
     */
    RT longNonquadraticMembraneEnergy(const Dune::GFE::CosseratStrain<field_type,3,gridDim>& U) const
    {
        RT result = 0;

        // shear-stretch energy
        Dune::FieldMatrix<field_type,dim-1,dim-1> sym2x2;
        for (int i=0; i<dim-1; i++)
            for (int j=0; j<dim-1; j++)
                sym2x2[i][j] = 0.5 * (U.matrix()[i][j] + U.matrix()[j][i]) - (i==j);

        result += mu_ * sym2x2.frobenius_norm2();

        // first order drill energy
        Dune::FieldMatrix<field_type,dim-1,dim-1> skew2x2;
        for (int i=0; i<dim-1; i++)
            for (int j=0; j<dim-1; j++)
                skew2x2[i][j] = 0.5 * (U.matrix()[i][j] - U.matrix()[j][i]);

        result += mu_c_ * skew2x2.frobenius_norm2();


        // classical transverse shear energy
        result += kappa_ * (mu_ + mu_c_)/2 * (U.matrix()[2][0]*U.matrix()[2][0] + U.matrix()[2][1]*U.matrix()[2][1]);

        // elongational stretch energy
        RT detU = U.determinant();
        result += (mu_*lambda_)/(2*mu_ + lambda_) * 0.5 * ((detU-1)*(detU-1) + (1.0/detU -1)*(1.0/detU -1));

        return result;
    }

    RT curvatureEnergy(const Tensor3<field_type,3,3,gridDim>& DR) const
    {
        using std::pow;
#ifdef DONT_USE_CURL
        return mu_ * pow(L_c_ * L_c_ * DR.frobenius_norm2(),q_/2.0);
#else
        return mu_ * pow(L_c_ * L_c_ * curl(DR).frobenius_norm2(),q_/2.0);
#endif
    }

    RT bendingEnergy(const Dune::FieldMatrix<field_type,dim,dim>& R, const Tensor3<field_type,3,3,gridDim>& DR) const
    {
        // left-multiply the derivative of the third director (in DR[][2][]) with R^T
        Dune::FieldMatrix<field_type,3,3> RT_DR3(0);
        for (int i=0; i<3; i++)
            for (int j=0; j<gridDim; j++)
                for (int k=0; k<3; k++)
                    RT_DR3[i][j] += R[k][i] * DR[k][2][j];

        return mu_ * Dune::GFE::sym(RT_DR3).frobenius_norm2()
               + mu_c_ * Dune::GFE::skew(RT_DR3).frobenius_norm2()
               + mu_*lambda_/(2*mu_+lambda_) * Dune::GFE::traceSquared(RT_DR3);
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
    const std::function<Dune::FieldVector<double,3>(Dune::FieldVector<double,dimworld>)> neumannFunction_;

    /** \brief The function implementing a volume load */
    const std::function<Dune::FieldVector<double,3>(Dune::FieldVector<double,dimworld>)> volumeLoad_;
};

template <class Basis, int dim, class field_type>
typename CosseratEnergyLocalStiffness<Basis,dim,field_type>::RT
CosseratEnergyLocalStiffness<Basis,dim,field_type>::
energy(const typename Basis::LocalView& localView,
       const std::vector<RigidBodyMotion<field_type,dim> >& localSolution) const
{
    RT energy = 0;

    auto element = localView.element();

    using namespace Dune::TypeTree::Indices;
    const auto& localFiniteElement = LocalFiniteElementFactory<Basis,0>::get(localView,_0);
#ifdef PROJECTED_INTERPOLATION
    typedef Dune::GFE::LocalProjectedFEFunction<gridDim, DT, decltype(localFiniteElement), TargetSpace> LocalGFEFunctionType;
#else
    typedef LocalGeodesicFEFunction<gridDim, DT, decltype(localFiniteElement), TargetSpace> LocalGFEFunctionType;
#endif
    LocalGFEFunctionType localGeodesicFEFunction(localFiniteElement,localSolution);

    int quadOrder = (element.type().isSimplex()) ? localFiniteElement.localBasis().order()
                                                 : localFiniteElement.localBasis().order() * gridDim;

    const auto& quad = Dune::QuadratureRules<DT, gridDim>::rule(element.type(), quadOrder);

    for (size_t pt=0; pt<quad.size(); pt++) {

        // Local position of the quadrature point
        const Dune::FieldVector<DT,gridDim>& quadPos = quad[pt].position();

        const DT integrationElement = element.geometry().integrationElement(quadPos);

        const auto jacobianInverseTransposed = element.geometry().jacobianInverseTransposed(quadPos);

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
        static_assert(dim>=gridDim, "Codim of the grid must be nonnegative");

        //
        Dune::FieldMatrix<field_type,dim,dim> R;
        value.q.matrix(R);

        Dune::GFE::CosseratStrain<field_type,dim,gridDim> U(derivative,R);

        //////////////////////////////////////////////////////////
        //  Compute the derivative of the rotation
        //  Note: we need it in matrix coordinates
        //////////////////////////////////////////////////////////

        Tensor3<field_type,3,3,gridDim> DR = value.quaternionTangentToMatrixTangent(derivative);

        // Add the local energy density
        if (gridDim==2) {
#ifdef QUADRATIC_MEMBRANE_ENERGY
            //energy += weight * thickness_ * quadraticMembraneEnergy(U.matrix());
            energy += weight * thickness_ * longQuadraticMembraneEnergy(U);
#else
            //energy += weight * thickness_ * nonquadraticMembraneEnergy(U);
            energy += weight * thickness_ * longNonquadraticMembraneEnergy(U);
#endif
            energy += weight * thickness_ * curvatureEnergy(DR);
            energy += weight * std::pow(thickness_,3) / 12.0 * bendingEnergy(R,DR);
        } else if (gridDim==3) {
            energy += weight * quadraticMembraneEnergy(U);
            energy += weight * curvatureEnergy(DR);
        } else
            DUNE_THROW(Dune::NotImplemented, "CosseratEnergyStiffness for 1d grids");

        ///////////////////////////////////////////////////////////
        // Volume load contribution
        ///////////////////////////////////////////////////////////

        if (not volumeLoad_)
            continue;

        // Value of the volume load density at the current position
        auto volumeLoadDensity = volumeLoad_(element.geometry().global(quad[pt].position()));

        // Only translational dofs are affected by the volume load
        for (size_t i=0; i<volumeLoadDensity.size(); i++)
            energy -= thickness_ * (volumeLoadDensity[i] * value.r[i]) * quad[pt].weight() * integrationElement;
    }


    //////////////////////////////////////////////////////////////////////////////
    //   Assemble boundary contributions
    //////////////////////////////////////////////////////////////////////////////

    if (not neumannFunction_)
        return energy;

    for (auto&& it : intersections(neumannBoundary_->gridView(),element) )
    {
        if (not neumannBoundary_ or not neumannBoundary_->contains(it))
            continue;

        const Dune::QuadratureRule<DT, gridDim-1>& quad
            = Dune::QuadratureRules<DT, gridDim-1>::rule(it.type(), quadOrder);

        for (size_t pt=0; pt<quad.size(); pt++) {

            // Local position of the quadrature point
            const Dune::FieldVector<DT,gridDim>& quadPos = it.geometryInInside().global(quad[pt].position());

            const DT integrationElement = it.geometry().integrationElement(quad[pt].position());

            // The value of the local function
            RigidBodyMotion<field_type,dim> value = localGeodesicFEFunction.evaluate(quadPos);

            // Value of the Neumann data at the current position
            auto neumannValue = neumannFunction_(it.geometry().global(quad[pt].position()));

            // Only translational dofs are affected by the Neumann force
            for (size_t i=0; i<neumannValue.size(); i++)
                energy -= thickness_ * (neumannValue[i] * value.r[i]) * quad[pt].weight() * integrationElement;

        }

    }

    return energy;
}

template <class Basis, int dim, class field_type>
typename CosseratEnergyLocalStiffness<Basis,dim,field_type>::RT
CosseratEnergyLocalStiffness<Basis,dim,field_type>::
energy(const typename Basis::LocalView& localView,
       const std::vector<RealTuple<field_type,dim> >& localDeformationConfiguration,
       const std::vector<Rotation<field_type,dim> >& localOrientationConfiguration) const
{
    auto element = localView.element();

    RT energy = 0;

    using namespace Dune::TypeTree::Indices;
    const auto& deformationLocalFiniteElement = LocalFiniteElementFactory<Basis,0>::get(localView,_0);
    const auto& orientationLocalFiniteElement = LocalFiniteElementFactory<Basis,1>::get(localView,_1);

#ifdef PROJECTED_INTERPOLATION
    typedef Dune::GFE::LocalProjectedFEFunction<gridDim, DT, decltype(deformationLocalFiniteElement), RealTuple<field_type,dim> >
    LocalDeformationGFEFunctionType;
#else
    typedef LocalGeodesicFEFunction<gridDim, DT, decltype(deformationLocalFiniteElement), RealTuple<field_type,dim> >
    LocalDeformationGFEFunctionType;
#endif
    LocalDeformationGFEFunctionType localDeformationGFEFunction(deformationLocalFiniteElement,localDeformationConfiguration);

#ifdef PROJECTED_INTERPOLATION
    typedef Dune::GFE::LocalProjectedFEFunction<gridDim, DT, decltype(orientationLocalFiniteElement), Rotation<field_type,dim> > LocalOrientationGFEFunctionType;
#else
    typedef LocalGeodesicFEFunction<gridDim, DT, decltype(orientationLocalFiniteElement), Rotation<field_type,dim> > LocalOrientationGFEFunctionType;
#endif
    LocalOrientationGFEFunctionType localOrientationGFEFunction(orientationLocalFiniteElement,localOrientationConfiguration);

    // \todo Implement smarter quadrature rule selection for more efficiency, i.e., less evaluations of the Rotation GFE function
    int quadOrder = deformationLocalFiniteElement.localBasis().order() * ((element.type().isSimplex()) ? 1 : gridDim);

    const auto& quad = Dune::QuadratureRules<DT, gridDim>::rule(element.type(), quadOrder);

    for (size_t pt=0; pt<quad.size(); pt++)
    {
        // Local position of the quadrature point
        const Dune::FieldVector<DT,gridDim>& quadPos = quad[pt].position();

        const DT integrationElement = element.geometry().integrationElement(quadPos);

        const auto jacobianInverseTransposed = element.geometry().jacobianInverseTransposed(quadPos);

        DT weight = quad[pt].weight() * integrationElement;

        // The value of the local deformation
        RealTuple<field_type,dim> deformationValue = localDeformationGFEFunction.evaluate(quadPos);
        Rotation<field_type,dim>  orientationValue = localOrientationGFEFunction.evaluate(quadPos);

        // The derivative of the local function defined on the reference element
        typename LocalDeformationGFEFunctionType::DerivativeType deformationReferenceDerivative = localDeformationGFEFunction.evaluateDerivative(quadPos,deformationValue);
        typename LocalOrientationGFEFunctionType::DerivativeType orientationReferenceDerivative = localOrientationGFEFunction.evaluateDerivative(quadPos,orientationValue);

        // The derivative of the function defined on the actual element
        typename LocalDeformationGFEFunctionType::DerivativeType deformationDerivative;
        typename LocalOrientationGFEFunctionType::DerivativeType orientationDerivative;

        for (size_t comp=0; comp<deformationReferenceDerivative.N(); comp++)
            jacobianInverseTransposed.mv(deformationReferenceDerivative[comp], deformationDerivative[comp]);

        for (size_t comp=0; comp<orientationReferenceDerivative.N(); comp++)
            jacobianInverseTransposed.mv(orientationReferenceDerivative[comp], orientationDerivative[comp]);

        /////////////////////////////////////////////////////////
        // compute U, the Cosserat strain
        /////////////////////////////////////////////////////////
        static_assert(dim>=gridDim, "Codim of the grid must be nonnegative");

        //
        Dune::FieldMatrix<field_type,dim,dim> R;
        orientationValue.matrix(R);

        Dune::GFE::CosseratStrain<field_type,dim,gridDim> U(deformationDerivative,R);

        //////////////////////////////////////////////////////////
        //  Compute the derivative of the rotation
        //  Note: we need it in matrix coordinates
        //////////////////////////////////////////////////////////

        Tensor3<field_type,3,3,gridDim> DR = orientationValue.quaternionTangentToMatrixTangent(orientationDerivative);

        // Add the local energy density
        if (gridDim==2) {
#ifdef QUADRATIC_MEMBRANE_ENERGY
            //energy += weight * thickness_ * quadraticMembraneEnergy(U.matrix());
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

    for (auto&& it : intersections(neumannBoundary_->gridView(),element) )
    {
        if (not neumannBoundary_ or not neumannBoundary_->contains(it))
            continue;

        const auto& quad = Dune::QuadratureRules<DT, gridDim-1>::rule(it.type(), quadOrder);

        for (size_t pt=0; pt<quad.size(); pt++) {

            // Local position of the quadrature point
            const Dune::FieldVector<DT,gridDim>& quadPos = it.geometryInInside().global(quad[pt].position());

            const DT integrationElement = it.geometry().integrationElement(quad[pt].position());

            // The value of the local function
            RealTuple<field_type,dim> deformationValue = localDeformationGFEFunction.evaluate(quadPos);

            // Value of the Neumann data at the current position
            auto neumannValue = neumannFunction_(it.geometry().global(quad[pt].position()));

            // Only translational dofs are affected by the Neumann force
            for (size_t i=0; i<neumannValue.size(); i++)
                energy += thickness_ * (neumannValue[i] * deformationValue.globalCoordinates()[i]) * quad[pt].weight() * integrationElement;

        }

    }

    return energy;
}

#endif   //#ifndef COSSERAT_ENERGY_LOCAL_STIFFNESS_HH

