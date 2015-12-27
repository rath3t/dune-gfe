#ifndef HARMONIC_ENERGY_LOCAL_STIFFNESS_HH
#define HARMONIC_ENERGY_LOCAL_STIFFNESS_HH

#include <dune/common/fmatrix.hh>
#include <dune/geometry/quadraturerules.hh>

#include "localgeodesicfestiffness.hh"
#include "localgeodesicfefunction.hh"

template<class Basis, class TargetSpace>
class HarmonicEnergyLocalStiffness
    : public LocalGeodesicFEStiffness<Basis,TargetSpace>
{
    // grid types
    typedef typename Basis::GridView GridView;
    typedef typename GridView::ctype DT;
    typedef typename TargetSpace::ctype RT;
    typedef typename GridView::template Codim<0>::Entity Entity;

    // some other sizes
    enum {gridDim=GridView::dimension};

public:

    //! Dimension of a tangent space
    enum { blocksize = TargetSpace::TangentVector::dimension };

    /** \brief Assemble the energy for a single element */
    RT energy (const typename Basis::LocalView& localView,
               const std::vector<TargetSpace>& localSolution) const override;

};

template <class Basis, class TargetSpace>
typename HarmonicEnergyLocalStiffness<Basis, TargetSpace>::RT
HarmonicEnergyLocalStiffness<Basis, TargetSpace>::
energy(const typename Basis::LocalView& localView,
       const std::vector<TargetSpace>& localSolution) const
{
    typedef typename GridView::template Codim<0>::Entity::Geometry Geometry;

    RT energy = 0;

    const auto& localFiniteElement = localView.tree().finiteElement();
    typedef LocalGeodesicFEFunction<gridDim, double, decltype(localFiniteElement), TargetSpace> LocalGFEFunctionType;
    LocalGFEFunctionType localGeodesicFEFunction(localFiniteElement,localSolution);

    int quadOrder = (localFiniteElement.type().isSimplex()) ? (localFiniteElement.localBasis().order()-1) * 2
                                                 : localFiniteElement.localBasis().order() * 2 * gridDim;

    const auto element = localView.element();

    const Dune::QuadratureRule<double, gridDim>& quad
        = Dune::QuadratureRules<double, gridDim>::rule(localFiniteElement.type(), quadOrder);

    for (size_t pt=0; pt<quad.size(); pt++) {

        // Local position of the quadrature point
        const Dune::FieldVector<double,gridDim>& quadPos = quad[pt].position();

        const double integrationElement = element.geometry().integrationElement(quadPos);

        const typename Geometry::JacobianInverseTransposed& jacobianInverseTransposed = element.geometry().jacobianInverseTransposed(quadPos);

        double weight = quad[pt].weight() * integrationElement;

        // The derivative of the local function defined on the reference element
        typename LocalGFEFunctionType::DerivativeType referenceDerivative = localGeodesicFEFunction.evaluateDerivative(quadPos);

        // The derivative of the function defined on the actual element
        typename LocalGFEFunctionType::DerivativeType derivative(0);

        for (size_t comp=0; comp<referenceDerivative.N(); comp++)
            jacobianInverseTransposed.umv(referenceDerivative[comp], derivative[comp]);

        // Add the local energy density
        // The Frobenius norm is the correct norm here if the metric of TargetSpace is the identity.
        // (And if the metric of the domain space is the identity, which it always is here.)
        energy += weight * derivative.frobenius_norm2();

    }

    return 0.5 * energy;
}

#endif

