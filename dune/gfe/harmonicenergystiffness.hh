#ifndef HARMONIC_ENERGY_LOCAL_STIFFNESS_HH
#define HARMONIC_ENERGY_LOCAL_STIFFNESS_HH

#include <dune/common/fmatrix.hh>
#include <dune/geometry/quadraturerules.hh>

#include "localgeodesicfestiffness.hh"
#include "localgeodesicfefunction.hh"

template<class GridView, class LocalFiniteElement, class TargetSpace>
class HarmonicEnergyLocalStiffness
    : public LocalGeodesicFEStiffness<GridView,LocalFiniteElement,TargetSpace>
{
    // grid types
    typedef typename GridView::Grid::ctype DT;
    typedef typename TargetSpace::ctype RT;
    typedef typename GridView::template Codim<0>::Entity Entity;

    // some other sizes
    enum {gridDim=GridView::dimension};

public:

    //! Dimension of a tangent space
    enum { blocksize = TargetSpace::TangentVector::dimension };

    /** \brief Assemble the energy for a single element */
    RT energy (const Entity& e,
               const LocalFiniteElement& localFiniteElement,
               const std::vector<TargetSpace>& localSolution) const;

};

template <class GridView, class LocalFiniteElement, class TargetSpace>
typename HarmonicEnergyLocalStiffness<GridView, LocalFiniteElement, TargetSpace>::RT
HarmonicEnergyLocalStiffness<GridView, LocalFiniteElement, TargetSpace>::
energy(const Entity& element,
       const LocalFiniteElement& localFiniteElement,
       const std::vector<TargetSpace>& localSolution) const
{
    assert(element.type() == localFiniteElement.type());
    typedef typename GridView::template Codim<0>::Entity::Geometry Geometry;

    RT energy = 0;
    typedef LocalGeodesicFEFunction<gridDim, double, LocalFiniteElement, TargetSpace> LocalGFEFunctionType;
    LocalGFEFunctionType localGeodesicFEFunction(localFiniteElement,localSolution);

    int quadOrder = (element.type().isSimplex()) ? (localFiniteElement.localBasis().order()-1) * 2
                                                 : localFiniteElement.localBasis().order() * 2 * gridDim;



    const Dune::QuadratureRule<double, gridDim>& quad
        = Dune::QuadratureRules<double, gridDim>::rule(element.type(), quadOrder);

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

