#ifndef HARMONIC_ENERGY_LOCAL_STIFFNESS_HH
#define HARMONIC_ENERGY_LOCAL_STIFFNESS_HH

#include <dune/common/fmatrix.hh>
#include <dune/geometry/quadraturerules.hh>

#include "localgeodesicfestiffness.hh"
#include "localgeodesicfefunction.hh"

#ifdef HARMONIC_ENERGY_FD_GRADIENT
#warning Finite-difference approximation of the energy gradient
#endif

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

#ifndef HARMONIC_ENERGY_FD_GRADIENT
    // The finite difference gradient method is in the base class.
    // If the cpp macro is not set we overload it here.
    /** \brief Assemble the gradient of the energy functional on one element */
    virtual void assembleEmbeddedGradient(const Entity& element,
                                          const LocalFiniteElement& localFiniteElement,
                                          const std::vector<TargetSpace>& solution,
                                  std::vector<typename TargetSpace::EmbeddedTangentVector>& gradient) const;

    virtual void assembleGradient(const Entity& element,
                                  const LocalFiniteElement& localFiniteElement,
                                  const std::vector<TargetSpace>& localSolution,
                                  std::vector<typename TargetSpace::TangentVector>& localGradient) const;
#endif
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

#ifndef HARMONIC_ENERGY_FD_GRADIENT
template <class GridView, class LocalFiniteElement, class TargetSpace>
void HarmonicEnergyLocalStiffness<GridView, LocalFiniteElement, TargetSpace>::
assembleEmbeddedGradient(const Entity& element,
                         const LocalFiniteElement& localFiniteElement,
                         const std::vector<TargetSpace>& localSolution,
                         std::vector<typename TargetSpace::EmbeddedTangentVector>& localGradient) const
{
    typedef typename GridView::template Codim<0>::Entity::Geometry Geometry;

    // initialize gradient
    localGradient.resize(localSolution.size());
    std::fill(localGradient.begin(), localGradient.end(), typename TargetSpace::EmbeddedTangentVector(0));

    // Set up local gfe function from the  local coefficients
    typedef LocalGeodesicFEFunction<gridDim, double, LocalFiniteElement, TargetSpace> LocalGFEFunctionType;
    LocalGFEFunctionType localGeodesicFEFunction(localFiniteElement,localSolution);

    // I am not sure about the correct quadrature order
    int quadOrder = (element.type().isSimplex()) ? (localFiniteElement.localBasis().order()-1) * 2
                                                 : (localFiniteElement.localBasis().order()-1) * 2 * gridDim;

    // numerical quadrature loop
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
        typename LocalGFEFunctionType::DerivativeType derivative;

        for (size_t comp=0; comp<referenceDerivative.N(); comp++)
            jacobianInverseTransposed.mv(referenceDerivative[comp], derivative[comp]);

        // loop over all the element's degrees of freedom and compute the gradient wrt it
        for (size_t i=0; i<localSolution.size(); i++) {

            Tensor3<double, TargetSpace::EmbeddedTangentVector::dimension,TargetSpace::EmbeddedTangentVector::dimension,gridDim> referenceDerivativeDerivative;
#ifdef HARMONIC_ENERGY_FD_INNER_GRADIENT
#warning Using finite differences to compute the inner gradients!
            localGeodesicFEFunction.evaluateFDDerivativeOfGradientWRTCoefficient(quadPos, i, referenceDerivativeDerivative);
#else
            localGeodesicFEFunction.evaluateDerivativeOfGradientWRTCoefficient(quadPos, i, referenceDerivativeDerivative);
#endif

            // multiply the transformation from the reference element to the actual element
            Tensor3<double, TargetSpace::EmbeddedTangentVector::dimension,TargetSpace::EmbeddedTangentVector::dimension,gridDim> derivativeDerivative;
            for (int ii=0; ii<TargetSpace::EmbeddedTangentVector::dimension; ii++)
                for (int jj=0; jj<TargetSpace::EmbeddedTangentVector::dimension; jj++)
                    for (int kk=0; kk<gridDim; kk++) {
                        derivativeDerivative[ii][jj][kk] = 0;
                        for (int ll=0; ll<gridDim; ll++)
                            derivativeDerivative[ii][jj][kk] += referenceDerivativeDerivative[ii][jj][ll] * jacobianInverseTransposed[kk][ll];
                    }

            for (int j=0; j<derivative.rows; j++) {

                for (int k=0; k<derivative.cols; k++) {

                    for (int l=0; l<TargetSpace::EmbeddedTangentVector::dimension; l++)
                        localGradient[i][l] += weight*derivative[j][k] * derivativeDerivative[l][j][k];

                }

            }

        }

    }

}

template <class GridView, class LocalFiniteElement, class TargetSpace>
void HarmonicEnergyLocalStiffness<GridView, LocalFiniteElement, TargetSpace>::
assembleGradient(const Entity& element,
                 const LocalFiniteElement& localFiniteElement,
                 const std::vector<TargetSpace>& localSolution,
                 std::vector<typename TargetSpace::TangentVector>& localGradient) const
{
    std::vector<typename TargetSpace::EmbeddedTangentVector> embeddedLocalGradient;

    // first compute the gradient in embedded coordinates
    assembleEmbeddedGradient(element, localFiniteElement, localSolution, embeddedLocalGradient);

    // transform to coordinates on the tangent space
    localGradient.resize(embeddedLocalGradient.size());

    for (size_t i=0; i<localGradient.size(); i++)
        localSolution[i].orthonormalFrame().mv(embeddedLocalGradient[i], localGradient[i]);

}
#endif
#endif

