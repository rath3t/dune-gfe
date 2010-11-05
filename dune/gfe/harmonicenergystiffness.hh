#ifndef HARMONIC_ENERGY_LOCAL_STIFFNESS_HH
#define HARMONIC_ENERGY_LOCAL_STIFFNESS_HH

//#define HARMONIC_ENERGY_FD_GRADIENT

#include <dune/common/fmatrix.hh>
#include <dune/grid/common/quadraturerules.hh>

#include "localgeodesicfestiffness.hh"
#include "localgeodesicfefunction.hh"

template<class GridView, class TargetSpace>
class HarmonicEnergyLocalStiffness 
    : public LocalGeodesicFEStiffness<GridView,TargetSpace>
{
    // grid types
    typedef typename GridView::Grid::ctype DT;
    typedef typename TargetSpace::ctype RT;
    typedef typename GridView::template Codim<0>::Entity Entity;
    
    // some other sizes
    enum {gridDim=GridView::dimension};

public:
    
    //! Dimension of a tangent space
    enum { blocksize = TargetSpace::TangentVector::size };

    /** \brief Assemble the energy for a single element */
    RT energy (const Entity& e,
               const std::vector<TargetSpace>& localSolution) const;

#ifndef HARMONIC_ENERGY_FD_GRADIENT
    /** \brief Assemble the gradient of the energy functional on one element */
    virtual void assembleEmbeddedGradient(const Entity& element,
                                  const std::vector<TargetSpace>& solution,
                                  std::vector<typename TargetSpace::EmbeddedTangentVector>& gradient) const;
                                    
    virtual void assembleGradient(const Entity& element,
                                  const std::vector<TargetSpace>& localSolution,
                                  std::vector<typename TargetSpace::TangentVector>& localGradient) const;
#endif
};

template <class GridView, class TargetSpace>
typename HarmonicEnergyLocalStiffness<GridView, TargetSpace>::RT HarmonicEnergyLocalStiffness<GridView, TargetSpace>::
energy(const Entity& element,
       const std::vector<TargetSpace>& localSolution) const
{
    RT energy = 0;

    assert(element.type().isSimplex());
    
    LocalGeodesicFEFunction<gridDim, double, TargetSpace> localGeodesicFEFunction(localSolution);

    int quadOrder = 1;//gridDim;

    const Dune::QuadratureRule<double, gridDim>& quad 
        = Dune::QuadratureRules<double, gridDim>::rule(element.type(), quadOrder);
    
    for (size_t pt=0; pt<quad.size(); pt++) {
        
        // Local position of the quadrature point
        const Dune::FieldVector<double,gridDim>& quadPos = quad[pt].position();
        
        const double integrationElement = element.geometry().integrationElement(quadPos);

        const Dune::FieldMatrix<double,gridDim,gridDim>& jacobianInverseTransposed = element.geometry().jacobianInverseTransposed(quadPos);
        
        double weight = quad[pt].weight() * integrationElement;

        // The derivative of the local function defined on the reference element
        Dune::FieldMatrix<double, TargetSpace::EmbeddedTangentVector::size, gridDim> referenceDerivative = localGeodesicFEFunction.evaluateDerivative(quadPos);

        // The derivative of the function defined on the actual element
        Dune::FieldMatrix<double, TargetSpace::EmbeddedTangentVector::size, gridDim> derivative(0);

        for (size_t comp=0; comp<referenceDerivative.N(); comp++)
            jacobianInverseTransposed.umv(referenceDerivative[comp], derivative[comp]);

#if 0
        // old debugging: the image of the derivative mapping must be tangent to the TargetSpace
        TargetSpace value = localGeodesicFEFunction.evaluate(quadPos);

        for (int i=0; i<gridDim; i++) {
            double dotproduct = 0;
            for (int j=0; j<4; j++)
                dotproduct += value[j]*derivative[j][i];
            assert(std::fabs(dotproduct) < 1e-6);
        }
#endif

#if 0
        std::cout << "Derivative norm squared: " << derivative.frobenius_norm2() << std::endl;
#endif

        // Add the local energy density
        // The Frobenius norm is the correct norm here if the metric of TargetSpace is the identity.
        // (And if the metric of the domain space is the identity, which it always is here.)
        energy += weight * derivative.frobenius_norm2();

    }

    return 0.5 * energy;
}

#ifndef HARMONIC_ENERGY_FD_GRADIENT
template <class GridView, class TargetSpace>
void HarmonicEnergyLocalStiffness<GridView, TargetSpace>::
assembleEmbeddedGradient(const Entity& element,
                 const std::vector<TargetSpace>& localSolution,
                 std::vector<typename TargetSpace::EmbeddedTangentVector>& localGradient) const
{
    // initialize gradient
    localGradient.resize(localSolution.size());
    std::fill(localGradient.begin(), localGradient.end(), typename TargetSpace::EmbeddedTangentVector(0));

    // Set up local gfe function from the  local coefficients
    LocalGeodesicFEFunction<gridDim, double, TargetSpace> localGeodesicFEFunction(localSolution);

    // I am not sure about the correct quadrature order
    int quadOrder = 1;//gridDim;

    // numerical quadrature loop
    const Dune::QuadratureRule<double, gridDim>& quad 
        = Dune::QuadratureRules<double, gridDim>::rule(element.type(), quadOrder);
    
    for (size_t pt=0; pt<quad.size(); pt++) {
        
        // Local position of the quadrature point
        const Dune::FieldVector<double,gridDim>& quadPos = quad[pt].position();
        
        const double integrationElement = element.geometry().integrationElement(quadPos);

        const Dune::FieldMatrix<double,gridDim,gridDim>& jacobianInverseTransposed = element.geometry().jacobianInverseTransposed(quadPos);
        
        double weight = quad[pt].weight() * integrationElement;

        // The derivative of the local function defined on the reference element
        Dune::FieldMatrix<double, TargetSpace::EmbeddedTangentVector::size, gridDim> referenceDerivative = localGeodesicFEFunction.evaluateDerivative(quadPos);

        // The derivative of the function defined on the actual element
        Dune::FieldMatrix<double, TargetSpace::EmbeddedTangentVector::size, gridDim> derivative;

        for (size_t comp=0; comp<referenceDerivative.N(); comp++)
            jacobianInverseTransposed.mv(referenceDerivative[comp], derivative[comp]);
        
        // loop over all the element's degrees of freedom and compute the gradient wrt it
        for (size_t i=0; i<localSolution.size(); i++) {
         
            Tensor3<double, TargetSpace::EmbeddedTangentVector::size, gridDim,TargetSpace::EmbeddedTangentVector::size> derivativeDerivative;
            localGeodesicFEFunction.evaluateDerivativeOfGradientWRTCoefficient(quadPos, i, derivativeDerivative);
        
            for (int j=0; j<derivative.rows; j++) {
                
                for (int k=0; k<derivative.cols; k++) {
                    
                    localGradient[i].axpy(weight*derivative[j][k], derivativeDerivative[j][k]);
                    
                }
                
            }
            
        }

	}

}

template <class GridView, class TargetSpace>
void HarmonicEnergyLocalStiffness<GridView, TargetSpace>::
assembleGradient(const Entity& element,
                 const std::vector<TargetSpace>& localSolution,
                 std::vector<typename TargetSpace::TangentVector>& localGradient) const
{
    std::vector<typename TargetSpace::EmbeddedTangentVector> embeddedLocalGradient;

    // first compute the gradient in embedded coordinates
    assembleEmbeddedGradient(element, localSolution, embeddedLocalGradient);

    // transform to coordinates on the tangent space
    localGradient.resize(embeddedLocalGradient.size());

    for (size_t i=0; i<localGradient.size(); i++)
        localSolution[i].orthonormalFrame().mv(embeddedLocalGradient[i], localGradient[i]);

}
#endif
#endif

