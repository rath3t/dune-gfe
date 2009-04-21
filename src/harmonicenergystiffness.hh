#ifndef HARMONIC_ENERGY_LOCAL_STIFFNESS_HH
#define HARMONIC_ENERGY_LOCAL_STIFFNESS_HH

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

#if 0
    // types for matrics, vectors and boundary conditions
    typedef Dune::FieldMatrix<RT,m,m> MBlockType; // one entry in the stiffness matrix
    typedef Dune::FieldVector<RT,m> VBlockType;   // one entry in the global vectors
    typedef Dune::array<Dune::BoundaryConditions::Flags,m> BCBlockType;     // componentwise boundary conditions
#endif

#if 0
    //! Default Constructor
    HarmonicEnergyLocalStiffness ()
    {}
#endif

    /** \brief Assemble the energy for a single element */
    RT energy (const Entity& e,
               const std::vector<TargetSpace>& localSolution) const;

};

template <class GridView, class TargetSpace>
typename HarmonicEnergyLocalStiffness<GridView, TargetSpace>::RT HarmonicEnergyLocalStiffness<GridView, TargetSpace>::
energy(const Entity& element,
       const std::vector<TargetSpace>& localSolution) const
{
    RT energy = 0;
    
    LocalGeodesicFEFunction<gridDim, double, TargetSpace> localGeodesicFEFunction(element.type(), localSolution);

    int quadOrder = gridDim;

    const Dune::QuadratureRule<double, gridDim>& quad 
        = Dune::QuadratureRules<double, gridDim>::rule(element.type(), quadOrder);
    
    for (size_t pt=0; pt<quad.size(); pt++) {
        
        // Local position of the quadrature point
        const Dune::FieldVector<double,gridDim>& quadPos = quad[pt].position();
        
        const double integrationElement = element.geometry().integrationElement(quadPos);

        const Dune::FieldMatrix<double,gridDim,gridDim>& jacobianInverseTransposed = element.geometry().jacobianInverseTransposed(quadPos);
        
        double weight = quad[pt].weight() * integrationElement;

        // The derivative of the local function defined on the reference element
        Dune::FieldMatrix<double, TargetSpace::EmbeddedTangentVector::size, gridDim> referenceDerivative = localGeodesicFEFunction.evaluateDerivativeFD(quadPos);

        // The derivative of the function defined on the actual element
        Dune::FieldMatrix<double, TargetSpace::EmbeddedTangentVector::size, gridDim> derivative(0);

        for (int comp=0; comp<4; comp++)
            jacobianInverseTransposed.umv(referenceDerivative[comp], derivative[comp]);

        TargetSpace value = localGeodesicFEFunction.evaluate(quadPos);

        for (int i=0; i<gridDim; i++) {
            double dotproduct = 0;
            for (int j=0; j<4; j++)
                dotproduct += value[j]*derivative[j][i];
            assert(std::fabs(dotproduct) < 1e-6);
        }

#if 0
        double derivativenorm0 = 0;
        double derivativenorm1 = 0;
        for (int i=0; i<4; i++) {
            derivativenorm0 += derivative[i][0]*derivative[i][0];
            derivativenorm1 += derivative[i][1]*derivative[i][1];
        }
        std::cout << "Derivative norm: " << derivativenorm0 << ",  " << derivativenorm1 << std::endl;
#endif

        for (int comp=0; comp<4; comp++) {

            for (int dir=0; dir<gridDim; dir++)
                energy += weight * derivative[comp][dir] * derivative[comp][dir];

        }

    }

    return energy;
}

#endif

