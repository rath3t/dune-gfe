#ifndef HARMONIC_ENERGY_LOCAL_STIFFNESS_HH
#define HARMONIC_ENERGY_LOCAL_STIFFNESS_HH

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>
#include <dune/disc/operators/localstiffness.hh>

template<class GridView, class TargetSpace>
class HarmonicEnergyLocalStiffness 
    : public Dune::LocalGeodesicFEStiffness<GridView,TargetSpace>
{
    // grid types
    typedef typename GridView::Grid::ctype DT;
    typedef typename GridView::template Codim<0>::Entity Entity;
    typedef typename GridView::template Codim<0>::EntityPointer EntityPointer;
    
    // some other sizes
    enum {gridDim=GridView::dimension};

public:
    
    //! Dimension of a tangent space
    enum { blocksize = TargetSpace::TangentVector::size };

    // types for matrics, vectors and boundary conditions
    typedef Dune::FieldMatrix<RT,m,m> MBlockType; // one entry in the stiffness matrix
    typedef Dune::FieldVector<RT,m> VBlockType;   // one entry in the global vectors
    typedef Dune::array<Dune::BoundaryConditions::Flags,m> BCBlockType;     // componentwise boundary conditions

    //! Default Constructor
    HarmonicEnergyLocalStiffness ()
    {}

    /** \brief assemble local stiffness matrix for given element
    */
    void assemble (const Entity& e, 
                   const Dune::BlockVector<Dune::FieldVector<double, 6> >& localSolution,
                   int k=1)
    {
        DUNE_THROW(Dune::NotImplemented, "!");
    }

    void assembleBoundaryCondition (const Entity& e, int k=1)
    {
        DUNE_THROW(Dune::NotImplemented, "!");
    }

    
    RT energy (const Entity& e,
               const Dune::array<RigidBodyMotion<3>,2>& localSolution) const;

    /** \brief Assemble the element gradient of the energy functional */
    void assembleGradient(const Entity& element,
                          const Dune::array<RigidBodyMotion<3>,2>& solution,
                          const Dune::array<RigidBodyMotion<3>,2>& referenceConfiguration,
                          Dune::array<Dune::FieldVector<double,6>, 2>& gradient) const;
    
};

template <class GridType, class RT>
RT RodLocalStiffness<GridType, RT>::
energy(const Entity& element,
       const Dune::array<RigidBodyMotion<3>,2>& localSolution) const
{
    RT energy = 0;
    
    // ///////////////////////////////////////////////////////////////////////////////
    //   The following two loops are a reduced integration scheme.  We integrate
    //   the transverse shear and extensional energy with a first-order quadrature
    //   formula, even though it should be second order.  This prevents shear-locking.
    // ///////////////////////////////////////////////////////////////////////////////

    const Dune::QuadratureRule<double, 1>& shearingQuad 
        = Dune::QuadratureRules<double, 1>::rule(element.type(), shearQuadOrder);
    
    for (size_t pt=0; pt<shearingQuad.size(); pt++) {
        
        // Local position of the quadrature point
        const Dune::FieldVector<double,1>& quadPos = shearingQuad[pt].position();
        
        const double integrationElement = element.geometry().integrationElement(quadPos);
        
        double weight = shearingQuad[pt].weight() * integrationElement;
        
        Dune::FieldVector<double,6> strain = getStrain(localSolution, element, quadPos);
        
        // The reference strain
        Dune::FieldVector<double,6> referenceStrain = getStrain(localReferenceConfiguration, element, quadPos);
        
        for (int i=0; i<3; i++)
            energy += weight * 0.5 * A_[i] * (strain[i] - referenceStrain[i]) * (strain[i] - referenceStrain[i]);

    }
    
    // Get quadrature rule
    const Dune::QuadratureRule<double, 1>& bendingQuad 
        = Dune::QuadratureRules<double, 1>::rule(element.type(), bendingQuadOrder);
    
    for (size_t pt=0; pt<bendingQuad.size(); pt++) {
        
        // Local position of the quadrature point
        const Dune::FieldVector<double,1>& quadPos = bendingQuad[pt].position();
        
        double weight = bendingQuad[pt].weight() * element.geometry().integrationElement(quadPos);
        
        Dune::FieldVector<double,6> strain = getStrain(localSolution, element, quadPos);
        
        // The reference strain
        Dune::FieldVector<double,6> referenceStrain = getStrain(localReferenceConfiguration, element, quadPos);
        
        // Part II: the bending and twisting energy
        for (int i=0; i<3; i++)
            energy += weight * 0.5 * K_[i] * (strain[i+3] - referenceStrain[i+3]) * (strain[i+3] - referenceStrain[i+3]);
        
    }

    return energy;
}

#endif

