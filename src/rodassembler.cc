#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>

#include <dune/grid/common/quadraturerules.hh>

#include <dune/disc/shapefunctions/lagrangeshapefunctions.hh>

#include "src/rodlocalstiffness.hh"



template <class GridType>
void RodAssembler<GridType>::
getNeighborsPerVertex(Dune::MatrixIndexSet& nb) const
{
    const int gridDim = GridType::dimension;
    const typename GridType::Traits::LevelIndexSet& indexSet = grid_->levelIndexSet(grid_->maxLevel());
    
    int i, j;
    int n = grid_->size(grid_->maxLevel(), gridDim);
    
    nb.resize(n, n);
    
    ElementIterator it    = grid_->template lbegin<0>( grid_->maxLevel() );
    ElementIterator endit = grid_->template lend<0>  ( grid_->maxLevel() );
    
    for (; it!=endit; ++it) {
        
        for (i=0; i<it->template count<gridDim>(); i++) {
            
            for (j=0; j<it->template count<gridDim>(); j++) {
                
                int iIdx = indexSet.subIndex(*it,i,gridDim);
                int jIdx = indexSet.subIndex(*it,j,gridDim);
                
                nb.add(iIdx, jIdx);
                
            }
            
        }
        
    }
    
}


template <class GridType>
void RodAssembler<GridType>::
assembleMatrix(const std::vector<RigidBodyMotion<3> >& sol,
               Dune::BCRSMatrix<MatrixBlock>& matrix) const
{
    // ////////////////////////////////////////////////////
    //   Create local assembler
    // ////////////////////////////////////////////////////

    Dune::array<double,3> K = {K_[0], K_[1], K_[2]};
    Dune::array<double,3> A = {A_[0], A_[1], A_[2]};
    RodLocalStiffness<typename GridType::LeafGridView,double> localStiffness(K, A);

    const typename GridType::Traits::LevelIndexSet& indexSet = grid_->levelIndexSet(grid_->maxLevel());

    Dune::MatrixIndexSet neighborsPerVertex;
    getNeighborsPerVertex(neighborsPerVertex);
    
    matrix = 0;
    
    ElementIterator it    = grid_->template lbegin<0>( grid_->maxLevel() );
    ElementIterator endit = grid_->template lend<0> ( grid_->maxLevel() );

    for( ; it != endit; ++it ) {
        
        const Dune::LagrangeShapeFunctionSet<double, double, gridDim> & baseSet 
            = Dune::LagrangeShapeFunctions<double, double, gridDim>::general(it->type(), elementOrder);
        const int numOfBaseFct = baseSet.size();  
        
        // Extract local solution
        std::vector<RigidBodyMotion<3> > localSolution(numOfBaseFct);
        
        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[indexSet.subIndex(*it,i,gridDim)];

        localStiffness.localReferenceConfiguration_.resize(numOfBaseFct);
        
        for (int i=0; i<numOfBaseFct; i++)
            localStiffness.localReferenceConfiguration_[i] = referenceConfiguration_[indexSet.subIndex(*it,i,gridDim)];

        // setup matrix 
        localStiffness.assemble(*it, localSolution);

        // Add element matrix to global stiffness matrix
        for(int i=0; i<numOfBaseFct; i++) { 
            
            int row = indexSet.subIndex(*it,i,gridDim);

            for (int j=0; j<numOfBaseFct; j++ ) {
                
                int col = indexSet.subIndex(*it,j,gridDim);
                matrix[row][col] += localStiffness.mat(i,j);
                
            }
        }

    }

}

template <class GridType>
void RodAssembler<GridType>::
assembleGradient(const std::vector<RigidBodyMotion<3> >& sol,
                 Dune::BlockVector<Dune::FieldVector<double, blocksize> >& grad) const
{
    using namespace Dune;

    const typename GridType::Traits::LevelIndexSet& indexSet = grid_->levelIndexSet(grid_->maxLevel());
    const int maxlevel = grid_->maxLevel();

    if (sol.size()!=grid_->size(maxlevel, gridDim))
        DUNE_THROW(Exception, "Solution vector doesn't match the grid!");

    // ////////////////////////////////////////////////////
    //   Create local assembler
    // ////////////////////////////////////////////////////

    Dune::array<double,3> K = {K_[0], K_[1], K_[2]};
    Dune::array<double,3> A = {A_[0], A_[1], A_[2]};
    RodLocalStiffness<typename GridType::LeafGridView,double> localStiffness(K, A);

    grad.resize(sol.size());
    grad = 0;

    ElementIterator it    = grid_->template lbegin<0>(maxlevel);
    ElementIterator endIt = grid_->template lend<0>(maxlevel);

    // Loop over all elements
    for (; it!=endIt; ++it) {

        // A 1d grid has two vertices
        const int nDofs = 2;

        // Extract local solution
        std::vector<RigidBodyMotion<3> > localSolution(nDofs);
        
        for (int i=0; i<nDofs; i++)
            localSolution[i] = sol[indexSet.subIndex(*it,i,gridDim)];

        // Extract local reference configuration
        std::vector<RigidBodyMotion<3> > localReferenceConfiguration(nDofs);
        
        for (int i=0; i<nDofs; i++)
            localReferenceConfiguration[i] = referenceConfiguration_[indexSet.subIndex(*it,i,gridDim)];

        // Assemble local gradient
        std::vector<FieldVector<double,blocksize> > localGradient(nDofs);

        localStiffness.localReferenceConfiguration_ = localReferenceConfiguration;
        localStiffness.assembleGradient(*it, localSolution, localGradient);

        // Add to global gradient
        for (int i=0; i<nDofs; i++)
            grad[indexSet.subIndex(*it,i,gridDim)] += localGradient[i];

    }

}


template <class GridType>
double RodAssembler<GridType>::
computeEnergy(const std::vector<RigidBodyMotion<3> >& sol) const
{
    using namespace Dune;

    double energy = 0;
    
    const typename GridType::Traits::LeafIndexSet& indexSet = grid_->leafIndexSet();

    if (sol.size()!=indexSet.size(gridDim))
        DUNE_THROW(Exception, "Solution vector doesn't match the grid!");

    // ////////////////////////////////////////////////////
    //   Create local assembler
    // ////////////////////////////////////////////////////

    Dune::array<double,3> K = {K_[0], K_[1], K_[2]};
    Dune::array<double,3> A = {A_[0], A_[1], A_[2]};
    RodLocalStiffness<typename GridType::LeafGridView,double> localStiffness(K, A);

    std::vector<RigidBodyMotion<3> > localReferenceConfiguration(2);
    std::vector<RigidBodyMotion<3> > localSolution(2);

    ElementLeafIterator it    = grid_->template leafbegin<0>();
    ElementLeafIterator endIt = grid_->template leafend<0>();

    // Loop over all elements
    for (; it!=endIt; ++it) {

        for (int i=0; i<2; i++) {

            localReferenceConfiguration[i] = referenceConfiguration_[indexSet.subIndex(*it,i,gridDim)];
            localSolution[i]               = sol[indexSet.subIndex(*it,i,gridDim)];

        }

        localStiffness.localReferenceConfiguration_ = localReferenceConfiguration;
        energy += localStiffness.energy(*it, localSolution);

    }

    return energy;

}


template <class GridType>
void RodAssembler<GridType>::
getStrain(const std::vector<RigidBodyMotion<3> >& sol,
          Dune::BlockVector<Dune::FieldVector<double, blocksize> >& strain) const
{
    using namespace Dune;

    const typename GridType::Traits::LeafIndexSet& indexSet = grid_->leafIndexSet();

    if (sol.size()!=indexSet.size(gridDim))
        DUNE_THROW(Exception, "Solution vector doesn't match the grid!");

    // ////////////////////////////////////////////////////
    //   Create local assembler
    // ////////////////////////////////////////////////////

    Dune::array<double,3> K = {K_[0], K_[1], K_[2]};
    Dune::array<double,3> A = {A_[0], A_[1], A_[2]};
    RodLocalStiffness<typename GridType::LeafGridView,double> localStiffness(K, A);

    // Strain defined on each element
    strain.resize(indexSet.size(0));
    strain = 0;

    ElementLeafIterator it    = grid_->template leafbegin<0>();
    ElementLeafIterator endIt = grid_->template leafend<0>();

    // Loop over all elements
    for (; it!=endIt; ++it) {

        int elementIdx = indexSet.index(*it);

        // Extract local solution on this element
        const LagrangeShapeFunctionSet<double, double, gridDim> & baseSet 
            = Dune::LagrangeShapeFunctions<double, double, gridDim>::general(it->type(), elementOrder);
        int numOfBaseFct = baseSet.size();

        std::vector<RigidBodyMotion<3> > localSolution(2);
        
        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[indexSet.subIndex(*it,i,gridDim)];

        // Get quadrature rule
        const int polOrd = 2;
        const QuadratureRule<double, gridDim>& quad = QuadratureRules<double, gridDim>::rule(it->type(), polOrd);

        for (int pt=0; pt<quad.size(); pt++) {

            // Local position of the quadrature point
            const FieldVector<double,gridDim>& quadPos = quad[pt].position();

            double weight = quad[pt].weight() * it->geometry().integrationElement(quadPos);

            FieldVector<double,blocksize> localStrain = localStiffness.getStrain(localSolution, *it, quad[pt].position());
            
            // Sum it all up
            strain[elementIdx].axpy(weight, localStrain);

        }

        // /////////////////////////////////////////////////////////////////////////
        //   We want the average strain per element.  Therefore we have to divide
        //   the integral we just computed by the element volume.
        // /////////////////////////////////////////////////////////////////////////
        // we know the element is a line, therefore the integration element is the volume
        FieldVector<double,1> dummyPos(0.5);  
        strain[elementIdx] /= it->geometry().integrationElement(dummyPos);

    }

}

template <class GridType>
void RodAssembler<GridType>::
getStress(const std::vector<RigidBodyMotion<3> >& sol,
          Dune::BlockVector<Dune::FieldVector<double, blocksize> >& stress) const
{
    // Get the strain
    getStrain(sol,stress);

    // Get reference strain
    Dune::BlockVector<Dune::FieldVector<double, blocksize> > referenceStrain;
    getStrain(referenceConfiguration_, referenceStrain);

    // Linear diagonal constitutive law
    for (size_t i=0; i<stress.size(); i++) {
        for (int j=0; j<3; j++) {
            stress[i][j]   = (stress[i][j]   - referenceStrain[i][j])   * A_[j];
            stress[i][j+3] = (stress[i][j+3] - referenceStrain[i][j+3]) * K_[j];
        }
    }
}

template <class GridType>
Dune::FieldVector<double,3> RodAssembler<GridType>::
getResultantForce(const BoundaryPatch<GridType>& boundary, 
                  const std::vector<RigidBodyMotion<3> >& sol,
                  Dune::FieldVector<double,3>& canonicalTorque) const
{
    using namespace Dune;

    if (grid_ != &boundary.getGrid())
        DUNE_THROW(Dune::Exception, "The boundary patch has to match the grid of the assembler!");

    const typename GridType::Traits::LeafIndexSet& indexSet = grid_->leafIndexSet();

    if (sol.size()!=indexSet.size(gridDim))
        DUNE_THROW(Exception, "Solution vector doesn't match the grid!");

    /** \todo Eigentlich sollte das BoundaryPatch ein LeafBoundaryPatch sein */
    if (boundary.level() != grid_->maxLevel())
        DUNE_THROW(Exception, "The boundary patch has to refer to the max level!");

    FieldVector<double,3> canonicalStress(0);
    canonicalTorque = 0;

    // Loop over the given boundary
    ElementLeafIterator eIt    = grid_->template leafbegin<0>();
    ElementLeafIterator eEndIt = grid_->template leafend<0>();

    for (; eIt!=eEndIt; ++eIt) {

        typename EntityType::LeafIntersectionIterator nIt    = eIt->ileafbegin();
        typename EntityType::LeafIntersectionIterator nEndIt = eIt->ileafend();

        for (; nIt!=nEndIt; ++nIt) {

            if (!boundary.contains(*eIt, nIt->indexInInside()))
                continue;

            // //////////////////////////////////////////////
            //   Compute force across this boundary face
            // //////////////////////////////////////////////

            //   Create local assembler
            
            Dune::array<double,3> K = {K_[0], K_[1], K_[2]};
            Dune::array<double,3> A = {A_[0], A_[1], A_[2]};
            RodLocalStiffness<typename GridType::LeafGridView,double> localStiffness(K, A);

            double pos = nIt->intersectionSelfLocal().corner(0);

            Dune::array<RigidBodyMotion<3>,2> localSolution = {sol[indexSet.template subIndex<1>(*eIt,0)],
                                                          sol[indexSet.template subIndex<1>(*eIt,1)]};
            Dune::array<RigidBodyMotion<3>,2> localRefConf  = {referenceConfiguration_[indexSet.template subIndex<1>(*eIt,0)],
                                                          referenceConfiguration_[indexSet.template subIndex<1>(*eIt,1)]};

            FieldVector<double, blocksize> strain          = localStiffness.getStrain(localSolution, *eIt, pos);
            FieldVector<double, blocksize> referenceStrain = localStiffness.getStrain(localRefConf, *eIt, pos);

            FieldVector<double,3> localStress;
            for (int i=0; i<3; i++)
                localStress[i] = (strain[i] - referenceStrain[i]) * A_[i];

            FieldVector<double,3> localTorque;
            for (int i=0; i<3; i++)
                localTorque[i] = (strain[i+3] - referenceStrain[i+3]) * K_[i];

            // Transform stress given with respect to the basis given by the three directors to
            // the canonical basis of R^3

            FieldMatrix<double,3,3> orientationMatrix;
            sol[indexSet.template subIndex<1>(*eIt,nIt->numberInSelf())].q.matrix(orientationMatrix);
            
            orientationMatrix.umv(localStress, canonicalStress);
            
            orientationMatrix.umv(localTorque, canonicalTorque);
            // Reverse transformation to make sure we did the correct thing
//             assert( std::abs(localStress[0]-canonicalStress*sol[0].q.director(0)) < 1e-6 );
//             assert( std::abs(localStress[1]-canonicalStress*sol[0].q.director(1)) < 1e-6 );
//             assert( std::abs(localStress[2]-canonicalStress*sol[0].q.director(2)) < 1e-6 );

            // Multiply force times boundary normal to get the transmitted force
            /** \todo The minus sign comes from the coupling conditions.  It
                should really be in the Dirichlet-Neumann code. */
            canonicalStress *= -nIt->unitOuterNormal(FieldVector<double,0>(0))[0];
            canonicalTorque *= -nIt->unitOuterNormal(FieldVector<double,0>(0))[0];
            
        }

    }

    return canonicalStress;
}

