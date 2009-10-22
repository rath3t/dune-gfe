#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>

#include <dune/grid/common/quadraturerules.hh>

#include <dune/disc/shapefunctions/lagrangeshapefunctions.hh>

#include "src/rodlocalstiffness.hh"



template <class GridView>
void RodAssembler<GridView>::
assembleGradient(const std::vector<RigidBodyMotion<3> >& sol,
                 Dune::BlockVector<Dune::FieldVector<double, blocksize> >& grad) const
{
    using namespace Dune;

    const typename GridView::Traits::IndexSet& indexSet = gridView_.indexSet();

    if (sol.size()!=indexSet.size(gridDim))
        DUNE_THROW(Exception, "Solution vector doesn't match the grid!");

    grad.resize(sol.size());
    grad = 0;

    ElementIterator it    = gridView_.template begin<0>();
    ElementIterator endIt = gridView_.template end<0>();

    // Loop over all elements
    for (; it!=endIt; ++it) {

        // A 1d grid has two vertices
        const int nDofs = 2;

        // Extract local solution
        std::vector<RigidBodyMotion<3> > localSolution(nDofs);
        
        for (int i=0; i<nDofs; i++)
            localSolution[i] = sol[indexSet.subIndex(*it,i,gridDim)];

        // Assemble local gradient
        std::vector<FieldVector<double,blocksize> > localGradient(nDofs);

        this->localStiffness_->assembleGradient(*it, localSolution, localGradient);

        // Add to global gradient
        for (int i=0; i<nDofs; i++)
            grad[indexSet.subIndex(*it,i,gridDim)] += localGradient[i];

    }

    // ///////////////////////////////////////////////////////////////////////
    //   Add the contributions of the Neumann data.  Since the boundary is
    //   zero-dimensional these are not integrals but simply values
    //   added at the first and last vertex.
    // \todo We use again that the numbering goes from left to right!
    // ///////////////////////////////////////////////////////////////////////
    for (int i=0; i<3; i++) {
        grad[0][i]               += leftNeumannForce_[i];
        grad[0][i+3]             += leftNeumannTorque_[i];
        grad[grad.size()-1][i]   += rightNeumannForce_[i];
        grad[grad.size()-1][i+3] += rightNeumannTorque_[i];
    }

}


template <class GridView>
double RodAssembler<GridView>::
computeEnergy(const std::vector<RigidBodyMotion<3> >& sol) const
{
    double energy = GeodesicFEAssembler<GridView,RigidBodyMotion<3> >::computeEnergy(sol);

    // ///////////////////////////////////////////////////////////////////////
    //   Add the contributions of the Neumann data.  Since the boundary is
    //   zero-dimensional these are not integrals but simply values
    //   added at the first and last vertex.
    // \todo We use again that the numbering goes from left to right!
    // ///////////////////////////////////////////////////////////////////////

    energy += sol[0].r * leftNeumannForce_;
    //energy += Rotation<3,double>::expInv(sol[0].q) * leftNeumannTorque_;

    energy += sol.back().r * rightNeumannForce_;
    //energy += Rotation<3,double>::expInv(sol.back().q) * rightNeumannTorque_;

    return energy;

}


template <class GridView>
void RodAssembler<GridView>::
getStrain(const std::vector<RigidBodyMotion<3> >& sol,
          Dune::BlockVector<Dune::FieldVector<double, blocksize> >& strain) const
{
    using namespace Dune;

    const typename GridView::Traits::IndexSet& indexSet = gridView_.indexSet();

    if (sol.size()!=indexSet.size(gridDim))
        DUNE_THROW(Exception, "Solution vector doesn't match the grid!");

    // Strain defined on each element
    strain.resize(indexSet.size(0));
    strain = 0;

    ElementIterator it    = gridView_.template begin<0>();
    ElementIterator endIt = gridView_.template end<0>();

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

            FieldVector<double,blocksize> localStrain = dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->getStrain(localSolution, *it, quad[pt].position());
            
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

template <class GridView>
void RodAssembler<GridView>::
getStress(const std::vector<RigidBodyMotion<3> >& sol,
          Dune::BlockVector<Dune::FieldVector<double, blocksize> >& stress) const
{
    // Get the strain
    getStrain(sol,stress);

    // Get reference strain
    Dune::BlockVector<Dune::FieldVector<double, blocksize> > referenceStrain;
    getStrain(dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->referenceConfiguration_, referenceStrain);

    // Linear diagonal constitutive law
    for (size_t i=0; i<stress.size(); i++) {
        for (int j=0; j<3; j++) {
            stress[i][j]   = (stress[i][j]   - referenceStrain[i][j])   * dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->A_[j];
            stress[i][j+3] = (stress[i][j+3] - referenceStrain[i][j+3]) * dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->K_[j];
        }
    }
}

template <class GridView>
template <class PatchGridView>
Dune::FieldVector<double,3> RodAssembler<GridView>::
getResultantForce(const BoundaryPatchBase<PatchGridView>& boundary,
                  const std::vector<RigidBodyMotion<3> >& sol,
                  Dune::FieldVector<double,3>& canonicalTorque) const
{
    using namespace Dune;

    //    if (gridView_ != &boundary.gridView())
    //        DUNE_THROW(Dune::Exception, "The boundary patch has to match the grid view of the assembler!");

    const typename GridView::Traits::IndexSet& indexSet = gridView_.indexSet();

    if (sol.size()!=indexSet.size(gridDim))
        DUNE_THROW(Exception, "Solution vector doesn't match the grid!");

    FieldVector<double,3> canonicalStress(0);
    canonicalTorque = 0;

    // Loop over the given boundary
    typename BoundaryPatchBase<PatchGridView>::iterator it    = boundary.begin();
    typename BoundaryPatchBase<PatchGridView>::iterator endIt = boundary.end();

    for (; it!=endIt; ++it) {

            // //////////////////////////////////////////////
            //   Compute force across this boundary face
            // //////////////////////////////////////////////

            double pos = it->geometryInInside().corner(0);

            std::vector<RigidBodyMotion<3> > localSolution(2);
            localSolution[0] = sol[indexSet.subIndex(*it->inside(),0,1)];
            localSolution[1] = sol[indexSet.subIndex(*it->inside(),1,1)];

            std::vector<RigidBodyMotion<3> > localRefConf(2);
            localRefConf[0] = dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->referenceConfiguration_[indexSet.subIndex(*it->inside(),0,1)];
            localRefConf[1] = dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->referenceConfiguration_[indexSet.subIndex(*it->inside(),1,1)];

            FieldVector<double, blocksize> strain          = dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->getStrain(localSolution, *it->inside(), pos);
            FieldVector<double, blocksize> referenceStrain = dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->getStrain(localRefConf, *it->inside(), pos);

            FieldVector<double,3> localStress;
            for (int i=0; i<3; i++)
                localStress[i] = (strain[i] - referenceStrain[i]) * dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->A_[i];

            FieldVector<double,3> localTorque;
            for (int i=0; i<3; i++)
                localTorque[i] = (strain[i+3] - referenceStrain[i+3]) * dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->K_[i];

            // Transform stress given with respect to the basis given by the three directors to
            // the canonical basis of R^3

            FieldMatrix<double,3,3> orientationMatrix;
            sol[indexSet.subIndex(*it->inside(),it->indexInInside(),1)].q.matrix(orientationMatrix);
            
            orientationMatrix.umv(localStress, canonicalStress);
            
            orientationMatrix.umv(localTorque, canonicalTorque);
            // Reverse transformation to make sure we did the correct thing
//             assert( std::abs(localStress[0]-canonicalStress*sol[0].q.director(0)) < 1e-6 );
//             assert( std::abs(localStress[1]-canonicalStress*sol[0].q.director(1)) < 1e-6 );
//             assert( std::abs(localStress[2]-canonicalStress*sol[0].q.director(2)) < 1e-6 );

            // Multiply force times boundary normal to get the transmitted force
            /** \todo The minus sign comes from the coupling conditions.  It
                should really be in the Dirichlet-Neumann code. */
            canonicalStress *= -it->unitOuterNormal(FieldVector<double,0>(0))[0];
            canonicalTorque *= -it->unitOuterNormal(FieldVector<double,0>(0))[0];
            
    }

    return canonicalStress;
}

