#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>

#include <dune/grid/common/quadraturerules.hh>

#include <dune/localfunctions/lagrange/p1.hh>

#include "src/rodlocalstiffness.hh"



template <class GridView>
void RodAssembler<GridView,3>::
assembleGradient(const std::vector<RigidBodyMotion<3> >& sol,
                 Dune::BlockVector<Dune::FieldVector<double, blocksize> >& grad) const
{
    using namespace Dune;

    const typename GridView::Traits::IndexSet& indexSet = this->gridView_.indexSet();

    if (sol.size()!=indexSet.size(gridDim))
        DUNE_THROW(Exception, "Solution vector doesn't match the grid!");

    grad.resize(sol.size());
    grad = 0;

    ElementIterator it    = this->gridView_.template begin<0>();
    ElementIterator endIt = this->gridView_.template end<0>();

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

}


template <class GridView>
void RodAssembler<GridView,3>::
getStrain(const std::vector<RigidBodyMotion<3> >& sol,
          Dune::BlockVector<Dune::FieldVector<double, blocksize> >& strain) const
{
    using namespace Dune;

    const typename GridView::Traits::IndexSet& indexSet = this->gridView_.indexSet();

    if (sol.size()!=indexSet.size(gridDim))
        DUNE_THROW(Exception, "Solution vector doesn't match the grid!");

    // Strain defined on each element
    strain.resize(indexSet.size(0));
    strain = 0;

    ElementIterator it    = this->gridView_.template begin<0>();
    ElementIterator endIt = this->gridView_.template end<0>();

    // Loop over all elements
    for (; it!=endIt; ++it) {

        int elementIdx = indexSet.index(*it);

        // Extract local solution on this element
        Dune::P1LocalFiniteElement<double,double,gridDim> localFiniteElement;
        int numOfBaseFct = localFiniteElement.localCoefficients().size();

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
void RodAssembler<GridView,3>::
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
Dune::FieldVector<double,3> RodAssembler<GridView,3>::
getResultantForce(const BoundaryPatchBase<PatchGridView>& boundary,
                  const std::vector<RigidBodyMotion<3> >& sol,
                  Dune::FieldVector<double,3>& canonicalTorque) const
{
    using namespace Dune;

    //    if (gridView_ != &boundary.gridView())
    //        DUNE_THROW(Dune::Exception, "The boundary patch has to match the grid view of the assembler!");

    const typename GridView::Traits::IndexSet& indexSet = this->gridView_.indexSet();

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
            canonicalStress *= it->unitOuterNormal(FieldVector<double,0>(0))[0];
            canonicalTorque *= it->unitOuterNormal(FieldVector<double,0>(0))[0];
            
    }

    return canonicalStress;
}


template <class GridView>
void PlanarRodAssembler<GridView>::
assembleMatrix(const std::vector<RigidBodyMotion<2> >& sol,
               Dune::BCRSMatrix<MatrixBlock>& matrix)
{
    const typename GridView::IndexSet& indexSet = this->gridView_.indexSet();

    Dune::MatrixIndexSet neighborsPerVertex;
    this->getNeighborsPerVertex(neighborsPerVertex);
    
    matrix = 0;
    
    ElementIterator it    = this->gridView_.template begin<0>();
    ElementIterator endit = this->gridView_.template end<0>  ();

    Dune::Matrix<MatrixBlock> mat;
    
    for( ; it != endit; ++it ) {
        
        const int numOfBaseFct = 2;  
        
        mat.setSize(numOfBaseFct, numOfBaseFct);

        // Extract local solution
        std::vector<RigidBodyMotion<2> > localSolution(numOfBaseFct);
        
        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[indexSet.subIndex(*it,i,gridDim)];

        // setup matrix 
        getLocalMatrix( *it, localSolution, numOfBaseFct, mat);
        
        // Add element matrix to global stiffness matrix
        for(int i=0; i<numOfBaseFct; i++) { 
            
            int row = indexSet.subIndex(*it,i,gridDim);

            for (int j=0; j<numOfBaseFct; j++ ) {
                
                int col = indexSet.subIndex(*it,j,gridDim);
                matrix[row][col] += mat[i][j];
                
            }
        }

    }
    
}






template <class GridView>
template <class MatrixType>
void PlanarRodAssembler<GridView>::
getLocalMatrix( EntityType &entity, 
                const std::vector<RigidBodyMotion<2> >& localSolution,
                const int matSize, MatrixType& localMat) const
{
    const typename GridView::IndexSet& indexSet = this->gridView_.indexSet();

    /* ndof is the number of vectors of the element */
    int ndof = matSize;

    for (int i=0; i<matSize; i++)
        for (int j=0; j<matSize; j++)
            localMat[i][j] = 0;
    
    Dune::P1LocalFiniteElement<double,double,gridDim> localFiniteElement;

    // Get quadrature rule
    const Dune::QuadratureRule<double, gridDim>& quad = Dune::QuadratureRules<double, gridDim>::rule(entity.type(), 2);
    
    /* Loop over all integration points */
    for (int ip=0; ip<quad.size(); ip++) {
        
        // Local position of the quadrature point
        const Dune::FieldVector<double,gridDim>& quadPos = quad[ip].position();

        // calc Jacobian inverse before integration element is evaluated 
        const Dune::FieldMatrix<double,gridDim,gridDim>& inv = entity.geometry().jacobianInverseTransposed(quadPos);
        const double integrationElement = entity.geometry().integrationElement(quadPos);
        
        /* Compute the weight of the current integration point */
        double weight = quad[ip].weight() * integrationElement;
        
        /**********************************************/
        /* compute gradients of the shape functions   */
        /**********************************************/
        std::vector<Dune::FieldMatrix<double,1,gridDim> > referenceElementGradients(ndof);
        localFiniteElement.localBasis().evaluateJacobian(quadPos,referenceElementGradients);

        std::vector<Dune::FieldVector<double,gridDim> > shapeGrad(ndof);
        
        // multiply with jacobian inverse 
        for (int dof=0; dof<ndof; dof++)
            inv.mv(referenceElementGradients[dof][0], shapeGrad[dof]);
        
        
        std::vector<Dune::FieldVector<double,1> > shapeFunction;
        localFiniteElement.localBasis().evaluateFunction(quadPos,shapeFunction);

        // //////////////////////////////////
        //   Interpolate
        // //////////////////////////////////
        
        double x_s     = localSolution[0].r[0]*shapeGrad[0][0] + localSolution[1].r[0]*shapeGrad[1][0];
        double y_s     = localSolution[0].r[1]*shapeGrad[0][0] + localSolution[1].r[1]*shapeGrad[1][0];

        double theta   = localSolution[0].q.angle_*shapeFunction[0] + localSolution[1].q.angle_*shapeFunction[1];

        for (int i=0; i<matSize; i++) {

            for (int j=0; j<matSize; j++) {

                // \partial J^2 / \partial x_i \partial x_j
                localMat[i][j][0][0] += weight * shapeGrad[i][0] * shapeGrad[j][0]
                    * (A1 * cos(theta) * cos(theta) + A3 * sin(theta) * sin(theta));

                // \partial J^2 / \partial x_i \partial y_j
                localMat[i][j][0][1] += weight * shapeGrad[i][0] * shapeGrad[j][0]
                    * (-A1 + A3) * sin(theta)* cos(theta);

                // \partial J^2 / \partial x_i \partial theta_j
                localMat[i][j][0][2] += weight * shapeGrad[i][0] * shapeFunction[j]
                    * (-A1 * (x_s*sin(theta) + y_s*cos(theta)) * cos(theta)
                       - A1* (x_s*cos(theta) - y_s*sin(theta)) * sin(theta)
                       +A3 * (x_s*cos(theta) - y_s*sin(theta)) * sin(theta)
                       +A3 * (x_s*sin(theta) + y_s*cos(theta) - 1) * cos(theta));

                // \partial J^2 / \partial y_i \partial x_j
                localMat[i][j][1][0] += weight * shapeGrad[i][0] * shapeGrad[j][0]
                    * (-A1 * sin(theta)* cos(theta) + A3 * cos(theta) * sin(theta));

                // \partial J^2 / \partial y_i \partial y_j
                localMat[i][j][1][1] += weight * shapeGrad[i][0] * shapeGrad[j][0]
                    * (A1 * sin(theta)*sin(theta) + A3 * cos(theta)*cos(theta));

                // \partial J^2 / \partial y_i \partial theta_j
                localMat[i][j][1][2] += weight * shapeGrad[i][0] * shapeFunction[j]
                    * (A1  * (x_s * sin(theta) + y_s * cos(theta)) * sin(theta)
                       -A1 * (x_s * cos(theta) - y_s * sin(theta)) * cos(theta)
                       +A3 * (x_s * cos(theta) - y_s * sin(theta)) * cos(theta)
                       -A3 * (x_s * sin(theta) + y_s * cos(theta) - 1) * sin(theta));

                // \partial J^2 / \partial theta_i \partial x_j
                localMat[i][j][2][0] += weight * shapeFunction[i] * shapeGrad[j][0]
                    * (-A1 * (x_s*sin(theta) + y_s*cos(theta)) * cos(theta)
                       - A1* (x_s*cos(theta) - y_s*sin(theta)) * sin(theta)
                       +A3 * (x_s*cos(theta) - y_s*sin(theta)) * sin(theta)
                       +A3 * (x_s*sin(theta) + y_s*cos(theta) - 1) * cos(theta));

                // \partial J^2 / \partial theta_i \partial y_j
                localMat[i][j][2][1] += weight * shapeFunction[i] * shapeGrad[j][0]
                    * (A1  * (x_s * sin(theta) + y_s * cos(theta)) * sin(theta)
                       -A1 * (x_s * cos(theta) - y_s * sin(theta)) * cos(theta)
                       +A3 * (x_s * cos(theta) - y_s * sin(theta)) * cos(theta)
                       -A3 * (x_s * sin(theta) + y_s * cos(theta) - 1) * sin(theta));

                // \partial J^2 / \partial theta_i \partial theta_j
                localMat[i][j][2][2] += weight * B * shapeGrad[i][0] * shapeGrad[j][0];
                localMat[i][j][2][2] += weight * shapeFunction[i] * shapeFunction[j]
                    * (+ A1 * (x_s*sin(theta) + y_s*cos(theta)) * (x_s*sin(theta) + y_s*cos(theta))
                       + A1 * (x_s*cos(theta) - y_s*sin(theta)) * (-x_s*cos(theta)+ y_s*sin(theta))
                       + A3 * (x_s*cos(theta) - y_s*sin(theta)) * (x_s*cos(theta) - y_s*sin(theta))
                       - A3 * (x_s*sin(theta) + y_s*cos(theta) - 1) * (x_s*sin(theta) + y_s*cos(theta)));
                                                


            }
        
        }
        
        
    }
    
#if 0
    static int eleme = 0;
    printf("********* Element %d **********\n", eleme++);
    for (int row=0; row<matSize; row++) {
        
        for (int rcomp=0; rcomp<dim; rcomp++) {
            
            for (int col=0; col<matSize; col++) {
                
                for (int ccomp=0; ccomp<dim; ccomp++)
                    std::cout << mat[row][col][rcomp][ccomp] << "  ";
                
                std::cout << "    ";
                
            }
            
            std::cout << std::endl;
            
        }
        
        std::cout << std::endl;
        
    }
    exit(0);
#endif
    
}

template <class GridView>
void PlanarRodAssembler<GridView>::
assembleGradient(const std::vector<RigidBodyMotion<2> >& sol,
                 Dune::BlockVector<Dune::FieldVector<double, blocksize> >& grad) const
{
    const typename GridView::IndexSet& indexSet = this->gridView_.indexSet();

    if (sol.size()!=this->gridView_.size(gridDim))
        DUNE_THROW(Dune::Exception, "Solution vector doesn't match the grid!");

    grad.resize(sol.size());
    grad = 0;

    ElementIterator it    = this->gridView_.template begin<0>();
    ElementIterator endIt = this->gridView_.template end<0>();

    // Loop over all elements
    for (; it!=endIt; ++it) {

        // Extract local solution on this element
        Dune::P1LocalFiniteElement<double,double,gridDim> localFiniteElement;
        const int numOfBaseFct = localFiniteElement.localBasis().size();  
        
        RigidBodyMotion<2> localSolution[numOfBaseFct];
        
        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[indexSet.subIndex(*it,i,gridDim)];

        // Get quadrature rule
        const Dune::QuadratureRule<double, gridDim>& quad = Dune::QuadratureRules<double, gridDim>::rule(it->type(), 2);

        for (int pt=0; pt<quad.size(); pt++) {

            // Local position of the quadrature point
            const Dune::FieldVector<double,gridDim>& quadPos = quad[pt].position();
            
            const Dune::FieldMatrix<double,1,1>& inv = it->geometry().jacobianInverseTransposed(quadPos);
            const double integrationElement = it->geometry().integrationElement(quadPos);
        
            double weight = quad[pt].weight() * integrationElement;
            
            /**********************************************/
            /* compute gradients of the shape functions   */
            /**********************************************/
            std::vector<Dune::FieldMatrix<double,1,gridDim> > referenceElementGradients(numOfBaseFct);
            localFiniteElement.localBasis().evaluateJacobian(quadPos,referenceElementGradients);
            
            std::vector<Dune::FieldVector<double,gridDim> > shapeGrad(numOfBaseFct);
            
            // multiply with jacobian inverse 
            for (int dof=0; dof<numOfBaseFct; dof++)
                inv.mv(referenceElementGradients[dof][0], shapeGrad[dof]);

            // Get the values of the shape functions
            std::vector<Dune::FieldVector<double,1> > shapeFunction;
            localFiniteElement.localBasis().evaluateFunction(quadPos,shapeFunction);

            // //////////////////////////////////
            //   Interpolate
            // //////////////////////////////////

            double x_s     = localSolution[0].r[0]*shapeGrad[0][0] + localSolution[1].r[0]*shapeGrad[1][0];
            double y_s     = localSolution[0].r[1]*shapeGrad[0][0] + localSolution[1].r[1]*shapeGrad[1][0];
            double theta_s = localSolution[0].q.angle_*shapeGrad[0][0] + localSolution[1].q.angle_*shapeGrad[1][0];

            double theta   = localSolution[0].q.angle_*shapeFunction[0] + localSolution[1].q.angle_*shapeFunction[1];

            // /////////////////////////////////////////////
            //   Sum it all up
            // /////////////////////////////////////////////

            double partA1 = A1 * (x_s * cos(theta) - y_s * sin(theta));
            double partA3 = A3 * (x_s * sin(theta) + y_s * cos(theta) - 1);

            for (int dof=0; dof<numOfBaseFct; dof++) {

                int globalDof = indexSet.subIndex(*it,dof,gridDim);

                //printf("globalDof: %d   partA1: %g   partA3: %g\n", globalDof, partA1, partA3);

                // \partial J / \partial x^i
                grad[globalDof][0] += weight * (partA1 * cos(theta) + partA3 * sin(theta)) * shapeGrad[dof][0];

                // \partial J / \partial y^i
                grad[globalDof][1] += weight * (-partA1 * sin(theta) + partA3 * cos(theta)) * shapeGrad[dof][0];

                // \partial J / \partial \theta^i
                grad[globalDof][2] += weight * (B * theta_s * shapeGrad[dof][0]
                                               + partA1 * (-x_s * sin(theta) - y_s * cos(theta)) * shapeFunction[dof]
                                               + partA3 * ( x_s * cos(theta) - y_s * sin(theta)) * shapeFunction[dof]);

            }


        }

    }

}


template <class GridView>
double PlanarRodAssembler<GridView>::
computeEnergy(const std::vector<RigidBodyMotion<2> >& sol) const
{
    double energy = 0;

    const typename GridView::IndexSet& indexSet = this->gridView_.indexSet();

    if (sol.size()!=this->gridView_.size(gridDim))
        DUNE_THROW(Dune::Exception, "Solution vector doesn't match the grid!");

    ElementIterator it    = this->gridView_.template begin<0>();
    ElementIterator endIt = this->gridView_.template end<0>();

    // Loop over all elements
    for (; it!=endIt; ++it) {

        // Extract local solution on this element
        Dune::P1LocalFiniteElement<double,double,gridDim> localFiniteElement;

        int numOfBaseFct = localFiniteElement.localBasis().size();

        RigidBodyMotion<2> localSolution[numOfBaseFct];
        
        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[indexSet.subIndex(*it,i,gridDim)];

        // Get quadrature rule
        const Dune::QuadratureRule<double, gridDim>& quad = Dune::QuadratureRules<double, gridDim>::rule(it->type(), 2);

        for (int pt=0; pt<quad.size(); pt++) {

            // Local position of the quadrature point
            const Dune::FieldVector<double,gridDim>& quadPos = quad[pt].position();
            
            const Dune::FieldMatrix<double,1,1>& inv = it->geometry().jacobianInverseTransposed(quadPos);
            const double integrationElement = it->geometry().integrationElement(quadPos);
        
            double weight = quad[pt].weight() * integrationElement;
            
            /**********************************************/
            /* compute gradients of the shape functions   */
            /**********************************************/
            std::vector<Dune::FieldMatrix<double,1,gridDim> > referenceElementGradients(numOfBaseFct);
            localFiniteElement.localBasis().evaluateJacobian(quadPos,referenceElementGradients);
            
            std::vector<Dune::FieldVector<double,gridDim> > shapeGrad(numOfBaseFct);
            
            // multiply with jacobian inverse 
            for (int dof=0; dof<numOfBaseFct; dof++)
                inv.mv(referenceElementGradients[dof][0], shapeGrad[dof]);

            // Get the value of the shape functions
            std::vector<Dune::FieldVector<double,1> > shapeFunction;
            localFiniteElement.localBasis().evaluateFunction(quadPos,shapeFunction);

            // //////////////////////////////////
            //   Interpolate
            // //////////////////////////////////

            double x_s     = localSolution[0].r[0]*shapeGrad[0][0] + localSolution[1].r[0]*shapeGrad[1][0];
            double y_s     = localSolution[0].r[1]*shapeGrad[0][0] + localSolution[1].r[1]*shapeGrad[1][0];
            double theta_s = localSolution[0].q.angle_*shapeGrad[0][0] + localSolution[1].q.angle_*shapeGrad[1][0];

            double theta   = localSolution[0].q.angle_*shapeFunction[0] + localSolution[1].q.angle_*shapeFunction[1];

            // /////////////////////////////////////////////
            //   Sum it all up
            // /////////////////////////////////////////////

            double partA1 = x_s * cos(theta) - y_s * sin(theta);
            double partA3 = x_s * sin(theta) + y_s * cos(theta) - 1;


            energy += 0.5 * weight * (B * theta_s * theta_s
                                      + A1 * partA1 * partA1
                                      + A3 * partA3 * partA3);

        }

    }

    return energy;

}
