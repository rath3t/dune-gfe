#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>

#include <dune/grid/common/quadraturerules.hh>

#include <dune/disc/shapefunctions/lagrangeshapefunctions.hh>


template <class GridType, class RT>
RT RodLocalStiffness<GridType, RT>::
energy(const EntityPointer& element,
       const std::vector<Configuration>& localSolution,
       const std::vector<Configuration>& localReferenceConfiguration,
       int k)
{
    RT energy = 0;
    
    //const typename GridType::Traits::LeafIndexSet& indexSet = grid_->leafIndexSet();

    double elementEnergy = 0;
    
    // Extract local solution on this element
    const Dune::LagrangeShapeFunctionSet<double, double, 1> & baseSet 
        = Dune::LagrangeShapeFunctions<double, double, 1>::general(element->type(), k);
    int numOfBaseFct = baseSet.size();
    
    // ///////////////////////////////////////////////////////////////////////////////
    //   The following two loops are a reduced integration scheme.  We integrate
    //   the transverse shear and extensional energy with a first-order quadrature
    //   formula, even though it should be second order.  This prevents shear-locking
    // ///////////////////////////////////////////////////////////////////////////////

    const int shearingPolOrd = 2;
    const Dune::QuadratureRule<double, 1>& shearingQuad 
        = Dune::QuadratureRules<double, 1>::rule(element->type(), shearingPolOrd);
    
    for (size_t pt=0; pt<shearingQuad.size(); pt++) {
        
        // Local position of the quadrature point
        const Dune::FieldVector<double,1>& quadPos = shearingQuad[pt].position();
        
        const double integrationElement = element->geometry().integrationElement(quadPos);
        
        double weight = shearingQuad[pt].weight() * integrationElement;
        
        Dune::FieldVector<double,6> strain = getStrain(localSolution, element, quadPos);
        
        // The reference strain
        Dune::FieldVector<double,6> referenceStrain = getStrain(localReferenceConfiguration, element, quadPos);
        
        for (int i=0; i<3; i++) {
            energy += weight * 0.5 * A_[i] * (strain[i] - referenceStrain[i]) * (strain[i] - referenceStrain[i]);
            elementEnergy += weight * 0.5 * A_[i] * (strain[i] - referenceStrain[i]) * (strain[i] - referenceStrain[i]);
        }
    }
    
    // Get quadrature rule
    const int polOrd = 2;
    const Dune::QuadratureRule<double, 1>& bendingQuad = Dune::QuadratureRules<double, 1>::rule(element->type(), polOrd);
    
    for (size_t pt=0; pt<bendingQuad.size(); pt++) {
        
        // Local position of the quadrature point
        const Dune::FieldVector<double,1>& quadPos = bendingQuad[pt].position();
        
        double weight = bendingQuad[pt].weight() * element->geometry().integrationElement(quadPos);
        
        Dune::FieldVector<double,6> strain = getStrain(localSolution, element, quadPos);
        
        // The reference strain
        Dune::FieldVector<double,6> referenceStrain = getStrain(localReferenceConfiguration, element, quadPos);
        
        // Part II: the bending and twisting energy
        for (int i=0; i<3; i++) {
            energy += weight * 0.5 * K_[i] * (strain[i+3] - referenceStrain[i+3]) * (strain[i+3] - referenceStrain[i+3]);
            elementEnergy += weight * 0.5 * K_[i] * (strain[i+3] - referenceStrain[i+3]) * (strain[i+3] - referenceStrain[i+3]);
        }
        
    }

    return energy;
}

template <class GridType, class RT>
Dune::FieldVector<double, 6> RodLocalStiffness<GridType, RT>::
getStrain(const std::vector<Configuration>& localSolution,
          const EntityPointer& element,
          const Dune::FieldVector<double,1>& pos) const
{
    if (!element->isLeaf())
        DUNE_THROW(Dune::NotImplemented, "Only for leaf elements");

    assert(localSolution.size() == 2);

    // Strain defined on each element
    Dune::FieldVector<double, 6> strain(0);

    // Extract local solution on this element
    const Dune::LagrangeShapeFunctionSet<double, double, 1> & baseSet 
        = Dune::LagrangeShapeFunctions<double, double, 1>::general(element->type(), 1);
    int numOfBaseFct = baseSet.size();
    
    const Dune::FieldMatrix<double,1,1>& inv = element->geometry().jacobianInverseTransposed(pos);
    
    // ///////////////////////////////////////
    //   Compute deformation gradient
    // ///////////////////////////////////////
    Dune::FieldVector<double,1> shapeGrad[numOfBaseFct];
        
    for (int dof=0; dof<numOfBaseFct; dof++) {
            
        for (int i=0; i<1; i++)
            shapeGrad[dof][i] = baseSet[dof].evaluateDerivative(0,i,pos);
        
        // multiply with jacobian inverse 
        Dune::FieldVector<double,1> tmp(0);
        inv.umv(shapeGrad[dof], tmp);
        shapeGrad[dof] = tmp;
        
    }
        
    // //////////////////////////////////
    //   Interpolate
    // //////////////////////////////////
        
    Dune::FieldVector<double,3> r_s;
    for (int i=0; i<3; i++)
        r_s[i] = localSolution[0].r[i]*shapeGrad[0][0] + localSolution[1].r[i]*shapeGrad[1][0];
        
    // Interpolate the rotation at the quadrature point
    Quaternion<double> q = Quaternion<double>::interpolate(localSolution[0].q, localSolution[1].q, pos);
        
    // Get the derivative of the rotation at the quadrature point by interpolating in $H$
    Quaternion<double> q_s = Quaternion<double>::interpolateDerivative(localSolution[0].q, localSolution[1].q,
                                                                       pos, 1/shapeGrad[1]);
        
    // /////////////////////////////////////////////
    //   Sum it all up
    // /////////////////////////////////////////////
        
    // Part I: the shearing and stretching strain
    strain[0] = r_s * q.director(0);    // shear strain
    strain[1] = r_s * q.director(1);    // shear strain
    strain[2] = r_s * q.director(2);    // stretching strain
        
    // Part II: the Darboux vector
        
    Dune::FieldVector<double,3> u = darboux(q, q_s);
    
    strain[3] = u[0];
    strain[4] = u[1];
    strain[5] = u[2];

    return strain;
}


template <class GridType>
void Dune::RodAssembler<GridType>::
getNeighborsPerVertex(MatrixIndexSet& nb) const
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
                
                int iIdx = indexSet.template subIndex<gridDim>(*it,i);
                int jIdx = indexSet.template subIndex<gridDim>(*it,j);
                
                nb.add(iIdx, jIdx);
                
            }
            
        }
        
    }
    
}


template <class GridType>
void Dune::RodAssembler<GridType>::
assembleMatrix(const std::vector<Configuration>& sol,
               BCRSMatrix<MatrixBlock>& matrix) const
{
    const typename GridType::Traits::LevelIndexSet& indexSet = grid_->levelIndexSet(grid_->maxLevel());

    MatrixIndexSet neighborsPerVertex;
    getNeighborsPerVertex(neighborsPerVertex);
    
    matrix = 0;
    
    ElementIterator it    = grid_->template lbegin<0>( grid_->maxLevel() );
    ElementIterator endit = grid_->template lend<0> ( grid_->maxLevel() );

    Matrix<MatrixBlock> mat;
    
    for( ; it != endit; ++it ) {
        
        const LagrangeShapeFunctionSet<double, double, gridDim> & baseSet 
            = Dune::LagrangeShapeFunctions<double, double, gridDim>::general(it->type(), elementOrder);
        const int numOfBaseFct = baseSet.size();  
        
        mat.setSize(numOfBaseFct, numOfBaseFct);

        // Extract local solution
        std::vector<Configuration> localSolution(numOfBaseFct);
        
        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[indexSet.template subIndex<gridDim>(*it,i)];

        // setup matrix 
        getLocalMatrix( it, localSolution, sol, numOfBaseFct, mat);
        
        // Add element matrix to global stiffness matrix
        for(int i=0; i<numOfBaseFct; i++) { 
            
            int row = indexSet.template subIndex<gridDim>(*it,i);

            for (int j=0; j<numOfBaseFct; j++ ) {
                
                int col = indexSet.template subIndex<gridDim>(*it,j);
                matrix[row][col] += mat[i][j];
                
            }
        }

    }

}


template <class GridType>
template <class MatrixType>
void Dune::RodAssembler<GridType>::
getLocalMatrix( EntityPointer &entity, 
                const std::vector<Configuration>& localSolution,
                const std::vector<Configuration>& globalSolution,
                const int matSize, MatrixType& localMat) const
{
    /* ndof is the number of vectors of the element */
    int ndof = matSize;

    for (int i=0; i<matSize; i++)
        for (int j=0; j<matSize; j++)
            localMat[i][j] = 0;
    
    const LagrangeShapeFunctionSet<double, double, gridDim> & baseSet 
        = Dune::LagrangeShapeFunctions<double, double, gridDim>::general(entity->type(), elementOrder);

    // Get quadrature rule
    int polOrd = 2;
    const QuadratureRule<double, gridDim>& quad = QuadratureRules<double, gridDim>::rule(entity->type(), polOrd);
    
    /* Loop over all integration points */
    for (size_t ip=0; ip<quad.size(); ip++) {
        
        // Local position of the quadrature point
        const FieldVector<double,gridDim>& quadPos = quad[ip].position();

        // calc Jacobian inverse before integration element is evaluated 
        const FieldMatrix<double,gridDim,gridDim>& inv = entity->geometry().jacobianInverseTransposed(quadPos);
        const double integrationElement = entity->geometry().integrationElement(quadPos);
        
        /* Compute the weight of the current integration point */
        double weight = quad[ip].weight() * integrationElement;
        
        /**********************************************/
        /* compute gradients of the shape functions   */
        /**********************************************/
        FieldVector<double,gridDim> shapeGrad[2];
        FieldVector<double,gridDim> tmp;
        
        for (int dof=0; dof<ndof; dof++) {
            
            for (int i=0; i<gridDim; i++)
                tmp[i] = baseSet[dof].evaluateDerivative(0,i,quadPos);
            
            // multiply with jacobian inverse 
            shapeGrad[dof] = 0;
            inv.umv(tmp,shapeGrad[dof]);

        }
        
        
        FieldVector<double,1> shapeFunction[matSize];
        for(int i=0; i<matSize; i++) 
            shapeFunction[i] = baseSet[i].evaluateFunction(0,quadPos);

        // //////////////////////////////////
        //   Interpolate
        // //////////////////////////////////
        
        FieldVector<double,3> r_s;
        r_s[0] = localSolution[0].r[0]*shapeGrad[0] + localSolution[1].r[0]*shapeGrad[1];
        r_s[1] = localSolution[0].r[1]*shapeGrad[0] + localSolution[1].r[1]*shapeGrad[1];
        r_s[2] = localSolution[0].r[2]*shapeGrad[0] + localSolution[1].r[2]*shapeGrad[1];

        // Interpolate current rotation at this quadrature point
        Quaternion<double> q = Quaternion<double>::interpolate(localSolution[0].q, localSolution[1].q,quadPos[0]);

        // Contains \partial q / \partial v^i_j  at v = 0
        array<Quaternion<double>,3> dq_dvj;
        Quaternion<double> dq_dvij_ds[2][3];
        for (int i=0; i<2; i++)
            for (int j=0; j<3; j++) {
                
                for (int m=0; m<3; m++) {
                    dq_dvj[j][m]    = (j==m) * 0.5;
                    dq_dvij_ds[i][j][m] = (j==m) * 0.5 * shapeGrad[i];
                }
                
                dq_dvj[j][3]    = 0;
                dq_dvij_ds[i][j][3] = 0;
            }

        Quaternion<double> dq_dvj_dvl[3][3];
        Quaternion<double> dq_dvij_dvkl_ds[2][3][2][3];
        for (int i=0; i<2; i++) {
            
            for (int j=0; j<3; j++) {
                
                for (int k=0; k<2; k++) {
            
                    for (int l=0; l<3; l++) {

                        for (int m=0; m<3; m++) {
                            dq_dvj_dvl[j][l][m] = 0;
                            dq_dvij_dvkl_ds[i][j][k][l][m] = 0;
                        }

                        dq_dvj_dvl[j][l][3]    = -0.25 * (j==l);
                        dq_dvij_dvkl_ds[i][j][k][l][3] = -0.25 * (j==l) * shapeGrad[i] * shapeGrad[k];

                    }

                }

            }

        }        
        
        // Contains \parder d \parder v^i_j
        array<array<FieldVector<double,3>, 3>, 3> dd_dvj;
        q.getFirstDerivativesOfDirectors(dd_dvj);

        // Contains \parder {dm}{v^i_j}{v^k_l}
        FieldVector<double,3> dd_dvij_dvkl[3][3][3];
        
        for (int j=0; j<3; j++) {
            
            for (int l=0; l<3; l++) {
                
                FieldMatrix<double,4,4> A;
                for (int a=0; a<4; a++)
                    for (int b=0; b<4; b++) 
                        A[a][b] = (q.mult(dq_dvj[l]))[a] * (q.mult(dq_dvj[j]))[b]
                            + q[a] * q.mult(dq_dvj_dvl[l][j])[b];
                
                // d1
                dd_dvij_dvkl[0][j][l][0] = A[0][0] - A[1][1] - A[2][2] + A[3][3];
                dd_dvij_dvkl[0][j][l][1] = A[1][0] + A[0][1] + A[3][2] + A[2][3];
                dd_dvij_dvkl[0][j][l][2] = A[2][0] + A[0][2] - A[3][1] - A[1][3];
                
                // d2
                dd_dvij_dvkl[1][j][l][0] =  A[1][0] + A[0][1] - A[3][2] - A[2][3];
                dd_dvij_dvkl[1][j][l][1] = -A[0][0] + A[1][1] - A[2][2] + A[3][3];
                dd_dvij_dvkl[1][j][l][2] =  A[2][1] + A[1][2] + A[3][0] + A[0][3];
                
                // d3
                dd_dvij_dvkl[2][j][l][0] =  A[2][0] + A[0][2] + A[3][1] + A[1][3];
                dd_dvij_dvkl[2][j][l][1] =  A[2][1] + A[1][2] - A[3][0] - A[0][3];
                dd_dvij_dvkl[2][j][l][2] = -A[0][0] - A[1][1] + A[2][2] + A[3][3];
                
                
                dd_dvij_dvkl[0][j][l] *= 2;
                dd_dvij_dvkl[1][j][l] *= 2;
                dd_dvij_dvkl[2][j][l] *= 2;
                
            }
            
        }

        // Get the derivative of the rotation at the quadrature point by interpolating in $H$
        Quaternion<double> q_s = Quaternion<double>::interpolateDerivative(localSolution[0].q, localSolution[1].q,
                                                                           quadPos, 1/shapeGrad[1]);
        
        // The current strain
        FieldVector<double,blocksize> strain = getStrain(globalSolution, entity, quadPos);
        
        // The reference strain
        FieldVector<double,blocksize> referenceStrain = getStrain(referenceConfiguration_, entity, quadPos);

        // Contains \partial u (canonical) / \partial v^i_j  at v = 0
        FieldVector<double,3> du_dvij[2][3];
        FieldVector<double,3> du_dvij_dvkl[2][3][2][3];

        for (int i=0; i<2; i++) {

            for (int j=0; j<3; j++) {

                for (int m=0; m<3; m++) {
                    du_dvij[i][j][m]  = (q.mult(dq_dvj[j])).B(m)*q_s;
                    du_dvij[i][j][m] *= 2 * shapeFunction[i];
                    Quaternion<double> tmp = dq_dvj[j];
                    tmp *= shapeFunction[i];
                    du_dvij[i][j][m] += 2 * ( q.B(m)*(q.mult(dq_dvij_ds[i][j]) + q_s.mult(tmp)));
                }

#if 0
                for (int k=0; k<2; k++) {

                    for (int l=0; l<3; l++) {

                        Quaternion<double> tmp_ij = dq_dvj[j];
                        Quaternion<double> tmp_kl = dq_dvj[l];

                        tmp_ij *= shapeFunction[i];
                        tmp_kl *= shapeFunction[k];

                        Quaternion<double> tmp_ijkl = dq_dvj_dvl[j][l];
                        tmp_ijkl *= shapeFunction[i] * shapeFunction[k];

                        for (int m=0; m<3; m++) {

                            du_dvij_dvkl[i][j][k][l][m] = 2 * ( (q.mult(tmp_ijkl)).B(m) * q_s)
                                + 2 * ( (q.mult(tmp_ij)).B(m) * (q.mult(dq_dvij_ds[k][l]) + q_s.mult(tmp_kl)))
                                + 2 * ( (q.mult(tmp_kl)).B(m) * (q.mult(dq_dvij_ds[i][j]) + q_s.mult(tmp_ij)))
                                + 2 * ( q.B(m) * (q.mult(dq_dvij_dvkl_ds[i][j][k][l]) + q_s.mult(tmp_ijkl)));

                        }

                    }

                }
#else
#warning Using fd approximation of rotation strain Hessian.
                Dune::array<Dune::Matrix<Dune::FieldMatrix<double,3,3> >, 3> rotationHessian;
                rotationStrainHessian(localSolution, quadPos[0], shapeGrad, shapeFunction, rotationHessian);
                for (int k=0; k<2; k++)
                    for (int l=0; l<3; l++)
                        for (int m=0; m<3; m++)
                            du_dvij_dvkl[i][j][k][l][m] = rotationHessian[m][i][k][j][l];
#endif     
           
            }

        }

        // ///////////////////////////////////
        //   Sum it all up
        // ///////////////////////////////////
        array<FieldMatrix<double,2,6>, 6> strainDerivatives;
        strainDerivative(localSolution, quadPos[0], shapeGrad, shapeFunction, strainDerivatives);

        for (int i=0; i<matSize; i++) {

            for (int k=0; k<matSize; k++) {

                for (int j=0; j<3; j++) {

                    for (int l=0; l<3; l++) {

                        for (int m=0; m<3; m++) {

                            // ////////////////////////////////////////////
                            //   The translational part
                            // ////////////////////////////////////////////
                        
                            // \partial W^2 \partial r^i_j \partial r^k_l
                            localMat[i][k][j][l] += weight 
                                * A_[m] * shapeGrad[i] * q.director(m)[j] * shapeGrad[k] * q.director(m)[l];

                            // \partial W^2 \partial v^i_j \partial r^k_l
                            localMat[i][k][j+3][l] += weight
                                * ( A_[m] * strainDerivatives[m][k][l] * strainDerivatives[m][i][j+3]
                                    + A_[m] * (strain[m] - referenceStrain[m]) * shapeGrad[k] * dd_dvj[m][j][l]*shapeFunction[i]);

                            // \partial W^2 \partial r^i_j \partial v^k_l
                            localMat[i][k][j][l+3] += weight
                                * ( A_[m] * strainDerivatives[m][i][j] * strainDerivatives[m][k][l+3]
                                    + A_[m] * (strain[m] - referenceStrain[m]) * shapeGrad[i] * dd_dvj[m][l][j]*shapeFunction[k]);

                            // \partial W^2 \partial v^i_j \partial v^k_l
                            localMat[i][k][j+3][l+3] += weight
                                * (  A_[m] * (r_s * dd_dvj[m][l]*shapeFunction[k]) * (r_s * dd_dvj[m][j]*shapeFunction[i])
                                     + A_[m] * (strain[m] - referenceStrain[m]) * (r_s * dd_dvij_dvkl[m][j][l]) * shapeFunction[i] * shapeFunction[k]);

                            // ////////////////////////////////////////////
                            //   The rotational part
                            // ////////////////////////////////////////////
                            
                            // \partial W^2 \partial v^i_j \partial v^k_l
                            // All other derivatives are zero

                            double sum = du_dvij[k][l][m] * du_dvij[i][j][m];
                            
                            sum += (strain[m+3] - referenceStrain[m+3]) * du_dvij_dvkl[i][j][k][l][m];
                       
                            localMat[i][k][j+3][l+3] += weight *K_[m] * sum;

                        }

                    }

                }

            }
        
        }

    }

}

template <class GridType>
void Dune::RodAssembler<GridType>::
assembleMatrixFD(const std::vector<Configuration>& sol,
                 BCRSMatrix<MatrixBlock>& matrix) const
{
    double eps = 1e-2;

    typedef typename Dune::BCRSMatrix<Dune::FieldMatrix<double,6,6> >::row_type::iterator ColumnIterator;

    // ///////////////////////////////////////////////////////////
    //   Compute gradient by finite-difference approximation
    // ///////////////////////////////////////////////////////////
    std::vector<Configuration> forwardSolution = sol;
    std::vector<Configuration> backwardSolution = sol;

    std::vector<Configuration> forwardForwardSolution = sol;
    std::vector<Configuration> forwardBackwardSolution = sol;
    std::vector<Configuration> backwardForwardSolution = sol;
    std::vector<Configuration> backwardBackwardSolution = sol;

    // ////////////////////////////////////////////////////
    //   Create local assembler
    // ////////////////////////////////////////////////////

    Dune::array<double,3> K = {K_[0], K_[1], K_[2]};
    Dune::array<double,3> A = {A_[0], A_[1], A_[2]};
    RodLocalStiffness<GridType,double> localStiffness(K, A);

    // //////////////////////////////////////////////////////////////////
    //   Store pointers to all elements so we can access them by index
    // //////////////////////////////////////////////////////////////////

    const typename GridType::Traits::LeafIndexSet& indexSet = grid_->leafIndexSet();

    std::vector<EntityPointer> elements(grid_->size(0),grid_->template leafbegin<0>());
    
    typename GridType::template Codim<0>::LeafIterator eIt    = grid_->template leafbegin<0>();
    typename GridType::template Codim<0>::LeafIterator eEndIt = grid_->template leafend<0>();

    for (; eIt!=eEndIt; ++eIt)
        elements[indexSet.index(*eIt)] = eIt;

    std::vector<Configuration> localReferenceConfiguration(2);
    std::vector<Configuration> localSolution(2);

    // ///////////////////////////////////////////////////////////////
    //   Loop over all blocks of the outer matrix
    // ///////////////////////////////////////////////////////////////
    for (int i=0; i<matrix.N(); i++) {

        ColumnIterator cIt    = matrix[i].begin();
        ColumnIterator cEndIt = matrix[i].end();

        for (; cIt!=cEndIt; ++cIt) {

            // ////////////////////////////////////////////////////////////////////////////
            //   Compute a finite-difference approximation of this hessian matrix block
            // ////////////////////////////////////////////////////////////////////////////

            for (int j=0; j<6; j++) {

                for (int k=0; k<6; k++) {

                    if (i==cIt.index() && j==k) {

                        double forwardEnergy = 0;
                        double solutionEnergy = 0;
                        double backwardEnergy = 0;

                        infinitesimalVariation(forwardSolution[i], eps, j);
                        infinitesimalVariation(backwardSolution[i], -eps, j);

                        if (i != grid_->size(1)-1) {

                            localReferenceConfiguration[0] = referenceConfiguration_[i];
                            localReferenceConfiguration[1] = referenceConfiguration_[i+1];

                            localSolution[0] = forwardSolution[i];
                            localSolution[1] = forwardSolution[i+1];

                            forwardEnergy += localStiffness.energy(elements[i], localSolution, localReferenceConfiguration);

                            localSolution[0] = sol[i];
                            localSolution[1] = sol[i+1];

                            solutionEnergy += localStiffness.energy(elements[i], localSolution, localReferenceConfiguration);

                            localSolution[0] = backwardSolution[i];
                            localSolution[1] = backwardSolution[i+1];

                            backwardEnergy += localStiffness.energy(elements[i], localSolution, localReferenceConfiguration);

                        } 

                        if (i != 0) {

                            localReferenceConfiguration[0] = referenceConfiguration_[i-1];
                            localReferenceConfiguration[1] = referenceConfiguration_[i];

                            localSolution[0] = forwardSolution[i-1];
                            localSolution[1] = forwardSolution[i];

                            forwardEnergy += localStiffness.energy(elements[i-1], localSolution, localReferenceConfiguration);

                            localSolution[0] = sol[i-1];
                            localSolution[1] = sol[i];

                            solutionEnergy += localStiffness.energy(elements[i-1], localSolution, localReferenceConfiguration);

                            localSolution[0] = backwardSolution[i-1];
                            localSolution[1] = backwardSolution[i];

                            backwardEnergy += localStiffness.energy(elements[i-1], localSolution, localReferenceConfiguration);

                        } 

                        // Second derivative
                        (*cIt)[j][k] = (forwardEnergy - 2*solutionEnergy + backwardEnergy) / (eps*eps);
                        
                        forwardSolution[i]  = sol[i];
                        backwardSolution[i] = sol[i];

                    } else {

                        double forwardForwardEnergy = 0;
                        double forwardBackwardEnergy = 0;
                        double backwardForwardEnergy = 0;
                        double backwardBackwardEnergy = 0;

                        infinitesimalVariation(forwardForwardSolution[i],             eps, j);
                        infinitesimalVariation(forwardForwardSolution[cIt.index()],   eps, k);
                        infinitesimalVariation(forwardBackwardSolution[i],            eps, j);
                        infinitesimalVariation(forwardBackwardSolution[cIt.index()], -eps, k);
                        infinitesimalVariation(backwardForwardSolution[i],           -eps, j);
                        infinitesimalVariation(backwardForwardSolution[cIt.index()],  eps, k);
                        infinitesimalVariation(backwardBackwardSolution[i],          -eps, j);
                        infinitesimalVariation(backwardBackwardSolution[cIt.index()],-eps, k);

                        std::set<int> elementsInvolved;
                        if (i>0)
                            elementsInvolved.insert(i-1);
                        if (i<grid_->size(1)-1)
                            elementsInvolved.insert(i);
                        if (cIt.index()>0)
                            elementsInvolved.insert(cIt.index()-1);
                        if (cIt.index()<grid_->size(1)-1)
                            elementsInvolved.insert(cIt.index());

                        for (typename std::set<int>::iterator it = elementsInvolved.begin();
                             it != elementsInvolved.end();
                             ++it) {

                            localReferenceConfiguration[0] = referenceConfiguration_[*it];
                            localReferenceConfiguration[1] = referenceConfiguration_[*it+1];

                            localSolution[0] = forwardForwardSolution[*it];
                            localSolution[1] = forwardForwardSolution[*it+1];

                            forwardForwardEnergy += localStiffness.energy(elements[*it], localSolution, localReferenceConfiguration);

                            localSolution[0] = forwardBackwardSolution[*it];
                            localSolution[1] = forwardBackwardSolution[*it+1];

                            forwardBackwardEnergy += localStiffness.energy(elements[*it], localSolution, localReferenceConfiguration);

                            localSolution[0] = backwardForwardSolution[*it];
                            localSolution[1] = backwardForwardSolution[*it+1];

                            backwardForwardEnergy += localStiffness.energy(elements[*it], localSolution, localReferenceConfiguration);

                            localSolution[0] = backwardBackwardSolution[*it];
                            localSolution[1] = backwardBackwardSolution[*it+1];

                            backwardBackwardEnergy += localStiffness.energy(elements[*it], localSolution, localReferenceConfiguration);


                        }

                        (*cIt)[j][k] = (forwardForwardEnergy + backwardBackwardEnergy
                                        - forwardBackwardEnergy - backwardForwardEnergy) / (4*eps*eps);
                        
                        forwardForwardSolution[i]             = sol[i];
                        forwardForwardSolution[cIt.index()]   = sol[cIt.index()];
                        forwardBackwardSolution[i]            = sol[i];
                        forwardBackwardSolution[cIt.index()]  = sol[cIt.index()];
                        backwardForwardSolution[i]            = sol[i];
                        backwardForwardSolution[cIt.index()]  = sol[cIt.index()];
                        backwardBackwardSolution[i]           = sol[i];
                        backwardBackwardSolution[cIt.index()] = sol[cIt.index()];
                        
                    }
                            
                }

            }

        }

    }

}

template <class GridType>
void Dune::RodAssembler<GridType>::
strainDerivative(const std::vector<Configuration>& localSolution,
                 double pos,
                 FieldVector<double,1> shapeGrad[2],
                 FieldVector<double,1> shapeFunction[2],
                 Dune::array<FieldMatrix<double,2,6>, 6>& derivatives) const
{
    assert(localSolution.size()==2);

//     FieldVector<double,1> shapeGrad[2];
//     shapeGrad[0] = -1;
//     shapeGrad[1] =  1;

//     FieldVector<double,1> shapeFunction[2];
//     shapeFunction[0] = 1-pos;
//     shapeFunction[1] =  pos;


    FieldVector<double,3> r_s;
    for (int i=0; i<3; i++)
        r_s[i] = localSolution[0].r[i]*shapeGrad[0] + localSolution[1].r[i]*shapeGrad[1];

    // Interpolate current rotation at this quadrature point
    Quaternion<double> q = Quaternion<double>::interpolate(localSolution[0].q, localSolution[1].q,pos);

    // Contains \parder d \parder v^i_j
    array<array<FieldVector<double,3>, 3>, 3> dd_dvj;
    q.getFirstDerivativesOfDirectors(dd_dvj);

    Quaternion<double> q_s = Quaternion<double>::interpolateDerivative(localSolution[0].q, localSolution[1].q,
                                                                           pos, 1/shapeGrad[1]);

    array<Quaternion<double>,3> dq_dvj;
    Quaternion<double> dq_dvij_ds[2][3];
    for (int i=0; i<2; i++)
        for (int j=0; j<3; j++) {
            
            for (int m=0; m<3; m++) {
                dq_dvj[j][m]    = (j==m) * 0.5;
                dq_dvij_ds[i][j][m] = (j==m) * 0.5 * shapeGrad[i];
            }
            
            dq_dvj[j][3]    = 0;
            dq_dvij_ds[i][j][3] = 0;
        }

    // the strain component
    for (int m=0; m<3; m++) {

        // the shape function
        for (int i=0; i<2; i++) {

            // the partial derivative direction
            for (int j=0; j<3; j++) {

                // parder v_m / \partial r^j_i
                derivatives[m][i][j] = shapeGrad[i]*q.director(m)[j];

                derivatives[m][i][j+3] = r_s *dd_dvj[m][j] * shapeFunction[i];

                // \partial u_m
                derivatives[m+3][i][j] = 0;

                double du_dvij_i_j_m  = 2* ((q.mult(dq_dvj[j])).B(m)*q_s) * shapeFunction[i][0];
                Quaternion<double> tmp = dq_dvj[j];
                tmp *= shapeFunction[i];
#if 1
                du_dvij_i_j_m += 2 * ( q.B(m)*(q.mult(dq_dvij_ds[i][j]) + q_s.mult(tmp)));
#else
#warning Term omitted in strainDerivative
#endif
                derivatives[m+3][i][j+3] = du_dvij_i_j_m;

            }

        }

    }


}


template <class GridType>
void Dune::RodAssembler<GridType>::
strainHessian(const std::vector<Configuration>& localSolution,
              double pos,
              Dune::array<Matrix<FieldMatrix<double,6,6> >, 3>& translationDer,
              Dune::array<Matrix<FieldMatrix<double,3,3> >, 3>& rotationDer) const
{
    assert(localSolution.size()==2);

    FieldVector<double,1> shapeGrad[2];
    shapeGrad[0] = -1;
    shapeGrad[1] =  1;

    FieldVector<double,1> shapeFunction[2];
    shapeFunction[0] = 1-pos;
    shapeFunction[1] =  pos;


    FieldVector<double,3> r_s;
    for (int i=0; i<3; i++)
        r_s[i] = localSolution[0].r[i]*shapeGrad[0] + localSolution[1].r[i]*shapeGrad[1];

    // Interpolate current rotation at this quadrature point
    Quaternion<double> q = Quaternion<double>::interpolate(localSolution[0].q, localSolution[1].q,pos);

    // Contains \parder d \parder v^i_j
    array<array<FieldVector<double,3>, 3>, 3> dd_dvj;
    q.getFirstDerivativesOfDirectors(dd_dvj);

    Quaternion<double> q_s = Quaternion<double>::interpolateDerivative(localSolution[0].q, localSolution[1].q,
                                                                       pos, 1/shapeGrad[1]);

    array<Quaternion<double>,3> dq_dvj;
    Quaternion<double> dq_dvij_ds[2][3];
    for (int i=0; i<2; i++)
        for (int j=0; j<3; j++) {
            
            for (int m=0; m<3; m++) {
                dq_dvj[j][m]    = (j==m) * 0.5;
                dq_dvij_ds[i][j][m] = (j==m) * 0.5 * shapeGrad[i]/* * ((i==0) ? 1-pos[0] : pos[0])*/;
            }
            
            dq_dvj[j][3]    = 0;
            dq_dvij_ds[i][j][3] = 0;
        }

    Quaternion<double> dq_dvj_dvl[3][3];
    Quaternion<double> dq_dvij_dvkl_ds[2][3][2][3];
    for (int i=0; i<2; i++) {
        
        for (int j=0; j<3; j++) {
            
            for (int k=0; k<2; k++) {
                
                for (int l=0; l<3; l++) {
                    
                    for (int m=0; m<3; m++) {
                        dq_dvj_dvl[j][l][m] = 0;
                        dq_dvij_dvkl_ds[i][j][k][l][m] = 0;
                    }
                    
                    dq_dvj_dvl[j][l][3]    = -0.25 * (j==l);
                    dq_dvij_dvkl_ds[i][j][k][l][3] = -0.25 * (j==l) * shapeGrad[i] * shapeGrad[k]
                        /*  * ((i==0) ? 1-pos[0] : pos[0]) * ((k==0) ? 1-pos[0] : pos[0]) */;
                    
                }
                
            }
            
        }
        
    }        

    // the strain component
    for (int m=0; m<3; m++) {

        translationDer[m].setSize(2,2);
        translationDer[m] = 0;

        rotationDer[m].setSize(2,2);
        rotationDer[m] = 0;

        // the shape function
        for (int i=0; i<2; i++) {

            // the partial derivative direction
            for (int j=0; j<3; j++) {

                for (int k=0; k<2; k++) {

                    for (int l=0; l<3; l++) {



                        // //////////////////////////////////////////////////
                        //   The rotation part
                        // //////////////////////////////////////////////////
                        Quaternion<double> tmp_ij = dq_dvj[j];
                        Quaternion<double> tmp_kl = dq_dvj[l];

                        tmp_ij *= shapeFunction[i];
                        tmp_kl *= shapeFunction[k];

                        Quaternion<double> tmp_ijkl = dq_dvj_dvl[j][l];
                        tmp_ijkl *= shapeFunction[i] * shapeFunction[k];

                        rotationDer[m][i][k][j][l]  = 2 * ( (q.mult(tmp_ijkl)).B(m) * q_s);
                        rotationDer[m][i][k][j][l] += 2 * ( (q.mult(tmp_ij)).B(m) * (q.mult(dq_dvij_ds[k][l]) + q_s.mult(tmp_kl)));
#if 1
                        rotationDer[m][i][k][j][l] += 2 * ( (q.mult(tmp_kl)).B(m) * (q.mult(dq_dvij_ds[i][j]) + q_s.mult(tmp_ij)));
                        rotationDer[m][i][k][j][l] += 2 * ( q.B(m) * (q.mult(dq_dvij_dvkl_ds[i][j][k][l]) + q_s.mult(tmp_ijkl)));
#else
#warning Term omitted in strainHessian
#endif

                    }

                }

            }

        }

    }


}

template <class GridType>
void Dune::RodAssembler<GridType>::
rotationStrainHessian(const std::vector<Configuration>& x, 
                      double pos,
                      Dune::FieldVector<double,1> shapeGrad[2],
                      Dune::FieldVector<double,1> shapeFunction[2],
                      Dune::array<Dune::Matrix<Dune::FieldMatrix<double,3,3> >, 3>& rotationDer) const
{
    assert(x.size()==2);
    double eps = 1e-3;

    for (int m=0; m<3; m++) {
        
        rotationDer[m].setSize(2,2);
        rotationDer[m] = 0;

    }

    // ///////////////////////////////////////////////////////////
    //   Compute gradient by finite-difference approximation
    // ///////////////////////////////////////////////////////////
    std::vector<Configuration> forwardSolution = x;
    std::vector<Configuration> backwardSolution = x;
    
    for (int k=0; k<2; k++) {
        
        for (int l=0; l<3; l++) {

 //            infinitesimalVariation(forwardSolution[k], eps, l+3);
//             infinitesimalVariation(backwardSolution[k], -eps, l+3);
            forwardSolution[k].q = forwardSolution[k].q.mult(Quaternion<double>::exp((l==0)*eps, 
                                                                                     (l==1)*eps, 
                                                                                     (l==2)*eps));
            backwardSolution[k].q = backwardSolution[k].q.mult(Quaternion<double>::exp(-(l==0)*eps, 
                                                                                       -(l==1)*eps, 
                                                                                       -(l==2)*eps));
                
            Dune::array<Dune::FieldMatrix<double,2,6>, 6> forwardStrainDer;
            strainDerivative(forwardSolution, pos, shapeGrad, shapeFunction, forwardStrainDer);

            Dune::array<Dune::FieldMatrix<double,2,6>, 6> backwardStrainDer;
            strainDerivative(backwardSolution, pos, shapeGrad, shapeFunction, backwardStrainDer);
            
            for (int i=0; i<2; i++) {
                for (int j=0; j<3; j++) {
                    for (int m=0; m<3; m++) {
                        rotationDer[m][i][k][j][l] = (forwardStrainDer[m][i][j]-backwardStrainDer[m][i][j]) / (2*eps);
                    }
                }
            }

            forwardSolution = x;
            backwardSolution = x;
            
        }
        
    }
    
}


template <class GridType>
void Dune::RodAssembler<GridType>::
assembleGradient(const std::vector<Configuration>& sol,
                 BlockVector<FieldVector<double, blocksize> >& grad) const
{
    const typename GridType::Traits::LevelIndexSet& indexSet = grid_->levelIndexSet(grid_->maxLevel());
    const int maxlevel = grid_->maxLevel();

    if (sol.size()!=grid_->size(maxlevel, gridDim))
        DUNE_THROW(Exception, "Solution vector doesn't match the grid!");

    grad.resize(sol.size());
    grad = 0;

    ElementIterator it    = grid_->template lbegin<0>(maxlevel);
    ElementIterator endIt = grid_->template lend<0>(maxlevel);

    // Loop over all elements
    for (; it!=endIt; ++it) {

        // Extract local solution on this element
        const LagrangeShapeFunctionSet<double, double, gridDim> & baseSet 
            = Dune::LagrangeShapeFunctions<double, double, gridDim>::general(it->type(), elementOrder);
        const int numOfBaseFct = baseSet.size();  
        
        Configuration localSolution[numOfBaseFct];
        
        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[indexSet.template subIndex<gridDim>(*it,i)];

        // Get quadrature rule
        int polOrd = 2;
        const QuadratureRule<double, gridDim>& quad = QuadratureRules<double, gridDim>::rule(it->type(), polOrd);

        for (int pt=0; pt<quad.size(); pt++) {

            // Local position of the quadrature point
            const FieldVector<double,gridDim>& quadPos = quad[pt].position();
            
            const FieldMatrix<double,1,1>& inv = it->geometry().jacobianInverseTransposed(quadPos);
            const double integrationElement = it->geometry().integrationElement(quadPos);
        
            double weight = quad[pt].weight() * integrationElement;
            
            // ///////////////////////////////////////
            //   Compute deformation gradient
            // ///////////////////////////////////////
            double shapeGrad[numOfBaseFct];
            
            for (int dof=0; dof<numOfBaseFct; dof++) {
                
                shapeGrad[dof] = baseSet[dof].evaluateDerivative(0,0,quadPos);

                // multiply with jacobian inverse 
                FieldVector<double,gridDim> tmp(0);
                inv.umv(shapeGrad[dof], tmp);
                shapeGrad[dof] = tmp;
                
            }

            // Get the value of the shape functions
            double shapeFunction[2];
            for(int i=0; i<2; i++) 
                shapeFunction[i] = baseSet[i].evaluateFunction(0,quadPos);

            // //////////////////////////////////
            //   Interpolate
            // //////////////////////////////////

            FieldVector<double,3> r_s;
            r_s[0] = localSolution[0].r[0]*shapeGrad[0] + localSolution[1].r[0]*shapeGrad[1];
            r_s[1] = localSolution[0].r[1]*shapeGrad[0] + localSolution[1].r[1]*shapeGrad[1];
            r_s[2] = localSolution[0].r[2]*shapeGrad[0] + localSolution[1].r[2]*shapeGrad[1];

            // Interpolate current rotation at this quadrature point
            Quaternion<double> hatq = Quaternion<double>::interpolate(localSolution[0].q, localSolution[1].q,quadPos[0]);

            // Get the derivative of the rotation at the quadrature point by interpolating in $H$
            Quaternion<double> hatq_s = Quaternion<double>::interpolateDerivative(localSolution[0].q, localSolution[1].q,
                                                                                  quadPos, 1/shapeGrad[1]);

            // The current strain
            FieldVector<double,blocksize> strain = getStrain(sol, it, quadPos);

            // The reference strain
            FieldVector<double,blocksize> referenceStrain = getStrain(referenceConfiguration_, it, quadPos);

            // Contains \partial q / \partial v^i_j  at v = 0
            array<Quaternion<double>,3> dq_dvj;

            array<Quaternion<double>,3> dq_dvj_ds;

            for (int j=0; j<3; j++) {
                    
                for (int m=0; m<3; m++) {
                    dq_dvj[j][m]     = (j==m) * 0.5;
                    dq_dvj_ds[j][m] = (j==m) * 0.5;
                }

                dq_dvj[j][3]    = 0;
                dq_dvj_ds[j][3] = 0;
            }

            // dd_dvij[k][i][j] = \parder {d_k} {v^i_j}
            array<array<FieldVector<double,3>, 3>, 3> dd_dvj;
            hatq.getFirstDerivativesOfDirectors(dd_dvj);

            
            // /////////////////////////////////////////////
            //   Sum it all up
            // /////////////////////////////////////////////

            for (int i=0; i<numOfBaseFct; i++) {

                int globalDof = indexSet.template subIndex<gridDim>(*it,i);

                // /////////////////////////////////////////////
                //   The translational part
                // /////////////////////////////////////////////
                
                // \partial \bar{W} / \partial r^i_j
                for (int j=0; j<3; j++) {

                    for (int m=0; m<3; m++) 
                        grad[globalDof][j] += weight 
                            * (  A_[m] * (strain[m] - referenceStrain[m]) * shapeGrad[i] * hatq.director(m)[j]);

                }

                // \partial \bar{W}_v / \partial v^i_j
                for (int j=0; j<3; j++) {

                    for (int m=0; m<3; m++) 
                        grad[globalDof][3+j] += weight 
                            * (A_[m] * (strain[m] - referenceStrain[m]) * (r_s*dd_dvj[m][j]*shapeFunction[i]));

                }

                // /////////////////////////////////////////////
                //   The rotational part
                // /////////////////////////////////////////////
                
                // \partial \bar{W}_v / \partial v^i_j
                for (int j=0; j<3; j++) {

                    for (int m=0; m<3; m++) {

                        double du_dvij_m;

                        du_dvij_m = (hatq.mult(dq_dvj[j])).B(m) * hatq_s;
                        du_dvij_m *= shapeFunction[i];

                        Quaternion<double> tmp = dq_dvj[j];
                        tmp *= shapeFunction[i];
                        Quaternion<double> tmp_ds = dq_dvj_ds[j];
                        tmp_ds *= shapeGrad[i];
                        
                        du_dvij_m += hatq.B(m) * (hatq.mult(tmp_ds) + hatq_s.mult(tmp));

                        du_dvij_m *= 2;

                        grad[globalDof][3+j] += weight * K_[m] 
                            * (strain[m+3]-referenceStrain[m+3]) * du_dvij_m;

                    }

                }

            }

        }

    }

}


template <class GridType>
double Dune::RodAssembler<GridType>::
computeEnergy(const std::vector<Configuration>& sol) const
{
    double energy = 0;
    
    const typename GridType::Traits::LeafIndexSet& indexSet = grid_->leafIndexSet();

    if (sol.size()!=indexSet.size(gridDim))
        DUNE_THROW(Exception, "Solution vector doesn't match the grid!");

    ElementLeafIterator it    = grid_->template leafbegin<0>();
    ElementLeafIterator endIt = grid_->template leafend<0>();

    // Loop over all elements
    for (; it!=endIt; ++it) {
        double elementEnergy = 0;
        // Extract local solution on this element
        const LagrangeShapeFunctionSet<double, double, gridDim> & baseSet 
            = Dune::LagrangeShapeFunctions<double, double, gridDim>::general(it->type(), elementOrder);
        int numOfBaseFct = baseSet.size();

        Configuration localSolution[numOfBaseFct];
        
        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[indexSet.template subIndex<gridDim>(*it,i)];

        // ///////////////////////////////////////////////////////////////////////////////
        //   The following two loops are a reduced integration scheme.  We integrate
        //   the transverse shear and extensional energy with a first-order quadrature
        //   formula, even though it should be second order.  This prevents shear-locking
        // ///////////////////////////////////////////////////////////////////////////////
        const int shearingPolOrd = 2;
        const QuadratureRule<double, gridDim>& shearingQuad 
            = QuadratureRules<double, gridDim>::rule(it->type(), shearingPolOrd);

        for (size_t pt=0; pt<shearingQuad.size(); pt++) {

            // Local position of the quadrature point
            const FieldVector<double,gridDim>& quadPos = shearingQuad[pt].position();
            
            const double integrationElement = it->geometry().integrationElement(quadPos);
        
            double weight = shearingQuad[pt].weight() * integrationElement;
            
            FieldVector<double,blocksize> strain = getStrain(sol, it, quadPos);

            // The reference strain
            FieldVector<double,blocksize> referenceStrain = getStrain(referenceConfiguration_, it, quadPos);

            for (int i=0; i<3; i++) {
                energy += weight * 0.5 * A_[i] * (strain[i] - referenceStrain[i]) * (strain[i] - referenceStrain[i]);
                elementEnergy += weight * 0.5 * A_[i] * (strain[i] - referenceStrain[i]) * (strain[i] - referenceStrain[i]);
            }
        }

        // Get quadrature rule
        const int polOrd = 2;
        const QuadratureRule<double, gridDim>& bendingQuad = QuadratureRules<double, gridDim>::rule(it->type(), polOrd);

        for (size_t pt=0; pt<bendingQuad.size(); pt++) {

            // Local position of the quadrature point
            const FieldVector<double,gridDim>& quadPos = bendingQuad[pt].position();
            
            double weight = bendingQuad[pt].weight() * it->geometry().integrationElement(quadPos);
            
            FieldVector<double,blocksize> strain = getStrain(sol, it, quadPos);

            // The reference strain
            FieldVector<double,blocksize> referenceStrain = getStrain(referenceConfiguration_, it, quadPos);

            // Part II: the bending and twisting energy
            for (int i=0; i<3; i++) {
                energy += weight * 0.5 * K_[i] * (strain[i+3] - referenceStrain[i+3]) * (strain[i+3] - referenceStrain[i+3]);
                elementEnergy += weight * 0.5 * K_[i] * (strain[i+3] - referenceStrain[i+3]) * (strain[i+3] - referenceStrain[i+3]);
            }

        }

        //std::cout << "ElementEnergy: " << elementEnergy << std::endl;
    }

    return energy;

}


template <class GridType>
void Dune::RodAssembler<GridType>::
getStrain(const std::vector<Configuration>& sol,
          BlockVector<FieldVector<double, blocksize> >& strain) const
{
    const typename GridType::Traits::LeafIndexSet& indexSet = grid_->leafIndexSet();

    if (sol.size()!=indexSet.size(gridDim))
        DUNE_THROW(Exception, "Solution vector doesn't match the grid!");

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

        Configuration localSolution[numOfBaseFct];
        
        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[indexSet.template subIndex<gridDim>(*it,i)];

        // Get quadrature rule
        const int polOrd = 2;
        const QuadratureRule<double, gridDim>& quad = QuadratureRules<double, gridDim>::rule(it->type(), polOrd);

        for (int pt=0; pt<quad.size(); pt++) {

            // Local position of the quadrature point
            const FieldVector<double,gridDim>& quadPos = quad[pt].position();

            double weight = quad[pt].weight() * it->geometry().integrationElement(quadPos);

            FieldVector<double,blocksize> localStrain = getStrain(sol, it, quad[pt].position());
            
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
Dune::FieldVector<double, 6> Dune::RodAssembler<GridType>::getStrain(const std::vector<Configuration>& sol,
                                                                       const EntityPointer& element,
                                                                       double pos) const
{
    if (!element->isLeaf())
        DUNE_THROW(Dune::NotImplemented, "Only for leaf elements");

    const typename GridType::Traits::LeafIndexSet& indexSet = grid_->leafIndexSet();

    if (sol.size()!=indexSet.size(gridDim))
        DUNE_THROW(Exception, "Configuration vector doesn't match the grid!");

    // Strain defined on each element
    FieldVector<double, blocksize> strain(0);

    // Extract local solution on this element
    const LagrangeShapeFunctionSet<double, double, gridDim> & baseSet 
        = Dune::LagrangeShapeFunctions<double, double, gridDim>::general(element->type(), elementOrder);
    int numOfBaseFct = baseSet.size();
    
    Configuration localSolution[numOfBaseFct];
    
    for (int i=0; i<numOfBaseFct; i++)
        localSolution[i] = sol[indexSet.template subIndex<gridDim>(*element,i)];
    
    const FieldMatrix<double,1,1>& inv = element->geometry().jacobianInverseTransposed(pos);
    
    // ///////////////////////////////////////
    //   Compute deformation gradient
    // ///////////////////////////////////////
    FieldVector<double,gridDim> shapeGrad[numOfBaseFct];
        
    for (int dof=0; dof<numOfBaseFct; dof++) {
            
        for (int i=0; i<gridDim; i++)
            shapeGrad[dof][i] = baseSet[dof].evaluateDerivative(0,i,pos);
        //std::cout << "Gradient " << dof << ": " << shape_grads[dof] << std::endl;
        
        // multiply with jacobian inverse 
        FieldVector<double,gridDim> tmp(0);
        inv.umv(shapeGrad[dof], tmp);
        shapeGrad[dof] = tmp;
        //std::cout << "Gradient " << dof << ": " << shape_grads[dof] << std::endl;
        
    }
        
    // //////////////////////////////////
    //   Interpolate
    // //////////////////////////////////
        
    FieldVector<double,3> r_s;
    r_s[0] = localSolution[0].r[0]*shapeGrad[0][0] + localSolution[1].r[0]*shapeGrad[1][0];
    r_s[1] = localSolution[0].r[1]*shapeGrad[0][0] + localSolution[1].r[1]*shapeGrad[1][0];
    r_s[2] = localSolution[0].r[2]*shapeGrad[0][0] + localSolution[1].r[2]*shapeGrad[1][0];
        
    // Interpolate the rotation at the quadrature point
    Quaternion<double> q = Quaternion<double>::interpolate(localSolution[0].q, localSolution[1].q, pos);
        
    // Get the derivative of the rotation at the quadrature point by interpolating in $H$
    Quaternion<double> q_s = Quaternion<double>::interpolateDerivative(localSolution[0].q, localSolution[1].q,
                                                                       pos, 1/shapeGrad[1]);
        
    // /////////////////////////////////////////////
    //   Sum it all up
    // /////////////////////////////////////////////
        
    // Part I: the shearing and stretching strain
    //std::cout << "tangent : " << r_s << std::endl;
    strain[0] = r_s * q.director(0);    // shear strain
    strain[1] = r_s * q.director(1);    // shear strain
    strain[2] = r_s * q.director(2);    // stretching strain
        
    //std::cout << "strain : " << v << std::endl;
        
    // Part II: the Darboux vector
        
    FieldVector<double,3> u = darboux(q, q_s);
    
    strain[3] = u[0];
    strain[4] = u[1];
    strain[5] = u[2];

    return strain;
}

template <class GridType>
Dune::FieldVector<double,3> Dune::RodAssembler<GridType>::
getResultantForce(const BoundaryPatch<GridType>& boundary, 
                  const std::vector<Configuration>& sol,
                  FieldVector<double,3>& canonicalTorque) const
{
    const typename GridType::Traits::LeafIndexSet& indexSet = grid_->leafIndexSet();

    if (sol.size()!=indexSet.size(gridDim))
        DUNE_THROW(Exception, "Solution vector doesn't match the grid!");

    FieldVector<double,3> canonicalStress(0);
    canonicalTorque = 0;

    // Loop over the given boundary
    ElementLeafIterator eIt    = grid_->template leafbegin<0>();
    ElementLeafIterator eEndIt = grid_->template leafend<0>();

    for (; eIt!=eEndIt; ++eIt) {

        typename EntityType::LeafIntersectionIterator nIt    = eIt->ileafbegin();
        typename EntityType::LeafIntersectionIterator nEndIt = eIt->ileafend();

        for (; nIt!=nEndIt; ++nIt) {

            if (!boundary.contains(*eIt, nIt.numberInSelf()))
                continue;

            // //////////////////////////////////////////////
            //   Compute force across this boundary face
            // //////////////////////////////////////////////

            
            double pos = nIt.intersectionSelfLocal()[0];

            FieldVector<double, blocksize> strain = getStrain(sol, eIt, pos);
            FieldVector<double, blocksize> referenceStrain = getStrain(referenceConfiguration_, eIt, pos);

            FieldVector<double,3> localStress;
            for (int i=0; i<3; i++)
                localStress[i] = (strain[i] - referenceStrain[i]) * A_[i];

            FieldVector<double,3> localTorque;
            for (int i=0; i<3; i++)
                localTorque[i] = (strain[i+3] - referenceStrain[i+3]) * K_[i];

            // Transform stress given with respect to the basis given by the three directors to
            // the canonical basis of R^3
            /** \todo Hardwired: Entry 0 is the leftmost entry! */
            FieldMatrix<double,3,3> orientationMatrix;
            sol[0].q.matrix(orientationMatrix);
            
            orientationMatrix.umv(localStress, canonicalStress);
            
            orientationMatrix.umv(localTorque, canonicalTorque);
            // Reverse transformation to make sure we did the correct thing
 //            assert( std::abs(localStress[0]-canonicalStress*sol[0].q.director(0)) < 1e-6 );
//             assert( std::abs(localStress[1]-canonicalStress*sol[0].q.director(1)) < 1e-6 );
//             assert( std::abs(localStress[2]-canonicalStress*sol[0].q.director(2)) < 1e-6 );
            
        }

    }

    return canonicalStress;
}

