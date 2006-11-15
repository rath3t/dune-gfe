#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>

#include <dune/grid/common/quadraturerules.hh>

#include <dune/disc/shapefunctions/lagrangeshapefunctions.hh>

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
            = Dune::LagrangeShapeFunctions<double, double, gridDim>::general(it->geometry().type(), elementOrder);
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
void Dune::RodAssembler<GridType>::
getFirstDerivativesOfDirectors(const Quaternion<double>& q, 
                               Dune::FixedArray<Dune::FixedArray<Dune::FixedArray<Dune::FieldVector<double,3>, 3>, 2>, 3>& dd_dvij,
                               const Dune::FixedArray<Dune::FixedArray<Quaternion<double>, 3>, 2>& dq_dvij)
{
 
    // Contains \parder d \parder v^i_j
    
    for (int i=0; i<2; i++) {
        
        for (int j=0; j<3; j++) {
            
            // d1
            dd_dvij[0][i][j][0] = q[0]*(dq_dvij[i][j].mult(q))[0] - q[1]*(dq_dvij[i][j].mult(q))[1] 
                - q[2]*(dq_dvij[i][j].mult(q))[2] + q[3]*(dq_dvij[i][j].mult(q))[3];
            
            dd_dvij[0][i][j][1] = (dq_dvij[i][j].mult(q))[0]*q[1] + q[0]*(dq_dvij[i][j].mult(q))[1]
                + (dq_dvij[i][j].mult(q))[2]*q[3] + q[2]*(dq_dvij[i][j].mult(q))[3];
            
            dd_dvij[0][i][j][2] = (dq_dvij[i][j].mult(q))[0]*q[2] + q[0]*(dq_dvij[i][j].mult(q))[2]
                - (dq_dvij[i][j].mult(q))[1]*q[3] - q[1]*(dq_dvij[i][j].mult(q))[3];
            
            // d2
            dd_dvij[1][i][j][0] = (dq_dvij[i][j].mult(q))[0]*q[1] + q[0]*(dq_dvij[i][j].mult(q))[1]
                - (dq_dvij[i][j].mult(q))[2]*q[3] - q[2]*(dq_dvij[i][j].mult(q))[3];
            
            dd_dvij[1][i][j][1] = - q[0]*(dq_dvij[i][j].mult(q))[0] + q[1]*(dq_dvij[i][j].mult(q))[1] 
                - q[2]*(dq_dvij[i][j].mult(q))[2] + q[3]*(dq_dvij[i][j].mult(q))[3];
            
            dd_dvij[1][i][j][2] = (dq_dvij[i][j].mult(q))[1]*q[2] + q[1]*(dq_dvij[i][j].mult(q))[2]
                + (dq_dvij[i][j].mult(q))[0]*q[3] + q[0]*(dq_dvij[i][j].mult(q))[3];
            
            // d3
            dd_dvij[2][i][j][0] = (dq_dvij[i][j].mult(q))[0]*q[2] + q[0]*(dq_dvij[i][j].mult(q))[2]
                + (dq_dvij[i][j].mult(q))[1]*q[3] + q[1]*(dq_dvij[i][j].mult(q))[3];
            
            dd_dvij[2][i][j][1] = (dq_dvij[i][j].mult(q))[1]*q[2] + q[1]*(dq_dvij[i][j].mult(q))[2]
                - (dq_dvij[i][j].mult(q))[0]*q[3] - q[0]*(dq_dvij[i][j].mult(q))[3];
            
            dd_dvij[2][i][j][2] = - q[0]*(dq_dvij[i][j].mult(q))[0] - q[1]*(dq_dvij[i][j].mult(q))[1] 
                + q[2]*(dq_dvij[i][j].mult(q))[2] + q[3]*(dq_dvij[i][j].mult(q))[3];
            
            
            dd_dvij[0][i][j] *= 2;
            dd_dvij[1][i][j] *= 2;
            dd_dvij[2][i][j] *= 2;
            
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
    const typename GridType::Traits::LevelIndexSet& indexSet = grid_->levelIndexSet(grid_->maxLevel());

    /* ndof is the number of vectors of the element */
    int ndof = matSize;

    for (int i=0; i<matSize; i++)
        for (int j=0; j<matSize; j++)
            localMat[i][j] = 0;
    
    const LagrangeShapeFunctionSet<double, double, gridDim> & baseSet 
        = Dune::LagrangeShapeFunctions<double, double, gridDim>::general(entity->geometry().type(), elementOrder);

    // Get quadrature rule
    int polOrd = 2;
    const QuadratureRule<double, gridDim>& quad = QuadratureRules<double, gridDim>::rule(entity->geometry().type(), polOrd);
    
    /* Loop over all integration points */
    for (int ip=0; ip<quad.size(); ip++) {
        
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
        FieldVector<double,gridDim> shapeGrad[ndof];
        FieldVector<double,gridDim> tmp;
        
        for (int dof=0; dof<ndof; dof++) {
            
            for (int i=0; i<gridDim; i++)
                tmp[i] = baseSet[dof].evaluateDerivative(0,i,quadPos);
            
            // multiply with jacobian inverse 
            shapeGrad[dof] = 0;
            inv.umv(tmp,shapeGrad[dof]);

        }
        
        
        double shapeFunction[matSize];
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
        Quaternion<double> hatq = Quaternion<double>::interpolate(localSolution[0].q, localSolution[1].q,quadPos[0]);

        // Contains \partial q / \partial v^i_j  at v = 0
        FixedArray<FixedArray<Quaternion<double>,3>,2> dq_dvij;
        Quaternion<double> dq_dvij_ds[2][3];
        for (int i=0; i<2; i++)
            for (int j=0; j<3; j++) {
                
                for (int m=0; m<3; m++) {
                    dq_dvij[i][j][m]    = (j==m) * 0.5 * shapeFunction[i];
                    dq_dvij_ds[i][j][m] = (j==m) * 0.5 * shapeGrad[i];
                }
                
                dq_dvij[i][j][3]    = 0;
                dq_dvij_ds[i][j][3] = 0;
            }

        Quaternion<double> dq_dvij_dvkl[2][3][2][3];
        Quaternion<double> dq_dvij_dvkl_ds[2][3][2][3];
        for (int i=0; i<2; i++) {
            
            for (int j=0; j<3; j++) {
                
                for (int k=0; k<2; k++) {
            
                    for (int l=0; l<3; l++) {

                        for (int m=0; m<3; m++) {
                            dq_dvij_dvkl[i][j][k][l][m] = 0;
                            dq_dvij_dvkl_ds[i][j][k][l][m] = 0;
                        }

                        dq_dvij_dvkl[i][j][k][l][3] = -0.25 * (j==l) * shapeFunction[i] * shapeFunction[k];
                        dq_dvij_dvkl_ds[i][j][k][l][3] = -0.25 * (j==l) * shapeGrad[i] * shapeGrad[k];

                    }

                }

            }

        }        
        
        // Contains \parder d \parder v^i_j
        FixedArray<FixedArray<FixedArray<FieldVector<double,3>, 3>, 2>, 3> dd_dvij;
        getFirstDerivativesOfDirectors(hatq, dd_dvij, dq_dvij);

        // Contains \parder {dm}{v^i_j}{v^k_l}
        FieldVector<double,3> dd_dvij_dvkl[3][2][3][2][3];
        
        for (int i=0; i<2; i++) {
            
            for (int j=0; j<3; j++) {
                
                for (int k=0; k<2; k++) {
            
                    for (int l=0; l<3; l++) {

                        FieldMatrix<double,4,4> A;
                        for (int a=0; a<4; a++)
                            for (int b=0; b<4; b++) 
                                A[a][b] = (dq_dvij[k][l].mult(hatq))[a] * (dq_dvij[i][j].mult(hatq))[b]
                                    + hatq[a] * dq_dvij_dvkl[i][j][k][l].mult(hatq)[b];
                
                        // d1
                        dd_dvij_dvkl[0][i][j][k][l][0] = A[0][0] - A[1][1] - A[2][2] + A[3][3];
                        dd_dvij_dvkl[0][i][j][k][l][1] = A[1][0] + A[0][1] + A[3][2] + A[2][3];
                        dd_dvij_dvkl[0][i][j][k][l][2] = A[2][0] + A[0][2] - A[3][1] - A[1][3];
                        
                        // d2
                        dd_dvij_dvkl[1][i][j][k][l][0] =  A[1][0] + A[0][1] - A[3][2] - A[2][3];
                        dd_dvij_dvkl[1][i][j][k][l][1] = -A[0][0] + A[1][1] - A[2][2] + A[3][3];
                        dd_dvij_dvkl[1][i][j][k][l][2] =  A[2][1] + A[1][2] + A[3][0] + A[0][3];
                        
                        // d3
                        dd_dvij_dvkl[2][i][j][k][l][0] =  A[2][0] + A[0][2] + A[3][1] + A[1][3];
                        dd_dvij_dvkl[2][i][j][k][l][1] =  A[2][1] + A[1][2] - A[3][0] - A[0][3];
                        dd_dvij_dvkl[2][i][j][k][l][2] = -A[0][0] - A[1][1] + A[2][2] + A[3][3];
                        
                        
                        dd_dvij_dvkl[0][i][j][k][l] *= 2;
                        dd_dvij_dvkl[1][i][j][k][l] *= 2;
                        dd_dvij_dvkl[2][i][j][k][l] *= 2;
                        
                    }
                    
                }

            }

        }

        // Get the derivative of the rotation at the quadrature point by interpolating in $H$
        Quaternion<double> hatq_s;
        hatq_s[0] = localSolution[0].q[0]*shapeGrad[0] + localSolution[1].q[0]*shapeGrad[1];
        hatq_s[1] = localSolution[0].q[1]*shapeGrad[0] + localSolution[1].q[1]*shapeGrad[1];
        hatq_s[2] = localSolution[0].q[2]*shapeGrad[0] + localSolution[1].q[2]*shapeGrad[1];
        hatq_s[3] = localSolution[0].q[3]*shapeGrad[0] + localSolution[1].q[3]*shapeGrad[1];
        
        // The Darboux vector
        //FieldVector<double,3> u = darboux(hatq, hatq_s);
        FieldVector<double,3> darbouxCan = darbouxCanonical(hatq, hatq_s);

        // The current strain
        FieldVector<double,blocksize> strain = getStrain(globalSolution, entity, quadPos);
        
        // The reference strain
        FieldVector<double,blocksize> referenceStrain = getStrain(referenceConfiguration_, entity, quadPos);

        // Contains \partial u (canonical) / \partial v^i_j  at v = 0
        FieldVector<double,3> duCan_dvij[2][3];
        FieldVector<double,3> duLocal_dvij[2][3];
        FieldVector<double,3> duCan_dvij_dvkl[2][3][2][3];

        for (int i=0; i<2; i++) {

            for (int j=0; j<3; j++) {
                
                for (int m=0; m<3; m++) {
                    duCan_dvij[i][j][m]  = 2*(B(m, dq_dvij[i][j].mult(hatq))*hatq_s);
                    duCan_dvij[i][j][m] += 2*(B(m,hatq)*(dq_dvij_ds[i][j].mult(hatq) + dq_dvij[i][j].mult(hatq_s)));
                }

                // Don't fuse the two loops, we need the complete duCan_dvij[i][j]
                for (int m=0; m<3; m++)
                    duLocal_dvij[i][j][m] = duCan_dvij[i][j] * hatq.director(m) + darbouxCan*dd_dvij[m][i][j];

                for (int k=0; k<2; k++) {

                    for (int l=0; l<3; l++) {

                        for (int m=0; m<3; m++)
                            duCan_dvij_dvkl[i][j][k][l] = 2 * (B(m,dq_dvij_dvkl[i][j][k][l].mult(hatq)) * hatq_s)
                                + 2 * (B(m,dq_dvij[i][j].mult(hatq)) * (dq_dvij_ds[k][l].mult(hatq) + dq_dvij[k][l].mult(hatq_s)))
                                + 2 * (B(m,dq_dvij[k][l].mult(hatq)) * (dq_dvij_ds[i][j].mult(hatq) + dq_dvij[i][j].mult(hatq_s)))
                                + 2 * (B(m,hatq) * (dq_dvij_dvkl_ds[i][j][k][l].mult(hatq) + dq_dvij_dvkl[i][j][k][l].mult(hatq_s)));

                    }

                }
                
            }

        }

        // ///////////////////////////////////
        //   Sum it all up
        // ///////////////////////////////////
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
                                * A_[m] * shapeGrad[i] * hatq.director(m)[j] * shapeGrad[k] * hatq.director(m)[l];

                            // \partial W^2 \partial v^i_j \partial r^k_l
                            localMat[i][k][j][l+3] += weight
                                * ( A_[m] * shapeGrad[k]*hatq.director(m)[l]*(r_s*dd_dvij[m][i][j])
                                    + A_[m] * (strain[m] - referenceStrain[m]) * shapeGrad[k] * dd_dvij[m][i][j][l]);

                            // This is programmed stupidly.  To ensure the symmetry it is enough to make
                            // the following assignment once and not for each quadrature point
                            localMat[k][i][l+3][j] = localMat[i][k][j][l+3];

                            // \partial W^2 \partial v^i_j \partial v^k_l
                            localMat[i][k][j+3][l+3] += weight
                                * (  A_[m] * (r_s * dd_dvij[m][k][l]) * (r_s * dd_dvij[m][i][j])
                                     + A_[m] * (strain[m] - referenceStrain[m]) * (r_s * dd_dvij_dvkl[m][i][j][k][l]) );

                            // ////////////////////////////////////////////
                            //   The rotational part
                            // ////////////////////////////////////////////
                            
                            // \partial W^2 \partial v^i_j \partial v^k_l
                            // All other derivatives are zero

                            double sum = duLocal_dvij[k][l][m] * (duCan_dvij[i][j] * hatq.director(m) + darbouxCan*dd_dvij[m][i][j]);
                            
                            sum += (strain[m+3] - referenceStrain[m+3]) * (duCan_dvij_dvkl[i][j][k][l] * hatq.director(m)
                                                                           + duCan_dvij[i][j] * dd_dvij[m][k][l]
                                                                           + duCan_dvij[k][l] * dd_dvij[m][i][j]
                                                                           + darbouxCan * dd_dvij_dvkl[m][i][j][k][l]);
                       
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
            = Dune::LagrangeShapeFunctions<double, double, gridDim>::general(it->geometry().type(), elementOrder);
        const int numOfBaseFct = baseSet.size();  
        
        Configuration localSolution[numOfBaseFct];
        
        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[indexSet.template subIndex<gridDim>(*it,i)];

        // Get quadrature rule
        int polOrd = 2;
        const QuadratureRule<double, gridDim>& quad = QuadratureRules<double, gridDim>::rule(it->geometry().type(), polOrd);

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
            Quaternion<double> hatq_s;
            hatq_s[0] = localSolution[0].q[0]*shapeGrad[0] + localSolution[1].q[0]*shapeGrad[1];
            hatq_s[1] = localSolution[0].q[1]*shapeGrad[0] + localSolution[1].q[1]*shapeGrad[1];
            hatq_s[2] = localSolution[0].q[2]*shapeGrad[0] + localSolution[1].q[2]*shapeGrad[1];
            hatq_s[3] = localSolution[0].q[3]*shapeGrad[0] + localSolution[1].q[3]*shapeGrad[1];

            // The current strain
            FieldVector<double,blocksize> strain = getStrain(sol, it, quadPos);

            // The reference strain
            FieldVector<double,blocksize> referenceStrain = getStrain(referenceConfiguration_, it, quadPos);

            // Contains \partial q / \partial v^i_j  at v = 0
            FixedArray<FixedArray<Quaternion<double>,3>,2> dq_dvij;

            Quaternion<double> dq_dvij_ds[2][3];
            for (int i=0; i<2; i++)
                for (int j=0; j<3; j++) {

                    for (int m=0; m<3; m++) {
                        dq_dvij[i][j][m]    = (j==m) * 0.5 * shapeFunction[i];
                        dq_dvij_ds[i][j][m] = (j==m) * 0.5 * shapeGrad[i];
                    }

                    dq_dvij[i][j][3]    = 0;
                    dq_dvij_ds[i][j][3] = 0;
                }

            // dd_dvij[k][i][j] = \parder {d_k} {v^i_j}
            FixedArray<FixedArray<FixedArray<FieldVector<double,3>, 3>, 2>, 3> dd_dvij;
            getFirstDerivativesOfDirectors(hatq, dd_dvij, dq_dvij);

            
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

                    grad[globalDof][j] += weight 
                        * ((  A_[0] * (strain[0] - referenceStrain[0]) * shapeGrad[i] * hatq.director(0)[j])
                           + (A_[1] * (strain[1] - referenceStrain[1]) * shapeGrad[i] * hatq.director(1)[j])
                           + (A_[2] * (strain[2] - referenceStrain[2]) * shapeGrad[i] * hatq.director(2)[j]));

                }

                // \partial \bar{W}_v / \partial v^i_j
                for (int j=0; j<3; j++) {

                    grad[globalDof][3+j] += weight 
                        * (  (A_[0] * (strain[0] - referenceStrain[0]) * (r_s*dd_dvij[0][i][j]))
                           + (A_[1] * (strain[1] - referenceStrain[1]) * (r_s*dd_dvij[1][i][j]))
                           + (A_[2] * (strain[2] - referenceStrain[2]) * (r_s*dd_dvij[2][i][j])));

                }

                // /////////////////////////////////////////////
                //   The rotational part
                // /////////////////////////////////////////////
                
                // \partial \bar{W}_v / \partial v^i_j
                for (int j=0; j<3; j++) {

                    FieldVector<double,3> du_dvij;
                    for (int m=0; m<3; m++) {
                        du_dvij[m] = B(m, dq_dvij[i][j].mult(hatq)) * hatq_s;
                        du_dvij[m] += B(m, hatq) * (dq_dvij_ds[i][j].mult(hatq) + dq_dvij[i][j].mult(hatq_s));
                    }
                    du_dvij *= 2;

                    FieldVector<double,3> darbouxCan = darbouxCanonical(hatq, hatq_s);
                    for (int m=0; m<3; m++) {

                        double addend1 = du_dvij * hatq.director(m);
                        double addend2 = darbouxCan * dd_dvij[m][i][j];

                        grad[globalDof][3+j] += weight * K_[m] 
                            * (strain[m+3]-referenceStrain[m+3]) * (addend1 + addend2);

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

        // Extract local solution on this element
        const LagrangeShapeFunctionSet<double, double, gridDim> & baseSet 
            = Dune::LagrangeShapeFunctions<double, double, gridDim>::general(it->geometry().type(), elementOrder);
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
            = QuadratureRules<double, gridDim>::rule(it->geometry().type(), shearingPolOrd);

        for (int pt=0; pt<shearingQuad.size(); pt++) {

            // Local position of the quadrature point
            const FieldVector<double,gridDim>& quadPos = shearingQuad[pt].position();
            
            const double integrationElement = it->geometry().integrationElement(quadPos);
        
            double weight = shearingQuad[pt].weight() * integrationElement;
            
            FieldVector<double,blocksize> strain = getStrain(sol, it, quadPos);

            // The reference strain
            FieldVector<double,blocksize> referenceStrain = getStrain(referenceConfiguration_, it, quadPos);

            for (int i=0; i<3; i++)
                energy += weight * 0.5 * A_[i] * (strain[i] - referenceStrain[i]) * (strain[i] * referenceStrain[i]);

        }

        // Get quadrature rule
        const int polOrd = 2;
        const QuadratureRule<double, gridDim>& bendingQuad = QuadratureRules<double, gridDim>::rule(it->geometry().type(), polOrd);

        for (int pt=0; pt<bendingQuad.size(); pt++) {

            // Local position of the quadrature point
            const FieldVector<double,gridDim>& quadPos = bendingQuad[pt].position();
            
            double weight = bendingQuad[pt].weight() * it->geometry().integrationElement(quadPos);
            
            FieldVector<double,blocksize> strain = getStrain(sol, it, quadPos);

            // The reference strain
            FieldVector<double,blocksize> referenceStrain = getStrain(referenceConfiguration_, it, quadPos);

            // Part II: the bending and twisting energy
            for (int i=0; i<3; i++)
                energy += weight * 0.5 * K_[i] * (strain[i+3] - referenceStrain[i+3]) * (strain[i+3] * referenceStrain[i+3]);

        }

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
            = Dune::LagrangeShapeFunctions<double, double, gridDim>::general(it->geometry().type(), elementOrder);
        int numOfBaseFct = baseSet.size();

        Configuration localSolution[numOfBaseFct];
        
        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[indexSet.template subIndex<gridDim>(*it,i)];

        // Get quadrature rule
        const int polOrd = 2;
        const QuadratureRule<double, gridDim>& quad = QuadratureRules<double, gridDim>::rule(it->geometry().type(), polOrd);

        for (int pt=0; pt<quad.size(); pt++) {

            // Local position of the quadrature point
            const FieldVector<double,gridDim>& quadPos = quad[pt].position();

            double weight = quad[pt].weight() * it->geometry().integrationElement(quadPos);

            FieldVector<double,blocksize> localStrain = getStrain(sol, it, quad[pt].position());
            
            // Sum it all up
            strain.axpy(weight, localStrain);

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
        = Dune::LagrangeShapeFunctions<double, double, gridDim>::general(element->geometry().type(), elementOrder);
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
    Quaternion<double> q_s;
    q_s[0] = localSolution[0].q[0]*shapeGrad[0][0] + localSolution[1].q[0]*shapeGrad[1][0];
    q_s[1] = localSolution[0].q[1]*shapeGrad[0][0] + localSolution[1].q[1]*shapeGrad[1][0];
    q_s[2] = localSolution[0].q[2]*shapeGrad[0][0] + localSolution[1].q[2]*shapeGrad[1][0];
    q_s[3] = localSolution[0].q[3]*shapeGrad[0][0] + localSolution[1].q[3]*shapeGrad[1][0];
        
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
getResultantForce(const std::vector<Configuration>& sol) const
{
    const typename GridType::Traits::LeafIndexSet& indexSet = grid_->leafIndexSet();

    if (sol.size()!=indexSet.size(gridDim))
        DUNE_THROW(Exception, "Solution vector doesn't match the grid!");

    // HARDWIRED: the leftmost point on the grid
    EntityPointer leftElement = grid_->template leafbegin<0>();
    assert(leftElement->ileafbegin().boundary());

    double pos = 0;

    FieldVector<double, blocksize> strain = getStrain(sol, leftElement, pos);
    FieldVector<double, blocksize> referenceStrain = getStrain(referenceConfiguration_, leftElement, pos);

    FieldVector<double,3> localStress;
    for (int i=0; i<3; i++)
        localStress[i] = (strain[i] - referenceStrain[i]) * A_[i];

    #warning Transformation fehlt
    return localStress;
}
