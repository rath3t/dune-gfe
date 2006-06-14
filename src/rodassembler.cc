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
               BCRSMatrix<MatrixBlock>& matrix)
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
        
        mat.resize(numOfBaseFct, numOfBaseFct);

        // Extract local solution
        std::vector<Configuration> localSolution(numOfBaseFct);
        
        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[indexSet.template subIndex<gridDim>(*it,i)];

        // setup matrix 
        getLocalMatrix( *it, localSolution, numOfBaseFct, mat);
        
        // Add element matrix to global stiffness matrix
        for(int i=0; i<numOfBaseFct; i++) { 
            
            //int row = functionSpace_.mapToGlobal( *it , i );
            int row = indexSet.template subIndex<gridDim>(*it,i);

            for (int j=0; j<numOfBaseFct; j++ ) {
                
                //int col = functionSpace_.mapToGlobal( *it , j );    
                int col = indexSet.template subIndex<gridDim>(*it,j);
                matrix[row][col] += mat[i][j];
                
            }
        }

    }

}






template <class GridType>
template <class MatrixType>
void Dune::RodAssembler<GridType>::
getLocalMatrix( EntityType &entity, 
                const std::vector<Configuration>& localSolution,
                const int matSize, MatrixType& localMat) const
{
    const typename GridType::Traits::LevelIndexSet& indexSet = grid_->levelIndexSet(grid_->maxLevel());

    /* ndof is the number of vectors of the element */
    int ndof = matSize;

    for (int i=0; i<matSize; i++)
        for (int j=0; j<matSize; j++)
            localMat[i][j] = 0;
    
    const LagrangeShapeFunctionSet<double, double, gridDim> & baseSet 
        = Dune::LagrangeShapeFunctions<double, double, gridDim>::general(entity.geometry().type(), elementOrder);

    // Get quadrature rule
    int polOrd = 2;
    const QuadratureRule<double, gridDim>& quad = QuadratureRules<double, gridDim>::rule(entity.geometry().type(), polOrd);
    
    /* Loop over all integration points */
    for (int ip=0; ip<quad.size(); ip++) {
        
        // Local position of the quadrature point
        const FieldVector<double,gridDim>& quadPos = quad[ip].position();

        // calc Jacobian inverse before integration element is evaluated 
        const FieldMatrix<double,gridDim,gridDim>& inv = entity.geometry().jacobianInverseTransposed(quadPos);
        const double integrationElement = entity.geometry().integrationElement(quadPos);
        
        /* Compute the weight of the current integration point */
        double weight = quad[ip].weight() * integrationElement;
        
        /**********************************************/
        /* compute gradients of the shape functions   */
        /**********************************************/
        FieldVector<double,gridDim> shapeGrad[ndof];
        
        for (int dof=0; dof<ndof; dof++) {
            
            for (int i=0; i<gridDim; i++)
                shapeGrad[dof][i] = baseSet[dof].evaluateDerivative(0,i,quadPos);
            
            // multiply with jacobian inverse 
            FieldVector<double,gridDim> tmp(0);
            inv.umv(shapeGrad[dof], tmp);
            shapeGrad[dof] = tmp;

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

        // Interpolate current rotation at this quadrature point and normalize
        // to get a unit quaternion again
        Quaternion<double> hatq;
        hatq[0] = localSolution[0].q[0]*shapeFunction[0] + localSolution[1].q[0]*shapeFunction[1];
        hatq[1] = localSolution[0].q[1]*shapeFunction[0] + localSolution[1].q[1]*shapeFunction[1];
        hatq[2] = localSolution[0].q[2]*shapeFunction[0] + localSolution[1].q[2]*shapeFunction[1];
        hatq[3] = localSolution[0].q[3]*shapeFunction[0] + localSolution[1].q[3]*shapeFunction[1];
        hatq.normalize();

        // Contains \partial q / \partial v^i_j  at v = 0
        Quaternion<double> dq_dvij[2][3];
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
        FieldVector<double,3> dd_dvij[3][2][3];
        
        for (int i=0; i<2; i++) {
            
            for (int j=0; j<3; j++) {
                
                // d1
                dd_dvij[0][i][j][0] = hatq[0]*(dq_dvij[i][j].mult(hatq))[0] - hatq[1]*(dq_dvij[i][j].mult(hatq))[1] 
                    - hatq[2]*(dq_dvij[i][j].mult(hatq))[2] + hatq[3]*(dq_dvij[i][j].mult(hatq))[3];
                
                dd_dvij[0][i][j][1] = (dq_dvij[i][j].mult(hatq))[0]*hatq[1] + hatq[0]*(dq_dvij[i][j].mult(hatq))[1]
                    + (dq_dvij[i][j].mult(hatq))[2]*hatq[3] + hatq[2]*(dq_dvij[i][j].mult(hatq))[3];
                
                dd_dvij[0][i][j][2] = (dq_dvij[i][j].mult(hatq))[0]*hatq[2] + hatq[0]*(dq_dvij[i][j].mult(hatq))[2]
                    - (dq_dvij[i][j].mult(hatq))[1]*hatq[3] - hatq[1]*(dq_dvij[i][j].mult(hatq))[3];
                
                // d2
                dd_dvij[1][i][j][0] = (dq_dvij[i][j].mult(hatq))[0]*hatq[1] + hatq[0]*(dq_dvij[i][j].mult(hatq))[1]
                    - (dq_dvij[i][j].mult(hatq))[2]*hatq[3] - hatq[2]*(dq_dvij[i][j].mult(hatq))[3];
                
                dd_dvij[1][i][j][1] = - hatq[0]*(dq_dvij[i][j].mult(hatq))[0] + hatq[1]*(dq_dvij[i][j].mult(hatq))[1] 
                    - hatq[2]*(dq_dvij[i][j].mult(hatq))[2] + hatq[3]*(dq_dvij[i][j].mult(hatq))[3];
                
                dd_dvij[1][i][j][2] = (dq_dvij[i][j].mult(hatq))[1]*hatq[2] + hatq[1]*(dq_dvij[i][j].mult(hatq))[2]
                    - (dq_dvij[i][j].mult(hatq))[0]*hatq[3] - hatq[0]*(dq_dvij[i][j].mult(hatq))[3];
                
                // d3
                dd_dvij[2][i][j][0] = (dq_dvij[i][j].mult(hatq))[0]*hatq[2] + hatq[0]*(dq_dvij[i][j].mult(hatq))[2]
                    + (dq_dvij[i][j].mult(hatq))[1]*hatq[3] + hatq[1]*(dq_dvij[i][j].mult(hatq))[3];
                
                dd_dvij[2][i][j][1] = (dq_dvij[i][j].mult(hatq))[0]*hatq[2] + hatq[0]*(dq_dvij[i][j].mult(hatq))[2]
                    - (dq_dvij[i][j].mult(hatq))[1]*hatq[3] - hatq[1]*(dq_dvij[i][j].mult(hatq))[3];
                
                dd_dvij[2][i][j][2] = - hatq[0]*(dq_dvij[i][j].mult(hatq))[0] - hatq[1]*(dq_dvij[i][j].mult(hatq))[1] 
                    + hatq[2]*(dq_dvij[i][j].mult(hatq))[2] + hatq[3]*(dq_dvij[i][j].mult(hatq))[3];
                
                
                dd_dvij[0][i][j] *= 2;
                dd_dvij[1][i][j] *= 2;
                dd_dvij[2][i][j] *= 2;
                
            }
            
        }


        // Contains \parder dm \parder v^i_j
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
                        dd_dvij_dvkl[1][i][j][k][l][2] =  A[2][1] + A[1][2] - A[3][0] - A[0][3];
                        
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
        
        FieldVector<double,3> u;  // The Darboux vector
        u[0] = 2 * ( hatq[3]*hatq_s[0] + hatq[2]*hatq_s[1] - hatq[1]*hatq_s[2] - hatq[0]*hatq_s[3]);
        u[1] = 2 * (-hatq[2]*hatq_s[0] + hatq[3]*hatq_s[1] + hatq[0]*hatq_s[2] - hatq[1]*hatq_s[3]);
        u[2] = 2 * ( hatq[1]*hatq_s[0] - hatq[0]*hatq_s[1] + hatq[3]*hatq_s[2] - hatq[2]*hatq_s[3]);

        // Contains \partial q / \partial v^i_j  at v = 0
        double dum_dvij[3][2][3];

        for (int i=0; i<2; i++)
            for (int j=0; j<3; j++) {
                
                for (int m=0; m<3; m++) 
                    dum_dvij[m][i][j] = B(m, dq_dvij[i][j].mult(hatq))*hatq_s + B(m,hatq)*(dq_dvij_ds[i][j].mult(hatq));
                
            }

        // ///////////////////////////////////
        //   Sum it all up
        // ///////////////////////////////////
        for (int i=0; i<matSize; i++) {

            for (int k=0; k<matSize; k++) {

                for (int j=0; j<3; j++) {

                    for (int l=0; l<3; l++) {

                        // ////////////////////////////////////////////
                        //   The translational part
                        // ////////////////////////////////////////////
                        
                        // \partial W^2 \partial r^i_j \partial r^k_l
                        localMat[i][k][j][l] += weight 
                            * ( A1 * shapeGrad[i] * hatq.director(0)[j] * shapeGrad[k] * hatq.director(0)[l]
                                + A2 * shapeGrad[i] * hatq.director(1)[j] * shapeGrad[k] * hatq.director(1)[l]
                                + A3 * shapeGrad[i] * hatq.director(2)[j] * shapeGrad[k] * hatq.director(2)[l]);

                        // \partial W^2 \partial v^i_j \partial r^k_l
                        localMat[i][k][j][l+3] += weight
                            * (A1 * shapeGrad[k]*hatq.director(0)[l]*(r_s*dd_dvij[0][i][j])
                               + A1 * (r_s*hatq.director(0)) * shapeGrad[k] * dd_dvij[0][i][j][l]
                               + A2 * shapeGrad[k]*hatq.director(1)[l]*(r_s*dd_dvij[1][i][j])
                               + A2 * (r_s*hatq.director(1)) * shapeGrad[k] * dd_dvij[1][i][j][l]
                               + A3 * shapeGrad[k]*hatq.director(2)[l]*(r_s*dd_dvij[2][i][j])
                               + A3 * (r_s*hatq.director(2)-1) * shapeGrad[k] * dd_dvij[2][i][j][l]);

                        localMat[i][k][j+3][l] = localMat[i][k][j][l+3];

                        // \partial W^2 \partial v^i_j \partial v^k_l
                        localMat[i][k][j+3][l+3] += weight
                            * (A1 * (r_s * dd_dvij[0][k][l]) * (r_s * dd_dvij[0][i][j])
                               + A1 * (r_s * hatq.director(0)) * (r_s * dd_dvij_dvkl[0][i][j][k][l])
                               + A2 * (r_s * dd_dvij[1][k][l]) * (r_s * dd_dvij[1][i][j])
                               + A2 * (r_s * hatq.director(1)) * (r_s * dd_dvij_dvkl[1][i][j][k][l])
                               + A3 * (r_s * dd_dvij[2][k][l]) * (r_s * dd_dvij[2][i][j])
                               + A3 * (r_s * hatq.director(2)) * (r_s * dd_dvij_dvkl[2][i][j][k][l]));

                        // ////////////////////////////////////////////
                        //   The rotational part
                        // ////////////////////////////////////////////
                        // Stupid: I want those as an array
                        double K[3] = {K1, K2, K3};

                        // \partial W^2 \partial v^i_j \partial v^k_l
                        // All other derivatives are zero
                        for (int m=0; m<3; m++) {

                            double sum = dum_dvij[m][k][l] * (B(m,dq_dvij[i][j].mult(hatq)) * hatq_s);
                            
                            sum += dum_dvij[m][k][l] * (B(m,hatq)*(dq_dvij_ds[i][j].mult(hatq) + dq_dvij[i][j].mult(hatq_s)));

                            sum += u[m] * (B(m, dq_dvij_dvkl[i][j][k][l].mult(hatq)) * hatq_s);

                            sum += u[m] * (B(m, dq_dvij[i][j].mult(hatq)) * dq_dvij_ds[k][l].mult(hatq));

                            sum += u[m] * (B(m, dq_dvij[k][l].mult(hatq)) * 
                                           (dq_dvij_ds[i][j].mult(hatq) + dq_dvij[i][j].mult(hatq_s)));

                            sum += u[m] * (B(m, hatq) * 
                                           (dq_dvij_dvkl_ds[i][j][k][l].mult(hatq) + (dq_dvij_dvkl[i][j][k][l].mult(hatq_s))));
                            
                            localMat[i][k][j+3][l+3] += 2*weight *K[m] * sum;

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

            // Interpolate current rotation at this quadrature point and normalize
            // to get a unit quaternion again
            Quaternion<double> hatq;
            hatq[0] = localSolution[0].q[0]*shapeFunction[0] + localSolution[1].q[0]*shapeFunction[1];
            hatq[1] = localSolution[0].q[1]*shapeFunction[0] + localSolution[1].q[1]*shapeFunction[1];
            hatq[2] = localSolution[0].q[2]*shapeFunction[0] + localSolution[1].q[2]*shapeFunction[1];
            hatq[3] = localSolution[0].q[3]*shapeFunction[0] + localSolution[1].q[3]*shapeFunction[1];
            hatq.normalize();

            // Get the derivative of the rotation at the quadrature point by interpolating in $H$
            Quaternion<double> hatq_s;
            hatq_s[0] = localSolution[0].q[0]*shapeGrad[0] + localSolution[1].q[0]*shapeGrad[1];
            hatq_s[1] = localSolution[0].q[1]*shapeGrad[0] + localSolution[1].q[1]*shapeGrad[1];
            hatq_s[2] = localSolution[0].q[2]*shapeGrad[0] + localSolution[1].q[2]*shapeGrad[1];
            hatq_s[3] = localSolution[0].q[3]*shapeGrad[0] + localSolution[1].q[3]*shapeGrad[1];

            FieldVector<double,3> u;  // The Darboux vector
            u[0] = 2 * ( hatq[3]*hatq_s[0] + hatq[2]*hatq_s[1] - hatq[1]*hatq_s[2] - hatq[0]*hatq_s[3]);
            u[1] = 2 * (-hatq[2]*hatq_s[0] + hatq[3]*hatq_s[1] + hatq[0]*hatq_s[2] - hatq[1]*hatq_s[3]);
            u[2] = 2 * ( hatq[1]*hatq_s[0] - hatq[0]*hatq_s[1] + hatq[3]*hatq_s[2] - hatq[2]*hatq_s[3]);

            // Contains \partial q / \partial v^i_j  at v = 0
            Quaternion<double> dq_dvij[2][3];
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

            // Contains \parder
            FieldVector<double,3> dd_dvij[3][2][3];

            for (int i=0; i<2; i++) {

                for (int j=0; j<3; j++) {

                    // d1
                    dd_dvij[0][i][j][0] = hatq[0]*(dq_dvij[i][j].mult(hatq))[0] - hatq[1]*(dq_dvij[i][j].mult(hatq))[1] 
                        - hatq[2]*(dq_dvij[i][j].mult(hatq))[2] + hatq[3]*(dq_dvij[i][j].mult(hatq))[3];
                    
                    dd_dvij[0][i][j][1] = (dq_dvij[i][j].mult(hatq))[0]*hatq[1] + hatq[0]*(dq_dvij[i][j].mult(hatq))[1]
                        + (dq_dvij[i][j].mult(hatq))[2]*hatq[3] + hatq[2]*(dq_dvij[i][j].mult(hatq))[3];

                    dd_dvij[0][i][j][2] = (dq_dvij[i][j].mult(hatq))[0]*hatq[2] + hatq[0]*(dq_dvij[i][j].mult(hatq))[2]
                        - (dq_dvij[i][j].mult(hatq))[1]*hatq[3] - hatq[1]*(dq_dvij[i][j].mult(hatq))[3];

                    // d2
                    dd_dvij[1][i][j][0] = (dq_dvij[i][j].mult(hatq))[0]*hatq[1] + hatq[0]*(dq_dvij[i][j].mult(hatq))[1]
                        - (dq_dvij[i][j].mult(hatq))[2]*hatq[3] - hatq[2]*(dq_dvij[i][j].mult(hatq))[3];

                    dd_dvij[1][i][j][1] = - hatq[0]*(dq_dvij[i][j].mult(hatq))[0] + hatq[1]*(dq_dvij[i][j].mult(hatq))[1] 
                        - hatq[2]*(dq_dvij[i][j].mult(hatq))[2] + hatq[3]*(dq_dvij[i][j].mult(hatq))[3];
                    
                    dd_dvij[1][i][j][2] = (dq_dvij[i][j].mult(hatq))[1]*hatq[2] + hatq[1]*(dq_dvij[i][j].mult(hatq))[2]
                        - (dq_dvij[i][j].mult(hatq))[0]*hatq[3] - hatq[0]*(dq_dvij[i][j].mult(hatq))[3];

                    // d3
                    dd_dvij[2][i][j][0] = (dq_dvij[i][j].mult(hatq))[0]*hatq[2] + hatq[0]*(dq_dvij[i][j].mult(hatq))[2]
                        + (dq_dvij[i][j].mult(hatq))[1]*hatq[3] + hatq[1]*(dq_dvij[i][j].mult(hatq))[3];

                    dd_dvij[2][i][j][1] = (dq_dvij[i][j].mult(hatq))[0]*hatq[2] + hatq[0]*(dq_dvij[i][j].mult(hatq))[2]
                        - (dq_dvij[i][j].mult(hatq))[1]*hatq[3] - hatq[1]*(dq_dvij[i][j].mult(hatq))[3];

                    dd_dvij[2][i][j][2] = - hatq[0]*(dq_dvij[i][j].mult(hatq))[0] - hatq[1]*(dq_dvij[i][j].mult(hatq))[1] 
                        + hatq[2]*(dq_dvij[i][j].mult(hatq))[2] + hatq[3]*(dq_dvij[i][j].mult(hatq))[3];
                    

                    dd_dvij[0][i][j] *= 2;
                    dd_dvij[1][i][j] *= 2;
                    dd_dvij[2][i][j] *= 2;

                }

            }

            // /////////////////////////////////////////////
            //   Sum it all up
            // /////////////////////////////////////////////

            for (int dof=0; dof<numOfBaseFct; dof++) {

                int globalDof = indexSet.template subIndex<gridDim>(*it,dof);

                // /////////////////////////////////////////////
                //   The translational part
                // /////////////////////////////////////////////
                
                // \partial \bar{W} / \partial r^i_j
                for (int j=0; j<3; j++) {

                    grad[globalDof][j] += weight 
                        * ((A1 * (r_s*hatq.director(0)) * shapeGrad[dof] * hatq.director(0)[j])
                           + (A2 * (r_s*hatq.director(1)) * shapeGrad[dof] * hatq.director(1)[j])
                           + (A3 * (r_s*hatq.director(2) - 1) * shapeGrad[dof] * hatq.director(2)[j]));

                }

                // \partial \bar{W}_v / \partial v^i_j
                for (int j=0; j<3; j++) {

                    grad[globalDof][3+j] += weight 
                        * ((A1 * (r_s*hatq.director(0)) * (r_s*dd_dvij[0][dof][j]))
                           + (A2 * (r_s*hatq.director(1)) * (r_s*dd_dvij[1][dof][j]))
                           + (A3 * (r_s*hatq.director(2) - 1) * (r_s*dd_dvij[2][dof][j])));

                }

                // /////////////////////////////////////////////
                //   The rotational part
                // /////////////////////////////////////////////
                
                // Stupid: I want those as an array
                double K[3] = {K1, K2, K3};

                // \partial \bar{W}_v / \partial v^i_j
                for (int j=0; j<3; j++) {

                    for (int m=0; m<3; m++) {

                        double addend1 = B(m,(dq_dvij[dof][j].mult(hatq))) * hatq_s;
                        double addend2 = B(m,hatq) * (dq_dvij_ds[dof][j].mult(hatq) + dq_dvij[dof][j].mult(hatq_s));

                        grad[globalDof][3+j] += 2*weight*K[m]*u[m] * (addend1 + addend2);

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

        // Get quadrature rule
        const int polOrd = 2;
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
            std::vector<FieldVector<double,gridDim> > shapeGrad(numOfBaseFct);
            
            for (int dof=0; dof<numOfBaseFct; dof++) {
                
                for (int i=0; i<gridDim; i++)
                    shapeGrad[dof][i] = baseSet[dof].evaluateDerivative(0,i,quadPos);
                //std::cout << "Gradient " << dof << ": " << shape_grads[dof] << std::endl;
                
                // multiply with jacobian inverse 
                FieldVector<double,gridDim> tmp(0);
                inv.umv(shapeGrad[dof], tmp);
                shapeGrad[dof] = tmp;
                //std::cout << "Gradient " << dof << ": " << shape_grads[dof] << std::endl;

            }

            // Get the value of the shape functions
            double shapeFunction[2];
            for(int i=0; i<2; i++) 
                shapeFunction[i] = baseSet[i].evaluateFunction(0,quadPos);

            // //////////////////////////////////
            //   Interpolate
            // //////////////////////////////////

            FieldVector<double,3> r_s;
            r_s[0] = localSolution[0].r[0]*shapeGrad[0][0] + localSolution[1].r[0]*shapeGrad[1][0];
            r_s[1] = localSolution[0].r[1]*shapeGrad[0][0] + localSolution[1].r[1]*shapeGrad[1][0];
            r_s[2] = localSolution[0].r[2]*shapeGrad[0][0] + localSolution[1].r[2]*shapeGrad[1][0];

            // Get the rotation at the quadrature point by interpolating in $H$ and normalizing
            Quaternion<double> q;
            q[0] = localSolution[0].q[0]*shapeFunction[0] + localSolution[1].q[0]*shapeFunction[1];
            q[1] = localSolution[0].q[1]*shapeFunction[0] + localSolution[1].q[1]*shapeFunction[1];
            q[2] = localSolution[0].q[2]*shapeFunction[0] + localSolution[1].q[2]*shapeFunction[1];
            q[3] = localSolution[0].q[3]*shapeFunction[0] + localSolution[1].q[3]*shapeFunction[1];

            // The interpolated quaternion is not a unit quaternion anymore.  We simply normalize
            q.normalize();
            
            // Get the derivative of the rotation at the quadrature point by interpolating in $H$
            Quaternion<double> q_s;
            q_s[0] = localSolution[0].q[0]*shapeGrad[0][0] + localSolution[1].q[0]*shapeGrad[1][0];
            q_s[1] = localSolution[0].q[1]*shapeGrad[0][0] + localSolution[1].q[1]*shapeGrad[1][0];
            q_s[2] = localSolution[0].q[2]*shapeGrad[0][0] + localSolution[1].q[2]*shapeGrad[1][0];
            q_s[3] = localSolution[0].q[3]*shapeGrad[0][0] + localSolution[1].q[3]*shapeGrad[1][0];

            // /////////////////////////////////////////////
            //   Sum it all up
            // /////////////////////////////////////////////

            // Part I: the shearing and stretching energy
            //std::cout << "tangent : " << r_s << std::endl;
            FieldVector<double,3> v;
            v[0] = r_s * q.director(0);    // shear strain
            v[1] = r_s * q.director(1);    // shear strain
            v[2] = r_s * q.director(2);    // stretching strain

            //std::cout << "strain : " << v << std::endl;

            energy += weight * 0.5*A1*v[0]*v[0] + 0.5*A2*v[1]*v[1] + 0.5*A3*(v[2]-1)*(v[2]-1);

            // Part II: the bending and twisting energy
            
            FieldVector<double,3> u;  // The Darboux vector
            u[0] = 2 * ( q[3]*q_s[0] + q[2]*q_s[1] - q[1]*q_s[2] - q[0]*q_s[3]);
            u[1] = 2 * (-q[2]*q_s[0] + q[3]*q_s[1] + q[0]*q_s[2] - q[1]*q_s[3]);
            u[2] = 2 * ( q[1]*q_s[0] - q[0]*q_s[1] + q[3]*q_s[2] - q[2]*q_s[3]);

            //std::cout << "Darboux vector : " << u << std::endl;

            energy += weight * 0.5 * (K1*u[0]*u[0] + K2*u[1]*u[1] + K3*u[2]*u[2]);

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
            
            const FieldMatrix<double,1,1>& inv = it->geometry().jacobianInverseTransposed(quadPos);
            const double integrationElement = it->geometry().integrationElement(quadPos);
        
            double weight = quad[pt].weight() * integrationElement;
            
            // ///////////////////////////////////////
            //   Compute deformation gradient
            // ///////////////////////////////////////
            std::vector<FieldVector<double,gridDim> > shapeGrad(numOfBaseFct);
            
            for (int dof=0; dof<numOfBaseFct; dof++) {
                
                for (int i=0; i<gridDim; i++)
                    shapeGrad[dof][i] = baseSet[dof].evaluateDerivative(0,i,quadPos);
                //std::cout << "Gradient " << dof << ": " << shape_grads[dof] << std::endl;
                
                // multiply with jacobian inverse 
                FieldVector<double,gridDim> tmp(0);
                inv.umv(shapeGrad[dof], tmp);
                shapeGrad[dof] = tmp;
                //std::cout << "Gradient " << dof << ": " << shape_grads[dof] << std::endl;

            }

            // Get the value of the shape functions
            double shapeFunction[2];
            for(int i=0; i<2; i++) 
                shapeFunction[i] = baseSet[i].evaluateFunction(0,quadPos);

            // //////////////////////////////////
            //   Interpolate
            // //////////////////////////////////

            FieldVector<double,3> r_s;
            r_s[0] = localSolution[0].r[0]*shapeGrad[0][0] + localSolution[1].r[0]*shapeGrad[1][0];
            r_s[1] = localSolution[0].r[1]*shapeGrad[0][0] + localSolution[1].r[1]*shapeGrad[1][0];
            r_s[2] = localSolution[0].r[2]*shapeGrad[0][0] + localSolution[1].r[2]*shapeGrad[1][0];

            // Get the rotation at the quadrature point by interpolating in $H$ and normalizing
            Quaternion<double> q;
            q[0] = localSolution[0].q[0]*shapeFunction[0] + localSolution[1].q[0]*shapeFunction[1];
            q[1] = localSolution[0].q[1]*shapeFunction[0] + localSolution[1].q[1]*shapeFunction[1];
            q[2] = localSolution[0].q[2]*shapeFunction[0] + localSolution[1].q[2]*shapeFunction[1];
            q[3] = localSolution[0].q[3]*shapeFunction[0] + localSolution[1].q[3]*shapeFunction[1];

            // The interpolated quaternion is not a unit quaternion anymore.  We simply normalize
            q.normalize();
            
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
            FieldVector<double,3> v;
            v[0] = r_s * q.director(0);    // shear strain
            v[1] = r_s * q.director(1);    // shear strain
            v[2] = r_s * q.director(2);    // stretching strain

            //std::cout << "strain : " << v << std::endl;

            // Part II: the Darboux vector
            
            FieldVector<double,3> u; 
            u[0] = 2 * ( q[3]*q_s[0] + q[2]*q_s[1] - q[1]*q_s[2] - q[0]*q_s[3]);
            u[1] = 2 * (-q[2]*q_s[0] + q[3]*q_s[1] + q[0]*q_s[2] - q[1]*q_s[3]);
            u[2] = 2 * ( q[1]*q_s[0] - q[0]*q_s[1] + q[3]*q_s[2] - q[2]*q_s[3]);

            // Sum it all up
            int elementIdx = indexSet.index(*it);
            strain[elementIdx][0] += weight * v[0];
            strain[elementIdx][1] += weight * v[1];
            strain[elementIdx][2] += weight * v[2];
            strain[elementIdx][3] += weight * u[0];
            strain[elementIdx][4] += weight * u[1];
            strain[elementIdx][5] += weight * u[2];

        }

    }

}
