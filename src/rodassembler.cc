#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/common/matrix.hh>

#include <dune/quadrature/quadraturerules.hh>



template <class FunctionSpaceType, int polOrd>
void Dune::RodAssembler<FunctionSpaceType, polOrd>::
getNeighborsPerVertex(MatrixIndexSet& nb) const
{
    const int gridDim = GridType::dimension;
    
    int i, j;
    int n = grid_->size(grid_->maxlevel(), gridDim);
    
    nb.resize(n, n);
    
    ElementIterator it    = grid_->template lbegin<0>( grid_->maxlevel() );
    ElementIterator endit = grid_->template lend<0>  ( grid_->maxlevel() );
    
    for (; it!=endit; ++it) {
        
        for (i=0; i<it->template count<gridDim>(); i++) {
            
            for (j=0; j<it->template count<gridDim>(); j++) {
                
                int iIdx = it->template subIndex<gridDim>(i);
                int jIdx = it->template subIndex<gridDim>(j);
                
                nb.add(iIdx, jIdx);
                
            }
            
        }
        
    }
    
}

template <class FunctionSpaceType, int polOrd>
void Dune::RodAssembler<FunctionSpaceType, polOrd>::
assembleMatrix(const BlockVector<FieldVector<double, blocksize> >& sol,
               BCRSMatrix<MatrixBlock>& matrix)
{

    MatrixIndexSet neighborsPerVertex;
    getNeighborsPerVertex(neighborsPerVertex);
    
    matrix = 0;
    
    ElementIterator it    = grid_->template lbegin<0>( grid_->maxlevel() );
    ElementIterator endit = grid_->template lend<0> ( grid_->maxlevel() );
    
    Matrix<MatrixBlock> mat;
    
    for( ; it != endit; ++it ) {
        
        const BaseFunctionSetType & baseSet = functionSpace_.getBaseFunctionSet( *it );
        const int numOfBaseFct = baseSet.getNumberOfBaseFunctions();  
        
        mat.resize(numOfBaseFct, numOfBaseFct);

        // Extract local solution
        BlockVector<FieldVector<double, blocksize> > localSolution(numOfBaseFct);
        
        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[functionSpace_.mapToGlobal(*it,i)];

        // setup matrix 
        getLocalMatrix( *it, localSolution, numOfBaseFct, mat);
        
        // Add element matrix to global stiffness matrix
        for(int i=0; i<numOfBaseFct; i++) { 
            
            int row = functionSpace_.mapToGlobal( *it , i );

            for (int j=0; j<numOfBaseFct; j++ ) {
                
                int col = functionSpace_.mapToGlobal( *it , j );    
                matrix[row][col] += mat[i][j];
                
            }
        }

    }
    
}






template <class FunctionSpaceType, int polOrd>
template <class MatrixType>
void Dune::RodAssembler<FunctionSpaceType, polOrd>::
getLocalMatrix( EntityType &entity, 
                const BlockVector<FieldVector<double, blocksize> >& localSolution,
                const int matSize, MatrixType& localMat) const
{
    
    /* ndof is the number of vectors of the element */
    int ndof = matSize;

    for (int i=0; i<matSize; i++)
        for (int j=0; j<matSize; j++)
            localMat[i][j] = 0;
    
    typedef typename FunctionSpaceType::BaseFunctionSetType BaseFunctionSetType;
    const BaseFunctionSetType & baseSet = this->functionSpace_.getBaseFunctionSet( entity );
    
    // Get quadrature rule
    const QuadratureRule<double, gridDim>& quad = QuadratureRules<double, gridDim>::rule(entity.geometry().type(), polOrd);
    
    /* Loop over all integration points */
    for (int ip=0; ip<quad.size(); ip++) {
        
        // Local position of the quadrature point
        const FieldVector<double,gridDim>& quadPos = quad[ip].position();

        // calc Jacobian inverse before integration element is evaluated 
        const FieldMatrix<double,gridDim,gridDim>& inv = entity.geometry().jacobianInverse(quadPos);
        const double integrationElement = entity.geometry().integrationElement(quadPos);
        
        /* Compute the weight of the current integration point */
        double weight = quad[ip].weight() * integrationElement;
        
        /**********************************************/
        /* compute gradients of the shape functions   */
        /**********************************************/
        Array<JacobianRange> shapeGrad(ndof);
        
        for (int dof=0; dof<ndof; dof++) {
            
            baseSet.jacobian(dof, quadPos, shapeGrad[dof]);
            
            // multiply with jacobian inverse 
            JacobianRange tmp(0);
            inv.umv(shapeGrad[dof][0], tmp[0]);
            shapeGrad[dof][0] = tmp[0];

        }
        
        
        RangeType shapeFunction[matSize];
        for(int i=0; i<matSize; i++) 
            baseSet.eval(i,quadPos,shapeFunction[i]);
        
        // //////////////////////////////////
        //   Interpolate
        // //////////////////////////////////
        
        double x_s     = localSolution[0][0]*shapeGrad[0][0][0] + localSolution[1][0]*shapeGrad[1][0][0];
        double y_s     = localSolution[0][1]*shapeGrad[0][0][0] + localSolution[1][1]*shapeGrad[1][0][0];
        //double theta_s = localSolution[0][2]*shapeGrad[0][0][0] + localSolution[1][2]*shapeGrad[1][0][0];

        double theta   = localSolution[0][2]*shapeFunction[0] + localSolution[1][2]*shapeFunction[1];

        for (int i=0; i<matSize; i++) {

            for (int j=0; j<matSize; j++) {

                // \partial J^2 / \partial x_i \partial x_j
                localMat[i][j][0][0] += weight * shapeGrad[i][0][0] * shapeGrad[j][0][0]
                    * (A1 * cos(theta) * cos(theta) + A3 * sin(theta) * sin(theta));

                // \partial J^2 / \partial x_i \partial y_j
                localMat[i][j][0][1] += weight * shapeGrad[i][0][0] * shapeGrad[j][0][0]
                    * (-A1 + A3) * sin(theta)* cos(theta);

                // \partial J^2 / \partial x_i \partial theta_j
                localMat[i][j][0][2] += weight * shapeGrad[i] * shapeFunction[j]
                    * (-A1 * (x_s*sin(theta) + y_s*cos(theta)) * cos(theta)
                       - A1* (x_s*cos(theta) - y_s*sin(theta)) * sin(theta)
                       +A3 * (x_s*cos(theta) - y_s*sin(theta)) * sin(theta)
                       +A3 * (x_s*sin(theta) + y_s*cos(theta) - 1) * cos(theta));

                // \partial J^2 / \partial y_i \partial x_j
                localMat[i][j][1][0] += weight * shapeGrad[i][0][0] * shapeGrad[j][0][0]
                    * (-A1 * sin(theta)* cos(theta) + A3 * cos(theta) * sin(theta));

                // \partial J^2 / \partial y_i \partial y_j
                localMat[i][j][1][1] += weight * shapeGrad[i] * shapeGrad[j]
                    * (A1 * sin(theta)*sin(theta) + A3 * cos(theta)*cos(theta));

                // \partial J^2 / \partial y_i \partial theta_j
                localMat[i][j][1][2] += weight * shapeGrad[i] * shapeFunction[j]
                    * (A1  * (x_s * sin(theta) + y_s * cos(theta)) * sin(theta)
                       -A1 * (x_s * cos(theta) - y_s * sin(theta)) * cos(theta)
                       +A3 * (x_s * cos(theta) - y_s * sin(theta)) * cos(theta)
                       -A3 * (x_s * sin(theta) + y_s * cos(theta) - 1) * sin(theta));

                // \partial J^2 / \partial theta_i \partial x_j
                localMat[i][j][2][0] += weight * shapeFunction[i] * shapeGrad[j]
                    * (-A1 * (x_s*sin(theta) + y_s*cos(theta)) * cos(theta)
                       - A1* (x_s*cos(theta) - y_s*sin(theta)) * sin(theta)
                       +A3 * (x_s*cos(theta) - y_s*sin(theta)) * sin(theta)
                       +A3 * (x_s*sin(theta) + y_s*cos(theta) - 1) * cos(theta));

                // \partial J^2 / \partial theta_i \partial y_j
                localMat[i][j][2][1] += weight * shapeFunction[i] * shapeGrad[j]
                    * (A1  * (x_s * sin(theta) + y_s * cos(theta)) * sin(theta)
                       -A1 * (x_s * cos(theta) - y_s * sin(theta)) * cos(theta)
                       +A3 * (x_s * cos(theta) - y_s * sin(theta)) * cos(theta)
                       -A3 * (x_s * sin(theta) + y_s * cos(theta) - 1) * sin(theta));

                // \partial J^2 / \partial theta_i \partial theta_j
                localMat[i][j][2][2] += weight * B * shapeGrad[i] * shapeGrad[j];
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

template <class FunctionSpaceType, int polOrd>
void Dune::RodAssembler<FunctionSpaceType, polOrd>::
assembleGradient(const BlockVector<FieldVector<double, blocksize> >& sol,
                 BlockVector<FieldVector<double, blocksize> >& grad) const
{
    const int maxlevel = grid_->maxlevel();

    if (sol.size()!=grid_->size(maxlevel, gridDim))
        DUNE_THROW(Exception, "Solution vector doesn't match the grid!");

    grad.resize(sol.size());
    grad = 0;

    ElementIterator it    = grid_->template lbegin<0>(maxlevel);
    ElementIterator endIt = grid_->template lend<0>(maxlevel);

    // Loop over all elements
    for (; it!=endIt; ++it) {

        // Extract local solution on this element
        const BaseFunctionSetType & baseSet = functionSpace_.getBaseFunctionSet( *it );
        const int numOfBaseFct = baseSet.getNumberOfBaseFunctions();  
        
        FieldVector<double, blocksize> localSolution[numOfBaseFct];
        
        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[functionSpace_.mapToGlobal(*it,i)];

        // Get quadrature rule
        const QuadratureRule<double, gridDim>& quad = QuadratureRules<double, gridDim>::rule(it->geometry().type(), polOrd);

        for (int pt=0; pt<quad.size(); pt++) {

            // Local position of the quadrature point
            const FieldVector<double,gridDim>& quadPos = quad[pt].position();
            
            const FieldMatrix<double,1,1>& inv = it->geometry().jacobianInverse(quadPos);
            const double integrationElement = it->geometry().integrationElement(quadPos);
        
            double weight = quad[pt].weight() * integrationElement;
            
            // ///////////////////////////////////////
            //   Compute deformation gradient
            // ///////////////////////////////////////
            Array<JacobianRange> shapeGrad(numOfBaseFct);
            
            for (int dof=0; dof<numOfBaseFct; dof++) {
                
                baseSet.jacobian(dof, quadPos, shapeGrad[dof]);
                
                // multiply with jacobian inverse 
                JacobianRange tmp(0);
                inv.umv(shapeGrad[dof][0], tmp[0]);
                shapeGrad[dof][0] = tmp[0];
                
            }

            // Get the value of the shape functions
            RangeType shapeFunction[2];
            for(int i=0; i<2; i++) 
                baseSet.eval(i,quadPos,shapeFunction[i]);

            // //////////////////////////////////
            //   Interpolate
            // //////////////////////////////////

            double x_s     = localSolution[0][0]*shapeGrad[0][0][0] + localSolution[1][0]*shapeGrad[1][0][0];
            double y_s     = localSolution[0][1]*shapeGrad[0][0][0] + localSolution[1][1]*shapeGrad[1][0][0];
            double theta_s = localSolution[0][2]*shapeGrad[0][0][0] + localSolution[1][2]*shapeGrad[1][0][0];

            double theta   = localSolution[0][2]*shapeFunction[0] + localSolution[1][2]*shapeFunction[1];

            // /////////////////////////////////////////////
            //   Sum it all up
            // /////////////////////////////////////////////

            double partA1 = A1 * (x_s * cos(theta) - y_s * sin(theta));
            double partA3 = A3 * (x_s * sin(theta) + y_s * cos(theta) - 1);

            for (int dof=0; dof<numOfBaseFct; dof++) {

                int globalDof = functionSpace_.mapToGlobal(*it,dof);

                //printf("globalDof: %d   partA1: %g   partA3: %g\n", globalDof, partA1, partA3);

                // \partial J / \partial x^i
                grad[globalDof][0] += weight * (partA1 * cos(theta) + partA3 * sin(theta)) * shapeGrad[dof][0][0];

                // \partial J / \partial y^i
                grad[globalDof][1] += weight * (-partA1 * sin(theta) + partA3 * cos(theta)) * shapeGrad[dof][0][0];

                // \partial J / \partial \theta^i
                grad[globalDof][2] += weight * (B * theta_s * shapeGrad[dof][0][0]
                                               + partA1 * (-x_s * sin(theta) - y_s * cos(theta)) * shapeFunction[dof]
                                               + partA3 * ( x_s * cos(theta) - y_s * sin(theta)) * shapeFunction[dof]);

            }


        }

    }

}


template <class FunctionSpaceType, int polOrd>
double Dune::RodAssembler<FunctionSpaceType, polOrd>::
computeEnergy(const BlockVector<FieldVector<double, blocksize> >& sol) const
{
    const int maxlevel = grid_->maxlevel();
    double energy = 0;

    if (sol.size()!=grid_->size(maxlevel, gridDim))
        DUNE_THROW(Exception, "Solution vector doesn't match the grid!");

    ElementIterator it    = grid_->template lbegin<0>(maxlevel);
    ElementIterator endIt = grid_->template lend<0>(maxlevel);

    // Loop over all elements
    for (; it!=endIt; ++it) {

        // Extract local solution on this element
        const BaseFunctionSetType & baseSet = functionSpace_.getBaseFunctionSet( *it );
        const int numOfBaseFct = baseSet.getNumberOfBaseFunctions();  
        
        FieldVector<double, blocksize> localSolution[numOfBaseFct];
        
        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[functionSpace_.mapToGlobal(*it,i)];

        // Get quadrature rule
        const QuadratureRule<double, gridDim>& quad = QuadratureRules<double, gridDim>::rule(it->geometry().type(), polOrd);

        for (int pt=0; pt<quad.size(); pt++) {

            // Local position of the quadrature point
            const FieldVector<double,gridDim>& quadPos = quad[pt].position();
            
            const FieldMatrix<double,1,1>& inv = it->geometry().jacobianInverse(quadPos);
            const double integrationElement = it->geometry().integrationElement(quadPos);
        
            double weight = quad[pt].weight() * integrationElement;
            
            // ///////////////////////////////////////
            //   Compute deformation gradient
            // ///////////////////////////////////////
            Array<JacobianRange> shapeGrad(numOfBaseFct);
            
            for (int dof=0; dof<numOfBaseFct; dof++) {
                
                baseSet.jacobian(dof, quadPos, shapeGrad[dof]);
                //std::cout << "Gradient " << dof << ": " << shape_grads[dof] << std::endl;
                
                // multiply with jacobian inverse 
                JacobianRange tmp(0);
                inv.umv(shapeGrad[dof][0], tmp[0]);
                shapeGrad[dof][0] = tmp[0];
                //std::cout << "Gradient " << dof << ": " << shape_grads[dof] << std::endl;

                
                
            }

            // Get the value of the shape functions
            RangeType shapeFunction[2];
            for(int i=0; i<2; i++) 
                baseSet.eval(i,quadPos,shapeFunction[i]);

            // //////////////////////////////////
            //   Interpolate
            // //////////////////////////////////

            double x_s     = localSolution[0][0]*shapeGrad[0][0][0] + localSolution[1][0]*shapeGrad[1][0][0];
            double y_s     = localSolution[0][1]*shapeGrad[0][0][0] + localSolution[1][1]*shapeGrad[1][0][0];
            double theta_s = localSolution[0][2]*shapeGrad[0][0][0] + localSolution[1][2]*shapeGrad[1][0][0];

            double theta   = localSolution[0][2]*shapeFunction[0] + localSolution[1][2]*shapeFunction[1];

            // /////////////////////////////////////////////
            //   Sum it all up
            // /////////////////////////////////////////////

            double partA1 = x_s * cos(theta) - y_s * sin(theta);
            double partA3 = x_s * sin(theta) + y_s * cos(theta) - 1;


            energy += 0.5 * weight * (B * theta_s * theta_s
                                      + A1 * partA1 * partA1
                                      + A3 * partA3 * partA3);

//             printf("energy %g\n", energy);
        }

    }

    return energy;

}
