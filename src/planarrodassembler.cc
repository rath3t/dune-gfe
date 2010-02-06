#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>

#include <dune/grid/common/quadraturerules.hh>

#include <dune/localfunctions/lagrange/p1.hh>

template <class GridType, int polOrd>
void Dune::PlanarRodAssembler<GridType, polOrd>::
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
                
                int iIdx = indexSet.subIndex(*it,i,gridDim);
                int jIdx = indexSet.subIndex(*it,j,gridDim);
                
                nb.add(iIdx, jIdx);
                
            }
            
        }
        
    }
    
}

template <class GridType, int polOrd>
void Dune::PlanarRodAssembler<GridType, polOrd>::
assembleMatrix(const std::vector<RigidBodyMotion<2> >& sol,
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






template <class GridType, int polOrd>
template <class MatrixType>
void Dune::PlanarRodAssembler<GridType, polOrd>::
getLocalMatrix( EntityType &entity, 
                const std::vector<RigidBodyMotion<2> >& localSolution,
                const int matSize, MatrixType& localMat) const
{
    const typename GridType::Traits::LevelIndexSet& indexSet = grid_->levelIndexSet(grid_->maxLevel());

    /* ndof is the number of vectors of the element */
    int ndof = matSize;

    for (int i=0; i<matSize; i++)
        for (int j=0; j<matSize; j++)
            localMat[i][j] = 0;
    
    P1LocalFiniteElement<double,double,gridDim> localFiniteElement;

    // Get quadrature rule
    const QuadratureRule<double, gridDim>& quad = QuadratureRules<double, gridDim>::rule(entity.type(), polOrd);
    
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
        std::vector<FieldMatrix<double,1,gridDim> > referenceElementGradients(ndof);
        localFiniteElement.localBasis().evaluateJacobian(quadPos,referenceElementGradients);

        std::vector<FieldVector<double,gridDim> > shapeGrad(ndof);
        
        // multiply with jacobian inverse 
        for (int dof=0; dof<ndof; dof++)
            inv.mv(referenceElementGradients[dof][0], shapeGrad[dof]);
        
        
        std::vector<FieldVector<double,1> > shapeFunction;
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

template <class GridType, int polOrd>
void Dune::PlanarRodAssembler<GridType, polOrd>::
assembleGradient(const std::vector<RigidBodyMotion<2> >& sol,
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
        P1LocalFiniteElement<double,double,gridDim> localFiniteElement;
        const int numOfBaseFct = localFiniteElement.localBasis().size();  
        
        RigidBodyMotion<2> localSolution[numOfBaseFct];
        
        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[indexSet.subIndex(*it,i,gridDim)];

        // Get quadrature rule
        const QuadratureRule<double, gridDim>& quad = QuadratureRules<double, gridDim>::rule(it->type(), polOrd);

        for (int pt=0; pt<quad.size(); pt++) {

            // Local position of the quadrature point
            const FieldVector<double,gridDim>& quadPos = quad[pt].position();
            
            const FieldMatrix<double,1,1>& inv = it->geometry().jacobianInverseTransposed(quadPos);
            const double integrationElement = it->geometry().integrationElement(quadPos);
        
            double weight = quad[pt].weight() * integrationElement;
            
            /**********************************************/
            /* compute gradients of the shape functions   */
            /**********************************************/
            std::vector<FieldMatrix<double,1,gridDim> > referenceElementGradients(numOfBaseFct);
            localFiniteElement.localBasis().evaluateJacobian(quadPos,referenceElementGradients);
            
            std::vector<FieldVector<double,gridDim> > shapeGrad(numOfBaseFct);
            
            // multiply with jacobian inverse 
            for (int dof=0; dof<numOfBaseFct; dof++)
                inv.mv(referenceElementGradients[dof][0], shapeGrad[dof]);

            // Get the values of the shape functions
            std::vector<FieldVector<double,1> > shapeFunction;
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


template <class GridType, int polOrd>
double Dune::PlanarRodAssembler<GridType, polOrd>::
computeEnergy(const std::vector<RigidBodyMotion<2> >& sol) const
{
    const int maxlevel = grid_->maxLevel();
    double energy = 0;

    const typename GridType::Traits::LevelIndexSet& indexSet = grid_->levelIndexSet(maxlevel);

    if (sol.size()!=grid_->size(maxlevel, gridDim))
        DUNE_THROW(Exception, "Solution vector doesn't match the grid!");

    ElementIterator it    = grid_->template lbegin<0>(maxlevel);
    ElementIterator endIt = grid_->template lend<0>(maxlevel);

    // Loop over all elements
    for (; it!=endIt; ++it) {

        // Extract local solution on this element
        P1LocalFiniteElement<double,double,gridDim> localFiniteElement;

        int numOfBaseFct = localFiniteElement.localBasis().size();

        RigidBodyMotion<2> localSolution[numOfBaseFct];
        
        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[indexSet.subIndex(*it,i,gridDim)];

        // Get quadrature rule
        const QuadratureRule<double, gridDim>& quad = QuadratureRules<double, gridDim>::rule(it->type(), polOrd);

        for (int pt=0; pt<quad.size(); pt++) {

            // Local position of the quadrature point
            const FieldVector<double,gridDim>& quadPos = quad[pt].position();
            
            const FieldMatrix<double,1,1>& inv = it->geometry().jacobianInverseTransposed(quadPos);
            const double integrationElement = it->geometry().integrationElement(quadPos);
        
            double weight = quad[pt].weight() * integrationElement;
            
            /**********************************************/
            /* compute gradients of the shape functions   */
            /**********************************************/
            std::vector<FieldMatrix<double,1,gridDim> > referenceElementGradients(numOfBaseFct);
            localFiniteElement.localBasis().evaluateJacobian(quadPos,referenceElementGradients);
            
            std::vector<FieldVector<double,gridDim> > shapeGrad(numOfBaseFct);
            
            // multiply with jacobian inverse 
            for (int dof=0; dof<numOfBaseFct; dof++)
                inv.mv(referenceElementGradients[dof][0], shapeGrad[dof]);

            // Get the value of the shape functions
            std::vector<FieldVector<double,1> > shapeFunction;
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
