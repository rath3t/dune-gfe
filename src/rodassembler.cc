#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/common/matrix.hh>

#include <dune/quadrature/quadraturerules.hh>

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
    
    //const BaseFunctionSetType & baseSet = this->functionSpace_.getBaseFunctionSet( entity );
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
        Array<FieldVector<double,gridDim> > shapeGrad(ndof);
        
        for (int dof=0; dof<ndof; dof++) {
            
            //baseSet.jacobian(dof, quadPos, shapeGrad[dof]);
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
        
#if 0
        double x_s     = localSolution[0][0]*shapeGrad[0][0] + localSolution[1][0]*shapeGrad[1][0];
        double y_s     = localSolution[0][1]*shapeGrad[0][0] + localSolution[1][1]*shapeGrad[1][0];

        double theta   = localSolution[0][2]*shapeFunction[0] + localSolution[1][2]*shapeFunction[1];

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
#endif
        
        
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
            Array<FieldVector<double,gridDim> > shapeGrad(numOfBaseFct);
            
            for (int dof=0; dof<numOfBaseFct; dof++) {
                
                for (int i=0; i<gridDim; i++)
                    shapeGrad[dof][i] = baseSet[dof].evaluateDerivative(0,i,quadPos);

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
            r_s[0]     = localSolution[0].r[0]*shapeGrad[0][0] + localSolution[1].r[0]*shapeGrad[1][0];
            r_s[1]     = localSolution[0].r[1]*shapeGrad[0][0] + localSolution[1].r[1]*shapeGrad[1][0];
            r_s[2]     = localSolution[0].r[2]*shapeGrad[0][0] + localSolution[1].r[2]*shapeGrad[1][0];

//             double theta_s = localSolution[0][2]*shapeGrad[0][0] + localSolution[1][2]*shapeGrad[1][0];

//             double theta   = localSolution[0][2]*shapeFunction[0] + localSolution[1][2]*shapeFunction[1];

            // /////////////////////////////////////////////
            //   Sum it all up
            // /////////////////////////////////////////////

#if 0
            double partA1 = A1 * (x_s * cos(theta) - y_s * sin(theta));
            double partA3 = A3 * (x_s * sin(theta) + y_s * cos(theta) - 1);

            for (int dof=0; dof<numOfBaseFct; dof++) {

                int globalDof = indexSet.template subIndex<gridDim>(*it,dof);
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


#endif
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
            
            // Get the derivative of the rotation at the quadrature point by interpolating in $H$ and normalizing
            Quaternion<double> q_s;
            q_s[0] = localSolution[0].q[0]*shapeGrad[0][0] + localSolution[1].q[0]*shapeGrad[1][0];
            q_s[1] = localSolution[0].q[1]*shapeGrad[0][0] + localSolution[1].q[1]*shapeGrad[1][0];
            q_s[2] = localSolution[0].q[2]*shapeGrad[0][0] + localSolution[1].q[2]*shapeGrad[1][0];
            q_s[3] = localSolution[0].q[3]*shapeGrad[0][0] + localSolution[1].q[3]*shapeGrad[1][0];

            q.normalize();

            // /////////////////////////////////////////////
            //   Sum it all up
            // /////////////////////////////////////////////

            // Part I: the shearing and stretching energy
            std::cout << "tangent : " << r_s << std::endl;
            FieldVector<double,3> v;
            v[0] = r_s * q.director(0);    // shear strain
            v[1] = r_s * q.director(1);    // shear strain
            v[2] = r_s * q.director(2);    // stretching strain

            std::cout << "strain : " << v << std::endl;

            energy += 0.5*A1*v[0]*v[0] + 0.5*A2*v[1]*v[1] + 0.5*A3*(v[2]-1)*(v[2]-1);

            // Part II: the bending and twisting energy
            
            FieldVector<double,3> u;  // The Darboux vector
            u[0] = 2 * ( q[3]*q_s[0] + q[2]*q_s[1] - q[1]*q_s[2] - q[0]*q_s[3]);
            u[1] = 2 * (-q[2]*q_s[0] + q[3]*q_s[1] + q[0]*q_s[2] - q[1]*q_s[3]);
            u[2] = 2 * ( q[1]*q_s[0] - q[0]*q_s[1] + q[3]*q_s[2] - q[2]*q_s[3]);

            std::cout << "Darboux vector : " << u << std::endl;

            energy += 0.5 * (K1*u[0]*u[0] + K2*u[1]*u[1] + K3*u[2]*u[2]);

        }

    }

    return energy;

}
