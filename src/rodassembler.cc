#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>

#include <dune/grid/common/quadraturerules.hh>

#include <dune/disc/shapefunctions/lagrangeshapefunctions.hh>


template <class GridType, class RT>
RT RodLocalStiffness<GridType, RT>::
energy(const Entity& element,
       const Dune::array<Configuration,2>& localSolution,
       const Dune::array<Configuration,2>& localReferenceConfiguration,
       int k)
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


template <class GridType, class RT>
void RodLocalStiffness<GridType, RT>::
interpolationDerivative(const Quaternion<RT>& q0, const Quaternion<RT>& q1, double s,
                        Dune::array<Quaternion<double>,6>& grad)
{
    // Clear output array
    for (int i=0; i<6; i++)
        grad[i] = 0;

    // The derivatives with respect to w^1

    // Compute q_1^{-1}q_0
    Quaternion<RT> q1InvQ0 = q1;
    q1InvQ0.invert();
    q1InvQ0 = q1InvQ0.mult(q0);

    {
    // Compute v = (1-s) \exp^{-1} ( q_1^{-1} q_0)
    Dune::FieldVector<RT,3> v = Quaternion<RT>::expInv(q1InvQ0);
    v *= (1-s);

    Dune::FieldMatrix<RT,4,3> dExp_v = Quaternion<RT>::Dexp(v);

    Dune::FieldMatrix<RT,3,4> dExpInv = Quaternion<RT>::DexpInv(q1InvQ0);

    Dune::FieldMatrix<RT,4,4> mat(0);
    for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
            for (int k=0; k<3; k++)
                mat[i][j] += (1-s) * dExp_v[i][k] * dExpInv[k][j];

    for (int i=0; i<3; i++) {

        Quaternion<RT> dw;
        for (int j=0; j<4; j++)
            dw[j] = 0.5 * (i==j);  // dExp[j][i] at v=0
        
        dw = q1InvQ0.mult(dw);
        
        mat.umv(dw,grad[i]);
        grad[i] = q1.mult(grad[i]);

    }
    }


    // The derivatives with respect to w^1

    // Compute q_0^{-1}
    Quaternion<RT> q0InvQ1 = q0;
    q0InvQ1.invert();
    q0InvQ1 = q0InvQ1.mult(q1);

    {
    // Compute v = s \exp^{-1} ( q_0^{-1} q_1)
    Dune::FieldVector<RT,3> v = Quaternion<RT>::expInv(q0InvQ1);
    v *= s;

    Dune::FieldMatrix<RT,4,3> dExp_v = Quaternion<RT>::Dexp(v);

    Dune::FieldMatrix<RT,3,4> dExpInv = Quaternion<RT>::DexpInv(q0InvQ1);

    Dune::FieldMatrix<RT,4,4> mat(0);
    for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
            for (int k=0; k<3; k++)
                mat[i][j] += s * dExp_v[i][k] * dExpInv[k][j];

    for (int i=3; i<6; i++) {

        Quaternion<RT> dw;
        for (int j=0; j<4; j++)
            dw[j] = 0.5 * ((i-3)==j);  // dExp[j][i-3] at v=0

        dw = q0InvQ1.mult(dw);
        
        mat.umv(dw,grad[i]);
        grad[i] = q0.mult(grad[i]);

    }
    }
}



template <class GridType, class RT>
void RodLocalStiffness<GridType, RT>::
interpolationVelocityDerivative(const Quaternion<RT>& q0, const Quaternion<RT>& q1, double s,
                                double intervalLength, Dune::array<Quaternion<double>,6>& grad)
{
    // Clear output array
    for (int i=0; i<6; i++)
        grad[i] = 0;

    // Compute q_0^{-1}
    Quaternion<RT> q0Inv = q0;
    q0Inv.invert();


    // Compute v = s \exp^{-1} ( q_0^{-1} q_1)
    Dune::FieldVector<RT,3> v = Quaternion<RT>::expInv(q0Inv.mult(q1));
    v *= s/intervalLength;

    Dune::FieldMatrix<RT,4,3> dExp_v = Quaternion<RT>::Dexp(v);

    Dune::array<Dune::FieldMatrix<RT,3,3>, 4> ddExp;
    Quaternion<RT>::DDexp(v, ddExp);

    Dune::FieldMatrix<RT,3,4> dExpInv = Quaternion<RT>::DexpInv(q0Inv.mult(q1));

    Dune::FieldMatrix<RT,4,4> mat(0);
    for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
            for (int k=0; k<3; k++)
                mat[i][j] += 1/intervalLength * dExp_v[i][k] * dExpInv[k][j];

    
    // /////////////////////////////////////////////////
    // The derivatives with respect to w^0
    // /////////////////////////////////////////////////
    for (int i=0; i<3; i++) {

        // \partial exp \partial w^1_j at 0
        Quaternion<RT> dw;
        for (int j=0; j<4; j++)
            dw[j] = 0.5*(i==j);  // dExp_v_0[j][i];

        // \xi = \exp^{-1} q_0^{-1} q_1
        Dune::FieldVector<RT,3> xi = Quaternion<RT>::expInv(q0Inv.mult(q1));

        Quaternion<RT> addend0;
        addend0 = 0;
        dExp_v.umv(xi,addend0);
        addend0 = dw.mult(addend0);
        addend0 /= intervalLength;

        //  \parder{\xi}{w^1_j} = ...
        Quaternion<RT> dwConj = dw;
        dwConj.conjugate();
        //dwConj[3] -= 2 * dExp_v_0[3][i];   the last row of dExp_v_0 is zero
        dwConj = dwConj.mult(q0Inv.mult(q1));

        Dune::FieldVector<RT,3> dxi(0);
        Quaternion<RT>::DexpInv(q0Inv.mult(q1)).umv(dwConj, dxi);

        Quaternion<RT> vHv;
        for (int j=0; j<4; j++) {
            vHv[j] = 0;
            // vHv[j] = dxi * DDexp * xi
            for (int k=0; k<3; k++)
                for (int l=0; l<3; l++)
                    vHv[j] += ddExp[j][k][l]*dxi[k]*xi[l];
        }

        vHv *= s/intervalLength/intervalLength;

        // Third addend
        mat.umv(dwConj,grad[i]);

        // add up
        grad[i] += addend0;
        grad[i] += vHv;

        grad[i] = q0.mult(grad[i]);
    }


    // /////////////////////////////////////////////////
    // The derivatives with respect to w^1
    // /////////////////////////////////////////////////
    for (int i=3; i<6; i++) {

        // \partial exp \partial w^1_j at 0
        Quaternion<RT> dw;
        for (int j=0; j<4; j++)
            dw[j] = 0.5 * ((i-3)==j);  // dw[j] = dExp_v_0[j][i-3];

        // \xi = \exp^{-1} q_0^{-1} q_1
        Dune::FieldVector<RT,3> xi = Quaternion<RT>::expInv(q0Inv.mult(q1));

        //  \parder{\xi}{w^1_j} = ...
        Dune::FieldVector<RT,3> dxi(0);
        dExpInv.umv(q0Inv.mult(q1.mult(dw)), dxi);

        Quaternion<RT> vHv;
        for (int j=0; j<4; j++) {
            // vHv[j] = dxi * DDexp * xi
            vHv[j] = 0;
            for (int k=0; k<3; k++)
                for (int l=0; l<3; l++)
                    vHv[j] += ddExp[j][k][l]*dxi[k]*xi[l];
        }

        vHv *= s/intervalLength/intervalLength;

        // ///////////////////////////////////
        //   second addend
        // ///////////////////////////////////
            
        
        dw = q0Inv.mult(q1.mult(dw));
        
        mat.umv(dw,grad[i]);
        grad[i] += vHv;

        grad[i] = q0.mult(grad[i]);

    }

}

template <class GridType, class RT>
Dune::FieldVector<double, 6> RodLocalStiffness<GridType, RT>::
getStrain(const Dune::array<Configuration,2>& localSolution,
          const Entity& element,
          const Dune::FieldVector<double,1>& pos) const
{
    if (!element.isLeaf())
        DUNE_THROW(Dune::NotImplemented, "Only for leaf elements");

    assert(localSolution.size() == 2);

    // Strain defined on each element
    Dune::FieldVector<double, 6> strain(0);

    // Extract local solution on this element
    const Dune::LagrangeShapeFunctionSet<double, double, 1> & baseSet 
        = Dune::LagrangeShapeFunctions<double, double, 1>::general(element.type(), 1);
    int numOfBaseFct = baseSet.size();
    
    const Dune::FieldMatrix<double,1,1>& inv = element.geometry().jacobianInverseTransposed(pos);
    
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
                                                                       pos);
    // Transformation from the reference element
    q_s *= inv[0][0];
        
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

template <class GridType, class RT>
void RodLocalStiffness<GridType, RT>::
assembleGradient(const Entity& element,
                 const Dune::array<Configuration,2>& solution,
                 const Dune::array<Configuration,2>& referenceConfiguration,
                 Dune::array<Dune::FieldVector<double,6>, 2>& gradient) const
{
    using namespace Dune;

    // Extract local solution on this element
    const Dune::LagrangeShapeFunctionSet<double, double, 1> & baseSet 
        = Dune::LagrangeShapeFunctions<double, double, 1>::general(element.type(), 1); // first order
    const int numOfBaseFct = baseSet.size();  
        
    // init
    for (size_t i=0; i<gradient.size(); i++)
        gradient[i] = 0;

    double intervalLength = element.geometry()[1][0] - element.geometry()[0][0];

    // ///////////////////////////////////////////////////////////////////////////////////
    //   Reduced integration to avoid locking:  assembly is split into a shear part 
    //   and a bending part.  Different quadrature rules are used for the two parts.
    //   This avoids locking.
    // ///////////////////////////////////////////////////////////////////////////////////
    
    // Get quadrature rule
    const QuadratureRule<double, 1>& shearingQuad = QuadratureRules<double, 1>::rule(element.type(), shearQuadOrder);

    for (int pt=0; pt<shearingQuad.size(); pt++) {
        
        // Local position of the quadrature point
        const FieldVector<double,1>& quadPos = shearingQuad[pt].position();
        
        const FieldMatrix<double,1,1>& inv = element.geometry().jacobianInverseTransposed(quadPos);
        const double integrationElement = element.geometry().integrationElement(quadPos);
        
        double weight = shearingQuad[pt].weight() * integrationElement;
        
        // ///////////////////////////////////////
        //   Compute deformation gradient
        // ///////////////////////////////////////
        double shapeGrad[numOfBaseFct];
        
        for (int dof=0; dof<numOfBaseFct; dof++) {
            
            shapeGrad[dof] = baseSet[dof].evaluateDerivative(0,0,quadPos);
            
            // multiply with jacobian inverse 
            FieldVector<double,1> tmp(0);
            inv.umv(shapeGrad[dof], tmp);
            shapeGrad[dof] = tmp;
            
        }
        
        // //////////////////////////////////
        //   Interpolate
        // //////////////////////////////////
        
        FieldVector<double,3> r_s;
        for (int i=0; i<3; i++)
            r_s[i] = solution[0].r[i]*shapeGrad[0] + solution[1].r[i]*shapeGrad[1];
        
        // Interpolate current rotation at this quadrature point
        Quaternion<double> q = Quaternion<double>::interpolate(solution[0].q, solution[1].q,quadPos[0]);
        
        // The current strain
        FieldVector<double,blocksize> strain = getStrain(solution, element, quadPos);
        
        // The reference strain
        FieldVector<double,blocksize> referenceStrain = getStrain(referenceConfiguration, element, quadPos);
        
        
        // dd_dvij[m][i][j] = \parder {(d_k)_i} {q}
        array<FieldMatrix<double,3 , 4>, 3> dd_dq;
        q.getFirstDerivativesOfDirectors(dd_dq);
        
        // First derivatives of the position
        array<Quaternion<double>,6> dq_dwij;
        interpolationDerivative(solution[0].q, solution[1].q, quadPos, dq_dwij);

        // /////////////////////////////////////////////
        //   Sum it all up
        // /////////////////////////////////////////////

        for (int i=0; i<numOfBaseFct; i++) {
            
            // /////////////////////////////////////////////
            //   The translational part
            // /////////////////////////////////////////////
            
            // \partial \bar{W} / \partial r^i_j
            for (int j=0; j<3; j++) {
                
                for (int m=0; m<3; m++) 
                    gradient[i][j] += weight 
                        * (  A_[m] * (strain[m] - referenceStrain[m]) * shapeGrad[i] * q.director(m)[j]);
                
            }
            
            // \partial \bar{W}_v / \partial v^i_j
            for (int j=0; j<3; j++) {
                
                for (int m=0; m<3; m++) {
                    FieldVector<double,3> tmp(0);
                    dd_dq[m].umv(dq_dwij[3*i+j], tmp);
                    gradient[i][3+j] += weight 
                        * A_[m] * (strain[m] - referenceStrain[m]) * (r_s*tmp);
                }
            }
            
        }
        
    }

    // /////////////////////////////////////////////////////
    //   Now: the bending/torsion part
    // /////////////////////////////////////////////////////

    // Get quadrature rule
    const QuadratureRule<double, 1>& bendingQuad = QuadratureRules<double, 1>::rule(element.type(), bendingQuadOrder);

    for (int pt=0; pt<bendingQuad.size(); pt++) {
        
        // Local position of the quadrature point
        const FieldVector<double,1>& quadPos = bendingQuad[pt].position();
        
        const FieldMatrix<double,1,1>& inv = element.geometry().jacobianInverseTransposed(quadPos);
        const double integrationElement = element.geometry().integrationElement(quadPos);
        
        double weight = bendingQuad[pt].weight() * integrationElement;
        
        // Interpolate current rotation at this quadrature point
        Quaternion<double> q = Quaternion<double>::interpolate(solution[0].q, solution[1].q,quadPos[0]);
        
        // Get the derivative of the rotation at the quadrature point by interpolating in $H$
        Quaternion<double> q_s = Quaternion<double>::interpolateDerivative(solution[0].q, solution[1].q,
                                                                           quadPos);
        // Transformation from the reference element
        q_s *= inv[0][0];
        
        
        // The current strain
        FieldVector<double,blocksize> strain = getStrain(solution, element, quadPos);
        
        // The reference strain
        FieldVector<double,blocksize> referenceStrain = getStrain(referenceConfiguration, element, quadPos);
        
        // First derivatives of the position
        array<Quaternion<double>,6> dq_dwij;
        interpolationDerivative(solution[0].q, solution[1].q, quadPos, dq_dwij);

        array<Quaternion<double>,6> dq_ds_dwij;
        interpolationVelocityDerivative(solution[0].q, solution[1].q, quadPos[0]*intervalLength, intervalLength, 
                                        dq_ds_dwij);

        // /////////////////////////////////////////////
        //   Sum it all up
        // /////////////////////////////////////////////

        for (int i=0; i<numOfBaseFct; i++) {
            
            // /////////////////////////////////////////////
            //   The rotational part
            // /////////////////////////////////////////////

            // \partial \bar{W}_v / \partial v^i_j
            for (int j=0; j<3; j++) {
                
                for (int m=0; m<3; m++) {
                    
                    // Compute derivative of the strain
                    double du_dvij_m = 2 * (dq_dwij[i*3+j].B(m) * q_s)
                        + 2* ( q.B(m) * dq_ds_dwij[i*3+j]);
                    
                    // Sum it up
                    gradient[i][3+j] += weight * K_[m] 
                        * (strain[m+3]-referenceStrain[m+3]) * du_dvij_m;
                    
                }
                
            }

        }
        
    }
}

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
                
                int iIdx = indexSet.template subIndex<gridDim>(*it,i);
                int jIdx = indexSet.template subIndex<gridDim>(*it,j);
                
                nb.add(iIdx, jIdx);
                
            }
            
        }
        
    }
    
}


template <class GridType>
void RodAssembler<GridType>::
assembleMatrix(const std::vector<Configuration>& sol,
               Dune::BCRSMatrix<MatrixBlock>& matrix) const
{
    using namespace Dune;

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
        //getLocalMatrix( it, localSolution, sol, numOfBaseFct, mat);
        DUNE_THROW(NotImplemented, "getLocalMatrix");

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
void RodAssembler<GridType>::
assembleMatrixFD(const std::vector<Configuration>& sol,
                 Dune::BCRSMatrix<MatrixBlock>& matrix) const
{
    using namespace Dune;

    double eps = 1e-4;

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

    Dune::array<Configuration,2> localReferenceConfiguration;
    Dune::array<Configuration,2> localSolution;

    // ///////////////////////////////////////////////////////////////
    //   Loop over all blocks of the outer matrix
    // ///////////////////////////////////////////////////////////////
    for (int i=0; i<matrix.N(); i++) {

        ColumnIterator cIt    = matrix[i].begin();
        ColumnIterator cEndIt = matrix[i].end();

        for (; cIt!=cEndIt; ++cIt) {

            // compute only the upper right triangular matrix
            if (cIt.index() < i)
                continue;

            // ////////////////////////////////////////////////////////////////////////////
            //   Compute a finite-difference approximation of this hessian matrix block
            // ////////////////////////////////////////////////////////////////////////////

            for (int j=0; j<6; j++) {

                for (int k=0; k<6; k++) {

                    // compute only the upper right triangular matrix
                    if (i==cIt.index() && k<j)
                        continue;

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

                            forwardEnergy += localStiffness.energy(*elements[i], localSolution, localReferenceConfiguration);

                            localSolution[0] = sol[i];
                            localSolution[1] = sol[i+1];

                            solutionEnergy += localStiffness.energy(*elements[i], localSolution, localReferenceConfiguration);

                            localSolution[0] = backwardSolution[i];
                            localSolution[1] = backwardSolution[i+1];

                            backwardEnergy += localStiffness.energy(*elements[i], localSolution, localReferenceConfiguration);

                        } 

                        if (i != 0) {

                            localReferenceConfiguration[0] = referenceConfiguration_[i-1];
                            localReferenceConfiguration[1] = referenceConfiguration_[i];

                            localSolution[0] = forwardSolution[i-1];
                            localSolution[1] = forwardSolution[i];

                            forwardEnergy += localStiffness.energy(*elements[i-1], localSolution, localReferenceConfiguration);

                            localSolution[0] = sol[i-1];
                            localSolution[1] = sol[i];

                            solutionEnergy += localStiffness.energy(*elements[i-1], localSolution, localReferenceConfiguration);

                            localSolution[0] = backwardSolution[i-1];
                            localSolution[1] = backwardSolution[i];

                            backwardEnergy += localStiffness.energy(*elements[i-1], localSolution, localReferenceConfiguration);

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

                            forwardForwardEnergy += localStiffness.energy(*elements[*it], localSolution, localReferenceConfiguration);

                            localSolution[0] = forwardBackwardSolution[*it];
                            localSolution[1] = forwardBackwardSolution[*it+1];

                            forwardBackwardEnergy += localStiffness.energy(*elements[*it], localSolution, localReferenceConfiguration);

                            localSolution[0] = backwardForwardSolution[*it];
                            localSolution[1] = backwardForwardSolution[*it+1];

                            backwardForwardEnergy += localStiffness.energy(*elements[*it], localSolution, localReferenceConfiguration);

                            localSolution[0] = backwardBackwardSolution[*it];
                            localSolution[1] = backwardBackwardSolution[*it+1];

                            backwardBackwardEnergy += localStiffness.energy(*elements[*it], localSolution, localReferenceConfiguration);


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

    // ///////////////////////////////////////////////////////////////
    //   Symmetrize the matrix
    //   This is possible expensive, but I want to be absolute sure
    //   that the matrix is symmetric.
    // ///////////////////////////////////////////////////////////////
    for (int i=0; i<matrix.N(); i++) {

        ColumnIterator cIt    = matrix[i].begin();
        ColumnIterator cEndIt = matrix[i].end();

        for (; cIt!=cEndIt; ++cIt) {

            if (cIt.index()>i)
                continue;


            if (cIt.index()==i) {

                for (int j=1; j<6; j++)
                    for (int k=0; k<j; k++)
                        (*cIt)[j][k] = (*cIt)[k][j];

            } else {

                const FieldMatrix<double,6,6>& other = matrix[cIt.index()][i];

                for (int j=0; j<6; j++)
                    for (int k=0; k<6; k++)
                        (*cIt)[j][k] = other[k][j];


            }


        }

    }

}


template <class GridType>
void RodAssembler<GridType>::
assembleGradient(const std::vector<Configuration>& sol,
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
    RodLocalStiffness<GridType,double> localStiffness(K, A);

    grad.resize(sol.size());
    grad = 0;

    ElementIterator it    = grid_->template lbegin<0>(maxlevel);
    ElementIterator endIt = grid_->template lend<0>(maxlevel);

    // Loop over all elements
    for (; it!=endIt; ++it) {

        // A 1d grid has two vertices
        const int nDofs = 2;

        // Extract local solution
        array<Configuration,nDofs> localSolution;
        
        for (int i=0; i<nDofs; i++)
            localSolution[i] = sol[indexSet.template subIndex<gridDim>(*it,i)];

        // Extract local reference configuration
        array<Configuration,nDofs> localReferenceConfiguration;
        
        for (int i=0; i<nDofs; i++)
            localReferenceConfiguration[i] = referenceConfiguration_[indexSet.template subIndex<gridDim>(*it,i)];

        // Assemble local gradient
        array<FieldVector<double,blocksize>, nDofs> localGradient;

        localStiffness.assembleGradient(*it, localSolution, localReferenceConfiguration, localGradient);

        // Add to global gradient
        for (int i=0; i<nDofs; i++)
            grad[indexSet.template subIndex<gridDim>(*it,i)] += localGradient[i];

    }

}


template <class GridType>
double RodAssembler<GridType>::
computeEnergy(const std::vector<Configuration>& sol) const
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
    RodLocalStiffness<GridType,double> localStiffness(K, A);

    Dune::array<Configuration,2> localReferenceConfiguration;
    Dune::array<Configuration,2> localSolution;

    ElementLeafIterator it    = grid_->template leafbegin<0>();
    ElementLeafIterator endIt = grid_->template leafend<0>();

    // Loop over all elements
    for (; it!=endIt; ++it) {

        for (int i=0; i<2; i++) {

            localReferenceConfiguration[i] = referenceConfiguration_[indexSet.template subIndex<gridDim>(*it,i)];
            localSolution[i]               = sol[indexSet.template subIndex<gridDim>(*it,i)];

        }

        energy += localStiffness.energy(*it, localSolution, localReferenceConfiguration);

    }

    return energy;

}


template <class GridType>
void RodAssembler<GridType>::
getStrain(const std::vector<Configuration>& sol,
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
    RodLocalStiffness<GridType,double> localStiffness(K, A);

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

        array<Configuration,2> localSolution;
        
        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[indexSet.template subIndex<gridDim>(*it,i)];

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
getStress(const std::vector<Configuration>& sol,
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
                  const std::vector<Configuration>& sol,
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

            if (!boundary.contains(*eIt, nIt->numberInSelf()))
                continue;

            // //////////////////////////////////////////////
            //   Compute force across this boundary face
            // //////////////////////////////////////////////

            //   Create local assembler
            
            Dune::array<double,3> K = {K_[0], K_[1], K_[2]};
            Dune::array<double,3> A = {A_[0], A_[1], A_[2]};
            RodLocalStiffness<GridType,double> localStiffness(K, A);

            double pos = nIt->intersectionSelfLocal()[0];

            Dune::array<Configuration,2> localSolution = {sol[indexSet.template subIndex<1>(*eIt,0)],
                                                          sol[indexSet.template subIndex<1>(*eIt,1)]};
            Dune::array<Configuration,2> localRefConf  = {referenceConfiguration_[indexSet.template subIndex<1>(*eIt,0)],
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

