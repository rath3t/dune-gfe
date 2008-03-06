#ifndef AVERAGE_INTERFACE_HH
#define AVERAGE_INTERFACE_HH

#include <dune/common/fmatrix.hh>
#include <dune/disc/shapefunctions/lagrangeshapefunctions.hh>

#include <dune/ag-common/dgindexset.hh>
#include <dune/ag-common/crossproduct.hh>
#include <dune/ag-common/surfmassmatrix.hh>
#include "svd.hh"
#include "lapackpp.h"
#undef max

template <class GridType>
class PressureAverager : public Ipopt::TNLP
{
    typedef double field_type;

    typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> > MatrixType;
    typedef typename MatrixType::row_type RowType;
    

    enum {dim=GridType::dimension};

public:
    /** \brief Constructor */
    PressureAverager(const BoundaryPatch<GridType>* patch,
                     Dune::BlockVector<Dune::FieldVector<double,dim> >* result,
                     const Dune::FieldVector<double,dim>& resultantForce,
                     const Dune::FieldVector<double,dim>& resultantTorque,
                     const Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> >* massMatrix,
                     const Dune::BlockVector<Dune::FieldVector<double,1> >* nodalWeights,
                     const Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> >* constraintJacobian)
        : jacobianCutoff_(1e-8), patch_(patch), x_(result),
          massMatrix_(massMatrix), nodalWeights_(nodalWeights),
          constraintJacobian_(constraintJacobian),
          resultantForce_(resultantForce), resultantTorque_(resultantTorque)
    {
        patchArea_ = patch->area();
    }
    
    /** default destructor */
    virtual ~PressureAverager() {};

  /**@name Overloaded from TNLP */
  //@{
  /** Method to return some info about the nlp */
  virtual bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m, Ipopt::Index& nnz_jac_g,
                            Ipopt::Index& nnz_h_lag, IndexStyleEnum& index_style);

  /** Method to return the bounds for my problem */
  virtual bool get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u,
                               Ipopt::Index m, Ipopt::Number* g_l, Ipopt::Number* g_u);

  /** Method to return the starting point for the algorithm */
  virtual bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x,
                                  bool init_z, Ipopt::Number* z_L, Ipopt::Number* z_U,
                                  Ipopt::Index m, bool init_lambda,
                                  Ipopt::Number* lambda);

  /** Method to return the objective value */
  virtual bool eval_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number& obj_value);

  /** Method to return the gradient of the objective */
  virtual bool eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number* grad_f);

  /** Method to return the constraint residuals */
  virtual bool eval_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Index m, Ipopt::Number* g);

  /** Method to return:
   *   1) The structure of the jacobian (if "values" is NULL)
   *   2) The values of the jacobian (if "values" is not NULL)
   */
  virtual bool eval_jac_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                          Ipopt::Index m, Ipopt::Index nele_jac, Ipopt::Index* iRow, Ipopt::Index *jCol,
                          Ipopt::Number* values);

  /** Method to return:
   *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
   *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
   */
  virtual bool eval_h(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                      Ipopt::Number obj_factor, Ipopt::Index m, const Ipopt::Number* lambda,
                      bool new_lambda, Ipopt::Index nele_hess, Ipopt::Index* iRow,
                      Ipopt::Index* jCol, Ipopt::Number* values);

  //@}

  /** @name Solution Methods */
  //@{
  /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
    virtual void finalize_solution(Ipopt::SolverReturn status,
                                 Ipopt::Index n, const Ipopt::Number* x, const Ipopt::Number* z_L, const Ipopt::Number* z_U,
                                 Ipopt::Index m, const Ipopt::Number* g, const Ipopt::Number* lambda,
                                 Ipopt::Number obj_value,
                                   const Ipopt::IpoptData* ip_data,
                                   Ipopt::IpoptCalculatedQuantities* ip_cq);
  //@}

    // /////////////////////////////////
    //   Data
    // /////////////////////////////////

    /** \brief All entries in the constraint Jacobian smaller than the value
        here are removed.  This increases stability.
    */
    const double jacobianCutoff_;

    const BoundaryPatch<GridType>* patch_;

    double patchArea_;

    const Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> >* massMatrix_;

    const Dune::BlockVector<Dune::FieldVector<double,1> >* nodalWeights_;

    const Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> >* constraintJacobian_;

    Dune::BlockVector<Dune::FieldVector<double,dim> >* x_;

    Dune::FieldVector<double,dim> resultantForce_;
    Dune::FieldVector<double,dim> resultantTorque_;

private:
  /**@name Methods to block default compiler methods.
   */
  //@{
  //  PressureAverager();
  PressureAverager(const PressureAverager&);
  PressureAverager& operator=(const PressureAverager&);
  //@}
};

// returns the size of the problem
template <class GridType>
bool PressureAverager<GridType>::
get_nlp_info(Ipopt::Index& n, Ipopt::Index& m, Ipopt::Index& nnz_jac_g,
             Ipopt::Index& nnz_h_lag, IndexStyleEnum& index_style)
{
    // One variable for each vertex on the coupling boundary, and three for the closest constant field
    n = patch_->numVertices()*dim + dim;

    // prescribed total forces and moments
    m = 2*dim;

    // number of nonzeroes in the constraint Jacobian
    // leave out the very small ones, as they create instabilities
    nnz_jac_g = 0;
    for (int i=0; i<m; i++) {
            
        const RowType& jacobianRow = (*constraintJacobian_)[i];
            
        for (typename RowType::ConstIterator cIt = jacobianRow.begin(); cIt!=jacobianRow.end(); ++cIt)
            if ( (*cIt)[0][0] > jacobianCutoff_ )
                nnz_jac_g++;
        
    }
    
    // We only need the lower left corner of the Hessian (since it is symmetric)
    if (!massMatrix_)
        DUNE_THROW(SolverError, "No mass matrix has been supplied!");

    nnz_h_lag = 0;

    // use the C style indexing (0-based)
    index_style = Ipopt::TNLP::C_STYLE;
    
    return true;
}


// returns the variable bounds
template <class GridType>
bool PressureAverager<GridType>::
get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u,
                Ipopt::Index m, Ipopt::Number* g_l, Ipopt::Number* g_u)
{
    // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
    // If desired, we could assert to make sure they are what we think they are.
    //assert(n == x_->dim());
    //assert(m == 0);

    // Be on the safe side: unset all variable bounds
    for (size_t i=0; i<n; i++) {
        x_l[i] = -std::numeric_limits<double>::max();
        x_u[i] =  std::numeric_limits<double>::max();
    }

    for (int i=0; i<dim; i++) {
        g_l[i]     = g_u[i]     = resultantForce_[i];
        g_l[i+dim] = g_u[i+dim] = resultantTorque_[i];
    }

  return true;
}

// returns the initial point for the problem
template <class GridType>
bool PressureAverager<GridType>::
get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x,
                   bool init_z, Ipopt::Number* z_L, Ipopt::Number* z_U,
                   Ipopt::Index m, bool init_lambda, Ipopt::Number* lambda)
{
    // Here, we assume we only have starting values for x, if you code
    // your own NLP, you can provide starting values for the dual variables
    // if you wish
    assert(init_x == true);
    assert(init_z == false);
    assert(init_lambda == false);
    
    // initialize to the given starting point
    for (int i=0; i<n; i++)
        x[i] = 0;

    return true;
}

// returns the value of the objective function
template <class GridType>
bool PressureAverager<GridType>::
eval_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number& obj_value)
{
//     std::cout << "x:" << std::endl;
//     for (int i=0; i<n; i++)
//         std::cout << x[i] << std::endl;

    // Init return value
    obj_value = 0;

    ////////////////////////////////////
    // Compute x^T*A*x
    ////////////////////////////////////
    
    for (int rowIdx=0; rowIdx<massMatrix_->N(); rowIdx++) {
        
        const typename MatrixType::row_type& row = (*massMatrix_)[rowIdx];
        
        typename MatrixType::row_type::ConstIterator cIt   = row.begin();
        typename MatrixType::row_type::ConstIterator cEndIt = row.end();
        
        for (; cIt!=cEndIt; ++cIt)
            for (int i=0; i<dim; i++)
                obj_value += x[dim*rowIdx+i] * x[dim*cIt.index()+i] * (*cIt)[0][0];

    }

    // -b(x)
    for (int i=0; i<nodalWeights_->size(); i++)
        for (int j=0; j<dim; j++)
            obj_value -= 2 * x[n-dim + j] * x[i*dim+j] * (*nodalWeights_)[i];
  
    // += c^2 * \int 1 ds
    for (int i=0; i<dim; i++)
        obj_value += patchArea_ * (x[n-dim + i] * x[n-dim + i]);

    //std::cout << "IPOPT Energy: " << obj_value << std::endl;
    //exit(0);

  return true;
}

// return the gradient of the objective function grad_{x} f(x)
template <class GridType>
bool PressureAverager<GridType>::
eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number* grad_f)
{
    //std::cout << "### eval_grad_f ###" << std::endl;

    // \nabla J = A(x,.) - b(x)
    for (int i=0; i<n; i++)
        grad_f[i] = 0;

    printmatrix(std::cout, *massMatrix_, "mass", "--");

    for (int i=0; i<massMatrix_->N(); i++) {

        const typename MatrixType::row_type& row = (*massMatrix_)[i];

        typename MatrixType::row_type::ConstIterator cIt   = row.begin();
        typename MatrixType::row_type::ConstIterator cEndIt = row.end();
        
        for (; cIt!=cEndIt; ++cIt) 
            for (int j=0; j<dim; j++)
                grad_f[i*dim+j] += 2 * (*cIt)[0][0] * x[cIt.index()*dim+j];

        for (int j=0; j<dim; j++)
            grad_f[i*dim+j] -= 2 * x[n-dim+j] * (*nodalWeights_)[i];
        
    }

    for (int i=0; i<dim; i++) {

        for (int j=0; j<nodalWeights_->size(); j++) 
            grad_f[n-dim+i] -= 2* (*nodalWeights_)[j]*x[j*dim+i];

        grad_f[n-dim+i] += 2*x[n-dim+i]*patchArea_;
        
    }

    for (int i=0; i<n; i++) {
        std::cout << "x = " <<  x[i] << std::endl;
        std::cout << "grad = " <<  grad_f[i] << std::endl;
    }

  return true;
}

// return the value of the constraints: g(x)
template <class GridType>
bool PressureAverager<GridType>::
eval_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Index m, Ipopt::Number* g)
{
    for (int i=0; i<m; i++) {

        // init
        g[i] = 0;

        const RowType& jacobianRow = (*constraintJacobian_)[i];

        for (typename RowType::ConstIterator cIt = jacobianRow.begin(); cIt!=jacobianRow.end(); ++cIt)
            if ( (*cIt)[0][0] > jacobianCutoff_ )
                g[i] += (*cIt)[0][0] * x[cIt.index()];

    }
        
  return true;
}

// return the structure or values of the jacobian
template <class GridType>
bool PressureAverager<GridType>::
eval_jac_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
           Ipopt::Index m, Ipopt::Index nele_jac, Ipopt::Index* iRow, Ipopt::Index *jCol,
           Ipopt::Number* values)
{
    int idx = 0;

    if (values==NULL) {

        for (int i=0; i<m; i++) {
            
            const RowType& jacobianRow = (*constraintJacobian_)[i];
            
            for (typename RowType::ConstIterator cIt = jacobianRow.begin(); cIt!=jacobianRow.end(); ++cIt) {
                if ( (*cIt)[0][0] > jacobianCutoff_ ) {
                    iRow[idx] = i;
                    jCol[idx] = cIt.index();
                    idx++;
                }
            }
            
        }

    } else {

        for (int i=0; i<m; i++) {
            
            const RowType& jacobianRow = (*constraintJacobian_)[i];
            
            for (typename RowType::ConstIterator cIt = jacobianRow.begin(); cIt!=jacobianRow.end(); ++cIt)
                if ( (*cIt)[0][0] > jacobianCutoff_ )
                    values[idx++] = (*cIt)[0][0];
            
        }



    }

 return true;
}

//return the structure or values of the hessian
template <class GridType>
bool PressureAverager<GridType>::
eval_h(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
       Ipopt::Number obj_factor, Ipopt::Index m, const Ipopt::Number* lambda,
       bool new_lambda, Ipopt::Index nele_hess, Ipopt::Index* iRow,
       Ipopt::Index* jCol, Ipopt::Number* values)
{
    // We are using a quasi-Hessian approximation
    return false;
}

template <class GridType>
void PressureAverager<GridType>::
finalize_solution(Ipopt::SolverReturn status,
                  Ipopt::Index n, const Ipopt::Number* x, const Ipopt::Number* z_L, const Ipopt::Number* z_U,
                  Ipopt::Index m, const Ipopt::Number* g, const Ipopt::Number* lambda,
                  Ipopt::Number obj_value,
                  const Ipopt::IpoptData* ip_data,
                  Ipopt::IpoptCalculatedQuantities* ip_cq)
{
    x_->resize(patch_->numVertices());
    
    for (int i=0; i<x_->size(); i++)
        for (int j=0; j<dim; j++)
            (*x_)[i][j] = x[i*dim+j];

    std::cout << "Closest constant: ";// << x[n-dim] << "  " << x[n-dim+1] << "  " << [x-dim+2] << std::endl;

}



// Given a resultant force and torque (from a rod problem), this method computes the corresponding
// Neumann data for a 3d elasticity problem.
template <class GridType>
void computeAveragePressureIPOpt(const Dune::FieldVector<double,GridType::dimension>& resultantForce,
                            const Dune::FieldVector<double,GridType::dimension>& resultantTorque,
                            const BoundaryPatch<GridType>& interface,
                            const Configuration& crossSection,
                            Dune::BlockVector<Dune::FieldVector<double, GridType::dimension> >& pressure)
{
    const GridType& grid = interface.getGrid();
    const int level      = interface.level();
    const typename GridType::Traits::LevelIndexSet& indexSet = grid.levelIndexSet(level);
    const int dim        = GridType::dimension;
    typedef typename GridType::ctype ctype;
    typedef double field_type;

    typedef typename GridType::template Codim<dim>::LevelIterator VertexIterator;

    // Create the matrix of constraints
    Dune::BCRSMatrix<Dune::FieldMatrix<field_type,1,1> > matrix(2*dim, dim*interface.numVertices(),
                                                                Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> >::random);

    for (int i=0; i<dim; i++) {
        matrix.setrowsize(i,     interface.numVertices());
        matrix.setrowsize(i+dim, dim*interface.numVertices());
    }

    matrix.endrowsizes();

    for (int i=0; i<dim; i++)
        for (int j=0; j<interface.numVertices(); j++)
            matrix.addindex(i, dim*j+i);

    for (int i=0; i<dim; i++)
        for (int j=0; j<dim*interface.numVertices(); j++)
            matrix.addindex(i+dim, j);

    matrix.endindices();

    matrix = 0;

    // Create the surface mass matrix
    Dune::BCRSMatrix<Dune::FieldMatrix<field_type,1,1> > massMatrix;
    assembleSurfaceMassMatrix<GridType,1>(interface, massMatrix);

    // Make global-to-local array
    std::vector<int> globalToLocal;
    interface.makeGlobalToLocal(globalToLocal);

    // Make array of nodal weights
    Dune::BlockVector<Dune::FieldVector<double,1> > nodalWeights(interface.numVertices());
    nodalWeights = 0;

    typename GridType::template Codim<0>::LevelIterator eIt    = indexSet.template begin<0,Dune::All_Partition>();
    typename GridType::template Codim<0>::LevelIterator eEndIt = indexSet.template end<0,Dune::All_Partition>();

    for (; eIt!=eEndIt; ++eIt) {

        typename GridType::template Codim<0>::Entity::LevelIntersectionIterator nIt    = eIt->ilevelbegin();
        typename GridType::template Codim<0>::Entity::LevelIntersectionIterator nEndIt = eIt->ilevelend();
        
        for (; nIt!=nEndIt; ++nIt) {
            
            if (!interface.contains(*eIt,nIt))
                continue;

            const Dune::LagrangeShapeFunctionSet<ctype, field_type, dim-1>& baseSet
                = Dune::LagrangeShapeFunctions<ctype, field_type, dim-1>::general(nIt.intersectionGlobal().type(),1);

            const Dune::ReferenceElement<double,dim>& refElement = Dune::ReferenceElements<double, dim>::general(eIt->type());

            // four rows because a face may have no more than four vertices
            Dune::FieldVector<double,4> mu(0);
            Dune::FieldVector<double,3> mu_tilde[4][3];
            
            for (int i=0; i<4; i++)
                for (int j=0; j<3; j++)
                    mu_tilde[i][j] = 0;

            for (int i=0; i<nIt.intersectionGlobal().corners(); i++) {
                
                const Dune::QuadratureRule<double, dim-1>& quad 
                    = Dune::QuadratureRules<double, dim-1>::rule(nIt.intersectionGlobal().type(), dim-1);
                
                for (size_t qp=0; qp<quad.size(); qp++) {
                    
                    // Local position of the quadrature point
                    const Dune::FieldVector<double,dim-1>& quadPos = quad[qp].position();
                    
                    const double integrationElement         = nIt.intersectionGlobal().integrationElement(quadPos);
                    
                    // \mu_i = \int_t \varphi_i \ds
                    mu[i] += quad[qp].weight() * integrationElement * baseSet[i].evaluateFunction(0,quadPos);
                    
                    // \tilde{\mu}_i^j = \int_t \varphi_i \times (x - x_0) \ds
                    Dune::FieldVector<double,dim> worldPos = nIt.intersectionGlobal().global(quadPos);

                    for (int j=0; j<dim; j++) {

                        // Vector-valued basis function
                        Dune::FieldVector<double,dim> phi_i(0);
                        phi_i[j] = baseSet[i].evaluateFunction(0,quadPos);
                        
                        mu_tilde[i][j].axpy(quad[qp].weight() * integrationElement,
                                            crossProduct(worldPos-crossSection.r, phi_i));

                    }
                    
                }
                
            }

            // Set up matrix
            for (int i=0; i<baseSet.size(); i++) {
                
                int faceIdxi = refElement.subEntity(nIt.numberInSelf(), 1, i, dim);
                int subIndex = globalToLocal[indexSet.template subIndex<dim>(*eIt, faceIdxi)];

                nodalWeights[subIndex] += mu[i];
                for (int j=0; j<dim; j++)
                    matrix[j][subIndex*dim+j] += mu[i];

                for (int j=0; j<3; j++)
                    for (int k=0; k<3; k++)
                        matrix[dim+k][dim*subIndex+j] += mu_tilde[i][j][k];

            }

        }

    }

    //printmatrix(std::cout, matrix, "jacobian", "--");
    //printmatrix(std::cout, massMatrix, "mass", "--");

    // /////////////////////////////////////////////////////////////////////////////////////
    //   Set up and start the interior-point solver
    // /////////////////////////////////////////////////////////////////////////////////////

    // Create a new instance of IpoptApplication
    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = new Ipopt::IpoptApplication();
    
    // Change some options
    app->Options()->SetNumericValue("tol", 1e-8);
    app->Options()->SetIntegerValue("max_iter", 20);
    app->Options()->SetStringValue("mu_strategy", "adaptive");
    app->Options()->SetStringValue("output_file", "ipopt.out");
    app->Options()->SetStringValue("hessian_approximation", "limited-memory");

    // Intialize the IpoptApplication and process the options
    Ipopt::ApplicationReturnStatus status;
    status = app->Initialize();
    if (status != Ipopt::Solve_Succeeded) 
        DUNE_THROW(SolverError, "Error during IPOpt initialization!");
    
    // Ask Ipopt to solve the problem
    Dune::BlockVector<Dune::FieldVector<double,dim> > localPressure;
    Ipopt::SmartPtr<Ipopt::TNLP> defectSolverSmart = new PressureAverager<GridType>(&interface,
                                                                                    &localPressure,
                                                                                    resultantForce,
                                                                                    resultantTorque,
                                                                                    &massMatrix,
                                                                                    &nodalWeights,
                                                                                    &matrix);
    status = app->OptimizeTNLP(defectSolverSmart);
    
    if (status != Ipopt::Solve_Succeeded)
        DUNE_THROW(SolverError, "Solving the defect problem failed!");

    // //////////////////////////////////////////////////////////////////////////////
    //   Get result
    // //////////////////////////////////////////////////////////////////////////////

    // set up output array
    pressure.resize(indexSet.size(dim));
    pressure = 0;

    for (size_t i=0; i<globalToLocal.size(); i++)
        if (globalToLocal[i]>=0)
            pressure[i] = localPressure[globalToLocal[i]];

    // /////////////////////////////////////////////////////////////////////////////////////
    //   Compute the overall force and torque to see whether the preceding code is correct
    // /////////////////////////////////////////////////////////////////////////////////////
#if 1
    Dune::FieldVector<double,3> outputForce(0), outputTorque(0);

    eIt    = indexSet.template begin<0,Dune::All_Partition>();
    eEndIt = indexSet.template end<0,Dune::All_Partition>();

    for (; eIt!=eEndIt; ++eIt) {

        typename GridType::template Codim<0>::Entity::LevelIntersectionIterator nIt    = eIt->ilevelbegin();
        typename GridType::template Codim<0>::Entity::LevelIntersectionIterator nEndIt = eIt->ilevelend();
        
        for (; nIt!=nEndIt; ++nIt) {
            
            if (!interface.contains(*eIt,nIt))
                continue;

            const Dune::LagrangeShapeFunctionSet<double, double, dim-1>& baseSet
                = Dune::LagrangeShapeFunctions<double, double, dim-1>::general(nIt.intersectionGlobal().type(),1);
            
            const Dune::QuadratureRule<double, dim-1>& quad 
                = Dune::QuadratureRules<double, dim-1>::rule(nIt.intersectionGlobal().type(), dim-1);
            
            const Dune::ReferenceElement<double,dim>& refElement = Dune::ReferenceElements<double, dim>::general(eIt->type());

            for (size_t qp=0; qp<quad.size(); qp++) {
                
                // Local position of the quadrature point
                const Dune::FieldVector<double,dim-1>& quadPos = quad[qp].position();
                
                const double integrationElement         = nIt.intersectionGlobal().integrationElement(quadPos);
                
                // Evaluate function
                Dune::FieldVector<double,dim> localPressure(0);
                
                for (size_t i=0; i<baseSet.size(); i++) {

                    int faceIdxi = refElement.subEntity(nIt.numberInSelf(), 1, i, dim);
                    int subIndex = indexSet.template subIndex<dim>(*eIt, faceIdxi);
                    
                    localPressure.axpy(baseSet[i].evaluateFunction(0,quadPos),
                                       pressure[subIndex]);

                }

                // Sum up the total force
                outputForce.axpy(quad[qp].weight()*integrationElement, localPressure);

                // Sum up the total torque   \int (x - x_0) \times f dx
                Dune::FieldVector<double,dim> worldPos = nIt.intersectionGlobal().global(quadPos);
                outputTorque.axpy(quad[qp].weight()*integrationElement, 
                                  crossProduct(worldPos - crossSection.r, localPressure));

            }

        }

    }

    outputForce  -= resultantForce;
    outputTorque -= resultantTorque;
    assert( outputForce.infinity_norm() < 1e-6 );
    assert( outputTorque.infinity_norm() < 1e-6 );
//     std::cout << "Output force:  " << outputForce << std::endl;
//     std::cout << "Output torque: " << outputTorque << "      " << resultantTorque[0]/outputTorque[0] << std::endl;
#endif

}


// Given a resultant force and torque (from a rod problem), this method computes the corresponding
// Neumann data for a 3d elasticity problem.
template <class GridType>
void computeAveragePressure(const Dune::FieldVector<double,GridType::dimension>& resultantForce,
                            const Dune::FieldVector<double,GridType::dimension>& resultantTorque,
                            const BoundaryPatch<GridType>& interface,
                            const Configuration& crossSection,
                            Dune::BlockVector<Dune::FieldVector<double, GridType::dimension> >& pressure)
{
    const GridType& grid = interface.getGrid();
    const int level      = interface.level();
    const typename GridType::Traits::LevelIndexSet& indexSet = grid.levelIndexSet(level);
    const int dim        = GridType::dimension;
    typedef typename GridType::ctype ctype;
    typedef double field_type;

    typedef typename GridType::template Codim<dim>::LevelIterator VertexIterator;

    // Get total interface area
    ctype area = interface.area();

    // set up output array
    DGIndexSet<GridType> dgIndexSet(grid,level);
    dgIndexSet.setup(grid,level);
    pressure.resize(dgIndexSet.size());
    pressure = 0;
    
    typename GridType::template Codim<0>::LevelIterator eIt    = indexSet.template begin<0,Dune::All_Partition>();
    typename GridType::template Codim<0>::LevelIterator eEndIt = indexSet.template end<0,Dune::All_Partition>();

    for (; eIt!=eEndIt; ++eIt) {

        typename GridType::template Codim<0>::Entity::LevelIntersectionIterator nIt    = eIt->ilevelbegin();
        typename GridType::template Codim<0>::Entity::LevelIntersectionIterator nEndIt = eIt->ilevelend();
        
        for (; nIt!=nEndIt; ++nIt) {
            
            if (!interface.contains(*eIt,nIt))
                continue;

            const Dune::LagrangeShapeFunctionSet<ctype, field_type, dim-1>& baseSet
                = Dune::LagrangeShapeFunctions<ctype, field_type, dim-1>::general(nIt.intersectionGlobal().type(),1);

            // four rows because a face may have no more than four vertices
            Dune::FieldVector<double,4> mu(0);
            Dune::FieldVector<double,3> mu_tilde[4][3];
            
            for (int i=0; i<4; i++)
                for (int j=0; j<3; j++)
                    mu_tilde[i][j] = 0;

            for (int i=0; i<nIt.intersectionGlobal().corners(); i++) {
                
                const Dune::QuadratureRule<double, dim-1>& quad 
                    = Dune::QuadratureRules<double, dim-1>::rule(nIt.intersectionGlobal().type(), dim-1);
                
                for (size_t qp=0; qp<quad.size(); qp++) {
                    
                    // Local position of the quadrature point
                    const Dune::FieldVector<double,dim-1>& quadPos = quad[qp].position();
                    
                    const double integrationElement         = nIt.intersectionGlobal().integrationElement(quadPos);
                    
                    // \mu_i = \int_t \varphi_i \ds
                    mu[i] += quad[qp].weight() * integrationElement * baseSet[i].evaluateFunction(0,quadPos);
                    
                    // \tilde{\mu}_i^j = \int_t \varphi_i \times (x - x_0) \ds
                    Dune::FieldVector<double,dim> worldPos = nIt.intersectionGlobal().global(quadPos);

                    for (int j=0; j<dim; j++) {

                        // Vector-valued basis function
                        Dune::FieldVector<double,dim> phi_i(0);
                        phi_i[j] = baseSet[i].evaluateFunction(0,quadPos);
                        
                        mu_tilde[i][j].axpy(quad[qp].weight() * integrationElement,
                                            crossProduct(worldPos-crossSection.r, phi_i));

                    }
                    
                }
                
            }

            // Set up matrix
#if 0  // DUNE style
            Dune::Matrix<Dune::FieldMatrix<double,1,1> > matrix(6, 3*baseSet.size());
            matrix = 0;
            for (int i=0; i<baseSet.size(); i++)
                for (int j=0; j<3; j++)
                    matrix[j][i*3+j] = mu[i];

            for (int i=0; i<baseSet.size(); i++)
                for (int j=0; j<3; j++)
                    for (int k=0; k<3; k++)
                        matrix[3+k][3*i+j] = mu_tilde[i][j][k];

            Dune::BlockVector<Dune::FieldVector<double,1> > u(3*baseSet.size());
            Dune::FieldVector<double,6> b;

            // Scale the resultant force and torque with this segments area percentage.
            // That way the resulting pressure gets distributed fairly uniformly.
            ctype segmentArea = nIt.intersectionGlobal().volume() / area;

            for (int i=0; i<3; i++) {
                b[i]   = resultantForce[i] * segmentArea;
                b[i+3] = resultantTorque[i] * segmentArea;
            }

            matrix.solve(u,b);

#else   // LaPack++ style
            LaGenMatDouble matrix(6, 3*baseSet.size());
            matrix = 0;
            for (int i=0; i<baseSet.size(); i++)
                for (int j=0; j<3; j++)
                    matrix(j, i*3+j) = mu[i];

            for (int i=0; i<baseSet.size(); i++)
                for (int j=0; j<3; j++)
                    for (int k=0; k<3; k++)
                        matrix(3+k, 3*i+j) = mu_tilde[i][j][k];

            LaVectorDouble u(3*baseSet.size());
            LaVectorDouble b(6);

            // Scale the resultant force and torque with this segments area percentage.
            // That way the resulting pressure gets distributed fairly uniformly.
            ctype segmentArea = nIt.intersectionGlobal().volume() / area;

            for (int i=0; i<3; i++) {
                b(i)   = resultantForce[i] * segmentArea;
                b(i+3) = resultantTorque[i] * segmentArea;
            }

            for (int i=0; i<6; i++) {
                for (int j=0; j<3*baseSet.size(); j++)
                    std::cout << matrix(i,j) << "  ";
                std::cout << std::endl;
            }

            LaLinearSolve(matrix, u, b);
#endif
//             std::cout << b << std::endl;
//             std::cout << matrix << std::endl;
            //std::cout << u << std::endl;

            for (int i=0; i<baseSet.size(); i++)
                for (int j=0; j<3; j++)
                    pressure[dgIndexSet(*eIt, nIt.numberInSelf())+i][j]   = u(i*3+j);

        }

    }


    // /////////////////////////////////////////////////////////////////////////////////////
    //   Compute the overall force and torque to see whether the preceding code is correct
    // /////////////////////////////////////////////////////////////////////////////////////
#ifdef NDEBUG
    Dune::FieldVector<double,3> outputForce(0), outputTorque(0);

    eIt    = indexSet.template begin<0,Dune::All_Partition>();
    eEndIt = indexSet.template end<0,Dune::All_Partition>();

    for (; eIt!=eEndIt; ++eIt) {

        typename GridType::template Codim<0>::Entity::LevelIntersectionIterator nIt    = eIt->ilevelbegin();
        typename GridType::template Codim<0>::Entity::LevelIntersectionIterator nEndIt = eIt->ilevelend();
        
        for (; nIt!=nEndIt; ++nIt) {
            
            if (!interface.contains(*eIt,nIt))
                continue;

            const Dune::LagrangeShapeFunctionSet<double, double, dim-1>& baseSet
                = Dune::LagrangeShapeFunctions<double, double, dim-1>::general(nIt.intersectionGlobal().type(),1);
            
            const Dune::QuadratureRule<double, dim-1>& quad 
                = Dune::QuadratureRules<double, dim-1>::rule(nIt.intersectionGlobal().type(), dim-1);
            
            for (size_t qp=0; qp<quad.size(); qp++) {
                
                // Local position of the quadrature point
                const Dune::FieldVector<double,dim-1>& quadPos = quad[qp].position();
                
                const double integrationElement         = nIt.intersectionGlobal().integrationElement(quadPos);
                
                // Evaluate function
                Dune::FieldVector<double,dim> localPressure(0);
                
                for (size_t i=0; i<baseSet.size(); i++) 
                    localPressure.axpy(baseSet[i].evaluateFunction(0,quadPos),
                                       pressure[dgIndexSet(*eIt,nIt.numberInSelf())+i]);


                // Sum up the total force
                outputForce.axpy(quad[qp].weight()*integrationElement, localPressure);

                // Sum up the total torque   \int (x - x_0) \times f dx
                Dune::FieldVector<double,dim> worldPos = nIt.intersectionGlobal().global(quadPos);
                outputTorque.axpy(quad[qp].weight()*integrationElement, 
                                  crossProduct(worldPos - crossSection.r, localPressure));

            }

        }

    }

    outputForce  -= resultantForce;
    outputTorque -= resultantTorque;
    assert( outputForce.infinity_norm() < 1e-6 );
    assert( outputTorque.infinity_norm() < 1e-6 );
//     std::cout << "Output force:  " << outputForce << std::endl;
//     std::cout << "Output torque: " << outputTorque << "      " << resultantTorque[0]/outputTorque[0] << std::endl;
#endif

}



template <class GridType>
void averageSurfaceDGFunction(const GridType& grid,
                              const Dune::BlockVector<Dune::FieldVector<double,GridType::dimension> >& dgFunction,
                              Dune::BlockVector<Dune::FieldVector<double,GridType::dimension> >& p1Function,
                              const DGIndexSet<GridType>& dgIndexSet)
{
    const int dim = GridType::dimension;

    const typename GridType::Traits::LeafIndexSet& indexSet = grid.leafIndexSet();

    p1Function.resize(indexSet.size(dim));
    p1Function = 0;

    std::vector<int> counter(indexSet.size(dim), 0);

    typename GridType::template Codim<0>::LeafIterator eIt    = grid.template leafbegin<0>();
    typename GridType::template Codim<0>::LeafIterator eEndIt = grid.template leafend<0>();

    for (; eIt!=eEndIt; ++eIt) {

        typename GridType::template Codim<0>::Entity::LeafIntersectionIterator nIt    = eIt->ileafbegin();
        typename GridType::template Codim<0>::Entity::LeafIntersectionIterator nEndIt = eIt->ileafend();

        for (; nIt!=nEndIt; ++nIt) {

            if (!nIt.boundary())
                continue;

            const Dune::ReferenceElement<double,dim>& refElement 
                = Dune::ReferenceElements<double, dim>::general(eIt->type());

            for (int i=0; i<refElement.size(nIt.numberInSelf(),1,dim); i++) {

                int idxInElement = refElement.subEntity(nIt.numberInSelf(),1, i, dim);
                
                p1Function[indexSet.template subIndex<dim>(*eIt,idxInElement)] 
                    += dgFunction[dgIndexSet(*eIt,nIt.numberInSelf())+i];
                
                counter[indexSet.template subIndex<dim>(*eIt,idxInElement)]++;
                
            }

        }

    }

    for (int i=0; i<p1Function.size(); i++)
        if (counter[i]!=0)
            p1Function[i] /= counter[i];

}


template <class GridType>
void computeAverageInterface(const BoundaryPatch<GridType>& interface,
                             const Dune::BlockVector<Dune::FieldVector<double,GridType::dimension> > deformation,
                             Configuration& average)
{
    using namespace Dune;

    typedef typename GridType::template Codim<0>::LevelIterator ElementIterator;
    typedef typename GridType::template Codim<0>::Entity EntityType;
    typedef typename EntityType::LevelIntersectionIterator NeighborIterator;

    const GridType& grid = interface.getGrid();
    const int level      = interface.level();
    const typename GridType::Traits::LevelIndexSet& indexSet = grid.levelIndexSet(level);
    const int dim        = GridType::dimension;

    // ///////////////////////////////////////////
    //   Initialize output configuration
    // ///////////////////////////////////////////
    average.r = 0;
    
    double interfaceArea = 0;
    FieldMatrix<double,dim,dim> deformationGradient(0);

    // ///////////////////////////////////////////
    //   Loop and integrate over the interface
    // ///////////////////////////////////////////
    ElementIterator eIt    = grid.template lbegin<0>(level);
    ElementIterator eEndIt = grid.template lend<0>(level);
    for (; eIt!=eEndIt; ++eIt) {

        NeighborIterator nIt    = eIt->ilevelbegin();
        NeighborIterator nEndIt = eIt->ilevelend();

        for (; nIt!=nEndIt; ++nIt) {

            if (!interface.contains(*eIt, nIt))
                continue;

            const typename NeighborIterator::Geometry& segmentGeometry = nIt.intersectionGlobal();

            // Get quadrature rule
            const QuadratureRule<double, dim-1>& quad = QuadratureRules<double, dim-1>::rule(segmentGeometry.type(), dim-1);

            // Get set of shape functions on this segment
            const typename LagrangeShapeFunctionSetContainer<double,double,dim>::value_type& sfs
                = LagrangeShapeFunctions<double,double,dim>::general(eIt->type(),1);

            /* Loop over all integration points */
            for (int ip=0; ip<quad.size(); ip++) {
                
                // Local position of the quadrature point
                const FieldVector<double,dim> quadPos = nIt.intersectionSelfLocal().global(quad[ip].position());
                
                const double integrationElement = segmentGeometry.integrationElement(quad[ip].position());

                // Evaluate base functions
                FieldVector<double,dim> posAtQuadPoint(0);

                for(int i=0; i<eIt->geometry().corners(); i++) {

                    int idx = indexSet.template subIndex<dim>(*eIt, i);

                    // Deformation at the quadrature point 
                    posAtQuadPoint.axpy(sfs[i].evaluateFunction(0,quadPos), deformation[idx]);
                }

                const FieldMatrix<double,dim,dim>& inv = eIt->geometry().jacobianInverseTransposed(quadPos);
                
                /**********************************************/
                /* compute gradients of the shape functions   */
                /**********************************************/
                std::vector<FieldVector<double, dim> > shapeGrads(eIt->geometry().corners());
                
                for (int dof=0; dof<eIt->geometry().corners(); dof++) {
                    
                    FieldVector<double,dim> tmp;
                    for (int i=0; i<dim; i++)
                        tmp[i] = sfs[dof].evaluateDerivative(0, i, quadPos);
                    
                    // multiply with jacobian inverse 
                    shapeGrads[dof] = 0;
                    inv.umv(tmp, shapeGrads[dof]);
                }

                /****************************************************/
                // The deformation gradient of the deformation
                // in formulas: F(i,j) = $\partial \phi_i / \partial x_j$
                // or F(i,j) = Id + $\partial u_i / \partial x_j$
                /****************************************************/
                FieldMatrix<double, dim, dim> F;
                for (int i=0; i<dim; i++) {
                    
                    for (int j=0; j<dim; j++) {
                        
                        F[i][j] = (i==j) ? 1 : 0;
                        
                        for (int k=0; k<eIt->geometry().corners(); k++)
                            F[i][j] += deformation[indexSet.template subIndex<dim>(*eIt, k)][i]*shapeGrads[k][j];
                        
                    }
                    
                }

                // Sum up interface area
                interfaceArea += quad[ip].weight() * integrationElement;

                // Sum up average displacement
                average.r.axpy(quad[ip].weight() * integrationElement, posAtQuadPoint);

                // Sum up average deformation gradient
                for (int i=0; i<dim; i++)
                    for (int j=0; j<dim; j++)
                        deformationGradient[i][j] += F[i][j] * quad[ip].weight() * integrationElement;

            }

        }

    }


    // average deformation of the interface is the integral over its
    // deformation divided by its area
    average.r /= interfaceArea;

    // average deformation is the integral over the deformation gradient
    // divided by its area
    deformationGradient /= interfaceArea;
    //std::cout << "deformationGradient: " << std::endl << deformationGradient << std::endl;

    // Get the rotational part of the deformation gradient by performing a 
    // polar composition.
    FieldVector<double,dim> W;
    FieldMatrix<double,dim,dim> V, VT, U;

    FieldMatrix<double,dim,dim> deformationGradientBackup = deformationGradient;

#if 1
    // returns a decomposition U W VT, where U is returned in the first argument
    svdcmp<double,dim,dim>(deformationGradient, W, V);

    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
            VT[i][j] = V[j][i];

    deformationGradient.rightmultiply(VT);
#else
    lapackSVD(deformationGradientBackup, U, W, VT);
    deformationGradient = U;
    deformationGradient.rightmultiply(VT);
#endif
    //std::cout << deformationGradient << std::endl;

    // deformationGradient now contains the orthogonal part of the polar decomposition
    assert( std::abs(1-deformationGradient.determinant()) < 1e-3);

    average.q.set(deformationGradient);
}

#endif
