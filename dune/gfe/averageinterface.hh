#ifndef AVERAGE_INTERFACE_HH
#define AVERAGE_INTERFACE_HH

#include <dune/common/fmatrix.hh>
#include <dune/localfunctions/lagrange/pqkfactory.hh>

#include <dune/fufem/dgindexset.hh>
#include <dune/fufem/crossproduct.hh>
#include <dune/fufem/surfmassmatrix.hh>
#include <dune/fufem/functions/basisgridfunction.hh>

#include <dune/solvers/common/numproc.hh>

#include "svd.hh"
#include "rigidbodymotion.hh"

#ifdef HAVE_IPOPT
#include "coin/IpTNLP.hpp"
#include "coin/IpIpoptApplication.hpp"
#else
#error You need IPOpt for this header!
#endif

template <class GridView>
class PressureAverager : public Ipopt::TNLP
{
    typedef double field_type;

    typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> > MatrixType;
    typedef typename MatrixType::row_type RowType;
    

    enum {dim=GridView::dimension};

public:
    /** \brief Constructor */
    PressureAverager(const BoundaryPatchBase<GridView>* patch,
                     Dune::BlockVector<Dune::FieldVector<double,dim> >* result,
                     const Dune::FieldVector<double,dim>& resultantForce,
                     const Dune::FieldVector<double,dim>& resultantTorque,
                     const Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> >* massMatrix,
                     const Dune::BlockVector<Dune::FieldVector<double,1> >* nodalWeights,
                     const Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> >* constraintJacobian)
        : jacobianCutoff_(1e-12), patch_(patch), x_(result),
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
        here are removed.  Apparently this is unnecessary though.
    */
    const double jacobianCutoff_;

    const BoundaryPatchBase<GridView>* patch_;

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
template <class GridView>
bool PressureAverager<GridView>::
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
            if ( std::abs((*cIt)[0][0]) > jacobianCutoff_ )
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
template <class GridView>
bool PressureAverager<GridView>::
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
template <class GridView>
bool PressureAverager<GridView>::
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
    for (int i=0; i<n/dim; i++)
        for (int j=0; j<dim; j++)
            x[i*dim+j] = resultantForce_[j]/patchArea_;

    return true;
}

// returns the value of the objective function
template <class GridView>
bool PressureAverager<GridView>::
eval_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number& obj_value)
{
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

    // 
    for (int i=0; i<nodalWeights_->size(); i++)
        for (int j=0; j<dim; j++)
            obj_value -= 2 * x[n-dim + j] * x[i*dim+j] * (*nodalWeights_)[i];
  
    // += c^2 * \int 1 ds
    for (int i=0; i<dim; i++)
        obj_value += patchArea_ * (x[n-dim + i] * x[n-dim + i]);

  return true;
}

// return the gradient of the objective function grad_{x} f(x)
template <class GridView>
bool PressureAverager<GridView>::
eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number* grad_f)
{
    //std::cout << "### eval_grad_f ###" << std::endl;

    // \nabla J = A(x,.) - b(x)
    for (int i=0; i<n; i++)
        grad_f[i] = 0;

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

//     for (int i=0; i<n; i++) {
//         std::cout << "x = " <<  x[i] << std::endl;
//         std::cout << "grad = " <<  grad_f[i] << std::endl;
//     }

  return true;
}

// return the value of the constraints: g(x)
template <class GridView>
bool PressureAverager<GridView>::
eval_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Index m, Ipopt::Number* g)
{
    for (int i=0; i<m; i++) {

        // init
        g[i] = 0;

        const RowType& jacobianRow = (*constraintJacobian_)[i];

        for (typename RowType::ConstIterator cIt = jacobianRow.begin(); cIt!=jacobianRow.end(); ++cIt)
            if ( std::abs((*cIt)[0][0]) > jacobianCutoff_ ) {
                //printf("adding %g times %g\n", (*cIt)[0][0], x[cIt.index()]);
                g[i] += (*cIt)[0][0] * x[cIt.index()];
            }

    }

//     for (int i=0; i<m; i++)
//         std::cout << "g[" << i << "]: " << g[i] << std::endl;
        
  return true;
}

// return the structure or values of the jacobian
template <class GridView>
bool PressureAverager<GridView>::
eval_jac_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
           Ipopt::Index m, Ipopt::Index nele_jac, Ipopt::Index* iRow, Ipopt::Index *jCol,
           Ipopt::Number* values)
{
    int idx = 0;

    if (values==NULL) {

        for (int i=0; i<m; i++) {
            
            const RowType& jacobianRow = (*constraintJacobian_)[i];
            
            for (typename RowType::ConstIterator cIt = jacobianRow.begin(); cIt!=jacobianRow.end(); ++cIt) {
                if ( std::abs((*cIt)[0][0]) > jacobianCutoff_ ) {
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
                if ( std::abs((*cIt)[0][0]) > jacobianCutoff_ )
                    values[idx++] = (*cIt)[0][0];
            
        }



    }

 return true;
}

//return the structure or values of the hessian
template <class GridView>
bool PressureAverager<GridView>::
eval_h(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
       Ipopt::Number obj_factor, Ipopt::Index m, const Ipopt::Number* lambda,
       bool new_lambda, Ipopt::Index nele_hess, Ipopt::Index* iRow,
       Ipopt::Index* jCol, Ipopt::Number* values)
{
    // We are using a quasi-Hessian approximation
    return false;
}

template <class GridView>
void PressureAverager<GridView>::
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

}

template <class GridView>
void weakToStrongBoundaryStress(const BoundaryPatchBase<GridView>& boundary,
                               const Dune::BlockVector<Dune::FieldVector<double, GridView::dimension> >& weakBoundaryStress,
                               Dune::BlockVector<Dune::FieldVector<double, GridView::dimension> >& strongBoundaryStress)
{
    static const int dim = GridView::dimension;
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,dim,dim> > MatrixType;
    typedef Dune::BlockVector<Dune::FieldVector<double,dim> >    VectorType;
    
    // turn residual (== weak form of the Neumann forces) to the strong form
    MatrixType surfaceMassMatrix;
    assembleSurfaceMassMatrix(boundary, surfaceMassMatrix);
        
    std::vector<int> globalToLocal;
    boundary.makeGlobalToLocal(globalToLocal);
    VectorType localWeakBoundaryStress(surfaceMassMatrix.N());
    assert(globalToLocal.size()==weakBoundaryStress.size());
    for (int i=0; i<globalToLocal.size(); i++)
        if (globalToLocal[i] >= 0)
            localWeakBoundaryStress[globalToLocal[i]] = weakBoundaryStress[i];
        
    VectorType localStrongBoundaryStress(surfaceMassMatrix.N());
    localStrongBoundaryStress = 0;
        
    // solve with a cg solver
    Dune::MatrixAdapter<MatrixType,VectorType,VectorType> op(surfaceMassMatrix);
    Dune::SeqILU0<MatrixType,VectorType,VectorType> ilu0(surfaceMassMatrix,1.0);
    Dune::CGSolver<VectorType> cg(op,ilu0,1E-4,100,0);
    Dune::InverseOperatorResult statistics;
    cg.apply(localStrongBoundaryStress, localWeakBoundaryStress, statistics);
    
    VectorType neumannForces(weakBoundaryStress.size());
    neumannForces = 0;
    for (int i=0; i<globalToLocal.size(); i++)
        if (globalToLocal[i] >= 0)
            strongBoundaryStress[i] = localStrongBoundaryStress[globalToLocal[i]];
    
}

/** \param center Compute total torque around this point
 */
template <class GridView>
void computeTotalForceAndTorque(const BoundaryPatchBase<GridView>& interface,
                                const Dune::BlockVector<Dune::FieldVector<double, GridView::dimension> >& boundaryStress,
                                const Dune::FieldVector<double,3>& center,
                                Dune::FieldVector<double,3>& totalForce, Dune::FieldVector<double,3>& totalTorque)
{
    const int dim = GridView::dimension;

    // ///////////////////////////////////////////
    //   Initialize output configuration
    // ///////////////////////////////////////////
    totalForce = 0;
    totalTorque = 0;
    
    ////////////////////////////////////////////////////////////////
    //  Interpret the Neumann value coefficients as a P1 function
    ////////////////////////////////////////////////////////////////
    typedef P1NodalBasis<GridView,double> P1Basis;
    P1Basis p1Basis(interface.gridView());
    BasisGridFunction<P1Basis, Dune::BlockVector<Dune::FieldVector<double, GridView::dimension> > > neumannFunction(p1Basis, boundaryStress);
    
    // ///////////////////////////////////////////
    //   Loop and integrate over the interface
    // ///////////////////////////////////////////

    typename BoundaryPatchBase<GridView>::iterator it    = interface.begin();
    typename BoundaryPatchBase<GridView>::iterator endIt = interface.end();

    for (; it!=endIt; ++it) {

        const typename BoundaryPatchBase<GridView>::iterator::Intersection::Geometry& segmentGeometry = it->geometry();

        // Get quadrature rule
        const Dune::QuadratureRule<double, dim-1>& quad = Dune::QuadratureRules<double, dim-1>::rule(segmentGeometry.type(), dim-1);

        /* Loop over all integration points */
        for (size_t ip=0; ip<quad.size(); ip++) {
                
            // Local position of the quadrature point
            const Dune::FieldVector<double,dim> quadPos = it->geometryInInside().global(quad[ip].position());
            const Dune::FieldVector<double,dim> worldPos = it->geometry().global(quad[ip].position());
            
            const double integrationElement = segmentGeometry.integrationElement(quad[ip].position());

            Dune::FieldVector<double,dim> value;
            neumannFunction.evaluateLocal(*it->inside(), quadPos, value);
            
            totalForce.axpy(quad[ip].weight() * integrationElement, value);
            totalTorque.axpy(quad[ip].weight() * integrationElement, crossProduct(worldPos-center,value));
            
        }

    }

    
}

// Given a resultant force and torque (from a rod problem), this method computes the corresponding
// Neumann data for a 3d elasticity problem.
template <class GridView>
void computeAveragePressure(const typename RigidBodyMotion<GridView::dimension>::TangentVector& resultantForceTorque,
                            const BoundaryPatchBase<GridView>& interface,
                            const Dune::FieldVector<double,GridView::dimension>& centerOfTorque,
                            Dune::BlockVector<Dune::FieldVector<double, GridView::dimension> >& pressure)
{
    const GridView& gridView                    = interface.gridView();
    const typename GridView::IndexSet& indexSet = gridView.indexSet();
    const int dim                               = GridView::dimension;
    typedef typename GridView::ctype ctype;
    typedef double field_type;

    Dune::PQkLocalFiniteElementCache<double,double, dim-1, 1> finiteElementCache;

    typedef typename GridView::template Codim<dim>::Iterator VertexIterator;

    // Create the matrix of constraints
    Dune::BCRSMatrix<Dune::FieldMatrix<field_type,1,1> > constraints(2*dim, dim*interface.numVertices(),
                                                                     Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> >::random);

    for (int i=0; i<dim; i++) {
        constraints.setrowsize(i,     interface.numVertices());
        constraints.setrowsize(i+dim, dim*interface.numVertices());
    }

    constraints.endrowsizes();

    for (int i=0; i<dim; i++)
        for (int j=0; j<interface.numVertices(); j++)
            constraints.addindex(i, dim*j+i);

    for (int i=0; i<dim; i++)
        for (int j=0; j<dim*interface.numVertices(); j++)
            constraints.addindex(i+dim, j);

    constraints.endindices();

    constraints = 0;

    // Create the surface mass matrix
    Dune::BCRSMatrix<Dune::FieldMatrix<field_type,1,1> > massMatrix;
    assembleSurfaceMassMatrix<GridView, 1>(interface, massMatrix);

    // Make global-to-local array
    std::vector<int> globalToLocal;
    interface.makeGlobalToLocal(globalToLocal);

    // Make array of nodal weights
    Dune::BlockVector<Dune::FieldVector<double,1> > nodalWeights(interface.numVertices());
    nodalWeights = 0;

    typename BoundaryPatchBase<GridView>::iterator it    = interface.begin();
    typename BoundaryPatchBase<GridView>::iterator endIt = interface.end();

    for (; it!=endIt; ++it) {

        // Get shape functions
        const typename Dune::PQkLocalFiniteElementCache<double,double, dim-1, 1>::FiniteElementType& 
            localFiniteElement = finiteElementCache.get(it->type());

            const Dune::GenericReferenceElement<double,dim>& refElement = Dune::GenericReferenceElements<double, dim>::general(it->inside()->type());

            // four rows because a face may have no more than four vertices
            Dune::FieldVector<double,4> mu(0);
            Dune::FieldVector<double,3> mu_tilde[4][3];
            
            for (int i=0; i<4; i++)
                for (int j=0; j<3; j++)
                    mu_tilde[i][j] = 0;

            for (int i=0; i<it->geometry().corners(); i++) {
                
                const Dune::QuadratureRule<double, dim-1>& quad 
                    = Dune::QuadratureRules<double, dim-1>::rule(it->type(), dim-1);
                
                for (size_t qp=0; qp<quad.size(); qp++) {
                    
                    // Local position of the quadrature point
                    const Dune::FieldVector<double,dim-1>& quadPos = quad[qp].position();
                    
                    const double integrationElement = it->geometry().integrationElement(quadPos);
                    
                    std::vector<Dune::FieldVector<double,1> > shapeFunctionValues;
                    localFiniteElement.localBasis().evaluateFunction(quadPos, shapeFunctionValues);

                    // \mu_i = \int_t \varphi_i \ds
                    mu[i] += quad[qp].weight() * integrationElement * shapeFunctionValues[i];
                    
                    // \tilde{\mu}_i^j = \int_t \varphi_i \times (x - x_0) \ds
                    Dune::FieldVector<double,dim> worldPos = it->geometry().global(quadPos);

                    for (int j=0; j<dim; j++) {

                        // Vector-valued basis function
                        Dune::FieldVector<double,dim> phi_i(0);
                        phi_i[j] = shapeFunctionValues[i];
                        
                        mu_tilde[i][j].axpy(quad[qp].weight() * integrationElement,
                                            crossProduct(Dune::FieldVector<double,dim>(worldPos-centerOfTorque), phi_i));

                    }
                    
                }
                
            }

            // Set up matrix
            for (int i=0; i<localFiniteElement.localCoefficients().size(); i++) {
                
                int faceIdxi = refElement.subEntity(it->indexInInside(), 1, i, dim);
                int subIndex = globalToLocal[indexSet.subIndex(*it->inside(), faceIdxi, dim)];

                nodalWeights[subIndex] += mu[i];
                for (int j=0; j<dim; j++)
                    constraints[j][subIndex*dim+j] += mu[i];

                for (int j=0; j<3; j++)
                    for (int k=0; k<3; k++)
                        constraints[dim+k][dim*subIndex+j] += mu_tilde[i][j][k];

            }

    }

    //printmatrix(std::cout, constraints, "jacobian", "--");
    //printmatrix(std::cout, massMatrix, "mass", "--");

    // /////////////////////////////////////////////////////////////////////////////////////
    //   Set up and start the interior-point solver
    // /////////////////////////////////////////////////////////////////////////////////////

    Dune::FieldVector<double,dim> resultantForce, resultantTorque;
    for (int i=0; i<dim; i++) {
        resultantForce[i]  = resultantForceTorque[i];
        resultantTorque[i] = resultantForceTorque[dim+i];
    }
    
    // Create a new instance of IpoptApplication
    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = new Ipopt::IpoptApplication();
    
    // Change some options
    app->Options()->SetNumericValue("tol", 1e-8);
    app->Options()->SetIntegerValue("max_iter", 1000);
    app->Options()->SetStringValue("mu_strategy", "adaptive");
    app->Options()->SetStringValue("output_file", "ipopt.out");
    app->Options()->SetStringValue("hessian_approximation", "limited-memory");
    app->Options()->SetStringValue("jac_c_constant", "yes");
    app->Options()->SetIntegerValue("print_level", 0);

    // Intialize the IpoptApplication and process the options
    Ipopt::ApplicationReturnStatus status;
    status = app->Initialize();
    if (status != Ipopt::Solve_Succeeded)
        DUNE_THROW(SolverError, "Error during IPOpt initialization!");
    
    // Ask Ipopt to solve the problem
    Dune::BlockVector<Dune::FieldVector<double,dim> > localPressure;
    Ipopt::SmartPtr<Ipopt::TNLP> defectSolverSmart = new PressureAverager<GridView>(&interface,
                                                                                    &localPressure,
                                                                                    resultantForce,
                                                                                    resultantTorque,
                                                                                    &massMatrix,
                                                                                    &nodalWeights,
                                                                                    &constraints);
    status = app->OptimizeTNLP(defectSolverSmart);
    
    if (status != Ipopt::Solve_Succeeded
        && status != Ipopt::Solved_To_Acceptable_Level) {
        //DUNE_THROW(SolverError, "Solving the defect problem failed!");
        std::cout << "IPOpt returned error code " << status << "!" << std::endl;
    }

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

    Dune::FieldVector<double,3> outputForce(0), outputTorque(0);

    computeTotalForceAndTorque(interface, pressure, centerOfTorque, outputForce, outputTorque);

    outputForce  -= resultantForce;
    outputTorque -= resultantTorque;
    assert( outputForce.infinity_norm() < 1e-6 );
    assert( outputTorque.infinity_norm() < 1e-6 );
//     std::cout << "Output force:  " << outputForce << std::endl;
//     std::cout << "Output torque: " << outputTorque << "      " << resultantTorque[0]/outputTorque[0] << std::endl;

}

template <class GridView>
void computeAverageInterface(const BoundaryPatchBase<GridView>& interface,
                             const Dune::BlockVector<Dune::FieldVector<double,GridView::dimension> > deformation,
                             RigidBodyMotion<3>& average)
{
    using namespace Dune;

    typedef typename GridView::template Codim<0>::Iterator ElementIterator;
    typedef typename GridView::template Codim<0>::Entity EntityType;
    typedef typename EntityType::LevelIntersectionIterator NeighborIterator;

    const GridView& gridView = interface.gridView();
    const typename GridView::IndexSet& indexSet = gridView.indexSet();
    const int dim        = GridView::dimension;

    Dune::PQkLocalFiniteElementCache<double,double, dim, 1> finiteElementCache;

    // ///////////////////////////////////////////
    //   Initialize output configuration
    // ///////////////////////////////////////////
    average.r = 0;
    
    double interfaceArea = 0;
    FieldMatrix<double,dim,dim> deformationGradient(0);

    // ///////////////////////////////////////////
    //   Loop and integrate over the interface
    // ///////////////////////////////////////////

    typename BoundaryPatchBase<GridView>::iterator it    = interface.begin();
    typename BoundaryPatchBase<GridView>::iterator endIt = interface.end();

    for (; it!=endIt; ++it) {

        const typename NeighborIterator::Intersection::Geometry& segmentGeometry = it->geometry();

            // Get quadrature rule
            const QuadratureRule<double, dim-1>& quad = QuadratureRules<double, dim-1>::rule(segmentGeometry.type(), dim-1);

            // Get set of shape functions on this segment
            const typename Dune::PQkLocalFiniteElementCache<double,double, dim, 1>::FiniteElementType& 
                localFiniteElement = finiteElementCache.get(it->inside()->type());

            /* Loop over all integration points */
            for (int ip=0; ip<quad.size(); ip++) {
                
                // Local position of the quadrature point
                const FieldVector<double,dim> quadPos = it->geometryInInside().global(quad[ip].position());
                
                const double integrationElement = segmentGeometry.integrationElement(quad[ip].position());

                std::vector<Dune::FieldVector<double,1> > values;
                localFiniteElement.localBasis().evaluateFunction(quadPos,values);

                // Evaluate base functions
                FieldVector<double,dim> posAtQuadPoint(0);

                for(int i=0; i<it->inside()->geometry().corners(); i++) {

                    int idx = indexSet.subIndex(*it->inside(), i, dim);

                    // Deformation at the quadrature point 
                    posAtQuadPoint.axpy(values[i], deformation[idx]);
                }

                const FieldMatrix<double,dim,dim>& inv = it->inside()->geometry().jacobianInverseTransposed(quadPos);
                
                /**********************************************/
                /* compute gradients of the shape functions   */
                /**********************************************/
                std::vector<FieldMatrix<double, 1, dim> > shapeGrads;
                localFiniteElement.localBasis().evaluateJacobian(quadPos,shapeGrads);
                
                for (int dof=0; dof<it->inside()->geometry().corners(); dof++) {
                    
                    FieldVector<double,dim> tmp = shapeGrads[dof][0];
                    
                    // multiply with jacobian inverse 
                    shapeGrads[dof] = 0;
                    inv.umv(tmp, shapeGrads[dof][0]);
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
                        
                        for (int k=0; k<it->inside()->geometry().corners(); k++)
                            F[i][j] += deformation[indexSet.subIndex(*it->inside(), k, dim)][i]*shapeGrads[k][0][j];
                        
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

    // returns a decomposition U W VT, where U is returned in the first argument
    svdcmp<double,dim,dim>(deformationGradient, W, V);

    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
            VT[i][j] = V[j][i];

    deformationGradient.rightmultiply(VT);
    //std::cout << deformationGradient << std::endl;

    // deformationGradient now contains the orthogonal part of the polar decomposition
    assert( std::abs(1-deformationGradient.determinant()) < 1e-3);

    average.q.set(deformationGradient);
}

/** \brief Set a Dirichlet value that corresponds to a given rigid body motion
 */
template <class GridView>
void setRotation(const BoundaryPatchBase<GridView>& dirichletBoundary,
                 Dune::BlockVector<Dune::FieldVector<double,GridView::dimension> >& deformation,
                 const RigidBodyMotion<3>& referenceInterface,
                 const RigidBodyMotion<3>& lambda)
{
    const typename GridView::IndexSet& indexSet = dirichletBoundary.gridView().indexSet();
    const int dim        = GridView::dimension;
    const int dimworld   = GridView::dimensionworld;

    // Get the relative rotation, first as a quaternion...
    Rotation<3,double> relativeRotation;
    relativeRotation = referenceInterface.q.inverse();
    relativeRotation = lambda.q.mult(relativeRotation);

    // ... then as a matrix
    Dune::FieldMatrix<double,3,3> rotation;
    relativeRotation.matrix(rotation);

    // ///////////////////////////////////////////
    //   Loop over all vertices
    // ///////////////////////////////////////////

    typename BoundaryPatchBase<GridView>::iterator it    = dirichletBoundary.begin();
    typename BoundaryPatchBase<GridView>::iterator endIt = dirichletBoundary.end();

    for (; it!=endIt; ++it) {
        
        int indexInInside = it->indexInInside();
        int nCorners  = Dune::GenericReferenceElements<double,dim>::general(it->inside()->type()).size(indexInInside, 1, dim);

        for (int i=0; i<nCorners; i++) {
            int cornerIdx = Dune::GenericReferenceElements<double,dim>::general(it->inside()->type()).subEntity(it->indexInInside(), 1, i, dim);
            int globalIdx = indexSet.subIndex(*it->inside(), cornerIdx, dim);

            // Get vertex position
            const Dune::FieldVector<double,dimworld> pos = it->inside()->geometry().corner(cornerIdx);
            
            // Action of the rigid body motion
            Dune::FieldVector<double,dimworld> rpos;
            rotation.mv(pos-referenceInterface.r, rpos);
            rpos += lambda.r;
            
            // We compute _displacements_, not positions
            rpos -= pos;
            
            deformation[globalIdx] = rpos;
            
        }
        
    }
    
}

#endif
