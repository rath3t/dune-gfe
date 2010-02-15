#ifndef LOCAL_GEODESIC_FE_STIFFNESS_HH
#define LOCAL_GEODESIC_FE_STIFFNESS_HH

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>

#include "localstiffness.hh"
#include "rigidbodymotion.hh"
#include "unitvector.hh"

template<class GridView, class TargetSpace>
class LocalGeodesicFEStiffness 
    : public Dune::LocalStiffness<GridView,double,TargetSpace::TangentVector::size>
{

    // grid types
    typedef typename GridView::Grid::ctype DT;
    typedef typename TargetSpace::ctype RT;
    typedef typename GridView::template Codim<0>::Entity Entity;
    
    // some other sizes
    enum {gridDim=GridView::dimension};

    /** \brief For the fd approximations 
    */
    static void infinitesimalVariation(RigidBodyMotion<3>& c, double eps, int i)
    {
        if (i<3)
            c.r[i] += eps;
        else
            c.q = c.q.mult(Rotation<3,double>::exp((i==3)*eps, 
                                                   (i==4)*eps, 
                                                   (i==5)*eps));
    }

    static void infinitesimalVariation(Rotation<3,double>& c, double eps, int i)
    {
        c = c.mult(Rotation<3,double>::exp((i==0)*eps, 
                                           (i==1)*eps, 
                                           (i==2)*eps));
    }

public:
    
    //! Each block is x, y, theta in 2d, T (R^3 \times SO(3)) in 3d
    enum { blocksize = TargetSpace::TangentVector::size };

    // define the number of components of your system, this is used outside
    // to allocate the correct size of (dense) blocks with a FieldMatrix
    enum {m=blocksize};

    // types for matrics, vectors and boundary conditions
    typedef Dune::FieldMatrix<RT,m,m> MBlockType; // one entry in the stiffness matrix
    typedef Dune::FieldVector<RT,m> VBlockType;   // one entry in the global vectors

    /** \brief Assemble the local stiffness matrix at the current position

    This default implementation used finite-difference approximations to compute the second derivatives
    */
    virtual void assemble(const Entity& e,
                  const std::vector<TargetSpace>& localSolution);
    
    /** \brief assemble local stiffness matrix for given element and order
    */
    void assemble (const Entity& e, 
                   const Dune::BlockVector<Dune::FieldVector<double, blocksize> >& localSolution,
                   int k=1)
    {
        DUNE_THROW(Dune::NotImplemented, "!");
    }

    /** \todo Remove this once this methods is not in base class LocalStiffness anymore */
    void assemble (const Entity& e, int k=1)
    {
        DUNE_THROW(Dune::NotImplemented, "!");
    }

    void assembleBoundaryCondition (const Entity& e, int k=1)
    {
        DUNE_THROW(Dune::NotImplemented, "!");
    }

    
    virtual RT energy (const Entity& e,
                       const std::vector<TargetSpace>& localSolution) const = 0;

    /** \brief Assemble the element gradient of the energy functional 

    The default implementation in this class uses a finite difference approximation */
    virtual void assembleGradient(const Entity& element,
                                  const std::vector<TargetSpace>& solution,
                                  std::vector<Dune::FieldVector<double,blocksize> >& gradient) const;
    
};

template <class GridView, class TargetSpace>
void LocalGeodesicFEStiffness<GridView, TargetSpace>::
assembleGradient(const Entity& element,
                 const std::vector<TargetSpace>& localSolution,
                 std::vector<Dune::FieldVector<double,blocksize> >& localGradient) const
{
    // ///////////////////////////////////////////////////////////
    //   Compute gradient by finite-difference approximation
    // ///////////////////////////////////////////////////////////

    double eps = 1e-6;

    localGradient.resize(localSolution.size());

    std::vector<TargetSpace> forwardSolution = localSolution;
    std::vector<TargetSpace> backwardSolution = localSolution;

    for (size_t i=0; i<localSolution.size(); i++) {
        
        for (int j=0; j<blocksize; j++) {
            
            infinitesimalVariation(forwardSolution[i],   eps, j);
            infinitesimalVariation(backwardSolution[i], -eps, j);
            
            localGradient[i][j] = (energy(element,forwardSolution) - energy(element,backwardSolution))
                / (2*eps);
            
            forwardSolution[i]  = localSolution[i];
            backwardSolution[i] = localSolution[i];
        }
        
    }

}


template <class GridType, class TargetSpace>
void LocalGeodesicFEStiffness<GridType,TargetSpace>::
assemble(const Entity& element,
         const std::vector<TargetSpace>& localSolution)
{
    // 1 degree of freedom per element vertex
    int nDofs = element.template count<gridDim>();

    // Clear assemble data
    this->setcurrentsize(nDofs);

    this->A = 0;

    double eps = 1e-4;

    typedef typename Dune::Matrix<Dune::FieldMatrix<double,blocksize,blocksize> >::row_type::iterator ColumnIterator;

    // ///////////////////////////////////////////////////////////
    //   Compute gradient by finite-difference approximation
    // ///////////////////////////////////////////////////////////
    std::vector<TargetSpace> forwardSolution  = localSolution;
    std::vector<TargetSpace> backwardSolution = localSolution;

    std::vector<TargetSpace> forwardForwardSolution   = localSolution;
    std::vector<TargetSpace> forwardBackwardSolution  = localSolution;
    std::vector<TargetSpace> backwardForwardSolution  = localSolution;
    std::vector<TargetSpace> backwardBackwardSolution = localSolution;

    // ///////////////////////////////////////////////////////////////
    //   Loop over all blocks of the element matrix
    // ///////////////////////////////////////////////////////////////
    for (int i=0; i<this->A.N(); i++) {

        ColumnIterator cIt    = this->A[i].begin();
        ColumnIterator cEndIt = this->A[i].end();

        for (; cIt!=cEndIt; ++cIt) {

            // compute only the upper right triangular matrix
            if (cIt.index() < i)
                continue;

            // ////////////////////////////////////////////////////////////////////////////
            //   Compute a finite-difference approximation of this hessian matrix block
            // ////////////////////////////////////////////////////////////////////////////

            for (int j=0; j<blocksize; j++) {

                for (int k=0; k<blocksize; k++) {

                    // compute only the upper right triangular matrix
                    if (i==cIt.index() && k<j)
                        continue;

                    // Diagonal entries
                    if (i==cIt.index() && j==k) {

                        infinitesimalVariation(forwardSolution[i], eps, j);
                        infinitesimalVariation(backwardSolution[i], -eps, j);

                        double forwardEnergy  = energy(element, forwardSolution);
                        
                        double solutionEnergy = energy(element, localSolution);
                        
                        double backwardEnergy = energy(element, backwardSolution);

                        // Second derivative
                        (*cIt)[j][k] = (forwardEnergy - 2*solutionEnergy + backwardEnergy) / (eps*eps);
                        
                        forwardSolution[i]  = localSolution[i];
                        backwardSolution[i] = localSolution[i];

                    } else {

                        // Off-diagonal entries
                        infinitesimalVariation(forwardForwardSolution[i],             eps, j);
                        infinitesimalVariation(forwardForwardSolution[cIt.index()],   eps, k);
                        infinitesimalVariation(forwardBackwardSolution[i],            eps, j);
                        infinitesimalVariation(forwardBackwardSolution[cIt.index()], -eps, k);
                        infinitesimalVariation(backwardForwardSolution[i],           -eps, j);
                        infinitesimalVariation(backwardForwardSolution[cIt.index()],  eps, k);
                        infinitesimalVariation(backwardBackwardSolution[i],          -eps, j);
                        infinitesimalVariation(backwardBackwardSolution[cIt.index()],-eps, k);

                        double forwardForwardEnergy = energy(element, forwardForwardSolution);
                        
                        double forwardBackwardEnergy = energy(element, forwardBackwardSolution);
                        
                        double backwardForwardEnergy = energy(element, backwardForwardSolution);
                        
                        double backwardBackwardEnergy = energy(element, backwardBackwardSolution);
                        
                        (*cIt)[j][k] = (forwardForwardEnergy + backwardBackwardEnergy
                                        - forwardBackwardEnergy - backwardForwardEnergy) / (4*eps*eps);
                        
                        forwardForwardSolution[i]             = localSolution[i];
                        forwardForwardSolution[cIt.index()]   = localSolution[cIt.index()];
                        forwardBackwardSolution[i]            = localSolution[i];
                        forwardBackwardSolution[cIt.index()]  = localSolution[cIt.index()];
                        backwardForwardSolution[i]            = localSolution[i];
                        backwardForwardSolution[cIt.index()]  = localSolution[cIt.index()];
                        backwardBackwardSolution[i]           = localSolution[i];
                        backwardBackwardSolution[cIt.index()] = localSolution[cIt.index()];
                        
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
    for (int i=0; i<this->A.N(); i++) {

        ColumnIterator cIt    = this->A[i].begin();
        ColumnIterator cEndIt = this->A[i].end();

        for (; cIt!=cEndIt; ++cIt) {

            if (cIt.index()>i)
                continue;


            if (cIt.index()==i) {

                for (int j=1; j<blocksize; j++)
                    for (int k=0; k<j; k++)
                        (*cIt)[j][k] = (*cIt)[k][j];

            } else {

                const Dune::FieldMatrix<double,blocksize,blocksize>& other = this->A[cIt.index()][i];

                for (int j=0; j<blocksize; j++)
                    for (int k=0; k<blocksize; k++)
                        (*cIt)[j][k] = other[k][j];


            }


        }

    }

}



/** \brief Specialization for unit vectors */
template<class GridView, int dim>
class LocalGeodesicFEStiffness <GridView,UnitVector<dim> >
    : public Dune::LocalStiffness<GridView,double,UnitVector<dim>::TangentVector::size>
{
    typedef UnitVector<dim> TargetSpace;

    // grid types
    typedef typename GridView::Grid::ctype DT;
    typedef typename TargetSpace::ctype RT;
    typedef typename GridView::template Codim<0>::Entity Entity;
    
    // some other sizes
    enum {gridDim=GridView::dimension};

    /** \brief For the fd approximations 
    */
    static Dune::FieldVector<double,dim> infinitesimalVariation(UnitVector<dim>& c, double eps, int i)
    {
        Dune::FieldVector<double,dim> result = c.globalCoordinates();
        result[i] += eps;
        return result;
    }

public:
    
    //! Each block is x, y, theta in 2d, T (R^3 \times SO(3)) in 3d
    enum { blocksize = TargetSpace::TangentVector::size };

    // define the number of components of your system, this is used outside
    // to allocate the correct size of (dense) blocks with a FieldMatrix
    enum {m=blocksize};

    // types for matrics, vectors and boundary conditions
    typedef Dune::FieldMatrix<RT,m,m> MBlockType; // one entry in the stiffness matrix
    typedef Dune::FieldVector<RT,m> VBlockType;   // one entry in the global vectors

    /** \brief Assemble the local stiffness matrix at the current position

    This default implementation used finite-difference approximations to compute the second derivatives
    */
    virtual void assemble(const Entity& e,
                  const std::vector<TargetSpace>& localSolution);
    
    /** \brief assemble local stiffness matrix for given element and order
    */
    void assemble (const Entity& e, 
                   const Dune::BlockVector<Dune::FieldVector<double, blocksize> >& localSolution,
                   int k=1)
    {
        DUNE_THROW(Dune::NotImplemented, "!");
    }

    /** \todo Remove this once this methods is not in base class LocalStiffness anymore */
    void assemble (const Entity& e, int k=1)
    {
        DUNE_THROW(Dune::NotImplemented, "!");
    }

    void assembleBoundaryCondition (const Entity& e, int k=1)
    {
        DUNE_THROW(Dune::NotImplemented, "!");
    }

    
    virtual RT energy (const Entity& e,
                       const std::vector<TargetSpace>& localSolution) const = 0;

    /** \brief Assemble the element gradient of the energy functional 

    The default implementation in this class uses a finite difference approximation */
    virtual void assembleGradient(const Entity& element,
                                  const std::vector<TargetSpace>& solution,
                                  std::vector<Dune::FieldVector<double,blocksize> >& gradient) const;
    
};

template <class GridView, int dim>
void LocalGeodesicFEStiffness<GridView, UnitVector<dim> >::
assembleGradient(const Entity& element,
                 const std::vector<TargetSpace>& localSolution,
                 std::vector<Dune::FieldVector<double,blocksize> >& localGradient) const
{

    // ///////////////////////////////////////////////////////////
    //   Compute gradient by finite-difference approximation
    // ///////////////////////////////////////////////////////////

    double eps = 1e-6;

    localGradient.resize(localSolution.size());

    std::vector<TargetSpace> forwardSolution = localSolution;
    std::vector<TargetSpace> backwardSolution = localSolution;

    for (size_t i=0; i<localSolution.size(); i++) {
        
        for (int j=0; j<blocksize; j++) {
            
            infinitesimalVariation(forwardSolution[i],   eps, j);
            infinitesimalVariation(backwardSolution[i], -eps, j);
            
            localGradient[i][j] = (energy(element,forwardSolution) - energy(element,backwardSolution))
                / (2*eps);
            
            forwardSolution[i]  = localSolution[i];
            backwardSolution[i] = localSolution[i];
        }

        // Project gradient in embedding space onto the tangent space
        localGradient[i] = localSolution[i].projectOntoTangentSpace(localGradient[i]);
        
    }

}


template <class GridType, int dim>
void LocalGeodesicFEStiffness<GridType,UnitVector<dim> >::
assemble(const Entity& element,
         const std::vector<TargetSpace>& localSolution)
{
#warning Dummy Hessian implementation

    // 1 degree of freedom per element vertex
    int nDofs = element.template count<gridDim>();

    // Clear assemble data
    this->setcurrentsize(nDofs);

    this->A = 0;

    for (int i=0; i<nDofs; i++)
        for (int j=0; j<blocksize; j++)
            this->A[i][i][j][j] = 1;
#if 0
    double eps = 1e-4;

    typedef typename Dune::Matrix<Dune::FieldMatrix<double,blocksize,blocksize> >::row_type::iterator ColumnIterator;

    // ///////////////////////////////////////////////////////////
    //   Compute gradient by finite-difference approximation
    // ///////////////////////////////////////////////////////////
    std::vector<TargetSpace> forwardSolution  = localSolution;
    std::vector<TargetSpace> backwardSolution = localSolution;

    std::vector<TargetSpace> forwardForwardSolution   = localSolution;
    std::vector<TargetSpace> forwardBackwardSolution  = localSolution;
    std::vector<TargetSpace> backwardForwardSolution  = localSolution;
    std::vector<TargetSpace> backwardBackwardSolution = localSolution;

    // ///////////////////////////////////////////////////////////////
    //   Loop over all blocks of the element matrix
    // ///////////////////////////////////////////////////////////////
    for (int i=0; i<this->A.N(); i++) {

        ColumnIterator cIt    = this->A[i].begin();
        ColumnIterator cEndIt = this->A[i].end();

        for (; cIt!=cEndIt; ++cIt) {

            // compute only the upper right triangular matrix
            if (cIt.index() < i)
                continue;

            // ////////////////////////////////////////////////////////////////////////////
            //   Compute a finite-difference approximation of this hessian matrix block
            // ////////////////////////////////////////////////////////////////////////////

            for (int j=0; j<blocksize; j++) {

                for (int k=0; k<blocksize; k++) {

                    // compute only the upper right triangular matrix
                    if (i==cIt.index() && k<j)
                        continue;

                    // Diagonal entries
                    if (i==cIt.index() && j==k) {

                        infinitesimalVariation(forwardSolution[i], eps, j);
                        infinitesimalVariation(backwardSolution[i], -eps, j);

                        double forwardEnergy  = energy(element, forwardSolution);
                        
                        double solutionEnergy = energy(element, localSolution);
                        
                        double backwardEnergy = energy(element, backwardSolution);

                        // Second derivative
                        (*cIt)[j][k] = (forwardEnergy - 2*solutionEnergy + backwardEnergy) / (eps*eps);
                        
                        forwardSolution[i]  = localSolution[i];
                        backwardSolution[i] = localSolution[i];

                    } else {

                        // Off-diagonal entries
                        infinitesimalVariation(forwardForwardSolution[i],             eps, j);
                        infinitesimalVariation(forwardForwardSolution[cIt.index()],   eps, k);
                        infinitesimalVariation(forwardBackwardSolution[i],            eps, j);
                        infinitesimalVariation(forwardBackwardSolution[cIt.index()], -eps, k);
                        infinitesimalVariation(backwardForwardSolution[i],           -eps, j);
                        infinitesimalVariation(backwardForwardSolution[cIt.index()],  eps, k);
                        infinitesimalVariation(backwardBackwardSolution[i],          -eps, j);
                        infinitesimalVariation(backwardBackwardSolution[cIt.index()],-eps, k);

                        double forwardForwardEnergy = energy(element, forwardForwardSolution);
                        
                        double forwardBackwardEnergy = energy(element, forwardBackwardSolution);
                        
                        double backwardForwardEnergy = energy(element, backwardForwardSolution);
                        
                        double backwardBackwardEnergy = energy(element, backwardBackwardSolution);
                        
                        (*cIt)[j][k] = (forwardForwardEnergy + backwardBackwardEnergy
                                        - forwardBackwardEnergy - backwardForwardEnergy) / (4*eps*eps);
                        
                        forwardForwardSolution[i]             = localSolution[i];
                        forwardForwardSolution[cIt.index()]   = localSolution[cIt.index()];
                        forwardBackwardSolution[i]            = localSolution[i];
                        forwardBackwardSolution[cIt.index()]  = localSolution[cIt.index()];
                        backwardForwardSolution[i]            = localSolution[i];
                        backwardForwardSolution[cIt.index()]  = localSolution[cIt.index()];
                        backwardBackwardSolution[i]           = localSolution[i];
                        backwardBackwardSolution[cIt.index()] = localSolution[cIt.index()];
                        
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
    for (int i=0; i<this->A.N(); i++) {

        ColumnIterator cIt    = this->A[i].begin();
        ColumnIterator cEndIt = this->A[i].end();

        for (; cIt!=cEndIt; ++cIt) {

            if (cIt.index()>i)
                continue;


            if (cIt.index()==i) {

                for (int j=1; j<blocksize; j++)
                    for (int k=0; k<j; k++)
                        (*cIt)[j][k] = (*cIt)[k][j];

            } else {

                const Dune::FieldMatrix<double,blocksize,blocksize>& other = this->A[cIt.index()][i];

                for (int j=0; j<blocksize; j++)
                    for (int k=0; k<blocksize; k++)
                        (*cIt)[j][k] = other[k][j];


            }


        }

    }
#endif
}


#endif

