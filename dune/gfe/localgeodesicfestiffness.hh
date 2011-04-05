#ifndef LOCAL_GEODESIC_FE_STIFFNESS_HH
#define LOCAL_GEODESIC_FE_STIFFNESS_HH

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>

#include "rigidbodymotion.hh"
#include "unitvector.hh"
#include "realtuple.hh"

// Forward declaration
template<class GridView, class TargetSpace>
class LocalGeodesicFEStiffness;

template<class GridView, class TargetSpace, bool globalIsometricCoordinates>
class LocalGeodesicFEStiffnessImp
{

    typedef typename GridView::template Codim<0>::Entity Entity;
    
    public:

    template <int N>
    static void infinitesimalVariation(RealTuple<N>& c, double eps, int i)
    {
        Dune::FieldVector<double,N> v(0);
        v[i] = eps;
        c = RealTuple<N>::exp(c,v).globalCoordinates();
    }

    /** \brief For the fd approximations 
    */
    template <int N>
    static void infinitesimalVariation(UnitVector<N>& c, double eps, int i)
    {
        Dune::FieldVector<double,N> result = c.globalCoordinates();
        result[i] += eps;
        c = result;
    }

    static void assembleEmbeddedGradient(const Entity& element,
                                  const std::vector<TargetSpace>& localSolution,
                                  std::vector<typename TargetSpace::EmbeddedTangentVector>& localGradient,
                                  const LocalGeodesicFEStiffness<GridView,TargetSpace>* energyObject)
    {

        const int embeddedBlocksize = TargetSpace::EmbeddedTangentVector::size;

        // ///////////////////////////////////////////////////////////
        //   Compute gradient by finite-difference approximation
        // ///////////////////////////////////////////////////////////

        double eps = 1e-6;

        localGradient.resize(localSolution.size());

        std::vector<TargetSpace> forwardSolution = localSolution;
        std::vector<TargetSpace> backwardSolution = localSolution;

        for (size_t i=0; i<localSolution.size(); i++) {
        
            for (int j=0; j<embeddedBlocksize; j++) {
            
                // The return value does not have unit norm.  But assigning it to a UnitVector object
                // will normalize it.  This amounts to an extension of the energy functional 
                // to a neighborhood around S^n
                forwardSolution[i]  = localSolution[i];
                backwardSolution[i] = localSolution[i];
                LocalGeodesicFEStiffnessImp<GridView,TargetSpace,globalIsometricCoordinates>::infinitesimalVariation(forwardSolution[i],  eps, j);
                LocalGeodesicFEStiffnessImp<GridView,TargetSpace,globalIsometricCoordinates>::infinitesimalVariation(backwardSolution[i], -eps, j);

                localGradient[i][j] = (energyObject->energy(element,forwardSolution) - energyObject->energy(element,backwardSolution)) / (2*eps);

                forwardSolution[i]  = localSolution[i];
                backwardSolution[i] = localSolution[i];
            }

            // Project gradient in embedding space onto the tangent space
            localGradient[i] = localSolution[i].projectOntoTangentSpace(localGradient[i]);
        
        }

    }

    public:
    static void assembleGradient(const Entity& element,
                          const std::vector<TargetSpace>& localSolution,
                          std::vector<typename TargetSpace::TangentVector>& localGradient,
                                 const LocalGeodesicFEStiffness<GridView,TargetSpace>* energyObject)
    {
        std::vector<typename TargetSpace::EmbeddedTangentVector> embeddedLocalGradient;

        // first compute the gradient in embedded coordinates
        assembleEmbeddedGradient(element, localSolution, embeddedLocalGradient, energyObject);

        // transform to coordinates on the tangent space
        localGradient.resize(embeddedLocalGradient.size());

        for (size_t i=0; i<localGradient.size(); i++)
            localSolution[i].orthonormalFrame().mv(embeddedLocalGradient[i], localGradient[i]);

    }
};


template<class GridView, class TargetSpace>
class LocalGeodesicFEStiffnessImp<GridView,TargetSpace,false>
{
    typedef typename GridView::template Codim<0>::Entity Entity;

public:
    
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

    /** \brief For the fd approximations 
    */
    static void infinitesimalVariation(RigidBodyMotion<2>& c, double eps, int i)
    {
        if (i<2)
            c.r[i] += eps;
        else
            c.q = c.q.mult(Rotation<2,double>::exp(Dune::FieldVector<double,1>(eps)));
    }

    static void infinitesimalVariation(Rotation<3,double>& c, double eps, int i)
    {
        c = c.mult(Rotation<3,double>::exp((i==0)*eps, 
                                           (i==1)*eps, 
                                           (i==2)*eps));
    }

    static void infinitesimalVariation(Rotation<2,double>& c, double eps, int i)
    {
        Dune::FieldVector<double,1> v(eps);
        c = Rotation<2,double>::exp(c,v);
    }
    
    static void assembleGradient(const Entity& element,
                          const std::vector<TargetSpace>& localSolution,
                          std::vector<typename TargetSpace::TangentVector>& localGradient,
                                 const LocalGeodesicFEStiffness<GridView,TargetSpace>* energyObject)
    {
        // ///////////////////////////////////////////////////////////
        //   Compute gradient by finite-difference approximation
        // ///////////////////////////////////////////////////////////

        double eps = 1e-6;

        localGradient.resize(localSolution.size());

        std::vector<TargetSpace> forwardSolution = localSolution;
        std::vector<TargetSpace> backwardSolution = localSolution;

        for (size_t i=0; i<localSolution.size(); i++) {
        
            for (int j=0; j<TargetSpace::TangentVector::size; j++) {
            
                infinitesimalVariation(forwardSolution[i],   eps, j);
                infinitesimalVariation(backwardSolution[i], -eps, j);
            
                localGradient[i][j] = (energyObject->energy(element,forwardSolution) - energyObject->energy(element,backwardSolution))
                    / (2*eps);
            
                forwardSolution[i]  = localSolution[i];
                backwardSolution[i] = localSolution[i];
            }
        
        }

    }

};


template<class GridView, class TargetSpace>
class LocalGeodesicFEStiffness 
{

    // grid types
    typedef typename GridView::Grid::ctype DT;
    typedef typename TargetSpace::ctype RT;
    typedef typename GridView::template Codim<0>::Entity Entity;
    
    // some other sizes
    enum {gridDim=GridView::dimension};

public:
    
    //! Each block is x, y, theta in 2d, T (R^3 \times SO(3)) in 3d
    enum { blocksize = TargetSpace::TangentVector::size };

    static const bool globalIsometricCoordinates = TargetSpace::globalIsometricCoordinates;

    /** \brief Assemble the local stiffness matrix at the current position

    This default implementation used finite-difference approximations to compute the second derivatives
    */
    virtual void assembleHessian(const Entity& e,
                  const std::vector<TargetSpace>& localSolution);
   
    virtual RT energy (const Entity& e,
                       const std::vector<TargetSpace>& localSolution) const = 0;

    /** \brief Assemble the element gradient of the energy functional 

    The default implementation in this class uses a finite difference approximation */
    virtual void assembleGradient(const Entity& element,
                                  const std::vector<TargetSpace>& solution,
                                  std::vector<Dune::FieldVector<double,blocksize> >& gradient) const;

    // assembled data
    Dune::Matrix<Dune::FieldMatrix<double,blocksize,blocksize> > A_;
    
};

template <class GridView, class TargetSpace>
void LocalGeodesicFEStiffness<GridView, TargetSpace>::
assembleGradient(const Entity& element,
                 const std::vector<TargetSpace>& localSolution,
                 std::vector<Dune::FieldVector<double,blocksize> >& localGradient) const
{
    LocalGeodesicFEStiffnessImp<GridView,TargetSpace,globalIsometricCoordinates>::assembleGradient(element, localSolution, localGradient, this);
}


template <class GridView, class TargetSpace>
void LocalGeodesicFEStiffness<GridView,TargetSpace>::
assembleHessian(const Entity& element,
         const std::vector<TargetSpace>& localSolution)
{
    // 1 degree of freedom per element vertex
    int nDofs = element.template count<gridDim>();

    // Clear assemble data
    A_.setSize(nDofs,nDofs);

    A_ = 0;

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
    for (size_t i=0; i<A_.N(); i++) {

        ColumnIterator cIt    = A_[i].begin();
        ColumnIterator cEndIt = A_[i].end();

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

                        LocalGeodesicFEStiffnessImp<GridView,TargetSpace,globalIsometricCoordinates>::infinitesimalVariation(forwardSolution[i], eps, j);
                        LocalGeodesicFEStiffnessImp<GridView,TargetSpace,globalIsometricCoordinates>::infinitesimalVariation(backwardSolution[i], -eps, j);

                        double forwardEnergy  = energy(element, forwardSolution);
                        
                        double solutionEnergy = energy(element, localSolution);
                        
                        double backwardEnergy = energy(element, backwardSolution);

                        // Second derivative
                        (*cIt)[j][k] = (forwardEnergy - 2*solutionEnergy + backwardEnergy) / (eps*eps);
                        
                        forwardSolution[i]  = localSolution[i];
                        backwardSolution[i] = localSolution[i];

                    } else {

                        // Off-diagonal entries
                        LocalGeodesicFEStiffnessImp<GridView,TargetSpace,globalIsometricCoordinates>::infinitesimalVariation(forwardForwardSolution[i],             eps, j);
                        LocalGeodesicFEStiffnessImp<GridView,TargetSpace,globalIsometricCoordinates>::infinitesimalVariation(forwardForwardSolution[cIt.index()],   eps, k);
                        LocalGeodesicFEStiffnessImp<GridView,TargetSpace,globalIsometricCoordinates>::infinitesimalVariation(forwardBackwardSolution[i],            eps, j);
                        LocalGeodesicFEStiffnessImp<GridView,TargetSpace,globalIsometricCoordinates>::infinitesimalVariation(forwardBackwardSolution[cIt.index()], -eps, k);
                        LocalGeodesicFEStiffnessImp<GridView,TargetSpace,globalIsometricCoordinates>::infinitesimalVariation(backwardForwardSolution[i],           -eps, j);
                        LocalGeodesicFEStiffnessImp<GridView,TargetSpace,globalIsometricCoordinates>::infinitesimalVariation(backwardForwardSolution[cIt.index()],  eps, k);
                        LocalGeodesicFEStiffnessImp<GridView,TargetSpace,globalIsometricCoordinates>::infinitesimalVariation(backwardBackwardSolution[i],          -eps, j);
                        LocalGeodesicFEStiffnessImp<GridView,TargetSpace,globalIsometricCoordinates>::infinitesimalVariation(backwardBackwardSolution[cIt.index()],-eps, k);

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
    for (size_t i=0; i<A_.N(); i++) {

        ColumnIterator cIt    = A_[i].begin();
        ColumnIterator cEndIt = A_[i].end();

        for (; cIt!=cEndIt; ++cIt) {

            if (cIt.index()>i)
                continue;


            if (cIt.index()==i) {

                for (int j=1; j<blocksize; j++)
                    for (int k=0; k<j; k++)
                        (*cIt)[j][k] = (*cIt)[k][j];

            } else {

                const Dune::FieldMatrix<double,blocksize,blocksize>& other = A_[cIt.index()][i];

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
{
    typedef UnitVector<dim> TargetSpace;

    // grid types
    typedef typename GridView::Grid::ctype DT;
    typedef typename TargetSpace::ctype RT;
    typedef typename GridView::template Codim<0>::Entity Entity;
    
    // some other sizes
    enum {gridDim=GridView::dimension};

public:
    
    //! Each block is x, y, theta in 2d, T (R^3 \times SO(3)) in 3d
    enum { blocksize = TargetSpace::TangentVector::size };

    //! Each block is x, y, theta in 2d, T (R^3 \times SO(3)) in 3d
    enum { embeddedBlocksize = TargetSpace::EmbeddedTangentVector::size };

    static const bool globalIsometricCoordinates = TargetSpace::globalIsometricCoordinates;

    /** \brief Assemble the local stiffness matrix at the current position

    This default implementation used finite-difference approximations to compute the second derivatives
    */
    virtual void assembleHessian(const Entity& e,
                  const std::vector<TargetSpace>& localSolution);
    
    virtual RT energy (const Entity& e,
                       const std::vector<TargetSpace>& localSolution) const = 0;

#if 0
    /** \brief Assemble the element gradient of the energy functional 

    The default implementation in this class uses a finite difference approximation */
    virtual void assembleEmbeddedGradient(const Entity& element,
                                          const std::vector<TargetSpace>& solution,
                                          std::vector<typename TargetSpace::EmbeddedTangentVector>& gradient) const;
#endif
                                          
    /** \brief Assemble the element gradient of the energy functional 

    The default implementation in this class uses a finite difference approximation */
    virtual void assembleGradient(const Entity& element,
                                  const std::vector<TargetSpace>& solution,
                                  std::vector<typename TargetSpace::TangentVector>& gradient) const;
    
    void embeddedGradientOfEmbeddedGradient(const Entity& element,
                                            const std::vector<TargetSpace>& localSolution,
                                            int component,
                                            std::vector<Dune::FieldMatrix<double,embeddedBlocksize,embeddedBlocksize> >& gradient) const {
        
        double eps = 1e-6;
        
        gradient.resize(localSolution.size());
        std::fill(gradient.begin(), gradient.end(), Dune::FieldMatrix<double,embeddedBlocksize,embeddedBlocksize>(0));
        
        std::vector<TargetSpace> forwardSolution = localSolution;
        std::vector<TargetSpace> backwardSolution = localSolution;
        
        
        for (int j=0; j<embeddedBlocksize; j++) {
            
            // The return value does not have unit norm.  But assigning it to a UnitVector object
            // will normalize it.  This amounts to an extension of the energy functional 
            // to a neighborhood around S^n
            forwardSolution[component]  = localSolution[component];
            backwardSolution[component] = localSolution[component];
            LocalGeodesicFEStiffnessImp<GridView,TargetSpace,globalIsometricCoordinates>::infinitesimalVariation(forwardSolution[component],  eps, j);
            LocalGeodesicFEStiffnessImp<GridView,TargetSpace,globalIsometricCoordinates>::infinitesimalVariation(backwardSolution[component], -eps, j);
            
            std::vector<Dune::FieldVector<double,embeddedBlocksize> > forwardGradient;
            std::vector<Dune::FieldVector<double,embeddedBlocksize> > backwardGradient;
            LocalGeodesicFEStiffnessImp<GridView,TargetSpace,globalIsometricCoordinates>::assembleEmbeddedGradient(element, forwardSolution, forwardGradient,this);
            LocalGeodesicFEStiffnessImp<GridView,TargetSpace,globalIsometricCoordinates>::assembleEmbeddedGradient(element, backwardSolution, backwardGradient,this);

            for (int k=0; k<localSolution.size(); k++)
                for (int l=0; l<embeddedBlocksize; l++)
                    gradient[k][j][l] = (forwardGradient[k][l] - backwardGradient[k][l]) / (2*eps);

            forwardSolution[component]  = localSolution[component];
            backwardSolution[component] = localSolution[component];

            // Project each column vector onto the tangent space

        // Project gradient in embedding space onto the tangent space
            for (size_t i=0; i<localSolution.size(); i++)
                for (int j=0; j<embeddedBlocksize; j++) {
                    Dune::FieldVector<double,embeddedBlocksize> tmp;
                    for (int k=0; k<embeddedBlocksize; k++)
                        tmp[k] = gradient[i][k][j];
                    tmp = localSolution[i].projectOntoTangentSpace(tmp);
                    for (int k=0; k<embeddedBlocksize; k++)
                        gradient[i][k][j] = tmp[k];
                }

        }

    }

    // assembled data
    Dune::Matrix<Dune::FieldMatrix<double,blocksize,blocksize> > A_;
    
};


template <class GridView, int dim>
void LocalGeodesicFEStiffness<GridView, UnitVector<dim> >::
assembleGradient(const Entity& element,
                 const std::vector<TargetSpace>& localSolution,
                 std::vector<typename TargetSpace::TangentVector>& localGradient) const
{
    LocalGeodesicFEStiffnessImp<GridView,TargetSpace,globalIsometricCoordinates>::assembleGradient(element, localSolution, localGradient,this);
}

// ///////////////////////////////////////////////////////////
//   Compute gradient by finite-difference approximation
// ///////////////////////////////////////////////////////////
template <class GridType, int dim>
void LocalGeodesicFEStiffness<GridType,UnitVector<dim> >::
assembleHessian(const Entity& element,
         const std::vector<TargetSpace>& localSolution)
{
    // 1 degree of freedom per element vertex
    int nDofs = element.template count<gridDim>();

    // Clear assemble data
    A_.setSize(nDofs, nDofs);

    A_ = 0;

#if 0
#warning Dummy Hessian implementation
    for (int i=0; i<nDofs; i++)
        for (int j=0; j<blocksize; j++)
            A_[i][i][j][j] = 1;
#else

    // first compute the Hessian in the embedding space
    Dune::Matrix<Dune::FieldMatrix<double,embeddedBlocksize,embeddedBlocksize> > embeddedHessian(nDofs,nDofs);

    std::vector<Dune::FieldMatrix<double,embeddedBlocksize,embeddedBlocksize> > embeddedGradient;

    for (size_t i=0; i<localSolution.size(); i++) {

        embeddedGradientOfEmbeddedGradient(element,localSolution, i, embeddedGradient);

        for (int j=0; j<localSolution.size(); j++)
            embeddedHessian[i][j] = embeddedGradient[j];

    }

    // transform to local tangent space bases
    std::vector<Dune::FieldMatrix<double,blocksize,embeddedBlocksize> > orthonormalFrames(nDofs);
    std::vector<Dune::FieldMatrix<double,embeddedBlocksize,blocksize> > orthonormalFramesTransposed(nDofs);

    for (size_t i=0; i<nDofs; i++) {
        orthonormalFrames[i] = localSolution[i].orthonormalFrame();

        for (int j=0; j<embeddedBlocksize; j++)
            for (int k=0; k<blocksize; k++)
                orthonormalFramesTransposed[i][j][k] = orthonormalFrames[i][k][j];

    }

    for (size_t i=0; i<nDofs; i++)
        for (size_t j=0; j<nDofs; j++) {

            Dune::FieldMatrix<double,blocksize,embeddedBlocksize> tmp;
            Dune::FMatrixHelp::multMatrix(orthonormalFrames[i],embeddedHessian[i][j],tmp);
            A_[i][j] = tmp.rightmultiplyany(orthonormalFramesTransposed[j]);
            

        }
    
#endif
}

#endif

