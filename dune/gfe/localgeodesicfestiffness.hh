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

template<class GridView, class TargetSpace>
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
                LocalGeodesicFEStiffnessImp<GridView,TargetSpace>::infinitesimalVariation(forwardSolution[i],  eps, j);
                LocalGeodesicFEStiffnessImp<GridView,TargetSpace>::infinitesimalVariation(backwardSolution[i], -eps, j);

                localGradient[i][j] = (energyObject->energy(element,forwardSolution) - energyObject->energy(element,backwardSolution)) / (2*eps);

                forwardSolution[i]  = localSolution[i];
                backwardSolution[i] = localSolution[i];
            }

            // Project gradient in embedding space onto the tangent space
            localGradient[i] = localSolution[i].projectOntoTangentSpace(localGradient[i]);
        
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

    //! Each block is x, y, theta in 2d, T (R^3 \times SO(3)) in 3d
    enum { embeddedBlocksize = TargetSpace::EmbeddedTangentVector::size };

    /** \brief Assemble the local stiffness matrix at the current position

    This default implementation used finite-difference approximations to compute the second derivatives

    The formula for the Riemannian Hessian has been taken from Absil, Mahony, Sepulchre:
    'Optimization algorithms on matrix manifolds', page 107.  There it says that
    \f[
        \langle Hess f(x)[\xi], \eta \rangle
            = \frac 12 \frac{d^2}{dt^2} \Big(f(\exp_x(t(\xi + \eta))) - f(\exp_x(t\xi)) - f(\exp_x(t\eta))\Big)\Big|_{t=0}.
    \f]
    We compute that using a finite difference approximation.

    */
    virtual void assembleHessian(const Entity& e,
                  const std::vector<TargetSpace>& localSolution);
    
    virtual RT energy (const Entity& e,
                       const std::vector<TargetSpace>& localSolution) const = 0;

    /** \brief Assemble the element gradient of the energy functional using a finite-difference approximation

    This is mainly for debugging purposes.
    */
    virtual void assembleEmbeddedFDGradient(const Entity& element,
                                            const std::vector<TargetSpace>& solution,
                                            std::vector<typename TargetSpace::EmbeddedTangentVector>& gradient) const;
                                          
    /** \brief Assemble the element gradient of the energy functional 

    The default implementation in this class uses a finite difference approximation */
    virtual void assembleGradient(const Entity& element,
                                  const std::vector<TargetSpace>& solution,
                                  std::vector<typename TargetSpace::TangentVector>& gradient) const;
    
    // assembled data
    Dune::Matrix<Dune::FieldMatrix<double,blocksize,blocksize> > A_;
    
};


template <class GridView, class TargetSpace>
void LocalGeodesicFEStiffness<GridView, TargetSpace>::
assembleGradient(const Entity& element,
                 const std::vector<TargetSpace>& localSolution,
                 std::vector<typename TargetSpace::TangentVector>& localGradient) const
{
    std::vector<typename TargetSpace::EmbeddedTangentVector> embeddedLocalGradient;

    // first compute the gradient in embedded coordinates
    LocalGeodesicFEStiffnessImp<GridView,TargetSpace>::assembleEmbeddedGradient(element, localSolution, embeddedLocalGradient, this);

    // transform to coordinates on the tangent space
    localGradient.resize(embeddedLocalGradient.size());

    for (size_t i=0; i<localGradient.size(); i++)
        localSolution[i].orthonormalFrame().mv(embeddedLocalGradient[i], localGradient[i]);
}

template <class GridView, class TargetSpace>
void LocalGeodesicFEStiffness<GridView, TargetSpace>::
assembleEmbeddedFDGradient(const Entity& element,
                 const std::vector<TargetSpace>& localSolution,
                 std::vector<typename TargetSpace::EmbeddedTangentVector>& localGradient) const
{
    LocalGeodesicFEStiffnessImp<GridView,TargetSpace>::assembleEmbeddedGradient(element, localSolution, localGradient, this);
}


// ///////////////////////////////////////////////////////////
//   Compute gradient by finite-difference approximation
// ///////////////////////////////////////////////////////////
template <class GridType, class TargetSpace>
void LocalGeodesicFEStiffness<GridType, TargetSpace>::
assembleHessian(const Entity& element,
         const std::vector<TargetSpace>& localSolution)
{
    // 1 degree of freedom per element vertex
    size_t nDofs = element.template count<gridDim>();

    // Clear assemble data
    A_.setSize(nDofs, nDofs);

    A_ = 0;


    
    
    const double eps = 1e-4;
        
    std::vector<Dune::FieldMatrix<double,blocksize,embeddedBlocksize> > B(localSolution.size());
    for (size_t i=0; i<B.size(); i++)
        B[i] = localSolution[i].orthonormalFrame();

    // Precompute negative energy at the current configuration
    // (negative because that is how we need it as part of the 2nd-order fd formula)
    double centerValue   = -energy(element, localSolution);
        
    // Precompute energy infinitesimal corrections in the directions of the local basis vectors
    std::vector<Dune::array<double,blocksize> > forwardEnergy(nDofs);
    std::vector<Dune::array<double,blocksize> > backwardEnergy(nDofs);
    for (size_t i=0; i<localSolution.size(); i++) {
        for (size_t i2=0; i2<blocksize; i2++) {
            Dune::FieldVector<double,embeddedBlocksize> epsXi = B[i][i2];
            epsXi *= eps;
            Dune::FieldVector<double,embeddedBlocksize> minusEpsXi = epsXi;
            minusEpsXi  *= -1;
            
            std::vector<TargetSpace> forwardSolution  = localSolution;
            std::vector<TargetSpace> backwardSolution = localSolution;
            
            forwardSolution[i]  = TargetSpace::exp(localSolution[i],epsXi);
            backwardSolution[i] = TargetSpace::exp(localSolution[i],minusEpsXi);
    
            forwardEnergy[i][i2]  = energy(element, forwardSolution);
            backwardEnergy[i][i2] = energy(element, backwardSolution);
    
        }
        
    }
    
    // finite-difference approximation
    for (size_t i=0; i<localSolution.size(); i++) {
        for (size_t i2=0; i2<blocksize; i2++) {
            for (size_t j=0; j<localSolution.size(); j++) {
                for (size_t j2=0; j2<blocksize; j2++) {

                    Dune::FieldVector<double,embeddedBlocksize> epsXi  = B[i][i2];    epsXi *= eps;
                    Dune::FieldVector<double,embeddedBlocksize> epsEta = B[j][j2];   epsEta *= eps;
            
                    Dune::FieldVector<double,embeddedBlocksize> minusEpsXi  = epsXi;   minusEpsXi  *= -1;
                    Dune::FieldVector<double,embeddedBlocksize> minusEpsEta = epsEta;  minusEpsEta *= -1;
                        
                    std::vector<TargetSpace> forwardSolutionXiEta  = localSolution;
                    std::vector<TargetSpace> backwardSolutionXiEta  = localSolution;
            
                    if (i==j)
                        forwardSolutionXiEta[i] = TargetSpace::exp(localSolution[i],epsXi+epsEta);
                    else {
                        forwardSolutionXiEta[i] = TargetSpace::exp(localSolution[i],epsXi);
                        forwardSolutionXiEta[j] = TargetSpace::exp(localSolution[j],epsEta);
                    }
                        
                    if (i==j)
                        backwardSolutionXiEta[i] = TargetSpace::exp(localSolution[i],minusEpsXi+minusEpsEta);
                    else {
                        backwardSolutionXiEta[i] = TargetSpace::exp(localSolution[i],minusEpsXi);
                        backwardSolutionXiEta[j] = TargetSpace::exp(localSolution[j],minusEpsEta);
                    }

                    double forwardValue  = energy(element, forwardSolutionXiEta) - forwardEnergy[i][i2] - forwardEnergy[j][j2];
                    double backwardValue = energy(element, backwardSolutionXiEta) - backwardEnergy[i][i2] - backwardEnergy[j][j2];
            
                    A_[i][j][i2][j2] = 0.5 * (forwardValue - 2*centerValue + backwardValue) / (eps*eps);
                        
                }
            }
        }
    }
        
}

#endif

