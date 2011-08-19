#ifndef LOCAL_GEODESIC_FE_STIFFNESS_HH
#define LOCAL_GEODESIC_FE_STIFFNESS_HH

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>


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
    
    //! Dimension of a tangent space
    enum { blocksize = TargetSpace::TangentVector::dimension };

    //! Dimension of the embedding space
    enum { embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension };

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
                  const Dune::array<TargetSpace,gridDim+1>& localSolution);

    /** \brief Compute the energy at the current configuration */
    virtual RT energy (const Entity& e,
                       const Dune::array<TargetSpace,gridDim+1>& localSolution) const = 0;

    /** \brief Assemble the element gradient of the energy functional 

    The default implementation in this class uses a finite difference approximation */
    virtual void assembleGradient(const Entity& element,
                                  const Dune::array<TargetSpace,gridDim+1>& solution,
                                  std::vector<typename TargetSpace::TangentVector>& gradient) const;
    
    // assembled data
    Dune::Matrix<Dune::FieldMatrix<double,blocksize,blocksize> > A_;
    
};


template <class GridView, class TargetSpace>
void LocalGeodesicFEStiffness<GridView, TargetSpace>::
assembleGradient(const Entity& element,
                 const Dune::array<TargetSpace,gridDim+1>& localSolution,
                 std::vector<typename TargetSpace::TangentVector>& localGradient) const
{

    // ///////////////////////////////////////////////////////////
    //   Compute gradient by finite-difference approximation
    // ///////////////////////////////////////////////////////////

    double eps = 1e-6;

    localGradient.resize(localSolution.size());

    Dune::array<TargetSpace,gridDim+1> forwardSolution = localSolution;
    Dune::array<TargetSpace,gridDim+1> backwardSolution = localSolution;

    for (size_t i=0; i<localSolution.size(); i++) {
        
        // basis vectors of the tangent space of the i-th entry of localSolution
        const Dune::FieldMatrix<double,blocksize,embeddedBlocksize> B = localSolution[i].orthonormalFrame();

        for (int j=0; j<blocksize; j++) {
            
            typename TargetSpace::EmbeddedTangentVector forwardCorrection = B[j];
            forwardCorrection *= eps;
            
            typename TargetSpace::EmbeddedTangentVector backwardCorrection = B[j];
            backwardCorrection *= -eps;
            
            forwardSolution[i]  = TargetSpace::exp(localSolution[i], forwardCorrection);
            backwardSolution[i] = TargetSpace::exp(localSolution[i], backwardCorrection);

            localGradient[i][j] = (energy(element,forwardSolution) - energy(element,backwardSolution)) / (2*eps);

        }

        forwardSolution[i]  = localSolution[i];
        backwardSolution[i] = localSolution[i];
        
    }

}


// ///////////////////////////////////////////////////////////
//   Compute gradient by finite-difference approximation
// ///////////////////////////////////////////////////////////
template <class GridType, class TargetSpace>
void LocalGeodesicFEStiffness<GridType, TargetSpace>::
assembleHessian(const Entity& element,
         const Dune::array<TargetSpace,gridDim+1>& localSolution)
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
            
            Dune::array<TargetSpace,gridDim+1> forwardSolution  = localSolution;
            Dune::array<TargetSpace,gridDim+1> backwardSolution = localSolution;
            
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
                        
                    Dune::array<TargetSpace,gridDim+1> forwardSolutionXiEta  = localSolution;
                    Dune::array<TargetSpace,gridDim+1> backwardSolutionXiEta  = localSolution;
            
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

