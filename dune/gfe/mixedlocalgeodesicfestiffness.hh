#ifndef DUNE_GFE_MIXEDLOCALGEODESICFESTIFFNESS_HH
#define DUNE_GFE_MIXEDLOCALGEODESICFESTIFFNESS_HH

#include "omp.h"

#include <dune/common/fmatrix.hh>
#include <dune/istl/matrix.hh>


template<class Basis, class DeformationTargetSpace, class OrientationTargetSpace>
class MixedLocalGeodesicFEStiffness
{
    // grid types
    typedef typename Basis::GridView GridView;
    typedef typename GridView::ctype DT;
    typedef typename DeformationTargetSpace::ctype RT;
    typedef typename GridView::template Codim<0>::Entity Entity;

    // some other sizes
    enum {gridDim=GridView::dimension};

public:

    //! Dimension of a tangent space
    enum { deformationBlocksize = DeformationTargetSpace::TangentVector::dimension };
    enum { orientationBlocksize = OrientationTargetSpace::TangentVector::dimension };

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
    virtual void assembleGradientAndHessian(const typename Basis::LocalView& localView,
                                            const std::vector<DeformationTargetSpace>& localDisplacementConfiguration,
                                            const std::vector<OrientationTargetSpace>& localOrientationConfiguration,
                                            std::vector<typename DeformationTargetSpace::TangentVector>& localDeformationGradient,
                                            std::vector<typename OrientationTargetSpace::TangentVector>& localOrientationGradient)
    {
      DUNE_THROW(Dune::NotImplemented, "!");
    }

    /** \brief Compute the energy at the current configuration */
    virtual RT energy (const typename Basis::LocalView& localView,
                       const std::vector<DeformationTargetSpace>& localDeformationConfiguration,
                       const std::vector<OrientationTargetSpace>& localOrientationConfiguration) const = 0;
#if 0
    /** \brief Assemble the element gradient of the energy functional

    The default implementation in this class uses a finite difference approximation */
    virtual void assembleGradient(const Entity& element,
                                  const LocalFiniteElement& localFiniteElement,
                                  const std::vector<TargetSpace>& solution,
                                  std::vector<typename TargetSpace::TangentVector>& gradient) const;
    {
      DUNE_THROW(Dune::NotImplemented, "!");
    }
#endif
    // assembled data
    Dune::Matrix<Dune::FieldMatrix<RT, deformationBlocksize, deformationBlocksize> > A00_;
    Dune::Matrix<Dune::FieldMatrix<RT, deformationBlocksize, orientationBlocksize> > A01_;
    Dune::Matrix<Dune::FieldMatrix<RT, orientationBlocksize, deformationBlocksize> > A10_;
    Dune::Matrix<Dune::FieldMatrix<RT, orientationBlocksize, orientationBlocksize> > A11_;

};

#endif

