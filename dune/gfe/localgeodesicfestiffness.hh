#ifndef DUNE_GFE_LOCAL_GEODESIC_FE_STIFFNESS_HH
#define DUNE_GFE_LOCAL_GEODESIC_FE_STIFFNESS_HH

#include <dune/common/fmatrix.hh>
#include <dune/istl/matrix.hh>

#include <dune/gfe/localfirstordermodel.hh>

template<class Basis, class TargetSpace>
class LocalGeodesicFEStiffness
: public Dune::GFE::LocalFirstOrderModel<Basis,TargetSpace>
{
    // grid types
    typedef typename Basis::GridView GridView;
    typedef typename GridView::ctype DT;
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
    virtual void assembleGradientAndHessian(const typename Basis::LocalView& localView,
                                 const std::vector<TargetSpace>& localSolution,
                                 std::vector<typename TargetSpace::TangentVector>& localGradient);

    /** \brief Compute the energy at the current configuration */
    virtual RT energy (const typename Basis::LocalView& localView,
                       const std::vector<TargetSpace>& localSolution) const = 0;

    /** \brief Assemble the element gradient of the energy functional

    The default implementation in this class uses a finite difference approximation */
    virtual void assembleGradient(const typename Basis::LocalView& localView,
                                  const std::vector<TargetSpace>& solution,
                                  std::vector<typename TargetSpace::TangentVector>& gradient) const;

    // assembled data
    Dune::Matrix<Dune::FieldMatrix<RT,blocksize,blocksize> > A_;

};


template <class Basis, class TargetSpace>
void LocalGeodesicFEStiffness<Basis, TargetSpace>::
assembleGradient(const typename Basis::LocalView& localView,
                 const std::vector<TargetSpace>& localSolution,
                 std::vector<typename TargetSpace::TangentVector>& localGradient) const
{
  DUNE_THROW(Dune::NotImplemented, "!");
}


// ///////////////////////////////////////////////////////////
//   Compute gradient by finite-difference approximation
// ///////////////////////////////////////////////////////////
template <class Basis, class TargetSpace>
void LocalGeodesicFEStiffness<Basis, TargetSpace>::
assembleGradientAndHessian(const typename Basis::LocalView& localView,
                const std::vector<TargetSpace>& localSolution,
                std::vector<typename TargetSpace::TangentVector>& localGradient)
{
  DUNE_THROW(Dune::NotImplemented, "!");
}

#endif

