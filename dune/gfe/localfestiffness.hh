#ifndef LOCAL_FE_STIFFNESS_HH
#define LOCAL_FE_STIFFNESS_HH

#include <dune/common/fmatrix.hh>
#include <dune/istl/matrix.hh>


template<class GridView, class LocalFiniteElement, class VectorType>
class LocalFEStiffness
{
    // grid types
    typedef typename GridView::Grid::ctype DT;
    typedef typename VectorType::value_type::field_type RT;
    typedef typename GridView::template Codim<0>::Entity Entity;

    // some other sizes
    enum {gridDim=GridView::dimension};

public:

    //! Dimension of a tangent space
    enum { blocksize = VectorType::value_type::dimension };

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
    virtual void assembleGradientAndHessian(const Entity& e,
                                 const LocalFiniteElement& localFiniteElement,
                                 const VectorType& localConfiguration,
                                 const std::vector<Dune::FieldVector<double,3> >& localPointLoads,
                                 VectorType& localGradient);

    /** \brief Compute the energy at the current configuration */
    virtual RT energy (const Entity& e,
                       const LocalFiniteElement& localFiniteElement,
                       const VectorType& localConfiguration,
                       const std::vector<Dune::FieldVector<double,3> >& localPointLoads) const = 0;

    // assembled data
    Dune::Matrix<Dune::FieldMatrix<RT,blocksize,blocksize> > A_;

};


// ///////////////////////////////////////////////////////////
//   Compute gradient by finite-difference approximation
// ///////////////////////////////////////////////////////////
template <class GridType, class LocalFiniteElement, class VectorType>
void LocalFEStiffness<GridType, LocalFiniteElement, VectorType>::
assembleGradientAndHessian(const Entity& element,
                const LocalFiniteElement& localFiniteElement,
                const VectorType& localConfiguration,
                const std::vector<Dune::FieldVector<double,3> >& localPointLoads,
                VectorType& localGradient)
{
  DUNE_THROW(Dune::NotImplemented, "!");
}

#endif

