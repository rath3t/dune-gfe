#ifndef DUNE_GFE_LOCAL_GEODESIC_FE_FD_STIFFNESS_HH
#define DUNE_GFE_LOCAL_GEODESIC_FE_FD_STIFFNESS_HH

#include <dune/common/fmatrix.hh>
#include <dune/istl/matrix.hh>

#include <dune/gfe/localgeodesicfestiffness.hh>

/** \brief Assembles energy gradient and Hessian with finite difference approximations
 */
template<class Basis, class TargetSpace, class field_type=double>
class LocalGeodesicFEFDStiffness
    : public LocalGeodesicFEStiffness<Basis,TargetSpace>
{
    // grid types
    typedef typename Basis::GridView GridView;
    typedef typename GridView::ctype DT;
    typedef typename TargetSpace::ctype RT;
    typedef typename GridView::template Codim<0>::Entity Entity;

    typedef typename TargetSpace::template rebind<field_type>::other ATargetSpace;

    // some other sizes
    enum {gridDim=GridView::dimension};

public:

    //! Dimension of a tangent space
    enum { blocksize = TargetSpace::TangentVector::dimension };

    //! Dimension of the embedding space
    enum { embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension };

    LocalGeodesicFEFDStiffness(const LocalGeodesicFEStiffness<Basis, ATargetSpace>* energy)
    : localEnergy_(energy)
    {}

    /** \brief Compute the energy at the current configuration */
    virtual RT energy (const typename Basis::LocalView& localView,
               const std::vector<TargetSpace>& localSolution) const
    {
      return localEnergy_->energy(localView,localSolution);
    }

    /** \brief Assemble the element gradient of the energy functional

       The default implementation in this class uses a finite difference approximation */
    virtual void assembleGradient(const typename Basis::LocalView& localView,
                                  const std::vector<TargetSpace>& solution,
                                  std::vector<typename TargetSpace::TangentVector>& gradient) const;

    /** \brief Assemble the local tangent matrix and gradient at the current position

      This implementation uses finite-difference approximations

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


    const LocalGeodesicFEStiffness<Basis, ATargetSpace>* localEnergy_;

};

template <class Basis, class TargetSpace, class field_type>
void LocalGeodesicFEFDStiffness<Basis, TargetSpace, field_type>::
assembleGradient(const typename Basis::LocalView& localView,
                 const std::vector<TargetSpace>& localSolution,
                 std::vector<typename TargetSpace::TangentVector>& localGradient) const
{

    // ///////////////////////////////////////////////////////////
    //   Compute gradient by finite-difference approximation
    // ///////////////////////////////////////////////////////////

    field_type eps = 1e-6;

    std::vector<ATargetSpace> localASolution(localSolution.size());
    std::vector<typename ATargetSpace::CoordinateType> aRaw(localSolution.size());
    for (size_t i=0; i<localSolution.size(); i++) {
      typename TargetSpace::CoordinateType raw = localSolution[i].globalCoordinates();
      for (size_t j=0; j<raw.size(); j++)
          aRaw[i][j] = raw[j];
      localASolution[i] = aRaw[i];  // may contain a projection onto M -- needs to be done in adouble
    }

    localGradient.resize(localSolution.size());

    std::vector<ATargetSpace> forwardSolution  = localASolution;
    std::vector<ATargetSpace> backwardSolution = localASolution;

    for (size_t i=0; i<localSolution.size(); i++) {

        // basis vectors of the tangent space of the i-th entry of localSolution
        const Dune::FieldMatrix<field_type,blocksize,embeddedBlocksize> B = localSolution[i].orthonormalFrame();

        for (int j=0; j<blocksize; j++) {

            typename ATargetSpace::EmbeddedTangentVector forwardCorrection = B[j];
            forwardCorrection *= eps;

            typename ATargetSpace::EmbeddedTangentVector backwardCorrection = B[j];
            backwardCorrection *= -eps;

            forwardSolution[i]  = ATargetSpace::exp(localASolution[i], forwardCorrection);
            backwardSolution[i] = ATargetSpace::exp(localASolution[i], backwardCorrection);

            field_type foo = (localEnergy_->energy(localView,forwardSolution) - localEnergy_->energy(localView, backwardSolution)) / (2*eps);
#ifdef MULTIPRECISION
            localGradient[i][j] = foo.template convert_to<double>();
#else
            localGradient[i][j] = foo;
#endif

        }

        forwardSolution[i]  = localASolution[i];
        backwardSolution[i] = localASolution[i];

    }

}


/////////////////////////////////////////////////////////////////////////////////
//   Compute gradient and Hessian together
//   To compute the Hessian we need to compute the gradient anyway, so we may
//   as well return it.  This saves assembly time.
/////////////////////////////////////////////////////////////////////////////////
template <class Basis, class TargetSpace, class field_type>
void LocalGeodesicFEFDStiffness<Basis, TargetSpace, field_type>::
assembleGradientAndHessian(const typename Basis::LocalView& localView,
                const std::vector<TargetSpace>& localSolution,
                std::vector<typename TargetSpace::TangentVector>& localGradient)
{
    // Number of degrees of freedom for this element
    size_t nDofs = localSolution.size();

    // Clear assemble data
    this->A_.setSize(nDofs, nDofs);

    this->A_ = 0;

#ifdef MULTIPRECISION
    const field_type eps = 1e-10;
#else
    const field_type eps = 1e-4;
#endif

    std::vector<ATargetSpace> localASolution(localSolution.size());
    std::vector<typename ATargetSpace::CoordinateType> aRaw(localSolution.size());
    for (size_t i=0; i<localSolution.size(); i++) {
      typename TargetSpace::CoordinateType raw = localSolution[i].globalCoordinates();
      for (size_t j=0; j<raw.size(); j++)
          aRaw[i][j] = raw[j];
      localASolution[i] = aRaw[i];
    }

    std::vector<Dune::FieldMatrix<double,blocksize,embeddedBlocksize> > B(localSolution.size());
    for (size_t i=0; i<B.size(); i++)
        B[i] = localSolution[i].orthonormalFrame();

    // Precompute negative energy at the current configuration
    // (negative because that is how we need it as part of the 2nd-order fd formula)
    field_type centerValue   = -localEnergy_->energy(localView, localASolution);

    // Precompute energy infinitesimal corrections in the directions of the local basis vectors
    std::vector<std::array<field_type,blocksize> > forwardEnergy(nDofs);
    std::vector<std::array<field_type,blocksize> > backwardEnergy(nDofs);

    //#pragma omp parallel for schedule (dynamic)
    for (size_t i=0; i<localSolution.size(); i++) {
        for (size_t i2=0; i2<blocksize; i2++) {
            typename ATargetSpace::EmbeddedTangentVector epsXi = B[i][i2];
            epsXi *= eps;
            typename ATargetSpace::EmbeddedTangentVector minusEpsXi = epsXi;
            minusEpsXi  *= -1;

            std::vector<ATargetSpace> forwardSolution  = localASolution;
            std::vector<ATargetSpace> backwardSolution = localASolution;

            forwardSolution[i]  = ATargetSpace::exp(localASolution[i],epsXi);
            backwardSolution[i] = ATargetSpace::exp(localASolution[i],minusEpsXi);

            forwardEnergy[i][i2]  = localEnergy_->energy(localView, forwardSolution);
            backwardEnergy[i][i2] = localEnergy_->energy(localView, backwardSolution);

        }

    }

    //////////////////////////////////////////////////////////////
    //   Compute gradient by finite-difference approximation
    //////////////////////////////////////////////////////////////

    localGradient.resize(localSolution.size());

    for (size_t i=0; i<localSolution.size(); i++)
        for (int j=0; j<blocksize; j++)
        {
          field_type foo = (forwardEnergy[i][j] - backwardEnergy[i][j]) / (2*eps);
#ifdef MULTIPRECISION
          localGradient[i][j] = foo.template convert_to<double>();
#else
          localGradient[i][j] = foo;
#endif
        }

    ///////////////////////////////////////////////////////////////////////////
    //   Compute Riemannian Hesse matrix by finite-difference approximation.
    //   We loop over the lower left triangular half of the matrix.
    //   The other half follows from symmetry.
    ///////////////////////////////////////////////////////////////////////////
    //#pragma omp parallel for schedule (dynamic)
    for (size_t i=0; i<localSolution.size(); i++) {
        for (size_t i2=0; i2<blocksize; i2++) {
            for (size_t j=0; j<=i; j++) {
                for (size_t j2=0; j2<((i==j) ? i2+1 : blocksize); j2++) {

                    std::vector<ATargetSpace> forwardSolutionXiEta   = localASolution;
                    std::vector<ATargetSpace> backwardSolutionXiEta  = localASolution;

                    typename ATargetSpace::EmbeddedTangentVector epsXi  = B[i][i2];    epsXi *= eps;
                    typename ATargetSpace::EmbeddedTangentVector epsEta = B[j][j2];   epsEta *= eps;

                    typename ATargetSpace::EmbeddedTangentVector minusEpsXi  = epsXi;   minusEpsXi  *= -1;
                    typename ATargetSpace::EmbeddedTangentVector minusEpsEta = epsEta;  minusEpsEta *= -1;

                    if (i==j)
                        forwardSolutionXiEta[i] = ATargetSpace::exp(localASolution[i],epsXi+epsEta);
                    else {
                        forwardSolutionXiEta[i] = ATargetSpace::exp(localASolution[i],epsXi);
                        forwardSolutionXiEta[j] = ATargetSpace::exp(localASolution[j],epsEta);
                    }

                    if (i==j)
                        backwardSolutionXiEta[i] = ATargetSpace::exp(localASolution[i],minusEpsXi+minusEpsEta);
                    else {
                        backwardSolutionXiEta[i] = ATargetSpace::exp(localASolution[i],minusEpsXi);
                        backwardSolutionXiEta[j] = ATargetSpace::exp(localASolution[j],minusEpsEta);
                    }

                    field_type forwardValue  = localEnergy_->energy(localView, forwardSolutionXiEta) - forwardEnergy[i][i2] - forwardEnergy[j][j2];
                    field_type backwardValue = localEnergy_->energy(localView, backwardSolutionXiEta) - backwardEnergy[i][i2] - backwardEnergy[j][j2];

                    field_type foo = 0.5 * (forwardValue - 2*centerValue + backwardValue) / (eps*eps);
#ifdef MULTIPRECISION
                    this->A_[i][j][i2][j2] = this->A_[j][i][j2][i2] = foo.template convert_to<double>();
#else
                    this->A_[i][j][i2][j2] = this->A_[j][i][j2][i2] = foo;
#endif
                }
            }
        }
    }
}

#endif

