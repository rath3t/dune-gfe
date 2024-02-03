#ifndef DUNE_GFE_LOCAL_GEODESIC_FE_STIFFNESS_HH
#define DUNE_GFE_LOCAL_GEODESIC_FE_STIFFNESS_HH

#include <dune/common/fmatrix.hh>
#include <dune/istl/matrix.hh>

#include <dune/gfe/assemblers/localfirstordermodel.hh>

namespace Dune::GFE
{
  namespace Impl
  {
    template<class TargetSpace>
    class LocalStiffnessTypes
    {
      // Number type
      typedef typename TargetSpace::ctype RT;

      //! Dimension of a tangent space
      constexpr static auto blocksize = TargetSpace::TangentVector::dimension;

    public:

      // Type of the local Hessian
      using Hessian = Matrix<FieldMatrix<RT, blocksize, blocksize> >;
    };
  }
}

template<class Basis, class TargetSpace>
class LocalGeodesicFEStiffness
  : public Dune::GFE::LocalFirstOrderModel<Basis,TargetSpace>
{
  // Number type
  typedef typename TargetSpace::ctype RT;

public:

  //! Dimension of a tangent space
  constexpr static int blocksize = TargetSpace::TangentVector::dimension;

  //! Dimension of the embedding space
  constexpr static int embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension;

  /** \brief Assemble the local gradient and stiffness matrix at the current position

   */
  virtual void assembleGradientAndHessian(const typename Basis::LocalView& localView,
                                          const std::vector<TargetSpace>& localSolution,
                                          std::vector<typename TargetSpace::TangentVector>& localGradient,
                                          typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::Hessian& localHessian) const = 0;

  /** \brief Compute the energy at the current configuration */
  virtual RT energy (const typename Basis::LocalView& localView,
                     const std::vector<TargetSpace>& localSolution) const = 0;

  /** \brief Assemble the element gradient of the energy functional
   */
  virtual void assembleGradient(const typename Basis::LocalView& localView,
                                const std::vector<TargetSpace>& solution,
                                std::vector<typename TargetSpace::TangentVector>& gradient) const = 0;
};

#endif
