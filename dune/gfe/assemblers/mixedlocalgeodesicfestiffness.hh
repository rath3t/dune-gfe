#ifndef DUNE_GFE_MIXEDLOCALGEODESICFESTIFFNESS_HH
#define DUNE_GFE_MIXEDLOCALGEODESICFESTIFFNESS_HH

#include <dune/common/fmatrix.hh>

#include <dune/istl/matrix.hh>
#include <dune/istl/multitypeblockmatrix.hh>

#include <dune/gfe/assemblers/localfirstordermodel.hh>

namespace Dune::GFE
{
  namespace Impl
  {
    template<class TargetSpace>
    class MixedLocalStiffnessTypes
      : public LocalFirstOrderModelTypes<TargetSpace>
    {
      // Number type
      typedef typename TargetSpace::ctype RT;

      using DeformationTargetSpace = std::decay_t<decltype(std::declval<TargetSpace>()[Dune::Indices::_0])>;
      using OrientationTargetSpace = std::decay_t<decltype(std::declval<TargetSpace>()[Dune::Indices::_1])>;

      //! Dimension of a tangent space
      constexpr static int blocksize0 = DeformationTargetSpace::TangentVector::dimension;
      constexpr static int blocksize1 = OrientationTargetSpace::TangentVector::dimension;

    public:

      // Type of the local Hessian
      using Row0 = MultiTypeBlockVector<Matrix<FieldMatrix<RT, blocksize0, blocksize0> >,
          Matrix<FieldMatrix<RT, blocksize0, blocksize1> > >;
      using Row1 = MultiTypeBlockVector<Matrix<FieldMatrix<RT, blocksize1, blocksize0> >,
          Matrix<FieldMatrix<RT, blocksize1, blocksize1> > >;

      using MixedHessian = Dune::MultiTypeBlockMatrix<Row0, Row1>;
    };
  }
}

/** \brief Abstract base class for second-order energy approximations on one grid element
 *
 * \tparam TargetSpace The space we map into.  MUST be a ProductManifold with two factors
 */
template<class Basis, class TargetSpace>
class MixedLocalGeodesicFEStiffness
  : public Dune::GFE::LocalFirstOrderModel<Basis,TargetSpace>
{
  using DeformationTargetSpace = std::decay_t<decltype(std::declval<TargetSpace>()[Dune::Indices::_0])>;
  using OrientationTargetSpace = std::decay_t<decltype(std::declval<TargetSpace>()[Dune::Indices::_1])>;

  // Number type
  typedef typename TargetSpace::ctype RT;

public:

  //! Dimension of a tangent space
  constexpr static int blocksize0 = DeformationTargetSpace::TangentVector::dimension;
  constexpr static int blocksize1 = OrientationTargetSpace::TangentVector::dimension;

  /** \brief Assemble the local stiffness matrix at the current position
   */
  virtual void assembleGradientAndHessian(const typename Basis::LocalView& localView,
                                          const typename Dune::GFE::Impl::MixedLocalStiffnessTypes<TargetSpace>::CompositeCoefficients& localConfiguration,
                                          std::vector<typename DeformationTargetSpace::TangentVector>& localDeformationGradient,
                                          std::vector<typename OrientationTargetSpace::TangentVector>& localOrientationGradient,
                                          typename Dune::GFE::Impl::MixedLocalStiffnessTypes<TargetSpace>::MixedHessian& localHessian) const = 0;
};

#endif
