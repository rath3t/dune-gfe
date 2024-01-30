#ifndef DUNE_GFE_MIXEDLOCALGEODESICFESTIFFNESS_HH
#define DUNE_GFE_MIXEDLOCALGEODESICFESTIFFNESS_HH

#include <dune/common/fmatrix.hh>

#include <dune/istl/matrix.hh>
#include <dune/istl/multitypeblockmatrix.hh>


/** \brief Abstract base class for second-order energy approximations on one grid element
 *
 * \tparam TargetSpace The space we map into.  MUST be a ProductManifold with two factors
 */
template<class Basis, class TargetSpace>
class MixedLocalGeodesicFEStiffness
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

  // assembled tangent matrix
  using Row0 = Dune::MultiTypeBlockVector<Dune::Matrix<Dune::FieldMatrix<RT, blocksize0, blocksize0> >,
      Dune::Matrix<Dune::FieldMatrix<RT, blocksize0, blocksize1> > >;
  using Row1 = Dune::MultiTypeBlockVector<Dune::Matrix<Dune::FieldMatrix<RT, blocksize1, blocksize0> >,
      Dune::Matrix<Dune::FieldMatrix<RT, blocksize1, blocksize1> > >;

  Dune::MultiTypeBlockMatrix<Row0, Row1> A_;
};

#endif
