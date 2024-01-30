#ifndef DUNE_GFE_MIXEDLOCALGEODESICFESTIFFNESS_HH
#define DUNE_GFE_MIXEDLOCALGEODESICFESTIFFNESS_HH

#include <dune/common/fmatrix.hh>
#include <dune/istl/matrix.hh>


/** \brief Abstract base class for second-order energy approximations on one grid element
 *
 * \tparam TargetSpace The space we map into.  MUST be a ProductManifold with two factors
 */
template<class Basis, class TargetSpace>
class MixedLocalGeodesicFEStiffness
{
  using DeformationTargetSpace = std::decay_t<decltype(std::declval<TargetSpace>()[Dune::Indices::_0])>;
  using OrientationTargetSpace = std::decay_t<decltype(std::declval<TargetSpace>()[Dune::Indices::_1])>;

  // grid types
  typedef typename Basis::GridView GridView;
  typedef typename GridView::ctype DT;
  typedef typename DeformationTargetSpace::ctype RT;
  typedef typename GridView::template Codim<0>::Entity Entity;

  // some other sizes
  constexpr static int gridDim = GridView::dimension;

public:

  //! Dimension of a tangent space
  constexpr static int deformationBlocksize = DeformationTargetSpace::TangentVector::dimension;
  constexpr static int orientationBlocksize = OrientationTargetSpace::TangentVector::dimension;

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

  // assembled data
  Dune::Matrix<Dune::FieldMatrix<RT, deformationBlocksize, deformationBlocksize> > A00_;
  Dune::Matrix<Dune::FieldMatrix<RT, deformationBlocksize, orientationBlocksize> > A01_;
  Dune::Matrix<Dune::FieldMatrix<RT, orientationBlocksize, deformationBlocksize> > A10_;
  Dune::Matrix<Dune::FieldMatrix<RT, orientationBlocksize, orientationBlocksize> > A11_;

};

#endif
