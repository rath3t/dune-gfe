#ifndef DUNE_GFE_LOCAL_GEODESIC_FE_STIFFNESS_HH
#define DUNE_GFE_LOCAL_GEODESIC_FE_STIFFNESS_HH

#include <dune/common/fmatrix.hh>
#include <dune/istl/matrix.hh>
#include <dune/istl/multitypeblockmatrix.hh>

#include <dune/gfe/assemblers/localfirstordermodel.hh>

namespace Dune::GFE
{
  namespace Impl
  {
    /** \brief A class exporting container types for sets local Hesse matrices
     *
     * This generic template handles TargetSpaces that are not product manifolds.
     */
    template<class TargetSpace>
    class LocalStiffnessTypes
      : public LocalEnergyTypes<TargetSpace>
    {
      // Number type
      typedef typename TargetSpace::ctype RT;

      //! Dimension of a tangent space
      constexpr static auto blocksize = TargetSpace::TangentVector::dimension;

    public:

      // Type of the local Hessian
      using Hessian = Matrix<FieldMatrix<RT, blocksize, blocksize> >;

      using Row = MultiTypeBlockVector<Matrix<FieldMatrix<RT, blocksize, blocksize> > >;
      using CompositeHessian = MultiTypeBlockMatrix<Row>;
    };

    /** \brief A class exporting container types for sets local Hesse matrices
     *
     * This is the specialization for product manifolds.
     */
    template<class ... Factors>
    class LocalStiffnessTypes<ProductManifold<Factors...> >
      : public LocalEnergyTypes<ProductManifold<Factors...> >
    {
      using TargetSpace = ProductManifold<Factors...>;

      // Number type
      typedef typename ProductManifold<Factors...>::ctype RT;

      using DeformationTargetSpace = std::decay_t<decltype(std::declval<TargetSpace>()[Dune::Indices::_0])>;
      using OrientationTargetSpace = std::decay_t<decltype(std::declval<TargetSpace>()[Dune::Indices::_1])>;

      // Dimension of the product tangent space
      constexpr static auto blocksize = TargetSpace::TangentVector::dimension;

      // Dimensions of the individual factor tangent spaces
      constexpr static auto blocksize0 = DeformationTargetSpace::TangentVector::dimension;
      constexpr static auto blocksize1 = OrientationTargetSpace::TangentVector::dimension;

    public:

      // Type of the local Hessian
      using Hessian = Matrix<FieldMatrix<RT, blocksize, blocksize> >;

      // Type of the local Hessian
      using Row0 = MultiTypeBlockVector<Matrix<FieldMatrix<RT, blocksize0, blocksize0> >,
          Matrix<FieldMatrix<RT, blocksize0, blocksize1> > >;
      using Row1 = MultiTypeBlockVector<Matrix<FieldMatrix<RT, blocksize1, blocksize0> >,
          Matrix<FieldMatrix<RT, blocksize1, blocksize1> > >;

      using CompositeHessian = MultiTypeBlockMatrix<Row0, Row1>;
    };
  }
}

template<class Basis, class TargetSpace>
class LocalGeodesicFEStiffness
  : public Dune::GFE::LocalFirstOrderModel<Basis,TargetSpace>
{
public:

  /** \brief Assemble the local gradient and stiffness matrix at the current position

   */
  virtual void assembleGradientAndHessian(const typename Basis::LocalView& localView,
                                          const typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::Coefficients& coefficients,
                                          std::vector<double>& localGradient,
                                          typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::Hessian& localHessian) const = 0;

  /** \brief Assemble the local gradient and stiffness matrix at the current position -- Composite version
   */
  virtual void assembleGradientAndHessian(const typename Basis::LocalView& localView,
                                          const typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::CompositeCoefficients& coefficients,
                                          std::vector<double>& localGradient,
                                          typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::CompositeHessian& localHessian) const = 0;
};

#endif
