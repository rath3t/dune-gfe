#ifndef DUNE_GFE_MIXED_RIEMANNIAN_PROXIMAL_NEWTON_SOLVER_HH
#define DUNE_GFE_MIXED_RIEMANNIAN_PROXIMAL_NEWTON_SOLVER_HH

#include <vector>

#include <dune/common/bitsetvector.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/multitypeblockmatrix.hh>

#include <dune/grid/utility/globalindexset.hh>

#include <dune/solvers/solvers/cholmodsolver.hh>

#include <dune/gfe/parallel/mapperfactory.hh>


namespace Dune::GFE
{

  /** \brief Riemannian proximal Newton solver for geodesic finite-element problems */
  template <class MixedBasis,
      class Basis0,
      class TargetSpace0,
      class Basis1,
      class TargetSpace1,
      class BitVector>
  class MixedRiemannianProximalNewtonSolver
    : public NumProc
  {
    using GridType = typename MixedBasis::GridView::Grid;

    using TargetSpace = ProductManifold<TargetSpace0,TargetSpace1>;

    const static int blocksize0 = TargetSpace0::TangentVector::dimension;
    const static int blocksize1 = TargetSpace1::TangentVector::dimension;

    const static int gridDim = GridType::dimension;

    // Centralize the field type here
    using field_type = double;

    using MatrixType00 = BCRSMatrix<FieldMatrix<field_type, blocksize0, blocksize0> >;
    using MatrixType01 = BCRSMatrix<FieldMatrix<field_type, blocksize0, blocksize1> >;
    using MatrixType10 = BCRSMatrix<FieldMatrix<field_type, blocksize1, blocksize0> >;
    using MatrixType11 = BCRSMatrix<FieldMatrix<field_type, blocksize1, blocksize1> >;
    typedef MultiTypeBlockMatrix<MultiTypeBlockVector<MatrixType00,MatrixType01>,
        MultiTypeBlockVector<MatrixType10,MatrixType11> > MatrixType;

    using CorrectionType0 = BlockVector<FieldVector<field_type, blocksize0> >;
    using CorrectionType1 = BlockVector<FieldVector<field_type, blocksize1> >;
    using CorrectionType  = MultiTypeBlockVector<CorrectionType0, CorrectionType1>;
    using SolutionType    = TupleVector<std::vector<TargetSpace0>, std::vector<TargetSpace1> >;
#if HAVE_MPI
    typedef typename MapperFactory<Basis0>::GlobalMapper GlobalMapper0;
    typedef typename MapperFactory<Basis1>::GlobalMapper GlobalMapper1;
    typedef typename MapperFactory<Basis0>::LocalMapper LocalMapper0;
    typedef typename MapperFactory<Basis1>::LocalMapper LocalMapper1;
#endif

  public:

    MixedRiemannianProximalNewtonSolver()
      : NumProc(NumProc::FULL)
    {}

    void setup(const GridType& grid,
               const MixedGFEAssembler<MixedBasis, TargetSpace>* assembler,
               const SolutionType& x,
               const BitVector& dirichletNodes,
               double tolerance,
               int maxProximalNewtonSteps,
               double initialRegularization,
               bool instrumented);
    void solve();

    void setInitialIterate(const SolutionType& x)
    {
      x_ = x;
    }

    SolutionType getSol() const
    {
      return x_;
    }

  protected:
#if HAVE_MPI
    std::unique_ptr<GlobalMapper0> globalMapper0_;
    std::unique_ptr<GlobalMapper1> globalMapper1_;
#endif
    /** \brief The grid */
    const GridType* grid_;

    /** \brief The solution vectors */
    SolutionType x_;

    /** \brief The initial regularization parameter for the proximal Newton step */
    double initialRegularization_;
    double tolerance_;

    /** \brief Maximum number of proximal-newton steps */
    std::size_t maxProximalNewtonSteps_;

    /** \brief Hesse matrix */
    std::unique_ptr<MatrixType> hessianMatrix_;

    /** \brief The assembler for the material law */
    const MixedGFEAssembler<MixedBasis, TargetSpace>* assembler_;

    /** \brief The solver for the quadratic inner problems */
    std::shared_ptr<Solvers::CholmodSolver<MatrixType, CorrectionType, BitVector> > innerSolver_;
  };

}  // namespace Dune::GFE

#include "mixedriemannianpnsolver.cc"

#endif
