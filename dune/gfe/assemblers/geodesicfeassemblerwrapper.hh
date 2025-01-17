#ifndef GLOBAL_GEODESIC_FE_ASSEMBLERWRAPPER_HH
#define GLOBAL_GEODESIC_FE_ASSEMBLERWRAPPER_HH

#include <dune/gfe/assemblers/mixedgfeassembler.hh>
#include <dune/common/tuplevector.hh>
#include <dune/gfe/spaces/productmanifold.hh>

namespace Dune::GFE
{

  /** \brief A wrapper that wraps a MixedGFEAssembler into an assembler that does not distinguish between the two finite element spaces

      It reimplements the same methods as these two and transfers the structure of the gradient and the Hessian
   */

  template <class Basis, class ScalarBasis, class TargetSpace>
  class GeodesicFEAssemblerWrapper
  {
    using MixedSpace0 = std::tuple_element_t<0,TargetSpace>;
    using MixedSpace1 = std::tuple_element_t<1,TargetSpace>;

    typedef typename Basis::GridView GridView;

    //! Dimension of the grid.
    constexpr static int gridDim = GridView::dimension;

    //! Dimension of the tangent space
    constexpr static int blocksize = TargetSpace::TangentVector::dimension;
    constexpr static int blocksize0 = MixedSpace0::TangentVector::dimension;
    constexpr static int blocksize1 = MixedSpace1::TangentVector::dimension;

    //!
    typedef Dune::FieldMatrix<double, blocksize, blocksize> MatrixBlock;
    typedef typename MixedGFEAssembler<Basis, TargetSpace>::MatrixType MatrixType;

  protected:
    MixedGFEAssembler<Basis, TargetSpace>* mixedAssembler_;

  public:
    const ScalarBasis& basis_;

    /** \brief Constructor for a given grid */
    GeodesicFEAssemblerWrapper(MixedGFEAssembler<Basis, TargetSpace>* mixedAssembler, ScalarBasis& basis)
      : mixedAssembler_(mixedAssembler),
      basis_(basis)
    {
      hessianMixed_ = std::make_unique<MatrixType>();
    }

    /** \brief Assemble the tangent stiffness matrix and the functional gradient together
     *
     * This is more efficient than computing them separately, because you need the gradient
     * anyway to compute the Riemannian Hessian.
     */
    virtual void assembleGradientAndHessian(const std::vector<TargetSpace>& sol,
                                            Dune::BlockVector<Dune::FieldVector<double, blocksize> >& gradient,
                                            Dune::BCRSMatrix<MatrixBlock>& hessian,
                                            bool computeOccupationPattern=true) const;

    /** \brief Compute the energy of a deformation state */
    virtual double computeEnergy(const std::vector<TargetSpace>& sol) const;

    /** \brief Get the occupation structure of the Hessian */
    virtual void getNeighborsPerVertex(Dune::MatrixIndexSet& nb) const;

    /** \brief Get the basis. */
    const ScalarBasis& getBasis() const {
      return basis_;
    }

  private:
    auto splitVector(const std::vector<TargetSpace>& sol) const;
    std::unique_ptr<MatrixType> hessianMixed_;
  }; // end class

}  // namespace Dune::GFE


template <class Basis, class ScalarBasis, class TargetSpace>
auto Dune::GFE::GeodesicFEAssemblerWrapper<Basis, ScalarBasis, TargetSpace>::
splitVector(const std::vector<TargetSpace>& sol) const {
  using namespace Indices;
  // Split the solution into the Deformation and the Rotational part
  auto n = basis_.size();
  assert (sol.size() == n);
  Dune::TupleVector<std::vector<MixedSpace0>,std::vector<MixedSpace1> > solutionSplit;
  solutionSplit[_0].resize(n);
  solutionSplit[_1].resize(n);
  for (std::size_t i = 0; i < n; i++) {
    solutionSplit[_0][i] = sol[i][_0];       // Deformation part
    solutionSplit[_1][i] = sol[i][_1];       // Rotational part
  }
  return solutionSplit;
}

template <class Basis, class ScalarBasis, class TargetSpace>
void Dune::GFE::GeodesicFEAssemblerWrapper<Basis, ScalarBasis, TargetSpace>::
getNeighborsPerVertex(Dune::MatrixIndexSet& nb) const
{
  auto n = basis_.size();
  nb.resize(n, n);
  //Retrieve occupation structure for the mixed space and convert it to the non-mixed space
  //In the mixed space, each index set is for one part of the composite basis
  //So: nb00 is for the displacement part, nb11 is for the rotation part and nb01 and nb10 (where nb01^T = nb10) is for the coupling
  Dune::MatrixIndexSet nb00, nb01, nb10, nb11;
  mixedAssembler_->getMatrixPattern(nb00, nb01, nb10, nb11);
  auto size0 = nb00.rows();
  auto size1 = nb11.rows();
  assert(size0 == size1);
  assert(size0 == n);
  //After checking if the sizes match, we can just copy over the occupation pattern
  //as all occupation patterns work on the same basis combination, so they are all equal
  nb = nb00;
}

template <class Basis, class ScalarBasis, class TargetSpace>
void Dune::GFE::GeodesicFEAssemblerWrapper<Basis, ScalarBasis, TargetSpace>::
assembleGradientAndHessian(const std::vector<TargetSpace>& sol,
                           Dune::BlockVector<Dune::FieldVector<double, blocksize> >& gradient,
                           Dune::BCRSMatrix<MatrixBlock>& hessian,
                           bool computeOccupationPattern) const
{
  using namespace Dune::Indices;
  auto n = basis_.size();

  // Get a split up version of the input
  auto solutionSplit = splitVector(sol);

  // Define the matrix and the gradient in the block structure
  Dune::BlockVector<Dune::FieldVector<double, blocksize0> > gradient0(n);
  Dune::BlockVector<Dune::FieldVector<double, blocksize1> > gradient1(n);

  if (computeOccupationPattern) {
    Dune::MatrixIndexSet neighborsPerVertex;
    getNeighborsPerVertex(neighborsPerVertex);
    neighborsPerVertex.exportIdx((*hessianMixed_)[_0][_0]);
    neighborsPerVertex.exportIdx((*hessianMixed_)[_0][_1]);
    neighborsPerVertex.exportIdx((*hessianMixed_)[_1][_0]);
    neighborsPerVertex.exportIdx((*hessianMixed_)[_1][_1]);
    neighborsPerVertex.exportIdx(hessian);
  }

  *hessianMixed_ = 0;
  mixedAssembler_->assembleGradientAndHessian(solutionSplit[_0], solutionSplit[_1], gradient0, gradient1, *hessianMixed_, false);

  // Transform gradient and hessian back to the non-mixed structure
  hessian = 0;
  gradient.resize(n);
  gradient = 0;
  for (std::size_t i = 0; i < n; i++) {
    for (int j = 0; j < blocksize0 + blocksize1; j++)
      gradient[i][j] = j < blocksize0 ? gradient0[i][j] : gradient1[i][j - blocksize0];
  }

  // All 4 matrices are nxn;
  // Each FieldMatrix in (*hessianMixed_)[_0][_0] is blocksize0 x blocksize0
  // Each FieldMatrix in (*hessianMixed_)[_1][_0] is blocksize1 x blocksize0
  // Each FieldMatrix in (*hessianMixed_)[_0][_1] is blocksize0 x blocksize1
  // Each FieldMatrix in (*hessianMixed_)[_1][_1] is blocksize1 x blocksize1
  // The hessian will then be a nxn matrix where each FieldMatrix is (blocksize0+blocksize1)x(blocksize0+blocksize1)

  for (size_t l = 0; l < blocksize0; l++) {
    // Extract Upper Left Block of mixed matrix
    for (auto rowIt = (*hessianMixed_)[_0][_0].begin(); rowIt != (*hessianMixed_)[_0][_0].end(); ++rowIt)
      for (auto colIt = rowIt->begin(); colIt != rowIt->end(); ++colIt)
        for(size_t k = 0; k < blocksize0; k++)
          hessian[rowIt.index()][colIt.index()][k][l] = (*colIt)[k][l];
    // Extract Lower Left Block of mixed matrix
    for (auto rowIt = (*hessianMixed_)[_1][_0].begin(); rowIt != (*hessianMixed_)[_1][_0].end(); ++rowIt)
      for (auto colIt = rowIt->begin(); colIt != rowIt->end(); ++colIt)
        for(size_t k = 0; k < blocksize1; k++)
          hessian[rowIt.index()][colIt.index()][k + blocksize0][l] = (*colIt)[k][l];
  }
  for (size_t l = 0; l < blocksize1; l++) {
    // Extract Upper Right Block of mixed matrix
    for (auto rowIt = (*hessianMixed_)[_0][_1].begin(); rowIt != (*hessianMixed_)[_0][_1].end(); ++rowIt)
      for (auto colIt = rowIt->begin(); colIt != rowIt->end(); ++colIt)
        for(size_t k = 0; k < blocksize0; k++)
          hessian[rowIt.index()][colIt.index()][k][l + blocksize0] = (*colIt)[k][l];
    // Extract Lower Right Block of mixed matrix
    for (auto rowIt = (*hessianMixed_)[_1][_1].begin(); rowIt != (*hessianMixed_)[_1][_1].end(); ++rowIt)
      for (auto colIt = rowIt->begin(); colIt != rowIt->end(); ++colIt)
        for(size_t k = 0; k < blocksize1; k++)
          hessian[rowIt.index()][colIt.index()][k + blocksize0][l + blocksize0] = (*colIt)[k][l];
  }
}

template <class Basis, class ScalarBasis, class TargetSpace>
double Dune::GFE::GeodesicFEAssemblerWrapper<Basis, ScalarBasis, TargetSpace>::
computeEnergy(const std::vector<TargetSpace>& sol) const
{
  using namespace Dune::Indices;
  auto solutionSplit = splitVector(sol);
  return mixedAssembler_->computeEnergy(solutionSplit[_0], solutionSplit[_1]);
}
#endif //GLOBAL_GEODESIC_FE_ASSEMBLERWRAPPER_HH
