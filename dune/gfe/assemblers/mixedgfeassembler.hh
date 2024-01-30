#ifndef DUNE_GFE_MIXEDGFEASSEMBLER_HH
#define DUNE_GFE_MIXEDGFEASSEMBLER_HH

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>
#include <dune/istl/multitypeblockmatrix.hh>

#include <dune/gfe/assemblers/mixedlocalgeodesicfestiffness.hh>


/** \brief A global FE assembler for problems involving functions that map into non-Euclidean spaces
 */
template <class Basis, class TargetSpace0, class TargetSpace1>
class MixedGFEAssembler {

  typedef typename Basis::GridView GridView;
  typedef typename GridView::template Codim<0>::template Partition<Dune::Interior_Partition>::Iterator ElementIterator;

  //! Dimension of the grid.
  constexpr static int gridDim = GridView::dimension;

  //! Dimension of a tangent space
  constexpr static int blocksize0 = TargetSpace0::TangentVector::dimension;
  constexpr static int blocksize1 = TargetSpace1::TangentVector::dimension;

  //!
  typedef Dune::BCRSMatrix<Dune::FieldMatrix<double, blocksize0, blocksize0> > MatrixBlock00;
  typedef Dune::BCRSMatrix<Dune::FieldMatrix<double, blocksize0, blocksize1> > MatrixBlock01;
  typedef Dune::BCRSMatrix<Dune::FieldMatrix<double, blocksize1, blocksize0> > MatrixBlock10;
  typedef Dune::BCRSMatrix<Dune::FieldMatrix<double, blocksize1, blocksize1> > MatrixBlock11;

public:
  typedef Dune::MultiTypeBlockMatrix<Dune::MultiTypeBlockVector<MatrixBlock00,MatrixBlock01>,
      Dune::MultiTypeBlockVector<MatrixBlock10,MatrixBlock11> > MatrixType;
  const Basis basis_;

  MixedLocalGeodesicFEStiffness<Basis, Dune::GFE::ProductManifold<TargetSpace0,TargetSpace1> >* localStiffness_;

public:

  /** \brief Constructor for a given grid */
  MixedGFEAssembler(const Basis& basis,
                    MixedLocalGeodesicFEStiffness<Basis, Dune::GFE::ProductManifold<TargetSpace0, TargetSpace1> >* localStiffness)
    : basis_(basis),
    localStiffness_(localStiffness)
  {}

  /** \brief Assemble the tangent stiffness matrix and the functional gradient together
   *
   * This is more efficient than computing them separately, because you need the gradient
   * anyway to compute the Riemannian Hessian.
   */
  virtual void assembleGradientAndHessian(const std::vector<TargetSpace0>& configuration0,
                                          const std::vector<TargetSpace1>& configuration1,
                                          Dune::BlockVector<Dune::FieldVector<double, blocksize0> >& gradient0,
                                          Dune::BlockVector<Dune::FieldVector<double, blocksize1> >& gradient1,
                                          MatrixType& hessian,
                                          bool computeOccupationPattern=true) const;

  /** \brief Compute the energy of a deformation state */
  virtual double computeEnergy(const std::vector<TargetSpace0>& configuration0,
                               const std::vector<TargetSpace1>& configuration1) const;

  //protected:
  void getMatrixPattern(Dune::MatrixIndexSet& nb00,
                        Dune::MatrixIndexSet& nb01,
                        Dune::MatrixIndexSet& nb10,
                        Dune::MatrixIndexSet& nb11) const;

}; // end class



template <class Basis, class TargetSpace0, class TargetSpace1>
void MixedGFEAssembler<Basis,TargetSpace0,TargetSpace1>::
getMatrixPattern(Dune::MatrixIndexSet& nb00,
                 Dune::MatrixIndexSet& nb01,
                 Dune::MatrixIndexSet& nb10,
                 Dune::MatrixIndexSet& nb11) const
{
  nb00.resize(basis_.size({0}), basis_.size({0}));
  nb01.resize(basis_.size({0}), basis_.size({1}));
  nb10.resize(basis_.size({1}), basis_.size({0}));
  nb11.resize(basis_.size({1}), basis_.size({1}));

  // A view on the FE basis on a single element
  auto localView = basis_.localView();

  // Loop over grid elements
  for (const auto& element : elements(basis_.gridView(), Dune::Partitions::interior))
  {
    // Bind the local FE basis view to the current element
    localView.bind(element);
    // Add element stiffness matrix onto the global stiffness matrix
    for (size_t i=0; i<localView.size(); i++)
    {
      // The global index of the i-th local degree of freedom of the element 'e'
      auto row = localView.index(i);

      for (size_t j=0; j<localView.size(); j++ )
      {
        // The global index of the j-th local degree of freedom of the element 'e'
        auto col = localView.index(j);

        if (row[0]==0 and col[0]==0)
          nb00.add(row[1],col[1]);
        if (row[0]==0 and col[0]==1)
          nb01.add(row[1],col[1]);
        if (row[0]==1 and col[0]==0)
          nb10.add(row[1],col[1]);
        if (row[0]==1 and col[0]==1)
          nb11.add(row[1],col[1]);

      }

    }

  }

}

template <class Basis, class TargetSpace0, class TargetSpace1>
void MixedGFEAssembler<Basis,TargetSpace0,TargetSpace1>::
assembleGradientAndHessian(const std::vector<TargetSpace0>& configuration0,
                           const std::vector<TargetSpace1>& configuration1,
                           Dune::BlockVector<Dune::FieldVector<double, blocksize0> >& gradient0,
                           Dune::BlockVector<Dune::FieldVector<double, blocksize1> >& gradient1,
                           MatrixType& hessian,
                           bool computeOccupationPattern) const
{
  if (computeOccupationPattern) {

    Dune::MatrixIndexSet pattern00;
    Dune::MatrixIndexSet pattern01;
    Dune::MatrixIndexSet pattern10;
    Dune::MatrixIndexSet pattern11;

    getMatrixPattern(pattern00, pattern01, pattern10, pattern11);

    using namespace Dune::TypeTree::Indices;
    pattern00.exportIdx(hessian[_0][_0]);
    pattern01.exportIdx(hessian[_0][_1]);
    pattern10.exportIdx(hessian[_1][_0]);
    pattern11.exportIdx(hessian[_1][_1]);

  }

  hessian = 0;

  gradient0.resize(configuration0.size());
  gradient0 = 0;
  gradient1.resize(configuration1.size());
  gradient1 = 0;

  // A view on the FE basis on a single element
  auto localView = basis_.localView();
  for (const auto& element : elements(basis_.gridView(), Dune::Partitions::interior))
  {
    // Bind the local FE basis view to the current element
    localView.bind(element);
    using namespace Dune::TypeTree::Indices;

    const int nDofs0 = localView.tree().child(_0,0).finiteElement().size();
    const int nDofs1 = localView.tree().child(_1,0).finiteElement().size();
    // This loop reads out the pattern for a local matrix; in each element, we have localView.size() degrees of freedom; from the composite and powerbasis layers
    // nDofs0 are the degrees of freedom for *one* subspacebasis of the power basis of the displacement part;
    // nDofs1 are the degrees of freedom for *one* subspacebasis of the power basis of the rotational part
    // this is why the indices (_0,0) and (_1,0) are used: _0 takes the whole displacement part and _1 the whole rotational part; and 0 the first subspacebasis respectively
    // Extract local solution
    std::vector<TargetSpace0> localConfiguration0(nDofs0);
    std::vector<TargetSpace1> localConfiguration1(nDofs1);

    for (int i=0; i<nDofs0+nDofs1; i++)
    {
      int localIndexI = 0;
      if (i < nDofs0) {
        auto& node = localView.tree().child(_0,0);
        localIndexI = node.localIndex(i);
      } else {
        auto& node = localView.tree().child(_1,0);
        localIndexI = node.localIndex(i-nDofs0);
      }
      auto multiIndex = localView.index(localIndexI);
      //CompositeBasis number is contained in multiIndex[0], the Subspacebasis is contained in multiIndex[2]
      //multiIndex[1] contains the actual index
      if (multiIndex[0] == 0)
        localConfiguration0[i] = configuration0[multiIndex[1]];
      else if (multiIndex[0] == 1)
        localConfiguration1[i-nDofs0] = configuration1[multiIndex[1]];
    }

    std::vector<Dune::FieldVector<double,blocksize0> > localGradient0(nDofs0);
    std::vector<Dune::FieldVector<double,blocksize1> > localGradient1(nDofs1);

    // setup local matrix and gradient
    localStiffness_->assembleGradientAndHessian(localView,
                                                localConfiguration0, localConfiguration1,
                                                localGradient0, localGradient1);

    // Add element matrix to global stiffness matrix
    for (int i=0; i<nDofs0+nDofs1; i++)
    {
      int localIndexRow = 0;
      if (i < nDofs0) {
        auto& node = localView.tree().child(_0,0);
        localIndexRow = node.localIndex(i);
      } else {
        auto& node = localView.tree().child(_1,0);
        localIndexRow = node.localIndex(i-nDofs0);
      }

      auto row = localView.index(localIndexRow);

      for (int j=0; j<nDofs0+nDofs1; j++ )
      {
        int localIndexCol = 0;
        if (j < nDofs0) {
          auto& node = localView.tree().child(_0,0);
          localIndexCol = node.localIndex(j);
        } else {
          auto& node = localView.tree().child(_1,0);
          localIndexCol = node.localIndex(j-nDofs0);
        }

        auto col = localView.index(localIndexCol);

        if (row[0]==0 and col[0]==0)
          hessian[_0][_0][row[1]][col[1]] += localStiffness_->A_[_0][_0][i][j];

        if (row[0]==0 and col[0]==1)
          hessian[_0][_1][row[1]][col[1]] += localStiffness_->A_[_0][_1][i][j-nDofs0];

        if (row[0]==1 and col[0]==0)
          hessian[_1][_0][row[1]][col[1]] += localStiffness_->A_[_1][_0][i-nDofs0][j];

        if (row[0]==1 and col[0]==1)
          hessian[_1][_1][row[1]][col[1]] += localStiffness_->A_[_1][_1][i-nDofs0][j-nDofs0];
      }

      // Add local gradient to global gradient
      if (row[0] == 0)
        gradient0[row[1]] += localGradient0[i];
      else
        gradient1[row[1]] += localGradient1[i-nDofs0];
    }

  }
}

template <class Basis, class TargetSpace0, class TargetSpace1>
double MixedGFEAssembler<Basis, TargetSpace0, TargetSpace1>::
computeEnergy(const std::vector<TargetSpace0>& configuration0,
              const std::vector<TargetSpace1>& configuration1) const
{
  double energy = 0;

  if (configuration0.size()!=basis_.size({0}))
    DUNE_THROW(Dune::Exception, "Configuration vector 0 doesn't match the basis!");

  if (configuration1.size()!=basis_.size({1}))
    DUNE_THROW(Dune::Exception, "Configuration vector 1 doesn't match the basis!");

  // A view on the FE basis on a single element
  auto localView = basis_.localView();

  // Loop over all elements
  for (const auto& element : elements(basis_.gridView(), Dune::Partitions::interior))
  {
    // Bind the local FE basis view to the current element
    localView.bind(element);

    // Number of degrees of freedom on this element
    using namespace Dune::TypeTree::Indices;
    const int nDofs0 = localView.tree().child(_0,0).finiteElement().size();
    const int nDofs1 = localView.tree().child(_1,0).finiteElement().size();

    std::vector<TargetSpace0> localConfiguration0(nDofs0);
    std::vector<TargetSpace1> localConfiguration1(nDofs1);

    for (int i=0; i<nDofs0+nDofs1; i++)
    {
      int localIndexI = 0;
      if (i < nDofs0) {
        auto& node = localView.tree().child(_0,0);
        localIndexI = node.localIndex(i);
      } else {
        auto& node = localView.tree().child(_1,0);
        localIndexI = node.localIndex(i-nDofs0);
      }

      auto multiIndex = localView.index(localIndexI);

      // The CompositeBasis number is contained in multiIndex[0]
      // multiIndex[1] contains the actual index
      if (multiIndex[0] == 0)
        localConfiguration0[i] = configuration0[multiIndex[1]];
      else if (multiIndex[0] == 1)
        localConfiguration1[i-nDofs0] = configuration1[multiIndex[1]];
    }

    energy += localStiffness_->energy(localView,
                                      localConfiguration0,
                                      localConfiguration1);

  }

  return energy;

}

#endif
