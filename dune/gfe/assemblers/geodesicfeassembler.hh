#ifndef GLOBAL_GEODESIC_FE_ASSEMBLER_HH
#define GLOBAL_GEODESIC_FE_ASSEMBLER_HH

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>

#include "localgeodesicfestiffness.hh"

#include <dune/solvers/common/wrapownshare.hh>


namespace Dune::GFE
{

  /** \brief A global FE assembler for problems involving functions that map into non-Euclidean spaces
   */
  template <class Basis, class TargetSpace>
  class GeodesicFEAssembler {

    using field_type = typename TargetSpace::field_type;
    typedef typename Basis::GridView GridView;
    using LocalStiffness = LocalGeodesicFEStiffness<Basis, TargetSpace>;

    //! Dimension of the grid.
    constexpr static int gridDim = GridView::dimension;

    //! Dimension of a tangent space
    constexpr static int blocksize = TargetSpace::TangentVector::dimension;

    //!
    typedef Dune::FieldMatrix<double, blocksize, blocksize> MatrixBlock;


  protected:

    //! The global basis
    const Basis basis_;

    //! The local stiffness operator
    std::shared_ptr<LocalStiffness> localStiffness_;

  public:

    /** \brief Constructor for a given grid */
    GeodesicFEAssembler(const Basis& basis)
      : basis_(basis)
    {}

    /** \brief Constructor for a given grid */
    template <class LocalStiffnessT>
    GeodesicFEAssembler(const Basis& basis,
                        LocalStiffnessT&& localStiffness)
      : basis_(basis),
      localStiffness_(Dune::Solvers::wrap_own_share<LocalStiffness>(std::forward<LocalStiffnessT>(localStiffness)))
    {}

    /** \brief Set the local stiffness assembler. This can be a temporary, l-value or shared pointer. */
    template <class LocalStiffnessT>
    void setLocalStiffness(LocalStiffnessT&& localStiffness) {
      localStiffness_ = Dune::Solvers::wrap_own_share<LocalStiffness>(std::forward<LocalStiffnessT>(localStiffness));
    }

    /** \brief Get the local stiffness operator. */
    const LocalStiffness& getLocalStiffness() const {
      return *localStiffness_;
    }

    /** \brief Get the local stiffness operator. */
    LocalStiffness& getLocalStiffness() {
      return *localStiffness_;
    }

    /** \brief Get the basis. */
    const Basis& getBasis() const {
      return basis_;
    }

    /** \brief Assemble the tangent stiffness matrix and the functional gradient together
     *
     * This is more efficient than computing them separately, because you need the gradient
     * anyway to compute the Riemannian Hessian.
     */
    virtual void assembleGradientAndHessian(const std::vector<TargetSpace>& sol,
                                            Dune::BlockVector<Dune::FieldVector<field_type, blocksize> >& gradient,
                                            Dune::BCRSMatrix<MatrixBlock>& hessian,
                                            bool computeOccupationPattern=true) const;

    /** \brief Assemble the gradient */
    virtual void assembleGradient(const std::vector<TargetSpace>& sol,
                                  Dune::BlockVector<Dune::FieldVector<double, blocksize> >& grad) const;

    /** \brief Compute the energy of a deformation state */
    virtual double computeEnergy(const std::vector<TargetSpace>& sol) const;

    //protected:
    void getNeighborsPerVertex(Dune::MatrixIndexSet& nb) const;

  }; // end class



  template <class Basis, class TargetSpace>
  void GeodesicFEAssembler<Basis,TargetSpace>::
  getNeighborsPerVertex(Dune::MatrixIndexSet& nb) const
  {
    auto n = basis_.size();

    nb.resize(n, n);

    // A view on the FE basis on a single element
    auto localView = basis_.localView();

    for (const auto& element : elements(basis_.gridView(), Dune::Partitions::interior))
    {
      // Bind the local FE basis view to the current element
      localView.bind(element);

      const auto& lfe = localView.tree().finiteElement();

      for (size_t i=0; i<lfe.size(); i++) {

        for (size_t j=0; j<lfe.size(); j++) {

          auto iIdx = localView.index(i);
          auto jIdx = localView.index(j);

          nb.add(iIdx, jIdx);

        }

      }

    }

  }

  template <class Basis, class TargetSpace>
  void GeodesicFEAssembler<Basis,TargetSpace>::
  assembleGradientAndHessian(const std::vector<TargetSpace>& sol,
                             Dune::BlockVector<Dune::FieldVector<field_type, blocksize> > &gradient,
                             Dune::BCRSMatrix<MatrixBlock>& hessian,
                             bool computeOccupationPattern) const
  {
    if (computeOccupationPattern) {

      Dune::MatrixIndexSet neighborsPerVertex;
      getNeighborsPerVertex(neighborsPerVertex);
      neighborsPerVertex.exportIdx(hessian);

    }

    hessian = 0;

    gradient.resize(sol.size());
    gradient = 0;

    // A view on the FE basis on a single element
    auto localView = basis_.localView();

    for (const auto& element : elements(basis_.gridView(), Dune::Partitions::interior))
    {
      localView.bind(element);

      const int numOfBaseFct = localView.tree().size();

      // Extract local solution
      std::vector<TargetSpace> localSolution(numOfBaseFct);

      for (int i=0; i<numOfBaseFct; i++)
        localSolution[i] = sol[localView.index(i)];

      std::vector<double> localGradient(numOfBaseFct*blocksize);
      Dune::Matrix<double> localHessian(numOfBaseFct*blocksize, numOfBaseFct*blocksize);

      // setup local matrix and gradient
      localStiffness_->assembleGradientAndHessian(localView, localSolution, localGradient, localHessian);

      // Add element matrix to global stiffness matrix
      for(int i=0; i<numOfBaseFct; i++) {

        auto row = localView.index(i);

        for (int j=0; j<numOfBaseFct; j++ ) {

          auto col = localView.index(j);

          for (std::size_t ii=0; ii<blocksize; ii++)
            for (std::size_t jj=0; jj<blocksize; jj++)
              hessian[row][col][ii][jj] += localHessian[i*blocksize+ii][j*blocksize+jj];

        }
      }

      // Add local gradient to global gradient
      for (int i=0; i<numOfBaseFct; i++)
        for (int j=0; j<blocksize; j++)
          gradient[localView.index(i)][j] += localGradient[i*blocksize + j];

    }

  }

  template <class Basis, class TargetSpace>
  void GeodesicFEAssembler<Basis,TargetSpace>::
  assembleGradient(const std::vector<TargetSpace>& sol,
                   Dune::BlockVector<Dune::FieldVector<double, blocksize> >& grad) const
  {
    if (sol.size()!=basis_.size())
      DUNE_THROW(Dune::Exception, "Solution vector doesn't match the grid!");

    grad.resize(sol.size());
    grad = 0;

    // A view on the FE basis on a single element
    auto localView = basis_.localView();

    // Loop over all elements
    for (const auto& element : elements(basis_.gridView(), Dune::Partitions::interior))
    {
      localView.bind(element);

      // The number of degrees of freedom of the current element
      const auto nDofs = localView.tree().size();

      // Extract local solution
      std::vector<TargetSpace> localSolution(nDofs);

      for (size_t i=0; i<nDofs; i++)
        localSolution[i] = sol[localView.index(i)];

      // Assemble local gradient
      std::vector<double> localGradient(nDofs*blocksize);

      localStiffness_->assembleGradient(localView, localSolution, localGradient);

      // Add to global gradient
      for (size_t i=0; i<nDofs; i++)
        for (size_t j=0; j<blocksize; j++)
          grad[localView.index(i)[0]][j] += localGradient[i*blocksize+j];
    }

  }


  template <class Basis, class TargetSpace>
  double GeodesicFEAssembler<Basis, TargetSpace>::
  computeEnergy(const std::vector<TargetSpace>& sol) const
  {
    double energy = 0;

    if (sol.size() != basis_.size())
      DUNE_THROW(Dune::Exception, "Coefficient vector doesn't match the function space basis!");

    // A view on the FE basis on a single element
    auto localView = basis_.localView();

    // Loop over all elements
    for (const auto& element : elements(basis_.gridView(), Dune::Partitions::interior))
    {
      localView.bind(element);

      // Number of degrees of freedom on this element
      size_t nDofs = localView.tree().size();

      std::vector<TargetSpace> localSolution(nDofs);

      for (size_t i=0; i<nDofs; i++)
        localSolution[i] = sol[localView.index(i)[0]];

      energy += localStiffness_->energy(localView, localSolution);

    }

    return energy;

  }

}  // namespace Dune::GFE

#endif
