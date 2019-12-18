#ifndef GLOBAL_GEODESIC_FE_ASSEMBLER_HH
#define GLOBAL_GEODESIC_FE_ASSEMBLER_HH

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>

#include "localgeodesicfestiffness.hh"


/** \brief A global FE assembler for problems involving functions that map into non-Euclidean spaces
 */
template <class Basis, class TargetSpace>
class GeodesicFEAssembler {

    typedef typename Basis::GridView GridView;

    //! Dimension of the grid.
    enum { gridDim = GridView::dimension };

    //! Dimension of a tangent space
    enum { blocksize = TargetSpace::TangentVector::dimension };

    //!
    typedef Dune::FieldMatrix<double, blocksize, blocksize> MatrixBlock;

public:
    const Basis basis_;

protected:

    LocalGeodesicFEStiffness<Basis,TargetSpace>* localStiffness_;

public:

    /** \brief Constructor for a given grid */
    GeodesicFEAssembler(const Basis& basis,
                        LocalGeodesicFEStiffness<Basis, TargetSpace>* localStiffness)
        : basis_(basis),
          localStiffness_(localStiffness)
    {}

    /** \brief Assemble the tangent stiffness matrix and the functional gradient together
     *
     * This is more efficient than computing them separately, because you need the gradient
     * anyway to compute the Riemannian Hessian.
     */
    virtual void assembleGradientAndHessian(const std::vector<TargetSpace>& sol,
                                            Dune::BlockVector<Dune::FieldVector<double, blocksize> >& gradient,
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
                           Dune::BlockVector<Dune::FieldVector<double, blocksize> >& gradient,
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

        std::vector<Dune::FieldVector<double,blocksize> > localGradient(numOfBaseFct);

        // setup local matrix and gradient
        localStiffness_->assembleGradientAndHessian(localView, localSolution, localGradient);

        // Add element matrix to global stiffness matrix
        for(int i=0; i<numOfBaseFct; i++) {

            auto row = localView.index(i);

            for (int j=0; j<numOfBaseFct; j++ ) {

                auto col = localView.index(j);
                hessian[row][col] += localStiffness_->A_[i][j];

            }
        }

        // Add local gradient to global gradient
        for (int i=0; i<numOfBaseFct; i++)
            gradient[localView.index(i)] += localGradient[i];

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

        // A 1d grid has two vertices
        const auto nDofs = localView.tree().size();

        // Extract local solution
        std::vector<TargetSpace> localSolution(nDofs);

        for (size_t i=0; i<nDofs; i++)
            localSolution[i] = sol[localView.index(i)];

        // Assemble local gradient
        std::vector<Dune::FieldVector<double,blocksize> > localGradient(nDofs);

        localStiffness_->assembleGradient(localView, localSolution, localGradient);

        // Add to global gradient
        for (size_t i=0; i<nDofs; i++)
            grad[localView.index(i)[0]] += localGradient[i];
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



#endif

