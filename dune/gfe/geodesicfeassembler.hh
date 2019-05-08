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
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
    auto localIndexSet = basis_.localIndexSet();
#endif

    for (const auto& element : elements(basis_.gridView(), Dune::Partitions::interior))
    {
        // Bind the local FE basis view to the current element
        localView.bind(element);
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
        localIndexSet.bind(localView);
#endif

        const auto& lfe = localView.tree().finiteElement();

        for (size_t i=0; i<lfe.size(); i++) {

            for (size_t j=0; j<lfe.size(); j++) {

#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
                auto iIdx = localIndexSet.index(i)[0];
                auto jIdx = localIndexSet.index(j)[0];
#else
                auto iIdx = localView.index(i);
                auto jIdx = localView.index(j);
#endif

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
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
    auto localIndexSet = basis_.localIndexSet();
#endif

    for (const auto& element : elements(basis_.gridView(), Dune::Partitions::interior))
    {
        localView.bind(element);
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
        localIndexSet.bind(localView);
#endif

        const int numOfBaseFct = localView.tree().size();

        // Extract local solution
        std::vector<TargetSpace> localSolution(numOfBaseFct);

        for (int i=0; i<numOfBaseFct; i++)
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
            localSolution[i] = sol[localIndexSet.index(i)[0]];
#else
            localSolution[i] = sol[localView.index(i)];
#endif

        std::vector<Dune::FieldVector<double,blocksize> > localGradient(numOfBaseFct);

        // setup local matrix and gradient
        localStiffness_->assembleGradientAndHessian(localView, localSolution, localGradient);

        // Add element matrix to global stiffness matrix
        for(int i=0; i<numOfBaseFct; i++) {

#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
            auto row = localIndexSet.index(i)[0];
#else
            auto row = localView.index(i);
#endif

            for (int j=0; j<numOfBaseFct; j++ ) {

#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
                auto col = localIndexSet.index(j)[0];
#else
                auto col = localView.index(j);
#endif
                hessian[row][col] += localStiffness_->A_[i][j];

            }
        }

        // Add local gradient to global gradient
        for (int i=0; i<numOfBaseFct; i++)
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
            gradient[localIndexSet.index(i)[0]] += localGradient[i];
#else
            gradient[localView.index(i)] += localGradient[i];
#endif

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
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
    auto localIndexSet = basis_.localIndexSet();
#endif

    // Loop over all elements
    for (const auto& element : elements(basis_.gridView(), Dune::Partitions::interior))
    {
        localView.bind(element);
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
        localIndexSet.bind(localView);
#endif

        // A 1d grid has two vertices
        const auto nDofs = localView.tree().size();

        // Extract local solution
        std::vector<TargetSpace> localSolution(nDofs);

        for (size_t i=0; i<nDofs; i++)
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
            localSolution[i] = sol[localIndexSet.index(i)[0]];
#else
            localSolution[i] = sol[localView.index(i)];
#endif

        // Assemble local gradient
        std::vector<Dune::FieldVector<double,blocksize> > localGradient(nDofs);

        localStiffness_->assembleGradient(localView, localSolution, localGradient);

        // Add to global gradient
        for (size_t i=0; i<nDofs; i++)
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
            grad[localIndexSet.index(i)[0]] += localGradient[i];
#else
            grad[localView.index(i)[0]] += localGradient[i];
#endif
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
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
    auto localIndexSet = basis_.localIndexSet();
#endif

    // Loop over all elements
    for (const auto& element : elements(basis_.gridView(), Dune::Partitions::interior))
    {
        localView.bind(element);
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
        localIndexSet.bind(localView);
#endif

        // Number of degrees of freedom on this element
        size_t nDofs = localView.tree().size();

        std::vector<TargetSpace> localSolution(nDofs);

        for (size_t i=0; i<nDofs; i++)
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
            localSolution[i] = sol[localIndexSet.index(i)[0]];
#else
            localSolution[i] = sol[localView.index(i)[0]];
#endif

        energy += localStiffness_->energy(localView, localSolution);

    }

    return energy;

}



#endif

