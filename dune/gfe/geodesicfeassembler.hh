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
    typedef typename GridView::template Codim<0>::template Partition<Dune::Interior_Partition>::Iterator ElementIterator;

    //! Dimension of the grid.
    enum { gridDim = GridView::dimension };

    //! Dimension of a tangent space
    enum { blocksize = TargetSpace::TangentVector::dimension };

    //!
    typedef Dune::FieldMatrix<double, blocksize, blocksize> MatrixBlock;

public:
    const Basis basis_;
    const typename Basis::IndexSet basisIndexSet_;

protected:

    LocalGeodesicFEStiffness<GridView,
                             typename Basis::LocalView::Tree::FiniteElement,
                             TargetSpace>* localStiffness_;

public:

    /** \brief Constructor for a given grid */
    GeodesicFEAssembler(const Basis& basis,
                        LocalGeodesicFEStiffness<GridView,typename Basis::LocalView::Tree::FiniteElement, TargetSpace>* localStiffness)
        : basis_(basis),
          basisIndexSet_(basis_.indexSet()),
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
    auto n = basisIndexSet_.size();

    nb.resize(n, n);

    // A view on the FE basis on a single element
    typename Basis::LocalView localView(&basis_);
    auto localIndexSet = basisIndexSet_.localIndexSet();

    ElementIterator it    = basis_.gridView().template begin<0,Dune::Interior_Partition>();
    ElementIterator endit = basis_.gridView().template end<0,Dune::Interior_Partition>  ();

    for (; it!=endit; ++it) {

        // Bind the local FE basis view to the current element
        localView.bind(*it);
        localIndexSet.bind(localView);

        const auto& lfe = localView.tree().finiteElement();

        for (size_t i=0; i<lfe.size(); i++) {

            for (size_t j=0; j<lfe.size(); j++) {

                auto iIdx = localIndexSet.index(i)[0];
                auto jIdx = localIndexSet.index(j)[0];

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
    typename Basis::LocalView localView(&basis_);
    auto localIndexSet = basisIndexSet_.localIndexSet();

    ElementIterator it    = basis_.gridView().template begin<0,Dune::Interior_Partition>();
    ElementIterator endit = basis_.gridView().template end<0,Dune::Interior_Partition>  ();

    for( ; it != endit; ++it ) {

        localView.bind(*it);
        localIndexSet.bind(localView);

        const int numOfBaseFct = localView.tree().size();

        // Extract local solution
        std::vector<TargetSpace> localSolution(numOfBaseFct);

        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[localIndexSet.index(i)[0]];

        std::vector<Dune::FieldVector<double,blocksize> > localGradient(numOfBaseFct);

        // setup local matrix and gradient
        localStiffness_->assembleGradientAndHessian(*it, localView.tree().finiteElement(), localSolution, localGradient);

        // Add element matrix to global stiffness matrix
        for(int i=0; i<numOfBaseFct; i++) {

            auto row = localIndexSet.index(i)[0];

            for (int j=0; j<numOfBaseFct; j++ ) {

                auto col = localIndexSet.index(j)[0];
                hessian[row][col] += localStiffness_->A_[i][j];

            }
        }

        // Add local gradient to global gradient
        for (int i=0; i<numOfBaseFct; i++)
            gradient[localIndexSet.index(i)[0]] += localGradient[i];

    }

}

template <class Basis, class TargetSpace>
void GeodesicFEAssembler<Basis,TargetSpace>::
assembleGradient(const std::vector<TargetSpace>& sol,
                 Dune::BlockVector<Dune::FieldVector<double, blocksize> >& grad) const
{
    if (sol.size()!=basisIndexSet_.size())
        DUNE_THROW(Dune::Exception, "Solution vector doesn't match the grid!");

    grad.resize(sol.size());
    grad = 0;

    // A view on the FE basis on a single element
    typename Basis::LocalView localView(&basis_);
    auto localIndexSet = basisIndexSet_.localIndexSet();

    ElementIterator it    = basis_.gridView().template begin<0,Dune::Interior_Partition>();
    ElementIterator endIt = basis_.gridView().template end<0,Dune::Interior_Partition>();

    // Loop over all elements
    for (; it!=endIt; ++it) {

        localView.bind(*it);
        localIndexSet.bind(localView);

        // A 1d grid has two vertices
        const auto nDofs = localView.tree().size();

        // Extract local solution
        std::vector<TargetSpace> localSolution(nDofs);

        for (int i=0; i<nDofs; i++)
            localSolution[i] = sol[localIndexSet.index(i)[0]];

        // Assemble local gradient
        std::vector<Dune::FieldVector<double,blocksize> > localGradient(nDofs);

        localStiffness_->assembleGradient(*it, localView.tree().finiteElement(), localSolution, localGradient);

        // Add to global gradient
        for (int i=0; i<nDofs; i++)
            grad[localIndexSet.index(i)[0]] += localGradient[i];

    }

}


template <class Basis, class TargetSpace>
double GeodesicFEAssembler<Basis, TargetSpace>::
computeEnergy(const std::vector<TargetSpace>& sol) const
{
    double energy = 0;

    if (sol.size() != basisIndexSet_.size())
        DUNE_THROW(Dune::Exception, "Coefficient vector doesn't match the function space basis!");

    // A view on the FE basis on a single element
    typename Basis::LocalView localView(&basis_);
    auto localIndexSet = basisIndexSet_.localIndexSet();

    ElementIterator it    = basis_.gridView().template begin<0,Dune::Interior_Partition>();
    ElementIterator endIt = basis_.gridView().template end<0,Dune::Interior_Partition>();

    // Loop over all elements
    for (; it!=endIt; ++it) {

        localView.bind(*it);
        localIndexSet.bind(localView);

        // Number of degrees of freedom on this element
        size_t nDofs = localView.tree().size();

        std::vector<TargetSpace> localSolution(nDofs);

        for (size_t i=0; i<nDofs; i++)
            localSolution[i] = sol[localIndexSet.index(i)[0]];

        energy += localStiffness_->energy(*it, localView.tree().finiteElement(), localSolution);

    }

    return energy;

}



#endif

