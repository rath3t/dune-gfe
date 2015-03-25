#ifndef DUNE_GFE_FE_ASSEMBLER_HH
#define DUNE_GFE_FE_ASSEMBLER_HH

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>

//#include "localgeodesicfestiffness.hh"


/** \brief A global FE assembler for problems involving functions that map into non-Euclidean spaces
 */
template <class Basis, class VectorType>
class FEAssembler {

    typedef typename Basis::GridView GridView;
    typedef typename GridView::template Codim<0>::template Partition<Dune::Interior_Partition>::Iterator ElementIterator;

    //! Dimension of the grid.
    enum { gridDim = GridView::dimension };

    //! Dimension of a tangent space
    enum { blocksize = VectorType::value_type::dimension };

    //!
    typedef Dune::FieldMatrix<double, blocksize, blocksize> MatrixBlock;

public:
    const Basis basis_;

protected:

    LocalFEStiffness<GridView,
                             typename Basis::LocalFiniteElement,
                             VectorType>* localStiffness_;

public:

    /** \brief Constructor for a given grid */
    FEAssembler(const Basis& basis,
                LocalFEStiffness<GridView,typename Basis::LocalFiniteElement, VectorType>* localStiffness)
        : basis_(basis),
          localStiffness_(localStiffness)
    {}

    /** \brief Assemble the tangent stiffness matrix and the functional gradient together
     *
     * This is more efficient than computing them separately, because you need the gradient
     * anyway to compute the Riemannian Hessian.
     */
    virtual void assembleGradientAndHessian(const VectorType& sol,
                                            Dune::BlockVector<Dune::FieldVector<double, blocksize> >& gradient,
                                            Dune::BCRSMatrix<MatrixBlock>& hessian,
                                            bool computeOccupationPattern=true) const;

    /** \brief Compute the energy of a deformation state */
    virtual double computeEnergy(const VectorType& sol) const;

    //protected:
    void getNeighborsPerVertex(Dune::MatrixIndexSet& nb) const;

}; // end class



template <class Basis, class TargetSpace>
void FEAssembler<Basis,TargetSpace>::
getNeighborsPerVertex(Dune::MatrixIndexSet& nb) const
{
    int n = basis_.size();

    nb.resize(n, n);

    ElementIterator it    = basis_.getGridView().template begin<0,Dune::Interior_Partition>();
    ElementIterator endit = basis_.getGridView().template end<0,Dune::Interior_Partition>  ();

    for (; it!=endit; ++it) {

        const typename Basis::LocalFiniteElement& lfe = basis_.getLocalFiniteElement(*it);

        for (size_t i=0; i<lfe.localBasis().size(); i++) {

            for (size_t j=0; j<lfe.localBasis().size(); j++) {

                int iIdx = basis_.index(*it,i);
                int jIdx = basis_.index(*it,j);

                nb.add(iIdx, jIdx);

            }

        }

    }

}

template <class Basis, class VectorType>
void FEAssembler<Basis,VectorType>::
assembleGradientAndHessian(const VectorType& sol,
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

    ElementIterator it    = basis_.getGridView().template begin<0,Dune::Interior_Partition>();
    ElementIterator endit = basis_.getGridView().template end<0,Dune::Interior_Partition>  ();

    for( ; it != endit; ++it ) {

        const int numOfBaseFct = basis_.getLocalFiniteElement(*it).localBasis().size();

        // Extract local solution
        VectorType localSolution(numOfBaseFct);
        VectorType localPointLoads(numOfBaseFct);

        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i]   = sol[basis_.index(*it,i)];

        VectorType localGradient(numOfBaseFct);

        // setup local matrix and gradient
        localStiffness_->assembleGradientAndHessian(*it, basis_.getLocalFiniteElement(*it), localSolution, localGradient);

        // Add element matrix to global stiffness matrix
        for(int i=0; i<numOfBaseFct; i++) {

            int row = basis_.index(*it,i);

            for (int j=0; j<numOfBaseFct; j++ ) {

                int col = basis_.index(*it,j);
                hessian[row][col] += localStiffness_->A_[i][j];

            }
        }

        // Add local gradient to global gradient
        for (int i=0; i<numOfBaseFct; i++)
            gradient[basis_.index(*it,i)] += localGradient[i];

    }

}


template <class Basis, class VectorType>
double FEAssembler<Basis, VectorType>::
computeEnergy(const VectorType& sol) const
{
    double energy = 0;

    if (sol.size()!=basis_.size())
        DUNE_THROW(Dune::Exception, "Solution vector doesn't match the basis!");

    ElementIterator it    = basis_.getGridView().template begin<0,Dune::Interior_Partition>();
    ElementIterator endIt = basis_.getGridView().template end<0,Dune::Interior_Partition>();

    // Loop over all elements
    for (; it!=endIt; ++it) {

        // Number of degrees of freedom on this element
        size_t nDofs = basis_.getLocalFiniteElement(*it).localBasis().size();

        VectorType localSolution(nDofs);

        for (size_t i=0; i<nDofs; i++)
            localSolution[i]   = sol[basis_.index(*it,i)];

        energy += localStiffness_->energy(*it, basis_.getLocalFiniteElement(*it), localSolution);

    }

    return energy;

}



#endif

