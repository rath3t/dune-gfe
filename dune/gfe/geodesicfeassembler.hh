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
    typedef typename GridView::template Codim<0>::Iterator ElementIterator;

    //! Dimension of the grid.
    enum { gridDim = GridView::dimension };
    
    //! Dimension of a tangent space
    enum { blocksize = TargetSpace::TangentVector::dimension };
    
    //!
    typedef Dune::FieldMatrix<double, blocksize, blocksize> MatrixBlock;

protected:

    const Basis basis_; 

    LocalGeodesicFEStiffness<GridView,
                             typename Basis::LocalFiniteElement,
                             TargetSpace>* localStiffness_;

public:
    
    /** \brief Constructor for a given grid */
    GeodesicFEAssembler(const Basis& basis,
                        LocalGeodesicFEStiffness<GridView,typename Basis::LocalFiniteElement, TargetSpace>* localStiffness)
        : basis_(basis),
          localStiffness_(localStiffness)
    {}
    
    /** \brief Assemble the tangent stiffness matrix
     */
    virtual void assembleMatrix(const std::vector<TargetSpace>& sol,
                                Dune::BCRSMatrix<MatrixBlock>& matrix,
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
    int n = basis_.size();
    
    nb.resize(n, n);
    
    ElementIterator it    = basis_.getGridView().template begin<0>();
    ElementIterator endit = basis_.getGridView().template end<0>  ();
    
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

template <class Basis, class TargetSpace>
void GeodesicFEAssembler<Basis,TargetSpace>::
assembleMatrix(const std::vector<TargetSpace>& sol,
               Dune::BCRSMatrix<MatrixBlock>& matrix,
               bool computeOccupationPattern) const
{
    if (computeOccupationPattern) {

        Dune::MatrixIndexSet neighborsPerVertex;
        getNeighborsPerVertex(neighborsPerVertex);
        neighborsPerVertex.exportIdx(matrix);

    }

    matrix = 0;
    
    ElementIterator it    = basis_.getGridView().template begin<0>();
    ElementIterator endit = basis_.getGridView().template end<0>  ();

    for( ; it != endit; ++it ) {
        
        const int numOfBaseFct = basis_.getLocalFiniteElement(*it).localBasis().size();
        
        // Extract local solution
        std::vector<TargetSpace> localSolution(numOfBaseFct);
        
        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[basis_.index(*it,i)];

        // setup matrix 
        localStiffness_->assembleHessian(*it, basis_.getLocalFiniteElement(*it), localSolution);

        // Add element matrix to global stiffness matrix
        for(int i=0; i<numOfBaseFct; i++) { 
            
            int row = basis_.index(*it,i);

            for (int j=0; j<numOfBaseFct; j++ ) {
                
                int col = basis_.index(*it,j);
                matrix[row][col] += localStiffness_->A_[i][j];
                
            }
        }

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

    ElementIterator it    = basis_.getGridView().template begin<0>();
    ElementIterator endIt = basis_.getGridView().template end<0>();

    // Loop over all elements
    for (; it!=endIt; ++it) {

        // A 1d grid has two vertices
        const int nDofs = basis_.getLocalFiniteElement(*it).localBasis().size();

        // Extract local solution
        std::vector<TargetSpace> localSolution(nDofs);
        
        for (int i=0; i<nDofs; i++)
            localSolution[i] = sol[basis_.index(*it,i)];

        // Assemble local gradient
        std::vector<Dune::FieldVector<double,blocksize> > localGradient(nDofs);

        localStiffness_->assembleGradient(*it, basis_.getLocalFiniteElement(*it), localSolution, localGradient);

        // Add to global gradient
        for (int i=0; i<nDofs; i++)
            grad[basis_.index(*it,i)] += localGradient[i];

    }

}


template <class Basis, class TargetSpace>
double GeodesicFEAssembler<Basis, TargetSpace>::
computeEnergy(const std::vector<TargetSpace>& sol) const
{
    double energy = 0;
    
    if (sol.size()!=basis_.size())
        DUNE_THROW(Dune::Exception, "Solution vector doesn't match the grid!");

    ElementIterator it    = basis_.getGridView().template begin<0>();
    ElementIterator endIt = basis_.getGridView().template end<0>();

    // Loop over all elements
    for (; it!=endIt; ++it) {
        
        // Number of degrees of freedom on this element
        size_t nDofs = basis_.getLocalFiniteElement(*it).localBasis().size();

        std::vector<TargetSpace> localSolution(nDofs);

        for (size_t i=0; i<nDofs; i++)
            localSolution[i] = sol[basis_.index(*it,i)];

        energy += localStiffness_->energy(*it, basis_.getLocalFiniteElement(*it), localSolution);

    }

    return energy;

}



#endif

