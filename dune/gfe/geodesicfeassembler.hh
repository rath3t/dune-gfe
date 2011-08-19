#ifndef GLOBAL_GEODESIC_FE_ASSEMBLER_HH
#define GLOBAL_GEODESIC_FE_ASSEMBLER_HH

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>

#include "localgeodesicfestiffness.hh"


/** \brief A global FE assembler for problems involving functions that map into non-Euclidean spaces 
 */
template <class GridView, class TargetSpace>
class GeodesicFEAssembler {
    
    typedef typename GridView::template Codim<0>::Entity EntityType;
    typedef typename GridView::template Codim<0>::EntityPointer EntityPointer;
    typedef typename GridView::template Codim<0>::Iterator ElementIterator;
    
    //! Dimension of the grid.
    enum { gridDim = GridView::dimension };
    
    //! Dimension of a tangent space
    enum { blocksize = TargetSpace::TangentVector::dimension };
    
    //!
    typedef Dune::FieldMatrix<double, blocksize, blocksize> MatrixBlock;
    
protected:

    const GridView gridView_; 

    LocalGeodesicFEStiffness<GridView,TargetSpace>* localStiffness_;

public:
    
    /** \brief Constructor for a given grid */
    GeodesicFEAssembler(const GridView& gridView,
                        LocalGeodesicFEStiffness<GridView,TargetSpace>* localStiffness)
        : gridView_(gridView),
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



template <class GridView, class TargetSpace>
void GeodesicFEAssembler<GridView,TargetSpace>::
getNeighborsPerVertex(Dune::MatrixIndexSet& nb) const
{
    const typename GridView::IndexSet& indexSet = gridView_.indexSet();
    
    int n = gridView_.size(gridDim);
    
    nb.resize(n, n);
    
    ElementIterator it    = gridView_.template begin<0>();
    ElementIterator endit = gridView_.template end<0>  ();
    
    for (; it!=endit; ++it) {
        
        for (int i=0; i<it->template count<gridDim>(); i++) {
            
            for (int j=0; j<it->template count<gridDim>(); j++) {
                
                int iIdx = indexSet.subIndex(*it,i,gridDim);
                int jIdx = indexSet.subIndex(*it,j,gridDim);
                
                nb.add(iIdx, jIdx);
                
            }
            
        }
        
    }
    
}

template <class GridView, class TargetSpace>
void GeodesicFEAssembler<GridView,TargetSpace>::
assembleMatrix(const std::vector<TargetSpace>& sol,
               Dune::BCRSMatrix<MatrixBlock>& matrix,
               bool computeOccupationPattern) const
{
    const typename GridView::IndexSet& indexSet = gridView_.indexSet();

    if (computeOccupationPattern) {

        Dune::MatrixIndexSet neighborsPerVertex;
        getNeighborsPerVertex(neighborsPerVertex);
        neighborsPerVertex.exportIdx(matrix);

    }

    matrix = 0;
    
    ElementIterator it    = gridView_.template begin<0>();
    ElementIterator endit = gridView_.template end<0>  ();

    for( ; it != endit; ++it ) {
        
        const int numOfBaseFct = it->template count<gridDim>();  
        
        // Extract local solution
        Dune::array<TargetSpace,gridDim+1> localSolution;
        
        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[indexSet.subIndex(*it,i,gridDim)];

        // setup matrix 
        localStiffness_->assembleHessian(*it, localSolution);

        // Add element matrix to global stiffness matrix
        for(int i=0; i<numOfBaseFct; i++) { 
            
            int row = indexSet.subIndex(*it,i,gridDim);

            for (int j=0; j<numOfBaseFct; j++ ) {
                
                int col = indexSet.subIndex(*it,j,gridDim);
                matrix[row][col] += localStiffness_->A_[i][j];
                
            }
        }

    }

}

template <class GridView, class TargetSpace>
void GeodesicFEAssembler<GridView,TargetSpace>::
assembleGradient(const std::vector<TargetSpace>& sol,
                 Dune::BlockVector<Dune::FieldVector<double, blocksize> >& grad) const
{
    const typename GridView::IndexSet& indexSet = gridView_.indexSet();

    if (sol.size()!=gridView_.size(gridDim))
        DUNE_THROW(Dune::Exception, "Solution vector doesn't match the grid!");

    grad.resize(sol.size());
    grad = 0;

    ElementIterator it    = gridView_.template begin<0>();
    ElementIterator endIt = gridView_.template end<0>();

    // Loop over all elements
    for (; it!=endIt; ++it) {

        // A 1d grid has two vertices
        const int nDofs = it->template count<gridDim>();

        // Extract local solution
        Dune::array<TargetSpace,gridDim+1> localSolution;
        
        for (int i=0; i<nDofs; i++)
            localSolution[i] = sol[indexSet.subIndex(*it,i,gridDim)];

        // Assemble local gradient
        std::vector<Dune::FieldVector<double,blocksize> > localGradient(nDofs);

        localStiffness_->assembleGradient(*it, localSolution, localGradient);

        // Add to global gradient
        for (int i=0; i<nDofs; i++)
            grad[indexSet.subIndex(*it,i,gridDim)] += localGradient[i];

    }

}


template <class GridView, class TargetSpace>
double GeodesicFEAssembler<GridView, TargetSpace>::
computeEnergy(const std::vector<TargetSpace>& sol) const
{
    double energy = 0;
    
    const typename GridView::IndexSet& indexSet = gridView_.indexSet();

    if (sol.size()!=indexSet.size(gridDim))
        DUNE_THROW(Dune::Exception, "Solution vector doesn't match the grid!");

    Dune::array<TargetSpace,gridDim+1> localSolution;

    ElementIterator it    = gridView_.template begin<0>();
    ElementIterator endIt = gridView_.template end<0>();

    // Loop over all elements
    for (; it!=endIt; ++it) {

        for (int i=0; i<it->template count<gridDim>(); i++)
            localSolution[i]               = sol[indexSet.subIndex(*it,i,gridDim)];

        energy += localStiffness_->energy(*it, localSolution);

    }

    return energy;

}



#endif

