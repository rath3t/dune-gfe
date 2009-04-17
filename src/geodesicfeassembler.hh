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
    enum { blocksize = TargetSpace::TangentVector::size };
    
    //!
    typedef Dune::FieldMatrix<double, blocksize, blocksize> MatrixBlock;
    
    const GridView gridView_; 

    const LocalGeodesicFEStiffness<GridView,TargetSpace>* localStiffness_;

public:
    
    /** \brief Constructor for a given grid */
    GeodesicFEAssembler(const GridView& gridView) : 
        gridView_(gridView)
    {}
    
    /** \brief Assemble the tangent stiffness matrix
     */
    void assembleMatrix(const std::vector<TargetSpace>& sol,
                        Dune::BCRSMatrix<MatrixBlock>& matrix) const;
    
    /** \brief Assemble the gradient */
    void assembleGradient(const std::vector<TargetSpace>& sol,
                          Dune::BlockVector<Dune::FieldVector<double, blocksize> >& grad) const;

    /** \brief Assemble the gradient using a finite difference approximation */
    void assembleGradientFD(const std::vector<TargetSpace>& sol,
                          Dune::BlockVector<Dune::FieldVector<double, blocksize> >& grad) const;
    
    /** \brief Compute the energy of a deformation state */
    double computeEnergy(const std::vector<TargetSpace>& sol) const;
    
protected:
    void getNeighborsPerVertex(Dune::MatrixIndexSet& nb) const;
    
}; // end class



template <class GridView, class TargetSpace>
void GeodesicFEAssembler<GridView,TargetSpace>::
getNeighborsPerVertex(Dune::MatrixIndexSet& nb) const
{
    const typename GridView::IndexSet& indexSet = gridView_->indexSet();
    
    int n = gridView_->size(gridDim);
    
    nb.resize(n, n);
    
    ElementIterator it    = gridView_->template begin<0>();
    ElementIterator endit = gridView_->template end<0>  ();
    
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
double GeodesicFEAssembler<GridView, TargetSpace>::
computeEnergy(const std::vector<TargetSpace>& sol) const
{
    double energy = 0;
    
    const typename GridView::IndexSet& indexSet = gridView_.indexSet();

    if (sol.size()!=indexSet.size(gridDim))
        DUNE_THROW(Dune::Exception, "Solution vector doesn't match the grid!");

    std::vector<TargetSpace> localSolution;

    ElementIterator it    = gridView_.template begin<0>();
    ElementIterator endIt = gridView_.template end<0>();

    // Loop over all elements
    for (; it!=endIt; ++it) {

        localSolution.resize(it->template count<gridDim>());

        for (int i=0; i<it->template count<gridDim>(); i++)
            localSolution[i]               = sol[indexSet.subIndex(*it,i,gridDim)];

        energy += localStiffness_->energy(*it, localSolution);

    }

    return energy;

}



#endif

