#ifndef DUNE_EXTENSIBLE_ROD_ASSEMBLER_HH
#define DUNE_EXTENSIBLE_ROD_ASSEMBLER_HH

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/common/matrix.hh>

namespace Dune 
{

    /** \brief The FEM operator for an extensible, shearable rod
     */
    template <class FunctionSpaceType, int polOrd>
    class RodAssembler {
        
        //! The grid
        typedef typename FunctionSpaceType::GridType GridType;
        
        typedef typename GridType::template Codim<0>::Entity EntityType;
        typedef typename GridType::template Codim<0>::LevelIterator ElementIterator;
        typedef typename FunctionSpaceType::BaseFunctionSetType BaseFunctionSetType;
    

        //! Dimension of the grid.  This needs to be one!
        enum { gridDim = GridType::dimension };

        //! Each block is x, y, theta
        enum { blocksize = 3 };
        
        //!
        typedef FieldMatrix<double, blocksize, blocksize> MatrixBlock;
        
        //! ???
        typedef typename FunctionSpaceType::JacobianRange JacobianRange;
        
        //! ???
        typedef typename FunctionSpaceType::RangeField RangeFieldType;
        typedef typename FunctionSpaceType::Range       RangeType;
        
    public:
        
        /** \todo Does actually belong into the base class */
        const GridType* grid_; 
        
        /** \todo Does actually belong into the base class */
        const FunctionSpaceType& functionSpace_;

        /** \brief Material constants */
        double B;
        double A1;
        double A3;

        //! ???
        RodAssembler(const FunctionSpaceType &f) : 
            functionSpace_(f)
        { 
            grid_ = &f.getGrid();
            B = 1;
            A1 = 1;
            A3 = 1;
        }

        ~RodAssembler() {}

        void getNeighborsPerVertex(MatrixIndexSet& nb) const;
        
        /** \brief Assemble the tangent stiffness matrix and the right hand side
         */
        void assembleMatrix(const BlockVector<FieldVector<double, blocksize> >& sol,
                            BCRSMatrix<MatrixBlock>& matrix);
        
        /** \brief Compute the element tangent stiffness matrix  */
        template <class MatrixType>
        void getLocalMatrix( EntityType &entity, 
                             const BlockVector<FieldVector<double, blocksize> >& localSolution, 
                             const int matSize, MatrixType& mat) const;

        void assembleGradient(const BlockVector<FieldVector<double, blocksize> >& sol,
                              BlockVector<FieldVector<double, blocksize> >& grad) const;

        /** \brief Compute the energy of a deformation state */
        double computeEnergy(const BlockVector<FieldVector<double, blocksize> >& sol) const;
        
        
        
        
        
        
    }; // end class
    
} // end namespace 

#include "rodassembler.cc"

#endif

