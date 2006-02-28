#ifndef DUNE_EXTENSIBLE_ROD_ASSEMBLER_HH
#define DUNE_EXTENSIBLE_ROD_ASSEMBLER_HH

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/common/matrix.hh>
#include "configuration.hh"

namespace Dune 
{

    /** \brief The FEM operator for an extensible, shearable rod
     */
    template <class GridType>
    class RodAssembler {
        
        typedef typename GridType::template Codim<0>::Entity EntityType;
        typedef typename GridType::template Codim<0>::LevelIterator ElementIterator;
        typedef typename GridType::template Codim<0>::LeafIterator ElementLeafIterator;

        //! Dimension of the grid.  This needs to be one!
        enum { gridDim = GridType::dimension };

        enum { elementOrder = 1};

        //! Each block is x, y, theta
        enum { blocksize = 6 };
        
        //!
        typedef FieldMatrix<double, blocksize, blocksize> MatrixBlock;
        
        const GridType* grid_; 
        
        /** \brief Material constants */
        double K1, K2, K3;
        double A1, A2, A3;

    public:
        
        //! ???
        RodAssembler(const GridType &grid) : 
            grid_(&grid)
        { 
            K1 = K2 = K3 = 1;
            A1 = A2 = A3 = 1;
        }

        ~RodAssembler() {}

        void setParameters(double k1, double k2, double k3, 
                           double a1, double a2, double a3) {
            K1 = k1;
            K2 = k2;
            K3 = k3;
            A1 = a1;
            A2 = a2;
            A3 = a3;
        }


        /** \brief Assemble the tangent stiffness matrix and the right hand side
         */
        void assembleMatrix(const std::vector<Configuration>& sol,
                            BCRSMatrix<MatrixBlock>& matrix);
        
        void assembleGradient(const std::vector<Configuration>& sol,
                              BlockVector<FieldVector<double, blocksize> >& grad) const;

        /** \brief Compute the energy of a deformation state */
        double computeEnergy(const std::vector<Configuration>& sol) const;

        void getNeighborsPerVertex(MatrixIndexSet& nb) const;
        
    protected:

        /** \brief Compute the element tangent stiffness matrix  */
        template <class MatrixType>
        void getLocalMatrix( EntityType &entity, 
                             const std::vector<Configuration>& localSolution, 
                             const int matSize, MatrixType& mat) const;

        
        
        
        
        
        
    }; // end class
    
} // end namespace 

#include "rodassembler.cc"

#endif

