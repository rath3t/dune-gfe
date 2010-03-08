#ifndef DUNE_EXTENSIBLE_ROD_ASSEMBLER_HH
#define DUNE_EXTENSIBLE_ROD_ASSEMBLER_HH

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>

#include <dune/ag-common/boundarypatch.hh>

#include "rigidbodymotion.hh"
#include "rodlocalstiffness.hh"
#include "geodesicfeassembler.hh"

/** \brief The FEM operator for an extensible, shearable rod
 */
template <class GridView>
class RodAssembler : public GeodesicFEAssembler<GridView, RigidBodyMotion<3> >
{
        
    //typedef typename GridType::template Codim<0>::Entity EntityType;
    //typedef typename GridType::template Codim<0>::EntityPointer EntityPointer;
    typedef typename GridView::template Codim<0>::Iterator ElementIterator;

        //! Dimension of the grid.  This needs to be one!
        enum { gridDim = GridView::dimension };

        enum { elementOrder = 1};

        //! Each block is x, y, theta in 2d, T (R^3 \times SO(3)) in 3d
        enum { blocksize = 6 };
        
        //!
        typedef Dune::FieldMatrix<double, blocksize, blocksize> MatrixBlock;
        
public:
        //! ???
    RodAssembler(const GridView &gridView,
                 RodLocalStiffness<GridView,double>* localStiffness) 
        : GeodesicFEAssembler<GridView, RigidBodyMotion<3> >(gridView,localStiffness)
        { 
            std::vector<RigidBodyMotion<3> > referenceConfiguration(gridView.size(gridDim));

            typename GridView::template Codim<gridDim>::Iterator it    = gridView.template begin<gridDim>();
            typename GridView::template Codim<gridDim>::Iterator endIt = gridView.template end<gridDim>();

            for (; it != endIt; ++it) {

                int idx = gridView.indexSet().index(*it);

                referenceConfiguration[idx].r[0] = 0;
                referenceConfiguration[idx].r[1] = 0;
                referenceConfiguration[idx].r[2] = it->geometry().corner(0)[0];
                referenceConfiguration[idx].q = Rotation<3,double>::identity();
            }

            dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->setReferenceConfiguration(referenceConfiguration);
        }

        void assembleGradient(const std::vector<RigidBodyMotion<3> >& sol,
                              Dune::BlockVector<Dune::FieldVector<double, blocksize> >& grad) const;

        void getStrain(const std::vector<RigidBodyMotion<3> >& sol, 
                       Dune::BlockVector<Dune::FieldVector<double, blocksize> >& strain) const;

        void getStress(const std::vector<RigidBodyMotion<3> >& sol, 
                       Dune::BlockVector<Dune::FieldVector<double, blocksize> >& stress) const;

        /** \brief Return resultant force across boundary in canonical coordinates 

        \note Linear run-time in the size of the grid */
        template <class PatchGridView>
        Dune::FieldVector<double,3> getResultantForce(const BoundaryPatchBase<PatchGridView>& boundary,
                                                      const std::vector<RigidBodyMotion<3> >& sol,
                                                      Dune::FieldVector<double,3>& canonicalTorque) const;

    }; // end class


/** \brief The FEM operator for a 2D extensible, shearable rod
 */
template <class GridType>
class PlanarRodAssembler : public GeodesicFEAssembler<typename GridType::LeafGridView, RigidBodyMotion<2> >
{
    
    typedef typename GridType::template Codim<0>::Entity EntityType;
    typedef typename GridType::template Codim<0>::LevelIterator ElementIterator;
    
    //! Dimension of the grid.  This needs to be one!
    enum { gridDim = GridType::dimension };
    
    enum { elementOrder = 1};
    
    //! Each block is x, y, theta
    enum { blocksize = 3 };
    
    //!
    typedef Dune::FieldMatrix<double, blocksize, blocksize> MatrixBlock;
    
    const GridType* grid_; 
    
    /** \brief Material constants */
    double B;
    double A1;
    double A3;
    
public:
    
    //! ???
    PlanarRodAssembler(const GridType &grid) 
        : GeodesicFEAssembler<typename GridType::LeafGridView, RigidBodyMotion<2> >(grid.leafView(),NULL), 
        grid_(&grid)
    { 
        B = 1;
        A1 = 1;
        A3 = 1;
    }
    
    ~PlanarRodAssembler() {}
    
    void setParameters(double b, double a1, double a3) {
        B  = b;
        A1 = a1;
        A3 = a3;
    }
    
    /** \brief Assemble the tangent stiffness matrix and the right hand side
     */
    void assembleMatrix(const std::vector<RigidBodyMotion<2> >& sol,
                        Dune::BCRSMatrix<MatrixBlock>& matrix);
    
    void assembleGradient(const std::vector<RigidBodyMotion<2> >& sol,
                          Dune::BlockVector<Dune::FieldVector<double, blocksize> >& grad) const;
    
    /** \brief Compute the energy of a deformation state */
    double computeEnergy(const std::vector<RigidBodyMotion<2> >& sol) const;
    
    void getNeighborsPerVertex(Dune::MatrixIndexSet& nb) const;
    
protected:
    
    /** \brief Compute the element tangent stiffness matrix  */
    template <class MatrixType>
    void getLocalMatrix( EntityType &entity, 
                         const std::vector<RigidBodyMotion<2> >& localSolution, 
                         const int matSize, MatrixType& mat) const;
    
}; // end class

#include "rodassembler.cc"

#endif

