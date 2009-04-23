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
template <class GridType>
class RodAssembler : public GeodesicFEAssembler<typename GridType::LeafGridView, RigidBodyMotion<3> >
{
        
        typedef typename GridType::template Codim<0>::Entity EntityType;
        typedef typename GridType::template Codim<0>::EntityPointer EntityPointer;
        typedef typename GridType::template Codim<0>::LevelIterator ElementIterator;
        typedef typename GridType::template Codim<0>::LeafIterator ElementLeafIterator;

        //! Dimension of the grid.  This needs to be one!
        enum { gridDim = GridType::dimension };

        enum { elementOrder = 1};

        //! Each block is x, y, theta in 2d, T (R^3 \times SO(3)) in 3d
        enum { blocksize = 6 };
        
        //!
        typedef Dune::FieldMatrix<double, blocksize, blocksize> MatrixBlock;
        
        /** \todo public only for debugging! */
    public:
        const GridType* grid_; 
        
    public:
        
        //! ???
    RodAssembler(const GridType &grid,
                 RodLocalStiffness<typename GridType::LeafGridView,double>* localStiffness) : 
        GeodesicFEAssembler<typename GridType::LeafGridView, RigidBodyMotion<3> >(grid.leafView(),
                                                                                  localStiffness),
        grid_(&grid)
        { 
            std::vector<RigidBodyMotion<3> > referenceConfiguration(grid.size(gridDim));

            typename GridType::template Codim<gridDim>::LeafIterator it    = grid.template leafbegin<gridDim>();
            typename GridType::template Codim<gridDim>::LeafIterator endIt = grid.template leafend<gridDim>();

            for (; it != endIt; ++it) {

                int idx = grid.leafIndexSet().index(*it);

                referenceConfiguration[idx].r[0] = 0;
                referenceConfiguration[idx].r[1] = 0;
                referenceConfiguration[idx].r[2] = it->geometry().corner(0)[0];
                referenceConfiguration[idx].q = Rotation<3,double>::identity();
            }

            dynamic_cast<RodLocalStiffness<typename GridType::LeafGridView, double>* >(this->localStiffness_)->setReferenceConfiguration(referenceConfiguration);
        }

        /** \brief Assemble the tangent stiffness matrix
         */
        void assembleMatrix(const std::vector<RigidBodyMotion<3> >& sol,
                            Dune::BCRSMatrix<MatrixBlock>& matrix) const;

        void assembleGradient(const std::vector<RigidBodyMotion<3> >& sol,
                              Dune::BlockVector<Dune::FieldVector<double, blocksize> >& grad) const;

        /** \brief Compute the energy of a deformation state */
        double computeEnergy(const std::vector<RigidBodyMotion<3> >& sol) const;

        void getNeighborsPerVertex(Dune::MatrixIndexSet& nb) const;

        void getStrain(const std::vector<RigidBodyMotion<3> >& sol, 
                       Dune::BlockVector<Dune::FieldVector<double, blocksize> >& strain) const;

        void getStress(const std::vector<RigidBodyMotion<3> >& sol, 
                       Dune::BlockVector<Dune::FieldVector<double, blocksize> >& stress) const;

        /** \brief Return resultant force across boundary in canonical coordinates 

        \note Linear run-time in the size of the grid */
        Dune::FieldVector<double,3> getResultantForce(const BoundaryPatch<GridType>& boundary, 
                                                      const std::vector<RigidBodyMotion<3> >& sol,
                                                      Dune::FieldVector<double,3>& canonicalTorque) const;

    protected:

        template <class T>
        static Dune::FieldVector<T,3> darboux(const Quaternion<T>& q, const Dune::FieldVector<T,4>& q_s) 
        {
            Dune::FieldVector<double,3> u;  // The Darboux vector

            u[0] = 2 * (q.B(0) * q_s);
            u[1] = 2 * (q.B(1) * q_s);
            u[2] = 2 * (q.B(2) * q_s);

            return u;
        }
        
    }; // end class

#include "rodassembler.cc"

#endif

