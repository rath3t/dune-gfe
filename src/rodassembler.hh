#ifndef DUNE_EXTENSIBLE_ROD_ASSEMBLER_HH
#define DUNE_EXTENSIBLE_ROD_ASSEMBLER_HH

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>
#include <dune/disc/operators/localstiffness.hh>

#include <dune/ag-common/boundarypatch.hh>
#include "configuration.hh"


    /** \brief The FEM operator for an extensible, shearable rod
     */
    template <class GridType>
    class RodAssembler {
        
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
        
        /** \brief Material constants */
        Dune::array<double,3> K_;
        Dune::array<double,3> A_;

        /** \brief The stress-free configuration */
        std::vector<Configuration> referenceConfiguration_;

        /** \todo Only for the fd approximations */
        static void infinitesimalVariation(Configuration& c, double eps, int i)
        {
            if (i<3)
                c.r[i] += eps;
            else
                c.q = c.q.mult(Rotation<3,double>::exp((i==3)*eps, 
                                                       (i==4)*eps, 
                                                       (i==5)*eps));
        }
    
    public:
        
        //! ???
        RodAssembler(const GridType &grid) : 
            grid_(&grid)
        { 
            // Set dummy material parameters
            K_[0] = K_[1] = K_[2] = 1;
            A_[0] = A_[1] = A_[2] = 1;

            referenceConfiguration_.resize(grid.size(gridDim));

            typename GridType::template Codim<gridDim>::LeafIterator it    = grid.template leafbegin<gridDim>();
            typename GridType::template Codim<gridDim>::LeafIterator endIt = grid.template leafend<gridDim>();

            for (; it != endIt; ++it) {

                int idx = grid.leafIndexSet().index(*it);

                referenceConfiguration_[idx].r[0] = 0;
                referenceConfiguration_[idx].r[1] = 0;
                referenceConfiguration_[idx].r[2] = it->geometry().corner(0)[0];
                referenceConfiguration_[idx].q = Rotation<3,double>::identity();
            }

        }

        ~RodAssembler() {}

        void setParameters(double k1, double k2, double k3, 
                           double a1, double a2, double a3) {
            K_[0] = k1;
            K_[1] = k2;
            K_[2] = k3;
            A_[0] = a1;
            A_[1] = a2;
            A_[2] = a3;
        }

        /** \brief Set shape constants and material parameters
            \param A The rod section area
            \param J1, J2 The geometric moments (Flächenträgheitsmomente)
            \param E Young's modulus
            \param nu Poisson number
        */
        void setShapeAndMaterial(double A, double J1, double J2, double E, double nu) 
        {
            // shear modulus
            double G = E/(2+2*nu);

            K_[0] = E * J1;
            K_[1] = E * J2;
            K_[2] = G * (J1 + J2);

            A_[0] = G * A;
            A_[1] = G * A;
            A_[2] = E * A;

            //printf("%g %g %g   %g %g %g\n", K_[0], K_[1], K_[2], A_[0], A_[1], A_[2]);
            //exit(0);
        }

        /** \brief Assemble the tangent stiffness matrix
         */
        void assembleMatrix(const std::vector<Configuration>& sol,
                            Dune::BCRSMatrix<MatrixBlock>& matrix) const;

        /** \brief Assemble the tangent stiffness matrix using a finite difference approximation
         */
        void assembleMatrixFD(const std::vector<Configuration>& sol,
                              Dune::BCRSMatrix<MatrixBlock>& matrix) const;

        void assembleGradient(const std::vector<Configuration>& sol,
                              Dune::BlockVector<Dune::FieldVector<double, blocksize> >& grad) const;

        /** \brief Compute the energy of a deformation state */
        double computeEnergy(const std::vector<Configuration>& sol) const;

        void getNeighborsPerVertex(Dune::MatrixIndexSet& nb) const;

        void getStrain(const std::vector<Configuration>& sol, 
                       Dune::BlockVector<Dune::FieldVector<double, blocksize> >& strain) const;

        void getStress(const std::vector<Configuration>& sol, 
                       Dune::BlockVector<Dune::FieldVector<double, blocksize> >& stress) const;

        /** \brief Return resultant force across boundary in canonical coordinates 

        \note Linear run-time in the size of the grid */
        Dune::FieldVector<double,3> getResultantForce(const BoundaryPatch<GridType>& boundary, 
                                                      const std::vector<Configuration>& sol,
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

