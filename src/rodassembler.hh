#ifndef DUNE_EXTENSIBLE_ROD_ASSEMBLER_HH
#define DUNE_EXTENSIBLE_ROD_ASSEMBLER_HH

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>
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

            K1 = E * J1;
            K2 = E * J2;
            K3 = G * (J1 + J2);

            A1 = G * A;
            A2 = G * A;
            A3 = E * A;

            printf("%g %g %g   %g %g %g\n", K1, K2, K3, A1, A2, A3);
            //exit(0);
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

        void getStrain(const std::vector<Configuration>& sol, 
                       BlockVector<FieldVector<double, blocksize> >& strain) const;
        
    protected:

        /** \brief Compute the element tangent stiffness matrix  */
        template <class MatrixType>
        void getLocalMatrix( EntityType &entity, 
                             const std::vector<Configuration>& localSolution, 
                             const int matSize, MatrixType& mat) const;

        template <class T>
        static Quaternion<T> B(int m, const Quaternion<T>& q) {
            assert(m>=0 && m<3);
            Quaternion<T> r;
            if (m==0) {
                r[0] =  q[3];
                r[1] =  q[2];
                r[2] = -q[1];
                r[3] = -q[0];
            } else if (m==1) {
                r[0] = -q[2];
                r[1] =  q[3];
                r[2] =  q[0];
                r[3] = -q[1];
            } else {
                r[0] =  q[1];
                r[1] = -q[0];
                r[2] =  q[3];
                r[3] = -q[2];
            } 

            return r;
        }
        
        template <class T>
        static FieldVector<T,3> darboux(const Quaternion<T>& q, const FieldVector<T,4>& q_s) 
        {
            FieldVector<double,3> u;  // The Darboux vector
            u[0] = 2 * ( q[3]*q_s[0] + q[2]*q_s[1] - q[1]*q_s[2] - q[0]*q_s[3]);
            u[1] = 2 * (-q[2]*q_s[0] + q[3]*q_s[1] + q[0]*q_s[2] - q[1]*q_s[3]);
            u[2] = 2 * ( q[1]*q_s[0] - q[0]*q_s[1] + q[3]*q_s[2] - q[2]*q_s[3]);

            return u;
        }
        
        
    }; // end class
    
} // end namespace 

#include "rodassembler.cc"

#endif

