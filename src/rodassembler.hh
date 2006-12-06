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
        typedef typename GridType::template Codim<0>::EntityPointer EntityPointer;
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
        double K_[3];
        double A_[3];

        /** \brief The stress-free configuration */
        std::vector<Configuration> referenceConfiguration_;

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
                referenceConfiguration_[idx].r[2] = it->geometry()[0][0];
                referenceConfiguration_[idx].q = Quaternion<double>::identity();
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

            printf("%g %g %g   %g %g %g\n", K_[0], K_[1], K_[2], A_[0], A_[1], A_[2]);
            //exit(0);
        }

        /** \brief Assemble the tangent stiffness matrix and the right hand side
         */
        void assembleMatrix(const std::vector<Configuration>& sol,
                            BCRSMatrix<MatrixBlock>& matrix) const;
        
        void assembleGradient(const std::vector<Configuration>& sol,
                              BlockVector<FieldVector<double, blocksize> >& grad) const;

        /** \brief Compute the energy of a deformation state */
        double computeEnergy(const std::vector<Configuration>& sol) const;

        void getNeighborsPerVertex(MatrixIndexSet& nb) const;

        void getStrain(const std::vector<Configuration>& sol, 
                       BlockVector<FieldVector<double, blocksize> >& strain) const;

        /** \brief Get the strain at a particular point of the grid */
        FieldVector<double, 6> getStrain(const std::vector<Configuration>& sol,
                                                 const EntityPointer& element,
                                                 double pos) const;
                       
        
        /** \brief Return resultant force in canonical coordinates */
        FieldVector<double,3> getResultantForce(const std::vector<Configuration>& sol) const;

    protected:

        /** \brief Compute the element tangent stiffness matrix  
            \todo Handing over both the local and the global solution is pretty stupid. */
        template <class MatrixType>
        void getLocalMatrix( EntityPointer &entity, 
                             const std::vector<Configuration>& localSolution, 
                             const std::vector<Configuration>& globalSolution, 
                             const int matSize, MatrixType& mat) const;


        
        template <class T>
        static FieldVector<T,3> darboux(const Quaternion<T>& q, const FieldVector<T,4>& q_s) 
        {
            FieldVector<double,3> uCanonical = darbouxCanonical(q, q_s);  // The Darboux vector

            FieldVector<double,3> u;
            u[0] = uCanonical*q.director(0);
            u[1] = uCanonical*q.director(1);
            u[2] = uCanonical*q.director(2);
            return u;
        }

        template <class T>
        static FieldVector<T,3> darbouxCanonical(const Quaternion<T>& q, const FieldVector<T,4>& q_s) 
        {
            FieldVector<double,3> uCanonical;  // The Darboux vector
//             uCanonical[0] = 2 * ( q[3]*q_s[0] + q[2]*q_s[1] - q[1]*q_s[2] - q[0]*q_s[3]);
//             uCanonical[1] = 2 * (-q[2]*q_s[0] + q[3]*q_s[1] + q[0]*q_s[2] - q[1]*q_s[3]);
//             uCanonical[2] = 2 * ( q[1]*q_s[0] - q[0]*q_s[1] + q[3]*q_s[2] - q[2]*q_s[3]);

            uCanonical[0] = 2 * (q.B(0) * q_s);
            uCanonical[1] = 2 * (q.B(1) * q_s);
            uCanonical[2] = 2 * (q.B(2) * q_s);

            return uCanonical;
        }
        
        static void getFirstDerivativesOfDirectors(const Quaternion<double>& q, 
                                                   Dune::FixedArray<Dune::FixedArray<Dune::FixedArray<Dune::FieldVector<double,3>, 3>, 2>, 3>& dd_dvij,
                                                   const Dune::FixedArray<Dune::FixedArray<Quaternion<double>, 3>, 2>& dq_dvij);

    }; // end class
    
} // end namespace 

#include "rodassembler.cc"

#endif

