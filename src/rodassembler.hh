#ifndef DUNE_EXTENSIBLE_ROD_ASSEMBLER_HH
#define DUNE_EXTENSIBLE_ROD_ASSEMBLER_HH

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>
#include <dune/disc/operators/localstiffness.hh>

#include "../../common/boundarypatch.hh"
#include "configuration.hh"

template<class GridType, class RT>
class RodLocalStiffness 
    : public Dune::LocalStiffness<RodLocalStiffness<GridType,RT>,GridType,RT,6>
{
    // grid types
    typedef typename GridType::ctype DT;
    typedef typename GridType::template Codim<0>::Entity Entity;
    typedef typename GridType::template Codim<0>::EntityPointer EntityPointer;
    
    // some other sizes
    enum {dim=GridType::dimension};

    // Quadrature order used for the extension and shear energy
    enum {shearQuadOrder = 2};

    // Quadrature order used for the bending and torsion energy
    enum {bendingQuadOrder = 2};

public:
    
    //! Each block is x, y, theta in 2d, T (R^3 \times SO(3)) in 3d
    enum { blocksize = 6 };

    // define the number of components of your system, this is used outside
    // to allocate the correct size of (dense) blocks with a FieldMatrix
    enum {m=blocksize};

    enum {SIZE = Dune::LocalStiffness<RodLocalStiffness,GridType,RT,m>::SIZE};
    
    // types for matrics, vectors and boundary conditions
    typedef Dune::FieldMatrix<RT,m,m> MBlockType; // one entry in the stiffness matrix
    typedef Dune::FieldVector<RT,m> VBlockType;   // one entry in the global vectors
    typedef Dune::array<Dune::BoundaryConditions::Flags,m> BCBlockType;     // componentwise boundary conditions

    // /////////////////////////////////
    //   The material parameters
    // /////////////////////////////////
    
    /** \brief Material constants */
    double K_[3];
    double A_[3];

    //! Default Constructor
    RodLocalStiffness ()
    {
        // this->currentsize_ = 0;
        
        // For the time being:  all boundary conditions are homogeneous Neumann
        // This means no boundary condition handling is done at all
        for (int i=0; i<Dune::LocalStiffness<RodLocalStiffness,GridType,RT,m>::SIZE; i++)
            for (size_t j=0; j<this->bctype[i].size(); j++)
                this->bctype[i][j] = Dune::BoundaryConditions::neumann;
    }

    //! Default Constructor
    RodLocalStiffness (const Dune::array<double,3>& K, const Dune::array<double,3>& A)
    {
        for (int i=0; i<3; i++) {
            K_[i] = K[i];
            A_[i] = A[i];
        }
        // this->currentsize_ = 0;
        
        // For the time being:  all boundary conditions are homogeneous Neumann
        // This means no boundary condition handling is done at all
        for (int i=0; i<Dune::LocalStiffness<RodLocalStiffness,GridType,RT,m>::SIZE; i++)
            for (size_t j=0; j<this->bctype[i].size(); j++)
                this->bctype[i][j] = Dune::BoundaryConditions::neumann;
    }
    
    //! assemble local stiffness matrix for given element and order
    /*! On exit the following things have been done:
      - The stiffness matrix for the given entity and polynomial degree has been assembled and is
      accessible with the mat() method.
      - The boundary conditions have been evaluated and are accessible with the bc() method
      - The right hand side has been assembled. It contains either the value of the essential boundary
      condition or the assembled source term and neumann boundary condition. It is accessible via the rhs() method.
      @param[in]  e    a codim 0 entity reference
      \param[in]  localSolution Current local solution, because this is a nonlinear assembler
      @param[in]  k    order of Lagrange basis
    */
    template<typename TypeTag>
    void assemble (const Entity& e, 
                   const Dune::BlockVector<Dune::FieldVector<double, dim> >& localSolution,
                   int k=1);

    
    RT energy (const Entity& e,
               const Dune::array<Configuration,2>& localSolution,
               const Dune::array<Configuration,2>& localReferenceConfiguration,
               int k=1);

    static void interpolationDerivative(const Quaternion<RT>& q0, const Quaternion<RT>& q1, double s,
                                        Dune::array<Quaternion<double>,6>& grad);

    static void interpolationVelocityDerivative(const Quaternion<RT>& q0, const Quaternion<RT>& q1, double s,
                                                double intervalLength, Dune::array<Quaternion<double>,6>& grad);

    Dune::FieldVector<double, 6> getStrain(const Dune::array<Configuration,2>& localSolution,
                                           const Entity& element,
                                           const Dune::FieldVector<double,1>& pos) const;

    /** \brief Assemble the element gradient of the energy functional */
    void assembleGradient(const Entity& element,
                          const Dune::array<Configuration,2>& solution,
                          const Dune::array<Configuration,2>& referenceConfiguration,
                          Dune::array<Dune::FieldVector<double,6>, 2>& gradient) const;
    
    template <class T>
    static Dune::FieldVector<T,3> darboux(const Quaternion<T>& q, const Dune::FieldVector<T,4>& q_s) 
    {
        Dune::FieldVector<double,3> u;  // The Darboux vector
        
        u[0] = 2 * (q.B(0) * q_s);
        u[1] = 2 * (q.B(1) * q_s);
        u[2] = 2 * (q.B(2) * q_s);
        
        return u;
    }
        
};


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
        double K_[3];
        double A_[3];

        /** \brief The stress-free configuration */
        std::vector<Configuration> referenceConfiguration_;

        /** \todo Only for the fd approximations */
        static void infinitesimalVariation(Configuration& c, double eps, int i)
        {
            if (i<3)
                c.r[i] += eps;
            else
                c.q = c.q.mult(Quaternion<double>::exp((i==3)*eps, 
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

        void strainDerivative(const std::vector<Configuration>& localSolution,
                              double pos,
                              Dune::FieldVector<double,1> shapeGrad[2],
                              Dune::FieldVector<double,1> shapeFunction[2],
                              Dune::array<Dune::FieldMatrix<double,2,6>, 6>& derivatives) const;

        void rotationStrainHessian(const std::vector<Configuration>& x, 
                                   double pos,
                                   Dune::FieldVector<double,1> shapeGrad[2],
                                   Dune::FieldVector<double,1> shapeFunction[2],
                                   Dune::array<Dune::Matrix<Dune::FieldMatrix<double,3,3> >, 3>& rotationDer) const;
        
        void strainHessian(const std::vector<Configuration>& localSolution,
                           double pos,
                           Dune::array<Dune::Matrix<Dune::FieldMatrix<double,6,6> >, 3>& translationDer,
                           Dune::array<Dune::Matrix<Dune::FieldMatrix<double,3,3> >, 3>& rotationDer) const;

        void assembleGradient(const std::vector<Configuration>& sol,
                              Dune::BlockVector<Dune::FieldVector<double, blocksize> >& grad) const;

        /** \brief Compute the energy of a deformation state */
        double computeEnergy(const std::vector<Configuration>& sol) const;

        void getNeighborsPerVertex(Dune::MatrixIndexSet& nb) const;

        void getStrain(const std::vector<Configuration>& sol, 
                       Dune::BlockVector<Dune::FieldVector<double, blocksize> >& strain) const;

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

