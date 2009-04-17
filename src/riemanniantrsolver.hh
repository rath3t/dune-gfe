#ifndef ROD_SOLVER_HH
#define ROD_SOLVER_HH

#include <vector>

#include <dune/common/bitsetvector.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>

#include <dune-solvers/boxconstraint.hh>
#include <dune-solvers/norms/h1seminorm.hh>
#include <dune-solvers/solvers/loopsolver.hh>

#include "geodesicfeassembler.hh"

#include "rigidbodymotion.hh"

/** \brief Riemannian trust-region solver for 3d Cosserat rod problems */
template <class GridType>
class RodSolver : public IterativeSolver<std::vector<RigidBodyMotion<3> >, Dune::BitSetVector<6> >
{ 
    const static int blocksize = 6;

    // Centralize the field type here
    typedef double field_type;

    // Some types that I need
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<field_type, blocksize, blocksize> > MatrixType;
    typedef Dune::BlockVector<Dune::FieldVector<field_type, blocksize> >           CorrectionType;
    typedef std::vector<RigidBodyMotion<3> >                                             SolutionType;

public:

    RodSolver()
        : IterativeSolver<std::vector<RigidBodyMotion<3> >, Dune::BitSetVector<6> >(0,100,NumProc::FULL),
          hessianMatrix_(NULL), h1SemiNorm_(NULL)
    {}

    void setup(const GridType& grid, 
               const GeodesicFEAssembler<typename GridType::LeafGridView, RigidBodyMotion<3> >* rodAssembler,
               const SolutionType& x,
               const Dune::BitSetVector<blocksize>& dirichletNodes,
               double tolerance,
               int maxTrustRegionSteps,
               double initialTrustRegionRadius,
               int multigridIterations,
               double mgTolerance,
               int mu, 
               int nu1,
               int nu2,
               int baseIterations,
               double baseTolerance,
               bool instrumented);

    void solve();

    void setInitialSolution(const SolutionType& x) {
        x_ = x;
    }

    SolutionType getSol() const {return x_;}

protected:

    /** \brief The grid */
    const GridType* grid_;

    /** \brief The solution vector */
    SolutionType x_;

    /** \brief The initial trust-region radius in the maximum-norm */
    double initialTrustRegionRadius_;

    /** \brief Maximum number of trust-region steps */
    int maxTrustRegionSteps_;

    /** \brief Maximum number of iterations of the multigrid basesolver */
    int baseIt_;

    double baseTolerance_;

    /** \brief Number of coarse multigrid iterations (1 for a V-cycle, 2 for a W-cycle) */
    int mu_;

    /** \brief Number of multigrid presmoothing steps */
    int nu1_;

    /** \brief Number of multigrid postsmoothing steps */
    int nu2_;

    /** \brief Maximum number of multigrid iterations */
    int multigridIterations_;

    /** \brief Error tolerance of the multigrid QP solver */
    double qpTolerance_;

    /** \brief Hessian matrix */
    MatrixType* hessianMatrix_;

    /** \brief The assembler for the material law */
    const GeodesicFEAssembler<typename GridType::LeafGridView, RigidBodyMotion<3> >* assembler_;

    /** \brief The multigrid solver */
    LoopSolver<CorrectionType>* mmgSolver_;

    /** \brief Dummy fields containing 'true' everywhere.  The multigrid step
        expects them :-( */
    std::vector<Dune::BitSetVector<1> > hasObstacle_;

    /** \brief The norm used to measure multigrid convergence */
    H1SemiNorm<CorrectionType>* h1SemiNorm_;
    
    /** \brief If set to true we log convergence speed and other stuff */
    bool instrumented_;

};

#include "rodsolver.cc"

#endif
