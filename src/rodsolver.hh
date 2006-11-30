#ifndef ROD_SOLVER_HH
#define RODSOLVER

#include <dune/common/array.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>

#include "../../common/boxconstraint.hh"
#include "../../solver/iterativesolver.hh"

#include "rodassembler.hh"

#include "configuration.hh"

/** \brief Riemannian trust-region solver for 3d Cosserat rod problems */
template <class GridType>
class RodSolver
{ 
    const static int blocksize = 6;

    // Some types that I need
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<double, blocksize, blocksize> > MatrixType;
    typedef Dune::BlockVector<Dune::FieldVector<double, blocksize> >           CorrectionType;
    typedef std::vector<Configuration>                              SolutionType;

    static void setTrustRegionObstacles(double trustRegionRadius,
                                        std::vector<BoxConstraint<blocksize> >& trustRegionObstacles);
public:

    RodSolver()
        : hessianMatrix_(NULL)
    {}

    void setup(const GridType& grid, 
               const Dune::RodAssembler<GridType>* rodAssembler,
               const SolutionType& x,
               int maxTrustRegionSteps,
               double initialTrustRegionRadius,
               int multigridIterations,
               double mgTolerance,
               int mu, 
               int nu1,
               int nu2,
               int baseIterations,
               double baseTolerance);

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
    const Dune::RodAssembler<GridType>* rodAssembler_;

    /** \brief The multigrid solver */
    Dune::IterativeSolver<MatrixType, CorrectionType>* mmgSolver_;

    /** \brief The hierarchy of trust-region obstacles */
    Dune::Array<std::vector<BoxConstraint<blocksize> > > trustRegionObstacles_;

    /** \brief Dummy fields containing 'true' everywhere.  The multigrid step
        expects them :-( */
    Dune::Array<Dune::BitField> hasObstacle_;

    /** \brief The Dirichlet nodes on all levels */
    std::vector<Dune::BitField> dirichletNodes_;
};

#include "rodsolver.cc"

#endif
