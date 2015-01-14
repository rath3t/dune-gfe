// -*- tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=8 sw=4 sts=4:
#ifndef DUNE_GFE_TRUSTREGIONMMGBASESOLVER_HH
#define DUNE_GFE_TRUSTREGIONMMGBASESOLVER_HH

#include <dune/istl/solver.hh>
#include <dune/istl/umfpack.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/solvers/quadraticipopt.hh>

/** \brief Base solver for a monotone multigrid solver when used as the inner solver in a trust region method
  *
  * Monotone multigrid methods solve constrained problems even on the coarsest level.  Therefore, the choice
  * of possible coarse grid solvers is limited.  I have tried both Gauss-Seidel and IPOpt, and both of them
  * have not satisfied my.  Gauss-Seidel is slow when the coarse grid is not coarse enough.  The results computed
  * by IPOpt appear to be imprecise, and I didn't get good trust-region convergence.  I don't really understand
  * this -- it may be a bug in our IPOpt wrapper code.
  * Ideally, one would use a direct solver, in particular close to the solution.  However, a direct solver may
  * not produce admissible corrections.  Also, it may produce energy increase.  Therefore, the TrustRegionMMGBaseSolver
  * does a pragmatic approach: a solution is first computed using UMFPack, disregarding the obstacles.
  * If the result is admissible and decreases the energy we keep it.  Otherwise we fall back to IPOpt.
  */
template <class MatrixType, class VectorType>
class TrustRegionMMGBaseSolver
: public IterativeSolver<VectorType>,
  public CanIgnore<Dune::BitSetVector<VectorType::block_type::dimension> >
{
    typedef typename VectorType::field_type field_type;

    // For complex-valued data
    typedef typename Dune::FieldTraits<field_type>::real_type real_type;

    enum {blocksize = VectorType::value_type::dimension};
    typedef Dune::BitSetVector<blocksize> BitVectorType;

public:

    /** \brief Constructor taking all relevant data */
    TrustRegionMMGBaseSolver(Solver::VerbosityMode verbosity)
        : IterativeSolver<VectorType, BitVectorType>(100,     // maxIterations
                                                     1e-8,    // tolerance
                                                     nullptr,
                                                     verbosity,
                                                     false)  // false = use absolute error
    {}

    void setProblem(const MatrixType& matrix,
                    VectorType& x,
                    const VectorType& rhs) {
        matrix_ = &matrix;
        x_ = &x;
        rhs_ = &rhs;
    }

    /**  \brief Checks whether all relevant member variables are set
      *  \exception SolverError if the iteration step is not set up properly
      */
    virtual void check() const;

    virtual void preprocess();

    /**  \brief Loop, call the iteration procedure
      *  and monitor convergence
      */
    virtual void solve();

    //! The quadratic term in the quadratic energy, is assumed to be symmetric
    const MatrixType* matrix_;

    //! Vector to store the solution
    VectorType* x_;

    //! The linear term in the quadratic energy
    const VectorType* rhs_;

    std::vector<BoxConstraint<field_type,blocksize> >* obstacles_;

};


template <class MatrixType, class VectorType>
void TrustRegionMMGBaseSolver<MatrixType, VectorType>::check() const
{
    // check base class
    IterativeSolver<VectorType,BitVectorType>::check();
}

template <class MatrixType, class VectorType>
void TrustRegionMMGBaseSolver<MatrixType, VectorType>::preprocess()
{
    //this->iterationStep_->preprocess();
}

template <class MatrixType, class VectorType>
void TrustRegionMMGBaseSolver<MatrixType, VectorType>::solve()
{
  MatrixType modifiedStiffness = *matrix_;
  VectorType modifiedRhs = *rhs_;

  ///////////////////////////////////////////
  //   Modify Dirichlet rows
  ///////////////////////////////////////////

  for (size_t j=0; j<modifiedStiffness.N(); j++)
    for (int k=0; k<blocksize; k++)
      modifiedStiffness[j][j][k][k] += 0.0;

  for (size_t j=0; j<modifiedStiffness.N(); j++)
  {
    auto cIt    = modifiedStiffness[j].begin();
    auto cEndIt = modifiedStiffness[j].end();

    for (; cIt!=cEndIt; ++cIt)
    {
      for (int k=0; k<blocksize; k++)
        if ((*this->ignoreNodes_)[j][k])
        {
          (*cIt)[k] = 0;
          if (cIt.index()==j)
            (*cIt)[k][k] = 1.0;
        }

    }

    for (int k=0; k<blocksize; k++)
      if ((*this->ignoreNodes_)[j][k])
        modifiedRhs[j][k] = 0.0;
  }

  /////////////////////////////////////////////////////////////////
  //  Solve the system
  /////////////////////////////////////////////////////////////////
  Dune::InverseOperatorResult statistics;
  Dune::UMFPack<MatrixType> solver(modifiedStiffness);
  solver.setOption(UMFPACK_PRL, 0);   // no output at all
  solver.apply(*x_, modifiedRhs, statistics);

  // Model increase?  Then the matrix is not positive definite -- fall back to ipopt solver
  // The model decrease is $ m(x) - m(x+s) = -<g,s> - 0.5 <s, Hs>
  // Note that rhs = -g
  VectorType tmp(x_->size());
  tmp = 0;
  matrix_->umv(*x_, tmp);
  double modelDecrease = (modifiedRhs*(*x_)) - 0.5 * ((*x_)*tmp);

  if (std::isnan(modelDecrease) or modelDecrease < 0 or true)
  {
    std::cout << "Model increase: " << -modelDecrease << ", falling back to slower solver" << std::endl;

    std::cout << "Total VARIABLES: " << x_->size() << " x " << blocksize << " = " << x_->size()*blocksize << std::endl;
    std::cout << "Dirichlet DOFS: " << this->ignoreNodes_->count() << std::endl;

    QuadraticIPOptSolver<MatrixType, VectorType> baseSolver;
    baseSolver.verbosity_ = NumProc::REDUCED;
    baseSolver.tolerance_ = 1e-8;

    *x_ = 0;
    baseSolver.setProblem(*matrix_, *x_, *rhs_);
    baseSolver.obstacles_ = obstacles_;
    baseSolver.ignoreNodes_ = this->ignoreNodes_;

    baseSolver.solve();
  }

  // Check whether the solution is admissible wrt the obstacles
  bool admissible = true;
  assert (obstacles_->size() == x_->size());
  for (size_t i=0; i<obstacles_->size(); i++)
  {
    for (int j=0; j<blocksize; j++)
      if ( (*x_)[i][j] > (*obstacles_)[i].upper(j) or (*x_)[i][j] < (*obstacles_)[i].lower(j) )
      {
        admissible = false;
        break;
      }
    if (not admissible)
      break;
  }

  if (not admissible)
  {
    std::cout << "Inadmissible step, falling back to slower solver" << std::endl;

    QuadraticIPOptSolver<MatrixType, VectorType> baseSolver;
    baseSolver.verbosity_ = NumProc::REDUCED;
    baseSolver.tolerance_ = 1e-8;

    *x_ = 0;
    baseSolver.setProblem(*matrix_, *x_, *rhs_);
    baseSolver.obstacles_ = obstacles_;
    baseSolver.ignoreNodes_ = this->ignoreNodes_;

    baseSolver.solve();
  }

}


#endif
