#ifndef LINEAR_SOLVER_HH
#define LINEAR_SOLVER_HH

#include <dune/common/fmatrix.hh>

void linearSolver(const Dune::FieldMatrix<double,6,12>& A,
                  Dune::FieldVector<double,12>& x,
                  const Dune::FieldVector<double,6>& b);

void lapackSVD(const Dune::FieldMatrix<double,3,3>& A,
               Dune::FieldMatrix<double,3,3>& U,
               Dune::FieldVector<double,3>& sigma,
              Dune::FieldMatrix<double,3,3>& VT);

#endif
