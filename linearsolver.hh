#ifndef LINEAR_SOLVER_HH
#define LINEAR_SOLVER_HH

#include <dune/common/fmatrix.hh>
#include <dune/istl/matrix.hh>

void linearSolver(const Dune::Matrix<Dune::FieldMatrix<double,1,1> >& A,
                  Dune::BlockVector<Dune::FieldVector<double,1> >& x,
                  const Dune::FieldVector<double,6>& b);

void lapackSVD(const Dune::FieldMatrix<double,3,3>& A,
               Dune::FieldMatrix<double,3,3>& U,
               Dune::FieldVector<double,3>& sigma,
              Dune::FieldMatrix<double,3,3>& VT);

#endif
