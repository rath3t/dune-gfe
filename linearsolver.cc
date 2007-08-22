// The following line switches of the f2c.h in UG.  Boy, is this disgusting!
#define F2C_INCLUDE

#include "linearsolver.hh"
#include "lapackpp.h"

using namespace Dune;

// Solve a small linear system using lapack++
void linearSolver(const FieldMatrix<double,6,12>& A,
                  FieldVector<double,12>& x,
                  const FieldVector<double,6>& b)
{
    int N = 6;
    int M = 12;

    LaGenMatDouble matrix(6,12);

    for (int i=0; i<N; i++)
        for (int j=0; j<M; j++)
            matrix(i,j) = A[i][j];

    LaVectorDouble X(M);
    for (int i=0; i<M; i++)
        X(i) = x[i];

    LaVectorDouble B(N);
    for (int i=0; i<N; i++)
        B(i) = b[i];

    LaLinearSolve(matrix, X, B);

    for (int i=0; i<M; i++)
        x[i] = X(i);
}
