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

// Compute the svd using lapack++
void lapackSVD(const FieldMatrix<double,3,3>& A,
               FieldMatrix<double,3,3>& U,
               FieldVector<double,3>& sigma,
               FieldMatrix<double,3,3>& VT)
{
    LaGenMatDouble lpA(3,3), lpU(3,3), lpVT(3,3);
    LaVectorDouble lpSigma(3);

    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
            lpA(i,j) = A[i][j];

    LaSVD_IP(lpA, lpSigma, lpU, lpVT);

    for (int i=0; i<3; i++) {

        sigma[i] = lpSigma(i);

        for (int j=0; j<3; j++) {
            U[i][j]  = lpU(i,j);
            VT[i][j] = lpVT(i,j);
        }

    }
}
