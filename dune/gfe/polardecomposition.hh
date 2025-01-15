// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_GFE_POLARDECOMPOSITION_HH
#define DUNE_GFE_POLARDECOMPOSITION_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/version.hh>
#include <dune/common/exceptions.hh>
#include <dune/gfe/linearalgebra.hh>
#include <dune/gfe/spaces/rotation.hh>

///////////////////////////////////////////////////////////////////////////////////////////
//  Two Methods to compute the Polar Factor of a 3x3 matrix
///////////////////////////////////////////////////////////////////////////////////////////

namespace Dune::GFE
{

  class PolarDecomposition
  {
  public:

    /** \brief Compute the polar factor  of a matrix iteratively
                         Attention: For matrices that are quite far away from an orthogonal matrix,
                         this might return a matrix with determinant = -1 ! */
    template <class field_type>
    FieldMatrix<field_type,3,3> operator() (const FieldMatrix<field_type,3,3>& matrix, double tol = 0.001) const
    {
      size_t maxIterations = 100;
      // Use Higham's method
      auto polar = matrix;
      for (size_t i=0; i<maxIterations; i++)
      {
        auto oldPolar = polar;
        auto polarInvert = polar;
        polarInvert.invert();
        for (size_t j=0; j<polar.N(); j++)
          for (size_t k=0; k<polar.M(); k++)
            polar[j][k] = 0.5 * (polar[j][k] + polarInvert[k][j]);
        oldPolar -= polar;
        if (oldPolar.frobenius_norm() < tol) {
          break;
        }
      }
      return polar;
    }
  };

  class HighamNoferiniPolarDecomposition
  {
  public:
    /** \brief Compute the polar factor part of a matrix using the paper
                         "An algorithm to compute the polar decomposition of a 3 × 3 matrix" from Higham and Noferini (2015)
     *  \param matrix Compute the polar factor part of this matrix
     *  \param tau1 tolerance, default value taken from Higham and Noferini
     *  \param tau2 tolerance, default value taken from Higham and Noferini
     */
    template <class field_type>
    FieldMatrix<field_type, 3, 3> operator() (const FieldMatrix<field_type, 3, 3>& matrix, double tau1 = 0.0001, double tau2 = 0.0001) const
    {
      auto normedMatrix = matrix;
      normedMatrix /= normedMatrix.frobenius_norm();
      FieldMatrix<field_type,4,4> bilinearmatrix = bilinearMatrix(normedMatrix);
      auto detB = determinantLaplace(bilinearmatrix);
      auto detM = normedMatrix.determinant();
      field_type domEV = 0;
      if (detM < 0) {
        detM = -detM;
        bilinearmatrix = -bilinearmatrix;
      }

      if ( detB + 1./3 > tau1) {       // estimate dominant eigenvalue if well separated => use formula
        domEV = dominantEV( detB, detM );
      } else {       // eignevalue nearly a double root => use newtons method
        auto coefficients = characteristicCoefficients(bilinearmatrix);
        domEV = dominantEVNewton(coefficients,detB);
      }
      // compute iterationmatrix B_S
      FieldMatrix<field_type,4,4> BS  = {{domEV,0,0,0},{0,domEV,0,0},{0,0,domEV,0},{0,0,0,domEV}};
      BS -= bilinearmatrix;
      if (detB < 1 - tau2) {
        Rotation<field_type,3> v(obtainQuaternion(BS));
        FieldMatrix<field_type,3,3> mat;
        v.matrix(mat);
        return mat;
      } else {
        DUNE_THROW(NotImplemented, "The eigenvalues are not wellseparated => inverse iteration won't converge!");
      }
    }

  protected:

    /** \brief Compute the bilinear matrix for a given 3x3 matrix*/
    template <class field_type>
    static FieldMatrix<field_type,4,4> bilinearMatrix(const FieldMatrix<field_type,3,3> matrix)
    {
      FieldMatrix<field_type,4,4> bilinearmatrix;
      bilinearmatrix[0][0] = matrix[0][0] + matrix[1][1] + matrix[2][2];
      bilinearmatrix[0][1] = matrix[1][2] - matrix[2][1];
      bilinearmatrix[0][2] = matrix[2][0] - matrix[0][2];
      bilinearmatrix[0][3] = matrix[0][1] - matrix[1][0];
      bilinearmatrix[1][0] = bilinearmatrix[0][1];
      bilinearmatrix[1][1] = matrix[0][0] - matrix[1][1] - matrix[2][2];
      bilinearmatrix[1][2] = matrix[0][1] + matrix[1][0];
      bilinearmatrix[1][3] = matrix[2][0] + matrix[0][2];
      bilinearmatrix[2][0] = bilinearmatrix[0][2];
      bilinearmatrix[2][1] = bilinearmatrix[1][2];
      bilinearmatrix[2][2] = matrix[1][1] - matrix[0][0] - matrix[2][2];
      bilinearmatrix[2][3] = matrix[1][2] + matrix[2][1];
      bilinearmatrix[3][0] = bilinearmatrix[0][3];
      bilinearmatrix[3][1] = bilinearmatrix[1][3];
      bilinearmatrix[3][2] = bilinearmatrix[2][3];
      bilinearmatrix[3][3] = matrix[2][2] - matrix[0][0] - matrix[1][1];
      return bilinearmatrix;
    }

    /** \brief Compute the determinant of a 4x4 matrix using the Laplace method
                            The implementation is faster than calling matrix.determinant()*/
    template <class field_type>
    static field_type determinantLaplace( const FieldMatrix<field_type,4,4>& matrix)
    {
      field_type T2233 = matrix[2][2] * matrix[3][3];
      field_type T3223 = matrix[3][2] * matrix[2][3];
      field_type T1       = T2233 - T3223;

      field_type T1233 = matrix[1][2] * matrix[3][3];
      field_type T3213 = matrix[3][2] * matrix[1][3];
      field_type T2 = T1233 - T3213;

      field_type T1223 = matrix[1][2] * matrix[2][3];
      field_type T2213 = matrix[2][2] * matrix[1][3];
      field_type T3       = T1223 - T2213;

      field_type T0233 = matrix[0][2] * matrix[3][3];
      field_type T3203 = matrix[3][2] * matrix[0][3];
      field_type T4       =  T3203 - T0233;

      field_type T0223 = matrix[0][2] * matrix[2][3];
      field_type T2203 = matrix[2][2] * matrix[0][3];
      field_type T5       = T0223- T2203;

      field_type T0213 = matrix[0][2] * matrix[1][3];
      field_type T1203 = matrix[1][2] * matrix[0][3];
      field_type T6       = T0213-T1203;

      return matrix[0][0] * (matrix[1][1] * T1 - matrix[2][1] * T3 + matrix[3][1] * T3)
             - matrix[1][0] * (matrix[0][1] * T1 + matrix[2][1] * T4 + matrix[3][1] * T5)
             + matrix[2][0] * (matrix[0][1] * T2 + matrix[1][1] * T4 + matrix[3][1] * T6)
             - matrix[3][0] * (matrix[0][1] * T2 - matrix[1][1] * T5 + matrix[2][1] * T6);
    }

    /** \brief Return the factors a,b,c of the characteristic polynomial x^4+a*x^3+b*x^2+c*x + detB */
    template <class field_type>
    static FieldVector<field_type,3> characteristicCoefficients(const FieldMatrix<field_type,4,4>& matrix)
    {
      field_type a = - Dune::GFE::trace(matrix);
      field_type b =  matrix[0][0] * (matrix[1][1] + matrix[2][2]  + matrix[3][3])
                     +  matrix[3][3] * (matrix[1][1] + matrix[2][2])
                     - (matrix[1][0] *  matrix[0][1] + matrix[2][0]  * matrix[0][2] +
                        matrix[3][0] * matrix[0][3] + matrix[1][2] * matrix[2][1] +
                        matrix[1][3] * matrix[3][1] + matrix[3][2] * matrix[2][3]);

      field_type c =  matrix[0][0] * (matrix[1][2] *  matrix[2][1] + matrix[1][3] *  matrix[3][1] + matrix[2][3] * matrix[3][2] -(matrix[2][2] * matrix[3][3] + matrix[1][1] * (matrix[2][2] + matrix[3][3])))
                     +  matrix[0][1] * (matrix[1][0] * (matrix[2][2] + matrix[3][3]) -(matrix[1][2] * matrix[2][0] + matrix[1][3] * matrix[3][0]))
                     +  matrix[0][2] * (matrix[2][0] * (matrix[1][1] + matrix[3][3]) -(matrix[1][0] * matrix[2][1] + matrix[2][3] * matrix[3][0]))
                     +  matrix[0][3] * (matrix[3][0] * (matrix[1][1] + matrix[2][2]) -(matrix[1][0] * matrix[3][1] + matrix[2][0] * matrix[3][2]))
                     +  matrix[1][1] * (matrix[2][3] *  matrix[3][2] - matrix[2][2] *  matrix[3][3])
                     +  matrix[1][2] * (matrix[2][1] *  matrix[3][3] - matrix[2][3] *  matrix[3][1])
                     +  matrix[1][3] * (matrix[2][2] *  matrix[3][1] - matrix[2][1] *  matrix[3][2]);
      return {a, b, c};
    }

    /** \brief Return the dominant Eigenvalue of the matrix B */
    template <class field_type>
    static field_type dominantEV(field_type detB, field_type detM )
    {
      auto c = 8 * detM;
      auto sigma0 = 1 + 3 * detB;
      auto sigma1 = 27/16*c*c + 9 * detB - 1;
      auto alpha  = sigma1/std::pow(sigma0, 1.5 );
      auto z = 4/3 * (1 + std::sqrt(sigma0) * std::cos(std::acos(alpha)/3));
      auto s = std::sqrt(z)/ 2;
      auto x = 4-z+c/s;
      if (x > 0)
        return s + std::sqrt(x)/2;
      else
        return s;
    }

    /** \brief Return the dominant Eigenvalue of the matrix B.
                         This algorithm corresponds to Algorithm 3.4 in
                         "An algorithm to compute the polar decomposition
                            of a 3 × 3 matrix" from Higham and Noferini (2015)
     *  \param coefficients Coefficients of the characteristic polynomial for the matrix b
     *  \param detB Determinant of the matrix B
     *  \param tol Tolerance for the Newton method, the value is taken from the paper by Higham and Noferini
     */
    template <class field_type>
    static field_type dominantEVNewton(const FieldVector<field_type,3>& coefficients, field_type detB, double tol = 10e-5)
    {
      field_type charPol = 0;
      field_type charPolDeriv = 0;
      field_type lambdaOld = 3;
      field_type domEV = std::sqrt(3);
      while ((lambdaOld - domEV)/domEV > tol) {
        lambdaOld = domEV;
        charPol   = detB + domEV*( coefficients[2] + domEV*( coefficients[1] + domEV*( coefficients[0]+  domEV ) ) );
        charPolDeriv= coefficients[2] + domEV * ( 2*coefficients[1]+ domEV * ( 3*coefficients[0] + 4*domEV) );
        domEV = domEV- charPol / charPolDeriv;
      }
      return domEV;
    }

    /** \brief Return quaternion vector resulting from step 7 and 8 in Algorithm 3.2 from Higham and Noferini (2015)
                        Step 7: Calculate the LDLT factorization of the given matrix with diagonal pivoting
                        Step 8: Using L and the pivotmatix, calculate the vector v
     * \param shiftedB  shifted matrix B as given in chapter 3 of Higham an Noferini: shiftedB = lambda * Id - B, where lambda is the dominant eigenvalue of B
            \returns v       the respective polar factor in quaternion form
     *\
     */
    template <class field_type>
    static FieldVector<field_type,4> obtainQuaternion(const FieldMatrix<field_type,4,4> shiftedB)
    {
      FieldMatrix<field_type,4,4> P = 0;
      FieldMatrix<field_type,4,4> Pleft = 0;
      FieldMatrix<field_type,4,4> Pright = 0;
      FieldMatrix<field_type,4,4> L = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
      FieldVector<field_type,4>   D = {0,0,0,0};
      FieldVector<field_type,4> Bdiag = {shiftedB[0][0], shiftedB[1][1], shiftedB[2][2], shiftedB[3][3]};
      int maxi = 0;       // Index of the maximal element
      int secmax = 0;
      int secmin = 0;
      int mini = 0;       // Index of the minimal element

      // Find Pivotmatrix
      for (int i = 0; i < 4; ++i)       // going through the diagonal searching for the maximum
        if ( Bdiag[maxi] < Bdiag[i] )          // found a bigger one but smaller than the older ones
          maxi = i;

      Pleft[0][maxi]  = 1;
      Pright[maxi][0] = 1;              // Largest element to the first position

      for (int i = 0; i < 4; ++i)       // going through the diagonal searching for the minimum
        if ( Bdiag[mini] > Bdiag[i] )          // found a smaller one but smaller than the older ones
          mini = i;

      Pleft[3][mini]  = 1;
      Pright[mini][3] = 1;            // Smallest element to the last position


      for (int i = 0; i < 4; ++i) {
        if ( i != maxi && i != mini ) {
          for (int j = 0; j < 4; ++j) {
            if ( j != maxi && j != mini && j != i  ) {
              if ( Bdiag[i] < Bdiag[j] ) {
                Pleft[1][j]  = 1;                   // Second largest element at the second position
                Pright[j][1] = 1;
                Pleft[2][i]  = 1;                   // Third largest element at the third position
                Pright[i][2] = 1;
                secmin = i;
                secmax = j;
              } else {
                Pleft[2][j]  = 1;
                Pright[j][2] = 1;
                Pleft[1][i]  = 1;
                Pright[i][1] = 1;
                secmin = j;
                secmax = i;
              }
            }
          }
        }
      }

      //P = Pleft*shiftedB*Pright;
      P[maxi][maxi] = shiftedB[0][0];
      P[maxi][secmax] = shiftedB[0][1];
      P[maxi][secmin] = shiftedB[0][2];
      P[maxi][mini] = shiftedB[0][3];

      P[secmax][maxi] = shiftedB[1][0];
      P[secmax][secmax] = shiftedB[1][1];
      P[secmax][secmin] = shiftedB[1][2];
      P[secmax][mini] = shiftedB[1][3];

      P[secmin][maxi] = shiftedB[2][0];
      P[secmin][secmax] = shiftedB[2][1];
      P[secmin][secmin] = shiftedB[2][2];
      P[secmin][mini] = shiftedB[2][3];

      P[mini][maxi] = shiftedB[3][0];
      P[mini][secmax] = shiftedB[3][1];
      P[mini][secmin] = shiftedB[3][2];
      P[mini][mini] = shiftedB[3][3];

      // Choleskydecomposition

      for (int k = 0; k < 4; ++k) {       //columns
        D[k] = P[k][k];
        for (int j = 0; j < k; ++j)        //sum
          D[k] -= L[k][j]*L[k][j] * D[j];

        for (int i = k+1; i < 4; ++i) {         //rows
          L[i][k] = P[i][k];
          for (int j = 0; j < k; ++j)          //sum
            L[i][k] -= L[i][j]*L[k][j] * D[j];
          L[i][k] /= D[k];
        }
      }

      FieldVector<field_type,4> v = {0,0,0,0};
      FieldVector<field_type,4> unnormed = {0,0,0,1};
      auto neg = -L[3][2];
      unnormed[0] = L[1][0] * ( L[3][1] + L[2][1]*neg ) + L[2][0] * L[3][2] - L[3][0];
      unnormed[1] = L[2][1] * L[3][2] - L[3][1];
      unnormed[2] = neg;
      unnormed  /= unnormed.two_norm();
      v[0] = unnormed[maxi];
      v[1] = unnormed[secmax];
      v[2] = unnormed[secmin];
      v[3] = unnormed[mini];
      return v;
    }
  };

}  // namespace Dune::GFE

#endif
