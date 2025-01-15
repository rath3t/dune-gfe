#include <config.h>

#include <iostream>
#include <array>

#include <dune/common/fmatrix.hh>

#include <dune/gfe/spaces/rotation.hh>
#include "valuefactory.hh"

using namespace Dune;

void testDDExp()
{
  std::array<FieldVector<double,3>, 125> v;
  int ct = 0;
  double eps = 1e-4;

  for (int i=-2; i<3; i++)
    for (int j=-2; j<3; j++)
      for (int k=-2; k<3; k++) {
        v[ct][0] = i;
        v[ct][1] = j;
        v[ct][2] = k;
        ct++;
      }

  for (size_t i=0; i<v.size(); i++) {

    // Compute FD approximation of second derivative of exp
    std::array<Dune::FieldMatrix<double,3,3>, 4> fdDDExp;

    for (int j=0; j<3; j++) {

      for (int k=0; k<3; k++) {

        if (j==k) {

          GFE::SkewMatrix<double,3> forward(v[i]);
          forward.axial()[j] += eps;
          const auto forwardQ  = GFE::Rotation<double,3>::exp(forward);

          GFE::SkewMatrix<double,3> center(v[i]);
          const auto centerQ   = GFE::Rotation<double,3>::exp(center);

          GFE::SkewMatrix<double,3> backward(v[i]);
          backward.axial()[j] -= eps;
          const auto backwardQ = GFE::Rotation<double,3>::exp(backward);

          for (int l=0; l<4; l++)
            fdDDExp[l][j][j] = (forwardQ[l] - 2*centerQ[l] + backwardQ[l]) / (eps*eps);


        } else {

          GFE::SkewMatrix<double,3> ffV(v[i]);      ffV.axial()[j] += eps;     ffV.axial()[k] += eps;
          GFE::SkewMatrix<double,3> fbV(v[i]);      fbV.axial()[j] += eps;     fbV.axial()[k] -= eps;
          GFE::SkewMatrix<double,3> bfV(v[i]);      bfV.axial()[j] -= eps;     bfV.axial()[k] += eps;
          GFE::SkewMatrix<double,3> bbV(v[i]);      bbV.axial()[j] -= eps;     bbV.axial()[k] -= eps;

          const auto forwardForwardQ = GFE::Rotation<double,3>::exp(ffV);
          const auto forwardBackwardQ = GFE::Rotation<double,3>::exp(fbV);
          const auto backwardForwardQ = GFE::Rotation<double,3>::exp(bfV);
          const auto backwardBackwardQ = GFE::Rotation<double,3>::exp(bbV);

          for (int l=0; l<4; l++)
            fdDDExp[l][j][k] = (forwardForwardQ[l] + backwardBackwardQ[l]
                                - forwardBackwardQ[l] - backwardForwardQ[l]) / (4*eps*eps);

        }

      }

    }

    // Compute analytical second derivative of exp
    std::array<Dune::FieldMatrix<double,3,3>, 4> ddExp;
    GFE::Rotation<double,3>::DDexp(v[i], ddExp);

    for (int m=0; m<4; m++)
      for (int j=0; j<3; j++)
        for (int k=0; k<3; k++)
          if ( std::abs(fdDDExp[m][j][k] - ddExp[m][j][k]) > eps) {
            std::cout << "Error at v = " << v[i]
                      << "[" << m << ", " << j << ", " << k << "] "
                      << "    fd: " << fdDDExp[m][j][k]
                      << "    analytical: " << ddExp[m][j][k] << std::endl;
          }
  }
}

void testRotation(GFE::Rotation<double,3> q)
{
  // Make sure it really is a unit quaternion
  q.normalize();

  assert(std::abs(1-q.two_norm()) < 1e-12);

  // Turn it into a matrix
  FieldMatrix<double,3,3> matrix;
  q.matrix(matrix);

  // make sure it is an orthogonal matrix
  if (std::abs(1-matrix.determinant()) > 1e-12 )
    DUNE_THROW(Exception, "Expected determinant 1, but the computed value is " << matrix.determinant());

  assert( std::abs( matrix[0]*matrix[1] ) < 1e-12 );
  assert( std::abs( matrix[0]*matrix[2] ) < 1e-12 );
  assert( std::abs( matrix[1]*matrix[2] ) < 1e-12 );

  // Turn the matrix back into a quaternion, and check whether it is the same one
  // Since the quaternions form a double covering of SO(3), we may either get q back
  // or -q.  We have to check both.
  GFE::Rotation<double,3> newQ;
  newQ.set(matrix);

  GFE::Rotation<double,3> diff = newQ;
  diff -= q;

  GFE::Rotation<double,3> sum  = newQ;
  sum += q;

  if (diff.infinity_norm() > 1e-12 && sum.infinity_norm() > 1e-12)
    DUNE_THROW(Exception, "Backtransformation failed for " << q << ". ");

  // //////////////////////////////////////////////////////
  //   Check the director vectors
  // //////////////////////////////////////////////////////

  for (int i=0; i<3; i++)
    for (int j=0; j<3; j++)
      assert( std::abs(matrix[i][j] - q.director(j)[i]) < 1e-12 );

  // //////////////////////////////////////////////////////
  //   Check multiplication with another unit quaternion
  // //////////////////////////////////////////////////////

  for (int i=-2; i<2; i++)
    for (int j=-2; j<2; j++)
      for (int k=-2; k<2; k++)
        for (int l=-2; l<2; l++)
          if (i!=0 || j!=0 || k!=0 || l!=0) {

            GFE::Rotation<double,3> q2(GFE::Quaternion<double>(i,j,k,l));
            q2.normalize();

            // set up corresponding rotation matrix
            FieldMatrix<double,3,3> q2Matrix;
            q2.matrix(q2Matrix);

            // q2 = q2 * q   Quaternion multiplication
            q2 = q2.mult(q);

            // q2 = q2 * q   Matrix multiplication
            q2Matrix.rightmultiply(matrix);

            FieldMatrix<double,3,3> productMatrix;
            q2.matrix(productMatrix);

            // Make sure we got identical results
            productMatrix -= q2Matrix;
            assert(productMatrix.infinity_norm() < 1e-10);

          }

  // ////////////////////////////////////////////////////////////////
  //   Check the operators 'B' that create an orthonormal basis of H
  // ////////////////////////////////////////////////////////////////

  GFE::Quaternion<double> Bq[4];
  Bq[0] = q;
  Bq[1] = q.B(0);
  Bq[2] = q.B(1);
  Bq[3] = q.B(2);

  for (int i=0; i<4; i++) {

    for (int j=0; j<4; j++) {

      double prod = Bq[i]*Bq[j];
      assert( std::abs( prod - (i==j) ) < 1e-6 );

    }

  }

  //////////////////////////////////////////////////////////////////////
  //  Check whether the derivativeOfMatrixToQuaternion methods works
  //////////////////////////////////////////////////////////////////////

  GFE::Tensor3<double,4,3,3> derivative = GFE::Rotation<double,3>::derivativeOfMatrixToQuaternion(matrix);

  const double eps = 1e-8;
  GFE::Tensor3<double,4,3,3> derivativeFD;

  for (size_t i=0; i<3; i++)
  {
    for (size_t j=0; j<3; j++)
    {
      auto forwardMatrix = matrix;
      forwardMatrix[i][j] += eps;
      auto backwardMatrix = matrix;
      backwardMatrix[i][j] -= eps;

      GFE::Rotation<double,3> forwardRotation, backwardRotation;
      forwardRotation.set(forwardMatrix);
      backwardRotation.set(backwardMatrix);

      for (size_t k=0; k<4; k++)
        derivativeFD[k][i][j] = (forwardRotation.globalCoordinates()[k] - backwardRotation.globalCoordinates()[k]) / (2*eps);

    }
  }

  if ((derivative - derivativeFD).infinity_norm() > 1e-6)
  {
    std::cout << "At matrix:\n" << matrix << std::endl;

    std::cout << "Derivative of matrix to quaternion map does not match its FD approximation" << std::endl;
    std::cout << "Analytical derivative:" << std::endl;
    std::cout << derivative << std::endl;
    std::cout << "Finite difference approximation" << std::endl;
    std::cout << derivativeFD << std::endl;
    abort();
  }
}

//! test interpolation between two rotations
bool testInterpolation(const GFE::Rotation<double, 3>& a, const GFE::Rotation<double, 3>& b) {

  // Compute difference on T_a SO(3)
  auto newB = GFE::Rotation<double, 3>::interpolate(a, b, 1.0);

  // Compare matrix representation
  FieldMatrix<double, 3, 3> matB;
  b.matrix(matB);

  FieldMatrix<double, 3, 3> matNewB;
  newB.matrix(matNewB);

  matNewB -= matB;
  if (matNewB.infinity_norm() > 1e-14)
    std::cout << " Interpolation failed with difference " << matNewB.infinity_norm()  << std::endl;

  return (matNewB.infinity_norm() < 1e-14);
}

int main (int argc, char *argv[]) try
{
  std::vector<GFE::Rotation<double,3> > testPoints;
  GFE::ValueFactory<GFE::Rotation<double,3> >::get(testPoints);

  int nTestPoints = testPoints.size();

  // Test each element in the list
  for (int i=0; i<nTestPoints; i++)
    testRotation(testPoints[i]);

  bool passed(true);
  // Test interpolating between pairs of rotations
  for (int i=0; i<nTestPoints-1; i++)
    passed = passed and testInterpolation(testPoints[i], testPoints[i+1]);

  // //////////////////////////////////////////////
  //   Test second derivative of exp
  // //////////////////////////////////////////////
  testDDExp();

  return not passed;

}
catch (Exception& e) {

  std::cout << e.what() << std::endl;

}
