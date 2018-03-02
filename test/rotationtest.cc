#include <config.h>

#include <iostream>
#include <array>

#include <dune/common/fmatrix.hh>

#warning Do not include onedgrid.hh
#include <dune/grid/onedgrid.hh>

#include <dune/gfe/rotation.hh>
#warning Do not include rodlocalstiffness.hh
#include <dune/gfe/rodlocalstiffness.hh>
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

                    SkewMatrix<double,3> forward(v[i]);
                    forward.axial()[j] += eps;
                    Rotation<double,3> forwardQ  = Rotation<double,3>::exp(forward);

                    SkewMatrix<double,3> center(v[i]);
                    Rotation<double,3> centerQ   = Rotation<double,3>::exp(center);

                    SkewMatrix<double,3> backward(v[i]);
                    backward.axial()[j] -= eps;
                    Rotation<double,3> backwardQ = Rotation<double,3>::exp(backward);

                    for (int l=0; l<4; l++)
                        fdDDExp[l][j][j] = (forwardQ[l] - 2*centerQ[l] + backwardQ[l]) / (eps*eps);


                } else {

                    SkewMatrix<double,3> ffV(v[i]);      ffV.axial()[j] += eps;     ffV.axial()[k] += eps;
                    SkewMatrix<double,3> fbV(v[i]);      fbV.axial()[j] += eps;     fbV.axial()[k] -= eps;
                    SkewMatrix<double,3> bfV(v[i]);      bfV.axial()[j] -= eps;     bfV.axial()[k] += eps;
                    SkewMatrix<double,3> bbV(v[i]);      bbV.axial()[j] -= eps;     bbV.axial()[k] -= eps;

                    Rotation<double,3> forwardForwardQ = Rotation<double,3>::exp(ffV);
                    Rotation<double,3> forwardBackwardQ = Rotation<double,3>::exp(fbV);
                    Rotation<double,3> backwardForwardQ = Rotation<double,3>::exp(bfV);
                    Rotation<double,3> backwardBackwardQ = Rotation<double,3>::exp(bbV);

                    for (int l=0; l<4; l++)
                        fdDDExp[l][j][k] = (forwardForwardQ[l] + backwardBackwardQ[l]
                                            - forwardBackwardQ[l] - backwardForwardQ[l]) / (4*eps*eps);

                }

            }

        }

        // Compute analytical second derivative of exp
        std::array<Dune::FieldMatrix<double,3,3>, 4> ddExp;
        Rotation<double,3>::DDexp(v[i], ddExp);

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

void testDerivativeOfInterpolatedPosition()
{
    std::array<Rotation<double,3>, 6> q;

    FieldVector<double,3>  xAxis = {1,0,0};
    FieldVector<double,3>  yAxis = {0,1,0};
    FieldVector<double,3>  zAxis = {0,0,1};

    q[0] = Rotation<double,3>(xAxis, 0);
    q[1] = Rotation<double,3>(xAxis, M_PI/2);
    q[2] = Rotation<double,3>(yAxis, 0);
    q[3] = Rotation<double,3>(yAxis, M_PI/2);
    q[4] = Rotation<double,3>(zAxis, 0);
    q[5] = Rotation<double,3>(zAxis, M_PI/2);

    double eps = 1e-7;

    for (int i=0; i<6; i++) {

        for (int j=0; j<6; j++) {

            for (int k=0; k<7; k++) {

                double s = k/6.0;

                std::array<Quaternion<double>,6> fdGrad;

                // ///////////////////////////////////////////////////////////
                //   First: test the interpolated position
                // ///////////////////////////////////////////////////////////
                for (int l=0; l<3; l++) {
                    SkewMatrix<double,3> forward(FieldVector<double,3>(0));
                    SkewMatrix<double,3> backward(FieldVector<double,3>(0));
                    forward.axial()[l] += eps;
                    backward.axial()[l] -= eps;
                    fdGrad[l] =  Rotation<double,3>::interpolate(q[i].mult(Rotation<double,3>::exp(forward)), q[j], s);
                    fdGrad[l] -= Rotation<double,3>::interpolate(q[i].mult(Rotation<double,3>::exp(backward)), q[j], s);
                    fdGrad[l] /= 2*eps;

                    fdGrad[3+l] =  Rotation<double,3>::interpolate(q[i], q[j].mult(Rotation<double,3>::exp(forward)), s);
                    fdGrad[3+l] -= Rotation<double,3>::interpolate(q[i], q[j].mult(Rotation<double,3>::exp(backward)), s);
                    fdGrad[3+l] /= 2*eps;
                }

                // Compute analytical gradient
                std::array<Quaternion<double>,6> grad;
                RodLocalStiffness<OneDGrid,double>::interpolationDerivative(q[i], q[j], s, grad);

                for (int l=0; l<6; l++) {
                    Quaternion<double> diff = fdGrad[l];
                    diff -= grad[l];
                    if (diff.two_norm() > 1e-6) {
                        std::cout << "Error in position " << l << ":  fd: " << fdGrad[l]
                                  << "    analytical: " << grad[l] << std::endl;
                    }

                }

                // ///////////////////////////////////////////////////////////
                //   Second: test the interpolated velocity vector
                // ///////////////////////////////////////////////////////////

                for (const double intervalLength : {1.0/3, 2.0/3, 3.0/3, 4.0/3, 5.0/3, 6.0/3, 7.0/3})
                {
                    Dune::FieldVector<double,3> variation;

                    for (int m=0; m<3; m++) {
                        variation = 0;
                        variation[m] = eps;
                        fdGrad[m] =  Rotation<double,3>::interpolateDerivative(q[i].mult(Rotation<double,3>::exp(SkewMatrix<double,3>(variation))),
                                                                               q[j], s);
                        variation = 0;
                        variation[m] = -eps;
                        fdGrad[m] -= Rotation<double,3>::interpolateDerivative(q[i].mult(Rotation<double,3>::exp(SkewMatrix<double,3>(variation))),
                                                                               q[j], s);
                        fdGrad[m] /= 2*eps;

                        variation = 0;
                        variation[m] = eps;
                        fdGrad[3+m] =  Rotation<double,3>::interpolateDerivative(q[i], q[j].mult(Rotation<double,3>::exp(SkewMatrix<double,3>(variation))), s);
                        variation = 0;
                        variation[m] = -eps;
                        fdGrad[3+m] -= Rotation<double,3>::interpolateDerivative(q[i], q[j].mult(Rotation<double,3>::exp(SkewMatrix<double,3>(variation))), s);
                        fdGrad[3+m] /= 2*eps;

                    }

                    // Scale the finite difference gradient with the interval length
                    for (auto& parDer : fdGrad)
                        parDer /= intervalLength;

                    // Compute analytical velocity vector gradient
                    RodLocalStiffness<OneDGrid,double>::interpolationVelocityDerivative(q[i], q[j], s*intervalLength, intervalLength, grad);

                    for (int m=0; m<6; m++) {
                        Quaternion<double> diff = fdGrad[m];
                        diff -= grad[m];
                        if (diff.two_norm() > 1e-6) {
                            std::cout << "Error in velocity " << m
                                      << ":  s = " << s << " of (" << intervalLength << ")"
                                      << "   fd: " << fdGrad[m] << "    analytical: " << grad[m] << std::endl;
                        }

                    }

                }

            }

        }

    }

}



void testRotation(Rotation<double,3> q)
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
    Rotation<double,3> newQ;
    newQ.set(matrix);

    Rotation<double,3> diff = newQ;
    diff -= q;

    Rotation<double,3> sum  = newQ;
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

                        Rotation<double,3> q2(Quaternion<double>(i,j,k,l));
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

    Quaternion<double> Bq[4];
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

    Tensor3<double,4,3,3> derivative = Rotation<double,3>::derivativeOfMatrixToQuaternion(matrix);

    const double eps = 1e-8;
    Tensor3<double,4,3,3> derivativeFD;

    for (size_t i=0; i<3; i++)
    {
      for (size_t j=0; j<3; j++)
      {
        auto forwardMatrix = matrix;
        forwardMatrix[i][j] += eps;
        auto backwardMatrix = matrix;
        backwardMatrix[i][j] -= eps;

        Rotation<double,3> forwardRotation, backwardRotation;
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
bool testInterpolation(const Rotation<double, 3>& a, const Rotation<double, 3>& b) {

    // Compute difference on T_a SO(3)
    Rotation<double, 3> newB = Rotation<double, 3>::interpolate(a, b, 1.0);

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
    std::vector<Rotation<double,3> > testPoints;
    ValueFactory<Rotation<double,3> >::get(testPoints);

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

    // //////////////////////////////////////////////
    //   Test derivative of interpolated position
    // //////////////////////////////////////////////
    testDerivativeOfInterpolatedPosition();

    return not passed;

 } catch (Exception e) {

    std::cout << e << std::endl;

 }
