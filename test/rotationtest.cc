#include <config.h>

#include <iostream>

#include <dune/common/fmatrix.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/svd.hh>
#include "valuefactory.hh"

using namespace Dune;

void testDDExp()
{
    array<FieldVector<double,3>, 125> v;
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
        Dune::array<Dune::FieldMatrix<double,3,3>, 4> fdDDExp;

        for (int j=0; j<3; j++) {

            for (int k=0; k<3; k++) {

                if (j==k) {

                    Quaternion<double> forwardQ  = Quaternion<double>::exp(v[i][0] + (j==0)*eps,
                                                                           v[i][1] + (j==1)*eps,
                                                                           v[i][2] + (j==2)*eps);
                    Quaternion<double> centerQ   = Quaternion<double>::exp(v[i][0],v[i][1],v[i][2]);
                    Quaternion<double> backwardQ = Quaternion<double>::exp(v[i][0] - (j==0)*eps,
                                                                           v[i][1] - (j==1)*eps,
                                                                           v[i][2] - (j==2)*eps);

                    for (int l=0; l<4; l++)
                        fdDDExp[l][j][j] = (forwardQ[l] - 2*centerQ[l] + backwardQ[l]) / (eps*eps);


                } else {

                    SkewMatrix<double,3> ffV(v[i]);      ffV.axial()[j] += eps;     ffV.axial()[k] += eps;
                    SkewMatrix<double,3> fbV(v[i]);      fbV.axial()[j] += eps;     fbV.axial()[k] -= eps;
                    SkewMatrix<double,3> bfV(v[i]);      bfV.axial()[j] -= eps;     bfV.axial()[k] += eps;
                    SkewMatrix<double,3> bbV(v[i]);      bbV.axial()[j] -= eps;     bbV.axial()[k] -= eps;

                    Quaternion<double> forwardForwardQ = Quaternion<double>::exp(ffV);
                    Quaternion<double> forwardBackwardQ = Quaternion<double>::exp(fbV);
                    Quaternion<double> backwardForwardQ = Quaternion<double>::exp(bfV);
                    Quaternion<double> backwardBackwardQ = Quaternion<double>::exp(bbV);

                    for (int l=0; l<4; l++)
                        fdDDExp[l][j][k] = (forwardForwardQ[l] + backwardBackwardQ[l]
                                            - forwardBackwardQ[l] - backwardForwardQ[l]) / (4*eps*eps);

                }

            }

        }

        // Compute analytical second derivative of exp
        Dune::array<Dune::FieldMatrix<double,3,3>, 4> ddExp;
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
    array<Quaternion<double>, 6> q;

    FieldVector<double,3>  xAxis(0);    xAxis[0] = 1;
    FieldVector<double,3>  yAxis(0);    yAxis[1] = 1;
    FieldVector<double,3>  zAxis(0);    zAxis[2] = 1;

    q[0] = Quaternion<double>(xAxis, 0);
    q[1] = Quaternion<double>(xAxis, M_PI/2);
    q[2] = Quaternion<double>(yAxis, 0);
    q[3] = Quaternion<double>(yAxis, M_PI/2);
    q[4] = Quaternion<double>(zAxis, 0);
    q[5] = Quaternion<double>(zAxis, M_PI/2);

    double eps = 1e-7;

    for (int i=0; i<6; i++) {

        for (int j=0; j<6; j++) {

            for (int k=0; k<7; k++) {

                double s = k/6.0;

                array<Quaternion<double>,6> fdGrad;

                // ///////////////////////////////////////////////////////////
                //   First: test the interpolated position
                // ///////////////////////////////////////////////////////////
                fdGrad[0] =  Rotation<double,3>::interpolate(q[i].mult(Quaternion<double>::exp(eps,0,0)), q[j], s);
                fdGrad[0] -= Rotation<double,3>::interpolate(q[i].mult(Quaternion<double>::exp(-eps,0,0)), q[j], s);
                fdGrad[0] /= 2*eps;

                fdGrad[1] =  Rotation<double,3>::interpolate(q[i].mult(Quaternion<double>::exp(0,eps,0)), q[j], s);
                fdGrad[1] -= Rotation<double,3>::interpolate(q[i].mult(Quaternion<double>::exp(0,-eps,0)), q[j], s);
                fdGrad[1] /= 2*eps;

                fdGrad[2] =  Rotation<double,3>::interpolate(q[i].mult(Quaternion<double>::exp(0,0,eps)), q[j], s);
                fdGrad[2] -= Rotation<double,3>::interpolate(q[i].mult(Quaternion<double>::exp(0,0,-eps)), q[j], s);
                fdGrad[2] /= 2*eps;

                fdGrad[3] =  Rotation<double,3>::interpolate(q[i], q[j].mult(Quaternion<double>::exp(eps,0,0)), s);
                fdGrad[3] -= Rotation<double,3>::interpolate(q[i], q[j].mult(Quaternion<double>::exp(-eps,0,0)), s);
                fdGrad[3] /= 2*eps;

                fdGrad[4] =  Rotation<double,3>::interpolate(q[i], q[j].mult(Quaternion<double>::exp(0,eps,0)), s);
                fdGrad[4] -= Rotation<double,3>::interpolate(q[i], q[j].mult(Quaternion<double>::exp(0,-eps,0)), s);
                fdGrad[4] /= 2*eps;

                fdGrad[5] =  Rotation<double,3>::interpolate(q[i], q[j].mult(Quaternion<double>::exp(0,0,eps)), s);
                fdGrad[5] -= Rotation<double,3>::interpolate(q[i], q[j].mult(Quaternion<double>::exp(0,0,-eps)), s);
                fdGrad[5] /= 2*eps;

                // Compute analytical gradient
                array<Quaternion<double>,6> grad;
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

                for (int l=1; l<7; l++) {

                    double intervalLength = l/(double(3));
                    
                    fdGrad[0] =  Rotation<double,3>::interpolateDerivative(q[i].mult(Quaternion<double>::exp(eps,0,0)), 
                                                                           q[j], s, intervalLength);
                    fdGrad[0] -= Rotation<double,3>::interpolateDerivative(q[i].mult(Quaternion<double>::exp(-eps,0,0)), 
                                                                           q[j], s, intervalLength);
                    fdGrad[0] /= 2*eps;
                    
                    fdGrad[1] =  Rotation<double,3>::interpolateDerivative(q[i].mult(Quaternion<double>::exp(0,eps,0)), 
                                                                           q[j], s, intervalLength);
                    fdGrad[1] -= Rotation<double,3>::interpolateDerivative(q[i].mult(Quaternion<double>::exp(0,-eps,0)), 
                                                                           q[j], s, intervalLength);
                    fdGrad[1] /= 2*eps;
                    
                    fdGrad[2] =  Rotation<double,3>::interpolateDerivative(q[i].mult(Quaternion<double>::exp(0,0,eps)), 
                                                                           q[j], s, intervalLength);
                    fdGrad[2] -= Rotation<double,3>::interpolateDerivative(q[i].mult(Quaternion<double>::exp(0,0,-eps)), 
                                                                           q[j], s, intervalLength);
                    fdGrad[2] /= 2*eps;
                    
                    fdGrad[3] =  Rotation<double,3>::interpolateDerivative(q[i], q[j].mult(Quaternion<double>::exp(eps,0,0)), s, intervalLength);
                    fdGrad[3] -= Rotation<double,3>::interpolateDerivative(q[i], q[j].mult(Quaternion<double>::exp(-eps,0,0)), s, intervalLength);
                    fdGrad[3] /= 2*eps;
                    
                    fdGrad[4] =  Rotation<double,3>::interpolateDerivative(q[i], q[j].mult(Quaternion<double>::exp(0,eps,0)), s, intervalLength);
                    fdGrad[4] -= Rotation<double,3>::interpolateDerivative(q[i], q[j].mult(Quaternion<double>::exp(0,-eps,0)), s, intervalLength);
                    fdGrad[4] /= 2*eps;
                    
                    fdGrad[5] =  Rotation<double,3>::interpolateDerivative(q[i], q[j].mult(Quaternion<double>::exp(0,0,eps)), s, intervalLength);
                    fdGrad[5] -= Rotation<double,3>::interpolateDerivative(q[i], q[j].mult(Quaternion<double>::exp(0,0,-eps)), s, intervalLength);
                    fdGrad[5] /= 2*eps;
                    
                    // Compute analytical velocity vector gradient
                    RodLocalStiffness<OneDGrid,double>::interpolationVelocityDerivative(q[i], q[j], s, intervalLength, grad);
                    
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

    Quaternion<double> diff = newQ;
    diff -= q;

    Quaternion<double> sum  = newQ;
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

}

int main (int argc, char *argv[]) try
{
    std::vector<Rotation<double,3> > testPoints;
    ValueFactory<Rotation<double,3> >::get(testPoints);
    
    int nTestPoints = testPoints.size();
    
    // Test each element in the list
    for (int i=0; i<nTestPoints; i++)
        testRotation(testPoints[i]);

    // //////////////////////////////////////////////
    //   Test second derivative of exp
    // //////////////////////////////////////////////
    testDDExp();

    // //////////////////////////////////////////////
    //   Test derivative of interpolated position
    // //////////////////////////////////////////////
    testDerivativeOfInterpolatedPosition();

 } catch (Exception e) {

    std::cout << e << std::endl;

 }
