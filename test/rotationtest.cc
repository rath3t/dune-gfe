#include <config.h>

#include <iostream>

#include <dune/common/fmatrix.hh>

#include "src/quaternion.hh"
#include "src/svd.hh"

using namespace Dune;

void testUnitQuaternion(Quaternion<double> q)
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
    Quaternion<double> newQ;
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

                        Quaternion<double> q2(i,j,k,l);
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
    // Make a few random unit quaternions and pipe them into the test
    for (int i=-5; i<5; i++)
        for (int j=-5; j<5; j++)
            for (int k=-5; k<5; k++)
                for (int l=-5; l<5; l++)
                    if (i!=0 || j!=0 || k!=0 || l!=0)
                        testUnitQuaternion(Quaternion<double>(i,j,k,l));
    

 } catch (Exception e) {

    std::cout << e << std::endl;

 }
