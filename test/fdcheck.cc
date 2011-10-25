#include <config.h>

#include <dune/grid/onedgrid.hh>

#include <dune/istl/io.hh>

#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/quaternion.hh>
#include <dune/gfe/rodassembler.hh>

#include "fdcheck.hh"

// Number of degrees of freedom: 
// 7 (x, y, z, q_1, q_2, q_3, q_4) for a spatial rod
const int blocksize = 6;

using namespace Dune;

void testDDExp()
{
    std::tr1::array<FieldVector<double,3>, 125> v;
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

                    FieldVector<double,3> ffV = v[i];      ffV[j] += eps;     ffV[k] += eps;
                    FieldVector<double,3> fbV = v[i];      fbV[j] += eps;     fbV[k] -= eps;
                    FieldVector<double,3> bfV = v[i];      bfV[j] -= eps;     bfV[k] += eps;
                    FieldVector<double,3> bbV = v[i];      bbV[j] -= eps;     bbV[k] -= eps;

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
        Quaternion<double>::DDexp(v[i], ddExp);

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
    std::tr1::array<Quaternion<double>, 6> q;

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

                std::tr1::array<Quaternion<double>,6> fdGrad;

                // ///////////////////////////////////////////////////////////
                //   First: test the interpolated position
                // ///////////////////////////////////////////////////////////
                fdGrad[0] =  Quaternion<double>::interpolate(q[i].mult(Quaternion<double>::exp(eps,0,0)), q[j], s);
                fdGrad[0] -= Quaternion<double>::interpolate(q[i].mult(Quaternion<double>::exp(-eps,0,0)), q[j], s);
                fdGrad[0] /= 2*eps;

                fdGrad[1] =  Quaternion<double>::interpolate(q[i].mult(Quaternion<double>::exp(0,eps,0)), q[j], s);
                fdGrad[1] -= Quaternion<double>::interpolate(q[i].mult(Quaternion<double>::exp(0,-eps,0)), q[j], s);
                fdGrad[1] /= 2*eps;

                fdGrad[2] =  Quaternion<double>::interpolate(q[i].mult(Quaternion<double>::exp(0,0,eps)), q[j], s);
                fdGrad[2] -= Quaternion<double>::interpolate(q[i].mult(Quaternion<double>::exp(0,0,-eps)), q[j], s);
                fdGrad[2] /= 2*eps;

                fdGrad[3] =  Quaternion<double>::interpolate(q[i], q[j].mult(Quaternion<double>::exp(eps,0,0)), s);
                fdGrad[3] -= Quaternion<double>::interpolate(q[i], q[j].mult(Quaternion<double>::exp(-eps,0,0)), s);
                fdGrad[3] /= 2*eps;

                fdGrad[4] =  Quaternion<double>::interpolate(q[i], q[j].mult(Quaternion<double>::exp(0,eps,0)), s);
                fdGrad[4] -= Quaternion<double>::interpolate(q[i], q[j].mult(Quaternion<double>::exp(0,-eps,0)), s);
                fdGrad[4] /= 2*eps;

                fdGrad[5] =  Quaternion<double>::interpolate(q[i], q[j].mult(Quaternion<double>::exp(0,0,eps)), s);
                fdGrad[5] -= Quaternion<double>::interpolate(q[i], q[j].mult(Quaternion<double>::exp(0,0,-eps)), s);
                fdGrad[5] /= 2*eps;

                // Compute analytical gradient
                std::tr1::array<Quaternion<double>,6> grad;
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
                    
                    fdGrad[0] =  Quaternion<double>::interpolateDerivative(q[i].mult(Quaternion<double>::exp(eps,0,0)), 
                                                                           q[j], s, intervalLength);
                    fdGrad[0] -= Quaternion<double>::interpolateDerivative(q[i].mult(Quaternion<double>::exp(-eps,0,0)), 
                                                                           q[j], s, intervalLength);
                    fdGrad[0] /= 2*eps;
                    
                    fdGrad[1] =  Quaternion<double>::interpolateDerivative(q[i].mult(Quaternion<double>::exp(0,eps,0)), 
                                                                           q[j], s, intervalLength);
                    fdGrad[1] -= Quaternion<double>::interpolateDerivative(q[i].mult(Quaternion<double>::exp(0,-eps,0)), 
                                                                           q[j], s, intervalLength);
                    fdGrad[1] /= 2*eps;
                    
                    fdGrad[2] =  Quaternion<double>::interpolateDerivative(q[i].mult(Quaternion<double>::exp(0,0,eps)), 
                                                                           q[j], s, intervalLength);
                    fdGrad[2] -= Quaternion<double>::interpolateDerivative(q[i].mult(Quaternion<double>::exp(0,0,-eps)), 
                                                                           q[j], s, intervalLength);
                    fdGrad[2] /= 2*eps;
                    
                    fdGrad[3] =  Quaternion<double>::interpolateDerivative(q[i], q[j].mult(Quaternion<double>::exp(eps,0,0)), s, intervalLength);
                    fdGrad[3] -= Quaternion<double>::interpolateDerivative(q[i], q[j].mult(Quaternion<double>::exp(-eps,0,0)), s, intervalLength);
                    fdGrad[3] /= 2*eps;
                    
                    fdGrad[4] =  Quaternion<double>::interpolateDerivative(q[i], q[j].mult(Quaternion<double>::exp(0,eps,0)), s, intervalLength);
                    fdGrad[4] -= Quaternion<double>::interpolateDerivative(q[i], q[j].mult(Quaternion<double>::exp(0,-eps,0)), s, intervalLength);
                    fdGrad[4] /= 2*eps;
                    
                    fdGrad[5] =  Quaternion<double>::interpolateDerivative(q[i], q[j].mult(Quaternion<double>::exp(0,0,eps)), s, intervalLength);
                    fdGrad[5] -= Quaternion<double>::interpolateDerivative(q[i], q[j].mult(Quaternion<double>::exp(0,0,-eps)), s, intervalLength);
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


int main (int argc, char *argv[]) try
{
    // //////////////////////////////////////////////
    //   Test second derivative of exp
    // //////////////////////////////////////////////
    testDDExp();

    // //////////////////////////////////////////////
    //   Test derivative of interpolated position
    // //////////////////////////////////////////////
    testDerivativeOfInterpolatedPosition();
    exit(0);
    typedef std::vector<Configuration> SolutionType;

    // ///////////////////////////////////////
    //    Create the grid
    // ///////////////////////////////////////
    typedef OneDGrid GridType;
    GridType grid(1, 0, 1);

    SolutionType x(grid.size(1));

    // //////////////////////////
    //   Initial solution
    // //////////////////////////
    FieldVector<double,3>  xAxis(0);
    xAxis[0] = 1;
    FieldVector<double,3>  zAxis(0);
    zAxis[2] = 1;

    for (size_t i=0; i<x.size(); i++) {
        x[i].r[0] = 0;    // x
        x[i].r[1] = 0;                 // y
        x[i].r[2] = double(i)/(x.size()-1);                 // z
        //x[i].r[2] = i+5;
        x[i].q = Quaternion<double>::identity();
        //x[i].q = Quaternion<double>(zAxis, M_PI/2 * double(i)/(x.size()-1));
    }

    //x.back().r[1] = 0.1;
    //x.back().r[2] = 2;
    //x.back().q = Quaternion<double>(zAxis, M_PI/4);

    std::cout << "Left boundary orientation:" << std::endl;
    std::cout << "director 0:  " << x[0].q.director(0) << std::endl;
    std::cout << "director 1:  " << x[0].q.director(1) << std::endl;
    std::cout << "director 2:  " << x[0].q.director(2) << std::endl;
    std::cout << std::endl;
    std::cout << "Right boundary orientation:" << std::endl;
    std::cout << "director 0:  " << x[x.size()-1].q.director(0) << std::endl;
    std::cout << "director 1:  " << x[x.size()-1].q.director(1) << std::endl;
    std::cout << "director 2:  " << x[x.size()-1].q.director(2) << std::endl;
//     exit(0);

    //x[0].r[2] = -1;

    // ///////////////////////////////////////////
    //   Create a solver for the rod problem
    // ///////////////////////////////////////////
    RodAssembler<GridType> rodAssembler(grid);
    //rodAssembler.setShapeAndMaterial(0.01, 0.0001, 0.0001, 2.5e5, 0.3);
    //rodAssembler.setParameters(0,0,0,0,1,0);
    rodAssembler.setParameters(0,0,100,0,0,0);

    std::cout << "Energy: " << rodAssembler.computeEnergy(x) << std::endl;

    double pos = (argc==2) ? atof(argv[1]) : 0.5;

    FieldVector<double,1> shapeGrad[2];
    shapeGrad[0] = -1;
    shapeGrad[1] =  1;

    FieldVector<double,1> shapeFunction[2];
    shapeFunction[0] = 1-pos;
    shapeFunction[1] =  pos;

    exit(0);
    BlockVector<FieldVector<double,6> > rhs(x.size());
    BCRSMatrix<FieldMatrix<double,6,6> > hessianMatrix;
    MatrixIndexSet indices(grid.size(1), grid.size(1));
    rodAssembler.getNeighborsPerVertex(indices);
    indices.exportIdx(hessianMatrix);

    rodAssembler.assembleGradient(x, rhs);
    rodAssembler.assembleMatrix(x, hessianMatrix);
    
    gradientFDCheck(x, rhs, rodAssembler);
    hessianFDCheck(x, hessianMatrix, rodAssembler);
        
    // //////////////////////////////
 } catch (Exception e) {

    std::cout << e << std::endl;

 }
