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



int main (int argc, char *argv[]) try
{
    typedef std::vector<RigidBodyMotion<double,3> > SolutionType;

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
    RodAssembler<GridType,3> rodAssembler(grid);
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
