#include <config.h>

//#define DUNE_EXPRESSIONTEMPLATES
#include <dune/grid/onedgrid.hh>

#include <dune/istl/io.hh>

#include <dune/common/bitfield.hh>
#include "src/quaternion.hh"

#include "src/rodassembler.hh"

#include "src/configuration.hh"
#include "src/rodwriter.hh"

// Number of degrees of freedom: 
// 7 (x, y, z, q_1, q_2, q_3, q_4) for a spatial rod
const int blocksize = 6;

using namespace Dune;
using std::string;


int main (int argc, char *argv[]) try
{
    // Some types that I need
    typedef BCRSMatrix<FieldMatrix<double, blocksize, blocksize> > MatrixType;
    typedef BlockVector<FieldVector<double, blocksize> >           CorrectionType;
    typedef std::vector<Configuration>                              SolutionType;

    // Problem settings
    const int numRodBaseElements = 1;
    
    // ///////////////////////////////////////
    //    Create the grid
    // ///////////////////////////////////////
    typedef OneDGrid GridType;
    GridType grid(numRodBaseElements, 0, 1);

    SolutionType x(grid.size(1));

    // //////////////////////////
    //   Initial solution
    // //////////////////////////

    for (size_t i=0; i<x.size(); i++) {
        x[i].r[0] = 0;    // x
        x[i].r[1] = 0;
        x[i].r[2] = double(i)/(x.size()-1);                 // z
        x[i].q = Quaternion<double>::identity();
        //x[i].q = Quaternion<double>(zAxis, (double(i)*M_PI)/(2*(x.size()-1)) );
    }

    FieldVector<double,3> zAxis(0);  zAxis[2]=1;
    x.back().q = Quaternion<double>(zAxis, M_PI/4);

    // /////////////////////////////////////////////////////////////////////
    //   Create a second, rotated copy of the configuration
    // /////////////////////////////////////////////////////////////////////

    FieldVector<double,3> displacement;
    displacement[0] = 0;
    displacement[1] = 0;
    displacement[2] = 0;

    FieldVector<double,3> axis(0);  axis[0]=1;
    Quaternion<double> rotation(axis,M_PI/2);
//     std::cout << "Rotation:" << std::endl;
//     std::cout << "director 0:  " << rotation.director(0) << std::endl;
//     std::cout << "director 1:  " << rotation.director(1) << std::endl;
//     std::cout << "director 2:  " << rotation.director(2) << std::endl;

    SolutionType rotatedX = x;

    for (size_t i=0; i<rotatedX.size(); i++) {

        rotatedX[i].r = rotation.rotate(x[i].r);
        rotatedX[i].r += displacement;

        rotatedX[i].q = rotation.mult(x[i].q);

//         std::cout << "Vertex " << i << ": (" << rotatedX[i].r << ")" << std::endl;
// //         std::cout << "Right boundary orientation:" << std::endl;
//         std::cout << "director 0:  " << rotatedX[i].q.director(0) << std::endl;
//         std::cout << "director 1:  " << rotatedX[i].q.director(1) << std::endl;
//         std::cout << "director 2:  " << rotatedX[i].q.director(2) << std::endl;
    }

    writeRod(x,"rod");
    writeRod(rotatedX, "rotated");

    RodLocalStiffness<GridType,double> assembler;

    for (int i=1; i<2; i++) {

        double p = double(i)/2;

        assembler.getStrain(x,grid.lbegin<0>(0), p);
        assembler.getStrain(rotatedX,grid.lbegin<0>(0), p);

    }

 } catch (Exception e) {

    std::cout << e << std::endl;

 }
