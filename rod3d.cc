#include <config.h>

//#define DUNE_EXPRESSIONTEMPLATES
#include <dune/grid/onedgrid.hh>

#include <dune/istl/io.hh>

#include <dune/common/bitfield.hh>
#include "src/quaternion.hh"

#include "src/rodassembler.hh"
#include "src/rodsolver.hh"

#include "../solver/iterativesolver.hh"

#include "../common/geomestimator.hh"
#include "../common/energynorm.hh"

#include <dune/common/configparser.hh>
#include "src/configuration.hh"
#include "src/rodwriter.hh"

// Number of degrees of freedom: 
// 7 (x, y, z, q_1, q_2, q_3, q_4) for a spatial rod
const int blocksize = 6;

using namespace Dune;
using std::string;

int main (int argc, char *argv[]) try
{
    typedef std::vector<Configuration> SolutionType;

    // parse data file
    ConfigParser parameterSet;
    parameterSet.parseFile("rod3d.parset");

    // read solver settings
    const int numLevels        = parameterSet.get("numLevels", int(1));
    const int maxTrustRegionSteps   = parameterSet.get("maxNewtonSteps", int(0));
    const int multigridIterations   = parameterSet.get("numIt", int(0));
    const int nu1              = parameterSet.get("nu1", int(0));
    const int nu2              = parameterSet.get("nu2", int(0));
    const int mu               = parameterSet.get("mu", int(0));
    const int baseIterations      = parameterSet.get("baseIt", int(0));
    const double mgTolerance        = parameterSet.get("tolerance", double(0));
    const double baseTolerance    = parameterSet.get("baseTolerance", double(0));
    const double initialTrustRegionRadius = parameterSet.get("initialTrustRegionRadius", double(1));
    const int numRodBaseElements = parameterSet.get("numRodBaseElements", int(0));
    
    // ///////////////////////////////////////
    //    Create the grid
    // ///////////////////////////////////////
    typedef OneDGrid GridType;
    GridType grid(numRodBaseElements, 0, 1);

    grid.globalRefine(numLevels-1);

    std::vector<BitField> dirichletNodes(1);

    SolutionType x(grid.size(grid.maxLevel(),1));

    // //////////////////////////
    //   Initial solution
    // //////////////////////////

    for (int i=0; i<x.size(); i++) {
        x[i].r[0] = 0;    // x
        x[i].r[1] = 0;                 // y
        x[i].r[2] = double(i)/(x.size()-1);                 // z
        //x[i].r[2] = i+5;
        x[i].q = Quaternion<double>::identity();
    }

//     x[x.size()-1].r[0] = 0;
//     x[x.size()-1].r[1] = 0;
//     x[x.size()-1].r[2] = 0;
#if 1
    FieldVector<double,3>  zAxis(0);
    zAxis[2] = 1;
    x[x.size()-1].q = Quaternion<double>(zAxis, M_PI);
#endif

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

    dirichletNodes.resize(numLevels);
    for (int i=0; i<numLevels; i++) {
        
        dirichletNodes[i].resize( blocksize * grid.size(i,1), false );
        
        for (int j=0; j<blocksize; j++) {
            dirichletNodes[i][j] = true;
            dirichletNodes[i][dirichletNodes[i].size()-1-j] = true;
        }
    }
    
    // ///////////////////////////////////////////
    //   Create a solver for the rod problem
    // ///////////////////////////////////////////
    RodAssembler<GridType> rodAssembler(grid);
    rodAssembler.setShapeAndMaterial(0.01, 0.0001, 0.0001, 2.5e5, 0.3);
    
    RodSolver<GridType> rodSolver;
    rodSolver.setup(grid, 
                    &rodAssembler,
                    x,
                    maxTrustRegionSteps,
                    initialTrustRegionRadius,
                    multigridIterations,
                    mgTolerance,
                    mu, nu1, nu2,
                    baseIterations,
                    baseTolerance);

    // /////////////////////////////////////////////////////
    //   Solve!
    // /////////////////////////////////////////////////////

    std::cout << "Energy: " << rodAssembler.computeEnergy(x) << std::endl;
    
    rodSolver.setInitialSolution(x);
    rodSolver.solve();

    x = rodSolver.getSol();
        
    // //////////////////////////////
    //   Output result
    // //////////////////////////////
    writeRod(x, "rod3d.result");
    BlockVector<FieldVector<double, 6> > strain(x.size()-1);
    rodAssembler.getStrain(x,strain);
    //std::cout << strain << std::endl;
    //exit(0);
    
    writeRod(x, strain, "rod3d.strain");

 } catch (Exception e) {

    std::cout << e << std::endl;

 }
