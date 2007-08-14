#include <config.h>

#include <dune/common/bitfield.hh>
#include <dune/common/configparser.hh>

#include <dune/grid/onedgrid.hh>

#include <dune/istl/io.hh>


#include "../solver/iterativesolver.hh"

#include "../common/energynorm.hh"

#include "src/configuration.hh"
#include "src/roddifference.hh"
#include "src/rodwriter.hh"
#include "src/quaternion.hh"
#include "src/rodassembler.hh"
#include "src/rodsolver.hh"

// Number of degrees of freedom: 
// 7 (x, y, z, q_1, q_2, q_3, q_4) for a spatial rod
const int blocksize = 6;

using namespace Dune;
using std::string;

double computeEnergyNormSquared(const BlockVector<FieldVector<double,6> >& x,
                                const BCRSMatrix<FieldMatrix<double, 6, 6> >& matrix)
{
    BlockVector<FieldVector<double, 6> > tmp(x.size());
    tmp = 0;
    matrix.umv(x,tmp);
    return x*tmp;
}

int main (int argc, char *argv[]) try
{
    typedef std::vector<Configuration> SolutionType;

    // parse data file
    ConfigParser parameterSet;
    if (argc==2)
        parameterSet.parseFile(argv[1]);
    else
        parameterSet.parseFile("rod3d.parset");

    // read solver settings
    const int numLevels        = parameterSet.get("numLevels", int(1));
    const double tolerance        = parameterSet.get("tolerance", double(0));
    const int maxTrustRegionSteps   = parameterSet.get("maxNewtonSteps", int(0));
    const int multigridIterations   = parameterSet.get("numIt", int(0));
    const int nu1              = parameterSet.get("nu1", int(0));
    const int nu2              = parameterSet.get("nu2", int(0));
    const int mu               = parameterSet.get("mu", int(0));
    const int baseIterations      = parameterSet.get("baseIt", int(0));
    const double mgTolerance        = parameterSet.get("mgTolerance", double(0));
    const double baseTolerance    = parameterSet.get("baseTolerance", double(0));
    const double initialTrustRegionRadius = parameterSet.get("initialTrustRegionRadius", double(1));
    const int numRodBaseElements = parameterSet.get("numRodBaseElements", int(0));
    const bool instrumented      = parameterSet.get("instrumented", int(0));
    std::string resultPath           = parameterSet.get("resultPath", "");

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
        x[i].r[0] = 0;
        x[i].r[1] = 0;
        x[i].r[2] = double(i)/(x.size()-1);
        x[i].q = Quaternion<double>::identity();
    }

    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////
    x.back().r[0] = parameterSet.get("dirichletValueX", double(0));
    x.back().r[1] = parameterSet.get("dirichletValueY", double(0));
    x.back().r[2] = parameterSet.get("dirichletValueZ", double(0));

    FieldVector<double,3> axis;
    axis[0] = parameterSet.get("dirichletAxisX", double(0));
    axis[1] = parameterSet.get("dirichletAxisY", double(0));
    axis[2] = parameterSet.get("dirichletAxisZ", double(0));
    double angle = parameterSet.get("dirichletAngle", double(0));

    x.back().q = Quaternion<double>(axis, M_PI*angle/180);

    std::cout << "Left boundary orientation:" << std::endl;
    std::cout << "director 0:  " << x[0].q.director(0) << std::endl;
    std::cout << "director 1:  " << x[0].q.director(1) << std::endl;
    std::cout << "director 2:  " << x[0].q.director(2) << std::endl;
    std::cout << std::endl;
    std::cout << "Right boundary orientation:" << std::endl;
    std::cout << "director 0:  " << x[x.size()-1].q.director(0) << std::endl;
    std::cout << "director 1:  " << x[x.size()-1].q.director(1) << std::endl;
    std::cout << "director 2:  " << x[x.size()-1].q.director(2) << std::endl;

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
    //rodAssembler.setParameters(0,0,0,1,1,1);

    RodSolver<GridType> rodSolver;
    rodSolver.setup(grid, 
                    &rodAssembler,
                    x,
                    tolerance,
                    maxTrustRegionSteps,
                    initialTrustRegionRadius,
                    multigridIterations,
                    mgTolerance,
                    mu, nu1, nu2,
                    baseIterations,
                    baseTolerance,
                    instrumented);

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
    writeRod(x, resultPath + "rod3d.result");
    BlockVector<FieldVector<double, 6> > strain(x.size()-1);
    rodAssembler.getStrain(x,strain);
    //std::cout << strain << std::endl;
    //exit(0);

    //writeRod(x, "rod3d.strain");

    // //////////////////////////////////////////////////////////
    //   Recompute and compare against exact solution
    // //////////////////////////////////////////////////////////
    
    SolutionType exactSolution = x;

    // //////////////////////////////////////////////////////////
    //   Compute hessian of the rod functional at the exact solution
    //   for use of the energy norm it creates.
    // //////////////////////////////////////////////////////////

    BCRSMatrix<FieldMatrix<double, 6, 6> > hessian;
    MatrixIndexSet indices(exactSolution.size(), exactSolution.size());
    rodAssembler.getNeighborsPerVertex(indices);
    indices.exportIdx(hessian);
    rodAssembler.assembleMatrix(exactSolution, hessian);


    double error = std::numeric_limits<double>::max();
    double oldError = 0;

    SolutionType intermediateSolution(x.size());

    // Create statistics file
    std::ofstream statisticsFile((resultPath + "trStatistics").c_str());

    // Compute error of the initial 3d solution
    
    // This should really be exactSol-initialSol, but we're starting
    // from zero anyways
    //oldError += computeEnergyNormSquared(exactSol3d, *hessian3d);
    
#warning Rod error still missing

    oldError = std::sqrt(oldError);

    

    int i;
    for (i=0; i<maxTrustRegionSteps; i++) {
        
        // /////////////////////////////////////////////////////
        //   Read iteration from file
        // /////////////////////////////////////////////////////
        char iSolFilename[100];
        sprintf(iSolFilename, "tmp/intermediateSolution_%04d", i);
            
        FILE* fp = fopen(iSolFilename, "rb");
        if (!fp)
            DUNE_THROW(IOError, "Couldn't open intermediate solution '" << iSolFilename << "'");
        for (int j=0; j<intermediateSolution.size(); j++) {
            fread(&intermediateSolution[j].r, sizeof(double), 3, fp);
            fread(&intermediateSolution[j].q, sizeof(double), 4, fp);
        }
        
        fclose(fp);



        // /////////////////////////////////////////////////////
        //   Compute error
        // /////////////////////////////////////////////////////
        typedef BlockVector<FieldVector<double,6> > RodDifferenceType;
        RodDifferenceType rodDifference = computeRodDifference(exactSolution, intermediateSolution);
        
        error = std::sqrt(computeEnergyNormSquared(rodDifference, hessian));
        

        double convRate = error / oldError;

        // Output
        std::cout << "Trust-region iteration: " << i << "  error : " << error << ",      "
                  << "convrate " << convRate << std::endl;
        statisticsFile << i << "  " << error << "  " << convRate << std::endl;

        if (error < 1e-12)
          break;

        oldError = error;
        
    }            


    // //////////////////////////////
 } catch (Exception e) {

    std::cout << e << std::endl;

 }
