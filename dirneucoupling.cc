#include <config.h>

#define HAVE_IPOPT

#include <dune/grid/onedgrid.hh>
#include <dune/grid/uggrid.hh>

#include <dune/disc/elasticity/linearelasticityassembler.hh>
#include <dune/disc/operators/p1operator.hh>
#include <dune/istl/io.hh>
#include <dune/grid/io/file/amirameshreader.hh>
#include <dune/grid/io/file/amirameshwriter.hh>


#include <dune/common/bitfield.hh>
#include <dune/common/configparser.hh>

#include "../common/multigridstep.hh"
#include "../solver/iterativesolver.hh"
#include "../common/projectedblockgsstep.hh"
#include "../common/linearipopt.hh"
#include "../common/readbitfield.hh"
#include "../common/energynorm.hh"
#include "../common/boundarypatch.hh"
#include "../common/prolongboundarypatch.hh"
#include "../common/neumannassembler.hh"

#include "src/quaternion.hh"
#include "src/rodassembler.hh"
#include "src/configuration.hh"
#include "src/averageinterface.hh"
#include "src/rodsolver.hh"
#include "src/roddifference.hh"
#include "src/rodwriter.hh"

// Space dimension
const int dim = 3;

using namespace Dune;
using std::string;

template <class DiscFuncType, int blocksize>
double computeEnergyNormSquared(const DiscFuncType& u, 
                                const BCRSMatrix<FieldMatrix<double,blocksize,blocksize> >& A)
{
    DiscFuncType tmp(u.size());
    tmp = 0;
    A.umv(u, tmp);
    return u*tmp;
}

int main (int argc, char *argv[]) try
{
    // Some types that I need
    typedef BCRSMatrix<FieldMatrix<double, dim, dim> >   MatrixType;
    typedef BlockVector<FieldVector<double, dim> >       VectorType;
    typedef std::vector<Configuration>                   RodSolutionType;
    typedef BlockVector<FieldVector<double, 6> >         RodDifferenceType;

    // parse data file
    ConfigParser parameterSet;
    parameterSet.parseFile("dirneucoupling.parset");

    // read solver settings
    const int minLevel            = parameterSet.get("minLevel", int(0));
    const int maxLevel            = parameterSet.get("maxLevel", int(0));
    const double ddTolerance           = parameterSet.get("ddTolerance", double(0));
    const int maxDirichletNeumannSteps = parameterSet.get("maxDirichletNeumannSteps", int(0));
    const double trTolerance           = parameterSet.get("trTolerance", double(0));
    const int maxTrustRegionSteps = parameterSet.get("maxTrustRegionSteps", int(0));
    const int multigridIterations = parameterSet.get("numIt", int(0));
    const int nu1              = parameterSet.get("nu1", int(0));
    const int nu2              = parameterSet.get("nu2", int(0));
    const int mu               = parameterSet.get("mu", int(0));
    const int baseIterations   = parameterSet.get("baseIt", int(0));
    const double mgTolerance     = parameterSet.get("tolerance", double(0));
    const double baseTolerance = parameterSet.get("baseTolerance", double(0));
    const double initialTrustRegionRadius = parameterSet.get("initialTrustRegionRadius", double(0));
    const double damping       = parameterSet.get("damping", double(1));

    // Problem settings
    std::string path = parameterSet.get("path", "xyz");
    std::string objectName = parameterSet.get("gridFile", "xyz");
    std::string dirichletNodesFile  = parameterSet.get("dirichletNodes", "xyz");
    std::string dirichletValuesFile = parameterSet.get("dirichletValues", "xyz");
    std::string interfaceNodesFile  = parameterSet.get("interfaceNodes", "xyz");
    const int numRodBaseElements = parameterSet.get("numRodBaseElements", int(0));
    
    // ///////////////////////////////////////
    //    Create the rod grid
    // ///////////////////////////////////////
    typedef OneDGrid RodGridType;
    RodGridType rodGrid(numRodBaseElements, 0, 5);

    // ///////////////////////////////////////
    //    Create the grid for the 3d object
    // ///////////////////////////////////////
    typedef UGGrid<dim> GridType;
    GridType grid;
    grid.setRefinementType(GridType::COPY);
    
    AmiraMeshReader<GridType>::read(grid, path + objectName);

    std::vector<BitField> dirichletNodes(1);

    RodSolutionType rodX(rodGrid.size(0,1));

    // //////////////////////////
    //   Initial solution
    // //////////////////////////

    for (int i=0; i<rodX.size(); i++) {
        rodX[i].r = 0.5;
        rodX[i].r[2] = i+5;
        rodX[i].q = Quaternion<double>::identity();
    }

//     rodX[rodX.size()-1].r[0] = 0.5;
//     rodX[rodX.size()-1].r[1] = 0.5;
//     rodX[rodX.size()-1].r[2] = 11;

    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////
    rodX.back().r[0] = parameterSet.get("dirichletValueX", double(0));
    rodX.back().r[1] = parameterSet.get("dirichletValueY", double(0));
    rodX.back().r[2] = parameterSet.get("dirichletValueZ", double(0));

    FieldVector<double,3> axis;
    axis[0] = parameterSet.get("dirichletAxisX", double(0));
    axis[1] = parameterSet.get("dirichletAxisY", double(0));
    axis[2] = parameterSet.get("dirichletAxisZ", double(0));
    double angle = parameterSet.get("dirichletAngle", double(0));

    rodX.back().q = Quaternion<double>(axis, M_PI*angle/180);


//     std::cout << "Left boundary orientation:" << std::endl;
//     std::cout << "director 0:  " << rodX[0].q.director(0) << std::endl;
//     std::cout << "director 1:  " << rodX[0].q.director(1) << std::endl;
//     std::cout << "director 2:  " << rodX[0].q.director(2) << std::endl;
//     std::cout << std::endl;
    std::cout << "Right boundary orientation:" << std::endl;
    std::cout << "director 0:  " << rodX[rodX.size()-1].q.director(0) << std::endl;
    std::cout << "director 1:  " << rodX[rodX.size()-1].q.director(1) << std::endl;
    std::cout << "director 2:  " << rodX[rodX.size()-1].q.director(2) << std::endl;
//     exit(0);

    int toplevel = rodGrid.maxLevel();

    // /////////////////////////////////////////////////////
    //   Determine the Dirichlet nodes
    // /////////////////////////////////////////////////////
    Array<VectorType> dirichletValues;
    dirichletValues.resize(toplevel+1);
    dirichletValues[0].resize(grid.size(0, dim));
    AmiraMeshReader<int>::readFunction(dirichletValues[0], path + dirichletValuesFile);

    std::vector<BoundaryPatch<GridType> > dirichletBoundary;
    dirichletBoundary.resize(maxLevel+1);
    dirichletBoundary[0].setup(grid, 0);
    readBoundaryPatch(dirichletBoundary[0], path + dirichletNodesFile);
    PatchProlongator<GridType>::prolong(dirichletBoundary);

    dirichletNodes.resize(toplevel+1);
    for (int i=0; i<=toplevel; i++) {
        
        dirichletNodes[i].resize( dim*grid.size(i,dim));

        for (int j=0; j<grid.size(i,dim); j++)
            for (int k=0; k<dim; k++)
                dirichletNodes[i][j*dim+k] = dirichletBoundary[i].containsVertex(j);
        
    }
    
    // /////////////////////////////////////////////////////
    //   Determine the interface boundary
    // /////////////////////////////////////////////////////
    std::vector<BoundaryPatch<GridType> > interfaceBoundary;
    interfaceBoundary.resize(maxLevel+1);
    interfaceBoundary[0].setup(grid, 0);
    readBoundaryPatch(interfaceBoundary[0], path + interfaceNodesFile);
    PatchProlongator<GridType>::prolong(interfaceBoundary);

    // //////////////////////////////////////////
    //   Assemble 3d linear elasticity problem
    // //////////////////////////////////////////
    LeafP1Function<GridType,double,dim> u(grid),f(grid);
    LinearElasticityLocalStiffness<GridType,double> lstiff(2.5e5, 0.3);
    LeafP1OperatorAssembler<GridType,double,dim> hessian3d(grid);
    hessian3d.assemble(lstiff,u,f);

    // ////////////////////////////////////////////////////////////
    //    Create solution and rhs vectors
    // ////////////////////////////////////////////////////////////
    
    VectorType x3d(grid.size(toplevel,dim));
    VectorType rhs3d(grid.size(toplevel,dim));

    // No external forces
    rhs3d = 0;

    // Set initial solution
    x3d = 0;
    for (int i=0; i<x3d.size(); i++) 
        for (int j=0; j<dim; j++)
            if (dirichletNodes[toplevel][i*dim+j])
                x3d[i][j] = dirichletValues[toplevel][i][j];

    // ///////////////////////////////////////////
    //   Create a solver for the rod problem
    // ///////////////////////////////////////////
    RodAssembler<RodGridType> rodAssembler(rodGrid);
    rodAssembler.setShapeAndMaterial(1, 1/12, 1/12, 2.5e5, 0.3);

    RodSolver<RodGridType> rodSolver;
    rodSolver.setup(rodGrid, 
                    &rodAssembler,
                    rodX,
                    trTolerance,
                    maxTrustRegionSteps,
                    initialTrustRegionRadius,
                    multigridIterations,
                    mgTolerance,
                    mu, nu1, nu2,
                    baseIterations,
                    baseTolerance,
                    false);

    // ////////////////////////////////
    //   Create a multigrid solver
    // ////////////////////////////////

    // First create a gauss-seidel base solver
    LinearIPOptSolver<VectorType> baseSolver;
    baseSolver.verbosity_ = NumProc::QUIET;
    baseSolver.tolerance_ = baseTolerance;

    // Make pre and postsmoothers
    BlockGSStep<MatrixType, VectorType> presmoother, postsmoother;

    MultigridStep<MatrixType, VectorType> multigridStep(*hessian3d, x3d, rhs3d, 1);

    multigridStep.setMGType(mu, nu1, nu2);
    multigridStep.dirichletNodes_    = &dirichletNodes;
    multigridStep.basesolver_        = &baseSolver;
    multigridStep.presmoother_       = &presmoother;
    multigridStep.postsmoother_      = &postsmoother;    
    multigridStep.verbosity_         = Solver::REDUCED;



    EnergyNorm<MatrixType, VectorType> energyNorm(multigridStep);

    IterativeSolver<MatrixType, VectorType> solver(&multigridStep,
                                                   multigridIterations,
                                                   mgTolerance,
                                                   &energyNorm,
                                                   Solver::QUIET);

    // ////////////////////////////////////
    //   Create the transfer operators
    // ////////////////////////////////////
    for (int k=0; k<multigridStep.mgTransfer_.size(); k++)
        delete(multigridStep.mgTransfer_[k]);
    
    multigridStep.mgTransfer_.resize(toplevel);
    
    for (int i=0; i<multigridStep.mgTransfer_.size(); i++){
        TruncatedMGTransfer<VectorType>* newTransferOp = new TruncatedMGTransfer<VectorType>;
        newTransferOp->setup(grid,i,i+1);
        multigridStep.mgTransfer_[i] = newTransferOp;
    }

    // /////////////////////////////////////////////////////
    //   Dirichlet-Neumann Solver
    // /////////////////////////////////////////////////////

    // Init interface value
    Configuration referenceInterface = rodX[0];
    Configuration lambda = referenceInterface;

    //
    double normOfOldCorrection = 0;

    for (int i=0; i<maxDirichletNeumannSteps; i++) {
        
        std::cout << "----------------------------------------------------" << std::endl;
        std::cout << "      Dirichlet-Neumann Step Number: " << i << std::endl;
        std::cout << "----------------------------------------------------" << std::endl;
        
        // Backup of the current solution for the error computation later on
        VectorType oldSolution3d = x3d;

        // //////////////////////////////////////////////////
        //   Dirichlet step for the rod
        // //////////////////////////////////////////////////

        rodX[0] = lambda;
        rodSolver.setInitialSolution(rodX);
        rodSolver.solve();

        rodX = rodSolver.getSol();

        // ///////////////////////////////////////////////////////////
        //   Extract Neumann values and transfer it to the 3d object
        // ///////////////////////////////////////////////////////////

        BitField couplingBitfield(rodX.size(),false);
        // Using that index 0 is always the left boundary for a uniformly refined OneDGrid
        couplingBitfield[0] = true;
        BoundaryPatch<RodGridType> couplingBoundary(rodGrid, rodGrid.maxLevel(), couplingBitfield);

        FieldVector<double,dim> resultantTorque;
        FieldVector<double,dim> resultantForce  = rodAssembler.getResultantForce(couplingBoundary, rodX, resultantTorque);

        std::cout << "resultant force: " << resultantForce << std::endl;
        std::cout << "resultant torque: " << resultantTorque << std::endl;

        VectorType neumannValues(grid.size(dim));
        // Using that index 0 is always the left boundary for a uniformly refined OneDGrid
        computeAveragePressure<GridType>(resultantForce, resultantTorque, 
                                         interfaceBoundary[grid.maxLevel()], 
                                         rodX[0],
                                         neumannValues);

        rhs3d = 0;
        assembleAndAddNeumannTerm<GridType, VectorType>(interfaceBoundary[grid.maxLevel()],
                                                        neumannValues,
                                                        rhs3d);

        // ///////////////////////////////////////////////////////////
        //   Solve the Neumann problem for the 3d body
        // ///////////////////////////////////////////////////////////
        multigridStep.setProblem(*hessian3d, x3d, rhs3d, grid.maxLevel()+1);
        
        solver.preprocess();
        multigridStep.preprocess();
        
        solver.solve();
        
        x3d = multigridStep.getSol();

        // ///////////////////////////////////////////////////////////
        //   Extract new interface position and orientation
        // ///////////////////////////////////////////////////////////

        Configuration averageInterface;
        computeAverageInterface(interfaceBoundary[toplevel], x3d, averageInterface);

        std::cout << "average interface: " << averageInterface << std::endl;

        // ///////////////////////////////////////////////////////////
        //   Compute new damped interface value
        // ///////////////////////////////////////////////////////////

        for (int j=0; j<dim; j++)
            lambda.r[j] = (1-damping) * lambda.r[j] + damping * (referenceInterface.r[j] + averageInterface.r[j]);
        lambda.q = averageInterface.q.mult(referenceInterface.q);

        // ////////////////////////////////////////////////////////////////////////
        //   Write the two iterates to disk for later convergence rate measurement
        // ////////////////////////////////////////////////////////////////////////

        // First the 3d body
        char iSolFilename[100];
        sprintf(iSolFilename, "tmp/intermediate3dSolution_%04d", i);
            
        FILE* fp = fopen(iSolFilename, "wb");
        if (!fp)
            DUNE_THROW(SolverError, "Couldn't open file " << iSolFilename << " for writing");
            
        for (int j=0; j<x3d.size(); j++)
            for (int k=0; k<dim; k++)
                fwrite(&x3d[j][k], sizeof(double), 1, fp);

        fclose(fp);

        // Then the rod
        char iRodFilename[100];
        sprintf(iRodFilename, "tmp/intermediateRodSolution_%04d", i);

        FILE* fpRod = fopen(iRodFilename, "wb");
        if (!fpRod)
            DUNE_THROW(SolverError, "Couldn't open file " << iRodFilename << " for writing");
            
        for (int j=0; j<rodX.size(); j++) {

            for (int k=0; k<dim; k++)
                fwrite(&rodX[j].r[k], sizeof(double), 1, fpRod);

            for (int k=0; k<4; k++)  // 3d hardwired here!
                fwrite(&rodX[j].q[k], sizeof(double), 1, fpRod);

        }

        fclose(fpRod);

        // ////////////////////////////////////////////
        //   Compute error in the energy norm
        // ////////////////////////////////////////////

        // the 3d body
        double oldNorm = computeEnergyNormSquared(oldSolution3d, *hessian3d);
        oldSolution3d -= x3d;
        double normOfCorrection = computeEnergyNormSquared(oldSolution3d, *hessian3d);

        // the rod \todo missing
#warning Energy error of the rod still missing

        oldNorm = std::sqrt(oldNorm);

        normOfCorrection = std::sqrt(normOfCorrection);

        double relativeError = normOfCorrection / oldNorm;

        double convRate = normOfCorrection / normOfOldCorrection;

        normOfOldCorrection = normOfCorrection;

        // Output
        std::cout << "DD iteration: " << i << "  --  ||u^{n+1} - u^n||: " << relativeError << ",      "
                  << "convrate " << convRate << "\n";

        if (relativeError < ddTolerance)
            break;

    }



    // //////////////////////////////////////////////////////////
    //   Recompute and compare against exact solution
    // //////////////////////////////////////////////////////////
    VectorType exactSol3d       = x3d;
    RodSolutionType exactSolRod = rodX;

    // //////////////////////////////////////////////////////////
    //   Compute hessian of the rod functional at the exact solution
    //   for use of the energy norm it creates.
    // //////////////////////////////////////////////////////////

    BCRSMatrix<FieldMatrix<double, 6, 6> > hessianRod;
    MatrixIndexSet indices(exactSolRod.size(), exactSolRod.size());
    rodAssembler.getNeighborsPerVertex(indices);
    indices.exportIdx(hessianRod);
    rodAssembler.assembleMatrix(exactSolRod, hessianRod);


    double error = std::numeric_limits<double>::max();
    double oldError = 0;
    double totalConvRate = 1;

    VectorType      intermediateSol3d(x3d.size());
    RodSolutionType intermediateSolRod(rodX.size());

    // Compute error of the initial 3d solution
    
    // This should really be exactSol-initialSol, but we're starting
    // from zero anyways
    oldError += computeEnergyNormSquared(exactSol3d, *hessian3d);
    
    /** \todo Rod error still missing */

    oldError = std::sqrt(oldError);

    

    int i;
    for (i=0; i<maxDirichletNeumannSteps; i++) {
        
        // /////////////////////////////////////////////////////
        //   Read iteration from file
        // /////////////////////////////////////////////////////

        // Read 3d solution from file
        char iSolFilename[100];
        sprintf(iSolFilename, "tmp/intermediate3dSolution_%04d", i);
            
        FILE* fpInt = fopen(iSolFilename, "rb");
        if (!fpInt)
            DUNE_THROW(IOError, "Couldn't open intermediate solution '" << iSolFilename << "'");
        for (int j=0; j<intermediateSol3d.size(); j++)
            fread(&intermediateSol3d[j], sizeof(double), dim, fpInt);
        
        fclose(fpInt);

        // Read rod solution from file
        sprintf(iSolFilename, "tmp/intermediateRodSolution_%04d", i);
            
        fpInt = fopen(iSolFilename, "rb");
        if (!fpInt)
            DUNE_THROW(IOError, "Couldn't open intermediate solution '" << iSolFilename << "'");
        for (int j=0; j<intermediateSolRod.size(); j++) {
            fread(&intermediateSolRod[j].r, sizeof(double), dim, fpInt);
            fread(&intermediateSolRod[j].q, sizeof(double), 4, fpInt);
        }
        
        fclose(fpInt);



        // /////////////////////////////////////////////////////
        //   Compute error
        // /////////////////////////////////////////////////////
        
        VectorType solBackup0 = intermediateSol3d;
        solBackup0 -= exactSol3d;

        RodDifferenceType rodDifference = computeRodDifference(exactSolRod, intermediateSolRod);
        
        error = std::sqrt(computeEnergyNormSquared(solBackup0, *hessian3d)
                          +
                          computeEnergyNormSquared(rodDifference, hessianRod));
        

        double convRate = error / oldError;
        totalConvRate *= convRate;

        // Output
        std::cout << "DD iteration: " << i << "  error : " << error << ",      "
                  << "convrate " << convRate 
                  << "    total conv rate " << std::pow(totalConvRate, 1/((double)i+1)) << std::endl;

        if (error < 1e-12)
          break;

        oldError = error;
        
    }            

    std::cout << "damping: " << damping
              << "   convRate: " << std::pow(totalConvRate, 1/((double)i+1)) << std::endl;


    // //////////////////////////////
    //   Output result
    // //////////////////////////////
    LeafAmiraMeshWriter<GridType> amiraMeshWriter(grid);
    amiraMeshWriter.addVertexData(x3d, grid.leafIndexSet());
    amiraMeshWriter.write("grid.result");

    writeRod(rodX, "rod3d.result");

//     for (int i=0; i<rodX.size(); i++)
//         std::cout << rodX[i] << std::endl;

 } catch (Exception e) {

    std::cout << e << std::endl;

 }
