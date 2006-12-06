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
#include "src/rodwriter.hh"

// Space dimension
const int dim = 3;

using namespace Dune;
using std::string;

int main (int argc, char *argv[]) try
{
    // Some types that I need
    typedef BCRSMatrix<FieldMatrix<double, dim, dim> >   MatrixType;
    typedef BlockVector<FieldVector<double, dim> >       VectorType;
    typedef std::vector<Configuration>                   RodSolutionType;

    // parse data file
    ConfigParser parameterSet;
    parameterSet.parseFile("dirneucoupling.parset");

    // read solver settings
    const int minLevel            = parameterSet.get("minLevel", int(0));
    const int maxLevel            = parameterSet.get("maxLevel", int(0));
    const int maxDirichletNeumannSteps = parameterSet.get("maxDirichletNeumannSteps", int(0));
    const int maxTrustRegionSteps = parameterSet.get("maxTrustRegionSteps", int(0));
    const int multigridIterations = parameterSet.get("numIt", int(0));
    const int nu1              = parameterSet.get("nu1", int(0));
    const int nu2              = parameterSet.get("nu2", int(0));
    const int mu               = parameterSet.get("mu", int(0));
    const int baseIterations   = parameterSet.get("baseIt", int(0));
    const double mgTolerance     = parameterSet.get("tolerance", double(0));
    const double baseTolerance = parameterSet.get("baseTolerance", double(0));
    const int initialTrustRegionRadius = parameterSet.get("initialTrustRegionRadius", int(0));
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

    rodX[rodX.size()-1].r[0] = 0.5;
    rodX[rodX.size()-1].r[1] = 0.5;
    rodX[rodX.size()-1].r[2] = 11;
//     rodX[rodX.size()-1].q[0] = 0;
//     rodX[rodX.size()-1].q[1] = 0;
//     rodX[rodX.size()-1].q[2] = 1/sqrt(2);
//     rodX[rodX.size()-1].q[3] = 1/sqrt(2);

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
    rodAssembler.setShapeAndMaterial(1, 1, 1, 2.5e5, 0.3);

    RodSolver<RodGridType> rodSolver;
    rodSolver.setup(rodGrid, 
                    &rodAssembler,
                    rodX,
                    maxTrustRegionSteps,
                    initialTrustRegionRadius,
                    multigridIterations,
                    mgTolerance,
                    mu, nu1, nu2,
                    baseIterations,
                    baseTolerance);

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
                                                   Solver::FULL);

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

    for (int i=0; i<maxDirichletNeumannSteps; i++) {
        
        std::cout << "----------------------------------------------------" << std::endl;
        std::cout << "      Dirichlet-Neumann Step Number: " << i << std::endl;
        std::cout << "----------------------------------------------------" << std::endl;
        
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
        FieldVector<double,dim> resultantForce  = rodAssembler.getResultantForce(rodX);
        std::cout << "resultant force: " << resultantForce << std::endl;
#if 0
        FieldVector<double,dim> resultantTorque = rodAssembler.getResultantTorque(grid, rodX);
#endif
        VectorType neumannValues(grid.size(dim));
        neumannValues = 0;
        for (int j=0; j<neumannValues.size(); j++)
            if (interfaceBoundary[grid.maxLevel()].containsVertex(j))
                neumannValues[j] = resultantForce;

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
//         x3d = 0;
//         for (int i=0; i<x3d.size(); i++)
//             x3d[i][2] = 1;

        computeAverageInterface(interfaceBoundary[toplevel], x3d, averageInterface);

        std::cout << "average interface: " << averageInterface << std::endl;

        // ///////////////////////////////////////////////////////////
        //   Compute new damped interface value
        // ///////////////////////////////////////////////////////////

        for (int j=0; j<dim; j++)
            lambda.r[j] = (1-damping) * lambda.r[j] + damping * (referenceInterface.r[j] + averageInterface.r[j]);
        lambda.q = averageInterface.q.mult(referenceInterface.q);

    }

    // //////////////////////////////
    //   Output result
    // //////////////////////////////
    LeafAmiraMeshWriter<GridType>::writeGrid(grid, "grid.result");
    LeafAmiraMeshWriter<GridType>::writeBlockVector(grid, x3d, "grid.sol");
    writeRod(rodX, "rod3d.result");

    for (int i=0; i<rodX.size(); i++)
        std::cout << rodX[i] << std::endl;

 } catch (Exception e) {

    std::cout << e << std::endl;

 }
