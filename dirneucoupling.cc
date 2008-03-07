#include <config.h>

#include <dune/grid/onedgrid.hh>
#include <dune/grid/uggrid.hh>

#include <dune/disc/elasticity/linearelasticityassembler.hh>
#include <dune/disc/operators/p1operator.hh>
#include <dune/istl/io.hh>
#include <dune/grid/io/file/amirameshreader.hh>
#include <dune/grid/io/file/amirameshwriter.hh>


#include <dune/common/bitfield.hh>
#include <dune/common/configparser.hh>

#include <dune/ag-common/multigridstep.hh>
#include <dune/ag-common/iterativesolver.hh>
#include <dune/ag-common/projectedblockgsstep.hh>
#ifdef HAVE_IPOPT
#include <dune/ag-common/linearipopt.hh>
#endif
#ifdef HAVE_IPOPT_CPP
#include <dune/ag-common/quadraticipopt.hh>
#endif
#include <dune/ag-common/readbitfield.hh>
#include <dune/ag-common/energynorm.hh>
#include <dune/ag-common/boundarypatch.hh>
#include <dune/ag-common/prolongboundarypatch.hh>
#include <dune/ag-common/sampleonbitfield.hh>
#include <dune/ag-common/neumannassembler.hh>
#include <dune/ag-common/computestress.hh>

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
using std::vector;

// Some types that I need
//typedef BCRSMatrix<FieldMatrix<double, dim, dim> > OperatorType;
//typedef BlockVector<FieldVector<double, dim> >     VectorType;
typedef vector<Configuration>                      RodSolutionType;
typedef BlockVector<FieldVector<double, 6> >       RodDifferenceType;


// Make a straight rod from two given endpoints
void makeStraightRod(RodSolutionType& rod, int n,
                     const FieldVector<double,3>& beginning, const FieldVector<double,3>& end)
{
    // Compute the correct orientation
    Quaternion<double> orientation = Quaternion<double>::identity();

    FieldVector<double,3> zAxis(0);
    zAxis[2] = 1;
    FieldVector<double,3> axis = crossProduct(end-beginning, zAxis);

    if (axis.two_norm() != 0)
        axis /= -axis.two_norm();

    FieldVector<double,3> d3 = end-beginning;
    d3 /= d3.two_norm();

    double angle = std::acos(zAxis * d3);

    if (angle != 0)
        orientation = Quaternion<double>(axis, angle);

    // Set the values
    rod.resize(n);
    for (int i=0; i<n; i++) {

        rod[i].r = beginning;
        rod[i].r.axpy(double(i) / (n-1), end-beginning);
        rod[i].q = orientation;

    }


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
    if (argc==2)
        parameterSet.parseFile(argv[1]);
    else
        parameterSet.parseFile("dirneucoupling.parset");

    // read solver settings
    const int numLevels            = parameterSet.get<int>("numLevels");
    const double ddTolerance           = parameterSet.get<double>("ddTolerance");
    const int maxDirichletNeumannSteps = parameterSet.get<int>("maxDirichletNeumannSteps");
    const double trTolerance           = parameterSet.get<double>("trTolerance");
    const int maxTrustRegionSteps = parameterSet.get<int>("maxTrustRegionSteps");
    const int trVerbosity            = parameterSet.get<int>("trVerbosity");
    const int multigridIterations = parameterSet.get<int>("numIt");
    const int nu1              = parameterSet.get<int>("nu1");
    const int nu2              = parameterSet.get<int>("nu2");
    const int mu               = parameterSet.get<int>("mu");
    const int baseIterations   = parameterSet.get<int>("baseIt");
    const double mgTolerance     = parameterSet.get<double>("mgTolerance");
    const double baseTolerance = parameterSet.get<double>("baseTolerance");
    const double initialTrustRegionRadius = parameterSet.get<double>("initialTrustRegionRadius");
    const double damping         = parameterSet.get<double>("damping");
    string resultPath           = parameterSet.get("resultPath", "");

    // Problem settings
    string path = parameterSet.get<string>("path");
    string objectName = parameterSet.get<string>("gridFile");
    string dirichletNodesFile  = parameterSet.get<string>("dirichletNodes");
    string dirichletValuesFile = parameterSet.get<string>("dirichletValues");
    string interfaceNodesFile  = parameterSet.get<string>("interfaceNodes");
    const int numRodBaseElements    = parameterSet.get<int>("numRodBaseElements");

    double E      = parameterSet.get<double>("E");
    double nu     = parameterSet.get<double>("nu");

    // rod material parameters
    double rodA   = parameterSet.get<double>("rodA");
    double rodJ1  = parameterSet.get<double>("rodJ1");
    double rodJ2  = parameterSet.get<double>("rodJ2");
    double rodE   = parameterSet.get<double>("rodE");
    double rodNu  = parameterSet.get<double>("rodNu");

    std::tr1::array<FieldVector<double,3>,2> rodRestEndPoint;
    rodRestEndPoint[0][0] = parameterSet.get<double>("rodRestEndPoint0X");
    rodRestEndPoint[0][1] = parameterSet.get<double>("rodRestEndPoint0Y");
    rodRestEndPoint[0][2] = parameterSet.get<double>("rodRestEndPoint0Z");
    rodRestEndPoint[1][0] = parameterSet.get<double>("rodRestEndPoint1X");
    rodRestEndPoint[1][1] = parameterSet.get<double>("rodRestEndPoint1Y");
    rodRestEndPoint[1][2] = parameterSet.get<double>("rodRestEndPoint1Z");

    // ///////////////////////////////////////
    //    Create the rod grid
    // ///////////////////////////////////////
    typedef OneDGrid RodGridType;
    RodGridType rodGrid(numRodBaseElements, 0, (rodRestEndPoint[1]-rodRestEndPoint[0]).two_norm());

    // ///////////////////////////////////////
    //    Create the grid for the 3d object
    // ///////////////////////////////////////
    typedef UGGrid<dim> GridType;
    GridType grid;
    grid.setRefinementType(GridType::COPY);
    
    AmiraMeshReader<GridType>::read(grid, path + objectName);

    // //////////////////////////////////////
    //   Globally refine grids
    // //////////////////////////////////////

    rodGrid.globalRefine(numLevels-1);
    grid.globalRefine(numLevels-1);

    std::vector<BitField> dirichletNodes(1);

    RodSolutionType rodX(rodGrid.size(1));

    // //////////////////////////
    //   Initial solution
    // //////////////////////////
#if 0
    for (int i=0; i<rodX.size(); i++) {
        rodX[i].r[0] = 0.5;
        rodX[i].r[1] = 0.5;
        rodX[i].r[2] = 5 + (i* 5.0 /(rodX.size()-1));
        rodX[i].q = Quaternion<double>::identity();
    }
#endif
    makeStraightRod(rodX, rodGrid.size(1), rodRestEndPoint[0], rodRestEndPoint[1]);

    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////
    rodX.back().r[0] = parameterSet.get("dirichletValueX", rodRestEndPoint[1][0]);
    rodX.back().r[1] = parameterSet.get("dirichletValueY", rodRestEndPoint[1][1]);
    rodX.back().r[2] = parameterSet.get("dirichletValueZ", rodRestEndPoint[1][2]);

    FieldVector<double,3> axis;
    axis[0] = parameterSet.get("dirichletAxisX", double(0));
    axis[1] = parameterSet.get("dirichletAxisY", double(0));
    axis[2] = parameterSet.get("dirichletAxisZ", double(0));
    double angle = parameterSet.get("dirichletAngle", double(0));

    rodX.back().q = Quaternion<double>(axis, M_PI*angle/180);

    // Backup initial rod iterate for later reference
    RodSolutionType initialIterateRod = rodX;

    int toplevel = rodGrid.maxLevel();

    // /////////////////////////////////////////////////////
    //   Determine the Dirichlet nodes
    // /////////////////////////////////////////////////////
    std::vector<VectorType> dirichletValues;
    dirichletValues.resize(toplevel+1);
    dirichletValues[0].resize(grid.size(0, dim));
    AmiraMeshReader<int>::readFunction(dirichletValues[0], path + dirichletValuesFile);

    std::vector<BoundaryPatch<GridType> > dirichletBoundary;
    dirichletBoundary.resize(numLevels);
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

    sampleOnBitField(grid, dirichletValues, dirichletNodes);
    
    // /////////////////////////////////////////////////////
    //   Determine the interface boundary
    // /////////////////////////////////////////////////////
    std::vector<BoundaryPatch<GridType> > interfaceBoundary;
    interfaceBoundary.resize(numLevels);
    interfaceBoundary[0].setup(grid, 0);
    readBoundaryPatch(interfaceBoundary[0], path + interfaceNodesFile);
    PatchProlongator<GridType>::prolong(interfaceBoundary);

    // ////////////////////////////////////////// 
    //   Assemble 3d linear elasticity problem
    // //////////////////////////////////////////
    LeafP1Function<GridType,double,dim> u(grid),f(grid);
    LinearElasticityLocalStiffness<GridType,double> lstiff(E, nu);
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
    rodAssembler.setShapeAndMaterial(rodA, rodJ1, rodJ2, rodE, rodNu);


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

    switch (trVerbosity) {
    case 0:
        rodSolver.verbosity_ = Solver::QUIET;   break;
    case 1:
        rodSolver.verbosity_ = Solver::REDUCED;   break;
    default:
        rodSolver.verbosity_ = Solver::FULL;   break;
    }


    // ////////////////////////////////
    //   Create a multigrid solver
    // ////////////////////////////////

    // First create a gauss-seidel base solver
#ifdef HAVE_IPOPT
    LinearIPOptSolver<VectorType> baseSolver;
#endif
#ifdef HAVE_IPOPT_CPP
    QuadraticIPOptSolver<MatrixType,VectorType> baseSolver;
#endif
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
    multigridStep.verbosity_         = Solver::QUIET;



    EnergyNorm<MatrixType, VectorType> energyNorm(multigridStep);

    IterativeSolver<MatrixType, VectorType> solver(&multigridStep,
                                                   // IPOpt doesn't like to be started in the solution
                                                   (numLevels!=1) ? multigridIterations : 1,
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
    FieldVector<double,3> lambdaForce(0);
    FieldVector<double,3> lambdaTorque(0);

    //
    double normOfOldCorrection = 0;

    for (int i=0; i<maxDirichletNeumannSteps; i++) {
        
        std::cout << "----------------------------------------------------" << std::endl;
        std::cout << "      Dirichlet-Neumann Step Number: " << i << std::endl;
        std::cout << "----------------------------------------------------" << std::endl;
        
        // Backup of the current solution for the error computation later on
        VectorType      oldSolution3d  = x3d;
        RodSolutionType oldSolutionRod = rodX;

        // //////////////////////////////////////////////////
        //   Dirichlet step for the rod
        // //////////////////////////////////////////////////

        rodX[0] = lambda;
        rodSolver.setInitialSolution(rodX);
        rodSolver.solve();

        rodX = rodSolver.getSol();

//         for (int j=0; j<rodX.size(); j++)
//             std::cout << rodX[j] << std::endl;

        // ///////////////////////////////////////////////////////////
        //   Extract Neumann values and transfer it to the 3d object
        // ///////////////////////////////////////////////////////////

        BitField couplingBitfield(rodX.size(),false);
        // Using that index 0 is always the left boundary for a uniformly refined OneDGrid
        couplingBitfield[0] = true;
        BoundaryPatch<RodGridType> couplingBoundary(rodGrid, rodGrid.maxLevel(), couplingBitfield);

        FieldVector<double,dim> resultantForce, resultantTorque;
        resultantForce  = rodAssembler.getResultantForce(couplingBoundary, rodX, resultantTorque);

        std::cout << "resultant force: " << resultantForce << std::endl;
        std::cout << "resultant torque: " << resultantTorque << std::endl;

#if 1
        VectorType neumannValues(rhs3d.size());

        // Using that index 0 is always the left boundary for a uniformly refined OneDGrid
        computeAveragePressureIPOpt<GridType>(resultantForce, resultantTorque, 
                                              interfaceBoundary[grid.maxLevel()], 
                                              rodX[0],
                                              neumannValues);

        rhs3d = 0;
        assembleAndAddNeumannTerm<GridType, VectorType>(interfaceBoundary[grid.maxLevel()],
                                                        neumannValues,
                                                        rhs3d);

#else
#ifndef HAVE_LAPACKPP
#error You need LaPack++ for this!
#endif
        // For the time being the Neumann data coming from the rod is a dg function (== not continuous)
        // Maybe that is not necessary
        DGIndexSet<GridType> dgIndexSet(grid,grid.maxLevel());
        dgIndexSet.setup(grid,grid.maxLevel());

        VectorType neumannValues(dgIndexSet.size());

        // Using that index 0 is always the left boundary for a uniformly refined OneDGrid
        computeAveragePressure<GridType>(resultantForce, resultantTorque, 
                                              interfaceBoundary[grid.maxLevel()], 
                                              rodX[0],
                                              neumannValues);

        rhs3d = 0;
        assembleAndAddNeumannTerm<GridType, VectorType>(interfaceBoundary[grid.maxLevel()],
                                                        dgIndexSet,
                                                        neumannValues,
                                                        rhs3d);
#endif
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

        //averageInterface.r[0] = averageInterface.r[1] = 0;
        //averageInterface.q = Quaternion<double>::identity();
        std::cout << "average interface: " << averageInterface << std::endl;

        std::cout << "director 0:  " << averageInterface.q.director(0) << std::endl;
        std::cout << "director 1:  " << averageInterface.q.director(1) << std::endl;
        std::cout << "director 2:  " << averageInterface.q.director(2) << std::endl;
        std::cout << std::endl;

        // ///////////////////////////////////////////////////////////
        //   Compute new damped interface value
        // ///////////////////////////////////////////////////////////
        for (int j=0; j<dim; j++)
            lambda.r[j] = (1-damping) * lambda.r[j] 
                + damping * (referenceInterface.r[j] + averageInterface.r[j]);

        lambda.q = Quaternion<double>::interpolate(lambda.q, 
                                                   referenceInterface.q.mult(averageInterface.q), 
                                                   damping);

        std::cout << "Lambda: " << lambda << std::endl;

        // ////////////////////////////////////////////////////////////////////////
        //   Write the two iterates to disk for later convergence rate measurement
        // ////////////////////////////////////////////////////////////////////////

        // First the 3d body
        std::stringstream iAsAscii;
        iAsAscii << i;
        std::string iSolFilename = resultPath + "tmp/intermediate3dSolution_" + iAsAscii.str();
            
        FILE* fp = fopen(iSolFilename.c_str(), "wb");
        if (!fp)
            DUNE_THROW(SolverError, "Couldn't open file " << iSolFilename << " for writing");
            
        for (int j=0; j<x3d.size(); j++)
            for (int k=0; k<dim; k++)
                fwrite(&x3d[j][k], sizeof(double), 1, fp);

        fclose(fp);

        // Then the rod
        iSolFilename = resultPath + "tmp/intermediateRodSolution_" + iAsAscii.str();

        FILE* fpRod = fopen(iSolFilename.c_str(), "wb");
        if (!fpRod)
            DUNE_THROW(SolverError, "Couldn't open file " << iSolFilename << " for writing");
            
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
        double oldNorm = EnergyNorm<MatrixType,VectorType>::normSquared(oldSolution3d, *hessian3d);
        oldSolution3d -= x3d;
        double normOfCorrection = EnergyNorm<MatrixType,VectorType>::normSquared(oldSolution3d, *hessian3d);

        double max3dRelCorrection = 0;
        for (size_t j=0; j<x3d.size(); j++)
            for (int k=0; k<dim; k++)
                max3dRelCorrection = std::max(max3dRelCorrection, 
                                              std::fabs(oldSolution3d[j][k])/ std::fabs(x3d[j][k]));


        RodDifferenceType rodDiff = computeRodDifference(oldSolutionRod, rodX);
        double maxRodRelCorrection = 0;
        for (size_t j=0; j<rodX.size(); j++)
            for (int k=0; k<dim; k++)
                maxRodRelCorrection = std::max(maxRodRelCorrection, 
                                              std::fabs(rodDiff[j][k])/ std::fabs(rodX[j].r[k]));

        double maxRodCorrection = computeRodDifference(oldSolutionRod, rodX).infinity_norm();
        double max3dCorrection  = oldSolution3d.infinity_norm();
        std::cout << "rod correction: " << maxRodCorrection
                  << "    rod rel correction: " <<  maxRodRelCorrection
                  << "    3d correction: " <<  max3dCorrection
                  << "    3d rel correction: " <<  max3dRelCorrection << std::endl;
        

        oldNorm = std::sqrt(oldNorm);

        normOfCorrection = std::sqrt(normOfCorrection);

        double relativeError = normOfCorrection / oldNorm;

        double convRate = normOfCorrection / normOfOldCorrection;

        normOfOldCorrection = normOfCorrection;

        // Output
        std::cout << "DD iteration: " << i << "  --  ||u^{n+1} - u^n|| / ||u^n||: " << relativeError << ",      "
                  << "convrate " << convRate << "\n";

        //if (relativeError < ddTolerance)
        if (std::max(max3dRelCorrection,maxRodRelCorrection) < ddTolerance)
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
    rodAssembler.assembleMatrixFD(exactSolRod, hessianRod);


    double error = std::numeric_limits<double>::max();
    double oldError = 0;

    VectorType      intermediateSol3d(x3d.size());
    RodSolutionType intermediateSolRod(rodX.size());

    // Compute error of the initial 3d solution
    
    // This should really be exactSol-initialSol, but we're starting
    // from zero anyways
    oldError += EnergyNorm<MatrixType,VectorType>::normSquared(exactSol3d, *hessian3d);
    
    // Error of the initial rod iterate
    RodDifferenceType rodDifference = computeRodDifference(initialIterateRod, exactSolRod);
    oldError += EnergyNorm<BCRSMatrix<FieldMatrix<double,6,6> >,RodDifferenceType>::normSquared(rodDifference, hessianRod);

    oldError = std::sqrt(oldError);

    // Store the history of total conv rates so we can filter out numerical
    // dirt in the end.
    std::vector<double> totalConvRate(maxDirichletNeumannSteps+1);
    totalConvRate[0] = 1;


    double oldConvRate = 100;
    bool firstTime = true;
    std::stringstream levelAsAscii, dampingAsAscii;
    levelAsAscii << numLevels;
    dampingAsAscii << damping;
    std::string filename = resultPath + "convrate_" + levelAsAscii.str() + "_" + dampingAsAscii.str();

    int i;
    for (i=0; i<maxDirichletNeumannSteps; i++) {
        
        // /////////////////////////////////////////////////////
        //   Read iteration from file
        // /////////////////////////////////////////////////////

        // Read 3d solution from file
        std::stringstream iAsAscii;
        iAsAscii << i;
        std::string iSolFilename = resultPath + "tmp/intermediate3dSolution_" + iAsAscii.str();
            
        FILE* fpInt = fopen(iSolFilename.c_str(), "rb");
        if (!fpInt)
            DUNE_THROW(IOError, "Couldn't open intermediate solution '" << iSolFilename << "'");
        for (int j=0; j<intermediateSol3d.size(); j++)
            fread(&intermediateSol3d[j], sizeof(double), dim, fpInt);
        
        fclose(fpInt);

        // Read rod solution from file
        iSolFilename = resultPath + "tmp/intermediateRodSolution_" + iAsAscii.str();
            
        fpInt = fopen(iSolFilename.c_str(), "rb");
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
        
        error = std::sqrt(EnergyNorm<MatrixType,VectorType>::normSquared(solBackup0, *hessian3d)
                          +
                          EnergyNorm<BCRSMatrix<FieldMatrix<double,6,6> >,RodDifferenceType>::normSquared(rodDifference, hessianRod));
        

        double convRate = error / oldError;
        totalConvRate[i+1] = totalConvRate[i] * convRate;

        // Output
        std::cout << "DD iteration: " << i << "  error : " << error << ",      "
                  << "convrate " << convRate 
                  << "    total conv rate " << std::pow(totalConvRate[i+1], 1/((double)i+1)) << std::endl;

        // Convergence rates tend to stay fairly constant after a few initial iterates.
        // Only when we hit numerical dirt are they starting to wiggle around wildly.
        // We use this to detect 'the' convergence rate as the last averaged rate before
        // we hit the dirt.
        if (convRate > oldConvRate + 0.1 && i > 5 && firstTime) {

            std::cout << "Damping: " << damping
                      << "   convRate: " << std::pow(totalConvRate[i], 1/((double)i)) 
                      << std::endl;

            std::ofstream convRateFile(filename.c_str());
            convRateFile << damping << "   " 
                         << std::pow(totalConvRate[i], 1/((double)i)) 
                         << std::endl;

            firstTime = false;
        }

        if (error < 1e-12)
          break;

        oldError = error;
        oldConvRate = convRate;
        
    }            

    // Convergence without dirt: write the overall convergence rate now
    if (firstTime) {
        int backTrace = std::min(size_t(4),totalConvRate.size());
        std::cout << "Damping: " << damping
                  << "   convRate: " << std::pow(totalConvRate[i+1-backTrace], 1/((double)i+1-backTrace)) 
                  << std::endl;
        
        std::ofstream convRateFile(filename.c_str());
        convRateFile << damping << "   " 
                     << std::pow(totalConvRate[i+1-backTrace], 1/((double)i+1-backTrace)) 
                     << std::endl;
    }

    // //////////////////////////////
    //   Output result
    // //////////////////////////////
    LeafAmiraMeshWriter<GridType> amiraMeshWriter(grid);
    amiraMeshWriter.addVertexData(x3d, grid.leafIndexSet());

    BlockVector<FieldVector<double,1> > stress;
    Stress<GridType,dim>::getStress(grid, x3d, stress, E, nu);
    amiraMeshWriter.addVertexData(stress, grid.leafIndexSet());

    amiraMeshWriter.write(resultPath + "grid.result");



    writeRod(rodX, resultPath + "rod3d.result");

 } catch (Exception e) {

    std::cout << e << std::endl;

 }
