#include <config.h>

#include <dune/grid/onedgrid.hh>
#include <dune/grid/uggrid.hh>

#include <dune/istl/io.hh>
#include <dune/istl/solvers.hh>

#include <dune/grid/io/file/amirameshreader.hh>
#include <dune/grid/io/file/amirameshwriter.hh>

#include <dune/common/bitsetvector.hh>
#include <dune/common/configparser.hh>

#include <dune/solvers/iterationsteps/multigridstep.hh>
#include <dune/solvers/solvers/loopsolver.hh>
#include <dune/solvers/iterationsteps/projectedblockgsstep.hh>
#ifdef HAVE_IPOPT
#include <dune/solvers/solvers/quadraticipopt.hh>
#endif
#include <dune/fufem/readbitfield.hh>
#include <dune/solvers/norms/energynorm.hh>
#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/prolongboundarypatch.hh>
#include <dune/fufem/sampleonbitfield.hh>
#include <dune/fufem/computestress.hh>

#include <dune/fufem/functionspacebases/q1nodalbasis.hh>
#include <dune/fufem/assemblers/operatorassembler.hh>
#include <dune/fufem/assemblers/localassemblers/stvenantkirchhoffassembler.hh>

#include <dune/gfe/quaternion.hh>
#include <dune/gfe/rodassembler.hh>
#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/averageinterface.hh>
#include <dune/gfe/riemanniantrsolver.hh>
#include <dune/gfe/geodesicdifference.hh>
#include <dune/gfe/rodwriter.hh>
#include <dune/gfe/rodfactory.hh>
#include <dune/gfe/coupling/rodcontinuumcomplex.hh>
#include <dune/gfe/coupling/rodcontinuumsteklovpoincarestep.hh>

// Space dimension
const int dim = 3;

using namespace Dune;
using std::string;
using std::vector;

// Some types that I need
//typedef BCRSMatrix<FieldMatrix<double, dim, dim> > OperatorType;
//typedef BlockVector<FieldVector<double, dim> >     VectorType;
typedef vector<RigidBodyMotion<dim> >              RodSolutionType;
typedef BlockVector<FieldVector<double, 6> >       RodDifferenceType;


int main (int argc, char *argv[]) try
{
    // Some types that I need
    typedef BCRSMatrix<FieldMatrix<double, dim, dim> >   MatrixType;
    typedef BlockVector<FieldVector<double, dim> >       VectorType;

    // parse data file
    ConfigParser parameterSet;
    if (argc==2)
        parameterSet.parseFile(argv[1]);
    else
        parameterSet.parseFile("dirneucoupling.parset");

    // read solver settings
    const int numLevels            = parameterSet.get<int>("numLevels");
    string ddType                = parameterSet.get<string>("ddType");
    string preconditioner        = parameterSet.get<string>("preconditioner");
    const double ddTolerance     = parameterSet.get<double>("ddTolerance");
    const int maxDDIterations    = parameterSet.get<int>("maxDirichletNeumannSteps");
    const double damping         = parameterSet.get<double>("damping");
    
    
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

    Dune::array<FieldVector<double,3>,2> rodRestEndPoint;
    rodRestEndPoint[0] = parameterSet.get<FieldVector<double,3> >("rodRestEndPoint0");
    rodRestEndPoint[1] = parameterSet.get<FieldVector<double,3> >("rodRestEndPoint1");
    
    //////////////////////////////////////////////////////////////////
    //  Print the algorithm type so we have it in the log files
    //////////////////////////////////////////////////////////////////
    
    std::cout << "Algorithm:      " << ddType << std::endl;
    if (ddType=="RichardsonIteration")
        std::cout << "Preconditioner: " << preconditioner << std::endl;
    
    // ///////////////////////////////////////////////
    //    Create the rod grid and continuum grid
    // ///////////////////////////////////////////////
        
    typedef OneDGrid RodGridType;
    typedef UGGrid<dim> GridType;

    RodContinuumComplex<RodGridType,GridType> complex;
    
    complex.rods_["rod"].grid_ = shared_ptr<RodGridType>
                (new RodGridType(numRodBaseElements, 0, (rodRestEndPoint[1]-rodRestEndPoint[0]).two_norm()));

    complex.continua_["continuum"].grid_ = shared_ptr<GridType>(AmiraMeshReader<GridType>::read(path + objectName));
    complex.continua_["continuum"].grid_->setRefinementType(GridType::COPY);
    
    

    // //////////////////////////////////////
    //   Globally refine grids
    // //////////////////////////////////////

    complex.rods_["rod"].grid_->globalRefine(numLevels-1);
    complex.continua_["continuum"].grid_->globalRefine(numLevels-1);

    RodSolutionType rodX(complex.rods_["rod"].grid_->size(1));

    int toplevel = complex.rods_["rod"].grid_->maxLevel();

    // //////////////////////////
    //   Initial solution
    // //////////////////////////

    RodFactory<RodGridType::LeafGridView> rodFactory(complex.rods_["rod"].grid_->leafView());
    rodFactory.create(rodX, rodRestEndPoint[0], rodRestEndPoint[1]);

    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////
    rodX.back().r = parameterSet.get("dirichletValue", rodRestEndPoint[1]);

    FieldVector<double,3> axis = parameterSet.get("dirichletAxis", FieldVector<double,3>(0));
    double angle = parameterSet.get("dirichletAngle", double(0));

    rodX.back().q = Rotation<3,double>(axis, M_PI*angle/180);
    
    rodFactory.create(complex.rods_["rod"].dirichletValues_,
                      RigidBodyMotion<3>(FieldVector<double,3>(0), Rotation<3,double>::identity()));
    complex.rods_["rod"].dirichletValues_.back() = RigidBodyMotion<3>(parameterSet.get("dirichletValue", rodRestEndPoint[1]),
                                                                   Rotation<3,double>(axis, M_PI*angle/180));
    BitSetVector<1> rodDNodes(complex.rods_["rod"].dirichletValues_.size(), false);
    rodDNodes.back() = true;
    complex.rods_["rod"].dirichletBoundary_.setup(*complex.rods_["rod"].grid_,rodDNodes);

    // Backup initial rod iterate for later reference
    RodSolutionType initialIterateRod = rodX;

    // /////////////////////////////////////////////////////
    //   Determine the Dirichlet nodes
    // /////////////////////////////////////////////////////
    VectorType coarseDirichletValues(complex.continua_["continuum"].grid_->size(0, dim));
    AmiraMeshReader<int>::readFunction(coarseDirichletValues, path + dirichletValuesFile);

    LevelBoundaryPatch<GridType> coarseDirichletBoundary;
    coarseDirichletBoundary.setup(*complex.continua_["continuum"].grid_, 0);
    readBoundaryPatch(coarseDirichletBoundary, path + dirichletNodesFile);
    
    LeafBoundaryPatch<GridType> dirichletBoundary(*complex.continua_["continuum"].grid_);
    PatchProlongator<GridType>::prolong(coarseDirichletBoundary, dirichletBoundary);
    complex.continua_["continuum"].dirichletBoundary_ = dirichletBoundary;

    BitSetVector<dim> dirichletNodes( complex.continua_["continuum"].grid_->size(dim) );

    for (int i=0; i<dirichletNodes.size(); i++)
        dirichletNodes[i] = dirichletBoundary.containsVertex(i);

    sampleOnBitField(*complex.continua_["continuum"].grid_, 
                     coarseDirichletValues, 
                     complex.continua_["continuum"].dirichletValues_, 
                     dirichletNodes);
    
    /////////////////////////////////////////////////////////////////////
    //  Create the two interface boundary patches
    /////////////////////////////////////////////////////////////////////
    
    std::pair<std::string,std::string> interfaceName = std::make_pair("rod", "continuum");

    // first for the rod
    BitSetVector<1> rodCouplingBitfield(rodX.size(),false);
    // Using that index 0 is always the left boundary for a uniformly refined OneDGrid
    rodCouplingBitfield[0] = true;
    complex.couplings_[interfaceName].rodInterfaceBoundary_.setup(*complex.rods_["rod"].grid_, rodCouplingBitfield);

    // then for the continuum
    LevelBoundaryPatch<GridType> coarseInterfaceBoundary(*complex.continua_["continuum"].grid_, 0);
    readBoundaryPatch(coarseInterfaceBoundary, path + interfaceNodesFile);
    
    PatchProlongator<GridType>::prolong(coarseInterfaceBoundary, complex.couplings_[interfaceName].continuumInterfaceBoundary_);

    // ////////////////////////////////////////// 
    //   Assemble 3d linear elasticity problem
    // //////////////////////////////////////////

    typedef P1NodalBasis<GridType::LeafGridView,double> FEBasis;
    FEBasis basis(complex.continua_["continuum"].grid_->leafView());
    OperatorAssembler<FEBasis,FEBasis> assembler(basis, basis);

    StVenantKirchhoffAssembler<GridType, FEBasis::LocalFiniteElement, FEBasis::LocalFiniteElement> localAssembler(E, nu);
    MatrixType stiffnessMatrix3d;

    assembler.assemble(localAssembler, stiffnessMatrix3d);

    // ////////////////////////////////////////////////////////////
    //    Create solution and rhs vectors
    // ////////////////////////////////////////////////////////////
    
    VectorType x3d(complex.continua_["continuum"].grid_->size(toplevel,dim));
    VectorType rhs3d(complex.continua_["continuum"].grid_->size(toplevel,dim));

    // No external forces
    rhs3d = 0;

    // Set initial solution
    const VectorType& dirichletValues = complex.continua_["continuum"].dirichletValues_;
    x3d = 0;
    for (int i=0; i<x3d.size(); i++) 
        for (int j=0; j<dim; j++)
            if (dirichletNodes[i][j])
                x3d[i][j] = dirichletValues[i][j];

    // ///////////////////////////////////////////
    //   Dirichlet nodes for the rod problem
    // ///////////////////////////////////////////

    BitSetVector<6> rodDirichletNodes(complex.rods_["rod"].grid_->size(1));
    rodDirichletNodes.unsetAll();
        
    rodDirichletNodes[0] = true;
    rodDirichletNodes.back() = true;

    // ///////////////////////////////////////////
    //   Create a solver for the rod problem
    // ///////////////////////////////////////////

    RodLocalStiffness<RodGridType::LeafGridView,double> rodLocalStiffness(complex.rods_["rod"].grid_->leafView(),
                                                                       rodA, rodJ1, rodJ2, rodE, rodNu);

    RodAssembler<RodGridType::LeafGridView,3> rodAssembler(complex.rods_["rod"].grid_->leafView(), &rodLocalStiffness);

    RiemannianTrustRegionSolver<RodGridType,RigidBodyMotion<3> > rodSolver;
    rodSolver.setup(*complex.rods_["rod"].grid_, 
                    &rodAssembler,
                    rodX,
                    rodDirichletNodes,
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
    QuadraticIPOptSolver<MatrixType,VectorType> baseSolver;
#endif
    baseSolver.verbosity_ = NumProc::QUIET;
    baseSolver.tolerance_ = baseTolerance;

    // Make pre and postsmoothers
    BlockGSStep<MatrixType, VectorType> presmoother, postsmoother;

    MultigridStep<MatrixType, VectorType> multigridStep(stiffnessMatrix3d, x3d, rhs3d, toplevel+1);

    multigridStep.setMGType(mu, nu1, nu2);
    multigridStep.ignoreNodes_       = &dirichletNodes;
    multigridStep.basesolver_        = &baseSolver;
    multigridStep.setSmoother(&presmoother, &postsmoother);
    multigridStep.verbosity_         = Solver::QUIET;



    EnergyNorm<MatrixType, VectorType> energyNorm(multigridStep);

    shared_ptr< ::LoopSolver<VectorType> > solver = shared_ptr< ::LoopSolver<VectorType> >( new ::LoopSolver<VectorType>(&multigridStep,
                                                                                                                // IPOpt doesn't like to be started in the solution
                                                                                                                (numLevels!=1) ? multigridIterations : 1,
                                                                                                                mgTolerance,
                                                                                                                &energyNorm,
                                                                                                                Solver::QUIET) );

    // ////////////////////////////////////
    //   Create the transfer operators
    // ////////////////////////////////////
    for (int k=0; k<multigridStep.mgTransfer_.size(); k++)
        delete(multigridStep.mgTransfer_[k]);
    
    multigridStep.mgTransfer_.resize(toplevel);
    
    for (int i=0; i<multigridStep.mgTransfer_.size(); i++){
        CompressedMultigridTransfer<VectorType>* newTransferOp = new CompressedMultigridTransfer<VectorType>;
        newTransferOp->setup(*complex.continua_["continuum"].grid_,i,i+1);
        multigridStep.mgTransfer_[i] = newTransferOp;
    }
    

    // /////////////////////////////////////////////////////
    //   Dirichlet-Neumann Solver
    // /////////////////////////////////////////////////////

    RigidBodyMotion<3> referenceInterface = rodX[0];
    complex.couplings_[interfaceName].referenceInterface_ = referenceInterface;

    // Init interface value
    std::map<std::pair<std::string,std::string>, RigidBodyMotion<3> > lambda;
    lambda[interfaceName] = referenceInterface;
    
    FieldVector<double,3> lambdaForce(0);
    FieldVector<double,3> lambdaTorque(0);

    //
    double normOfOldCorrection = 1;
    int dnStepsActuallyTaken = 0;
    for (int i=0; i<maxDDIterations; i++) {
        
        std::cout << "----------------------------------------------------" << std::endl;
        std::cout << "      Domain-Decomposition- Step Number: " << i       << std::endl;
        std::cout << "----------------------------------------------------" << std::endl;
        
        // Backup of the current iterate for the error computation later on
        std::map<std::pair<std::string,std::string>, RigidBodyMotion<3> > oldLambda  = lambda;
        
        if (ddType=="FixedPointIteration") {

            // //////////////////////////////////////////////////
            //   Dirichlet step for the rod
            // //////////////////////////////////////////////////

            rodX[0] = lambda[interfaceName];
            rodSolver.setInitialSolution(rodX);
            rodSolver.solve();

            rodX = rodSolver.getSol();

//         for (int j=0; j<rodX.size(); j++)
//             std::cout << rodX[j] << std::endl;

            // ///////////////////////////////////////////////////////////
            //   Extract Neumann values and transfer it to the 3d object
            // ///////////////////////////////////////////////////////////

            FieldVector<double,dim> resultantForce, resultantTorque;
            resultantForce  = rodAssembler.getResultantForce(complex.couplings_[interfaceName].rodInterfaceBoundary_, rodX, resultantTorque);

            // Flip orientation
            resultantForce  *= -1;
            resultantTorque *= -1;
        
            std::cout << "resultant force: " << resultantForce << std::endl;
            std::cout << "resultant torque: " << resultantTorque << std::endl;

            VectorType neumannValues(rhs3d.size());

            // Using that index 0 is always the left boundary for a uniformly refined OneDGrid
            computeAveragePressure<GridType::LeafGridView>(resultantForce, resultantTorque, 
                                              complex.couplings_[interfaceName].continuumInterfaceBoundary_, 
                                              rodX[0].r,
                                              neumannValues);

            BoundaryFunctionalAssembler<FEBasis> boundaryFunctionalAssembler(basis, complex.couplings_[interfaceName].continuumInterfaceBoundary_);
            BasisGridFunction<FEBasis, VectorType> neumannValuesFunction(basis, neumannValues);
            NeumannBoundaryAssembler<GridType, FieldVector<double,dim> > localNeumannAssembler(neumannValuesFunction);
            boundaryFunctionalAssembler.assemble(localNeumannAssembler, rhs3d, true);

            // ///////////////////////////////////////////////////////////
            //   Solve the Neumann problem for the continuum
            // ///////////////////////////////////////////////////////////
            multigridStep.setProblem(stiffnessMatrix3d, x3d, rhs3d, complex.continua_["continuum"].grid_->maxLevel()+1);
        
            solver->preprocess();
            multigridStep.preprocess();
        
            solver->solve();
        
            x3d = multigridStep.getSol();

            // ///////////////////////////////////////////////////////////
            //   Extract new interface position and orientation
            // ///////////////////////////////////////////////////////////

            RigidBodyMotion<3> averageInterface;

            computeAverageInterface(complex.couplings_[interfaceName].continuumInterfaceBoundary_, x3d, averageInterface);

        //averageInterface.r[0] = averageInterface.r[1] = 0;
        //averageInterface.q = Quaternion<double>::identity();
            std::cout << "average interface: " << averageInterface << std::endl;

            std::cout << "director 0:  " << averageInterface.q.director(0) << std::endl;
            std::cout << "director 1:  " << averageInterface.q.director(1) << std::endl;
            std::cout << "director 2:  " << averageInterface.q.director(2) << std::endl;
            std::cout << std::endl;

            //////////////////////////////////////////////////////////////
            //   Compute new damped interface value
            //////////////////////////////////////////////////////////////
            for (int j=0; j<dim; j++)
                lambda[interfaceName].r[j] = (1-damping) * lambda[interfaceName].r[j] 
                    + damping * (referenceInterface.r[j] + averageInterface.r[j]);

            lambda[interfaceName].q = Rotation<3,double>::interpolate(lambda[interfaceName].q, 
                                                       referenceInterface.q.mult(averageInterface.q), 
                                                       damping);

           
        } else if (ddType=="RichardsonIteration") {
            
            Dune::array<double,2> alpha = parameterSet.get<array<double,2> >("NeumannNeumannDamping");
            
            RodContinuumSteklovPoincareStep<RodGridType,GridType> rodContinuumSteklovPoincareStep(complex,
                                                                                                  preconditioner,
                                                                                                  alpha,
                                                                                                  damping,
                                                                                                  &rodAssembler,
                                                                                                  &rodLocalStiffness,
                                                                                                  &rodSolver,
                                                                                                  &stiffnessMatrix3d,
                                                                                                  solver,
                                                                                                  &localAssembler);
            
            rodContinuumSteklovPoincareStep.iterate(lambda);
            
            // get the subdomain solutions
            rodX = rodContinuumSteklovPoincareStep.rodSubdomainSolutions_["rod"];
            x3d  = rodContinuumSteklovPoincareStep.continuumSubdomainSolutions_["continuum"];
            
        } else
            DUNE_THROW(NotImplemented, ddType << " is not a known domain decomposition algorithm");

        std::cout << "Lambda: " << lambda[interfaceName] << std::endl;

        // ////////////////////////////////////////////////////////////////////////
        //   Write the two iterates to disk for later convergence rate measurement
        // ////////////////////////////////////////////////////////////////////////

        // First the 3d body
        std::stringstream iAsAscii;
        iAsAscii << i;
        std::string iSolFilename = resultPath + "tmp/intermediate3dSolution_" + iAsAscii.str();

        LeafAmiraMeshWriter<GridType> amiraMeshWriter;
        amiraMeshWriter.addVertexData(x3d, complex.continua_["continuum"].grid_->leafView());
        amiraMeshWriter.write(iSolFilename);

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

        double lengthOfCorrection = RigidBodyMotion<3>::distance(oldLambda[interfaceName], lambda[interfaceName]);

        double convRate = lengthOfCorrection / normOfOldCorrection;

        normOfOldCorrection = lengthOfCorrection;

        // Output
        std::cout << "DD iteration: " << i << ",      "
                  << "interface correction: " << lengthOfCorrection << ",     "
                  << "convrate " << convRate << "\n";

        dnStepsActuallyTaken = i;

        if (lengthOfCorrection < ddTolerance)
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

    VectorType      intermediateSol3d(x3d.size());
    RodSolutionType intermediateSolRod(rodX.size());

    // Compute error of the initial 3d solution
    
    // This should really be exactSol-initialSol, but we're starting
    // from zero anyways
    oldError += EnergyNorm<MatrixType,VectorType>::normSquared(exactSol3d, stiffnessMatrix3d);
    
    // Error of the initial rod iterate
    RodDifferenceType rodDifference = computeGeodesicDifference(initialIterateRod, exactSolRod);
    oldError += EnergyNorm<BCRSMatrix<FieldMatrix<double,6,6> >,RodDifferenceType>::normSquared(rodDifference, hessianRod);

    oldError = std::sqrt(oldError);

    // Store the history of total conv rates so we can filter out numerical
    // dirt in the end.
    std::vector<double> totalConvRate(maxDDIterations+1);
    totalConvRate[0] = 1;


    double oldConvRate = 100;
    bool firstTime = true;
    std::stringstream levelAsAscii, dampingAsAscii;
    levelAsAscii << numLevels;
    dampingAsAscii << damping;
    std::string filename = resultPath + "convrate_" + levelAsAscii.str() + "_" + dampingAsAscii.str();

    int i;
    for (i=0; i<dnStepsActuallyTaken; i++) {
        
        // /////////////////////////////////////////////////////
        //   Read iteration from file
        // /////////////////////////////////////////////////////

        // Read 3d solution from file
        std::stringstream iAsAscii;
        iAsAscii << i;
        std::string iSolFilename = resultPath + "tmp/intermediate3dSolution_" + iAsAscii.str();
      
        AmiraMeshReader<int>::readFunction(intermediateSol3d, iSolFilename);

        // Read rod solution from file
        iSolFilename = resultPath + "tmp/intermediateRodSolution_" + iAsAscii.str();
            
        FILE* fpInt = fopen(iSolFilename.c_str(), "rb");
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

        RodDifferenceType rodDifference = computeGeodesicDifference(exactSolRod, intermediateSolRod);
        
        error = std::sqrt(EnergyNorm<MatrixType,VectorType>::normSquared(solBackup0, stiffnessMatrix3d)
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
    //   Delete temporary memory
    // //////////////////////////////
    std::string removeTmpCommand = "rm -rf " + resultPath + "tmp/intermediate*";
    system(removeTmpCommand.c_str());

    // //////////////////////////////
    //   Output result
    // //////////////////////////////
    LeafAmiraMeshWriter<GridType> amiraMeshWriter(*complex.continua_["continuum"].grid_);
    amiraMeshWriter.addVertexData(x3d, complex.continua_["continuum"].grid_->leafView());

    BlockVector<FieldVector<double,1> > stress;
    Stress<GridType>::getStress(*complex.continua_["continuum"].grid_, x3d, stress, E, nu);
    amiraMeshWriter.addCellData(stress, complex.continua_["continuum"].grid_->leafView());

    amiraMeshWriter.write(resultPath + "grid.result");



    writeRod(rodX, resultPath + "rod3d.result");

 } catch (Exception e) {

    std::cout << e << std::endl;

 }
