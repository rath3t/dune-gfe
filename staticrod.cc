#include <config.h>

#include <dune/grid/onedgrid.hh>

#include <dune/fem/lagrangebase.hh>

#include <dune/istl/io.hh>

//#include "../common/boundarytreatment.hh"
#include "../common/boundarypatch.hh"
#include <dune/common/bitfield.hh>
//#include "../common/readbitfield.hh"

#include "src/rodassembler.hh"
//#include "../common/linearipopt.hh"

#include "../common/projectedblockgsstep.hh"
#include <dune/solver/iterativesolver.hh>

#include "../common/geomestimator.hh"
#include "../common/energynorm.hh"
#include <dune/common/configparser.hh>

// Choose a solver
//#define IPOPT
#define GAUSS_SEIDEL
//#define MULTIGRID

//#define IPOPT_BASE

// Number of degrees of freedom: 
// 3 (x, y, theta) for a planar rod
const int blocksize = 3;

using namespace Dune;
using std::string;

int main (int argc, char *argv[]) try
{
    // Some types that I need
    typedef BCRSMatrix<FieldMatrix<double, blocksize, blocksize> > MatrixType;
    typedef BlockVector<FieldVector<double, blocksize> >     VectorType;

    // parse data file
    ConfigParser parameterSet;
    parameterSet.parseFile("staticrod.parset");

    // read solver settings
    const int minLevel         = parameterSet.get("minLevel", int(0));
    const int maxLevel         = parameterSet.get("maxLevel", int(0));
    double loadIncrement       = parameterSet.get("loadIncrement", double(0));
    const int maxNewtonSteps   = parameterSet.get("maxNewtonSteps", int(0));
    const int numIt            = parameterSet.get("numIt", int(0));
    const int nu1              = parameterSet.get("nu1", int(0));
    const int nu2              = parameterSet.get("nu2", int(0));
    const int mu               = parameterSet.get("mu", int(0));
    const int baseIt           = parameterSet.get("baseIt", int(0));
    const double tolerance     = parameterSet.get("tolerance", double(0));
    const double baseTolerance = parameterSet.get("baseTolerance", double(0));
    
    // Problem settings
    const int numRodElements = parameterSet.get("numRodElements", int(0));

    // ///////////////////////////////////////
    //    Create the two grids
    // ///////////////////////////////////////
    typedef OneDGrid<1,1> RodGridType;
    RodGridType rod(numRodElements, 0, 1);

    

    Array<BitField> dirichletNodes;
    dirichletNodes.resize(maxLevel+1);
    dirichletNodes[0].resize( blocksize * (numRodElements+1) );

    dirichletNodes[0].unsetAll();
    dirichletNodes[0][0] = dirichletNodes[0][1] = dirichletNodes[0][2] = true;
    dirichletNodes[0][blocksize*numRodElements+0] = true;
    dirichletNodes[0][blocksize*numRodElements+1] = true;
    dirichletNodes[0][blocksize*numRodElements+2] = true;

    // refine uniformly until minlevel
    for (int i=0; i<minLevel; i++)
        rod.globalRefine(1);

    int maxlevel = rod.maxlevel();

    // //////////////////////////////////////////////////////////
    //   Create obstacles
    // //////////////////////////////////////////////////////////

    Array<BitField> hasObstacle;
    hasObstacle.resize(maxLevel+1);
    hasObstacle[0].resize(numRodElements+1);
    hasObstacle[0].unsetAll();



    // //////////////////////////////////////////////////////////
    //    Create discrete function spaces
    // //////////////////////////////////////////////////////////

    typedef FunctionSpace < double , double, 1, 1 > RodFuncSpace;
    typedef DefaultGridIndexSet<RodGridType,LevelIndex> RodIndexSet;
    typedef LagrangeDiscreteFunctionSpace < RodFuncSpace, RodGridType,RodIndexSet,  1> RodFuncSpaceType;

    Array<RodIndexSet*> rodIndexSet(maxlevel+1);
    Array<const RodFuncSpaceType*> rodFuncSpace(maxlevel+1);

    for (int i=0; i<maxlevel+1; i++) {
        rodIndexSet[i]  = new RodIndexSet(rod, i);
        rodFuncSpace[i] = new RodFuncSpaceType(rod, *rodIndexSet[i], i);
    }


    // ////////////////////////////////////////////////////////////
    //    Create solution and rhs vectors
    // ////////////////////////////////////////////////////////////

    VectorType rhs;
    VectorType x;
    VectorType corr;

    MatrixType hessianMatrix;

    rhs.resize(rodFuncSpace[maxlevel]->size());
    x.resize(rodFuncSpace[maxlevel]->size());
    corr.resize(rodFuncSpace[maxlevel]->size());
    
    // Initial solution
    x = 0;

    for (int i=0; i<numRodElements+1; i++) {
        x[i][0] = i/((double)numRodElements);
        x[i][1] = 0;
        x[i][2] = M_PI/2;
    }

    x[0][1] = x[numRodElements][1] = 1;

    RodAssembler<RodFuncSpaceType,2> test(*rodFuncSpace[0]);
    test.assembleGradient(x, rhs);
    //std::cout << "Solution: " << std::endl << x << std::endl;
    //std::cout << "Gradient: " << std::endl << rhs << std::endl;
    std::cout << "Energy: " << test.computeEnergy(x) << std::endl;

    MatrixIndexSet indices(numRodElements+1, numRodElements+1);
    test.getNeighborsPerVertex(indices);
    indices.exportIdx(hessianMatrix);
    test.assembleMatrix(x,hessianMatrix);

    //printmatrix(std::cout, hessianMatrix, "hessianMatrix", "--");
    //exit(0);

    // Create a solver
#if defined IPOPT

    typedef LinearIPOptSolver<VectorType> SolverType;
    
    SolverType solver;
    solver.dirichletNodes_ = &totalDirichletNodes[maxlevel];
    solver.hasObstacle_    = &contactAssembler.hasObstacle_[maxlevel];
    solver.obstacles_      = &contactAssembler.obstacles_[maxlevel];
    solver.verbosity_      = Solver::FULL;

#elif defined GAUSS_SEIDEL

    typedef ProjectedBlockGSStep<MatrixType, VectorType> SmootherType;
    SmootherType projectedBlockGSStep(hessianMatrix, corr, rhs);
    projectedBlockGSStep.dirichletNodes_ = &dirichletNodes[maxlevel];
    projectedBlockGSStep.hasObstacle_    = &hasObstacle[maxlevel];
    projectedBlockGSStep.obstacles_      = NULL;//&contactAssembler.obstacles_[maxlevel];

    EnergyNorm<MatrixType, VectorType> energyNorm(projectedBlockGSStep);

    IterativeSolver<MatrixType, VectorType> solver;
    solver.iterationStep = &projectedBlockGSStep;
    solver.numIt = numIt;
    solver.verbosity_ = Solver::QUIET;
    solver.errorNorm_ = &energyNorm;
    solver.tolerance_ = tolerance;
    
#elif defined MULTIGRID

    // First create a base solver
#ifdef IPOPT_BASE

    LinearIPOptSolver<BlockVector<FieldVector<double,dim> >  > baseSolver;
    baseSolver.verbosity_ = Solver::FULL;

#else // Gauss-Seidel is the base solver

    ProjectedBlockGSStep<MatrixType, BlockVector<FieldVector<double,dim> >  > baseSolverStep;

    EnergyNorm<MatrixType, BlockVector<FieldVector<double,dim> >  > baseEnergyNorm(baseSolverStep);

    IterativeSolver<MatrixType, BlockVector<FieldVector<double,dim> >  > baseSolver;
    baseSolver.iterationStep = &baseSolverStep;
    baseSolver.numIt = baseIt;
    baseSolver.verbosity_ = Solver::QUIET;
    baseSolver.errorNorm_ = &baseEnergyNorm;
    baseSolver.tolerance_ = baseTolerance;
#endif

    // Make pre and postsmoothers
    ProjectedBlockGSStep<MatrixType, BlockVector<FieldVector<double,dim> >  > presmoother;
    ProjectedBlockGSStep<MatrixType, BlockVector<FieldVector<double,dim> >  > postsmoother;
    

    ContactMMGStep<MatrixType, BlockVector<FieldVector<double,dim> > , FuncSpaceType > contactMMGStep(maxlevel+1);

    contactMMGStep.setMGType(1, nu1, nu2);
    contactMMGStep.dirichletNodes_    = &totalDirichletNodes;
    contactMMGStep.basesolver_        = &baseSolver;
    contactMMGStep.presmoother_       = &presmoother;
    contactMMGStep.postsmoother_      = &postsmoother;    
    contactMMGStep.hasObstacle_       = &hasObstacle;
    contactMMGStep.obstacles_         = &contactAssembler.obstacles_;

    // Create the transfer operators
    contactMMGStep.mgTransfer_.resize(maxlevel);
    for (int i=0; i<contactMMGStep.mgTransfer_.size(); i++)
        contactMMGStep.mgTransfer_[i] = NULL;

    EnergyNorm<MatrixType, VectorType> energyNorm(contactMMGStep);

    IterativeSolver<MatrixType, BlockVector<FieldVector<double,dim> > > solver;
    solver.iterationStep = &contactMMGStep;
    solver.numIt = numIt;
    solver.verbosity_ = Solver::FULL;
    solver.errorNorm_ = &energyNorm;
    solver.tolerance_ = tolerance;

#else
    #warning You have to specify a solver!
#endif

    // ///////////////////////////////////////////////////
    //   Do a homotopy of the Dirichlet boundary data
    // ///////////////////////////////////////////////////
    double loadFactor = 0;

    do {

        RodAssembler<RodFuncSpaceType, 1> rodAssembler(*rodFuncSpace[maxlevel]);

        loadFactor += loadIncrement;

        std::cout << "####################################################" << std::endl;
        std::cout << "New load factor: " << loadFactor 
                  << "    new load increment: " << loadIncrement << std::endl;
        std::cout << "####################################################" << std::endl;

        // /////////////////////////////////////////////////////
        //   Newton Solver
        // /////////////////////////////////////////////////////

        for (int j=0; j<maxNewtonSteps; j++) {

            rhs = 0;
            corr = 0;

            rodAssembler.assembleGradient(x, rhs);
            rodAssembler.assembleMatrix(x, hessianMatrix);

            rhs *= -1;

            std::cout << "rhs: " << std::endl << rhs << std::endl;

#ifndef IPOPT
            solver.iterationStep->setProblem(hessianMatrix, corr, rhs);
#else
            solver.setProblem(hessianMatrix, corr, rhs);
#endif

            solver.preprocess();
#ifdef MULTIGRID

            contactMMGStep.preprocess();
#endif

            // /////////////////////////////
            //    Solve !
            // /////////////////////////////
             solver.solve();

#ifdef MULTIGRID
             totalCorr = contactMMGStep.getSol();
#endif

             std::cout << "Correction: \n" << corr << std::endl;

             // line search
             printf("------  Line Search ---------\n");
             int lSSteps = 10;
             double smallestEnergy = std::numeric_limits<double>::max();
             double smallestFactor = 1;
             for (int k=0; k<lSSteps; k++) {

                 double factor = double(k)/(lSSteps-1);
                 VectorType sCorr = corr;
                 sCorr *= factor;
                 sCorr += x;

                 double energy = rodAssembler.computeEnergy(sCorr);

                 if (energy < smallestEnergy) {
                     smallestEnergy = energy;
                     smallestFactor = factor;
                 }
                 //printf("factor: %g,  energy: %g\n", factor, energy);
             }

             std::cout << "Damping factor: " << smallestFactor << std::endl;

             //  Add correction to the current solution
             x.axpy(smallestFactor, corr);

             // Output result
             std::cout << "Solution:" << std::endl << x << std::endl;

             printf("infinity norm of the correction: %g\n", corr[0].infinity_norm());
             if (corr.infinity_norm() < 1e-8)
                 break;

        }
        
        
    } while (loadFactor < 1);



 } catch (Exception e) {

    std::cout << e << std::endl;

 }
