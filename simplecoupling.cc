#include <config.h>

//#define HAVE_IPOPT
//#define DUNE_EXPRESSIONTEMPLATES

#include <dune/grid/onedgrid.hh>
#include <dune/grid/uggrid.hh>

#include <dune/disc/elasticity/linearelasticityassembler.hh>
#include <dune/disc/operators/p1operator.hh>
#include <dune/istl/io.hh>
#include <dune/grid/io/file/amirameshreader.hh>
#include <dune/grid/io/file/amirameshwriter.hh>


#include <dune/common/bitfield.hh>
#include <dune/common/configparser.hh>

#include "../contact/src/contactmmgstep.hh"
#include "../solver/iterativesolver.hh"
#include "../common/projectedblockgsstep.hh"
#include "../common/geomestimator.hh"
#include "../common/readbitfield.hh"
#include "../common/energynorm.hh"
#include "../common/boundarypatch.hh"
#include "../common/prolongboundarypatch.hh"

#include "src/quaternion.hh"
#include "src/rodassembler.hh"
#include "src/configuration.hh"
#include "src/rodcoupling.hh"
#include "src/rodwriter.hh"

// Number of degrees of freedom of a correction for a rod configuration
// 6 (x, y, z, v_1, v_2, v_3) for a spatial rod
const int blocksize = 6;

// Space dimension
const int dim = 3;

using namespace Dune;
using std::string;

void setTrustRegionObstacles(double trustRegionRadius,
                             SimpleVector<BoxConstraint<dim> >& trustRegionObstacles,
                             const BitField& dirichletNodes)
{
    for (int i=0; i<trustRegionObstacles.size(); i++) {

        for (int k=0; k<dim; k++) {

           if (dirichletNodes[i*dim+k])
                continue;

            trustRegionObstacles[i].val[2*k]   = -trustRegionRadius;
            trustRegionObstacles[i].val[2*k+1] =  trustRegionRadius;

        }
        
    }

    //std::cout << trustRegionObstacles << std::endl;
//     exit(0);
}



int main (int argc, char *argv[]) try
{
    // Some types that I need
    typedef BCRSMatrix<FieldMatrix<double, dim, dim> >             MatrixType;
    typedef BlockVector<FieldVector<double, dim> >                 VectorType;

    typedef BCRSMatrix<FieldMatrix<double, blocksize, blocksize> > RodMatrixType;
    typedef BlockVector<FieldVector<double, blocksize> >           RodCorrectionType;
    typedef std::vector<Configuration>                               RodSolutionType;

    // parse data file
    ConfigParser parameterSet;
    parameterSet.parseFile("simplecoupling.parset");

    // read solver settings
    const int minLevel            = parameterSet.get("minLevel", int(0));
    const int maxLevel            = parameterSet.get("maxLevel", int(0));
    const int maxTrustRegionSteps = parameterSet.get("maxTrustRegionSteps", int(0));
    const int numIt            = parameterSet.get("numIt", int(0));
    const int nu1              = parameterSet.get("nu1", int(0));
    const int nu2              = parameterSet.get("nu2", int(0));
    const int mu               = parameterSet.get("mu", int(0));
    const int baseIt           = parameterSet.get("baseIt", int(0));
    const double tolerance     = parameterSet.get("tolerance", double(0));
    const double baseTolerance = parameterSet.get("baseTolerance", double(0));
    const bool paramBoundaries = parameterSet.get("paramBoundaries", int(0));

    // Problem settings
    std::string path = parameterSet.get("path", "xyz");
    std::string objectName = parameterSet.get("gridFile", "xyz");
    std::string parFile             = parameterSet.get("parFile", "xyz");
    std::string dirichletNodesFile  = parameterSet.get("dirichletNodes", "xyz");
    std::string dirichletValuesFile = parameterSet.get("dirichletValues", "xyz");
    const int numRodBaseElements = parameterSet.get("numRodBaseElements", int(0));
    
    
    // ///////////////////////////////////////
    //    Create the rod grid
    // ///////////////////////////////////////
    typedef OneDGrid<1,1> RodGridType;
    RodGridType rodGrid(numRodBaseElements, 0, 5);

    // ///////////////////////////////////////
    //    Create the grid for the 3d object
    // ///////////////////////////////////////
    typedef UGGrid<dim,dim> GridType;
    GridType grid;
    grid.setRefinementType(GridType::COPY);
    
    if (paramBoundaries) {
        AmiraMeshReader<GridType>::read(grid, path + objectName, path + parFile);
    } else {
        AmiraMeshReader<GridType>::read(grid, path + objectName);
    }


    Array<SimpleVector<BoxConstraint<dim> > > trustRegionObstacles(1);
    Array<BitField> hasObstacle(1);
    Array<BitField> dirichletNodes(1);

    double trustRegionRadius = 0.1;

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

    Array<BoundaryPatch<GridType> > dirichletBoundary;
    dirichletBoundary.resize(maxLevel+1);
    dirichletBoundary[0].setup(grid, 0);
    readBoundaryPatch(dirichletBoundary[0], path + dirichletNodesFile);
    prolongBoundaryPatch(dirichletBoundary);

    dirichletNodes.resize(toplevel+1);
    for (int i=0; i<=toplevel; i++) {
        
        dirichletNodes[i].resize( dim*grid.size(i,dim) + blocksize * rodGrid.size(i,1));
        dirichletNodes[i].unsetAll();

        for (int j=0; j<grid.size(i,dim); j++)
            for (int k=0; k<dim; k++)
                dirichletNodes[i][j*dim+k] = dirichletBoundary[i].containsVertex(j);
        
        for (int j=0; j<blocksize; j++)
            dirichletNodes[i][dirichletNodes[i].size()-1-j] = true;
        
    }
    
    // ////////////////////////////////////////////////////////////
    //    Create solution and rhs vectors
    // ////////////////////////////////////////////////////////////
    
    VectorType totalRhs, totalCorr;
    totalRhs.resize(grid.size(toplevel,dim) + 2*rodGrid.size(toplevel,1));
    totalCorr.resize(grid.size(toplevel,dim) + 2*rodGrid.size(toplevel,1));
    
    // ////////////////////////////////////////////////////////////
    //   Create and set up assembler for the separate problems
    // ////////////////////////////////////////////////////////////
    RodMatrixType rodHessian;
    RodAssembler<RodGridType> rodAssembler(rodGrid);
    
    //std::cout << "Energy: " << rodAssembler.computeEnergy(rodX) << std::endl;
    
    MatrixIndexSet indices(rodGrid.size(toplevel,1), rodGrid.size(toplevel,1));
    rodAssembler.getNeighborsPerVertex(indices);
    indices.exportIdx(rodHessian);

    RodCorrectionType rodRhs, rodCorr;
    
    // //////////////////////////////////////////
    //   Assemble 3d linear elasticity problem
    // //////////////////////////////////////////
    LeafP1Function<GridType,double,dim> u(grid),f(grid);
    LinearElasticityLocalStiffness<GridType,double> lstiff(2.5e5, 0.3);
    LeafP1OperatorAssembler<GridType,double,dim> hessian3d(grid);
    hessian3d.assemble(lstiff,u,f);

    VectorType x3d(grid.size(toplevel,dim));
    VectorType corr3d(grid.size(toplevel,dim));
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
    //   Set up the total matrix
    // ///////////////////////////////////////////
    MatrixType totalHessian;
    RodCoupling<MatrixType, VectorType, RodMatrixType, RodCorrectionType> coupling;

    coupling.setUpMatrix(totalHessian, *hessian3d, rodHessian);
    coupling.insert3dPart(totalHessian, *hessian3d);

    // //////////////////////////////////////////////////////////
    //   Create obstacles
    // //////////////////////////////////////////////////////////
    
    hasObstacle.resize(toplevel+1);
    for (int i=0; i<hasObstacle.size(); i++) {
        hasObstacle[i].resize(grid.size(i, dim) + 2*rodGrid.size(i,1));
        hasObstacle[i].setAll();
    }
    
    trustRegionObstacles.resize(toplevel+1);
    
    for (int i=0; i<toplevel+1; i++) {
        trustRegionObstacles[i].resize(grid.size(i,dim) + 2*rodGrid.size(i,1));
    }
    
    // ////////////////////////////////
    //   Create a multigrid solver
    // ////////////////////////////////

    // First create a gauss-seidel base solver
    ProjectedBlockGSStep<MatrixType, VectorType> baseSolverStep;

    EnergyNorm<MatrixType, VectorType> baseEnergyNorm(baseSolverStep);

    IterativeSolver<MatrixType, VectorType> baseSolver;
    baseSolver.iterationStep = &baseSolverStep;
    baseSolver.numIt = baseIt;
    baseSolver.verbosity_ = Solver::QUIET;
    baseSolver.errorNorm_ = &baseEnergyNorm;
    baseSolver.tolerance_ = baseTolerance;

    // Make pre and postsmoothers
    ProjectedBlockGSStep<MatrixType, VectorType> presmoother, postsmoother;

    ContactMMGStep<MatrixType, VectorType> contactMMGStep(totalHessian, totalCorr, totalRhs, 1);

    contactMMGStep.setMGType(mu, nu1, nu2);
    contactMMGStep.dirichletNodes_    = &dirichletNodes;
    contactMMGStep.basesolver_        = &baseSolver;
    contactMMGStep.presmoother_       = &presmoother;
    contactMMGStep.postsmoother_      = &postsmoother;    
    contactMMGStep.hasObstacle_       = &hasObstacle;
    contactMMGStep.obstacles_         = &trustRegionObstacles;
    contactMMGStep.verbosity_         = Solver::QUIET;



    EnergyNorm<MatrixType, VectorType> energyNorm(contactMMGStep);

    IterativeSolver<MatrixType, VectorType> solver;
    solver.iterationStep = &contactMMGStep;
    solver.numIt = numIt;
    solver.verbosity_ = Solver::FULL;
    solver.errorNorm_ = &energyNorm;
    solver.tolerance_ = tolerance;

    // ////////////////////////////////////
    //   Create the transfer operators
    // ////////////////////////////////////
    for (int k=0; k<contactMMGStep.mgTransfer_.size(); k++)
        delete(contactMMGStep.mgTransfer_[k]);
    
    contactMMGStep.mgTransfer_.resize(toplevel);
    
    for (int i=0; i<contactMMGStep.mgTransfer_.size(); i++){
        TruncatedMGTransfer<VectorType>* newTransferOp = new TruncatedMGTransfer<VectorType>;
        newTransferOp->setup(grid,i,i+1);
        contactMMGStep.mgTransfer_[i] = newTransferOp;
    }
    
    // /////////////////////////////////////////////////////
    //   Trust-Region Solver
    // /////////////////////////////////////////////////////
    for (int i=0; i<maxTrustRegionSteps; i++) {
        
        std::cout << "----------------------------------------------------" << std::endl;
        std::cout << "      Trust-Region Step Number: " << i << std::endl;
        std::cout << "----------------------------------------------------" << std::endl;
        
        std::cout << "### Trust-Region Radius: " << trustRegionRadius << " ###" << std::endl;
        std::cout << " --  Rod energy: " << rodAssembler.computeEnergy(rodX) << " --" << std::endl;
        VectorType tmp3d(x3d.size());
        tmp3d = 0;
        (*hessian3d).umv(x3d, tmp3d);
        double energy3d = 0.5 * (x3d*tmp3d);
        std::cout << " --  3d energy: " << energy3d << " --" << std::endl;

        totalCorr = 0;
        totalRhs  = 0;

        // Update the 3d part of the right hand side
        rhs3d = 0;  // The zero here is the true right hand side
        (*hessian3d).mmv(x3d, rhs3d);
        
        coupling.insert3dPart(totalRhs, rhs3d);
        
        // Update the rod part of the total Hessian and right hand side
        rodRhs = 0;
        rodAssembler.assembleGradient(rodX, rodRhs);
        rodRhs *= -1;
        rodAssembler.assembleMatrix(rodX, rodHessian);
        
        coupling.insertRodPart(totalHessian, rodHessian);
        coupling.insertRodPart(totalRhs, rodRhs);

        // Create trust-region obstacles
        setTrustRegionObstacles(trustRegionRadius,
                                trustRegionObstacles[toplevel],
                                dirichletNodes[toplevel]);
        
        // /////////////////////////////
        //    Solve !
        // /////////////////////////////
        dynamic_cast<MultigridStep<MatrixType,VectorType>*>(solver.iterationStep)->setProblem(totalHessian, totalCorr, totalRhs, toplevel+1);
        solver.preprocess();
        contactMMGStep.preprocess();
        
        solver.solve();
        
        totalCorr = contactMMGStep.getSol();
        
        //std::cout << "Correction: " << std::endl << totalCorr << std::endl;
        
        printf("infinity norm of the correction: %g\n", totalCorr.infinity_norm());
        if (totalCorr.infinity_norm() < 1e-5) {
            std::cout << "CORRECTION IS SMALL ENOUGH" << std::endl;
            break;
        }
        
        // ////////////////////////////////////////////////////
        //   Split overall correction into its separate parts
        // ////////////////////////////////////////////////////
        coupling.get3dPart(totalCorr, corr3d);
        coupling.getRodPart(totalCorr, rodCorr);

        std::cout << "3ddCorrection: " << std::endl << corr3d << std::endl;
        std::cout << "RodCorrection: " << std::endl << rodCorr << std::endl;
        //exit(0);
        // ////////////////////////////////////////////////////
        //   Check whether trust-region step can be accepted
        // ////////////////////////////////////////////////////
        
        RodSolutionType newRodIterate = rodX;
        for (int j=0; j<rodX.size(); j++) {
            
            // Add translational correction
            for (int k=0; k<3; k++)
                newRodIterate[j].r[k] += rodCorr[j][k];
            
            // Add rotational correction
            newRodIterate[j].q = newRodIterate[j].q.mult(Quaternion<double>::exp(rodCorr[j][3], rodCorr[j][4], rodCorr[j][5]));
            
        }
        
        VectorType new3dIterate = x3d;
        new3dIterate += corr3d;
//         std::cout << "newIterate: \n";
//         for (int j=0; j<newIterate.size(); j++)
//             printf("%d:  (%g %g %g)  (%g %g %g %g)\n", j,
//                    newIterate[j].r[0],newIterate[j].r[1],newIterate[j].r[2],
//                    newIterate[j].q[0],newIterate[j].q[1],newIterate[j].q[2],newIterate[j].q[3]);
        
        /** \todo Don't always recompute old energy */
        double oldRodEnergy = rodAssembler.computeEnergy(rodX); 
        double rodEnergy    = rodAssembler.computeEnergy(newRodIterate);

        // compute the model decrease for the 3d part
        // It is $ m(x) - m(x+s) = -<g,s> - 0.5 <s, Hs>
        // Note that rhs = -g
        tmp3d = 0;
        (*hessian3d).umv(corr3d, tmp3d);
        double modelDecrease3d = (rhs3d*corr3d) - 0.5 * (corr3d*tmp3d);

        // compute the model decrease for the rod
        // It is $ m(x) - m(x+s) = -<g,s> - 0.5 <s, Hs>
        // Note that rhs = -g
        RodCorrectionType tmp(rodCorr.size());
        tmp = 0;
        rodHessian.mmv(rodCorr, tmp);
        double rodModelDecrease = (rodRhs*rodCorr) - 0.5 * (rodCorr*tmp);
        
        std::cout << "Rod model decrease: " << rodModelDecrease 
                       << ",  rod functional decrease: " << oldRodEnergy - rodEnergy << std::endl;

        std::cout << "3d decrease: " << modelDecrease3d << std::endl;

        if (rodEnergy >= oldRodEnergy) {
            printf("Richtung ist keine Abstiegsrichtung!\n");
            //                  std::cout << "corr[0]\n" << corr[0] << std::endl;
            
            exit(0);
        }
        
        //  Add correction to the current solution
        rodX = newRodIterate;
        x3d  = new3dIterate;

    }

    // //////////////////////////////
    //   Output result
    // //////////////////////////////
    AmiraMeshWriter<GridType>::writeGrid(grid, "grid.result");
    AmiraMeshWriter<GridType>::writeBlockVector(grid, x3d, "grid.sol");
    writeRod(rodX, "rod3d.result");

 } catch (Exception e) {

    std::cout << e << std::endl;

 }
