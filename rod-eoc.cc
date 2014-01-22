#include <config.h>

#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/grid/onedgrid.hh>

#include <dune/istl/io.hh>

#include <dune/fufem/functionspacebases/p1nodalbasis.hh>
#include <dune/fufem/assemblers/operatorassembler.hh>
#include <dune/fufem/assemblers/localassemblers/laplaceassembler.hh>
#include <dune/fufem/assemblers/localassemblers/massassembler.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/geodesicdifference.hh>
#include <dune/gfe/rotation.hh>
#include <dune/gfe/rodassembler.hh>
#include <dune/gfe/riemanniantrsolver.hh>
#include <dune/gfe/geodesicfefunctionadaptor.hh>
#include <dune/gfe/rodwriter.hh>

typedef Dune::OneDGrid GridType;

typedef RigidBodyMotion<double,3> TargetSpace;
typedef std::vector<RigidBodyMotion<double,3> > SolutionType;

const int blocksize = TargetSpace::TangentVector::dimension;

using namespace Dune;
using std::string;

void solve (const GridType& grid,
            SolutionType& x,
            int numLevels,
            const TargetSpace& dirichletValue,
            const ParameterTree& parameters)
{
    // read solver setting
    const double innerTolerance           = parameters.get<double>("innerTolerance");
    const double tolerance                = parameters.get<double>("tolerance");
    const int maxTrustRegionSteps         = parameters.get<int>("maxTrustRegionSteps");
    const double initialTrustRegionRadius = parameters.get<double>("initialTrustRegionRadius");
    const int multigridIterations         = parameters.get<int>("numIt");

    // read rod parameter settings
    const double A               = parameters.get<double>("A");
    const double J1              = parameters.get<double>("J1");
    const double J2              = parameters.get<double>("J2");
    const double E               = parameters.get<double>("E");
    const double nu              = parameters.get<double>("nu");

    //   Create a local assembler
    RodLocalStiffness<OneDGrid::LeafGridView,double> localStiffness(grid.leafGridView(),
                                                                    A, J1, J2, E, nu);




    x.resize(grid.size(1));

    // //////////////////////////
    //   Initial solution
    // //////////////////////////

    for (int i=0; i<x.size(); i++) {
        x[i].r[0] = 0;
        x[i].r[1] = 0;
        x[i].r[2] = double(i)/(x.size()-1);
        x[i].q    = Rotation<double,3>::identity();
    }

    //  set Dirichlet value
    x.back() = dirichletValue;

    // Both ends are Dirichlet
    BitSetVector<blocksize> dirichletNodes(grid.size(1));
    dirichletNodes.unsetAll();

    dirichletNodes[0] = dirichletNodes.back() = true;

    // ///////////////////////////////////////////
    //   Create a solver for the rod problem
    // ///////////////////////////////////////////

    RodAssembler<GridType::LeafGridView,3> rodAssembler(grid.leafGridView(), &localStiffness);

    RiemannianTrustRegionSolver<GridType,RigidBodyMotion<double,3> > rodSolver;
#if 1
    rodSolver.setup(grid,
                    &rodAssembler,
                    x,
                    dirichletNodes,
                    tolerance,
                    maxTrustRegionSteps,
                    initialTrustRegionRadius,
                    multigridIterations,
                    innerTolerance,
                    1, 3, 3,
                    100,     // iterations of the base solver
                    1e-8,    // base tolerance
                    false);  // instrumentation
#else
    rodSolver.setupTCG(grid,
                       &rodAssembler,
                       x,
                       dirichletNodes,
                       tolerance,
                       maxTrustRegionSteps,
                       initialTrustRegionRadius,
                       multigridIterations,
                       innerTolerance,
                       false);  // instrumentation
#endif

    // /////////////////////////////////////////////////////
    //   Solve!
    // /////////////////////////////////////////////////////

    rodSolver.setInitialSolution(x);
    rodSolver.solve();

    x = rodSolver.getSol();
}

int main (int argc, char *argv[]) try
{
    // parse data file
    ParameterTree parameterSet;
    if (argc==2)
        ParameterTreeParser::readINITree(argv[1], parameterSet);
    else
        ParameterTreeParser::readINITree("rod-eoc.parset", parameterSet);

    // read solver settings
    const int numLevels        = parameterSet.get<int>("numLevels");
    const int nu1              = parameterSet.get<int>("nu1");
    const int nu2              = parameterSet.get<int>("nu2");
    const int mu               = parameterSet.get<int>("mu");
    const int baseIterations      = parameterSet.get<int>("baseIt");
    const double baseTolerance    = parameterSet.get<double>("baseTolerance");

    const int numRodBaseElements = parameterSet.get<int>("numRodBaseElements");

    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////

    RigidBodyMotion<double,3> dirichletValue;
    dirichletValue.r[0] = parameterSet.get<double>("dirichletValueX");
    dirichletValue.r[1] = parameterSet.get<double>("dirichletValueY");
    dirichletValue.r[2] = parameterSet.get<double>("dirichletValueZ");

    FieldVector<double,3> axis;
    axis[0] = parameterSet.get<double>("dirichletAxisX");
    axis[1] = parameterSet.get<double>("dirichletAxisY");
    axis[2] = parameterSet.get<double>("dirichletAxisZ");
    double angle = parameterSet.get<double>("dirichletAngle");

    dirichletValue.q = Rotation<double,3>(axis, M_PI*angle/180);

    // ///////////////////////////////////////////////////////////
    //   First compute the 'exact' solution on a very fine grid
    // ///////////////////////////////////////////////////////////

    //    Create the reference grid

    GridType referenceGrid(numRodBaseElements, 0, 1);

    referenceGrid.globalRefine(numLevels-1);

    //  Solve the rod Dirichlet problem
    SolutionType referenceSolution;
    solve(referenceGrid, referenceSolution, numLevels, dirichletValue, parameterSet);


    // //////////////////////////////////////////////////////////////////////
    //   Compute mass matrix and laplace matrix to emulate L2 and H1 norms
    // //////////////////////////////////////////////////////////////////////

    typedef P1NodalBasis<GridType::LeafGridView,double> FEBasis;
    FEBasis basis(referenceGrid.leafGridView());
    OperatorAssembler<FEBasis,FEBasis> operatorAssembler(basis, basis);

    LaplaceAssembler<GridType, FEBasis::LocalFiniteElement, FEBasis::LocalFiniteElement> laplaceLocalAssembler;
    MassAssembler<GridType, FEBasis::LocalFiniteElement, FEBasis::LocalFiniteElement> massMatrixLocalAssembler;

    typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> > ScalarMatrixType;
    ScalarMatrixType laplace, massMatrix;

    operatorAssembler.assemble(laplaceLocalAssembler, laplace);
    operatorAssembler.assemble(massMatrixLocalAssembler, massMatrix);

    // ///////////////////////////////////////////////////////////
    //   Compute on all coarser levels, and compare
    // ///////////////////////////////////////////////////////////

    for (int i=1; i<=numLevels; i++) {

        GridType grid(numRodBaseElements, 0, 1);
        grid.globalRefine(i-1);

        // compute again
        SolutionType solution;
        solve(grid, solution, i, dirichletValue, parameterSet);

        // Prolong solution to the very finest grid
        for (int j=i; j<numLevels; j++)
            GeodesicFEFunctionAdaptor<FEBasis,TargetSpace>::geodesicFEFunctionAdaptor(grid, solution);

        std::stringstream numberAsAscii;
        numberAsAscii << i;
        writeRod(solution, "rodGrid_" + numberAsAscii.str());

        assert(referenceSolution.size() == solution.size());

        BlockVector<TargetSpace::TangentVector> difference = computeGeodesicDifference(solution,referenceSolution);

        H1SemiNorm< BlockVector<TargetSpace::TangentVector> > h1Norm(laplace);
        H1SemiNorm< BlockVector<TargetSpace::TangentVector> > l2Norm(massMatrix);

        // Compute max-norm difference
        std::cout << "Level: " << i-1
                  << ",   max-norm error: " << difference.infinity_norm()
                  << std::endl;

        std::cout << "Level: " << i-1
                  << ",   L2 error: " << l2Norm(difference)
                  << std::endl;

        std::cout << "Level: " << i-1
                  << ",   H1 error: " << h1Norm(difference)
                  << std::endl;

    }

 } catch (Exception e) {

    std::cout << e << std::endl;

 }
