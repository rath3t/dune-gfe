#include <config.h>

//#define HARMONIC_ENERGY_FD_GRADIENT
//#define HARMONIC_ENERGY_FD_INNER_GRADIENT

#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/onedgrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/grid/io/file/amirameshwriter.hh>
#include <dune/grid/io/file/amirameshreader.hh>

#include <dune/fufem/functionspacebases/p1nodalbasis.hh>
#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/assemblers/operatorassembler.hh>
#include <dune/fufem/assemblers/localassemblers/laplaceassembler.hh>
#include <dune/fufem/assemblers/localassemblers/massassembler.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/h1seminorm.hh>

#include <dune/gfe/unitvector.hh>
#include <dune/gfe/harmonicenergystiffness.hh>
#include <dune/gfe/geodesicfeassembler.hh>
#include <dune/gfe/riemanniantrsolver.hh>
#include <dune/gfe/geodesicfefunctionadaptor.hh>

// grid dimension
const int dim = 3;

typedef UnitVector<3> TargetSpace;
typedef std::vector<TargetSpace> SolutionType;

const int blocksize = TargetSpace::TangentVector::size;

using namespace Dune;
using std::string;

template <class GridType>
void solve (const shared_ptr<GridType>& grid,
            SolutionType& x, 
            int numLevels,
            const ParameterTree& parameters)
{
    // read solver setting
    const double innerTolerance           = parameters.get<double>("innerTolerance");
    const double tolerance                = parameters.get<double>("tolerance");
    const int maxTrustRegionSteps         = parameters.get<int>("maxTrustRegionSteps");
    const double initialTrustRegionRadius = parameters.get<double>("initialTrustRegionRadius");
    const int multigridIterations         = parameters.get<int>("numIt");

    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////

    BitSetVector<1> allNodes(grid->size(dim));
    allNodes.setAll();
    BoundaryPatch<typename GridType::LeafGridView> dirichletBoundary(grid->leafView(), allNodes);

    BitSetVector<blocksize> dirichletNodes(grid->size(dim));
    for (int i=0; i<dirichletNodes.size(); i++)
        dirichletNodes[i] = dirichletBoundary.containsVertex(i);

    // //////////////////////////
    //   Initial solution
    // //////////////////////////

    x.resize(grid->size(dim));

        FieldVector<double,3> yAxis(0);
    yAxis[1] = 1;

    typename GridType::LeafGridView::template Codim<dim>::Iterator vIt    = grid->template leafbegin<dim>();
    typename GridType::LeafGridView::template Codim<dim>::Iterator vEndIt = grid->template leafend<dim>();

    for (; vIt!=vEndIt; ++vIt) {
        int idx = grid->leafIndexSet().index(*vIt);

        FieldVector<double,3> v;
        FieldVector<double,dim> pos = vIt->geometry().corner(0);
        FieldVector<double,3> axis;
        axis[0] = pos[0];  axis[1] = pos[1]; axis[2] = 1;
        Rotation<3,double> rotation(axis, pos.two_norm()*M_PI*3);

        if (dirichletNodes[idx][0]) {
#if 1
            FieldMatrix<double,3,3> rMat;
            rotation.matrix(rMat);
            v = rMat[2];
#else
            v[0] = std::sin(pos[0]*M_PI);
            v[1] = 0;
            v[2] = std::cos(pos[0]*M_PI);
#endif
        } else {
            v[0] = 1;
            v[1] = 0;
            v[2] = 0;
        }            

        x[idx] = v;

    }


    // ////////////////////////////////////////////////////////////
    //   Create an assembler for the Harmonic Energy Functional
    // ////////////////////////////////////////////////////////////

    HarmonicEnergyLocalStiffness<typename GridType::LeafGridView,TargetSpace> harmonicEnergyLocalStiffness;

    GeodesicFEAssembler<typename GridType::LeafGridView,TargetSpace> assembler(grid->leafView(),
                                                                      &harmonicEnergyLocalStiffness);
    
    // ///////////////////////////////////////////
    //   Create a solver for the rod problem
    // ///////////////////////////////////////////

    RiemannianTrustRegionSolver<GridType,TargetSpace> solver;

    solver.setup(*grid, 
                 &assembler,
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

    // /////////////////////////////////////////////////////
    //   Solve!
    // /////////////////////////////////////////////////////

    solver.setInitialSolution(x);
    solver.solve();

    x = solver.getSol();
}

int main (int argc, char *argv[]) try
{
    // parse data file
    ParameterTree parameterSet;
    if (argc==2)
        ParameterTreeParser::readINITree(argv[1], parameterSet);
    else
        ParameterTreeParser::readINITree("harmonicmaps-eoc.parset", parameterSet);

    // read solver settings
    const int numLevels        = parameterSet.get<int>("numLevels");
    const int baseIterations      = parameterSet.get<int>("baseIt");
    const double baseTolerance    = parameterSet.get<double>("baseTolerance");

    // only if a structured grid is used
    const int numBaseElements = parameterSet.get<int>("numBaseElements");
    FieldVector<double,dim> lowerLeft  = parameterSet.get<FieldVector<double,dim> >("lowerLeft");
    FieldVector<double,dim> upperRight = parameterSet.get<FieldVector<double,dim> >("upperRight");
    
    // ///////////////////////////////////////////////////////////
    //   First compute the 'exact' solution on a very fine grid
    // ///////////////////////////////////////////////////////////

    typedef std::conditional<dim==1,OneDGrid,UGGrid<dim> >::type GridType;

    //    Create the reference grid
    shared_ptr<GridType> referenceGrid;
    
    if (parameterSet.get<std::string>("gridType")=="structured") {
        array<unsigned int,dim> elements;
        elements.fill(numBaseElements);
        referenceGrid = StructuredGridFactory<GridType>::createSimplexGrid(lowerLeft,
                                                                           upperRight,
                                                                           elements);
    } else {
        referenceGrid = shared_ptr<GridType>(AmiraMeshReader<GridType>::read(parameterSet.get<std::string>("gridFile")));
    }
    
    referenceGrid->globalRefine(numLevels-1);

    //  Solve the rod Dirichlet problem
    SolutionType referenceSolution;
    solve(referenceGrid, referenceSolution, numLevels, parameterSet);

    BlockVector<FieldVector<double,3> > xEmbedded(referenceSolution.size());
    for (int j=0; j<referenceSolution.size(); j++)
        xEmbedded[j] = referenceSolution[j].globalCoordinates();
        
    LeafAmiraMeshWriter<GridType> amiramesh;
    amiramesh.addGrid(referenceGrid->leafView());
    amiramesh.addVertexData(xEmbedded, referenceGrid->leafView());
    amiramesh.write("reference_result.am");

    // //////////////////////////////////////////////////////////////////////
    //   Compute mass matrix and laplace matrix to emulate L2 and H1 norms
    // //////////////////////////////////////////////////////////////////////

    typedef P1NodalBasis<GridType::LeafGridView,double> FEBasis;
    FEBasis basis(referenceGrid->leafView());
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
    
    std::ofstream logFile("harmonicmaps-eoc.results");
    logFile << "# vertices max-norm, L2-norm, h1-seminorm" << std::endl;
    
    for (int i=1; i<numLevels; i++) {

        shared_ptr<GridType> grid;
        if (parameterSet.get<std::string>("gridType")=="structured") {
            array<unsigned int,dim> elements;
            elements.fill(numBaseElements);
            grid = StructuredGridFactory<GridType>::createSimplexGrid(lowerLeft,
                                                                      upperRight,
                                                                      elements);
        } else {
            grid = shared_ptr<GridType>(AmiraMeshReader<GridType>::read(parameterSet.get<std::string>("gridFile")));
        }

        grid->globalRefine(i-1);

        // compute again
        SolutionType solution;
        solve(grid, solution, i, parameterSet);

        // write solution
        std::stringstream numberAsAscii;
        numberAsAscii << i;

        BlockVector<FieldVector<double,3> > xEmbedded(solution.size());
        for (int j=0; j<solution.size(); j++)
            xEmbedded[j] = solution[j].globalCoordinates();
        
        LeafAmiraMeshWriter<GridType> amiramesh;
        amiramesh.addGrid(grid->leafView());
        amiramesh.addVertexData(xEmbedded, grid->leafView());
        amiramesh.write("harmonic_result_" + numberAsAscii.str() + ".am");

        // Prolong solution to the very finest grid
        for (int j=i; j<numLevels; j++)
            geodesicFEFunctionAdaptor(*grid, solution);

        // Interpret TargetSpace as isometrically embedded into an R^m, because this is
        // how the corresponding Sobolev spaces are defined.

        BlockVector<TargetSpace::CoordinateType> difference(referenceSolution.size());

        for (int j=0; j<referenceSolution.size(); j++)
            difference[j] = solution[j].globalCoordinates() - referenceSolution[j].globalCoordinates();

        H1SemiNorm< BlockVector<TargetSpace::CoordinateType> > h1Norm(laplace);
        H1SemiNorm< BlockVector<TargetSpace::CoordinateType> > l2Norm(massMatrix);

        // Compute max-norm difference
        std::cout << "Vertices: " << xEmbedded.size() << std::endl;
        std::cout << "Level: " << i-1 
                  << ",   max-norm error: " << difference.infinity_norm()
                  << std::endl;

        std::cout << "Level: " << i-1 
                  << ",   L2 error: " << l2Norm(difference)
                  << std::endl;

        std::cout << "Level: " << i-1 
                  << ",   H1 error: " << h1Norm(difference)
                  << std::endl;
                  
        logFile << xEmbedded.size() << "  " << difference.infinity_norm() 
                << "  " << l2Norm(difference)
                << "  " << h1Norm(difference)
                << std::endl;

    }    
        
 } catch (Exception e) {

    std::cout << e << std::endl;

 }
