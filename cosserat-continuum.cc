#include <config.h>

#include <fenv.h>

#define RIGIDBODYMOTION3

#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/onedgrid.hh>
#include <dune/grid/geometrygrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/grid/io/file/amirameshreader.hh>
#include <dune/grid/io/file/amirameshwriter.hh>

#include <dune/fufem/boundarypatch.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/unitvector.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/cosseratenergystiffness.hh>
#include <dune/gfe/cosseratvtkwriter.hh>
#include <dune/gfe/geodesicfeassembler.hh>
#include <dune/gfe/riemanniantrsolver.hh>

// grid dimension
const int dim = 2;

// Image space of the geodesic fe functions
#ifdef RIGIDBODYMOTION3
typedef RigidBodyMotion<double,3> TargetSpace;
#endif

// Tangent vector of the image space
const int blocksize = TargetSpace::TangentVector::dimension;

using namespace Dune;

#if 0
void dirichletValues(const FieldVector<double,dim>& in, FieldVector<double,3>& out,
                     double homotopy)
{
    out = 0;
    for (int i=0; i<dim; i++)
        out[i] = in[i];

    out[1] += homotopy;
}
#endif
void dirichletValues(const FieldVector<double,dim>& in, FieldVector<double,3>& out,
                     double homotopy
)
{
    double angle = 8*M_PI;
    angle *= homotopy;

    // center of rotation
    FieldVector<double,3> center(0);
    center[1] = 0.5;

    FieldMatrix<double,3,3> rotation(0);
    rotation[0][0] = 1;
    rotation[1][1] =  std::cos(angle);
    rotation[1][2] = -std::sin(angle);
    rotation[2][1] =  std::sin(angle);
    rotation[2][2] =  std::cos(angle);

    FieldVector<double,3> inEmbedded(0);
    for (int i=0; i<dim; i++)
        inEmbedded[i] = in[i];
    inEmbedded -= center;

    rotation.mv(inEmbedded, out);

    out += center;
}


struct NeumannFunction
    : public Dune::VirtualFunction<FieldVector<double,dim>, FieldVector<double,3> >
{
    NeumannFunction(double homotopyParameter)
    : homotopyParameter_(homotopyParameter)
    {}

    void evaluate(const FieldVector<double, dim>& x, FieldVector<double,3>& out) const {
        out = 0;
        out[2] = -40*homotopyParameter_;
    }

    double homotopyParameter_;
};


int main (int argc, char *argv[]) try
{
    //feenableexcept(FE_INVALID);

    typedef std::vector<TargetSpace> SolutionType;

    // parse data file
    ParameterTree parameterSet;
    ParameterTreeParser::readINITree("cosserat-continuum.parset", parameterSet);

    ParameterTreeParser::readOptions(argc, argv, parameterSet);

    // read solver settings
    const int numLevels                   = parameterSet.get<int>("numLevels");
    int numHomotopySteps                  = parameterSet.get<int>("numHomotopySteps");
    const double tolerance                = parameterSet.get<double>("tolerance");
    const int maxTrustRegionSteps         = parameterSet.get<int>("maxTrustRegionSteps");
    const double initialTrustRegionRadius = parameterSet.get<double>("initialTrustRegionRadius");
    const int multigridIterations         = parameterSet.get<int>("numIt");
    const int nu1                         = parameterSet.get<int>("nu1");
    const int nu2                         = parameterSet.get<int>("nu2");
    const int mu                          = parameterSet.get<int>("mu");
    const int baseIterations              = parameterSet.get<int>("baseIt");
    const double mgTolerance              = parameterSet.get<double>("mgTolerance");
    const double baseTolerance            = parameterSet.get<double>("baseTolerance");
    const bool instrumented               = parameterSet.get<bool>("instrumented");
    std::string resultPath                = parameterSet.get("resultPath", "");

    // read problem settings
    std::string path                = parameterSet.get<std::string>("path");
    std::string gridFile            = parameterSet.get<std::string>("gridFile");

    // ///////////////////////////////////////
    //    Create the grid
    // ///////////////////////////////////////
    typedef std::conditional<dim==1,OneDGrid,UGGrid<dim> >::type GridType;
    array<unsigned int,dim> elements;
    elements.fill(1);
    elements[0] = 10;
    FieldVector<double,dim> upper(1);
    upper[0] = 10;
    shared_ptr<GridType> grid = StructuredGridFactory<GridType>::createCubeGrid(FieldVector<double,dim>(0),
                                                                                      upper,
                                                                                      elements);

    grid->globalRefine(numLevels-1);

    SolutionType x(grid->size(dim));

    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////

    BitSetVector<blocksize> dirichletNodes(grid->size(dim), false);
    BitSetVector<1> neumannNodes(grid->size(dim), false);

    GridType::Codim<dim>::LeafIterator vIt    = grid->leafbegin<dim>();
    GridType::Codim<dim>::LeafIterator vEndIt = grid->leafend<dim>();

    for (; vIt!=vEndIt; ++vIt) {
        if (vIt->geometry().corner(0)[0] < 1.0+1e-3 /* or vIt->geometry().corner(0)[0] > upper[0]-1e-3*/ ) {
            // Only translation dofs are Dirichlet
            for (int j=0; j<3; j++)
                dirichletNodes[grid->leafIndexSet().index(*vIt)][j] = true;
        }
        if (vIt->geometry().corner(0)[0] > upper[0]-1e-3 ) {
            neumannNodes[grid->leafIndexSet().index(*vIt)][0] = true;
        }

    }

    //////////////////////////////////////////////////////////////////////////////
    //   Assemble Neumann term
    //////////////////////////////////////////////////////////////////////////////

    typedef P1NodalBasis<GridType::LeafGridView,double> P1Basis;
    P1Basis p1Basis(grid->leafView());

    BoundaryPatch<GridType::LeafGridView> neumannBoundary(grid->leafView(), neumannNodes);

    std::cout << "Neumann boundary has " << neumannBoundary.numFaces() << " faces\n";

    // //////////////////////////
    //   Initial solution
    // //////////////////////////

    vIt    = grid->leafbegin<dim>();

    for (; vIt!=vEndIt; ++vIt) {
        int idx = grid->leafIndexSet().index(*vIt);

        x[idx].r = 0;
        for (int i=0; i<dim; i++)
            x[idx].r[i] = vIt->geometry().corner(0)[i];

        // x[idx].q is the identity, set by the default constructor

    }

    ////////////////////////////////////////////////////////
    //   Main homotopy loop
    ////////////////////////////////////////////////////////

    for (int i=0; i<numHomotopySteps; i++) {

        double homotopyParameter = (i+1)*(1.0/numHomotopySteps);
        std::cout << "Homotopy step: " << i << ",    parameter: " << homotopyParameter << std::endl;


    // ////////////////////////////////////////////////////////////
    //   Create an assembler for the energy functional
    // ////////////////////////////////////////////////////////////

    const ParameterTree& materialParameters = parameterSet.sub("materialParameters");
    NeumannFunction neumannFunction(homotopyParameter);

    std::cout << "Material parameters:" << std::endl;
    materialParameters.report();

    CosseratEnergyLocalStiffness<GridType::LeafGridView,
                                 P1Basis::LocalFiniteElement,
                                 3> cosseratEnergyLocalStiffness(materialParameters,
                                                                 &neumannBoundary,
                                                                 &neumannFunction);

    GeodesicFEAssembler<P1Basis,TargetSpace> assembler(grid->leafView(),
                                                                      &cosseratEnergyLocalStiffness);

    // /////////////////////////////////////////////////
    //   Create a Riemannian trust-region solver
    // /////////////////////////////////////////////////

    RiemannianTrustRegionSolver<GridType,TargetSpace> solver;
    solver.setup(*grid,
                 &assembler,
                 x,
                 dirichletNodes,
                 tolerance,
                 maxTrustRegionSteps,
                 initialTrustRegionRadius,
                 multigridIterations,
                 mgTolerance,
                 mu, nu1, nu2,
                 baseIterations,
                 baseTolerance,
                 instrumented);

        ////////////////////////////////////////////////////////
        //   Set Dirichlet values
        ////////////////////////////////////////////////////////

        for (vIt=grid->leafbegin<dim>(); vIt!=vEndIt; ++vIt) {

            int idx = grid->leafIndexSet().index(*vIt);
            if (dirichletNodes[idx][0] and vIt->geometry().corner(0)[0] > upper[0]-1e-3) {

                // Only the positions have Dirichlet values
                dirichletValues(vIt->geometry().corner(0), x[idx].r,
                                homotopyParameter);

            }

        }

        // /////////////////////////////////////////////////////
        //   Solve!
        // /////////////////////////////////////////////////////

        std::cout << "Energy: " << assembler.computeEnergy(x) << std::endl;
        //exit(0);

        solver.setInitialSolution(x);
        solver.solve();

        x = solver.getSol();

    }

    // //////////////////////////////
    //   Output result
    // //////////////////////////////

    CosseratVTKWriter<GridType>::write(*grid,x, resultPath + "cosserat");

    // finally: compute the average deformation of the Neumann boundary
    // That is what we need for the locking tests
    FieldVector<double,3> averageDef(0);
    for (size_t i=0; i<x.size(); i++)
        if (neumannNodes[i][0])
            averageDef += x[i].r;
    averageDef /= neumannNodes.count();

    std::cout << "mu_c = " << parameterSet.get<double>("materialParameters.mu_c") << "  "
              << "kappa = " << parameterSet.get<double>("materialParameters.kappa") << "  "
              << numLevels << " levels,  average deflection: " << averageDef << std::endl;

    // //////////////////////////////
 } catch (Exception e) {

    std::cout << e << std::endl;

 }
