#include <config.h>

#include <fenv.h>

//#define LAPLACE_DEBUG
//#define HARMONIC_ENERGY_FD_GRADIENT

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

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/unitvector.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/harmonicenergystiffness.hh>
#include <dune/gfe/geodesicfeassembler.hh>
#include <dune/gfe/riemanniantrsolver.hh>

// grid dimension
const int dim = 2;

// Image space of the geodesic fe functions
#ifdef RIGIDBODYMOTION3
typedef RigidBodyMotion<3> TargetSpace;
#endif

// Tangent vector of the image space
const int blocksize = TargetSpace::TangentVector::size;

using namespace Dune;


template <class HostGridView>
class DeformationFunction
    : public Dune :: DiscreteCoordFunction< double, 3, DeformationFunction<HostGridView> >
{
    typedef DeformationFunction<HostGridView> This;
    typedef Dune :: DiscreteCoordFunction< double, 3, This > Base;

  public:

    DeformationFunction(const HostGridView& gridView,
                        const std::vector<RigidBodyMotion<3> >& deformedPosition)
        : gridView_(gridView),
          deformedPosition_(deformedPosition)
    {}

    void evaluate ( const typename HostGridView::template Codim<dim>::Entity& hostEntity, unsigned int corner,
                    FieldVector<double,3> &y ) const
    {

        const typename HostGridView::IndexSet& indexSet = gridView_.indexSet();

        int idx = indexSet.index(hostEntity);
        y = deformedPosition_[idx].r;
    }

    void evaluate ( const typename HostGridView::template Codim<0>::Entity& hostEntity, unsigned int corner,
                    FieldVector<double,3> &y ) const
    {

        const typename HostGridView::IndexSet& indexSet = gridView_.indexSet();

        int idx = indexSet.subIndex(hostEntity, corner,dim);

        y = deformedPosition_[idx].r;
    }

private:

    HostGridView gridView_;

    const std::vector<RigidBodyMotion<3> > deformedPosition_;

};


int main (int argc, char *argv[]) try
{
    //feenableexcept(FE_INVALID);

    typedef std::vector<TargetSpace> SolutionType;

    // parse data file
    ParameterTree parameterSet;
    if (argc==2)
        ParameterTreeParser::readINITree(argv[1], parameterSet);
    else
        ParameterTreeParser::readINITree("cosserat-continuum.parset", parameterSet);

    // read solver settings
    const int numLevels                   = parameterSet.get<int>("numLevels");
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
    elements.fill(3);
    shared_ptr<GridType> gridPtr = StructuredGridFactory<GridType>::createSimplexGrid(FieldVector<double,dim>(0),
                                                                                      FieldVector<double,dim>(1),
                                                                                      elements);
    GridType& grid = *gridPtr.get();

    grid.globalRefine(numLevels-1);

    SolutionType x(grid.size(dim));

    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////

    BitSetVector<1> allNodes(grid.size(dim));
    allNodes.setAll();
    LeafBoundaryPatch<GridType> dirichletBoundary(grid, allNodes);

    BitSetVector<blocksize> dirichletNodes(grid.size(dim));
    for (int i=0; i<dirichletNodes.size(); i++)
        dirichletNodes[i] = dirichletBoundary.containsVertex(i);
    
    // //////////////////////////
    //   Initial solution
    // //////////////////////////

    FieldVector<double,3> yAxis(0);
    yAxis[1] = 1;

    GridType::LeafGridView::Codim<dim>::Iterator vIt    = grid.leafbegin<dim>();
    GridType::LeafGridView::Codim<dim>::Iterator vEndIt = grid.leafend<dim>();

    for (; vIt!=vEndIt; ++vIt) {
        int idx = grid.leafIndexSet().index(*vIt);

        x[idx].r = 0;
        for (int i=0; i<dim; i++)
            x[idx].r[i] = vIt->geometry().corner(0)[i];

        // x[idx].q is the identity, set by the default constructor

        if (dirichletNodes[idx][0]) {
            
            // Only the positions have Dirichlet values
            x[idx].r[2] = vIt->geometry().corner(0)[0];

        }

    }
        
    // backup for error measurement later
    SolutionType initialIterate = x;

    // ////////////////////////////////////////////////////////////
    //   Create an assembler for the Harmonic Energy Functional
    // ////////////////////////////////////////////////////////////
#if 0
    HarmonicEnergyLocalStiffness<GridType::LeafGridView,TargetSpace> harmonicEnergyLocalStiffness;

    GeodesicFEAssembler<GridType::LeafGridView,TargetSpace> assembler(grid.leafView(),
                                                                      &harmonicEnergyLocalStiffness);

    // /////////////////////////////////////////////////
    //   Create a Riemannian trust-region solver
    // /////////////////////////////////////////////////

    RiemannianTrustRegionSolver<GridType,TargetSpace> solver;
    solver.setup(grid, 
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
    
    // /////////////////////////////////////////////////////
    //   Solve!
    // /////////////////////////////////////////////////////
    
    std::cout << "Energy: " << assembler.computeEnergy(x) << std::endl;
    //exit(0);

    solver.setInitialSolution(x);
    solver.solve();

    x = solver.getSol();
#endif

    // //////////////////////////////
    //   Output result
    // //////////////////////////////

    typedef GeometryGrid<GridType,DeformationFunction<GridType::LeafGridView> > DeformedGridType;
    
    DeformationFunction<GridType::LeafGridView> deformationFunction(grid.leafView(),x);
    
    DeformedGridType deformedGrid(grid, deformationFunction);


    LeafAmiraMeshWriter<DeformedGridType> amiramesh;
    amiramesh.writeSurfaceGrid(deformedGrid.leafView(), "cosseratGrid");
/*    amiramesh.addGrid(deformedGrid.leafView());
    amiramesh.write("cosseratGrid", 1);*/
    
    // //////////////////////////////
 } catch (Exception e) {

    std::cout << e << std::endl;

 }
