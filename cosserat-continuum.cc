#include <config.h>

#include <fenv.h>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>    // use of "Easy to Use" drivers
#include <adolc/taping.h>
#undef overwrite  // stupid: ADOL-C sets this to 1, so the name cannot be used

#include <dune/gfe/adolcnamespaceinjections.hh>
#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/onedgrid.hh>
#include <dune/grid/geometrygrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/grid/io/file/gmshreader.hh>

#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/functionspacebases/p1nodalbasis.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/cosseratenergystiffness.hh>
#include <dune/gfe/cosseratvtkwriter.hh>
#include <dune/gfe/geodesicfeassembler.hh>
#include <dune/gfe/riemanniantrsolver.hh>

// grid dimension
const int dim = 2;

// Image space of the geodesic fe functions
typedef RigidBodyMotion<double,3> TargetSpace;

// Tangent vector of the image space
const int blocksize = TargetSpace::TangentVector::dimension;

using namespace Dune;

#if 1
// Dirichlet boundary data for the shear/wrinkling example
void dirichletValues(const FieldVector<double,dim>& in, FieldVector<double,3>& out,
                     double homotopy)
{
    out = 0;
    for (int i=0; i<dim; i++)
        out[i] = in[i];

//     if (out[1] > 1-1e-3)
     if (out[1] > 0.18-1e-4)
        out[0] += 0.003*homotopy;
}
#endif
#if 0
// Dirichlet boundary data for the 'twisted-strip' example
void dirichletValues(const FieldVector<double,dim>& in, FieldVector<double,3>& out,
                     double homotopy
)
{
    if (not vIt->geometry().corner(0)[0] > upper[0]-1e-3)
        return;

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
#endif

/** \brief A constant vector-valued function, for simple Neumann boundary values */
struct NeumannFunction
    : public Dune::VirtualFunction<FieldVector<double,dim>, FieldVector<double,3> >
{
    NeumannFunction(const FieldVector<double,3> values,
                    double homotopyParameter)
    : values_(values),
      homotopyParameter_(homotopyParameter)
    {}

    void evaluate(const FieldVector<double, dim>& x, FieldVector<double,3>& out) const {
        out = 0;
        out.axpy(homotopyParameter_, values_);
    }

    FieldVector<double,3> values_;
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

    // ///////////////////////////////////////
    //    Create the grid
    // ///////////////////////////////////////
    typedef std::conditional<dim==1,OneDGrid,UGGrid<dim> >::type GridType;

    shared_ptr<GridType> grid;

    FieldVector<double,dim> lower = parameterSet.get<FieldVector<double,dim> >("lower");
    FieldVector<double,dim> upper = parameterSet.get<FieldVector<double,dim> >("upper");

    if (parameterSet.get<bool>("structuredGrid")) {

        array<unsigned int,dim> elements = parameterSet.get<array<unsigned int,dim> >("elements");
        grid = StructuredGridFactory<GridType>::createCubeGrid(lower, upper, elements);

    } else {
        std::string path                = parameterSet.get<std::string>("path");
        std::string gridFile            = parameterSet.get<std::string>("gridFile");
        grid = shared_ptr<GridType>(GmshReader<GridType>::read(path + "/" + gridFile));
    }

    grid->globalRefine(numLevels-1);

    typedef GridType::LeafGridView GridView;
    GridView gridView = grid->leafGridView();

    typedef P1NodalBasis<GridView,double> FEBasis;
    FEBasis feBasis(gridView);

    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////

    BitSetVector<1> dirichletVertices(feBasis.size(), false);
    BitSetVector<1> neumannNodes(feBasis.size(), false);

    GridType::Codim<dim>::LeafIterator vIt    = gridView.begin<dim>();
    GridType::Codim<dim>::LeafIterator vEndIt = gridView.end<dim>();

    const GridView::IndexSet& indexSet = gridView.indexSet();

    for (; vIt!=vEndIt; ++vIt) {

#if 0   // Boundary conditions for the cantilever example
        if (vIt->geometry().corner(0)[0] < 1.0+1e-3 /* or vIt->geometry().corner(0)[0] > upper[0]-1e-3*/ )
          dirichletVertices[indexSet.index(*vIt)] = true;
        if (vIt->geometry().corner(0)[0] > upper[0]-1e-3 )
            neumannNodes[indexSet.index(*vIt)] = true;
#endif
#if 1   // Boundary conditions for the shearing/wrinkling example
        if (vIt->geometry().corner(0)[1] < 1e-4  or vIt->geometry().corner(0)[1] > upper[1]-1e-4 )
          dirichletVertices[indexSet.index(*vIt)] = true;
#endif
#if 0   // Boundary conditions for the twisted-strip example
        if (vIt->geometry().corner(0)[0] < lower[0]+1e-3  or vIt->geometry().corner(0)[0] > upper[0]-1e-3 )
          dirichletVertices[indexSet.index(*vIt)] = true;
#endif
#if 0   // Boundary conditions for the L-shape example
        if (vIt->geometry().corner(0)[0] < 1.0)
          dirichletVertices[indexSet.index(*vIt)] = true;
        if (vIt->geometry().corner(0)[1] < -239 )
            neumannNodes[indexSet.index(*vIt)][0] = true;
#endif
    }

    BoundaryPatch<GridView> dirichletBoundary(gridView, dirichletVertices);
    BoundaryPatch<GridView> neumannBoundary(gridView, neumannNodes);

    std::cout << "Neumann boundary has " << neumannBoundary.numFaces() << " faces\n";



    BitSetVector<blocksize> dirichletNodes(feBasis.size(), false);
    for (size_t i=0; i<feBasis.size(); i++)
      if (dirichletVertices[i][0])
        for (int j=0; j<5; j++)
          dirichletNodes[i][j] = true;

    // //////////////////////////
    //   Initial iterate
    // //////////////////////////

    SolutionType x(feBasis.size());

    vIt = gridView.begin<dim>();

    for (; vIt!=vEndIt; ++vIt) {
        int idx = indexSet.index(*vIt);

        x[idx].r = 0;
        for (int i=0; i<dim; i++)
            x[idx].r[i] = vIt->geometry().corner(0)[i];

        x[idx].r[2] = 0.002*std::cos(1e4*vIt->geometry().corner(0)[0]);
        // x[idx].q is the identity, set by the default constructor

    }

    ////////////////////////////////////////////////////////
    //   Main homotopy loop
    ////////////////////////////////////////////////////////

    // Output initial iterate (of homotopy loop)
    CosseratVTKWriter<GridType>::write(*grid,x, resultPath + "cosserat_homotopy_0");

    for (int i=0; i<numHomotopySteps; i++) {

        double homotopyParameter = (i+1)*(1.0/numHomotopySteps);
        std::cout << "Homotopy step: " << i << ",    parameter: " << homotopyParameter << std::endl;


    // ////////////////////////////////////////////////////////////
    //   Create an assembler for the energy functional
    // ////////////////////////////////////////////////////////////

    const ParameterTree& materialParameters = parameterSet.sub("materialParameters");
    shared_ptr<NeumannFunction> neumannFunction;
    if (parameterSet.hasKey("neumannValues"))
        neumannFunction = make_shared<NeumannFunction>(parameterSet.get<FieldVector<double,3> >("neumannValues"),
                                                       homotopyParameter);

    std::cout << "Material parameters:" << std::endl;
    materialParameters.report();

    // Assembler using ADOL-C
    CosseratEnergyLocalStiffness<GridView,
                                 FEBasis::LocalFiniteElement,
                                 3,adouble> cosseratEnergyADOLCLocalStiffness(materialParameters,
                                                                              &neumannBoundary,
                                                                              neumannFunction.get());
    LocalGeodesicFEADOLCStiffness<GridView,
                                  FEBasis::LocalFiniteElement,
                                  TargetSpace> localGFEADOLCStiffness(&cosseratEnergyADOLCLocalStiffness);

    GeodesicFEAssembler<FEBasis,TargetSpace> assembler(gridView, &localGFEADOLCStiffness);

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

        for (vIt=gridView.begin<dim>(); vIt!=vEndIt; ++vIt) {

            int idx = indexSet.index(*vIt);
            if (dirichletNodes[idx][0]) {

                // Only the positions have Dirichlet values
                dirichletValues(vIt->geometry().corner(0), x[idx].r,
                                homotopyParameter);

            }

        }

        // /////////////////////////////////////////////////////
        //   Solve!
        // /////////////////////////////////////////////////////

        solver.setInitialIterate(x);
        solver.solve();

        x = solver.getSol();

        // Output result of each homotopy step
        std::stringstream iAsAscii;
        iAsAscii << i+1;
        CosseratVTKWriter<GridType>::write(*grid,x, resultPath + "cosserat_homotopy_" + iAsAscii.str());

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
