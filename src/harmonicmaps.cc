#include <config.h>

#include <fenv.h>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/typetraits.hh>
namespace Dune {
  template <>
  struct IsNumber<adouble>
  {
    constexpr static bool value = true;
  };
}

#include <array>

#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/onedgrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/io/file/vtk.hh>

#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
#include <dune/functions/functionspacebases/pqknodalbasis.hh>
#include <dune/functions/functionspacebases/bsplinebasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>

#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/functiontools/basisinterpolator.hh>
#include <dune/fufem/functiontools/boundarydofs.hh>
#include <dune/fufem/functionspacebases/dunefunctionsbasis.hh>
#include <dune/fufem/discretizationerror.hh>
#include <dune/fufem/dunepython.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/unitvector.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/localprojectedfefunction.hh>
#include <dune/gfe/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/harmonicenergystiffness.hh>
#include <dune/gfe/chiralskyrmionenergy.hh>
#include <dune/gfe/geodesicfeassembler.hh>
#include <dune/gfe/riemanniantrsolver.hh>
#include <dune/gfe/embeddedglobalgfefunction.hh>

// grid dimension
const int dim = 2;

// Image space of the geodesic fe functions
// typedef Rotation<double,2> TargetSpace;
// typedef Rotation<double,3> TargetSpace;
// typedef UnitVector<double,2> TargetSpace;
typedef UnitVector<double,3> TargetSpace;
// typedef UnitVector<double,4> TargetSpace;
// typedef RealTuple<double,1> TargetSpace;

// Tangent vector of the image space
const int blocksize = TargetSpace::TangentVector::dimension;

const int order = 1;

#define LAGRANGE

using namespace Dune;


int main (int argc, char *argv[])
{
    MPIHelper::instance(argc, argv);

    //feenableexcept(FE_INVALID);
    // Start Python interpreter
    Python::start();
    Python::Reference main = Python::import("__main__");
    Python::run("import math");

    //feenableexcept(FE_INVALID);
    Python::runStream()
        << std::endl << "import sys"
        << std::endl << "sys.path.append('/home/sander/dune/dune-gfe/problems')"
        << std::endl;


    typedef std::vector<TargetSpace> SolutionType;

    // parse data file
    ParameterTree parameterSet;
    if (argc < 2)
      DUNE_THROW(Exception, "Usage: ./harmonicmaps <parameter file>");

    ParameterTreeParser::readINITree(argv[1], parameterSet);

    ParameterTreeParser::readOptions(argc, argv, parameterSet);

    // read problem settings
    const int numLevels                   = parameterSet.get<int>("numLevels");
#ifndef LAGRANGE
    const int order                       = parameterSet.get<int>("order");
#endif
    // read solver settings
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
    std::string resultPath                = parameterSet.get("resultPath", "");

    // ///////////////////////////////////////
    //    Create the grid
    // ///////////////////////////////////////
    typedef std::conditional<dim==1,OneDGrid,UGGrid<dim> >::type GridType;

    shared_ptr<GridType> grid;
    FieldVector<double,dim> lower(0), upper(1);
    std::array<unsigned int,dim> elements;

    if (parameterSet.get<bool>("structuredGrid")) {

        lower = parameterSet.get<FieldVector<double,dim> >("lower");
        upper = parameterSet.get<FieldVector<double,dim> >("upper");

        elements = parameterSet.get<std::array<unsigned int,dim> >("elements");
        grid = StructuredGridFactory<GridType>::createCubeGrid(lower, upper, elements);

    } else {

        std::string path                = parameterSet.get<std::string>("path");
        std::string gridFile            = parameterSet.get<std::string>("gridFile");

        grid = shared_ptr<GridType>(GmshReader<GridType>::read(path + "/" + gridFile));
    }

    grid->globalRefine(numLevels-1);

    //////////////////////////////////////////////////////////////////////////////////
    //  Construct the scalar function space basis corresponding to the GFE space
    //////////////////////////////////////////////////////////////////////////////////

    using GridView = GridType::LeafGridView;
    GridView gridView = grid->leafGridView();
#ifdef LAGRANGE
    typedef Dune::Functions::PQkNodalBasis<GridView, order> FEBasis;
    FEBasis feBasis(gridView);
#else
    typedef Dune::Functions::BSplineBasis<GridView> FEBasis;
    for (auto& i : elements)
      i = i<<(numLevels-1);
    FEBasis feBasis(gridView, lower, upper, elements, order);
#endif
    SolutionType x(feBasis.size());

    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////

    BitSetVector<1> allNodes(grid->size(dim));
    allNodes.setAll();
    BoundaryPatch<GridView> dirichletBoundary(gridView, allNodes);

    BitSetVector<blocksize> dirichletNodes;

    typedef DuneFunctionsBasis<FEBasis> FufemFEBasis;
    FufemFEBasis fufemFeBasis(feBasis);

    constructBoundaryDofs(dirichletBoundary,fufemFeBasis,dirichletNodes);

    // //////////////////////////
    //   Initial iterate
    // //////////////////////////

    // Read initial iterate into a PythonFunction
    typedef VirtualDifferentiableFunction<FieldVector<double, dim>, TargetSpace::CoordinateType> FBase;

    Python::Module module = Python::import(parameterSet.get<std::string>("initialIterate"));
    auto pythonInitialIterate = module.get("fdf").toC<std::shared_ptr<FBase>>();

    std::vector<TargetSpace::CoordinateType> v;
#ifdef LAGRANGE
    ::Functions::interpolate(fufemFeBasis, v, *pythonInitialIterate);
#else
    Dune::Functions::interpolate(feBasis, v, *pythonInitialIterate, lower, upper, elements, order);
#endif

    for (size_t i=0; i<x.size(); i++)
      x[i] = v[i];

    // backup for error measurement later
    SolutionType initialIterate = x;

    // ////////////////////////////////////////////////////////////
    //   Create an assembler for the Harmonic Energy Functional
    // ////////////////////////////////////////////////////////////

    typedef TargetSpace::rebind<adouble>::other ATargetSpace;
    typedef LocalGeodesicFEFunction<dim, double, FEBasis::LocalView::Tree::FiniteElement, ATargetSpace> LocalInterpolationRule;
    //typedef GFE::LocalProjectedFEFunction<dim, double, FEBasis::LocalView::Tree::FiniteElement, ATargetSpace> LocalInterpolationRule;
    std::cout << "Using local interpolation: " << className<LocalInterpolationRule>() << std::endl;

    // Assembler using ADOL-C
    std::shared_ptr<LocalGeodesicFEStiffness<FEBasis,ATargetSpace> > localEnergy;

    std::string energy = parameterSet.get<std::string>("energy");
    if (energy == "harmonic")
    {

      localEnergy.reset(new HarmonicEnergyLocalStiffness<FEBasis, LocalInterpolationRule, ATargetSpace>);

    } else if (energy == "chiral_skyrmion")
    {

      localEnergy.reset(new GFE::ChiralSkyrmionEnergy<FEBasis, LocalInterpolationRule, adouble>(parameterSet.sub("energyParameters")));

    } else
      DUNE_THROW(Exception, "Unknown energy type '" << energy << "'");

    LocalGeodesicFEADOLCStiffness<FEBasis,TargetSpace> localGFEADOLCStiffness(localEnergy.get());

    GeodesicFEAssembler<FEBasis,TargetSpace> assembler(feBasis, &localGFEADOLCStiffness);

    // /////////////////////////////////////////////////
    //   Create a Riemannian trust-region solver
    // /////////////////////////////////////////////////

    RiemannianTrustRegionSolver<FEBasis,TargetSpace> solver;
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
                 false);   // instrumented

    // /////////////////////////////////////////////////////
    //   Solve!
    // /////////////////////////////////////////////////////

    std::cout << "Energy: " << assembler.computeEnergy(x) << std::endl;
    //exit(0);

    solver.setInitialIterate(x);
    solver.solve();

    x = solver.getSol();

    // //////////////////////////////
    //   Output result
    // //////////////////////////////

    typedef BlockVector<TargetSpace::CoordinateType> EmbeddedVectorType;
    EmbeddedVectorType xEmbedded(x.size());
    for (size_t i=0; i<x.size(); i++)
        xEmbedded[i] = x[i].globalCoordinates();

    auto xFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<TargetSpace::CoordinateType>(feBasis,TypeTree::hybridTreePath(),xEmbedded);

    std::string baseName = "harmonicmaps-result-" + std::to_string(order) + "-" + std::to_string(numLevels);

    SubsamplingVTKWriter<GridView> vtkWriter(gridView,order-1);
    vtkWriter.addVertexData(xFunction, VTK::FieldInfo("orientation", VTK::FieldInfo::Type::vector, xEmbedded[0].size()));
    vtkWriter.write(resultPath + baseName);

    // Write the corresponding coefficient vector: verbatim in binary, to be completely lossless
    std::ofstream outFile(baseName + ".data", std::ios_base::binary);
    MatrixVector::Generic::writeBinary(outFile, xEmbedded);
    outFile.close();

    return 0;
 }
