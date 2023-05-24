#include <config.h>

#include <fenv.h>
#include <array>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/typetraits.hh>

#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/io/file/vtk.hh>

#if HAVE_DUNE_FOAMGRID
#include <dune/foamgrid/foamgrid.hh>
#else
#include <dune/grid/onedgrid.hh>
#endif

#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>

#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/functiontools/boundarydofs.hh>
#include <dune/fufem/dunepython.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/localprojectedfefunction.hh>
#include <dune/gfe/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/harmonicenergy.hh>
#include <dune/gfe/chiralskyrmionenergy.hh>
#include <dune/gfe/geodesicfeassembler.hh>
#include <dune/gfe/riemanniantrsolver.hh>
#include <dune/gfe/embeddedglobalgfefunction.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rigidbodymotion.hh>
#include <dune/gfe/spaces/rotation.hh>
#include <dune/gfe/spaces/unitvector.hh>

// grid dimension
const int dim = 2;
const int dimworld = 2;

// Image space of the geodesic fe functions
// typedef Rotation<double,2> TargetSpace;
// typedef Rotation<double,3> TargetSpace;
// typedef RigidBodyMotion<double,3> TargetSpace;
// typedef UnitVector<double,2> TargetSpace;
typedef UnitVector<double,3> TargetSpace;
// typedef UnitVector<double,4> TargetSpace;
// typedef RealTuple<double,1> TargetSpace;

// Tangent vector of the image space
const int blocksize = TargetSpace::TangentVector::dimension;

const int order = 1;

using namespace Dune;

template <typename Writer, typename Basis, typename SolutionType>
void fillVTKWriter(Writer& vtkWriter, const Basis& feBasis, const SolutionType& x, std::string filename)
{
  typedef BlockVector<TargetSpace::CoordinateType> EmbeddedVectorType;
  EmbeddedVectorType xEmbedded(x.size());
  for (size_t i=0; i<x.size(); i++)
    xEmbedded[i] = x[i].globalCoordinates();

  if constexpr (std::is_same<TargetSpace, Rotation<double,3> >::value)
  {
    std::array<BlockVector<FieldVector<double,3> >,3> director;
    for (int i=0; i<3; i++)
      director[i].resize(x.size());

    for (size_t i=0; i<x.size(); i++)
    {
      FieldMatrix<double,3,3> m;
      x[i].matrix(m);
      director[0][i] = {m[0][0], m[1][0], m[2][0]};
      director[1][i] = {m[0][1], m[1][1], m[2][1]};
      director[2][i] = {m[0][2], m[1][2], m[2][2]};
    }

    auto dFunction0 = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,3> >(feBasis,director[0]);
    auto dFunction1 = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,3> >(feBasis,director[1]);
    auto dFunction2 = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,3> >(feBasis,director[2]);

    vtkWriter.addVertexData(dFunction0, VTK::FieldInfo("director0", VTK::FieldInfo::Type::vector, 3));
    vtkWriter.addVertexData(dFunction1, VTK::FieldInfo("director1", VTK::FieldInfo::Type::vector, 3));
    vtkWriter.addVertexData(dFunction2, VTK::FieldInfo("director2", VTK::FieldInfo::Type::vector, 3));

    // Needs to be in this scope; otherwise the stack-allocated dFunction?-objects will get
    // destructed before 'write' is called.
    vtkWriter.write(filename);
  }
  else
  {
    auto xFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<TargetSpace::CoordinateType>(feBasis,xEmbedded);

    vtkWriter.addVertexData(xFunction, VTK::FieldInfo("orientation", VTK::FieldInfo::Type::vector, xEmbedded[0].size()));

    vtkWriter.write(filename);
  }
}


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
        << std::endl << "import os"
        << std::endl << "sys.path.append(os.getcwd() + '/../../problems/')"
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
#if HAVE_DUNE_FOAMGRID
    typedef std::conditional<dim==1 or dim!=dimworld,FoamGrid<dim,dimworld>,UGGrid<dim> >::type GridType;
#else
    static_assert(dim==dimworld, "You need to have dune-foamgrid installed for dim != dimworld!");
    typedef std::conditional<dim==1,OneDGrid,UGGrid<dim> >::type GridType;
#endif

    std::shared_ptr<GridType> grid;
    FieldVector<double,dimworld> lower(0), upper(1);
    std::array<unsigned int,dim> elements;

    std::string structuredGridType = parameterSet["structuredGrid"];
    if (structuredGridType != "false" ) {

        lower = parameterSet.get<FieldVector<double,dimworld> >("lower");
        upper = parameterSet.get<FieldVector<double,dimworld> >("upper");

        elements = parameterSet.get<std::array<unsigned int,dim> >("elements");
        if (structuredGridType == "simplex")
            grid = StructuredGridFactory<GridType>::createSimplexGrid(lower, upper, elements);
        else if (structuredGridType == "cube")
            grid = StructuredGridFactory<GridType>::createCubeGrid(lower, upper, elements);
        else
          DUNE_THROW(Exception, "Unknown structured grid type '" << structuredGridType << "' found!");

    } else {

        std::string path                = parameterSet.get<std::string>("path");
        std::string gridFile            = parameterSet.get<std::string>("gridFile");

        grid = std::shared_ptr<GridType>(GmshReader<GridType>::read(path + "/" + gridFile));
    }

    grid->globalRefine(numLevels-1);

    //////////////////////////////////////////////////////////////////////////////////
    //  Construct the scalar function space basis corresponding to the GFE space
    //////////////////////////////////////////////////////////////////////////////////

    using GridView = GridType::LeafGridView;
    GridView gridView = grid->leafGridView();

    typedef Dune::Functions::LagrangeBasis<GridView, order> FEBasis;
    FEBasis feBasis(gridView);
    SolutionType x(feBasis.size());

    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////
    BitSetVector<1> dirichletVertices(gridView.size(dim), false);
    const GridView::IndexSet& indexSet = gridView.indexSet();

    // Make Python function that computes which vertices are on the Dirichlet boundary,
    // based on the vertex positions.
    std::string lambda = std::string("lambda x: (") + parameterSet.get<std::string>("dirichletVerticesPredicate") + std::string(")");
    auto pythonDirichletVertices = Python::make_function<bool>(Python::evaluate(lambda));

    for (auto&& vertex : vertices(gridView))
    {
      bool isDirichlet = pythonDirichletVertices(vertex.geometry().corner(0));
      dirichletVertices[indexSet.index(vertex)] = isDirichlet;
    }

    BoundaryPatch<GridView> dirichletBoundary(gridView, dirichletVertices);

    BitSetVector<blocksize> dirichletNodes(feBasis.size(), false);

    constructBoundaryDofs(dirichletBoundary,feBasis,dirichletNodes);

    // //////////////////////////
    //   Initial iterate
    // //////////////////////////

    // Read initial iterate into a Python function
    Python::Module module = Python::import(parameterSet.get<std::string>("initialIterate"));
    auto pythonInitialIterate = Python::makeFunction<TargetSpace::CoordinateType(const FieldVector<double,dimworld>&)>(module.get("f"));

    std::vector<TargetSpace::CoordinateType> v;
    using namespace Functions::BasisFactory;

      auto powerBasis = makeBasis(
        gridView,
        power<TargetSpace::CoordinateType::dimension>(
          lagrange<order>(),
          blockedInterleaved()
      ));

    Dune::Functions::interpolate(powerBasis, v, pythonInitialIterate);

    for (size_t i=0; i<x.size(); i++)
      x[i] = v[i];

    // backup for error measurement later
    SolutionType initialIterate = x;

    // ////////////////////////////////////////////////////////////
    //   Create an assembler for the Harmonic Energy Functional
    // ////////////////////////////////////////////////////////////

    typedef TargetSpace::rebind<adouble>::other ATargetSpace;
    using GeodesicInterpolationRule  = LocalGeodesicFEFunction<dim, double, FEBasis::LocalView::Tree::FiniteElement, ATargetSpace>;
    using ProjectedInterpolationRule = GFE::LocalProjectedFEFunction<dim, double, FEBasis::LocalView::Tree::FiniteElement, ATargetSpace>;

    // Assembler using ADOL-C
    std::shared_ptr<GFE::LocalEnergy<FEBasis,ATargetSpace> > localEnergy;

    std::string energy = parameterSet.get<std::string>("energy");
    if (energy == "harmonic")
    {
        if (parameterSet["interpolationMethod"] == "geodesic")
            localEnergy.reset(new HarmonicEnergy<FEBasis, GeodesicInterpolationRule, ATargetSpace>);
        else if (parameterSet["interpolationMethod"] == "projected")
            localEnergy.reset(new HarmonicEnergy<FEBasis, ProjectedInterpolationRule, ATargetSpace>);
        else
            DUNE_THROW(Exception, "Unknown interpolation method " << parameterSet["interpolationMethod"] << " requested!");

    } else if (energy == "chiral_skyrmion")
    {
//       // Doesn't work: we are not inside of a template
//       if constexpr (std::is_same<TargetSpace, UnitVector<double,3> >::value)
//       {
        if (parameterSet["interpolationMethod"] == "geodesic")
            localEnergy.reset(new GFE::ChiralSkyrmionEnergy<FEBasis, GeodesicInterpolationRule, adouble>(parameterSet.sub("energyParameters")));
        else if (parameterSet["interpolationMethod"] == "projected")
            localEnergy.reset(new GFE::ChiralSkyrmionEnergy<FEBasis, ProjectedInterpolationRule, adouble>(parameterSet.sub("energyParameters")));
        else
            DUNE_THROW(Exception, "Unknown interpolation method " << parameterSet["interpolationMethod"] << " requested!");
//       } else
//         DUNE_THROW(Exception, "Build program with TargetSpace = UnitVector<3> for the ChiralSkyrmion energy!");

    } else
      DUNE_THROW(Exception, "Unknown energy type '" << energy << "'");

    LocalGeodesicFEADOLCStiffness<FEBasis,TargetSpace> localGFEADOLCStiffness(localEnergy.get());

    GeodesicFEAssembler<FEBasis,TargetSpace> assembler(feBasis, localGFEADOLCStiffness);

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

    solver.setInitialIterate(x);
    solver.solve();

    x = solver.getSol();

    // //////////////////////////////
    //   Output result
    // //////////////////////////////

    SubsamplingVTKWriter<GridView> vtkWriter(gridView,Dune::refinementLevels(order-1));
    std::string baseName = "harmonicmaps-result-" + std::to_string(order) + "-" + std::to_string(numLevels);
    fillVTKWriter(vtkWriter, feBasis, x, resultPath + baseName);

    // Write the corresponding coefficient vector: verbatim in binary, to be completely lossless
    typedef BlockVector<TargetSpace::CoordinateType> EmbeddedVectorType;
    EmbeddedVectorType xEmbedded(x.size());
    for (size_t i=0; i<x.size(); i++)
      xEmbedded[i] = x[i].globalCoordinates();

    std::ofstream outFile(baseName + ".data", std::ios_base::binary);
    MatrixVector::Generic::writeBinary(outFile, xEmbedded);
    outFile.close();

    return 0;
 }
