#define MIXED_SPACE 0
#include <config.h>

#include <array>
#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/typetraits.hh>
#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/common/tuplevector.hh>

#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/uggrid.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/compositebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>

#include <dune/fufem/functiontools/boundarydofs.hh>
#include <dune/fufem/dunepython.hh>

#include <dune/gfe/assemblers/mixedlocalgfeadolcstiffness.hh>
#include <dune/gfe/assemblers/simofoxenergy.hh>
#include <dune/gfe/cosseratvtkwriter.hh>
#include <dune/gfe/embeddedglobalgfefunction.hh>
#include <dune/gfe/assemblers/mixedgfeassembler.hh>
#include <dune/gfe/mixedriemanniantrsolver.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/localprojectedfefunction.hh>
#include <dune/gfe/spaces/unitvector.hh>

#if !MIXED_SPACE
#include <dune/gfe/assemblers/geodesicfeassemblerwrapper.hh>
#include <dune/gfe/riemannianpnsolver.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#endif

#if HAVE_DUNE_VTK
#include <dune/vtk/vtkreader.hh>
#endif

template <int dim, class ctype, class LocalFiniteElement, class TS>
using LocalFEFunction = Dune::GFE::LocalProjectedFEFunction<dim,ctype,LocalFiniteElement,TS>;
//using LocalFEFunction = LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,TS>;

// Order of the approximation space for the midsurface position
const int midsurfaceOrder = 1;

// Order of the approximation space for the director
const int directorOrder = 1;

using namespace Dune;

#if !MIXED_SPACE
static_assert(midsurfaceOrder==directorOrder, "displacement and rotation order do not match!");
#endif

int main(int argc, char *argv[]) try
{
  // Initialize MPI, finalize is done automatically on exit
  Dune::MPIHelper &mpiHelper = MPIHelper::instance(argc, argv);

  // Start Python interpreter
  Python::start();
  Python::Reference main = Python::import("__main__");
  Python::run("import math");

  Python::runStream()
    << std::endl << "import sys"
    << std::endl << "import os"
    << std::endl << "sys.path.append(os.getcwd() + '/../../problems/')"
    << std::endl;

  using namespace TypeTree::Indices;
  using SolutionType = TupleVector<std::vector<RealTuple<double,3> >, std::vector<UnitVector<double,3> > >;

  // parse data file
  ParameterTree parameterSet;
  if (argc < 2)
    DUNE_THROW(Exception, "Usage: ./simofoxshell inputfile.dat");

  ParameterTreeParser::readINITree(argv[1], parameterSet);

  ParameterTreeParser::readOptions(argc, argv, parameterSet);

  // read solver settings
  const auto numLevels = parameterSet.get<int>("numLevels");
  const auto totalLoadSteps = parameterSet.get<int>("numHomotopySteps");
  const auto tolerance = parameterSet.get<double>("tolerance");
  const auto maxSolverSteps = parameterSet.get<int>("maxSolverSteps");
  const auto initialTrustRegionRadius = parameterSet.get<double>("initialTrustRegionRadius");
  const auto initialRegularization = parameterSet.get<double>("initialRegularization");
  const auto multigridIterations = parameterSet.get<int>("numIt");
  const auto nu1 = parameterSet.get<int>("nu1");
  const auto nu2 = parameterSet.get<int>("nu2");
  const auto mu = parameterSet.get<int>("mu");
  const auto baseIterations = parameterSet.get<int>("baseIt");
  const auto mgTolerance = parameterSet.get<double>("mgTolerance");
  const auto baseTolerance = parameterSet.get<double>("baseTolerance");
  const auto instrumented = parameterSet.get<bool>("instrumented");
  std::string resultPath = parameterSet.get("resultPath", "");

  /////////////////////////////////////////
  //    Create the grid
  /////////////////////////////////////////
  using Grid = UGGrid<2>;

  std::shared_ptr<Grid> grid;

  FieldVector<double, 2> lower(0), upper(1);

  std::string structuredGridType = parameterSet["structuredGrid"];
  if (parameterSet.get<bool>("structuredGrid"))
  {
    lower = parameterSet.get<FieldVector<double, 2> >("lower");
    upper = parameterSet.get<FieldVector<double, 2> >("upper");

    auto elements = parameterSet.get<std::array<unsigned int, 2> >("elements");
    grid = StructuredGridFactory<Grid>::createCubeGrid(lower, upper, elements);

  } else {
    auto path = parameterSet.get<std::string>("path");
    auto gridFile = parameterSet.get<std::string>("gridFile");

    // Guess the grid file format by looking at the file name suffix
    auto dotPos = gridFile.rfind('.');
    if (dotPos == std::string::npos)
      DUNE_THROW(IOError, "Could not determine grid input file format");
    std::string suffix = gridFile.substr(dotPos, gridFile.length() - dotPos);

    if (suffix == ".msh")
      grid = std::shared_ptr<Grid>(GmshReader<Grid>::read(path + "/" + gridFile));
    else if (suffix == ".vtu" or suffix == ".vtp")
#if HAVE_DUNE_VTK
      grid = VtkReader<Grid>::createGridFromFile(path + "/" + gridFile);
#else
      DUNE_THROW(NotImplemented, "Please install dune-vtk for VTK reading support!");
#endif
  }

  grid->globalRefine(numLevels - 1);
  grid->loadBalance();

  if (mpiHelper.rank() == 0)
    std::cout << "There are " << grid->leafGridView().comm().size() << " processes" << std::endl;

  using GridView = Grid::LeafGridView;
  GridView gridView = grid->leafGridView();

  using namespace Dune::Functions::BasisFactory;

  auto compositeBasis = makeBasis(gridView,
                                  composite(
                                    power<2>(lagrange<midsurfaceOrder>()),
                                    power<2>(lagrange<directorOrder>())));

  typedef Dune::Functions::LagrangeBasis<GridView, midsurfaceOrder> MidsurfaceFEBasis;
  typedef Dune::Functions::LagrangeBasis<GridView, directorOrder> DirectorFEBasis;

  MidsurfaceFEBasis midsurfaceFEBasis(gridView);
  DirectorFEBasis directorFEBasis(gridView);

  ///////////////////////////////////////////
  //   Read Dirichlet values
  ///////////////////////////////////////////

  BitSetVector<1> dirichletVertices(gridView.size(2), false);
  BitSetVector<1> neumannVertices(gridView.size(2), false);

  const GridView::IndexSet &indexSet = gridView.indexSet();

  // Make Python function that computes which vertices are on the Dirichlet boundary,
  // based on the vertex positions.
  std::string lambda = std::string("lambda x: (") + parameterSet.get<std::string>("dirichletVerticesPredicate") + std::string(")");
  auto pythonDirichletVertices = Python::make_function<bool>(Python::evaluate(lambda));

  // Same for the Neumann boundary
  lambda = std::string("lambda x: (") + parameterSet.get<std::string>("neumannVerticesPredicate", "0") + std::string(")");
  auto pythonNeumannVertices = Python::make_function<bool>(Python::evaluate(lambda));

  for (auto &&vertex: vertices(gridView))
  {
    bool isDirichlet = pythonDirichletVertices(vertex.geometry().corner(0));
    dirichletVertices[indexSet.index(vertex)] = isDirichlet;

    bool isNeumann = pythonNeumannVertices(vertex.geometry().corner(0));
    neumannVertices[indexSet.index(vertex)] = isNeumann;
  }

  BoundaryPatch<GridView> dirichletBoundary(gridView, dirichletVertices);
  BoundaryPatch<GridView> neumannBoundary(gridView, neumannVertices);

  if (mpiHelper.rank() == 0)
    std::cout << "Neumann boundary has " << neumannBoundary.numFaces() << " faces\n";

  BitSetVector<1> deformationDirichletNodes(midsurfaceFEBasis.size(), false);
  constructBoundaryDofs(dirichletBoundary, midsurfaceFEBasis, deformationDirichletNodes);

  BitSetVector<1> neumannNodes(midsurfaceFEBasis.size(), false);
  constructBoundaryDofs(neumannBoundary, directorFEBasis, neumannNodes);

  BitSetVector<3> deformationDirichletDofs(midsurfaceFEBasis.size(), false);
  for (size_t i = 0; i < midsurfaceFEBasis.size(); i++)
    if (deformationDirichletNodes[i][0])
      for (int j = 0; j < 3; j++)
        deformationDirichletDofs[i][j] = true;

  BitSetVector<1> orientationDirichletNodes(directorFEBasis.size(), false);
  constructBoundaryDofs(dirichletBoundary, directorFEBasis, orientationDirichletNodes);

  BitSetVector<2> orientationDirichletDofs(directorFEBasis.size(), false);
  for (size_t i = 0; i < directorFEBasis.size(); i++)
    if (orientationDirichletNodes[i][0])
      for (int j = 0; j < 2; j++)
        orientationDirichletDofs[i][j] = true;

  ////////////////////////////
  //   Initial iterate
  ////////////////////////////

  SolutionType x;

  x[_0].resize(midsurfaceFEBasis.size());

  lambda = std::string("lambda x: (") + parameterSet.get<std::string>("initialDeformation") + std::string(")");
  auto pythonInitialDeformation = Python::make_function<FieldVector<double, 3> >(Python::evaluate(lambda));

  auto deformationPowerBasis = makeBasis(gridView,power<3>(lagrange<midsurfaceOrder>()));

  std::vector<FieldVector<double, 3> > v;
  Functions::interpolate(deformationPowerBasis, v, pythonInitialDeformation);
  std::copy(v.begin(), v.end(), x[_0].begin());

  x[_1].resize(directorFEBasis.size());

  // The code currently assumes that the grid resides in the x-y plane
  // Therefore, all references directors point into z direction
  // If the reference is not planar one has to calculate the reference nodal directors.
  // E.g. for bilinear grid elements this can be just the average of the 4 element normals
  const FieldVector<double, 3> referenceDirectorVector({0, 0, 1});   //only plain reference!
  std::fill(x[_1].begin(), x[_1].end(), referenceDirectorVector);

  const SolutionType x0 = x;

  ////////////////////////////////////////////////////////
  //   Main homotopy loop
  ////////////////////////////////////////////////////////

  for (int loadStep = 0; loadStep < totalLoadSteps; loadStep++)
  {
    const double loadFactor = (loadStep + 1) * (1.0 / totalLoadSteps);
    if (mpiHelper.rank() == 0)
      std::cout << "Homotopy step: " << loadStep << ",    parameter: " << loadFactor << std::endl;

    //////////////////////////////////////////////////////////////
    //   Create an assembler for the energy functional
    //////////////////////////////////////////////////////////////

    const ParameterTree &materialParameters = parameterSet.sub("materialParameters");
    FieldVector<double, 3> neumannValues{0, 0, 0};
    if (parameterSet.hasKey("neumannValues"))
      neumannValues = parameterSet.get<FieldVector<double, 3> >("neumannValues");

    auto neumannFunction = [&](FieldVector<double, 2>) {
                             auto nV = neumannValues;
                             nV *= loadFactor;
                             return nV;
                           };

    // Output initial iterate (of homotopy loop)
    auto directorPowerBasis = makeBasis(gridView, power<3>(lagrange<directorOrder>()));

    BlockVector<FieldVector<double, 3> > displacementInitial(compositeBasis.size({0}));
    //calculates the global coordinates of midsurface displacements into displacementInitial
    std::transform(x[_0].begin(),x[_0].end(),x0[_0].begin(),displacementInitial.begin(),[](const auto& x_0Entry,const auto& x0_0Entry){
      return x_0Entry.globalCoordinates()-x0_0Entry.globalCoordinates();
    });

    //copies the global coordinates of the directors to directorInitial for a VTKWriter usable format
    BlockVector<FieldVector<double, 3> > directorInitial(compositeBasis.size({1}));
    std::transform(x[_1].begin(),x[_1].end(),directorInitial.begin(),[](const auto& x_1Entry){
      return x_1Entry.globalCoordinates();
    });

    auto displacementFunctionInitial = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double, 3> >(deformationPowerBasis,
                                                                                                                displacementInitial);
    auto directorFunctionInitial = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double, 3> >(directorPowerBasis,
                                                                                                            directorInitial);
    //  We need to subsample, because VTK cannot natively display real second-order functions
    SubsamplingVTKWriter<GridView> vtkWriter(gridView, Dune::refinementLevels(midsurfaceOrder - 1));
    vtkWriter.addVertexData(displacementFunctionInitial, VTK::FieldInfo("displacement", VTK::FieldInfo::Type::scalar, 3));
    vtkWriter.addVertexData(directorFunctionInitial, VTK::FieldInfo("director", VTK::FieldInfo::Type::scalar, 3));
    vtkWriter.write(resultPath + "simo-fox_homotopy_" + "_" + std::to_string(neumannValues[2]) + "_" + std::to_string(0));

    if (mpiHelper.rank() == 0) {
      std::cout << "Material parameters:" << std::endl;
      materialParameters.report();
    }

    // Assembler using ADOL-C
    Dune::GFE::SimoFoxEnergyLocalStiffness<decltype(compositeBasis), LocalFEFunction,adouble> simoFoxEnergyADOLCLocalStiffness(materialParameters,
                                                                                                    &neumannBoundary,
                                                                                                    neumannFunction,
                                                                                                    nullptr, x0);

    MixedLocalGFEADOLCStiffness<decltype(compositeBasis),
        RealTuple<double,3>,
        UnitVector<double,3> > localGFEADOLCStiffness(&simoFoxEnergyADOLCLocalStiffness);

    MixedGFEAssembler<decltype(compositeBasis),
        RealTuple<double,3>, UnitVector<double,3> > assembler(compositeBasis, &localGFEADOLCStiffness);
    ////////////////////////////////////////////////////////
    //   Set Dirichlet values
    ////////////////////////////////////////////////////////

    Python::Reference dirichletValuesClass = Python::import(parameterSet.get<std::string>("problem") + "-dirichlet-values");// + std::to_string(loadStep));

    Python::Callable C = dirichletValuesClass.get("DirichletValues");

    // Call a constructor.
    Python::Reference dirichletValuesPythonObject = C(loadFactor);

    // Extract object member functions as Dune functions
    auto deformationDirichletValues = Python::make_function<FieldVector<double, 3> >(dirichletValuesPythonObject.get("deformation"));

    BlockVector<FieldVector<double,3> > ddV(deformationPowerBasis.size());
    Functions::interpolate(deformationPowerBasis, ddV, deformationDirichletValues, deformationDirichletDofs);

    for (size_t j = 0; j < x[_0].size(); j++){
      if (deformationDirichletNodes[j][0]){
        x[_0][j] = ddV[j];
      }
    }

    // /////////////////////////////////////////////////
    //   Create a Riemannian trust-region solver
    // /////////////////////////////////////////////////
    if (parameterSet.get<std::string>("solvertype", "trustRegion") == "trustRegion") {

    MixedRiemannianTrustRegionSolver<Grid,
        decltype(compositeBasis),
        MidsurfaceFEBasis, RealTuple<double,3>,
        DirectorFEBasis, UnitVector<double,3> > solver;

    solver.setup(*grid,
                 &assembler,
                 midsurfaceFEBasis,
                 directorFEBasis,
                 x,
                 deformationDirichletDofs,
                 orientationDirichletDofs,
                 tolerance,
                 maxSolverSteps,
                 initialTrustRegionRadius,
                 multigridIterations,
                 mgTolerance,
                 mu, nu1, nu2,
                 baseIterations,
                 baseTolerance,
                 instrumented);

    solver.setScaling(parameterSet.get<FieldVector<double, 5> >("trustRegionScaling"));

    // /////////////////////////////////////////////////////
    //   Solve!
    // /////////////////////////////////////////////////////

    solver.setInitialIterate(x);
    solver.solve();

    x = solver.getSol();
    } else {
#if !MIXED_SPACE
      using TargetSpace = Dune::GFE::ProductManifold<RealTuple<double,3>,UnitVector<double,3>>;
      std::vector<TargetSpace> xTargetSpace(compositeBasis.size({0}));
      BitSetVector<TargetSpace::TangentVector::dimension> dirichletDofsTargetSpace(compositeBasis.size({0}), false);
      for (std::size_t i = 0; i < compositeBasis.size({0}); i++) {
        xTargetSpace[i][_0] = x[_0][i]; // Displacement part
        xTargetSpace[i][_1] = x[_1][i]; // Rotation part
        for (int j = 0; j < 3; j ++)
          dirichletDofsTargetSpace[i][j] = deformationDirichletDofs[i][j];
        for (int j = 3; j < TargetSpace::TangentVector::dimension; j ++)
          dirichletDofsTargetSpace[i][j] = orientationDirichletDofs[i][j-3];
      }
      using GFEAssemblerWrapper = Dune::GFE::GeodesicFEAssemblerWrapper<decltype(compositeBasis), MidsurfaceFEBasis, TargetSpace, RealTuple<double, 3>, UnitVector<double,3>>;
      GFEAssemblerWrapper assemblerNotMixed(&assembler, midsurfaceFEBasis);
      RiemannianProximalNewtonSolver<MidsurfaceFEBasis, TargetSpace, GFEAssemblerWrapper> solver;
      solver.setup(*grid,
                   &assemblerNotMixed,
                   xTargetSpace,
                   dirichletDofsTargetSpace,
                   tolerance,
                   maxSolverSteps,
                   initialRegularization,
                   instrumented);
      solver.setInitialIterate(xTargetSpace);
      solver.solve();
      xTargetSpace = solver.getSol();
      for (std::size_t i = 0; i < xTargetSpace.size(); i++) {
        x[_0][i] = xTargetSpace[i][_0];
        x[_1][i] = xTargetSpace[i][_1];
      }
#endif
    }
    // Output result of each load step
    std::stringstream iAsAscii;
    iAsAscii << loadStep + 1;

    //calculates the global coordinates of midsurface displacements into displacementInitial
    BlockVector<FieldVector<double, 3> > displacement(compositeBasis.size({0}));
    std::transform(x[_0].begin(),x[_0].end(),x0[_0].begin(),displacement.begin(),[](const auto& x_0Entry,const auto& x0_0Entry){
      return x_0Entry.globalCoordinates()-x0_0Entry.globalCoordinates();
    });

    //copies the global coordinates of the directors to directorInitial for a VTKWriter usable format
    BlockVector<FieldVector<double, 3> > director(compositeBasis.size({1}));
    std::transform(x[_1].begin(),x[_1].end(),director.begin(),[](const auto& x_1Entry){
      return x_1Entry.globalCoordinates();
    });

    auto displacementFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double, 3> >(deformationPowerBasis,
                                                                                                          displacement);
    auto directorFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double, 3> >(directorPowerBasis,
                                                                                                      director);
    //  We need to subsample, because VTK cannot natively display real second-order functions
    //  SubsamplingVTKWriter<GridView> vtkWriter(gridView, Dune::refinementLevels(midsurfaceOrder - 1));
    vtkWriter.addVertexData(displacementFunction, VTK::FieldInfo("displacement", VTK::FieldInfo::Type::scalar, 3));
    vtkWriter.addVertexData(directorFunction, VTK::FieldInfo("director", VTK::FieldInfo::Type::scalar, 3));
    vtkWriter.write(resultPath + "simo-fox_homotopy_" + "_" + std::to_string(neumannValues[2]) + "_" + std::to_string(loadStep + 1));
  }
}
catch (Exception &e) {
  std::cout << e.what() << std::endl;
}
