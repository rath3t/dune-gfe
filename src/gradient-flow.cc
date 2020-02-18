// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include <config.h>

#include <array>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/typetraits.hh>
#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/common/version.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/onedgrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/io/file/vtk.hh>

#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>

#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/functiontools/basisinterpolator.hh>
#include <dune/fufem/functiontools/boundarydofs.hh>
#include <dune/fufem/functionspacebases/dunefunctionsbasis.hh>
#include <dune/fufem/discretizationerror.hh>
#include <dune/fufem/dunepython.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include <dune/gfe/unitvector.hh>
#include <dune/gfe/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/geodesicfeassembler.hh>
#include <dune/gfe/riemanniantrsolver.hh>
#include <dune/gfe/globalgeodesicfefunction.hh>
#include <dune/gfe/embeddedglobalgfefunction.hh>
#include <dune/gfe/harmonicenergy.hh>
#include <dune/gfe/l2distancesquaredenergy.hh>
#include <dune/gfe/weightedsumenergy.hh>
#include <dune/gfe/periodic1dpq1nodalbasis.hh>

// grid dimension
const int dim = 1;
const int dimworld = dim;

// Image space of the geodesic fe functions
// typedef Rotation<double,2> TargetSpace;
// typedef Rotation<double,3> TargetSpace;
// typedef UnitVector<double,2> TargetSpace;
typedef UnitVector<double,3> TargetSpace;
// typedef UnitVector<double,4> TargetSpace;
// typedef RealTuple<double,1> TargetSpace;

// Tangent vector of the image space
const int blocksize = TargetSpace::TangentVector::dimension;

// Approximation order of the finite element space
const int order = 1;

using namespace Dune;


int main (int argc, char *argv[]) try
{
  // initialize MPI; this is needed even though we may never use it
  MPIHelper::instance(argc, argv);

  // Start Python interpreter
  Python::start();
  Python::Reference main = Python::import("__main__");
  Python::run("import math");

  //feenableexcept(FE_INVALID);
  Python::runStream()
      << std::endl << "import sys"
      << std::endl << "sys.path.append('../../problems/')"
      << std::endl;

  typedef std::vector<TargetSpace> SolutionType;

  // parse data file
  ParameterTree parameterSet;
  if (argc < 2)
    DUNE_THROW(Exception, "Usage: ./gradient-flow <parameter file>");

  ParameterTreeParser::readINITree(argv[1], parameterSet);
  ParameterTreeParser::readOptions(argc, argv, parameterSet);

  // read problem settings
  const int numLevels                   = parameterSet.get<int>("numLevels");
  const double timeStepSize             = parameterSet.get<double>("timeStepSize");
  const int numTimeSteps                = parameterSet.get<int>("numTimeSteps");

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

  /////////////////////////////////////////
  //    Create the grid
  /////////////////////////////////////////
  typedef std::conditional<dim==1,OneDGrid,UGGrid<dim> >::type GridType;

  std::shared_ptr<GridType> grid;
  FieldVector<double,dimworld> lower(0), upper(1);
  std::array<unsigned int,dim> elements;

  if (parameterSet.get<bool>("structuredGrid"))
  {
    lower = parameterSet.get<FieldVector<double,dimworld> >("lower");
    upper = parameterSet.get<FieldVector<double,dimworld> >("upper");

    elements = parameterSet.get<std::array<unsigned int,dim> >("elements");
    grid = StructuredGridFactory<GridType>::createCubeGrid(lower, upper, elements);
  }
  else
  {
    std::string path     = parameterSet.get<std::string>("path");
    std::string gridFile = parameterSet.get<std::string>("gridFile");

    grid = std::shared_ptr<GridType>(GmshReader<GridType>::read(path + "/" + gridFile));
  }

  grid->globalRefine(numLevels-1);

  //////////////////////////////////////////////////////////////////////////////////
  //  Construct the scalar function space basis corresponding to the GFE space
  //////////////////////////////////////////////////////////////////////////////////

  typedef Dune::Functions::LagrangeBasis<typename GridType::LeafGridView, order> FEBasis;
  //typedef Dune::Functions::Periodic1DPQ1NodalBasis<typename GridType::LeafGridView> FEBasis;
  FEBasis feBasis(grid->leafGridView());

  ///////////////////////////////////////////
  //   Read Dirichlet values
  ///////////////////////////////////////////

  typedef DuneFunctionsBasis<FEBasis> FufemFEBasis;
  FufemFEBasis fufemFeBasis(feBasis);

  BitSetVector<1> dirichletVertices(grid->leafGridView().size(dim), false);

  const auto& indexSet = grid->leafGridView().indexSet();

  // Make Python function that computes which vertices are on the Dirichlet boundary,
  // based on the vertex positions.
  std::string lambda = std::string("lambda x: (") + parameterSet.get<std::string>("dirichletVerticesPredicate") + std::string(")");
  PythonFunction<FieldVector<double,dimworld>, bool> pythonDirichletVertices(Python::evaluate(lambda));

  for (auto&& vertex : vertices(grid->leafGridView()))
  {
    bool isDirichlet = pythonDirichletVertices(vertex.geometry().corner(0));
    dirichletVertices[indexSet.index(vertex)] = isDirichlet;
  }

  BoundaryPatch<GridType::LeafGridView> dirichletBoundary(grid->leafGridView(), dirichletVertices);

  BitSetVector<blocksize> dirichletNodes(feBasis.size(), false);
  constructBoundaryDofs(dirichletBoundary,fufemFeBasis,dirichletNodes);

  ////////////////////////////
  //   Initial iterate
  ////////////////////////////

  // Read initial iterate into a PythonFunction
  typedef PythonFunction<FieldVector<double, dimworld>, TargetSpace::CoordinateType> FBase;

  Python::Module module = Python::import(parameterSet.get<std::string>("initialIterate"));
  auto pythonInitialIterate = module.get("f").toC<std::shared_ptr<FBase>>();

  std::vector<TargetSpace::CoordinateType> v;
  ::Functions::interpolate(fufemFeBasis, v, *pythonInitialIterate);

  SolutionType x(feBasis.size());

  for (size_t i=0; i<x.size(); i++)
    x[i] = v[i];

  //////////////////////////////////////////////////////////////
  //   Create an assembler for the harmonic energy functional
  //////////////////////////////////////////////////////////////

  // Assembler using ADOL-C
  typedef TargetSpace::rebind<adouble>::other ATargetSpace;

  auto l2DistanceSquaredEnergy = std::make_shared<L2DistanceSquaredEnergy<FEBasis, ATargetSpace> >();

  std::vector<std::shared_ptr<GFE::LocalEnergy<FEBasis,ATargetSpace> > > addends(2);
  using GeodesicInterpolationRule  = LocalGeodesicFEFunction<dim, double, FEBasis::LocalView::Tree::FiniteElement, ATargetSpace>;
  addends[0] = std::make_shared<HarmonicEnergy<FEBasis, GeodesicInterpolationRule, ATargetSpace> >();
  addends[1] = l2DistanceSquaredEnergy;

  std::vector<double> weights = {1.0, 1.0/(2*timeStepSize)};

  auto sumEnergy = std::make_shared< WeightedSumEnergy<FEBasis, ATargetSpace> >(addends, weights);

  LocalGeodesicFEADOLCStiffness<FEBasis,TargetSpace> localGFEADOLCStiffness(sumEnergy.get());

  GeodesicFEAssembler<FEBasis,TargetSpace> assembler(feBasis, &localGFEADOLCStiffness);

  ///////////////////////////////////////////////////
  //   Create a Riemannian trust-region solver
  ///////////////////////////////////////////////////

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
               false);   // instrumentation

  ///////////////////////////////////////////////////////
  //   Write initial iterate
  ///////////////////////////////////////////////////////

  typedef BlockVector<TargetSpace::CoordinateType> EmbeddedVectorType;
  EmbeddedVectorType xEmbedded(x.size());
  for (size_t i=0; i<x.size(); i++)
    xEmbedded[i] = x[i].globalCoordinates();

  auto xFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<TargetSpace::CoordinateType>(feBasis,
                                                                                                 TypeTree::hybridTreePath(),
                                                                                                 xEmbedded);

  SubsamplingVTKWriter<GridType::LeafGridView> vtkWriter(grid->leafGridView(),refinementLevels(0));
  vtkWriter.addVertexData(xFunction, VTK::FieldInfo("orientation", VTK::FieldInfo::Type::scalar, xEmbedded[0].size()));
  vtkWriter.write("gradientflow_result_0");

  // Write the corresponding coefficient vector: verbatim in binary, to be completely lossless
  std::ofstream outFile("gradientflow_result_0.data", std::ios_base::binary);
  MatrixVector::Generic::writeBinary(outFile, xEmbedded);
  outFile.close();

  ///////////////////////////////////////////////////////
  //   Time loop
  ///////////////////////////////////////////////////////

  auto previousTimeStep = x;

  for (int i=0; i<numTimeSteps; i++)
  {
    auto previousTimeStepFct = std::make_shared<GlobalGeodesicFEFunction<FufemFEBasis,TargetSpace> >(fufemFeBasis,previousTimeStep);
    l2DistanceSquaredEnergy->origin_ = previousTimeStepFct;

    solver.setInitialIterate(x);
    solver.solve();

    x = solver.getSol();

    previousTimeStep = x;

    ////////////////////////////////
    //   Output result
    ////////////////////////////////

    typedef BlockVector<TargetSpace::CoordinateType> EmbeddedVectorType;
    EmbeddedVectorType xEmbedded(x.size());
    for (size_t i=0; i<x.size(); i++)
      xEmbedded[i] = x[i].globalCoordinates();

    auto xFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<TargetSpace::CoordinateType>(feBasis,
                                                                                                   TypeTree::hybridTreePath(),
                                                                                                   xEmbedded);

    SubsamplingVTKWriter<GridType::LeafGridView> vtkWriter(grid->leafGridView(),refinementLevels(0));
    vtkWriter.addVertexData(xFunction, VTK::FieldInfo("orientation", VTK::FieldInfo::Type::scalar, xEmbedded[0].size()));
    vtkWriter.write("gradientflow_result_" + std::to_string(i+1));

    // Write the corresponding coefficient vector: verbatim in binary, to be completely lossless
    std::ofstream outFile("gradientflow_result_" + std::to_string(i+1) + ".data", std::ios_base::binary);
    MatrixVector::Generic::writeBinary(outFile, xEmbedded);
    outFile.close();

  }

  return 0;
}
catch (Exception& e)
{
  std::cout << e.what() << std::endl;
  return 1;
}
