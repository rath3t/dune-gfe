#include <config.h>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/drivers/drivers.h>
#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <optional>

#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/common/version.hh>

#include <dune/grid/onedgrid.hh>

#include <dune/istl/io.hh>

#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include <dune/vtk/vtkwriter.hh>
#include <dune/vtk/datacollectors/lagrangedatacollector.hh>

#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/functiontools/boundarydofs.hh>

#include <dune/gfe/assemblers/cosseratrodenergy.hh>
#include <dune/gfe/assemblers/geodesicfeassembler.hh>
#include <dune/gfe/assemblers/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/localprojectedfefunction.hh>
#include <dune/gfe/riemanniantrsolver.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>

using namespace Dune;
using namespace Dune::Indices;

using TargetSpace = GFE::ProductManifold<RealTuple<double,3>,Rotation<double,3> >;

const int blocksize = TargetSpace::TangentVector::dimension;

// Approximation order of the finite element space
constexpr int order = 2;


int main (int argc, char *argv[]) try
{
  MPIHelper::instance(argc, argv);

  // parse data file
  ParameterTree parameterSet;
  if (argc < 2)
    DUNE_THROW(Exception, "Usage: ./rod3d <parameter file>");

  ParameterTreeParser::readINITree(argv[1], parameterSet);

  ParameterTreeParser::readOptions(argc, argv, parameterSet);

  // read solver settings
  const int numLevels        = parameterSet.get<int>("numLevels");
  const double tolerance        = parameterSet.get<double>("tolerance");
  const int maxTrustRegionSteps   = parameterSet.get<int>("maxTrustRegionSteps");
  const double initialTrustRegionRadius = parameterSet.get<double>("initialTrustRegionRadius");
  const int multigridIterations   = parameterSet.get<int>("numIt");
  const int nu1              = parameterSet.get<int>("nu1");
  const int nu2              = parameterSet.get<int>("nu2");
  const int mu               = parameterSet.get<int>("mu");
  const int baseIterations      = parameterSet.get<int>("baseIt");
  const double mgTolerance        = parameterSet.get<double>("mgTolerance");
  const double baseTolerance    = parameterSet.get<double>("baseTolerance");
  const bool instrumented      = parameterSet.get<bool>("instrumented");
  std::string resultPath           = parameterSet.get("resultPath", "");

  // read rod parameter settings
  const double A               = parameterSet.get<double>("A");
  const double J1              = parameterSet.get<double>("J1");
  const double J2              = parameterSet.get<double>("J2");
  const double E               = parameterSet.get<double>("E");
  const double nu              = parameterSet.get<double>("nu");
  const int numRodBaseElements = parameterSet.get<int>("numRodBaseElements");

  // ///////////////////////////////////////
  //    Create the grid
  // ///////////////////////////////////////
  typedef OneDGrid GridType;
  GridType grid(numRodBaseElements, 0, 1);

  grid.globalRefine(numLevels-1);

  using GridView = GridType::LeafGridView;
  GridView gridView = grid.leafGridView();

  //////////////////////////////////////////////
  //  Create the stress-free configuration
  //////////////////////////////////////////////

  using ScalarBasis = Functions::LagrangeBasis<GridView,order>;
  ScalarBasis scalarBasis(gridView);

  std::vector<double> referenceConfigurationX(scalarBasis.size());

  auto identity = [](const FieldVector<double,1>& x) {
                    return x;
                  };

  Functions::interpolate(scalarBasis, referenceConfigurationX, identity);

  using Configuration = std::vector<TargetSpace>;
  Configuration referenceConfiguration(scalarBasis.size());

  for (std::size_t i=0; i<referenceConfiguration.size(); i++)
  {
    referenceConfiguration[i][_0] = {0, 0, referenceConfigurationX[i]};
    referenceConfiguration[i][_1] = Rotation<double,3>::identity();
  }

  // Select the reference configuration as initial iterate

  Configuration x = referenceConfiguration;

  // /////////////////////////////////////////
  //   Read Dirichlet values
  // /////////////////////////////////////////

  // A basis for the tangent space
  using namespace Functions::BasisFactory;

  auto tangentBasis = makeBasis(
    gridView,
    power<TargetSpace::TangentVector::dimension>(
      lagrange<order>(),
      blockedInterleaved()
      ));

  // Find all boundary dofs
  BoundaryPatch<GridView> dirichletBoundary(gridView,
                                            true);      // true: The entire boundary is Dirichlet boundary
  BitSetVector<TargetSpace::TangentVector::dimension> dirichletNodes(tangentBasis.size(), false);
#if DUNE_VERSION_GTE(DUNE_FUFEM, 2, 10)
  Fufem::markBoundaryPatchDofs(dirichletBoundary,tangentBasis,dirichletNodes);
#else
  constructBoundaryDofs(dirichletBoundary,tangentBasis,dirichletNodes);
#endif

  // Find the dof on the right boundary
  std::size_t rightBoundaryDof;
  for (std::size_t i=0; i<referenceConfigurationX.size(); i++)
    if (std::fabs(referenceConfigurationX[i] - 1.0) < 1e-6)
    {
      rightBoundaryDof = i;
      break;
    }

  // Set Dirichlet values
  x[rightBoundaryDof][_0] = parameterSet.get<FieldVector<double,3> >("dirichletValue");

  auto axis = parameterSet.get<FieldVector<double,3> >("dirichletAxis");
  double angle = parameterSet.get<double>("dirichletAngle");

  x[rightBoundaryDof][_1] = Rotation<double,3>(axis, M_PI*angle/180);

  // backup for error measurement later
  std::cout << "Right boundary orientation:" << std::endl;
  std::cout << "director 0:  " << x[rightBoundaryDof][_1].director(0) << std::endl;
  std::cout << "director 1:  " << x[rightBoundaryDof][_1].director(1) << std::endl;
  std::cout << "director 2:  " << x[rightBoundaryDof][_1].director(2) << std::endl;

  //////////////////////////////////////////////
  //  Create the energy and assembler
  //////////////////////////////////////////////

  using ATargetSpace = TargetSpace::rebind<adouble>::other;
  using GeodesicInterpolationRule  = LocalGeodesicFEFunction<1, double, ScalarBasis::LocalView::Tree::FiniteElement, ATargetSpace>;
  using ProjectedInterpolationRule = GFE::LocalProjectedFEFunction<1, double, ScalarBasis::LocalView::Tree::FiniteElement, ATargetSpace>;

  // Assembler using ADOL-C
  std::shared_ptr<GFE::LocalEnergy<ScalarBasis,ATargetSpace> > localRodEnergy;

  if (parameterSet["interpolationMethod"] == "geodesic")
  {
    auto energy = std::make_shared<GFE::CosseratRodEnergy<ScalarBasis, GeodesicInterpolationRule, adouble> >(gridView,
                                                                                                             A, J1, J2, E, nu);
    energy->setReferenceConfiguration(referenceConfiguration);
    localRodEnergy = energy;
  }
  else if (parameterSet["interpolationMethod"] == "projected")
  {
    auto energy = std::make_shared<GFE::CosseratRodEnergy<ScalarBasis, ProjectedInterpolationRule, adouble> >(gridView,
                                                                                                              A, J1, J2, E, nu);
    energy->setReferenceConfiguration(referenceConfiguration);
    localRodEnergy = energy;
  }
  else
    DUNE_THROW(Exception, "Unknown interpolation method " << parameterSet["interpolationMethod"] << " requested!");

  LocalGeodesicFEADOLCStiffness<ScalarBasis,
      TargetSpace> localStiffness(localRodEnergy);

  GeodesicFEAssembler<ScalarBasis,TargetSpace> rodAssembler(gridView, localStiffness);

  /////////////////////////////////////////////
  //   Create a solver for the rod problem
  /////////////////////////////////////////////

  RiemannianTrustRegionSolver<ScalarBasis,TargetSpace> rodSolver;

  rodSolver.setup(grid,
                  &rodAssembler,
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

  std::cout << "Energy: " << rodAssembler.computeEnergy(x) << std::endl;

  rodSolver.setInitialIterate(x);
  rodSolver.solve();

  x = rodSolver.getSol();

  // //////////////////////////////
  //   Output result
  // //////////////////////////////

#if DUNE_VERSION_GTE(DUNE_VTK, 2, 10)
  auto vtkWriter = Vtk::UnstructuredGridWriter(Vtk::LagrangeDataCollector(gridView,order));
#else
  using DataCollector = Vtk::LagrangeDataCollector<GridView,order>;
  DataCollector dataCollector(gridView);
  VtkUnstructuredGridWriter<GridView,DataCollector> vtkWriter(gridView, Vtk::FormatTypes::ASCII);
#endif

  // Make basis for R^3-valued data
  using namespace Functions::BasisFactory;

  auto worldBasis = makeBasis(
    gridView,
    power<3>(lagrange<order>())
    );

  // The rod displacement field
  BlockVector<FieldVector<double,3> > displacement(worldBasis.size());
  for (std::size_t i=0; i<x.size(); i++)
    displacement[i] = x[i][_0].globalCoordinates();

  std::vector<double> xEmbedding;
  Functions::interpolate(scalarBasis, xEmbedding, [](FieldVector<double,1> x){
    return x;
  });

  BlockVector<FieldVector<double,3> > gridEmbedding(xEmbedding.size());
  for (std::size_t i=0; i<gridEmbedding.size(); i++)
    gridEmbedding[i] = {xEmbedding[i], 0, 0};

  displacement -= gridEmbedding;

  auto displacementFunction = Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,3> >(worldBasis, displacement);
  vtkWriter.addPointData(displacementFunction, "displacement", 3);

  // The three director fields
  using FunctionType = decltype(displacementFunction);
  std::array<std::optional<FunctionType>, 3> directorFunction;
  std::array<BlockVector<FieldVector<double, 3> >, 3> director;
  for (int i=0; i<3; i++)
  {
    director[i].resize(worldBasis.size());
    for (std::size_t j=0; j<x.size(); j++)
      director[i][j] = x[j][_1].director(i);

    directorFunction[i] = Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,3> >(worldBasis, std::move(director[i]));
    vtkWriter.addPointData(*directorFunction[i], "director " + std::to_string(i), 3);
  }

  vtkWriter.write(resultPath + "rod3d-result");

}
catch (Exception& e)
{
  std::cout << e.what() << std::endl;
}
