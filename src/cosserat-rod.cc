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

#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include <dune/vtk/vtkwriter.hh>
#include <dune/vtk/datacollectors/lagrangedatacollector.hh>

#include <dune/fufem/dunepython.hh>

#include <dune/gfe/assemblers/cosseratrodenergy.hh>
#include <dune/gfe/assemblers/geodesicfeassembler.hh>
#include <dune/gfe/assemblers/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/cosseratvtkwriter.hh>
#include <dune/gfe/functions/embeddedglobalgfefunction.hh>
#include <dune/gfe/functions/localgeodesicfefunction.hh>
#include <dune/gfe/functions/localprojectedfefunction.hh>
#include <dune/gfe/riemanniantrsolver.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>

using namespace Dune;
using namespace Dune::Indices;

using TargetSpace = GFE::ProductManifold<GFE::RealTuple<double,3>,GFE::Rotation<double,3> >;

const int blocksize = TargetSpace::TangentVector::dimension;

// Approximation order of the finite element space
constexpr int order = 2;


int main (int argc, char *argv[]) try
{
  MPIHelper::instance(argc, argv);

  // Check for appropriate number of command line arguments
  if (argc < 3)
    DUNE_THROW(Exception, "Usage: ./cosserat-rod <python path> <parameter file>");

  // Start Python interpreter
  Python::start();
  auto pyMain = Python::main();

  Python::runStream()
    << std::endl << "import sys"
    << std::endl << "sys.path.append('" << argv[1] << "')"
    << std::endl;

  // Parse data file
  auto pyModule = pyMain.import(argv[2]);

  // Get main parameter set
  ParameterTree parameterSet;
  pyModule.get("parameterSet").toC(parameterSet);

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

  using namespace Dune::Functions::BasisFactory;

  using ScalarBasis = Functions::LagrangeBasis<GridView,order>;
  ScalarBasis scalarBasis(gridView);

  auto deformationPowerBasis = makeBasis(
    gridView,
    power<3>(
      lagrange<order>()
      ));

  // Matrix-valued basis for treating the microrotation as a matrix field
  auto orientationMatrixBasis = makeBasis(
    gridView,
    power<3>(
      power<3>(
        lagrange<order>()
        )
      ));


  using Configuration = std::vector<TargetSpace>;
  Configuration referenceConfiguration(scalarBasis.size());

  // Load the stress-free configuration from the Python options file
  Python::Callable referenceConfigurationPythonClass = pyModule.get("ReferenceConfiguration");
  Python::Reference referenceConfigurationPythonObject = referenceConfigurationPythonClass();

  // Extract object member functions as Dune functions
  auto referenceDeformationFunction = Python::make_function<FieldVector<double,3> >   (referenceConfigurationPythonObject.get("deformation"));
  auto referenceOrientationFunction = Python::make_function<FieldMatrix<double,3,3> > (referenceConfigurationPythonObject.get("orientation"));

  BlockVector<FieldVector<double,3> > ddV;
  Functions::interpolate(deformationPowerBasis, ddV, referenceDeformationFunction);

  BlockVector<FieldMatrix<double,3,3> > dOV;
  Functions::interpolate(orientationMatrixBasis, dOV, referenceOrientationFunction);

  for (std::size_t i = 0; i < deformationPowerBasis.size(); i++)
    referenceConfiguration[i][_0] = ddV[i];

  for (std::size_t i = 0; i < orientationMatrixBasis.size(); i++)
    referenceConfiguration[i][_1].set(dOV[i]);


  ///////////////////////////////////////////
  //   Read Dirichlet values
  ///////////////////////////////////////////

  // A basis for the tangent space
  auto tangentBasis = makeBasis(
    gridView,
    power<TargetSpace::TangentVector::dimension>(
      lagrange<order>(),
      blockedInterleaved()
      ));

  BitSetVector<TargetSpace::TangentVector::dimension> dirichletNodes(tangentBasis.size(), false);

  // Make Python function that computes which vertices are on the Dirichlet boundary, based on the vertex positions.
  std::string lambda = std::string("lambda x: (") + parameterSet.get<std::string>("dirichletVerticesPredicate") + std::string(")");
  auto pythonDirichletVertices = Python::make_function<FieldVector<bool,3> >(Python::evaluate(lambda));

  lambda = std::string("lambda x: (") + parameterSet.get<std::string>("dirichletRotationVerticesPredicate") + std::string(")");
  auto pythonOrientationDirichletVertices = Python::make_function<bool>(Python::evaluate(lambda));

  for (size_t i=0; i<tangentBasis.size(); i++)
  {
    FieldVector<bool,3> isDirichlet = pythonDirichletVertices(referenceConfiguration[i][_0].globalCoordinates());
    for (size_t j=0; j<3; j++)
      dirichletNodes[i][j] = isDirichlet[j];

    bool isDirichletOrientation = pythonOrientationDirichletVertices(referenceConfiguration[i][_0].globalCoordinates());
    for (size_t j=0; j<3; j++)
      dirichletNodes[i][j+3] = isDirichletOrientation;
  }

  std::cout << "Dirichlet boundary has " << dirichletNodes.count() << " degrees of freedom.\n";

  Configuration dirichletValues(scalarBasis.size());

  // Load the stress-free configuration from the Python options file
  Python::Callable initialConfigurationPythonClass = pyModule.get("DirichletValues");
  Python::Reference initialConfigurationPythonObject = initialConfigurationPythonClass();

  // Extract object member functions as Dune functions
  auto initialDeformationFunction = Python::make_function<FieldVector<double,3> >   (initialConfigurationPythonObject.get("deformation"));
  auto initialOrientationFunction = Python::make_function<FieldMatrix<double,3,3> > (initialConfigurationPythonObject.get("orientation"));

  Functions::interpolate(deformationPowerBasis, ddV, initialDeformationFunction);
  Functions::interpolate(orientationMatrixBasis, dOV, initialOrientationFunction);

  for (std::size_t i = 0; i < deformationPowerBasis.size(); i++)
    dirichletValues[i][_0] = ddV[i];

  for (std::size_t i = 0; i < orientationMatrixBasis.size(); i++)
    dirichletValues[i][_1].set(dOV[i]);

  // Select the Dirichlet value function as initial iterate
  Configuration x = dirichletValues;

  //////////////////////////////////////////////
  //  Create the energy and assembler
  //////////////////////////////////////////////

  using ATargetSpace = TargetSpace::rebind<adouble>::other;
  using GeodesicInterpolationRule  = GFE::LocalGeodesicFEFunction<ScalarBasis, ATargetSpace>;
  using ProjectedInterpolationRule = GFE::LocalProjectedFEFunction<ScalarBasis, ATargetSpace>;

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

  GFE::LocalGeodesicFEADOLCStiffness<ScalarBasis,
      TargetSpace> localStiffness(localRodEnergy);

  GFE::GeodesicFEAssembler<ScalarBasis,TargetSpace> rodAssembler(gridView, localStiffness);

  /////////////////////////////////////////////
  //   Create a solver for the rod problem
  /////////////////////////////////////////////

  GFE::RiemannianTrustRegionSolver<ScalarBasis,TargetSpace> rodSolver;

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

  // Make basis for R^3-valued data
  auto worldBasis = makeBasis(
    gridView,
    power<3>(lagrange<order>())
    );

  // Compute the displacement from the deformation, because that's more easily visualized
  // in ParaView
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

  // Copy the orientation part of the configuration; the CosseratVTKWriter wants it that way
  std::vector<GFE::Rotation<double,3> > orientationConfiguration(x.size());
  for (size_t i=0; i<x.size(); ++i)
    orientationConfiguration[i] = x[i][_1];

  using RotationInterpolationRule = GFE::LocalGeodesicFEFunction<ScalarBasis, GFE::Rotation<double,3> >;

  GFE::EmbeddedGlobalGFEFunction<ScalarBasis, RotationInterpolationRule,GFE::Rotation<double,3> > orientationFunction(scalarBasis,
                                                                                                                      orientationConfiguration);

  GFE::CosseratVTKWriter<GridView>::write(gridView,
                                          displacementFunction,
                                          orientationFunction,
                                          order,
                                          resultPath + "cosserat-rod-result-" + std::to_string(numLevels));

  // Write the corresponding coefficient vector: verbatim in binary, to be completely lossless
  // This data may be used by other applications measuring the discretization error
  BlockVector<TargetSpace::CoordinateType> xEmbedded(x.size());
  for (size_t i=0; i<x.size(); i++)
    xEmbedded[i] = x[i].globalCoordinates();

  std::ofstream outFile("cosserat-rod-result-" + std::to_string(numLevels) + ".data", std::ios_base::binary);
  MatrixVector::Generic::writeBinary(outFile, xEmbedded);
  outFile.close();
}
catch (Exception& e)
{
  std::cout << e.what() << std::endl;
}
