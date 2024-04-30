#include <config.h>

#include <array>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/parametertree.hh>
#include <dune/common/typetraits.hh>
#include <dune/common/bitsetvector.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/compositebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/subspacebasis.hh>

#include <dune/fufem/boundarypatch.hh>

#include <dune/gfe/assemblers/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/assemblers/mixedgfeassembler.hh>
#include <dune/gfe/assemblers/sumenergy.hh>
#include <dune/gfe/assemblers/localintegralenergy.hh>
#include <dune/gfe/densities/bulkcosseratdensity.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/mixedriemanniantrsolver.hh>
#include <dune/gfe/neumannenergy.hh>

// grid dimension
const int dim = 3;

// Order of the approximation space for the displacement
const int displacementOrder = 2;

// Order of the approximation space for the microrotations
const int rotationOrder = 1;

using namespace Dune;

int main (int argc, char *argv[])
{

  MPIHelper::instance(argc, argv);

  using SolutionType = TupleVector<std::vector<RealTuple<double,dim> >, std::vector<Rotation<double,dim> > >;

  // solver settings
  const double tolerance                = 1e-6;
  const int maxSolverSteps              = 1000;
  const double initialTrustRegionRadius = 1;
  const int multigridIterations         = 200;
  const int baseIterations              = 100;
  const double mgTolerance              = 1e-10;
  const double baseTolerance            = 1e-8;

  // ///////////////////////////////////////
  //    Create the grid
  // ///////////////////////////////////////
  using GridType = UGGrid<dim>;

  //Create a grid with 2 elements, one element will get refined to create different element types, the other one stays a cube
  std::shared_ptr<GridType> grid = StructuredGridFactory<GridType>::createCubeGrid({0,0,0}, {1,1,1}, {2,1,1});

  // Refine once
  for (auto&& e : elements(grid->leafGridView())) {
    bool refineThisElement = false;
    for (int i = 0; i < e.geometry().corners(); i++) {
      refineThisElement = refineThisElement || (e.geometry().corner(i)[0] > 0.99);
    }
    grid->mark(refineThisElement ? 1 : 0, e);
  }
  grid->adapt();
  grid->loadBalance();

  using GridView = GridType::LeafGridView;
  GridView gridView = grid->leafGridView();

  // ///////////////////////////////////////////
  //  Construct all needed function space bases
  // ///////////////////////////////////////////
  using namespace TypeTree::Indices;
  using namespace Functions::BasisFactory;

  const int dimRotation = Rotation<double,dim>::embeddedDim;
  auto compositeBasis = makeBasis(
    gridView,
    composite(
      power<dim>(
        lagrange<displacementOrder>()
        ),
      power<dimRotation>(
        lagrange<rotationOrder>()
        )
      ));
  using CompositeBasis = decltype(compositeBasis);

  using DeformationFEBasis = Functions::LagrangeBasis<GridView,displacementOrder>;
  using OrientationFEBasis = Functions::LagrangeBasis<GridView,rotationOrder>;

  DeformationFEBasis deformationFEBasis(gridView);
  OrientationFEBasis orientationFEBasis(gridView);

  // ////////////////////////////////////////////
  //  Determine Dirichlet dofs
  // ////////////////////////////////////////////
  // This identityBasis is only needed to interpolate the identity for both the displacement and the rotation
  auto identityBasis = makeBasis(
    gridView,
    composite(
      power<dim>(
        lagrange<displacementOrder>()
        ),
      power<dim>(
        lagrange<rotationOrder>()
        )
      ));
  auto deformationPowerBasis = Functions::subspaceBasis(identityBasis, _0);
  auto rotationPowerBasis = Functions::subspaceBasis(identityBasis, _1);

  MultiTypeBlockVector<BlockVector<FieldVector<double,dim> >,BlockVector<FieldVector<double,dim> > > identity;
  Functions::interpolate(deformationPowerBasis, identity, [&](FieldVector<double,dim> x){
    return x;
  });
  Functions::interpolate(rotationPowerBasis, identity, [&](FieldVector<double,dim> x){
    return x;
  });

  BitSetVector<dim> deformationDirichletDofs(deformationFEBasis.size(), false);
  BitSetVector<dim> orientationDirichletDofs(orientationFEBasis.size(), false);

  const GridView::IndexSet& indexSet = gridView.indexSet();

  // Make predicate function that computes which vertices are on the Dirichlet boundary, based on the vertex positions.
  auto isDirichlet = [](FieldVector<double,dim> coordinate)
                     {
                       return coordinate[0] < 0.01;
                     };

  for (size_t i=0; i<deformationFEBasis.size(); i++) {
    bool isDirichletDeformation = isDirichlet(identity[_0][i]);
    for (size_t j=0; j<dim; j++)
      deformationDirichletDofs[i][j] = isDirichletDeformation;
  }
  for (size_t i=0; i<orientationFEBasis.size(); i++) {
    bool isDirichletOrientation = isDirichlet(identity[_1][i]);
    for (size_t j=0; j<dim; j++)
      orientationDirichletDofs[i][j] = isDirichletOrientation;
  }

  // /////////////////////////////////////////
  //  Determine Neumann dofs and values
  // /////////////////////////////////////////

  std::function<bool(FieldVector<double,dim>)> isNeumann = [](FieldVector<double,dim> coordinate) {
                                                             return coordinate[0] > 0.99;
                                                           };

  BitSetVector<1> neumannVertices(gridView.size(dim), false);

  for (auto&& vertex : vertices(gridView))
    neumannVertices[indexSet.index(vertex)] = isNeumann(vertex.geometry().corner(0));

  auto neumannBoundary = std::make_shared<BoundaryPatch<GridType::LeafGridView> >(gridView, neumannVertices);

  FieldVector<double,dim> values_ = {0,0,0};
  auto neumannFunction = [&](FieldVector<double, dim>){
                           return values_;
                         };

  // //////////////////////////
  //  Initial iterate
  // //////////////////////////

  SolutionType x;

  x[_0].resize(compositeBasis.size({0}));
  x[_1].resize(compositeBasis.size({1}));

  // ////////////////////////////
  //  Parameters for the problem
  // ////////////////////////////

  ParameterTree parameters;
  parameters["mu"] = "1";
  parameters["lambda"] = "1";
  parameters["mu_c"] = "1";
  parameters["L_c"] = "0.1";
  parameters["q"] = "2";
  parameters["b1"] = "1";
  parameters["b2"] = "1";
  parameters["b3"] = "1";

  // ////////////////////////////
  //  Create an assembler
  // ////////////////////////////

  using RigidBodyMotion = GFE::ProductManifold<RealTuple<double,dim>,Rotation<double,dim> >;
  using ARigidBodyMotion = GFE::ProductManifold<RealTuple<adouble,dim>,Rotation<adouble,dim> >;

  GFE::SumEnergy<CompositeBasis, RealTuple<adouble,dim>,Rotation<adouble,dim> > sumEnergy;
  auto neumannEnergy = std::make_shared<GFE::NeumannEnergy<CompositeBasis, RealTuple<adouble,dim>, Rotation<adouble,dim> > >(neumannBoundary,neumannFunction);

  // Select which type of geometric interpolation to use
  using LocalDeformationInterpolationRule = LocalGeodesicFEFunction<dim, GridType::ctype, decltype(deformationFEBasis.localView().tree().finiteElement()), RealTuple<adouble,dim> >;
  using LocalOrientationInterpolationRule = LocalGeodesicFEFunction<dim, GridType::ctype, decltype(orientationFEBasis.localView().tree().finiteElement()), Rotation<adouble,dim> >;

  using LocalInterpolationRule = std::tuple<LocalDeformationInterpolationRule,LocalOrientationInterpolationRule>;

  auto bulkCosseratDensity = std::make_shared<GFE::BulkCosseratDensity<FieldVector<double,dim>,adouble> >(parameters);
  auto bulkCosseratEnergy = std::make_shared<GFE::LocalIntegralEnergy<CompositeBasis, LocalInterpolationRule, ARigidBodyMotion> >(bulkCosseratDensity);
  sumEnergy.addLocalEnergy(bulkCosseratEnergy);
  sumEnergy.addLocalEnergy(neumannEnergy);

  LocalGeodesicFEADOLCStiffness<CompositeBasis,RigidBodyMotion> localGFEADOLCStiffness(&sumEnergy);
  MixedGFEAssembler<CompositeBasis,
      RealTuple<double,dim>,
      Rotation<double,dim> > mixedAssembler(compositeBasis, localGFEADOLCStiffness);

  MixedRiemannianTrustRegionSolver<GridType,
      CompositeBasis,
      DeformationFEBasis, RealTuple<double,dim>,
      OrientationFEBasis, Rotation<double,dim> > solver;
  solver.setup(*grid,
               &mixedAssembler,
               deformationFEBasis,
               orientationFEBasis,
               x,
               deformationDirichletDofs,
               orientationDirichletDofs,
               tolerance,
               maxSolverSteps,
               initialTrustRegionRadius,
               multigridIterations,
               mgTolerance,
               3, 3, 1, // Multigrid V-cycle
               baseIterations,
               baseTolerance,
               false);

  solver.setInitialIterate(x);
  solver.solve();

  // //////////////////////////////////////////////
  //  Check if solver returns the expected values
  // //////////////////////////////////////////////

  size_t expectedFinalIteration = 14;
  double expectedEnergy = 0.67188585;

  if (solver.getStatistics().finalIteration != expectedFinalIteration)
  {
    std::cerr << "Trust-region solver did " << solver.getStatistics().finalIteration+1
              << " iterations, instead of the expected '" << expectedFinalIteration+1 << "'!" << std::endl;
    return 1;
  }

  if ( std::abs(solver.getStatistics().finalEnergy - expectedEnergy) > 1e-7)
  {
    std::cerr << std::setprecision(9);
    std::cerr << "Final energy is " << solver.getStatistics().finalEnergy
              << " but '" << expectedEnergy << "' was expected!" << std::endl;
    return 1;
  }

  return 0;
}
