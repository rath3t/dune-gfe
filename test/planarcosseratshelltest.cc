#include <config.h>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/parametertree.hh>
#include <dune/common/bitsetvector.hh>

#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/uggrid.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/compositebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/subspacebasis.hh>

#include <dune/fufem/boundarypatch.hh>

#include <dune/gfe/assemblers/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/assemblers/mixedgfeassembler.hh>
#include <dune/gfe/assemblers/sumenergy.hh>
#include <dune/gfe/assemblers/cosseratenergystiffness.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/mixedriemanniantrsolver.hh>
#include <dune/gfe/neumannenergy.hh>

// Dimension of the world space
const int dimworld = 3;

// Order of the approximation space for the displacement
const int displacementOrder = 2;

// Order of the approximation space for the microrotations
const int rotationOrder = 1;

using namespace Dune;

int main (int argc, char *argv[])
{
  MPIHelper::instance(argc, argv);

  using Configuration = TupleVector<std::vector<RealTuple<double,3> >, std::vector<Rotation<double,3> > >;

  // solver settings
  const double tolerance                = 1e-4;
  const int maxSolverSteps              = 20;
  const double initialTrustRegionRadius = 3.125;
  const int multigridIterations         = 100;
  const int baseIterations              = 100;
  const double mgTolerance              = 1e-10;
  const double baseTolerance            = 1e-8;

  /////////////////////////////////////////
  //   Create the grid
  /////////////////////////////////////////

  const int dim = 2;
  using GridType = UGGrid<dim>;

  const std::string path = std::string(DUNE_GRID_EXAMPLE_GRIDS_PATH) + "gmsh/";
  auto grid = GmshReader<GridType>::read(path + "hybrid-testgrid-2d.msh");

  //grid->globalRefine(1);

  using GridView = GridType::LeafGridView;
  GridView gridView = grid->leafGridView();

  // ///////////////////////////////////////////
  //  Construct all needed function space bases
  // ///////////////////////////////////////////
  using namespace Dune::Indices;
  using namespace Functions::BasisFactory;

  const int dimRotation = Rotation<double,dim>::embeddedDim;
  auto compositeBasis = makeBasis(
    gridView,
    composite(
      power<dimworld>(
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

  //////////////////////////////////////////////
  //  Determine Dirichlet dofs
  //////////////////////////////////////////////

  // This identityBasis is only needed to compute the positions of the Lagrange points
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

  BitSetVector<RealTuple<double,dimworld>::TangentVector::dimension> deformationDirichletDofs(deformationFEBasis.size(), false);
  BitSetVector<Rotation<double,dimworld>::TangentVector::dimension> orientationDirichletDofs(orientationFEBasis.size(), false);

  const GridView::IndexSet& indexSet = gridView.indexSet();

  // Make predicate function that computes which vertices are on the Dirichlet boundary, based on the vertex positions.
  auto isDirichlet = [](FieldVector<double,dim> coordinate)
                     {
                       return coordinate[0] < 0.01;
                     };

  for (size_t i=0; i<deformationFEBasis.size(); i++)
    deformationDirichletDofs[i] = isDirichlet(identity[_0][i]);

  for (size_t i=0; i<orientationFEBasis.size(); i++)
    orientationDirichletDofs[i] = isDirichlet(identity[_1][i]);

  ///////////////////////////////////////////
  //  Determine Neumann dofs and values
  ///////////////////////////////////////////

  std::function<bool(FieldVector<double,dim>)> isNeumann
    = [](FieldVector<double,dim> coordinate)
      {
        return coordinate[0] > 0.99 && coordinate[1] > 0.49;
      };

  BitSetVector<1> neumannVertices(gridView.size(dim), false);

  for (auto&& vertex : vertices(gridView))
    neumannVertices[indexSet.index(vertex)] = isNeumann(vertex.geometry().corner(0));

  auto neumannBoundary = std::make_shared<BoundaryPatch<GridType::LeafGridView> >(gridView, neumannVertices);

  FieldVector<double,dimworld> values_ = {0,0,60};
  auto neumannFunction = [&](FieldVector<double, dim>){
                           return values_;
                         };

  ////////////////////////////
  //  Initial iterate
  ////////////////////////////

  Configuration x;

  x[_0].resize(compositeBasis.size({0}));
  x[_1].resize(compositeBasis.size({1}));

  for (std::size_t i=0; i<x[_0].size(); ++i)
    x[_0][i] = {identity[_0][i][0], identity[_0][i][1], 0.0};

  for (auto& rotation : x[_1])
    rotation = Rotation<double,dimworld>::identity();

  //////////////////////////////////
  //  Parameters for the problem
  //////////////////////////////////

  ParameterTree parameters;
  parameters["thickness"] = "0.06";
  parameters["mu"] = "2.7191e+4";
  parameters["lambda"] = "4.4364e+4";
  parameters["mu_c"] = "0";
  parameters["L_c"] = "0.06";
  parameters["q"] = "2";
  parameters["kappa"] = "1";         // Shear correction factor
  // TODO: Remove the following three parameters. They are not actually used by the model
  parameters["b1"] = "1";
  parameters["b2"] = "1";
  parameters["b3"] = "1";

  //////////////////////////////
  //  Create an assembler
  //////////////////////////////

  using RigidBodyMotion = GFE::ProductManifold<RealTuple<double,dimworld>,Rotation<double,dimworld> >;

  auto sumEnergy = std::make_shared<GFE::SumEnergy<CompositeBasis, RealTuple<adouble,dimworld>,Rotation<adouble,dimworld> > >();
  auto neumannEnergy = std::make_shared<GFE::NeumannEnergy<CompositeBasis, RealTuple<adouble,dimworld>, Rotation<adouble,dimworld> > >(neumannBoundary,neumannFunction);

  auto planarCosseratShellEnergy = std::make_shared<CosseratEnergyLocalStiffness<CompositeBasis, dimworld, adouble> >(parameters);
  sumEnergy->addLocalEnergy(planarCosseratShellEnergy);
  sumEnergy->addLocalEnergy(neumannEnergy);

  LocalGeodesicFEADOLCStiffness<CompositeBasis,RigidBodyMotion> localGFEADOLCStiffness(sumEnergy);
  MixedGFEAssembler<CompositeBasis,RigidBodyMotion> mixedAssembler(compositeBasis, localGFEADOLCStiffness);

  MixedRiemannianTrustRegionSolver<GridType,
      CompositeBasis,
      DeformationFEBasis, RealTuple<double,dimworld>,
      OrientationFEBasis, Rotation<double,dimworld> > solver;
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

  solver.setScaling({1,1,1,0.01,0.01,0.01});

  solver.setInitialIterate(x);
  solver.solve();
  x = solver.getSol();

  ////////////////////////////////////////////////
  //  Check if solver returns the expected values
  ////////////////////////////////////////////////

  size_t expectedFinalIteration = 9;
  double expectedEnergy = -6.11748397;

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
