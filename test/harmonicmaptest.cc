#include <config.h>

#include <array>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/typetraits.hh>
#include <dune/common/bitsetvector.hh>
#include <dune/common/version.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/io/file/gmshreader.hh>

#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>

#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/functiontools/boundarydofs.hh>

#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/localprojectedfefunction.hh>
#include <dune/gfe/assemblers/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/assemblers/harmonicenergy.hh>
#include <dune/gfe/assemblers/geodesicfeassembler.hh>
#include <dune/gfe/riemanniantrsolver.hh>
#include <dune/gfe/spaces/unitvector.hh>

// grid dimension
const int dim = 2;

// Image space of the geodesic fe functions
typedef UnitVector<double,3> TargetSpace;

const int order = 1;

using namespace Dune;


int main (int argc, char *argv[])
{
  MPIHelper::instance(argc, argv);

  using SolutionType = std::vector<TargetSpace>;

  // read problem settings
  const int numLevels                   = 4;

  // read solver settings
  const double tolerance                = 1e-6;
  const int maxTrustRegionSteps         = 1000;
  const double initialTrustRegionRadius = 1;
  const int multigridIterations         = 200;
  const int baseIterations              = 100;
  const double mgTolerance              = 1e-10;
  const double baseTolerance            = 1e-8;

  // ///////////////////////////////////////
  //    Create the grid
  // ///////////////////////////////////////
  using GridType = UGGrid<dim>;

  std::shared_ptr<GridType> grid = GmshReader<GridType>::read("grids/irregular-square.msh");

  grid->globalRefine(numLevels-1);

  grid->loadBalance();

  using GridView = GridType::LeafGridView;
  GridView gridView = grid->leafGridView();

  //////////////////////////////////////////////////////////////////////////////////
  //  Construct the scalar function space basis corresponding to the GFE space
  //////////////////////////////////////////////////////////////////////////////////

  using FEBasis = Functions::LagrangeBasis<GridView, order>;
  FEBasis feBasis(gridView);
  SolutionType x(feBasis.size());

  using namespace Functions::BasisFactory;

  // A basis for the embedding space
  auto powerBasis = makeBasis(
    gridView,
    power<TargetSpace::CoordinateType::dimension>(
      lagrange<order>(),
      blockedInterleaved()
      ));

  // A basis for the tangent space
  auto tangentBasis = makeBasis(
    gridView,
    power<TargetSpace::TangentVector::dimension>(
      lagrange<order>(),
      blockedInterleaved()
      ));

  ///////////////////////////////////////////
  //  Determine Dirichlet values
  ///////////////////////////////////////////
  BitSetVector<1> dirichletVertices(gridView.size(dim), false);
  const GridView::IndexSet& indexSet = gridView.indexSet();

  // Make predicate function that computes which vertices are on the Dirichlet boundary,
  // based on the vertex positions.
  auto dirichletVerticesPredicate = [](FieldVector<double,dim> x)
                                    {
                                      return (x[0] < -4.9999 or x[0] > 4.9999 or x[1] < -4.9999 or x[1] > 4.9999);
                                    };

  for (auto&& vertex : vertices(gridView))
  {
    bool isDirichlet = dirichletVerticesPredicate(vertex.geometry().corner(0));
    dirichletVertices[indexSet.index(vertex)] = isDirichlet;
  }

  BoundaryPatch<GridView> dirichletBoundary(gridView, dirichletVertices);
  BitSetVector<TargetSpace::TangentVector::dimension> dirichletNodes(powerBasis.size(), false);
#if DUNE_VERSION_GTE(DUNE_FUFEM, 2, 10)
  Fufem::markBoundaryPatchDofs(dirichletBoundary,tangentBasis,dirichletNodes);
#else
  constructBoundaryDofs(dirichletBoundary,tangentBasis,dirichletNodes);
#endif

  ////////////////////////////
  //  Initial iterate
  ////////////////////////////

  // The inverse stereographic projection through the north pole
  auto initialIterateFunction = [](FieldVector<double,dim> x) -> TargetSpace::CoordinateType
                                {
                                  auto normSquared = x.two_norm2();
                                  return {2*x[0] / (normSquared+1), 2*x[1] / (normSquared+1), (normSquared-1)/ (normSquared+1)};
                                };

  std::vector<TargetSpace::CoordinateType> v;
  Dune::Functions::interpolate(powerBasis, v, initialIterateFunction);

  for (size_t i=0; i<x.size(); i++)
    x[i] = v[i];

  //////////////////////////////////////////////////////////////
  //  Create an assembler for the Harmonic Energy Functional
  //////////////////////////////////////////////////////////////

  using ATargetSpace = TargetSpace::rebind<adouble>::other;
  //using GeodesicInterpolationRule  = LocalGeodesicFEFunction<dim, double, FEBasis::LocalView::Tree::FiniteElement, ATargetSpace>;
  using ProjectedInterpolationRule = GFE::LocalProjectedFEFunction<dim, double, FEBasis::LocalView::Tree::FiniteElement, ATargetSpace>;

  //std::shared_ptr<GFE::LocalEnergy<FEBasis,ATargetSpace> > localEnergy = std::make_shared<HarmonicEnergy<FEBasis, GeodesicInterpolationRule, ATargetSpace> >();
  std::shared_ptr<GFE::LocalEnergy<FEBasis,ATargetSpace> > localEnergy = std::make_shared<HarmonicEnergy<FEBasis, ProjectedInterpolationRule, ATargetSpace> >();

  LocalGeodesicFEADOLCStiffness<FEBasis,TargetSpace> localGFEADOLCStiffness(localEnergy.get());

  GeodesicFEAssembler<FEBasis,TargetSpace> assembler(feBasis, localGFEADOLCStiffness);

  ///////////////////////////////////////////////////
  //  Create a Riemannian trust-region solver
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
               3, 3, 1,   // Multigrid V-cycle
               baseIterations,
               baseTolerance,
               false);   // Do not write intermediate iterates

  ///////////////////////////////////////////////////////
  //  Solve!
  ///////////////////////////////////////////////////////

  solver.setInitialIterate(x);
  solver.solve();

  x = solver.getSol();

  std::size_t expectedFinalIteration = 12;
  if (solver.getStatistics().finalIteration != expectedFinalIteration)
  {
    std::cerr << "Trust-region solver did " << solver.getStatistics().finalIteration+1
              << " iterations, instead of the expected '" << expectedFinalIteration+1 << "'!" << std::endl;
    return 1;
  }

  double expectedEnergy = 12.2927849;
  if ( std::abs(solver.getStatistics().finalEnergy - expectedEnergy) > 1e-7)
  {
    std::cerr << std::setprecision(9);
    std::cerr << "Final energy is " << solver.getStatistics().finalEnergy
              << " but '" << expectedEnergy << "' was expected!" << std::endl;
    return 1;
  }

  return 0;
}
