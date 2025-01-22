#include <config.h>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/drivers/drivers.h>
#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/bitsetvector.hh>
#include <dune/common/version.hh>

#include <dune/grid/onedgrid.hh>

#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>

#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/functiontools/boundarydofs.hh>

#include <dune/gfe/assemblers/cosseratrodenergy.hh>
#include <dune/gfe/assemblers/geodesicfeassembler.hh>
#include <dune/gfe/assemblers/localgeodesicfeadolcstiffness.hh>
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

  // Rod parameter settings
  const double A               = 1;
  const double J1              = 1;
  const double J2              = 1;
  const double E               = 2.5e5;
  const double nu              = 0.3;

  /////////////////////////////////////////
  //    Create the grid
  /////////////////////////////////////////
  using Grid = OneDGrid;
  Grid grid(5, 0, 1);

  grid.globalRefine(2);

  using GridView = Grid::LeafGridView;
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
    referenceConfiguration[i][_0] = {0.0, 0.0, referenceConfigurationX[i]};
    referenceConfiguration[i][_1] = GFE::Rotation<double,3>::identity();
  }

  // Select the reference configuration as initial iterate
  Configuration x = referenceConfiguration;

  ///////////////////////////////////////////
  //   Set Dirichlet values
  ///////////////////////////////////////////

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
                                            true);    // true: The entire boundary is Dirichlet boundary
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
  x[rightBoundaryDof][_0] = {1,0,0};

  FieldVector<double,3> axis = {1,0,0};
  double angle = 0;

  x[rightBoundaryDof][_1] = GFE::Rotation<double,3>(axis, M_PI*angle/180);

  //////////////////////////////////////////////
  //  Create the energy and assembler
  //////////////////////////////////////////////

  using ATargetSpace = TargetSpace::rebind<adouble>::other;
  using GeodesicInterpolationRule  = GFE::LocalGeodesicFEFunction<ScalarBasis, ATargetSpace>;
  using ProjectedInterpolationRule = GFE::LocalProjectedFEFunction<ScalarBasis, ATargetSpace>;

  // Assembler using ADOL-C
  std::shared_ptr<GFE::LocalEnergy<ScalarBasis,ATargetSpace> > localRodEnergy;

  std::string interpolationMethod = "geodesic";

  if (interpolationMethod == "geodesic")
  {
    auto energy = std::make_shared<GFE::CosseratRodEnergy<ScalarBasis, GeodesicInterpolationRule, adouble> >(gridView,
                                                                                                             A, J1, J2, E, nu);
    energy->setReferenceConfiguration(referenceConfiguration);
    localRodEnergy = energy;
  }
  else if (interpolationMethod == "projected")
  {
    auto energy = std::make_shared<GFE::CosseratRodEnergy<ScalarBasis, ProjectedInterpolationRule, adouble> >(gridView,
                                                                                                              A, J1, J2, E, nu);
    energy->setReferenceConfiguration(referenceConfiguration);
    localRodEnergy = energy;
  }
  else
    DUNE_THROW(Exception, "Unknown interpolation method " << interpolationMethod << " requested!");

  GFE::LocalGeodesicFEADOLCStiffness<ScalarBasis,
      TargetSpace> localStiffness(localRodEnergy);

  GFE::GeodesicFEAssembler<ScalarBasis,TargetSpace> rodAssembler(gridView, localStiffness);

  /////////////////////////////////////////////
  //   Create a solver for the rod problem
  /////////////////////////////////////////////

  GFE::RiemannianTrustRegionSolver<ScalarBasis,TargetSpace> solver;

  solver.setup(grid,
               &rodAssembler,
               x,
               dirichletNodes,
               1e-6,      // tolerance
               100,       // max trust region steps
               1,         // initial trust region radius,
               200,       // max multigrid iterations
               1e-10,     // multigrid tolerance
               1, 3, 3,   // V(3,3) multigrid cycle
               100,       // base iterations
               1e-8,      // base tolerance
               false);    // No instrumentation for the solver

  ///////////////////////////////////////////////////////
  //   Solve!
  ///////////////////////////////////////////////////////

  solver.setInitialIterate(x);
  solver.solve();

  x = solver.getSol();

  std::size_t expectedFinalIteration = 4;
  if (solver.getStatistics().finalIteration != expectedFinalIteration)
  {
    std::cerr << "Trust-region solver did " << solver.getStatistics().finalIteration+1
              << " iterations, instead of the expected '" << expectedFinalIteration+1 << "'!" << std::endl;
    return 1;
  }

  double expectedEnergy = 162852.7265417;
  if ( std::abs(solver.getStatistics().finalEnergy - expectedEnergy) > 1e-7)
  {
    std::cerr << std::setprecision(15);
    std::cerr << "Final energy is " << solver.getStatistics().finalEnergy
              << " but '" << expectedEnergy << "' was expected!" << std::endl;
    return 1;
  }

}
catch (Exception& e)
{
  std::cout << e.what() << std::endl;
}
