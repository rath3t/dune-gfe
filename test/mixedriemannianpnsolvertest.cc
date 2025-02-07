#include <config.h>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/bitsetvector.hh>
#include <dune/common/typetraits.hh>
#include <dune/common/tuplevector.hh>

#include <dune/functions/functionspacebases/compositebasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>

#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/functiontools/boundarydofs.hh>

#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/grid/uggrid.hh>

#include <dune/gfe/assemblers/geodesicfeassembler.hh>
#include <dune/gfe/assemblers/geodesicfeassemblerwrapper.hh>
#include <dune/gfe/assemblers/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/assemblers/localintegralenergy.hh>
#include <dune/gfe/assemblers/mixedgfeassembler.hh>
#include <dune/gfe/assemblers/sumenergy.hh>
#include <dune/gfe/densities/planarcosseratshelldensity.hh>
#include <dune/gfe/functions/localgeodesicfefunction.hh>
#include <dune/gfe/mixedriemannianpnsolver.hh>
#include <dune/gfe/neumannenergy.hh>
#include <dune/gfe/riemannianpnsolver.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>

/** \file
 * \brief This test compares the MixedRiemannianPNSolver to the RiemannianPNSolver.
 */

// grid dimension
const int gridDim = 2;

// target dimension
const int dim = 3;

//order of the finite element spaces, they need to be the same to compare!
const int displacementOrder = 2;
const int rotationOrder = displacementOrder;

using namespace Dune;
using namespace Indices;

//differentiation method: ADOL-C
using ValueType = adouble;

//Types for the mixed space
using DisplacementVector = std::vector<GFE::RealTuple<double,dim> >;
using RotationVector =  std::vector<GFE::Rotation<double,dim> >;
using Vector = TupleVector<DisplacementVector, RotationVector>;
using BlockTupleVector = TupleVector<BlockVector<GFE::RealTuple<double,dim> >, BlockVector<GFE::Rotation<double,dim> > >;
const int dimRotationTangent = GFE::Rotation<double,dim>::TangentVector::dimension;


int main (int argc, char *argv[])
{
  MPIHelper::instance(argc, argv);

  /////////////////////////////////////////////////////////////////////////
  //    Create the grid
  /////////////////////////////////////////////////////////////////////////
  using GridType = UGGrid<gridDim>;
  auto grid = StructuredGridFactory<GridType>::createCubeGrid({0.0,0.0}, {1.0,1.0}, {2,2});
  grid->globalRefine(2);
  grid->loadBalance();

  using GridView = GridType::LeafGridView;
  GridView gridView = grid->leafGridView();

  /////////////////////////////////////////////////////////////////////////
  //  Create a composite basis and a Lagrange FE basis
  /////////////////////////////////////////////////////////////////////////

  using namespace Functions::BasisFactory;

  auto compositeBasis = makeBasis(
    gridView,
    composite(
      power<dim>(
        lagrange<displacementOrder>()
        ),
      power<dim>(
        lagrange<rotationOrder>()
        )
      ));

  using CompositeBasis = decltype(compositeBasis);

  using DeformationFEBasis = Functions::LagrangeBasis<GridView,displacementOrder>;
  DeformationFEBasis deformationFEBasis(gridView);
  using RotationFEBasis = Functions::LagrangeBasis<GridView,rotationOrder>;
  RotationFEBasis rotationFEBasis(gridView);

  /////////////////////////////////////////////////////////////////////////
  //  Create the Neumann and Dirichlet boundary
  /////////////////////////////////////////////////////////////////////////

  std::function<bool(FieldVector<double,gridDim>)> isNeumann = [](FieldVector<double,gridDim> coordinate) {
                                                                 return coordinate[0] > 0.99; //Neumann for upper boundary
                                                               };
  std::function<bool(FieldVector<double,gridDim>)> isDirichlet = [](FieldVector<double,gridDim> coordinate) {
                                                                   return coordinate[0] < 0.01; //Dircichlet for lower boundary
                                                                 };

  BitSetVector<1> neumannVertices(gridView.size(gridDim), false);
  BitSetVector<1> dirichletVertices(gridView.size(gridDim), false);
  const GridView::IndexSet& indexSet = gridView.indexSet();

  for (auto&& vertex : vertices(gridView)) {
    neumannVertices[indexSet.index(vertex)] = isNeumann(vertex.geometry().corner(0));
    dirichletVertices[indexSet.index(vertex)] = isDirichlet(vertex.geometry().corner(0));
  }

  auto neumannBoundary = std::make_shared<BoundaryPatch<GridView> >(gridView, neumannVertices);
  BoundaryPatch<GridView> dirichletBoundary(gridView, dirichletVertices);

  BitSetVector<1> dirichletNodes(compositeBasis.size({0}),false);
#if DUNE_VERSION_GTE(DUNE_FUFEM, 2, 10)
  Fufem::markBoundaryPatchDofs(dirichletBoundary, deformationFEBasis, dirichletNodes);
#else
  constructBoundaryDofs(dirichletBoundary, deformationFEBasis, dirichletNodes);
#endif

  typedef MultiTypeBlockVector<std::vector<FieldVector<double,dim> >, std::vector<FieldVector<double,dimRotationTangent> > > VectorForBit;
  using BitVector = Solvers::DefaultBitVector_t<VectorForBit>;

  BitVector dirichletDofs;
  dirichletDofs[_0].resize(compositeBasis.size({0}));
  dirichletDofs[_1].resize(compositeBasis.size({1}));
  for (size_t i = 0; i < compositeBasis.size({0}); i++) {
    for (size_t j = 0; j < dim; j++)
      dirichletDofs[_0][i][j] = dirichletNodes[i][0];
    for (size_t j = 0; j < dimRotationTangent; j++)
      dirichletDofs[_1][i][j] = dirichletNodes[i][0];
  }

  /////////////////////////////////////////////////////////////////////////
  //  Create the energy functions with their parameters
  /////////////////////////////////////////////////////////////////////////

  //Surface-Cosserat-Energy-Parameters
  ParameterTree parameters;
  parameters["thickness"] = "1";
  parameters["mu"] = "2.7191e+4";
  parameters["lambda"] = "4.4364e+4";
  parameters["mu_c"] = "0";
  parameters["L_c"] = "0.01";
  parameters["q"] = "2";
  parameters["kappa"] = "1";


  FieldVector<double,dim> values_ = {3e4,2e4,1e4};
  auto neumannFunction = [&](FieldVector<double, gridDim>){
                           return values_;
                         };

  // The target space, with 'double' and 'adouble' as number types
  using RBM = GFE::ProductManifold<GFE::RealTuple<double,dim>,GFE::Rotation<double,dim> >;
  using ARBM = typename RBM::template rebind<adouble>::other;

  // The total energy
  auto sumEnergy = std::make_shared<GFE::SumEnergy<CompositeBasis, GFE::RealTuple<adouble,dim>,GFE::Rotation<adouble,dim> > >();

  // The Cosserat shell energy
  using AInterpolationRule = std::tuple<GFE::LocalGeodesicFEFunction<DeformationFEBasis, GFE::RealTuple<adouble,3> >,
      GFE::LocalGeodesicFEFunction<RotationFEBasis, GFE::Rotation<adouble,3> > >;
  AInterpolationRule localGFEFunctionA{GFE::LocalGeodesicFEFunction<DeformationFEBasis, GFE::RealTuple<adouble,3> >(deformationFEBasis),
                                       GFE::LocalGeodesicFEFunction<RotationFEBasis, GFE::Rotation<adouble,3> >(rotationFEBasis)};

  auto cosseratDensity = std::make_shared<GFE::PlanarCosseratShellDensity<GridType::Codim<0>::Entity, adouble> >(parameters);

  auto planarCosseratShellEnergy = std::make_shared<GFE::LocalIntegralEnergy<CompositeBasis,AInterpolationRule,ARBM> >(std::move(localGFEFunctionA), cosseratDensity);

  sumEnergy->addLocalEnergy(planarCosseratShellEnergy);

  // The Neumann surface load term
  auto neumannEnergy = std::make_shared<GFE::NeumannEnergy<CompositeBasis, GFE::RealTuple<adouble,dim>, GFE::Rotation<adouble,dim> > >(neumannBoundary,neumannFunction);
  sumEnergy->addLocalEnergy(neumannEnergy);

  // The local assembler
  GFE::LocalGeodesicFEADOLCStiffness<CompositeBasis,RBM> mixedLocalGFEADOLCStiffness(sumEnergy);

  // The global assembler
  GFE::MixedGFEAssembler<CompositeBasis, RBM> mixedAssembler(compositeBasis, mixedLocalGFEADOLCStiffness);

  using GFEAssemblerWrapper = GFE::GeodesicFEAssemblerWrapper<CompositeBasis, DeformationFEBasis, RBM>;
  GFEAssemblerWrapper assembler(&mixedAssembler, deformationFEBasis);

  /////////////////////////////////////////////////////////////////////////
  //  Prepare the iterate x where we want to assemble - identity in 2D with z = 0
  /////////////////////////////////////////////////////////////////////////
  auto deformationPowerBasis = makeBasis(
    gridView,
    power<gridDim>(
      lagrange<displacementOrder>()
      ));
  BlockVector<FieldVector<double,gridDim> > identity(compositeBasis.size({0}));
  Functions::interpolate(deformationPowerBasis, identity, [](FieldVector<double,gridDim> x){
    return x;
  });
  BlockVector<FieldVector<double,dim> > initialDeformation(compositeBasis.size({0}));
  initialDeformation = 0;

  Vector x;
  x[_0].resize(compositeBasis.size({0}));
  x[_1].resize(compositeBasis.size({1}));
  std::vector<RBM> xRBM(compositeBasis.size({0}));
  BitSetVector<RBM::TangentVector::dimension> dirichletDofsRBM(compositeBasis.size({0}), false);

  for (std::size_t i = 0; i < compositeBasis.size({0}); i++) {
    for (int j = 0; j < gridDim; j++)
      initialDeformation[i][j] = identity[i][j];
    x[_0][i] = initialDeformation[i];
    xRBM[i][_0] = x[_0][i];
    for (int j = 0; j < dim; j ++) { // Displacement part
      dirichletDofsRBM[i][j] = dirichletDofs[_0][i][j];
    }
    xRBM[i][_1] = x[_1][i]; // Rotation part
    for (int j = dim; j < RBM::TangentVector::dimension; j ++)
      dirichletDofsRBM[i][j] = dirichletDofs[_1][i][j-dim];
  }

  //////////////////////////////////////////////////////////////////////////////
  //  Create a MixedRiemannianPNSolver and a normal RiemannianPNSolver
  //  and compare one solver step!
  //////////////////////////////////////////////////////////////////////////////

  const double tolerance = 1e-7;
  const int maxSolverSteps = 1;
  const double initialRegularization = 100;
  const bool instrumented = false;
  GFE::MixedRiemannianProximalNewtonSolver<CompositeBasis, DeformationFEBasis, GFE::RealTuple<double,dim>, DeformationFEBasis, GFE::Rotation<double,dim>, BitVector> mixedSolver;
  mixedSolver.setup(*grid,
                    &mixedAssembler,
                    x,
                    dirichletDofs,
                    tolerance,
                    maxSolverSteps,
                    initialRegularization,
                    instrumented);
  mixedSolver.setInitialIterate(x);
  mixedSolver.solve();
  x = mixedSolver.getSol();

  GFE::RiemannianProximalNewtonSolver<DeformationFEBasis, RBM, GFEAssemblerWrapper> solver;
  solver.setup(*grid,
               &assembler,
               xRBM,
               dirichletDofsRBM,
               tolerance,
               maxSolverSteps,
               initialRegularization,
               instrumented);
  solver.setInitialIterate(xRBM);
  solver.solve();
  xRBM = solver.getSol();

  BlockTupleVector xMixed;
  BlockTupleVector xNotMixed;
  xNotMixed[_0].resize(compositeBasis.size({0}));
  xNotMixed[_1].resize(compositeBasis.size({1}));
  xMixed[_0].resize(compositeBasis.size({0}));
  xMixed[_1].resize(compositeBasis.size({1}));
  for (std::size_t i = 0; i < xRBM.size(); i++)
  {
    xNotMixed[_0][i] = xRBM[i][_0];
    xNotMixed[_1][i] = xRBM[i][_1];
    xMixed[_0][i] = x[_0][i];
    xMixed[_1][i] = x[_1][i];
    auto difference0 = xMixed[_0][i].globalCoordinates();
    auto difference1 = xMixed[_1][i].globalCoordinates();
    difference0 -= xNotMixed[_0][i].globalCoordinates();
    difference1 -= xNotMixed[_1][i].globalCoordinates();
    if (difference0.two_norm() > 1e-1 || difference1.two_norm() > 1e-1) {
      std::cerr << std::setprecision(9);
      std::cerr << "At index " << i << " the solution calculated by the MixedRiemannianPNSolver is "
                << xMixed[_0][i] << " and " << xMixed[_1][i] << " but "
                << xNotMixed[_0][i] << " and " << xNotMixed[_1][i]
                << " (calculated by the RiemannianProximalNewtonSolver) was expected!" << std::endl;
      return 1;
    }
  }
}
