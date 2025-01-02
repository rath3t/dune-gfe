#include <iostream>
#include <fstream>

#include <config.h>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>    // use of "Easy to Use" drivers
#include <adolc/taping.h>

#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/bitsetvector.hh>
#include <dune/common/indices.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/common/timer.hh>
#include <dune/common/tuplevector.hh>
#include <dune/common/version.hh>

#if DUNE_VERSION_GTE(DUNE_ELASTICITY, 2, 11)
#include <dune/elasticity/densities/exphenckydensity.hh>
#include <dune/elasticity/densities/henckydensity.hh>
#include <dune/elasticity/densities/mooneyrivlindensity.hh>
#include <dune/elasticity/densities/neohookedensity.hh>
#include <dune/elasticity/densities/stvenantkirchhoffdensity.hh>
#else
#include <dune/elasticity/materials/exphenckydensity.hh>
#include <dune/elasticity/materials/henckydensity.hh>
#include <dune/elasticity/materials/mooneyrivlindensity.hh>
#include <dune/elasticity/materials/neohookedensity.hh>
#include <dune/elasticity/materials/stvenantkirchhoffdensity.hh>
#endif

#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
#include <dune/functions/functionspacebases/compositebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>

#include <dune/fufem/functiontools/boundarydofs.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/io/file/vtk.hh>

#include <dune/gfe/cosseratvtkwriter.hh>
#include <dune/gfe/assemblers/localintegralenergy.hh>
#include <dune/gfe/assemblers/mixedgfeassembler.hh>
#include <dune/gfe/assemblers/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/neumannenergy.hh>
#include <dune/gfe/assemblers/surfacecosseratenergy.hh>
#include <dune/gfe/assemblers/sumenergy.hh>
#include <dune/gfe/densities/duneelasticitydensity.hh>

#if MIXED_SPACE
#include <dune/gfe/mixedriemannianpnsolver.hh>
#include <dune/gfe/mixedriemanniantrsolver.hh>
#else
#include <dune/gfe/assemblers/geodesicfeassemblerwrapper.hh>
#include <dune/gfe/riemannianpnsolver.hh>
#include <dune/gfe/riemanniantrsolver.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#endif

#include <dune/istl/multitypeblockvector.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>


const int dim = 3;
const int targetDim = 3;

const int displacementOrder = 2;
#if MIXED_SPACE
const int rotationOrder = 1;
#else
const int rotationOrder = 2;
#endif

const int stressFreeDataOrder = 2;

#if !MIXED_SPACE
static_assert(displacementOrder==rotationOrder, "displacement and rotation order do not match!");
#endif

//differentiation method
typedef adouble ValueType;

using namespace Dune;

int main (int argc, char *argv[])
{
  Dune::MPIHelper& mpiHelper = MPIHelper::instance(argc, argv);

  if (mpiHelper.rank()==0) {
    std::cout << "DISPLACEMENTORDER = " << displacementOrder << ", ROTATIONORDER = " << rotationOrder << std::endl;
#if MIXED_SPACE
    std::cout << "MIXED_SPACE = 1" << std::endl;
#else
    std::cout << "MIXED_SPACE = 0" << std::endl;
#endif
  }

  // read solver settings
  const double tolerance                = 1e-7;
  const int maxSolverSteps              = 100;
  const double initialTrustRegionRadius = 1;
  const int multigridIterations         = 100;
  const int nu1                         = 3;
  const int nu2                         = 3;
  const int mu                          = 1;
  const int baseIterations              = 100;
  const double mgTolerance              = 1e-7;
  const double baseTolerance            = 1e-7;

  /////////////////////////////////////////////////////////////
  //                      CREATE THE GRID
  /////////////////////////////////////////////////////////////
  using GridType = UGGrid<dim>;

  FieldVector<double,dim> lower(0), upper({4.0,1.0,1.0});

  std::array<unsigned int,dim> elements = {8,2,2};
  auto grid = StructuredGridFactory<GridType>::createCubeGrid(lower, upper, elements);

  grid->setRefinementType(GridType::RefinementType::COPY);

  grid->loadBalance();

  if (mpiHelper.rank()==0)
    std::cout << "There are " << grid->leafGridView().comm().size() << " processes" << std::endl;

  using GridView = GridType::LeafGridView;
  GridView gridView = grid->leafGridView();

  /////////////////////////////////////////////////////////////
  //                      DATA TYPES
  /////////////////////////////////////////////////////////////

  using namespace Dune::Indices;

  typedef std::vector<RealTuple<double,dim> > DisplacementVector;
  typedef std::vector<Rotation<double,dim> > RotationVector;
  const int dimRotation = Rotation<double,dim>::embeddedDim;
  typedef TupleVector<DisplacementVector, RotationVector> SolutionType;

  /////////////////////////////////////////////////////////////
  //                      FUNCTION SPACE
  /////////////////////////////////////////////////////////////

  using namespace Dune::Functions::BasisFactory;
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

  auto deformationPowerBasis = makeBasis(
    gridView,
    power<dim>(
      lagrange<displacementOrder>()
      ));

  using DeformationFEBasis = Functions::LagrangeBasis<GridView,displacementOrder>;
  using OrientationFEBasis = Functions::LagrangeBasis<GridView,rotationOrder>;
  DeformationFEBasis deformationFEBasis(gridView);
  OrientationFEBasis orientationFEBasis(gridView);

  using CompositeBasis = decltype(compositeBasis);

  /////////////////////////////////////////////////////////////
  //                      BOUNDARY DATA
  /////////////////////////////////////////////////////////////

  auto dirichletPredicate = [](const auto& x) {
                              return x[0] < 1e-4;
                            };
  auto neumannPredicate = [](const auto& x) {
                            return x[0] > 4-1e-4;
                          };
  auto surfaceShellPredicate = [](const auto& x) {
                                 return x[2] > 1-1e-4;
                               };

  BitSetVector<1> dirichletVerticesX(gridView.size(dim), false);
  BitSetVector<1> dirichletVerticesY(gridView.size(dim), false);
  BitSetVector<1> dirichletVerticesZ(gridView.size(dim), false);

  BitSetVector<1> neumannVertices(gridView.size(dim), false);
  BitSetVector<1> surfaceShellVertices(gridView.size(dim), false);

  const GridView::IndexSet& indexSet = gridView.indexSet();

  for (auto&& v : vertices(gridView))
  {
    bool isDirichlet = dirichletPredicate(v.geometry().corner(0));
    dirichletVerticesX[indexSet.index(v)] = isDirichlet;
    dirichletVerticesY[indexSet.index(v)] = isDirichlet;
    dirichletVerticesZ[indexSet.index(v)] = isDirichlet;

    neumannVertices[indexSet.index(v)] = neumannPredicate(v.geometry().corner(0));

    surfaceShellVertices[indexSet.index(v)] = surfaceShellPredicate(v.geometry().corner(0));
  }

  BoundaryPatch<GridView> dirichletBoundaryX(gridView, dirichletVerticesX);
  BoundaryPatch<GridView> dirichletBoundaryY(gridView, dirichletVerticesY);
  BoundaryPatch<GridView> dirichletBoundaryZ(gridView, dirichletVerticesZ);
  auto neumannBoundary = std::make_shared<BoundaryPatch<GridView> >(gridView, neumannVertices);
  BoundaryPatch<GridView> surfaceShellBoundary(gridView, surfaceShellVertices);

  std::cout << "On rank " << mpiHelper.rank() << ": Dirichlet boundary has [" << dirichletBoundaryX.numFaces() <<
    ", " << dirichletBoundaryY.numFaces() <<
    ", " << dirichletBoundaryZ.numFaces() <<"] faces\n";
  std::cout << "On rank " << mpiHelper.rank() << ": Neumann boundary has " << neumannBoundary->numFaces() << " faces\n";
  std::cout << "On rank " << mpiHelper.rank() << ": Shell boundary has " << surfaceShellBoundary.numFaces() << " faces\n";

  BitSetVector<1> dirichletNodesX(compositeBasis.size({0}),false);
  BitSetVector<1> dirichletNodesY(compositeBasis.size({0}),false);
  BitSetVector<1> dirichletNodesZ(compositeBasis.size({0}),false);
  BitSetVector<1> surfaceShellNodes(compositeBasis.size({1}),false);
#if DUNE_VERSION_GTE(DUNE_FUFEM, 2, 10)
  Fufem::markBoundaryPatchDofs(dirichletBoundaryX,deformationFEBasis,dirichletNodesX);
  Fufem::markBoundaryPatchDofs(dirichletBoundaryY,deformationFEBasis,dirichletNodesY);
  Fufem::markBoundaryPatchDofs(dirichletBoundaryZ,deformationFEBasis,dirichletNodesZ);
  Fufem::markBoundaryPatchDofs(surfaceShellBoundary,orientationFEBasis,surfaceShellNodes);
#else
  constructBoundaryDofs(dirichletBoundaryX,deformationFEBasis,dirichletNodesX);
  constructBoundaryDofs(dirichletBoundaryY,deformationFEBasis,dirichletNodesY);
  constructBoundaryDofs(dirichletBoundaryZ,deformationFEBasis,dirichletNodesZ);
  constructBoundaryDofs(surfaceShellBoundary,orientationFEBasis,surfaceShellNodes);
#endif

  //Create BitVector matching the tangential space
  const int dimRotationTangent = Rotation<double,dim>::TangentVector::dimension;
  typedef MultiTypeBlockVector<std::vector<FieldVector<double,dim> >, std::vector<FieldVector<double,dimRotationTangent> > > VectorForBit;
  typedef Solvers::DefaultBitVector_t<VectorForBit> BitVector;

  BitVector dirichletDofs;
  dirichletDofs[_0].resize(compositeBasis.size({0}));
  dirichletDofs[_1].resize(compositeBasis.size({1}));
  for (size_t i = 0; i < compositeBasis.size({0}); i++) {
    dirichletDofs[_0][i][0] = dirichletNodesX[i][0];
    dirichletDofs[_0][i][1] = dirichletNodesY[i][0];
    dirichletDofs[_0][i][2] = dirichletNodesZ[i][0];
  }
  for (size_t i = 0; i < compositeBasis.size({1}); i++) {
    for (int j = 0; j < dimRotationTangent; j++) {
      dirichletDofs[_1][i][j] = not surfaceShellNodes[i][0];
    }
  }

  /////////////////////////////////////////////////////////////
  //                      INITIAL DATA
  /////////////////////////////////////////////////////////////

  SolutionType x;
  x[_0].resize(compositeBasis.size({0}));
  x[_1].resize(compositeBasis.size({1}));

  BlockVector<FieldVector<double,dim> > identity(compositeBasis.size({0}));
  Dune::Functions::interpolate(deformationPowerBasis, identity, [](FieldVector<double,dim> x){
    return x;
  });

  for (std::size_t i=0; i<x[_0].size(); ++i)
    x[_0][i] = identity[i];

  for (std::size_t i=0; i<x[_1].size(); ++i)
    x[_1][i] = Rotation<double,dim>::identity();


  /////////////////////////////////////////////////////////////
  //               STRESS-FREE SURFACE SHELL DATA
  /////////////////////////////////////////////////////////////
  auto stressFreeFEBasis = makeBasis(
    gridView,
    power<dim>(
      lagrange<stressFreeDataOrder>(),
      blockedInterleaved()
      ));

  GlobalIndexSet<GridView> globalVertexIndexSet(gridView,dim);
  BlockVector<FieldVector<double,dim> > stressFreeShellVector(stressFreeFEBasis.size());

  // Read grid deformation from deformation function
  auto gridDeformation = [](FieldVector<double,dim> x){
                           return x;
                         };
  Functions::interpolate(stressFreeFEBasis, stressFreeShellVector, gridDeformation);

  auto stressFreeShellFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,dim> >(stressFreeFEBasis, stressFreeShellVector);

  /////////////////////////////////////////////////////////////
  //                      MAIN HOMOTOPY LOOP
  /////////////////////////////////////////////////////////////
  FieldVector<double,dim> neumannValues {3e6,0,0};
  std::cout << "Neumann values: " << neumannValues << std::endl;

  // Function for the Cosserat material parameters
  auto fThickness = [](const auto& x) {
                      return 0.1;
                    };
  auto fLame = [](const auto& x) -> FieldVector<double, 2> {
                 return {1e7, 1e7};
               };

  //////////////////////////////////////////////////////////////
  //   Create an assembler for the energy functional
  //////////////////////////////////////////////////////////////

  // A constant vector-valued function, for simple Neumann boundary values
  auto neumannFunction = std::function<FieldVector<double,dim>(FieldVector<double,dim>)>([&](FieldVector<double,dim> ) {
    return neumannValues * (-1.0);
  });

  ///////////////////////////////////////////////////
  //   Create the energy functional
  ///////////////////////////////////////////////////

  ParameterTree materialParameters;

  materialParameters["mooneyrivlin_energy"] = "ciarlet";
  materialParameters["mooneyrivlin_a"] = "1e6";
  materialParameters["mooneyrivlin_b"] = "1e6";
  materialParameters["mooneyrivlin_c"] = "1e6";

  materialParameters["mu_c"] = "0.0";
  materialParameters["L_c"] = "1e-4";
  materialParameters["b1"] = "1.0";
  materialParameters["b2"] = "1.0";
  materialParameters["b3"] = "1.0";

  using ActiveRigidBodyMotion = GFE::ProductManifold<RealTuple<adouble,dim>, Rotation<adouble,dim> >;

  std::shared_ptr<Elasticity::LocalDensity<dim,ValueType> > elasticDensity;
  elasticDensity = std::make_shared<Elasticity::MooneyRivlinDensity<dim,ValueType> >(materialParameters);

  auto elasticDensityWrapped = std::make_shared<GFE::DuneElasticityDensity<FieldVector<double,dim>,ActiveRigidBodyMotion,0> >(elasticDensity);

  // Select which type of geometric interpolation to use
  using LocalDeformationInterpolationRule = LocalGeodesicFEFunction<dim, GridType::ctype, decltype(deformationFEBasis.localView().tree().finiteElement()), RealTuple<adouble,dim> >;
  using LocalOrientationInterpolationRule = LocalGeodesicFEFunction<dim, GridType::ctype, decltype(orientationFEBasis.localView().tree().finiteElement()), Rotation<adouble,dim> >;

  using LocalInterpolationRule = std::tuple<LocalDeformationInterpolationRule,LocalOrientationInterpolationRule>;

  auto elasticEnergy = std::make_shared<GFE::LocalIntegralEnergy<CompositeBasis, LocalInterpolationRule, ActiveRigidBodyMotion> >(elasticDensityWrapped);
  auto neumannEnergy = std::make_shared<GFE::NeumannEnergy<CompositeBasis, RealTuple<ValueType,targetDim>, Rotation<ValueType,dim> > >(neumannBoundary,neumannFunction);

  auto cosseratShellDensity = std::make_shared<GFE::CosseratShellDensity<
      FieldVector<double,3>, adouble> >(
    materialParameters,
    fThickness,
    fLame);

  auto surfaceCosseratEnergy = std::make_shared<GFE::SurfaceCosseratEnergy<
      decltype(stressFreeShellFunction), CompositeBasis, ActiveRigidBodyMotion> >(
    cosseratShellDensity,
    &surfaceShellBoundary,
    stressFreeShellFunction);

  using RBM = GFE::ProductManifold<RealTuple<double, dim>,Rotation<double,dim> >;

  auto sumEnergy = std::make_shared<GFE::SumEnergy<CompositeBasis, RealTuple<ValueType,targetDim>, Rotation<ValueType,targetDim> > >();
  sumEnergy->addLocalEnergy(neumannEnergy);
  sumEnergy->addLocalEnergy(elasticEnergy);
  sumEnergy->addLocalEnergy(surfaceCosseratEnergy);

  LocalGeodesicFEADOLCStiffness<CompositeBasis,RBM> localGFEADOLCStiffness(sumEnergy);
  MixedGFEAssembler<CompositeBasis,RBM> mixedAssembler(compositeBasis, localGFEADOLCStiffness);

  ////////////////////////////////////////////////////////
  //   Set Dirichlet values
  ////////////////////////////////////////////////////////

  BitSetVector<RealTuple<double,dim>::TangentVector::dimension> deformationDirichletDofs(dirichletDofs[_0].size(), false);
  for (std::size_t i = 0; i < dirichletDofs[_0].size(); i++)
    for (int j = 0; j < RealTuple<double,dim>::TangentVector::dimension; j++)
      deformationDirichletDofs[i][j] = dirichletDofs[_0][i][j];

  BitSetVector<Rotation<double,dim>::TangentVector::dimension> orientationDirichletDofs(dirichletDofs[_1].size(), false);
  for (std::size_t i = 0; i < dirichletDofs[_1].size(); i++)
    for (int j = 0; j < Rotation<double,dim>::TangentVector::dimension; j++)
      orientationDirichletDofs[i][j] = dirichletDofs[_1][i][j];

#if !MIXED_SPACE
  std::vector<RBM> xRBM(compositeBasis.size({0}));
  BitSetVector<RBM::TangentVector::dimension> dirichletDofsRBM(compositeBasis.size({0}), false);
  for (std::size_t i = 0; i < compositeBasis.size({0}); i++) {
    for (int j = 0; j < dim; j ++) {   // Displacement part
      xRBM[i][_0].globalCoordinates()[j] = x[_0][i][j];
      dirichletDofsRBM[i][j] = dirichletDofs[_0][i][j];
    }
    xRBM[i][_1] = x[_1][i];   // Rotation part
    for (int j = dim; j < RBM::TangentVector::dimension; j ++)
      dirichletDofsRBM[i][j] = dirichletDofs[_1][i][j-dim];
  }
  typedef GFE::GeodesicFEAssemblerWrapper<CompositeBasis, DeformationFEBasis, RBM> GFEAssemblerWrapper;
  GFEAssemblerWrapper assembler(&mixedAssembler, deformationFEBasis);
#endif

  ///////////////////////////////////////////////////
  //   Create the chosen solver and solve!
  ///////////////////////////////////////////////////

  std::string solverType = "trustRegion";

  std::size_t finalIteration;
  double finalEnergy;

  if (solverType == "trustRegion")
  {
#if MIXED_SPACE
    MixedRiemannianTrustRegionSolver<GridType, CompositeBasis, DeformationFEBasis, RealTuple<double,dim>, OrientationFEBasis, Rotation<double,dim> > solver;
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
                 mu, nu1, nu2,
                 baseIterations,
                 baseTolerance,
                 false /* instrumented */);

    solver.setScaling({1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    solver.setInitialIterate(x);
    solver.solve();
    x = solver.getSol();
#else
    RiemannianTrustRegionSolver<DeformationFEBasis, RBM, GFEAssemblerWrapper> solver;
    solver.setup(*grid,
                 &assembler,
                 xRBM,
                 dirichletDofsRBM,
                 tolerance,
                 maxSolverSteps,
                 initialTrustRegionRadius,
                 multigridIterations,
                 mgTolerance,
                 mu, nu1, nu2,
                 baseIterations,
                 baseTolerance,
                 false /* instrumented */);

    solver.setScaling({1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    solver.setInitialIterate(xRBM);
    solver.solve();
    xRBM = solver.getSol();
    for (std::size_t i = 0; i < xRBM.size(); i++) {
      x[_0][i] = xRBM[i][_0];
      x[_1][i] = xRBM[i][_1];
    }
#endif
    finalIteration = solver.getStatistics().finalIteration;
    finalEnergy = solver.getStatistics().finalEnergy;
  } else {    // solverType == "proximalNewton"

#if MIXED_SPACE
    GFE::MixedRiemannianProximalNewtonSolver<CompositeBasis, DeformationFEBasis, RealTuple<double,dim>, OrientationFEBasis, Rotation<double,dim>, BitVector> solver;
    solver.setup(*grid,
                 &mixedAssembler,
                 x,
                 dirichletDofs,
                 tolerance,
                 maxSolverSteps,
                 1.0 /* initialRegularization */,
                 false /* instrumented */);
    solver.setInitialIterate(x);
    solver.solve();
    x = solver.getSol();
#else
    RiemannianProximalNewtonSolver<DeformationFEBasis, RBM, GFEAssemblerWrapper> solver;
    solver.setup(*grid,
                 &assembler,
                 xRBM,
                 dirichletDofsRBM,
                 tolerance,
                 maxSolverSteps,
                 1.0 /* initialRegularization */,
                 false /* instrumented */);
    solver.setScaling({1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    solver.setInitialIterate(xRBM);
    solver.solve();
    xRBM = solver.getSol();
    for (std::size_t i = 0; i < xRBM.size(); i++) {
      x[_0][i] = xRBM[i][_0];
      x[_1][i] = xRBM[i][_1];
    }

    finalIteration = solver.getStatistics().finalIteration;
    finalEnergy = solver.getStatistics().finalEnergy;
#endif
  }

  /////////////////////////////////
  //   Output result
  /////////////////////////////////

  // Compute the displacement, for visualization
  auto displacement = x[_0];
  for (std::size_t i = 0; i < compositeBasis.size({0}); i++)
    displacement[i] -= identity[i];

  auto displacementFunction = Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,dim> >(deformationPowerBasis, displacement);

  // We (used to) need to subsample, because VTK cannot natively display real second-order functions
  // TODO: This is not true anymore, update our code accordingly!
  SubsamplingVTKWriter<GridView> vtkWriter(gridView, Dune::refinementLevels(displacementOrder-1));
  vtkWriter.addVertexData(displacementFunction, VTK::FieldInfo("displacement", VTK::FieldInfo::Type::scalar, dim));
  vtkWriter.write("filmonsubstratetest-result");

  // The different configurations tested with the CI system all produce
  // slightly different results.  I don't really understand why this is
  // the case yet.  For the time being, just document the situation.
#if MIXED_SPACE
  std::size_t expectedFinalIteration = 10;
  double expectedEnergy = -13812728.2;
#else
#if DUNE_VERSION_LTE(DUNE_COMMON, 2, 9)
  std::size_t expectedFinalIteration = 11;
  double expectedEnergy = -13763856.8;
#elif HAVE_DUNE_CURVEDGEOMETRY
  std::size_t expectedFinalIteration = 12;
  double expectedEnergy = -13812920.2;
#else
  std::size_t expectedFinalIteration = 12; // or 11
  double expectedEnergy = -13763856.8;
#endif
#endif

#if !MIXED_SPACE && !HAVE_DUNE_CURVEDGEOMETRY
  // This case is covered by two CI jobs that only differ in the compiler they use.
  // Still, one job needs one iteration less than the other one...
  if (finalIteration != expectedFinalIteration && finalIteration != expectedFinalIteration-1)
#else
  if (finalIteration != expectedFinalIteration)
#endif
  {
    std::cerr << "Trust-region solver did " << finalIteration+1
              << " iterations, instead of the expected '" << expectedFinalIteration+1 << "'!" << std::endl;
    return 1;
  }

  if (std::abs(finalEnergy - expectedEnergy) > 1e-1)
  {
    std::cerr << std::setprecision(9);
    std::cerr << "Final energy is " << finalEnergy
              << " but '" << expectedEnergy << "' was expected!" << std::endl;
    return 1;
  }

  return 0;
}
