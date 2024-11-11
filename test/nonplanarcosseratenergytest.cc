#include "config.h"

#include <dune/common/test/testsuite.hh>

#include <dune/curvedgrid/curvedgrid.hh>
#include <dune/foamgrid/foamgrid.hh>

#include <dune/gmsh4/gmsh4reader.hh>
#include <dune/gmsh4/gridcreators/lagrangegridcreator.hh>

#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/functions/gridfunctions/analyticgridviewfunction.hh>

#include <dune/gfe/cosseratvtkwriter.hh>
#include <dune/gfe/assemblers/nonplanarcosseratshellenergy.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>


using namespace Dune;


template <typename FlatGridView, typename CurvedGridView, typename GridGeometry,
    typename DeformationFunction, typename OrientationFunction>
double calculateEnergy(const FlatGridView& flatGridView,
                       const CurvedGridView& curvedGridView,
                       const GridGeometry curvedGridGeometry,
                       const DeformationFunction deformationFunction,
                       const OrientationFunction orientationFunction)
{
  ParameterTree materialParameters;
  materialParameters["thickness"] = "0.1";
  materialParameters["mu"] = "3.8462e+05";
  materialParameters["lambda"] = "2.7149e+05";
  materialParameters["mu_c"] = "0";
  materialParameters["L_c"] = "1e-3";
  materialParameters["q"] = "2.5";
  materialParameters["kappa"] = "0.1";
  materialParameters["b1"] = "1";
  materialParameters["b2"] = "1";
  materialParameters["b3"] = "1";

  using FlatFEBasis = Functions::LagrangeBasis<FlatGridView,2>;
  FlatFEBasis flatFEBasis(flatGridView);

  using namespace Dune::Functions::BasisFactory;
  using namespace Dune::Indices;

  TupleVector<std::vector<RealTuple<double,3> >, std::vector<Rotation<double,3> > > configuration;
  configuration[_0].resize(flatFEBasis.size());
  configuration[_1].resize(flatFEBasis.size());

  /////////////////////////////////////////////////////////////////////////
  //  FE representation of the deformation field
  /////////////////////////////////////////////////////////////////////////

  auto curvedGridDeformationBasis = makeBasis(
    curvedGridView,
    power<3>(
      lagrange<2>()
      ));

  BlockVector<FieldVector<double,3> > deformationAsVector(flatFEBasis.size());
  Functions::interpolate(curvedGridDeformationBasis, deformationAsVector, deformationFunction);
  for (std::size_t i = 0; i < flatFEBasis.size(); i++)
    configuration[_0][i].globalCoordinates() = deformationAsVector[i];

  /////////////////////////////////////////////////////////////////////////
  //  FE representation of the microrotation field
  /////////////////////////////////////////////////////////////////////////

  auto curvedGridQuaternionBasis = makeBasis(
    curvedGridView,
    power<4>(
      lagrange<2>()
      ));

  // Turn matrix-valued function into quaternion-valued function
  auto orientationQuaternionFunction
    = [&orientationFunction](FieldVector<double,3> x) -> FieldVector<double,4>
                                       {
                                         Rotation<double,3> rotation;
                                         rotation.set(orientationFunction(x));
                                         return rotation;
                                       };

  BlockVector<FieldVector<double,4> > orientationAsVector(flatFEBasis.size());
  Functions::interpolate(curvedGridQuaternionBasis, orientationAsVector, orientationQuaternionFunction);
  for (std::size_t i = 0; i < flatFEBasis.size(); i++)
    configuration[_1][i] = orientationAsVector[i];

  /////////////////////////////////////////////////////////////////////////
  //  Write the configuration to a file (just for debugging)
  /////////////////////////////////////////////////////////////////////////

  auto directorBasis = makeBasis(
    flatGridView,
    power<3>(
      lagrange<2>()
      ));

  // TODO: Write the curved grid, not the flat one
  CosseratVTKWriter<FlatGridView>::write(flatGridView,
                                         curvedGridGeometry,
                                         directorBasis,
                                         configuration[_1],
                                         2, // VTK output element order
                                         "nonplanarcosseratenergytest-result.vtu");

  ///////////////////////////////////////////////////
  //  Construct the energy functional
  ///////////////////////////////////////////////////

  using GlobalCoordinate = typename FlatGridView::template Codim<0>::Entity::Geometry::GlobalCoordinate;
  auto density = std::make_shared<GFE::CosseratShellDensity<GlobalCoordinate, double> >(materialParameters);

  using ShellEnergy = NonplanarCosseratShellEnergy<FlatFEBasis,
      3,                                               // Dimension of the target space
      double,
      GridGeometry>;

  ShellEnergy nonplanarCosseratShellEnergy(density, &curvedGridGeometry);

  ///////////////////////////////////////////////////
  //  Compute the energy
  ///////////////////////////////////////////////////

  using TargetSpace = GFE::ProductManifold<RealTuple<double,3>,Rotation<double,3> >;

  double energy = 0;

  // A view on the FE basis on a single element
  auto localView = flatFEBasis.localView();

  // Loop over all elements
  for (const auto& element : elements(flatGridView))
  {
    localView.bind(element);

    // Number of degrees of freedom on this element
    size_t nDofs = localView.tree().size();
    std::vector<TargetSpace> localConfiguration(nDofs);
    for (size_t i=0; i<nDofs; i++)
    {
      localConfiguration[i][_0] = configuration[_0][localView.index(i)[0]];
      localConfiguration[i][_1] = configuration[_1][localView.index(i)[0]];
    }
    energy += nonplanarCosseratShellEnergy.energy(localView, localConfiguration);
  }
  return energy;
}



int main(int argc, char** argv)
{
  MPIHelper::instance(argc, argv);

  //////////////////////////////////
  //    Create the grid
  //////////////////////////////////

  using FlatGrid = FoamGrid<2,3>;

  GridFactory<FlatGrid> factory;
  Gmsh4::LagrangeGridCreator gridGeometry{factory};

  Gmsh4Reader reader{gridGeometry};
  reader.read(GRID_PATH "/sphere_order2.msh");
  auto flatGrid = factory.createGrid();

  // Wrap the flat grid to build a curved grid, with Lagrange elements of order 2
  CurvedGrid curvedGrid{*flatGrid, gridGeometry, 2};

  ///////////////////////////////////////////////////////
  //  Create configurations and check their energies
  ///////////////////////////////////////////////////////

  TestSuite test;

  // The reference configuration
  // ---------------------------------
  auto deformationIdentity = [](FieldVector<double,3> x){
                               return x;
                             };

  auto orientationIdentity = [](FieldVector<double,3> x) -> FieldMatrix<double,3,3>
                             {
                               return {{1,0,0}, {0,1,0}, {0,0,1}};
                             };

  double energyIdentity = calculateEnergy(flatGrid->leafGridView(), curvedGrid.leafGridView(), gridGeometry,
                                          deformationIdentity, orientationIdentity);

  test.check(std::fabs(energyIdentity) < 1e-12, "reference configuration has zero energy");

  // A translation
  // ---------------------------------
  auto deformationTranslation = [](FieldVector<double,3> x){
                                  return x + FieldVector<double,3>{1.5,1.5,1.5};
                                };

  double energyTranslation = calculateEnergy(flatGrid->leafGridView(), curvedGrid.leafGridView(), gridGeometry,
                                             deformationTranslation, orientationIdentity);

  test.check(std::fabs(energyTranslation) < 1e-12, "translated configuration has zero energy");

  // A rotation -- for simplicity about an axis
  // ---------------------------------
  double angle = M_PI/4;
  FieldMatrix<double,3,3> globalRotation = {{1,0,0},
    {0,cos(angle),-sin(angle)},
    {0,sin(angle),cos(angle)}};

  // A translation
  auto deformationRotation = [&globalRotation](FieldVector<double,3> x){
                               FieldVector<double,3> result;
                               globalRotation.mv(x,result);
                               return result;
                             };

  auto orientationRotation = [&globalRotation](FieldVector<double,3> x) -> FieldMatrix<double,3,3>
                             {
                               return globalRotation;
                             };


  double energyRotation = calculateEnergy(flatGrid->leafGridView(), curvedGrid.leafGridView(), gridGeometry,
                                          deformationRotation, orientationRotation);

  test.check(std::fabs(energyRotation) < 1e-12, "rotated configuration has zero energy");

  // Stretching
  // ---------------------------------
  auto deformationStretchY = [](FieldVector<double,3> x) -> FieldVector<double,3>
                             {
                               return {x[0], 2*x[1], x[2]};
                             };

  double energyStretchY = calculateEnergy(flatGrid->leafGridView(), curvedGrid.leafGridView(), gridGeometry,
                                          deformationStretchY, orientationIdentity);

  test.check(std::fabs(energyStretchY-357163.4280328181) < 1e-6,
             "stretched configuration has energy 357163.4280328181");



  // Something wild
  // ----------------------------------------
  auto deformationIrregular = [](FieldVector<double,3> x) -> FieldVector<double,3>
                              {
                                return {x[0] + sin(3*M_PI*(x[1]+x[2])),
                                        x[1] + sin(5*M_PI*(x[0]+x[2])),
                                        x[2] + sin(2*M_PI*(x[0]+x[1]))};
                              };

  auto orientationIrregular = [](FieldVector<double,3> x) -> FieldMatrix<double,3>
                              {
                                double angle = sqrt(x[0]*x[0] + x[1]*x[1]);

                                return {{1,0,0},
                                  {0,cos(angle),-sin(angle)},
                                  {0,sin(angle),cos(angle)}};
                              };

  double energyIrregular = calculateEnergy(flatGrid->leafGridView(), curvedGrid.leafGridView(), gridGeometry,
                                           deformationIrregular, orientationIrregular);

  test.check(std::fabs(energyIrregular-56331610.0823059) < 1e-6,
             "irregular configuration has energy 56331610.0823059");

  return test.exit();
}
