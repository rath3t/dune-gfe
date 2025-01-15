#include <config.h>

#include <dune/grid/onedgrid.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>

#include <dune/gfe/assemblers/cosseratrodenergy.hh>
#include <dune/gfe/functions/localgeodesicfefunction.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>


using namespace Dune;
using namespace Dune::Indices;


int main (int argc, char *argv[]) try
{
  // Type used for algebraic rod configurations
  using TargetSpace = GFE::ProductManifold<GFE::RealTuple<double,3>,GFE::Rotation<double,3> >;
  using SolutionType = std::vector<TargetSpace>;

  // Problem settings
  const int numRodBaseElements = 100;

  // ///////////////////////////////////////
  //    Create the grid
  // ///////////////////////////////////////
  typedef OneDGrid GridType;
  GridType grid(numRodBaseElements, 0, 1);
  using GridView = GridType::LeafGridView;
  GridView gridView = grid.leafGridView();

  using FEBasis = Functions::LagrangeBasis<GridView,1>;
  FEBasis feBasis(gridView);

  SolutionType x(feBasis.size());

  // //////////////////////////
  //   Initial solution
  // //////////////////////////

  for (size_t i=0; i<x.size(); i++)
  {
    double s = double(i)/(x.size()-1);
    x[i][_0] = {0.1*std::cos(2*M_PI*s), 0.1*std::sin(2*M_PI*s), s};
    x[i][_1] = GFE::Rotation<double,3>::identity();
    //x[i].q = GFE::Quaternion<double>(zAxis, (double(i)*M_PI)/(2*(x.size()-1)) );
  }

  FieldVector<double,3> zAxis(0);  zAxis[2]=1;
  x.back()[_1] = GFE::Rotation<double,3>(zAxis, M_PI/4);

  // /////////////////////////////////////////////////////////////////////
  //   Create a second, rotated copy of the configuration
  // /////////////////////////////////////////////////////////////////////

  FieldVector<double,3> displacement {0, 1, 0};

  FieldVector<double,3> axis = {1,0,0};
  GFE::Rotation<double,3> rotation(axis,M_PI/2);

  SolutionType rotatedX = x;

  for (size_t i=0; i<rotatedX.size(); i++)
  {
    rotatedX[i][_0] = rotation.rotate(x[i][_0].globalCoordinates());
    rotatedX[i][_0].globalCoordinates() += displacement;

    rotatedX[i][_1] = rotation.mult(x[i][_1]);
  }

  using GeodesicInterpolationRule  = GFE::LocalGeodesicFEFunction<1, double,
      FEBasis::LocalView::Tree::FiniteElement,
      TargetSpace>;

  GFE::CosseratRodEnergy<FEBasis,
      GeodesicInterpolationRule,
      double> localRodEnergy(gridView,
                             1,1,1,1e6,0.3);

  SolutionType referenceConfiguration(gridView.size(1));

  for (const auto& vertex : vertices(gridView))
  {
    auto idx = gridView.indexSet().index(vertex);

    referenceConfiguration[idx][_0] = {0.0, 0.0, vertex.geometry().corner(0)[0]};
    referenceConfiguration[idx][_1] = GFE::Rotation<double,3>::identity();
  }

  localRodEnergy.setReferenceConfiguration(referenceConfiguration);

  auto localView = feBasis.localView();
  localView.bind(*gridView.begin<0>());

  SolutionType localX = {x[0], x[1]};
  SolutionType localRotatedX = {rotatedX[0], rotatedX[1]};

  if (std::abs(localRodEnergy.energy(localView, localX) - localRodEnergy.energy(localView, localRotatedX)) > 1e-6)
    DUNE_THROW(Dune::Exception, "Rod energy not invariant under rigid body motions!");

}
catch (Exception& e) {

  std::cout << e.what() << std::endl;

}
