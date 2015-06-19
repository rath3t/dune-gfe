#include "config.h"

#include <dune/grid/uggrid.hh>
#include <dune/grid/geometrygrid.hh>
#include <dune/grid/io/file/vtk.hh>

#include <dune/localfunctions/lagrange/p1.hh>

#include <dune/gfe/unitvector.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/localprojectedfefunction.hh>
#include <dune/gfe/localtangentfefunction.hh>

using namespace Dune;

/** \brief Encapsulates the grid deformation for the GeometryGrid class */
template <class HostGridView>
class DeformationFunction
: public Dune :: DiscreteCoordFunction< double, 3, DeformationFunction<HostGridView> >
{
  typedef DeformationFunction<HostGridView> This;
  typedef Dune :: DiscreteCoordFunction< double, 3, This > Base;

  static const int dim = HostGridView::dimension;

public:

  DeformationFunction(const HostGridView& gridView,
                      const std::vector<FieldVector<double,3> >& deformedPosition)
  : gridView_(gridView),
  deformedPosition_(deformedPosition)
  {}

  void evaluate (const typename HostGridView::template Codim<dim>::Entity& hostEntity, unsigned int corner,
                 Dune::FieldVector<double,3> &y ) const
  {
    const typename HostGridView::IndexSet& indexSet = gridView_.indexSet();
    int idx = indexSet.index(hostEntity);
    y = deformedPosition_[idx];
  }

  void evaluate (const typename HostGridView::template Codim<0>::Entity& hostEntity, unsigned int corner,
                 Dune::FieldVector<double,3> &y ) const
  {
    const typename HostGridView::IndexSet& indexSet = gridView_.indexSet();
    int idx = indexSet.subIndex(hostEntity, corner,dim);
    y = deformedPosition_[idx];
  }

private:

  HostGridView gridView_;

  const std::vector<FieldVector<double,3> > deformedPosition_;

};


template <class TargetSpace, class LocalFEFunctionType>
void interpolate(std::vector<TargetSpace> coefficients,
                const std::string& nameSuffix)
{
  static const int dim = 2;

  typedef UGGrid<dim> GridType;

  GridFactory<GridType> factory;

  factory.insertVertex({0,0});
  factory.insertVertex({1,0});
  factory.insertVertex({0,1});

  GeometryType triangle;
  triangle.makeTriangle();
  factory.insertElement(triangle, {0,1,2});

  shared_ptr<GridType> grid = shared_ptr<GridType>(factory.createGrid());

  grid->globalRefine(6);

  typedef GridType::LeafGridView GridView;
  GridView gridView = grid->leafGridView();

  //////////////////////////////////////////////////////////////
  //   Set up an interpolation function on the sphere
  //////////////////////////////////////////////////////////////

  typedef P1LocalFiniteElement<double,double,dim> LocalFiniteElement;
  LocalFiniteElement localFiniteElement;

  LocalFEFunctionType localGeodesicFEFunction(localFiniteElement,coefficients);

  ///////////////////////////////////////////////////////////////
  //   Sample the interpolating function on the grid
  ///////////////////////////////////////////////////////////////

  std::vector<typename TargetSpace::CoordinateType> samples(gridView.size(dim));

  for (auto v=gridView.begin<dim>(); v!=gridView.end<dim>(); ++v)
    samples[gridView.indexSet().index(*v)] = localGeodesicFEFunction.evaluate(v->geometry().corner(0)).globalCoordinates();

  // sample a checkerboard pattern for nicer pictures
  uint pattern = 8;
  std::vector<double> colors(gridView.size(0));
  for (auto e=gridView.begin<0>(); e!=gridView.end<0>(); ++e)
  {
    FieldVector<double,dim> center = e->geometry().center();

    uint i = pattern * center[0];
    uint j = pattern * center[1];

    FieldVector<double,dim> local;
    local[0] = (center[0] - (i/double(pattern)))*pattern;
    local[1] = (center[1] - (j/double(pattern)))*pattern;

    colors[gridView.indexSet().index(*e)] = (local[0] + local[1] <= 1);
  }

  ///////////////////////////////////////////////////////////////
  //   Write a grid with the interpolated positions
  ///////////////////////////////////////////////////////////////


  typedef Dune::GeometryGrid<GridType,DeformationFunction<typename GridType::LeafGridView> > DeformedGridType;

  DeformationFunction<GridView> deformationFunction(gridView, samples);

  // stupid, can't instantiate deformedGrid with a const grid
  DeformedGridType deformedGrid(const_cast<GridType&>(*grid), deformationFunction);
  Dune::VTKWriter<typename DeformedGridType::LeafGridView> vtkWriter(deformedGrid.leafGridView());
  vtkWriter.addCellData(colors, "colors");
  vtkWriter.write("sphere-patch-" + nameSuffix);

}

int main(int argc, char* argv[]) try
{
  static const int dim = 2;

  //////////////////////////////////////////////////////////////
  //   Set up an interpolation function on the sphere
  //////////////////////////////////////////////////////////////

  typedef UnitVector<double,3> TargetSpace;

  std::vector<TargetSpace> coefficients(3);

  coefficients[0] = TargetSpace(FieldVector<double,3>({1,0,0}));
  coefficients[1] = TargetSpace(FieldVector<double,3>({0,1,0}));
  coefficients[2] = TargetSpace(FieldVector<double,3>({0,0,1}));

  typedef P1LocalFiniteElement<double,double,dim> LocalFiniteElement;
  typedef LocalGeodesicFEFunction<dim, double, LocalFiniteElement, TargetSpace>       LocalGFEFunctionType;
  typedef GFE::LocalProjectedFEFunction<dim, double, LocalFiniteElement, TargetSpace> LocalProjectedFEFunctionType;
  typedef GFE::LocalTangentFEFunction<dim, double, LocalFiniteElement, TargetSpace> LocalTangentFEFunctionType;

  interpolate<TargetSpace,LocalGFEFunctionType>(coefficients, "riemannian");
  interpolate<TargetSpace,LocalProjectedFEFunctionType>(coefficients, "projected");
  interpolate<TargetSpace,LocalTangentFEFunctionType>(coefficients, "tangent");

  std::vector<RealTuple<double,3> > flatCoefficients(3);

  flatCoefficients[0] = RealTuple<double,3>(FieldVector<double,3>({0,0,0}));
  flatCoefficients[1] = RealTuple<double,3>(FieldVector<double,3>({1,0,0}));
  flatCoefficients[2] = RealTuple<double,3>(FieldVector<double,3>({0,1,0}));

  typedef GFE::LocalProjectedFEFunction<dim, double, LocalFiniteElement, RealTuple<double,3> > RealTupleProjectedFEFunctionType;
  interpolate<RealTuple<double,3>, RealTupleProjectedFEFunctionType>(flatCoefficients, "flat");
} catch (Exception e) {

    std::cout << e << std::endl;

}
