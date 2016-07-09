#include "config.h"

#include <dune/grid/uggrid.hh>
#include <dune/grid/geometrygrid.hh>
#include <dune/grid/io/file/vtk.hh>

#include <dune/localfunctions/lagrange/p1.hh>
#include <dune/localfunctions/lagrange/p2.hh>

#include <dune/functions/functionspacebases/pq1nodalbasis.hh>
#include <dune/functions/gridfunctions/discretescalarglobalbasisfunction.hh>

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

/** \brief
 *
 * \param partialDerivative The dof with respect to which we are computing the variation
 */
template <class TargetSpace, class LocalFEFunctionType>
void interpolate(const LocalFEFunctionType& localGeodesicFEFunction,
                 int partialDerivative,
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

  grid->globalRefine(3);

  typedef GridType::LeafGridView GridView;
  GridView gridView = grid->leafGridView();

  ///////////////////////////////////////////////////////////////
  //   Sample the interpolating function on the grid
  ///////////////////////////////////////////////////////////////

  std::vector<typename TargetSpace::CoordinateType> samples(gridView.size(dim));

  for (auto v=gridView.begin<dim>(); v!=gridView.end<dim>(); ++v)
    samples[gridView.indexSet().index(*v)] = localGeodesicFEFunction.evaluate(v->geometry().corner(0)).globalCoordinates();

  // Sample a variation vector field
  std::vector<typename TargetSpace::EmbeddedTangentVector> variation0(gridView.size(dim));
  std::vector<typename TargetSpace::EmbeddedTangentVector> variation1(gridView.size(dim));
  for (const auto& v : vertices(gridView))
  {
    FieldMatrix<double,3,3> derivative;
    localGeodesicFEFunction.evaluateDerivativeOfValueWRTCoefficient(v.geometry().corner(0),
                                                                    partialDerivative,  // select the Lagrange node
                                                                    derivative);

    Dune::FieldMatrix<double,2,3> basis = localGeodesicFEFunction.coefficient(partialDerivative).orthonormalFrame();

    derivative.mtv(basis[0], variation0[gridView.indexSet().index(v)]);
    derivative.mtv(basis[1], variation1[gridView.indexSet().index(v)]);
  }

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

  typedef Functions::PQ1NodalBasis<typename DeformedGridType::LeafGridView > FEBasis;
  FEBasis feBasis(deformedGrid.leafGridView());

  Functions::DiscreteScalarGlobalBasisFunction<decltype(feBasis),decltype(variation0)> variation0Function(feBasis,variation0);
  Functions::DiscreteScalarGlobalBasisFunction<decltype(feBasis),decltype(variation1)> variation1Function(feBasis,variation1);
  auto localVariation0Function = localFunction(variation0Function);
  auto localVariation1Function = localFunction(variation1Function);

  Dune::VTKWriter<typename DeformedGridType::LeafGridView> vtkWriter(deformedGrid.leafGridView());
  vtkWriter.addCellData(colors, "colors");

  vtkWriter.addVertexData(localVariation0Function, VTK::FieldInfo("variation 0", VTK::FieldInfo::Type::scalar, variation0[0].size()));
  vtkWriter.addVertexData(localVariation1Function, VTK::FieldInfo("variation 1", VTK::FieldInfo::Type::scalar, variation1[0].size()));

  vtkWriter.write("sphere-patch-" + nameSuffix);
}

int main(int argc, char* argv[]) try
{
  static const int dim = 2;
  typedef UnitVector<double,3> TargetSpace;

  //////////////////////////////////////////////////////////////
  //   Set up a first-order interpolation function on the sphere
  //////////////////////////////////////////////////////////////

  std::vector<TargetSpace> coefficients(3);

  coefficients[0] = TargetSpace(FieldVector<double,3>({1,0,0}));
  coefficients[1] = TargetSpace(FieldVector<double,3>({0,1,0}));
  coefficients[2] = TargetSpace(FieldVector<double,3>({0,0,1}));

  typedef LocalGeodesicFEFunction<dim, double, P1LocalFiniteElement<double,double,dim>, TargetSpace> P1LocalGFEFunctionType;
  //typedef GFE::LocalProjectedFEFunction<dim, double, LocalFiniteElement, TargetSpace> LocalProjectedFEFunctionType;
  //typedef GFE::LocalTangentFEFunction<dim, double, LocalFiniteElement, TargetSpace> LocalTangentFEFunctionType;

  P1LocalFiniteElement<double,double,dim> localFiniteElement;
  P1LocalGFEFunctionType localGeodesicFEFunction(localFiniteElement,coefficients);
  interpolate<TargetSpace, P1LocalGFEFunctionType>(localGeodesicFEFunction, 0, "riemannian-p1");

  //interpolate<TargetSpace,LocalProjectedFEFunctionType>(coefficients, "projected");
  //interpolate<TargetSpace,LocalTangentFEFunctionType>(coefficients, "tangent");

  //////////////////////////////////////////////////////////////
  //   Set up a second-order interpolation function on the sphere
  //////////////////////////////////////////////////////////////

  coefficients.resize(6);

  coefficients[0] = TargetSpace(FieldVector<double,3>({1,0,0}));
  coefficients[1] = TargetSpace(FieldVector<double,3>({0.5,0.5,0.15}));
  coefficients[2] = TargetSpace(FieldVector<double,3>({0,1,0}));
  coefficients[3] = TargetSpace(FieldVector<double,3>({0.5,0.15,0.5}));
  coefficients[4] = TargetSpace(FieldVector<double,3>({0.15,0.5,0.5}));
  coefficients[5] = TargetSpace(FieldVector<double,3>({0,0,1}));

  typedef LocalGeodesicFEFunction<dim, double, P2LocalFiniteElement<double,double,dim>, TargetSpace> P2LocalGFEFunctionType;
  //typedef GFE::LocalProjectedFEFunction<dim, double, LocalFiniteElement, TargetSpace> LocalProjectedFEFunctionType;
  //typedef GFE::LocalTangentFEFunction<dim, double, LocalFiniteElement, TargetSpace> LocalTangentFEFunctionType;

  P2LocalFiniteElement<double,double,dim> p2LocalFiniteElement;
  P2LocalGFEFunctionType p2LocalGeodesicFEFunction(p2LocalFiniteElement,coefficients);
  interpolate<TargetSpace, P2LocalGFEFunctionType>(p2LocalGeodesicFEFunction, 0, "riemannian-p2-vertex");
  interpolate<TargetSpace, P2LocalGFEFunctionType>(p2LocalGeodesicFEFunction, 1, "riemannian-p2-edge");

  //interpolate<TargetSpace,LocalProjectedFEFunctionType>(coefficients, "projected");
  //interpolate<TargetSpace,LocalTangentFEFunctionType>(coefficients, "tangent");

} catch (Exception e) {

    std::cout << e << std::endl;

}
