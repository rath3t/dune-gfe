#include <iostream>
#include <fstream>

#include <config.h>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>    // use of "Easy to Use" drivers
#include <adolc/taping.h>

#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/common/version.hh>

#if DUNE_VERSION_GTE(DUNE_ELASTICITY, 2, 11)
#include <dune/elasticity/densities/stvenantkirchhoffdensity.hh>
#else
#include <dune/elasticity/materials/stvenantkirchhoffdensity.hh>
#endif

#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/gfe/filereader.hh>
#include <dune/gfe/assemblers/surfacecosseratstressassembler.hh>
#include <dune/gfe/functions/localgeodesicfefunction.hh>
#include <dune/gfe/spaces/rotation.hh>

#include <dune/matrix-vector/transpose.hh>

// grid dimension
#ifndef WORLD_DIM
#  define WORLD_DIM 3
#endif
const int dim = WORLD_DIM;

const int displacementOrder = 2;
const int rotationOrder = 2;

using namespace Dune;
using ValueType = adouble;


// Surface Shell Boundary
std::function<bool(FieldVector<double,dim>)> isSurfaceShellBoundary = [](FieldVector<double,dim> coordinate) {
                                                                        return coordinate[2] > 199.99 and coordinate[0] > 49.99 and coordinate[0] < 150.01;
                                                                      };

static bool sameEntries(FieldVector<double,3> a,FieldVector<double,3> b) {
  b[1] *= -1;
  b -= a;
  //If a.two_norm > 0, so if there is a displacement at this points: Scale the difference with the length of the vectors
  return a.two_norm() > 0 ? b.two_norm()/a.two_norm() < 0.05 : b.two_norm() < 0.05;
}

static bool sameEntries(FieldVector<double,4> a,FieldVector<double,4> b) {
  GFE::Rotation<double,dim> rotationA(a);
  GFE::Rotation<double,dim> rotationB(b);
  rotationA.mult(rotationB);
  GFE::Rotation<double,dim> identity;
  auto distance = GFE::Rotation<double,dim>::distance(rotationA, identity);
  return distance < 0.01;
}

template <int d>
static bool symmetryTest(std::unordered_map<std::string, FieldVector<double,d> > map, double axis) {
  bool isSame = true;
  std::string entry;
  for (auto it = map.begin(); it != map.end(); it++) {
    auto stringKey = it->first;
    std::stringstream entries(stringKey);
    FieldVector<double,dim> coordinate(0);
    int j = 0;
    while(entries >> entry)
      coordinate[j++] = std::stod(entry);
    if (coordinate[1] < axis) {
      coordinate[1] -= 2*axis;
      coordinate[1] *= -1;
      std::stringstream stream;
      stream << coordinate; //modified coordinate
      if (map.count(stream.str()) > 0 && isSurfaceShellBoundary(coordinate)) {
        bool thisIsSame = sameEntries(it->second, map.at(stream.str()));
        isSame = isSame && thisIsSame;
      }
    }
  }
  return isSame;
}

int main (int argc, char *argv[])
{
  MPIHelper::instance(argc, argv);

  /////////////////////////////////////////////////////////////
  //                      CREATE THE GRID
  /////////////////////////////////////////////////////////////
  typedef UGGrid<dim> GridType;

  std::shared_ptr<GridType> grid = StructuredGridFactory<GridType>::createCubeGrid({0,0,0}, {200, 60, 200}, {10,3,5});
  grid->setRefinementType(GridType::RefinementType::COPY);

  int numLevels = 4;

  while (numLevels > 0) {
    for (auto&& e : elements(grid->leafGridView())) {
      bool isSurfaceShell = false;
      for (int i = 0; i < e.geometry().corners(); i++) {
        isSurfaceShell = isSurfaceShell || isSurfaceShellBoundary(e.geometry().corner(i));
      }
      grid->mark(isSurfaceShell ? 1 : 0,e);
    }

    grid->adapt();

    numLevels--;
  }
  typedef GridType::LeafGridView GridView;
  GridView gridView = grid->leafGridView();

  /////////////////////////////////////////////////////////////
  //                   SURFACE SHELL BOUNDARY
  /////////////////////////////////////////////////////////////

  const GridView::IndexSet& indexSet = gridView.indexSet();
  BitSetVector<1> surfaceShellVertices(gridView.size(dim), false);
  for (auto&& v : vertices(gridView))
  {
    bool isSurfaceShell = isSurfaceShellBoundary(v.geometry().corner(0));
    surfaceShellVertices[indexSet.index(v)] = isSurfaceShell;
  }
  BoundaryPatch<GridView> surfaceShellBoundary(gridView, surfaceShellVertices);

  std::function<Dune::FieldVector<double,2>(Dune::FieldVector<double,dim>)> fLame = [](Dune::FieldVector<double,dim> x){
                                                                                      Dune::FieldVector<double,2> lameConstants {1.24E+07,2.52E+07}; //stiffness = 33
                                                                                      return lameConstants;
                                                                                    };

  /////////////////////////////////////////////////////////////
  //                      FUNCTION SPACE
  /////////////////////////////////////////////////////////////

  using namespace Functions::BasisFactory;
  auto basisOrderD = makeBasis(
    gridView,
    power<dim>(
      lagrange<displacementOrder>()
      ));

  auto basisOrderR = makeBasis(
    gridView,
    power<dim>(
      lagrange<rotationOrder>()
      ));

  /////////////////////////////////////////////////////////////
  //               READ IN TEST DATA
  /////////////////////////////////////////////////////////////

  auto deformationMap = GFE::transformFileToMap<dim>("./stressPlotData/stressPlotTestDeformation");
  auto initialDeformationMap = GFE::transformFileToMap<dim>("./stressPlotData/stressPlotTestInitialDeformation");
  const auto dimRotation = GFE::Rotation<double,dim>::embeddedDim;
  auto rotationMap = GFE::transformFileToMap<dimRotation>("./stressPlotData/stressPlotTestRotation");

  bool deformationIsSymmetric = symmetryTest<dim>(deformationMap, 30);
  bool rotationIsSymmetric = symmetryTest<dimRotation>(rotationMap, 30);

  if (!deformationIsSymmetric) {
    std::cerr << "The stressAssemblerTest checking for symmetry only works with a symmetric deformation input file! Please check the file for symmetry!" << std::endl;
    return 1;
  }
  if (!rotationIsSymmetric) {
    std::cout << "The rotation is not symmetric, but that is still ok!" << std::endl;
  }

  using DisplacementVector = std::vector<FieldVector<double,dim> >;
  DisplacementVector x;
  x.resize(basisOrderD.size());
  DisplacementVector xInitial;
  xInitial.resize(basisOrderD.size());
  DisplacementVector displacement;
  displacement.resize(basisOrderD.size());

  Functions::interpolate(basisOrderD, x, [](FieldVector<double,dim> x){
    return x;
  });
  Functions::interpolate(basisOrderD, xInitial, [](FieldVector<double,dim> x){
    return x;
  });

  for (std::size_t i = 0; i < basisOrderD.size(); i++) {
    std::stringstream stream;
    stream << x[i];
    //Look up the displacement for this vertex in the deformationMap
    displacement[i] = deformationMap.at(stream.str());
    x[i] += deformationMap.at(stream.str());
    xInitial[i] += initialDeformationMap.at(stream.str());
  }

  using RotationVector = std::vector<GFE::Rotation<double,dim> >;
  RotationVector rot;
  rot.resize(basisOrderR.size());
  DisplacementVector xOrderR;
  xOrderR.resize(basisOrderR.size());
  Functions::interpolate(basisOrderR, xOrderR, [](FieldVector<double,dim> x){
    return x;
  });
  for (std::size_t i = 0; i < basisOrderR.size(); i++) {
    std::stringstream stream;
    stream << xOrderR[i];
    GFE::Rotation<double,dim> rotation(rotationMap.at(stream.str()));
    FieldMatrix<double,dim,dim> rotationMatrix(0);
    rotation.matrix(rotationMatrix);
    rot[i].set(rotationMatrix);
  }

  MultipleCodimMultipleGeomTypeMapper<GridView> elementMapper(basisOrderD.gridView(),mcmgElementLayout());
  std::vector<int> indicesOfMirroredElements(0);
  indicesOfMirroredElements.resize(elementMapper.size());

  std::unordered_map<std::string, int> elementCenterMirrorMap;
  static constexpr auto partitionType = Partitions::interiorBorder;
  for (const auto& element : elements(basisOrderD.gridView(), partitionType)) {
    std::stringstream stream;
    stream << element.geometry().center();
    elementCenterMirrorMap.insert({stream.str(), elementMapper.index(element)});
  }

  /////////////////////////////////////////////////////////////
  //           STRESS ASSEMBLER
  /////////////////////////////////////////////////////////////

  auto quadOrder = 4;

  const Functions::LagrangeBasis<GridView,rotationOrder> scalarBasisR(gridView);

  using LocalGFEFunctionR = GFE::LocalGeodesicFEFunction<decltype(scalarBasisR),GFE::Rotation<double,dim> >;

  LocalGFEFunctionR localGFEFunction(scalarBasisR);

  auto stressAssembler = GFE::SurfaceCosseratStressAssembler<decltype(basisOrderD),
      decltype(basisOrderR),
      LocalGFEFunctionR,
      FieldVector<double,dim>,
      GFE::Rotation<double,dim> >
                           (basisOrderD, basisOrderR);

  std::shared_ptr<Elasticity::LocalDensity<dim,ValueType> > elasticDensity;
  ParameterTree materialParameters;
  materialParameters["mu"] = "2.7191e+4";
  materialParameters["lambda"] = "4.4364e+4";
  elasticDensity = std::make_shared<Elasticity::StVenantKirchhoffDensity<dim,ValueType> >(materialParameters);

  std::vector<FieldMatrix<double,dim,dim> > stressSubstrate1stPiolaKirchhoffTensor;
  std::vector<FieldMatrix<double,dim,dim> > stressSubstrateCauchyTensor;
  stressAssembler.assembleSubstrateStress<Elasticity::LocalDensity<dim,ValueType> >(x, elasticDensity.get(), quadOrder, stressSubstrate1stPiolaKirchhoffTensor, stressSubstrateCauchyTensor);
  std::vector<FieldMatrix<double,dim,dim> > stressShellBiotTensor;
  stressAssembler.assembleShellStress(localGFEFunction, rot, x, xInitial, fLame,/*mu_c*/ 0, surfaceShellBoundary, quadOrder, stressShellBiotTensor);

  //Now modify ONE value in the rotation function
  int i = 39787;
  FieldMatrix<double,dim,dim> rotationMatrix(0);
  FieldMatrix<double,dim,dim> transposed(0);
  rot[i].matrix(rotationMatrix);
  Dune::MatrixVector::transpose(transposed, rotationMatrix);
  rot[i].set(transposed);
  std::vector<FieldMatrix<double,dim,dim> > stressShellBiotTensorNotSymmetric;
  stressAssembler.assembleShellStress(localGFEFunction, rot, x, xInitial, fLame,/*mu_c*/ 0, surfaceShellBoundary, quadOrder, stressShellBiotTensorNotSymmetric);
  // ... and ONE in the displacement function
  x[i] *= 2;
  std::vector<FieldMatrix<double,dim,dim> > stressSubstrate1stPiolaKirchhoffTensorNotSymmetric;
  std::vector<FieldMatrix<double,dim,dim> > stressSubstrateCauchyTensorNotSymmetric;
  stressAssembler.assembleSubstrateStress<Elasticity::LocalDensity<dim,ValueType> >(x, elasticDensity.get(), quadOrder, stressSubstrate1stPiolaKirchhoffTensorNotSymmetric, stressSubstrateCauchyTensorNotSymmetric);
  // ... and make sure that the stress is not symmetric anymore!
  bool substrateModifiedIsNotSymmetric = false;
  bool shellModifiedIsNotSymmetric = false;

  auto axis = 30; //test for symmetry for the plane at y=30

  for (const auto& element : elements(basisOrderD.gridView(), partitionType)) {
    auto elementCenter = element.geometry().center();
    elementCenter[1] -= 2*axis;
    elementCenter[1] *= -1;

    int thisIndex = elementMapper.index(element);
    std::stringstream stream;
    stream << elementCenter;
    int mirroredIndex = elementCenterMirrorMap.at(stream.str());

    auto substrate = stressSubstrate1stPiolaKirchhoffTensor[thisIndex];
    auto mirroredSubstrate = stressSubstrate1stPiolaKirchhoffTensor[mirroredIndex];
    if ((substrate.frobenius_norm() - mirroredSubstrate.frobenius_norm())/mirroredSubstrate.frobenius_norm() > 0.05) {
      std::cerr << "The elements with center " << element.geometry().center()
                << " and " << elementCenter << " have different 1st-Piola-Kirchhoff stress values (assembled using order 3): " << std::endl
                << substrate << " and " << mirroredSubstrate << ", but should have the same!" << std::endl;
      return 1;
    }

    auto shell = stressShellBiotTensor[thisIndex];
    auto mirroredShell = stressShellBiotTensor[mirroredIndex];
    if ((shell.frobenius_norm()-mirroredShell.frobenius_norm())/mirroredShell.frobenius_norm() > 0.05) {
      std::cerr << "The elements with center " << element.geometry().center()
                << " and " << elementCenter << " have different Biot stress values (assembled using order 3): " << std::endl
                << shell << " and " << mirroredShell << ", but should have the same!" << std::endl;
      return 1;
    }

    auto substrateNotSymmetric = stressSubstrate1stPiolaKirchhoffTensorNotSymmetric[thisIndex];
    auto mirroredSubstrateNotSymmetric = stressSubstrate1stPiolaKirchhoffTensorNotSymmetric[mirroredIndex];
    substrateNotSymmetric -= mirroredSubstrateNotSymmetric;
    substrateModifiedIsNotSymmetric = substrateModifiedIsNotSymmetric or substrateNotSymmetric.frobenius_norm()/mirroredSubstrateNotSymmetric.frobenius_norm() > 0.05;

    auto shellNotSymmetric = stressShellBiotTensorNotSymmetric[thisIndex];
    auto mirroredShellNotSymmetric = stressShellBiotTensorNotSymmetric[mirroredIndex];
    shellNotSymmetric -= mirroredShellNotSymmetric;
    shellModifiedIsNotSymmetric = shellModifiedIsNotSymmetric or shellNotSymmetric.frobenius_norm()/mirroredShellNotSymmetric.frobenius_norm() > 0.05;
  }

  if (substrateModifiedIsNotSymmetric and shellModifiedIsNotSymmetric)
    return 0;

  std::cerr << "The modified functions still returned symmetric stress values!" << std::endl;
  return 1;
}
