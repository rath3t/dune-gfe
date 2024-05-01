#include <config.h>

#include <array>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/typetraits.hh>
#include <dune/common/bitsetvector.hh>
#include <dune/common/tuplevector.hh>

#include <dune/functions/functionspacebases/compositebasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>

#include <dune/fufem/boundarypatch.hh>

#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/grid/uggrid.hh>

#include <dune/gfe/assemblers/cosseratenergystiffness.hh>
#include <dune/gfe/assemblers/geodesicfeassembler.hh>
#include <dune/gfe/assemblers/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/assemblers/mixedgfeassembler.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>

#include <dune/gfe/assemblers/geodesicfeassemblerwrapper.hh>

#include <dune/istl/multitypeblockmatrix.hh>
#include <dune/istl/multitypeblockvector.hh>

// grid dimension
const int gridDim = 2;

// target dimension
const int dim = 3;

//order of the finite element space
const int displacementOrder = 2;
const int rotationOrder = 2;

using namespace Dune;
using namespace Indices;


//Types for the mixed space
using DisplacementVector = std::vector<RealTuple<double,dim> >;
using RotationVector =  std::vector<Rotation<double,dim> >;
using Vector = TupleVector<DisplacementVector, RotationVector>;
const int dimCR = Rotation<double,dim>::TangentVector::dimension; //dimCorrectionRotation = Dimension of the correction for rotations
using CorrectionType = MultiTypeBlockVector<BlockVector<FieldVector<double,dim> >, BlockVector<FieldVector<double,dimCR> > >;

using MatrixRow0 = MultiTypeBlockVector<BCRSMatrix<FieldMatrix<double,dim,dim> >,  BCRSMatrix<FieldMatrix<double,dim,dimCR> > >;
using MatrixRow1 = MultiTypeBlockVector<BCRSMatrix<FieldMatrix<double,dimCR,dim> >, BCRSMatrix<FieldMatrix<double,dimCR,dimCR> > >;
using MatrixType = MultiTypeBlockMatrix<MatrixRow0,MatrixRow1>;

//Types for the Non-mixed space
using RBM = GFE::ProductManifold<RealTuple<double,dim>,Rotation<double, dim> >;
const static int blocksize = RBM::TangentVector::dimension;
using CorrectionTypeWrapped = BlockVector<FieldVector<double, blocksize> >;
using MatrixTypeWrapped = BCRSMatrix<FieldMatrix<double, blocksize, blocksize> >;

int main (int argc, char *argv[])
{
  MPIHelper::instance(argc, argv);

  /////////////////////////////////////////////////////////////////////////
  //    Create the grid
  /////////////////////////////////////////////////////////////////////////
  using GridType = UGGrid<gridDim>;
  auto grid = StructuredGridFactory<GridType>::createCubeGrid({0,0}, {1,1}, {2,2});
  grid->globalRefine(2);
  grid->loadBalance();

  using GridView = GridType::LeafGridView;
  GridView gridView = grid->leafGridView();

  std::function<bool(FieldVector<double,gridDim>)> isNeumann = [](FieldVector<double,gridDim> coordinate) {
                                                                 return coordinate[0] > 0.99;
                                                               };

  BitSetVector<1> neumannVertices(gridView.size(gridDim), false);
  const GridView::IndexSet& indexSet = gridView.indexSet();

  for (auto&& vertex : vertices(gridView))
    neumannVertices[indexSet.index(vertex)] = isNeumann(vertex.geometry().corner(0));

  BoundaryPatch<GridView> neumannBoundary(gridView, neumannVertices);


  /////////////////////////////////////////////////////////////////////////
  //  Create a composite basis
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
  parameters["b1"] = "1";
  parameters["b2"] = "1";
  parameters["b3"] = "1";


  FieldVector<double,dim> values_ = {3e4,2e4,1e4};
  auto neumannFunction = [&](FieldVector<double, gridDim>){
                           return values_;
                         };

  auto cosseratEnergy = std::make_shared<CosseratEnergyLocalStiffness<decltype(compositeBasis), dim,adouble> >(parameters,
                                                                                                               &neumannBoundary,
                                                                                                               neumannFunction,
                                                                                                               nullptr);
  LocalGeodesicFEADOLCStiffness<CompositeBasis,
      GFE::ProductManifold<RealTuple<double,dim>,Rotation<double,dim> > > mixedLocalGFEADOLCStiffness(cosseratEnergy);
  MixedGFEAssembler<CompositeBasis,RBM> mixedAssembler(compositeBasis, mixedLocalGFEADOLCStiffness);

  using DeformationFEBasis = Functions::LagrangeBasis<GridView,displacementOrder>;
  DeformationFEBasis deformationFEBasis(gridView);
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
  for (std::size_t i = 0; i < compositeBasis.size({0}); i++) {
    for (int j = 0; j < gridDim; j++)
      initialDeformation[i][j] = identity[i][j];
    x[_0][i] = initialDeformation[i];
    xRBM[i][_0] = x[_0][i];
    xRBM[i][_1] = x[_1][i]; // Rotation part
  }

  //////////////////////////////////////////////////////////////////////////////
  //  Compute the energy, assemble the Gradient and Hessian using
  //  the GeodesicFEAssemblerWrapper and the MixedGFEAssembler and compare!
  //////////////////////////////////////////////////////////////////////////////
  CorrectionTypeWrapped gradient;
  MatrixTypeWrapped hessianMatrix;
  double energy = assembler.computeEnergy(xRBM);
  assembler.assembleGradientAndHessian(xRBM, gradient, hessianMatrix, true);
  double gradientTwoNorm = gradient.two_norm();
  double gradientInfinityNorm = gradient.infinity_norm();
  double matrixFrobeniusNorm = hessianMatrix.frobenius_norm();

  CorrectionType gradientMixed;
  gradientMixed[_0].resize(x[_0].size());
  gradientMixed[_1].resize(x[_1].size());
  MatrixType hessianMatrixMixed;
  double energyMixed = mixedAssembler.computeEnergy(x[_0], x[_1]);
  mixedAssembler.assembleGradientAndHessian(x[_0], x[_1], gradientMixed[_0], gradientMixed[_1], hessianMatrixMixed, true);
  double gradientMixedTwoNorm = gradientMixed.two_norm();
  double gradientMixedInfinityNorm = gradientMixed.infinity_norm();
  double matrixMixedFrobeniusNorm = hessianMatrixMixed.frobenius_norm();

  if (std::abs(energy - energyMixed)/energyMixed > 1e-8)
  {
    std::cerr << std::setprecision(9);
    std::cerr << "The energy calculated by the GeodesicFEAssemblerWrapper is " << energy << " but "
              << energyMixed << " (calculated by the MixedGFEAssembler) was expected!" << std::endl;
    return 1;
  }
  if ( std::abs(gradientTwoNorm - gradientMixedTwoNorm)/gradientMixedTwoNorm > 1e-8 ||
       std::abs(gradientInfinityNorm - gradientMixedInfinityNorm)/gradientMixedInfinityNorm > 1e-8)
  {
    std::cerr << std::setprecision(9);
    std::cerr << "The gradient infinity norm calculated by the GeodesicFEAssemblerWrapper is " << gradientInfinityNorm << " but "
              << gradientMixedInfinityNorm << " (calculated by the MixedGFEAssembler) was expected!" << std::endl;
    std::cerr << "The gradient norm calculated by the GeodesicFEAssemblerWrapper is " << gradientTwoNorm << " but "
              << gradientMixedTwoNorm << " (calculated by the MixedGFEAssembler) was expected!" << std::endl;
    return 1;
  }

  if (std::abs(matrixFrobeniusNorm - matrixMixedFrobeniusNorm)/matrixMixedFrobeniusNorm > 1e-8)
  {
    std::cerr << std::setprecision(9);
    std::cerr << "The matrix norm calculated by the GeodesicFEAssemblerWrapper is " << matrixFrobeniusNorm << " but "
              << matrixMixedFrobeniusNorm << " (calculated by the MixedGFEAssembler) was expected!" << std::endl;
    return 1;
  }
}
