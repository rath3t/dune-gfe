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
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>

#include <dune/fufem/boundarypatch.hh>

#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/grid/uggrid.hh>

#include <dune/gfe/cosseratenergystiffness.hh>
#include <dune/gfe/geodesicfeassembler.hh>
#include <dune/gfe/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/mixedgfeassembler.hh>
#include <dune/gfe/mixedlocalgfeadolcstiffness.hh>

#include <dune/istl/multitypeblockmatrix.hh>
#include <dune/istl/multitypeblockvector.hh>

// grid dimension
const int gridDim = 2;

// target dimension
const int dim = 3;

using namespace Dune;
using namespace Indices;

//differentiation method: ADOL-C
using ValueType = adouble;

//Types for the mixed space
using DisplacementVector = std::vector<RealTuple<double,dim>>;
using RotationVector =  std::vector<Rotation<double,dim>>;
using Vector = TupleVector<DisplacementVector, RotationVector>;
const int dimCR = Rotation<double,dim>::TangentVector::dimension; //dimCorrectionRotation = Dimension of the correction for rotations
using CorrectionTypeMixed = MultiTypeBlockVector<BlockVector<FieldVector<double,dim> >, BlockVector<FieldVector<double,dimCR>>>;

using MatrixRow0 = MultiTypeBlockVector<BCRSMatrix<FieldMatrix<double,dim,dim>>,  BCRSMatrix<FieldMatrix<double,dim,dimCR>>>;
using MatrixRow1 = MultiTypeBlockVector<BCRSMatrix<FieldMatrix<double,dimCR,dim>>, BCRSMatrix<FieldMatrix<double,dimCR,dimCR>>>;
using MatrixTypeMixed = MultiTypeBlockMatrix<MatrixRow0,MatrixRow1>;

//Types for the Non-mixed space
using RBM = RigidBodyMotion<double, dim>;
using RBMVector = std::vector<RBM>;
const static int blocksize = RBM::TangentVector::dimension;
using CorrectionType = BlockVector<FieldVector<double, blocksize> >;
using MatrixType = BCRSMatrix<FieldMatrix<double, blocksize, blocksize> >;

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

  /////////////////////////////////////////////////////////////////////////
  //  Create a composite basis
  /////////////////////////////////////////////////////////////////////////

  using namespace Functions::BasisFactory;

  auto compositeBasis = makeBasis(
    gridView,
    composite(
      power<dim>(
        lagrange<2>()
      ),
      power<dim>(
        lagrange<2>()
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

  //Mixed space
  CosseratEnergyLocalStiffness<decltype(compositeBasis), dim,adouble> cosseratEnergyMixed(parameters,
                                                                     nullptr,
                                                                     nullptr,
                                                                     nullptr);
  MixedLocalGFEADOLCStiffness<CompositeBasis,
                              RealTuple<double,dim>,
                              Rotation<double,dim> > mixedLocalGFEADOLCStiffnessVector(&cosseratEnergyMixed, false);
  MixedGFEAssembler<CompositeBasis,
                    RealTuple<double,dim>,
                    Rotation<double,dim> > mixedAssemblerVector(compositeBasis, &mixedLocalGFEADOLCStiffnessVector);

  MixedLocalGFEADOLCStiffness<CompositeBasis,
                              RealTuple<double,dim>,
                              Rotation<double,dim> > mixedLocalGFEADOLCStiffnessScalar(&cosseratEnergyMixed, true);
  MixedGFEAssembler<CompositeBasis,
                    RealTuple<double,dim>,
                    Rotation<double,dim> > mixedAssemblerScalar(compositeBasis, &mixedLocalGFEADOLCStiffnessScalar);
  
  //Non-mixed space
  using DeformationFEBasis = Dune::Functions::LagrangeBasis<GridView,2>;

  CosseratEnergyLocalStiffness<DeformationFEBasis, dim,adouble> cosseratEnergy(parameters,
                                                                     nullptr,
                                                                     nullptr,
                                                                     nullptr);

  LocalGeodesicFEADOLCStiffness<DeformationFEBasis,RBM> localGFEADOLCStiffnessVector(&cosseratEnergy, false);
  GeodesicFEAssembler<DeformationFEBasis,RBM> assemblerVector(gridView, localGFEADOLCStiffnessVector);

  LocalGeodesicFEADOLCStiffness<DeformationFEBasis,RBM> localGFEADOLCStiffnessScalar(&cosseratEnergy, true);
  GeodesicFEAssembler<DeformationFEBasis,RBM> assemblerScalar(gridView, localGFEADOLCStiffnessScalar);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //  Prepare the iterate x where we want to assemble - cos(identity) in 2D with z = x^2
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  auto deformationPowerBasis = makeBasis(
    gridView,
    power<gridDim>(
        lagrange<2>()
  ));
  BlockVector<FieldVector<double,gridDim> > identity(compositeBasis.size({0}));
  Functions::interpolate(deformationPowerBasis, identity, [](FieldVector<double,gridDim> x){ return x; });
  BlockVector<FieldVector<double,dim> > initialDeformation(compositeBasis.size({0}));
  initialDeformation = 0;

  Vector x;
  RBMVector xRBM;
  xRBM.resize(compositeBasis.size({0}));
  x[_0].resize(compositeBasis.size({0}));
  x[_1].resize(compositeBasis.size({1}));
  for (std::size_t i = 0; i < compositeBasis.size({0}); i++) {
    for (int j = 0; j < gridDim; j++)
      initialDeformation[i][j] = std::cos(identity[i][j]);
    initialDeformation[i][2] = identity[i][0]*identity[i][0];
    x[_0][i] = initialDeformation[i];
    xRBM[i].r = initialDeformation[i];
  }

  //////////////////////////////////////////////////////////////////////////////
  //  Assemble the Hessian using vector-mode and scalar-mode
  //////////////////////////////////////////////////////////////////////////////

  CorrectionTypeMixed gradientMixed;
  gradientMixed[_0].resize(x[_0].size());
  gradientMixed[_1].resize(x[_1].size());

  MatrixTypeMixed hessianMatrixMixedScalar;
  mixedAssemblerScalar.assembleGradientAndHessian(x[_0], x[_1], gradientMixed[_0], gradientMixed[_1], hessianMatrixMixedScalar, true);

  MatrixTypeMixed hessianMatrixMixedVector;
  mixedAssemblerVector.assembleGradientAndHessian(x[_0], x[_1], gradientMixed[_0], gradientMixed[_1], hessianMatrixMixedVector, true);

  auto differenceMixed = hessianMatrixMixedScalar;
  differenceMixed -= hessianMatrixMixedVector;

  CorrectionType gradient;
  gradient.resize(xRBM.size());

  MatrixType hessianMatrixScalar;
  assemblerScalar.assembleGradientAndHessian(xRBM, gradient, hessianMatrixScalar, true);

  MatrixType hessianMatrixVector;
  assemblerVector.assembleGradientAndHessian(xRBM, gradient, hessianMatrixVector, true);

  auto difference = hessianMatrixScalar;
  difference -= hessianMatrixVector;

  if (differenceMixed.frobenius_norm() > 1e-8)
  {
      std::cerr << "MixedLocalGFEADOLCStiffness: The ADOL-C scalar mode and vector mode produce different Hessians!"  << std::endl;
      return 1;
  }
  if (difference.frobenius_norm() > 1e-8)
  {
      std::cerr << "LocalGFEADOLCStiffness: The ADOL-C scalar mode and vector mode produce different Hessians!"  << std::endl;
      return 1;
  }
  return 0;
}
