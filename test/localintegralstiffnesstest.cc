#include <config.h>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <dune/fufem/utilities/adolcnamespaceinjections.hh>


#include <dune/common/parametertree.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/common/timer.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/compositebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/subspacebasis.hh>

#include <dune/gfe/assemblers/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/assemblers/mixedgfeassembler.hh>
#include <dune/gfe/assemblers/sumenergy.hh>
#include <dune/gfe/assemblers/geodesicfeassembler.hh>
#include <dune/gfe/assemblers/localintegralenergy.hh>
#include <dune/gfe/assemblers/localintegralstiffness.hh>
#include <dune/gfe/densities/localdensity.hh>
#include <dune/gfe/densities/bulkcosseratdensity.hh>
#include <dune/gfe/densities/harmonicdensity.hh>
#include <dune/gfe/functions/localprojectedfefunction.hh>
#include <dune/gfe/spaces/realtuple.hh>

// grid dimension
const int dim = 3;

using namespace Dune;
using namespace Dune::Indices;
using namespace Functions::BasisFactory;

enum InterpolationType {Geodesic, ProjectionBased, Nonconforming};


template <class GridView, InterpolationType interpolationType>
int testHarmonicMapIntoSphere(TestSuite& test, const GridView& gridView)
{
  using TargetSpace = GFE::UnitVector<double,dim>;

  // Finite element order
  const int order = 1;

  const static int blocksize = TargetSpace::TangentVector::dimension;

  using Correction = BlockVector<TargetSpace::TangentVector>;
  using Matrix = BCRSMatrix<FieldMatrix<double, blocksize, blocksize> >;

  //////////////////////////////////////////////////////////////////////////////////
  //  Construct all needed function space bases
  //////////////////////////////////////////////////////////////////////////////////

  auto tangentBasis = makeBasis(
    gridView,
    power<blocksize>(
      lagrange<order>()
      )
    );

  auto valueBasis = makeBasis(gridView, power<TargetSpace::CoordinateType::dimension>(lagrange<order>()));

  /////////////////////////////////////////////////////////////////////////
  //  Construct the configuration where to assemble the tangent problem
  /////////////////////////////////////////////////////////////////////////

  using CoefficientVector = std::vector<TargetSpace>;
  CoefficientVector x(tangentBasis.size());

  BlockVector<FieldVector<double,dim> > initialIterate;
  Functions::interpolate(valueBasis, initialIterate, [](FieldVector<double,dim> x)
                         -> FieldVector<double,dim> {
    // Interpret x[0] and x[1] as spherical coordinates
    auto phi = x[0];
    auto theta = x[1];
    return {sin(phi)*cos(theta), sin(phi)*sin(theta), cos(theta)};
  });
  for (size_t i = 0; i < x.size(); i++)
    x[i] = initialIterate[i];

  using ATargetSpace = typename TargetSpace::rebind<adouble>::other;

  // Select which type of geometric interpolation to use
  // TODO: Use the tangent basis here!
  using FEBasis = Functions::LagrangeBasis<GridView,order>;
  Functions::LagrangeBasis<GridView,order> feBasis(gridView);

  //////////////////////////////////////////////////////////////
  //  Set up the two assemblers
  //////////////////////////////////////////////////////////////

  std::shared_ptr<GFE::GeodesicFEAssembler<FEBasis,TargetSpace> > assemblerADOLC;
  std::shared_ptr<GFE::LocalGeodesicFEStiffness<FEBasis,TargetSpace> > localIntegralStiffness;

  using Element = typename GridView::template Codim<0>::Entity;
  auto harmonicDensity = std::make_shared<GFE::HarmonicDensity<Element,TargetSpace> >();
  auto harmonicDensityA = std::make_shared<GFE::HarmonicDensity<Element,ATargetSpace> >();

  if constexpr (interpolationType==Geodesic)
  {
    std::cout << "Using geodesic interpolation" << std::endl;
    using LocalInterpolationRule = GFE::LocalGeodesicFEFunction<dim,
        typename GridView::ctype,
        decltype(feBasis.localView().tree().finiteElement()),
        TargetSpace>;

    using ALocalInterpolationRule = GFE::LocalGeodesicFEFunction<dim,
        typename GridView::ctype,
        decltype(feBasis.localView().tree().finiteElement()),
        ATargetSpace>;

    // Assemble using the old assembler
    auto energy = std::make_shared<GFE::LocalIntegralEnergy<FEBasis, ALocalInterpolationRule,ATargetSpace> >(harmonicDensityA);

    auto localGFEADOLCStiffness = std::make_shared<GFE::LocalGeodesicFEADOLCStiffness<FEBasis,TargetSpace> >(energy);
    assemblerADOLC = std::make_shared<GFE::GeodesicFEAssembler<FEBasis,TargetSpace> >(feBasis, localGFEADOLCStiffness);

    // Assemble using the new assembler
    localIntegralStiffness = std::make_shared<GFE::LocalIntegralStiffness<FEBasis,LocalInterpolationRule,TargetSpace> >(harmonicDensity);
  }
  else
  {
    if (interpolationType==ProjectionBased)
      std::cout << "Using projection-based interpolation" << std::endl;
    else
      std::cout << "Using nonconforming interpolation" << std::endl;

    using LocalInterpolationRule = GFE::LocalProjectedFEFunction<dim,
        typename GridView::ctype,
        decltype(feBasis.localView().tree().finiteElement()),
        TargetSpace, interpolationType!=Nonconforming>;

    using ALocalInterpolationRule = GFE::LocalProjectedFEFunction<dim,
        typename GridView::ctype,
        decltype(feBasis.localView().tree().finiteElement()),
        ATargetSpace, interpolationType!=Nonconforming>;

    // Assemble using the old assembler
    auto energy = std::make_shared<GFE::LocalIntegralEnergy<FEBasis, ALocalInterpolationRule,ATargetSpace> >(harmonicDensityA);

    auto localGFEADOLCStiffness = std::make_shared<GFE::LocalGeodesicFEADOLCStiffness<FEBasis,TargetSpace> >(energy);
    assemblerADOLC = std::make_shared<GFE::GeodesicFEAssembler<FEBasis,TargetSpace> >(feBasis, localGFEADOLCStiffness);

    // Assemble using the new assembler
    localIntegralStiffness = std::make_shared<GFE::LocalIntegralStiffness<FEBasis,LocalInterpolationRule,TargetSpace> >(harmonicDensity);
  }

  // TODO: The assembler should really get the tangent basis.
  auto assemblerIntegralStiffness = std::make_shared<GFE::GeodesicFEAssembler<FEBasis,TargetSpace> >(feBasis, localIntegralStiffness);

  //////////////////////////////////////////////////////////////
  //  Assemble
  //////////////////////////////////////////////////////////////

  Correction gradient;
  Matrix hesseMatrix;

  Correction gradientSmart;
  Matrix hesseMatrixSmart;

  Timer assemblerTimer;
  assemblerADOLC->assembleGradientAndHessian(x, gradient, hesseMatrix,true);
  std::cout << "assemblerTimer took: " << assemblerTimer.elapsed() << std::endl;

  Timer assemblerSmartTimer;
  assemblerIntegralStiffness->assembleGradientAndHessian(x, gradientSmart, hesseMatrixSmart,true);
  std::cout << "assemblerSmartTimer took: " << assemblerSmartTimer.elapsed() << std::endl;

  //////////////////////////////////////////////////////////////
  //  Compare the results
  //////////////////////////////////////////////////////////////

  double gradientInfinityNorm = gradient.infinity_norm();
  double matrixFrobeniusNorm = hesseMatrix.frobenius_norm();

  double gradientSmartInfinityNorm = gradientSmart.infinity_norm();
  double matrixSmartFrobeniusNorm = hesseMatrixSmart.frobenius_norm();

  if (std::isnan(gradientInfinityNorm) || std::isnan(gradientSmartInfinityNorm) || std::abs(gradientInfinityNorm - gradientSmartInfinityNorm)/gradientInfinityNorm > 1e-3)
  {
    std::cerr << std::setprecision(9);
    std::cerr << "Gradient max norm from localADOLCStiffness is " << gradientInfinityNorm
              << ", from LocalIntegralStiffness is " << gradientSmartInfinityNorm << " :(" << std::endl;
    return 1;
  }

  if (std::isnan(matrixFrobeniusNorm) || std::isnan(matrixSmartFrobeniusNorm) || std::abs(matrixFrobeniusNorm - matrixSmartFrobeniusNorm)/matrixFrobeniusNorm > 1e-3)
  {
    std::cerr << std::setprecision(9);
    std::cerr << "Matrix norm from localADOLCStiffness is " << matrixFrobeniusNorm
              << ", from LocalIntegralStiffness is " << matrixSmartFrobeniusNorm << " :(" << std::endl;
    return 1;
  }

  // TODO: Use rangeFor
  for (auto rowIt = hesseMatrixSmart.begin(); rowIt != hesseMatrixSmart.end(); ++rowIt)
    for (auto colIt = rowIt->begin(); colIt != rowIt->end(); ++colIt)
      for (size_t k = 0; k < blocksize; k++)
        for (size_t l = 0; l < blocksize; l++)
          if (std::abs((*colIt)[k][l]) > 1e-6 and std::abs(hesseMatrix[rowIt.index()][colIt.index()][k][l]) > 1e-6
              and std::abs((*colIt)[k][l] - hesseMatrix[rowIt.index()][colIt.index()][k][l])/(*colIt)[k][l] > 1e-4) {
            std::cerr << "Matrix: Index " << " " << rowIt.index() << " " << colIt.index() << " " << k << " " << l  << " is wrong: " << hesseMatrix[rowIt.index()][colIt.index()][k][l] << " - " << (*colIt)[k][l] << std::endl;
            return 1;
          }

  return 0;
}



template <class GridView, InterpolationType interpolationType>
int testCosseratBulkModel(TestSuite& test, const GridView& gridView)
{
  // Order of the approximation spaces
  // Deliberately test with different orders, to find more bugs
  const int deformationOrder = 2;
  const int rotationOrder = 1;

  const static int deformationBlocksize = GFE::RealTuple<double,dim>::TangentVector::dimension;
  const static int orientationBlocksize = GFE::Rotation<double,dim>::TangentVector::dimension;

  using CorrectionType0 = BlockVector<FieldVector<double, deformationBlocksize> >;
  using CorrectionType1 = BlockVector<FieldVector<double, orientationBlocksize> >;

  using MatrixType00 = BCRSMatrix<FieldMatrix<double, deformationBlocksize, deformationBlocksize> >;
  using MatrixType01 = BCRSMatrix<FieldMatrix<double, deformationBlocksize, orientationBlocksize> >;
  using MatrixType10 = BCRSMatrix<FieldMatrix<double, orientationBlocksize, deformationBlocksize> >;
  using MatrixType11 = BCRSMatrix<FieldMatrix<double, orientationBlocksize, orientationBlocksize> >;

  using MatrixType = MultiTypeBlockMatrix<MultiTypeBlockVector<MatrixType00,MatrixType01>,
      MultiTypeBlockVector<MatrixType10,MatrixType11> >;

  //////////////////////////////////////////////////////////////////////////////////
  //  Construct all needed function space bases
  //////////////////////////////////////////////////////////////////////////////////

  const int dimRotation = GFE::Rotation<double,dim>::embeddedDim;
  auto compositeBasis = makeBasis(
    gridView,
    composite(
      power<dim>(
        lagrange<deformationOrder>()
        ),
      power<dimRotation>(
        lagrange<rotationOrder>()
        )
      ));

  using CompositeBasis = decltype(compositeBasis);

  /////////////////////////////////////////////////////////////////////////
  //  Construct the configuration where to assemble the tangent problem
  /////////////////////////////////////////////////////////////////////////

  using CoefficientVector = TupleVector<std::vector<GFE::RealTuple<double,dim> >,std::vector<GFE::Rotation<double,dim> > >;
  CoefficientVector x;
  x[_0].resize(compositeBasis.size({0}));
  x[_1].resize(compositeBasis.size({1}));

  MultiTypeBlockVector<BlockVector<FieldVector<double,dim> >,BlockVector<FieldVector<double,dimRotation> > > initialIterate;
  Functions::interpolate(Functions::subspaceBasis(compositeBasis, _0), initialIterate, [](FieldVector<double,dim> x){
    FieldVector<double,dim> y = {0.5*x[0]*x[0], 0.5*x[1] + 0.2*x[0], 4*std::abs(std::sin(x[2]))};
    return y;
  });
  for (size_t i = 0; i < x[_0].size(); i++)
    x[_0][i] = initialIterate[_0][i];

  Functions::interpolate(Functions::subspaceBasis(compositeBasis, _1), initialIterate, [](FieldVector<double,dim> x){
    // Just any rotation field, for testing
    FieldVector<double,4> y = {0.5*x[0], 0.7*x[1], 1.0*std::abs(std::sin(x[2])), 1.0 - 0.2*x[0]*x[1]};
    return y;
  });
  for (size_t i = 0; i < x[_1].size(); i++)
    x[_1][i] = GFE::Rotation<double,dim>(initialIterate[_1][i]);

  // Set of material parameters
  ParameterTree parameters;
  parameters["thickness"] = "0.1";
  parameters["mu"] = "1";
  parameters["lambda"] = "1";
  parameters["mu_c"] = "1";
  parameters["L_c"] = "0.1";
  parameters["curvatureType"] = "wryness";
  parameters["q"] = "2";
  parameters["kappa"] = "1";
  parameters["b1"] = "1";
  parameters["b2"] = "1";
  parameters["b3"] = "1";

  using RigidBodyMotion  = GFE::ProductManifold<GFE::RealTuple<double,dim>, GFE::Rotation<double,dim> >;
  using ARigidBodyMotion = GFE::ProductManifold<GFE::RealTuple<adouble,dim>, GFE::Rotation<adouble,dim> >;

  using Element = typename GridView::template Codim<0>::Entity;

  auto bulkCosseratDensity = std::make_shared<GFE::BulkCosseratDensity<Element,double> >(parameters);
  auto aBulkCosseratDensity = std::make_shared<GFE::BulkCosseratDensity<Element,adouble> >(parameters);

  // Select which type of geometric interpolation to use
  using DeformationFEBasis = Functions::LagrangeBasis<GridView,deformationOrder>;
  using OrientationFEBasis = Functions::LagrangeBasis<GridView,rotationOrder>;

  DeformationFEBasis deformationFEBasis(gridView);
  OrientationFEBasis orientationFEBasis(gridView);

  //////////////////////////////////////////////////////////////
  //  Set up the two assemblers
  //////////////////////////////////////////////////////////////

  std::shared_ptr<GFE::MixedGFEAssembler<CompositeBasis,RigidBodyMotion> > assemblerADOLC;
  std::shared_ptr<GFE::LocalGeodesicFEStiffness<CompositeBasis,RigidBodyMotion> > localIntegralStiffness;

  if constexpr (interpolationType==Geodesic)
  {
    std::cout << "Using geodesic interpolation" << std::endl;

    using LocalDeformationInterpolationRule = GFE::LocalGeodesicFEFunction<dim, typename GridView::ctype, decltype(deformationFEBasis.localView().tree().finiteElement()), GFE::RealTuple<double,dim> >;
    using LocalOrientationInterpolationRule = GFE::LocalGeodesicFEFunction<dim, typename GridView::ctype, decltype(orientationFEBasis.localView().tree().finiteElement()), GFE::Rotation<double,dim> >;

    using LocalInterpolationRule = std::tuple<LocalDeformationInterpolationRule,LocalOrientationInterpolationRule>;

    using ALocalDeformationInterpolationRule = GFE::LocalGeodesicFEFunction<dim, typename GridView::ctype, decltype(deformationFEBasis.localView().tree().finiteElement()), GFE::RealTuple<adouble,dim> >;
    using ALocalOrientationInterpolationRule = GFE::LocalGeodesicFEFunction<dim, typename GridView::ctype, decltype(orientationFEBasis.localView().tree().finiteElement()), GFE::Rotation<adouble,dim> >;

    using ALocalInterpolationRule = std::tuple<ALocalDeformationInterpolationRule,ALocalOrientationInterpolationRule>;

    // Assemble using the ADOL-C assembler
    auto energy = std::make_shared<GFE::LocalIntegralEnergy<CompositeBasis, ALocalInterpolationRule,ARigidBodyMotion> >(aBulkCosseratDensity);

    auto localGFEADOLCStiffness = std::make_shared<GFE::LocalGeodesicFEADOLCStiffness<CompositeBasis,RigidBodyMotion> >(energy);

    assemblerADOLC = std::make_shared<GFE::MixedGFEAssembler<CompositeBasis,RigidBodyMotion> >(compositeBasis, localGFEADOLCStiffness);

    // Assemble using the new assembler
    localIntegralStiffness = std::make_shared<GFE::LocalIntegralStiffness<CompositeBasis,LocalInterpolationRule,RigidBodyMotion> >(bulkCosseratDensity);
  }
  else
  {
    std::cout << "Using projection-based interpolation" << std::endl;

    using LocalDeformationInterpolationRule = GFE::LocalProjectedFEFunction<dim, typename GridView::ctype, decltype(deformationFEBasis.localView().tree().finiteElement()), GFE::RealTuple<double,dim> >;
    using LocalOrientationInterpolationRule = GFE::LocalProjectedFEFunction<dim, typename GridView::ctype, decltype(orientationFEBasis.localView().tree().finiteElement()), GFE::Rotation<double,dim> >;

    using LocalInterpolationRule = std::tuple<LocalDeformationInterpolationRule,LocalOrientationInterpolationRule>;

    using ALocalDeformationInterpolationRule = GFE::LocalProjectedFEFunction<dim, typename GridView::ctype, decltype(deformationFEBasis.localView().tree().finiteElement()), GFE::RealTuple<adouble,dim> >;
    using ALocalOrientationInterpolationRule = GFE::LocalProjectedFEFunction<dim, typename GridView::ctype, decltype(orientationFEBasis.localView().tree().finiteElement()), GFE::Rotation<adouble,dim> >;

    using ALocalInterpolationRule = std::tuple<ALocalDeformationInterpolationRule,ALocalOrientationInterpolationRule>;

    // Assemble using the ADOL-C assembler
    auto energy = std::make_shared<GFE::LocalIntegralEnergy<CompositeBasis, ALocalInterpolationRule,ARigidBodyMotion> >(aBulkCosseratDensity);

    auto localGFEADOLCStiffness = std::make_shared<GFE::LocalGeodesicFEADOLCStiffness<CompositeBasis,RigidBodyMotion> >(energy);

    assemblerADOLC = std::make_shared<GFE::MixedGFEAssembler<CompositeBasis,RigidBodyMotion> >(compositeBasis, localGFEADOLCStiffness);

    // Assemble using the new assembler
    localIntegralStiffness = std::make_shared<GFE::LocalIntegralStiffness<CompositeBasis,LocalInterpolationRule,RigidBodyMotion> >(bulkCosseratDensity);
  }

  GFE::MixedGFEAssembler<CompositeBasis,RigidBodyMotion> mixedAssemblerSmart(compositeBasis,
                                                                        localIntegralStiffness);

  //////////////////////////////////////////////////////////////
  //  Assemble
  //////////////////////////////////////////////////////////////

  CorrectionType0 gradient0;
  CorrectionType1 gradient1;
  MatrixType hesseMatrix;

  CorrectionType0 gradientSmart0;
  CorrectionType1 gradientSmart1;
  MatrixType hesseMatrixSmart;

  Timer assemblerTimer;
  assemblerADOLC->assembleGradientAndHessian(x[_0], x[_1], gradient0, gradient1, hesseMatrix,true);
  std::cout << "assemblerTimer took: " << assemblerTimer.elapsed() << std::endl;

  Timer assemblerSmartTimer;
  mixedAssemblerSmart.assembleGradientAndHessian(x[_0], x[_1], gradientSmart0, gradientSmart1, hesseMatrixSmart,true);
  std::cout << "assemblerSmartTimer took: " << assemblerSmartTimer.elapsed() << std::endl;

  //////////////////////////////////////////////////////////////
  //  Compare the results
  //////////////////////////////////////////////////////////////

  double gradient0InfinityNorm = gradient0.infinity_norm();
  double gradient1InfinityNorm = gradient1.infinity_norm();
  double matrix00FrobeniusNorm = hesseMatrix[_0][_0].frobenius_norm();
  double matrix01FrobeniusNorm = hesseMatrix[_0][_1].frobenius_norm();
  double matrix10FrobeniusNorm = hesseMatrix[_1][_0].frobenius_norm();
  double matrix11FrobeniusNorm = hesseMatrix[_1][_1].frobenius_norm();

  double gradientSmart0InfinityNorm = gradientSmart0.infinity_norm();
  double gradientSmart1InfinityNorm = gradientSmart1.infinity_norm();
  double matrixSmart00FrobeniusNorm = hesseMatrixSmart[_0][_0].frobenius_norm();
  double matrixSmart01FrobeniusNorm = hesseMatrixSmart[_0][_1].frobenius_norm();
  double matrixSmart10FrobeniusNorm = hesseMatrixSmart[_1][_0].frobenius_norm();
  double matrixSmart11FrobeniusNorm = hesseMatrixSmart[_1][_1].frobenius_norm();

  if (std::isnan(gradient0InfinityNorm) || std::isnan(gradientSmart0InfinityNorm) || std::abs(gradient0InfinityNorm - gradientSmart0InfinityNorm)/gradient0InfinityNorm > 1e-3)
  {
    std::cerr << std::setprecision(9);
    std::cerr << "Gradient (deformation part) max norm from localADOLCStiffness is " << gradient0InfinityNorm << ", from localADOLCIntegralStiffness is " << gradientSmart0InfinityNorm << " :(" << std::endl;
    return 1;
  }

  if (std::isnan(gradient1InfinityNorm) || std::isnan(gradientSmart1InfinityNorm) || std::abs(gradient1InfinityNorm - gradientSmart1InfinityNorm)/gradient1InfinityNorm > 1e-3)
  {
    std::cerr << std::setprecision(9);
    std::cerr << "Gradient (orientation part) max norm from localADOLCStiffness is " << gradient1InfinityNorm << ", from localADOLCIntegralStiffness is " << gradientSmart1InfinityNorm << " :(" << std::endl;
    return 1;
  }

  if (std::isnan(matrix00FrobeniusNorm) || std::isnan(matrixSmart00FrobeniusNorm) || std::abs(matrix00FrobeniusNorm - matrixSmart00FrobeniusNorm)/matrix00FrobeniusNorm > 1e-3)
  {
    std::cerr << std::setprecision(9);
    std::cerr << "Matrix00 norm from localADOLCStiffness is " << matrix00FrobeniusNorm << ", from localADOLCIntegralStiffness is " << matrixSmart00FrobeniusNorm << " :(" << std::endl;
    return 1;
  }

  for (auto rowIt = hesseMatrixSmart[_0][_0].begin(); rowIt != hesseMatrixSmart[_0][_0].end(); ++rowIt)
    for (auto colIt = rowIt->begin(); colIt != rowIt->end(); ++colIt)
      for (size_t k = 0; k < deformationBlocksize; k++)
        for (size_t l = 0; l < deformationBlocksize; l++)
          if (std::abs((*colIt)[k][l]) > 1e-6 and std::abs(hesseMatrix[_0][_0][rowIt.index()][colIt.index()][k][l]) > 1e-6
              and std::abs((*colIt)[k][l] - hesseMatrix[_0][_0][rowIt.index()][colIt.index()][k][l])/(*colIt)[k][l] > 1e-4) {
            std::cerr << "Matrix00: Index " << " " << rowIt.index() << " " << colIt.index() << " " << k << " " << l  << " is wrong: " << hesseMatrix[_0][_0][rowIt.index()][colIt.index()][k][l] << " - " << (*colIt)[k][l] << std::endl;
            return 1;
          }

  if (std::isnan(matrix01FrobeniusNorm) || std::isnan(matrixSmart01FrobeniusNorm) || std::abs(matrix01FrobeniusNorm - matrixSmart01FrobeniusNorm)/matrix01FrobeniusNorm > 1e-3)
  {
    std::cerr << std::setprecision(9);
    std::cerr << "Matrix01 norm from localADOLCStiffness is " << matrix01FrobeniusNorm << ", from localADOLCIntegralStiffness is " << matrixSmart01FrobeniusNorm << " :(" << std::endl;
    return 1;
  }

  for (auto rowIt = hesseMatrixSmart[_0][_1].begin(); rowIt != hesseMatrixSmart[_0][_1].end(); ++rowIt)
    for (auto colIt = rowIt->begin(); colIt != rowIt->end(); ++colIt)
      for (size_t k = 0; k < deformationBlocksize; k++)
        for (size_t l = 0; l < deformationBlocksize; l++)
          if (std::abs((*colIt)[k][l]) > 1e-6 and std::abs(hesseMatrix[_0][_1][rowIt.index()][colIt.index()][k][l]) > 1e-6
              and std::abs((*colIt)[k][l] - hesseMatrix[_0][_1][rowIt.index()][colIt.index()][k][l])/(*colIt)[k][l] > 1e-4) {
            std::cerr << "Matrix01 Index " << " " << rowIt.index() << " " << colIt.index() << " " << k << " " << l  << " is wrong: " << hesseMatrix[_0][_1][rowIt.index()][colIt.index()][k][l] << " - " << (*colIt)[k][l] << std::endl;
            return 1;
          }

  if (std::isnan(matrix10FrobeniusNorm) || std::isnan(matrixSmart10FrobeniusNorm) || std::abs(matrix10FrobeniusNorm - matrixSmart10FrobeniusNorm)/matrix10FrobeniusNorm > 1e-3)
  {
    std::cerr << std::setprecision(9);
    std::cerr << "Matrix10 norm from localADOLCStiffness is " << matrix10FrobeniusNorm << ", from localADOLCIntegralStiffness is " << matrixSmart10FrobeniusNorm << " :(" << std::endl;
    return 1;
  }

  for (auto rowIt = hesseMatrixSmart[_1][_0].begin(); rowIt != hesseMatrixSmart[_1][_0].end(); ++rowIt)
    for (auto colIt = rowIt->begin(); colIt != rowIt->end(); ++colIt)
      for (size_t k = 0; k < deformationBlocksize; k++)
        for (size_t l = 0; l < deformationBlocksize; l++)
          if (std::abs((*colIt)[k][l]) > 1e-6 and std::abs(hesseMatrix[_1][_0][rowIt.index()][colIt.index()][k][l]) > 1e-6
              and std::abs((*colIt)[k][l] - hesseMatrix[_1][_0][rowIt.index()][colIt.index()][k][l])/(*colIt)[k][l] > 1e-4) {
            std::cerr << "Matrix10 Index " << " " << rowIt.index() << " " << colIt.index() << " " << k << " " << l  << " is wrong: " << hesseMatrix[_1][_0][rowIt.index()][colIt.index()][k][l] << " - " << (*colIt)[k][l] << std::endl;
            return 1;
          }

  if (std::isnan(matrix11FrobeniusNorm) || std::isnan(matrixSmart00FrobeniusNorm) || std::abs(matrix11FrobeniusNorm - matrixSmart11FrobeniusNorm)/matrix11FrobeniusNorm > 1e-3)
  {
    std::cerr << std::setprecision(9);
    std::cerr << "Matrix11 norm from localADOLCStiffness is " << matrix11FrobeniusNorm << ", from localADOLCIntegralStiffness is " << matrixSmart11FrobeniusNorm << " :(" << std::endl;
    return 1;
  }

  for (auto rowIt = hesseMatrixSmart[_1][_1].begin(); rowIt != hesseMatrixSmart[_1][_1].end(); ++rowIt)
    for (auto colIt = rowIt->begin(); colIt != rowIt->end(); ++colIt)
      for (size_t k = 0; k < deformationBlocksize; k++)
        for (size_t l = 0; l < deformationBlocksize; l++)
          if (std::abs((*colIt)[k][l]) > 1e-6 and std::abs(hesseMatrix[_1][_1][rowIt.index()][colIt.index()][k][l]) > 1e-6
              and std::abs((*colIt)[k][l] - hesseMatrix[_1][_1][rowIt.index()][colIt.index()][k][l])/(*colIt)[k][l] > 1e-4) {
            std::cerr << "Matrix11 Index " << " " << rowIt.index() << " " << colIt.index() << " " << k << " " << l  << " is wrong: " << hesseMatrix[_1][_1][rowIt.index()][colIt.index()][k][l] << " - " << (*colIt)[k][l] << std::endl;
            return 1;
          }

  return 0;
}


int main (int argc, char *argv[])
{
  MPIHelper::instance(argc, argv);

  TestSuite test;

  /////////////////////////////////////////
  //   Create the grid
  /////////////////////////////////////////

  // Create a grid with 2 elements, one element will get refined to create different element types, the other one stays a cube
  using GridType = UGGrid<dim>;
  auto grid = StructuredGridFactory<GridType>::createCubeGrid({0,0,0}, {1,1,1}, {2,1,1});

  //Refine once
  for (auto&& e : elements(grid->leafGridView())) {
    bool refineThisElement = false;
    for (int i = 0; i < e.geometry().corners(); i++) {
      refineThisElement = refineThisElement || (e.geometry().corner(i)[0] > 0.99);
    }
    grid->mark(refineThisElement ? 1 : 0, e);
  }
  grid->adapt();
  grid->loadBalance();

  using GridView = GridType::LeafGridView;
  auto gridView = grid->leafGridView();

  /////////////////////////////////////////////
  //  Test different energies
  /////////////////////////////////////////////

  // TODO: Use test framework
  testHarmonicMapIntoSphere<GridView,Geodesic>(test, gridView);
  testHarmonicMapIntoSphere<GridView,ProjectionBased>(test, gridView);
  testHarmonicMapIntoSphere<GridView,Nonconforming>(test, gridView);

  testCosseratBulkModel<GridView,Geodesic>(test, gridView);
  testCosseratBulkModel<GridView,ProjectionBased>(test, gridView);

  return test.exit();
}
