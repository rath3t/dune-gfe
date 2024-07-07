#include <config.h>

#include <fenv.h>

#include <array>

//#define MULTIPRECISION

#ifdef MULTIPRECISION
#include <boost/multiprecision/mpfr.hpp>
#endif

#ifdef MULTIPRECISION
typedef boost::multiprecision::mpfr_float_50 FDType;
#else
typedef double FDType;
#endif

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>    // use of "Easy to Use" drivers
#include <adolc/taping.h>

#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/typetraits.hh>
#include <dune/common/fmatrix.hh>

#include <dune/geometry/quadraturerules.hh>

#include <dune/grid/yaspgrid.hh>

#include <dune/istl/io.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>

#include <dune/gfe/assemblers/localgeodesicfestiffness.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/assemblers/cosseratenergystiffness.hh>
#include <dune/gfe/assemblers/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/assemblers/localgeodesicfefdstiffness.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>

using namespace Dune;

// grid dimension
const int dim = 2;

// Image space of the geodesic fe functions
using TargetSpace = GFE::ProductManifold<RealTuple<double,3>,Rotation<double,3> >;

// Compare two matrices
void compareMatrices(const Matrix<double>& matrixA, std::string nameA,
                     const Matrix<double>& matrixB, std::string nameB)
{
  double maxAbsDifference = -1;
  double maxRelDifference = -1;

  for(size_t i=0; i<matrixA.N(); i++) {

    for (size_t j=0; j<matrixA.M(); j++ ) {

      const double valueA = matrixA[i][j];
      const double valueB = matrixB[i][j];

      double absDifference = valueA - valueB;
      double relDifference = std::abs(absDifference) / std::abs(valueA);
      maxAbsDifference = std::max(maxAbsDifference, std::abs(absDifference));
      if (not std::isinf(relDifference))
        maxRelDifference = std::max(maxRelDifference, relDifference);

      if (relDifference > 1)
        std::cout << i << ", " << j << "   ,       "
                  << nameA << ": " << valueA << ",           " << nameB << ": " << valueB << std::endl;
    }
  }

  std::cout << nameA << " vs. " << nameB << " -- max absolute / relative difference is " << maxAbsDifference << " / " << maxRelDifference << std::endl;
}


int main (int argc, char *argv[]) try
{
  MPIHelper::instance(argc, argv);

  typedef std::vector<TargetSpace> SolutionType;
  constexpr static int blocksize = TargetSpace::TangentVector::dimension;

  // ///////////////////////////////////////
  //    Create the grid
  // ///////////////////////////////////////
  typedef YaspGrid<dim> GridType;

  FieldVector<double,dim> upper = {{0.38, 0.128}};

  std::array<int,dim> elements = {{5, 5}};
  GridType grid(upper, elements);

  typedef GridType::LeafGridView GridView;
  GridView gridView = grid.leafGridView();

  typedef Functions::LagrangeBasis<GridView,1> FEBasis;
  FEBasis feBasis(gridView);

  // /////////////////////////////////////////
  //   Read Dirichlet values
  // /////////////////////////////////////////

  // //////////////////////////
  //   Initial iterate
  // //////////////////////////

  SolutionType x(feBasis.size());

  //////////////////////////////////////////7
  //  Read initial iterate from file
  //////////////////////////////////////////7
#if 0
  Dune::BlockVector<FieldVector<double,7> > xEmbedded(x.size());

  std::ifstream file("dangerous_iterate", std::ios::in|std::ios::binary);
  if (not (file))
    DUNE_THROW(SolverError, "Couldn't open file 'dangerous_iterate' for reading");

  GenericVector::readBinary(file, xEmbedded);

  file.close();

  for (int ii=0; ii<x.size(); ii++)
    x[ii] = xEmbedded[ii];
#else
  auto identity = [](const FieldVector<double,2>& x) -> FieldVector<double,3> {
                    return {x[0], x[1], 0};
                  };

  std::vector<FieldVector<double,3> > v;
  using namespace Functions::BasisFactory;

  auto powerBasis = makeBasis(
    gridView,
    power<3>(
      lagrange<1>(),
      blockedInterleaved()
      ));
  Functions::interpolate(powerBasis, v, identity);

  for (size_t i=0; i<x.size(); i++)
    x[i][Indices::_0] = v[i];
#endif

  // ////////////////////////////////////////////////////////////
  //   Create an assembler for the energy functional
  // ////////////////////////////////////////////////////////////

  ParameterTree materialParameters;
  materialParameters["thickness"] = "1";
  materialParameters["mu"] = "1";
  materialParameters["lambda"] = "1";
  materialParameters["mu_c"] = "1";
  materialParameters["L_c"] = "1";
  materialParameters["q"] = "2";
  materialParameters["kappa"] = "1";
  materialParameters["b1"] = "1";
  materialParameters["b2"] = "1";
  materialParameters["b3"] = "1";


  ///////////////////////////////////////////////////////////////////////
  //  Assemblers for the Riemannian derivatives
  ///////////////////////////////////////////////////////////////////////

  // Assembler using ADOL-C
  auto cosseratLocalEnergy = std::make_shared<CosseratEnergyLocalStiffness<FEBasis, 3,adouble> >(materialParameters, nullptr, nullptr, nullptr);

  LocalGeodesicFEADOLCStiffness<FEBasis,
      TargetSpace> localGFEADOLCStiffness(cosseratLocalEnergy);

  // Assembler using finite differences
  CosseratEnergyLocalStiffness<FEBasis, 3,FDType>
  cosseratEnergyFDLocalStiffness(materialParameters, nullptr, nullptr, nullptr);
  LocalGeodesicFEFDStiffness<FEBasis,
      TargetSpace,
      FDType> localGFEFDStiffness(&cosseratEnergyFDLocalStiffness);

  // Compute and compare matrices
  for (const auto& element : Dune::elements(gridView))
  {
    std::cout << "  ++++  element " << gridView.indexSet().index(element) << " ++++" << std::endl;

    auto localView     = feBasis.localView();
    localView.bind(element);

    const int numOfBaseFct = localView.size();

    // Extract local configuration
    std::vector<TargetSpace> localSolution(numOfBaseFct);

    for (int i=0; i<numOfBaseFct; i++)
      localSolution[i] = x[localView.index(i)];

    // Assemble Riemannian derivatives
    std::vector<double> localRiemannianADGradient(numOfBaseFct*blocksize);
    std::vector<double> localRiemannianFDGradient(numOfBaseFct*blocksize);

    Matrix<double> localRiemannianADHessian;
    Matrix<double> localRiemannianFDHessian;

    localGFEADOLCStiffness.assembleGradientAndHessian(localView,
                                                      localSolution,
                                                      localRiemannianADGradient,
                                                      localRiemannianADHessian);

    localGFEFDStiffness.assembleGradientAndHessian(localView,
                                                   localSolution,
                                                   localRiemannianFDGradient,
                                                   localRiemannianFDHessian);

    // compare
    compareMatrices(localRiemannianADHessian, "Riemannian AD", localRiemannianFDHessian, "Riemannian FD");

  }

  return 0;
}
catch (Exception& e)
{
  std::cout << e.what() << std::endl;
  return 1;
}
