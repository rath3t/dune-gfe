/** \file
    \brief Unit tests for classes that implement derivatives of interpolation functions
 */
#include <config.h>

#define DUNE_ISTL_WITH_CHECKING

#include <adolc/adolc.h>
#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/test/testsuite.hh>

#include <dune/grid/uggrid.hh>

#include <dune/istl/io.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>

#include <dune/gfe/functions/interpolationderivatives.hh>
#include <dune/gfe/spaces/unitvector.hh>
#include <dune/gfe/spaces/realtuple.hh>

#include "valuefactory.hh"



using namespace Dune;

/** \brief Compute derivatives of GFE interpolation with respect to the coefficients using finite differencts
 *
 * This class implements the InterpolationDerivatives interface but uses a finite difference
 * approximation to approximate those derivatives.  This is used for testing purposes only.
 *
 * \tparam LocalInterpolationRule The class that implements the interpolation from a set of coefficients
 *
 */
template <typename LocalInterpolationRule>
class FiniteDifferenceInterpolationDerivatives
{
  using TargetSpace = typename LocalInterpolationRule::TargetSpace;
  using Derivative = typename LocalInterpolationRule::DerivativeType;

  constexpr static auto blocksize = TargetSpace::TangentVector::dimension;
  constexpr static auto embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension;

  //////////////////////////////////////////////////////////////////////
  //  Data members
  //////////////////////////////////////////////////////////////////////

  // TODO: Do not hard-wirde this!
  static constexpr int domainDim = 2;

  FieldVector<double,domainDim> localPos_;
  FieldMatrix<double,domainDim,domainDim> geometryJacobianInverse_;

  const LocalInterpolationRule& localInterpolationRule_;

  std::vector<TargetSpace> coefficients_;

  // TODO: Don't hardcode FieldMatrix
  std::vector<FieldMatrix<double,blocksize,embeddedBlocksize> > orthonormalFrames_;

public:

  FiniteDifferenceInterpolationDerivatives(const LocalInterpolationRule& localInterpolationRule)
    : localInterpolationRule_(localInterpolationRule)
  {
    // Copy the coefficients into a dedicated array, for easier access
    coefficients_.resize(localInterpolationRule.size());
    for (std::size_t i=0; i<localInterpolationRule.size(); i++)
      coefficients_[i] = localInterpolationRule.coefficient(i);

    // Precompute the orthonormal frames
    orthonormalFrames_.resize(localInterpolationRule_.size());
    for (size_t i=0; i<localInterpolationRule_.size(); ++i)
      orthonormalFrames_[i] = localInterpolationRule_.coefficient(i).orthonormalFrame();
  }

  /** \brief Bind the objects to a particular evaluation point
   *
   * In particular, this computes the value of the interpolation function at that point,
   * and the derivative at that point with respect to space.
   *
   *  \param[in]  tapeNumber      Number of the ADOL-C tape if ADOL-C is used.  Dummy otherwise
   *  \param[in]  localPos        Local position where the FE function is evaluated
   *  \param[out] value           The function value at the local configuration
   *  \param[out] derivative      The derivative of the interpolation function
   *                              with respect to the evaluation point
   */
  template <typename Element>
  void bind(short tapeNumber,
            const Element& element,
            const typename Element::Geometry::LocalCoordinate& localPos,
            typename TargetSpace::CoordinateType& value,
            typename LocalInterpolationRule::DerivativeType& derivative)
  {
    localPos_ = localPos;

    value = localInterpolationRule_.evaluate(localPos).globalCoordinates();

    geometryJacobianInverse_ = element.geometry().jacobianInverse(localPos);

    auto referenceDerivative = localInterpolationRule_.evaluateDerivative(localPos, value);
    derivative = referenceDerivative * geometryJacobianInverse_;
  }

  /** \brief Compute first and second derivatives of the FE interpolation
   *
   * This code assumes that `bind` has been called before.
   *
   *  \param[in]  tapeNumber            The tape number to be used by ADOL-C.  Must be the same
   *                                    that was given to the `bind` method.
   *  \param[in]  weights               Vector of weights that the second derivative is contracted with
   *  \param[out] embeddedFirstDerivative       Derivative of the FE interpolation wrt the coefficients
   *  \param[out] firstDerivative       Derivative of the FE interpolation wrt the coefficients
   *  \param[out] secondDerivative      Second derivative of the FE interpolation,
   *                                    contracted with the weight vector
   */
  void evaluateDerivatives(short tapeNumber,
                           const std::vector<double>& adjoint,
                           Matrix<double>& euclideanFirstDerivative,
                           Matrix<double>& riemannianFirstDerivative,
                           Matrix<FieldMatrix<double,blocksize,blocksize> >& secondDerivative) const
  {
    ////////////////////////////////////////////////////////////////////////
    //  Compute Euclidean first derivative of the interpolation value
    ////////////////////////////////////////////////////////////////////////

    for (std::size_t coefficient=0; coefficient<localInterpolationRule_.size(); coefficient++)
    {
      std::vector<TargetSpace> cornersPlus  = coefficients_;
      std::vector<TargetSpace> cornersMinus = coefficients_;

      for (std::size_t j=0; j<TargetSpace::CoordinateType::size(); j++)
      {
        // Optimal variation size for first derivatives
        const double eps = std::sqrt(std::numeric_limits<double>::epsilon());

        // Variation in coordinates of the surrounding spaces
        typename TargetSpace::CoordinateType variation(0.0);
        variation[j] = eps;

        cornersPlus [coefficient] = TargetSpace(coefficients_[coefficient].globalCoordinates() + variation);
        cornersMinus[coefficient] = TargetSpace(coefficients_[coefficient].globalCoordinates() - variation);

        LocalInterpolationRule fPlus(localInterpolationRule_.localFiniteElement(),cornersPlus);
        LocalInterpolationRule fMinus(localInterpolationRule_.localFiniteElement(),cornersMinus);

        /////////////////////////////////////////////////////////////
        //  Compute first derivative of the interpolation value
        /////////////////////////////////////////////////////////////

        TargetSpace hPlus  = fPlus.evaluate(localPos_);
        TargetSpace hMinus = fMinus.evaluate(localPos_);

        for (std::size_t k=0; k<TargetSpace::CoordinateType::size(); k++)
          euclideanFirstDerivative[k][coefficient*TargetSpace::CoordinateType::size()+j]
            = (hPlus.globalCoordinates()[k] - hMinus.globalCoordinates()[k]) / (2*eps);

        /////////////////////////////////////////////////////////////
        //  Compute first derivative of the interpolation gradient
        /////////////////////////////////////////////////////////////
        auto hPlusDer  = fPlus.evaluateDerivative(localPos_) * geometryJacobianInverse_;
        auto hMinusDer = fMinus.evaluateDerivative(localPos_) * geometryJacobianInverse_;

        for (std::size_t k=0; k<hPlusDer.N(); k++)
          for (std::size_t l=0; l<hPlusDer.M(); l++)
            euclideanFirstDerivative[k*hPlusDer.M()+l+TargetSpace::CoordinateType::size()][coefficient*TargetSpace::CoordinateType::size()+j] = (hPlusDer[k][l] - hMinusDer[k][l]) / (2*eps);
      }
    }


    ////////////////////////////////////////////////////////////////////////
    //  Compute Riemannian first derivative of the interpolation value
    ////////////////////////////////////////////////////////////////////////

    for (std::size_t coefficient=0; coefficient<localInterpolationRule_.size(); coefficient++)
    {
      // the function value at the point where we are evaluating the derivative
      const auto B = orthonormalFrames_[coefficient];

      std::vector<TargetSpace> cornersPlus  = coefficients_;
      std::vector<TargetSpace> cornersMinus = coefficients_;

      for (std::size_t j=0; j<B.size(); j++)
      {
        // Optimal variation size for first derivatives
        const double eps = std::sqrt(std::numeric_limits<double>::epsilon());

        auto forwardVariation = B[j];
        forwardVariation *= eps;
        auto backwardVariation = B[j];
        backwardVariation *= -eps;

        cornersPlus [coefficient] = TargetSpace::exp(coefficients_[coefficient], forwardVariation);
        cornersMinus[coefficient] = TargetSpace::exp(coefficients_[coefficient], backwardVariation);

        LocalInterpolationRule fPlus(localInterpolationRule_.localFiniteElement(),cornersPlus);
        LocalInterpolationRule fMinus(localInterpolationRule_.localFiniteElement(),cornersMinus);

        /////////////////////////////////////////////////////////////
        //  Compute first derivative of the interpolation value
        /////////////////////////////////////////////////////////////

        TargetSpace hPlus  = fPlus.evaluate(localPos_);
        TargetSpace hMinus = fMinus.evaluate(localPos_);

        for (std::size_t k=0; k<TargetSpace::CoordinateType::size(); k++)
          riemannianFirstDerivative[k][coefficient*B.size()+j]
            = (hPlus.globalCoordinates()[k] - hMinus.globalCoordinates()[k]) / (2*eps);

        /////////////////////////////////////////////////////////////
        //  Compute first derivative of the interpolation gradient
        /////////////////////////////////////////////////////////////
        auto hPlusDer  = fPlus.evaluateDerivative(localPos_) * geometryJacobianInverse_;
        auto hMinusDer = fMinus.evaluateDerivative(localPos_) * geometryJacobianInverse_;

        for (std::size_t k=0; k<hPlusDer.N(); k++)
          for (std::size_t l=0; l<hPlusDer.M(); l++)
            riemannianFirstDerivative[k*hPlusDer.M()+l+TargetSpace::CoordinateType::size()][coefficient*B.size()+j] = (hPlusDer[k][l] - hMinusDer[k][l]) / (2*eps);
      }
    }


    ///////////////////////////////////////////////////////////////////////////
    //   Compute Riemannian Hesse matrix by finite-difference approximation.
    ///////////////////////////////////////////////////////////////////////////

    // Precompute value at the current configuration
    auto centerValue = localInterpolationRule_.evaluate(localPos_).globalCoordinates();
    auto centerDerivative = localInterpolationRule_.evaluateDerivative(localPos_)* geometryJacobianInverse_;

    // Precompute energy infinitesimal corrections in the directions of the local basis vectors
    std::vector<std::array<TargetSpace,blocksize> > forwardValue(coefficients_.size());
    std::vector<std::array<TargetSpace,blocksize> > backwardValue(coefficients_.size());

    std::vector<std::array<Derivative,blocksize> > forwardDer(coefficients_.size());
    std::vector<std::array<Derivative,blocksize> > backwardDer(coefficients_.size());

    BlockVector<FieldVector<double,blocksize> > canonicalValues(coefficients_.size());

    for (size_t i=0; i<coefficients_.size(); i++)
    {
      for (size_t i2=0; i2<blocksize; i2++)
      {
        // Optimal variation size for second derivatives
        const double eps = std::pow(std::numeric_limits<double>::epsilon(), 0.25);

        typename TargetSpace::EmbeddedTangentVector xi = orthonormalFrames_[i][i2];

        auto forwardSolution  = coefficients_;
        auto backwardSolution = coefficients_;

        forwardSolution[i]  = TargetSpace::exp(coefficients_[i], eps * xi);
        backwardSolution[i] = TargetSpace::exp(coefficients_[i], -1 * eps * xi);

        LocalInterpolationRule fPlus(localInterpolationRule_.localFiniteElement(),forwardSolution);
        LocalInterpolationRule fMinus(localInterpolationRule_.localFiniteElement(),backwardSolution);

        forwardValue[i][i2] = fPlus.evaluate(localPos_);
        backwardValue[i][i2] = fMinus.evaluate(localPos_);

        forwardDer[i][i2] = fPlus.evaluateDerivative(localPos_)* geometryJacobianInverse_;
        backwardDer[i][i2] = fMinus.evaluateDerivative(localPos_)* geometryJacobianInverse_;

        // Finite difference quotient for the second derivative
        auto valueDerivative = (forwardValue[i][i2].globalCoordinates() -2*centerValue + backwardValue[i][i2].globalCoordinates()) / (eps * eps);

        auto jacobianDerivative = (forwardDer[i][i2] -2*centerDerivative + backwardDer[i][i2]) / (eps * eps);

        // Multiply with the adjoint
        canonicalValues[i][i2] = 0;
        for (std::size_t j=0; j<valueDerivative.size(); j++)
          canonicalValues[i][i2] += adjoint[j] * valueDerivative[j];

        for (std::size_t j=0; j<jacobianDerivative.N(); j++)
          for (std::size_t j2=0; j2<jacobianDerivative.M(); j2++)
            canonicalValues[i][i2] += adjoint[valueDerivative.size() + j*jacobianDerivative.M() + j2] * jacobianDerivative[j][j2];
      }
    }

    for (size_t i=0; i<localInterpolationRule_.size(); i++)
    {
      for (size_t i2=0; i2<blocksize; i2++)
      {
        for (size_t j=0; j<localInterpolationRule_.size(); j++)
        {
          for (size_t j2=0; j2<blocksize; j2++)
          {
            // Optimal variation size for second derivatives
            const double eps = std::pow(std::numeric_limits<double>::epsilon(), 0.25);

            std::vector<TargetSpace> forwardSolutionXiEta   = coefficients_;
            std::vector<TargetSpace> backwardSolutionXiEta  = coefficients_;

            typename TargetSpace::EmbeddedTangentVector epsXi  = orthonormalFrames_[i][i2];
            epsXi *= eps;
            typename TargetSpace::EmbeddedTangentVector epsEta = orthonormalFrames_[j][j2];
            epsEta *= eps;

            if (i==j)
              forwardSolutionXiEta[i] = TargetSpace::exp(coefficients_[i],epsXi+epsEta);
            else {
              forwardSolutionXiEta[i] = TargetSpace::exp(coefficients_[i],epsXi);
              forwardSolutionXiEta[j] = TargetSpace::exp(coefficients_[j],epsEta);
            }

            if (i==j)
              backwardSolutionXiEta[i] = TargetSpace::exp(coefficients_[i], (-1)*epsXi + (-1)*epsEta);
            else {
              backwardSolutionXiEta[i] = TargetSpace::exp(coefficients_[i], (-1)*epsXi);
              backwardSolutionXiEta[j] = TargetSpace::exp(coefficients_[j], (-1)*epsEta);
            }

            LocalInterpolationRule fPlus(localInterpolationRule_.localFiniteElement(),forwardSolutionXiEta);
            LocalInterpolationRule fMinus(localInterpolationRule_.localFiniteElement(),backwardSolutionXiEta);

            /////////////////////////////////////////////////////////////////////////////////////
            //  Compute second derivative of the adjoint vector times the interpolation value
            /////////////////////////////////////////////////////////////////////////////////////

            auto forwardTmp  = fPlus.evaluate(localPos_).globalCoordinates();
            auto backwardTmp = fMinus.evaluate(localPos_).globalCoordinates();

            auto foo = (forwardTmp - 2*centerValue + backwardTmp) / (eps*eps);

            // Scalar product:  ... = adjoint * foo;
            secondDerivative[i][j][i2][j2] = 0;
            for (std::size_t k=0; k<foo.size(); k++)
              secondDerivative[i][j][i2][j2] += adjoint[k] * foo[k];

            /////////////////////////////////////////////////////////////////////////////////////
            //  Compute second derivative of the adjoint vector times the interpolation gradient
            /////////////////////////////////////////////////////////////////////////////////////

            auto forwardDerTmp  = fPlus.evaluateDerivative(localPos_)* geometryJacobianInverse_;
            auto backwardDerTmp = fMinus.evaluateDerivative(localPos_)* geometryJacobianInverse_;

            auto foo2 = (forwardDerTmp - 2*centerDerivative + backwardDerTmp) / (eps*eps);
            // Scalar product:  ... += adjoint * foo2;
            for (std::size_t k=0; k<foo2.N(); k++)
              for (std::size_t l=0; l<foo2.M(); l++)
                secondDerivative[i][j][i2][j2] += adjoint[k*foo2.M()+l+TargetSpace::CoordinateType::size()] * foo2[k][l];

            ////////////////////////////////////////////////////////////////////////////////////
            // Use a polarization identity to get the actual Hesse matrix entry
            ////////////////////////////////////////////////////////////////////////////////////

            secondDerivative[i][j][i2][j2] = 0.5 * (secondDerivative[i][j][i2][j2] - canonicalValues[i][i2] - canonicalValues[j][j2]);
          }
        }
      }
    }
  }

};


enum class InterpolationType {Geodesic, ProjectionBased};

template <class TargetSpace, InterpolationType interpolationType>
TestSuite checkDerivatives()
{
  TestSuite test;

  std::cout << "Testing class " << className<TargetSpace>() << std::endl;

  ////////////////////////////////////////////////////
  //  Make grid consisting of a single triangle
  ////////////////////////////////////////////////////

  static const int domainDim = 2;
  using Grid = UGGrid<domainDim>;
  GridFactory<Grid> factory;

  factory.insertVertex({1.0, 1.0});
  factory.insertVertex({2.0, 1.5});
  factory.insertVertex({2.5, 3.0});

  factory.insertElement(GeometryTypes::simplex(domainDim), {0,1,2});

  auto grid = factory.createGrid();
  auto gridView = grid->leafGridView();
  using GridView = decltype(gridView);

  /////////////////////////////////////////////////////////////////////////
  //  Construct a LocalInterpolationRule whose derivative we will compute
  /////////////////////////////////////////////////////////////////////////

  constexpr int order = 1;
  Functions::LagrangeBasis<GridView,order> scalarBasis(gridView);

  std::vector<TargetSpace> testPoints;
  ValueFactory<TargetSpace>::get(testPoints);

  // TODO: Make sure the list of test points is longer than this.
  const std::size_t nDofs = scalarBasis.dimension();

  std::vector<TargetSpace> localCoefficients(nDofs);
  for (std::size_t i=0; i<nDofs; i++)
    localCoefficients[i] = testPoints[i];

  /////////////////////////////////////////////////////////////////////////
  //  Construct the InterpolationDerivatives object that we will test
  /////////////////////////////////////////////////////////////////////////

  // Define the two possible interpolation rules
  using GeodesicInterpolationRule = LocalGeodesicFEFunction<domainDim,
      typename Grid::ctype,
      decltype(scalarBasis.localView().tree().finiteElement()),
      TargetSpace>;

  using ProjectionBasedInterpolationRule = GFE::LocalProjectedFEFunction<domainDim,
      typename Grid::ctype,
      decltype(scalarBasis.localView().tree().finiteElement()),
      TargetSpace>;

  // Select the one to test
  using LocalInterpolationRule = std::conditional_t<interpolationType==InterpolationType::Geodesic,
      GeodesicInterpolationRule,
      ProjectionBasedInterpolationRule>;


  auto localView = scalarBasis.localView();
  localView.bind(*gridView.begin<0>());
  LocalInterpolationRule localGFEFunction(localView.tree().finiteElement(),localCoefficients);

  GFE::InterpolationDerivatives<LocalInterpolationRule> interpolationDerivatives(localGFEFunction,
                                                                                 true,   // doValue
                                                                                 true);  // doDerivative

  /////////////////////////////////////////////////////////////////////////
  //  Construct the finite difference InterpolationDerivatives object
  //  that we will use to compare with
  /////////////////////////////////////////////////////////////////////////

  FiniteDifferenceInterpolationDerivatives<LocalInterpolationRule> interpolationDerivativesFD(localGFEFunction);

  /////////////////////////////////////////////////////////////////////////
  //  Bind the two objects to a test point, and verify that this
  //  produces identical results.
  /////////////////////////////////////////////////////////////////////////

  // InterpolationDerivatives uses ADOL-C by default.  Therefore, give a tape number
  const int tapeNumber = 0;

  const typename Grid::template Codim<0>::Entity::Geometry::LocalCoordinate position = {0.3, 0.3};

  typename TargetSpace::CoordinateType valueGlobalCoordinates;
  typename TargetSpace::CoordinateType valueFDGlobalCoordinates;

  typename LocalInterpolationRule::DerivativeType derivative;
  typename LocalInterpolationRule::DerivativeType derivativeFD;

  interpolationDerivatives.bind(tapeNumber,
                                localView.element(),
                                position,
                                valueGlobalCoordinates,
                                derivative);

  TargetSpace value(valueGlobalCoordinates);

  interpolationDerivativesFD.bind(tapeNumber,
                                  localView.element(),
                                  position,
                                  valueFDGlobalCoordinates,
                                  derivativeFD);

  TargetSpace valueFD(valueFDGlobalCoordinates);

  ///////////////////////////////////////////////////////
  //  Compute the derivatives, and compare them
  ///////////////////////////////////////////////////////

  constexpr auto blocksize = TargetSpace::TangentVector::dimension;
  constexpr auto embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension;
  // Number of dependent variables for the interpolation function
  // The sum of the variables for the interpolation value and the variables
  // for the derivative
  constexpr auto m = TargetSpace::CoordinateType::size() + embeddedBlocksize*domainDim;

  std::vector<double> weights(m);

  for (std::size_t i=0; i<m; i++)
  {
    std::fill(weights.begin(), weights.end(), 0.0);

    weights[i] = 1.0;


    Matrix<double> euclideanInterpolationGradient(m, nDofs*embeddedBlocksize);
    Matrix<double> riemannianInterpolationGradient(m, nDofs*blocksize);

    Matrix<FieldMatrix<double,blocksize,blocksize> > interpolationHessian(nDofs,nDofs);


    interpolationDerivatives.evaluateDerivatives(tapeNumber,
                                                 weights.data(),
                                                 euclideanInterpolationGradient,
                                                 riemannianInterpolationGradient,
                                                 interpolationHessian);

    Matrix<double> euclideanInterpolationGradientFD(m, nDofs*embeddedBlocksize);
    Matrix<double> riemannianInterpolationGradientFD(m, nDofs*blocksize);


    Matrix<FieldMatrix<double,blocksize,blocksize> > interpolationHessianFD(nDofs,nDofs);

    interpolationDerivativesFD.evaluateDerivatives(tapeNumber,
                                                   weights,
                                                   euclideanInterpolationGradientFD,
                                                   riemannianInterpolationGradientFD,
                                                   interpolationHessianFD);

    /////////////////////////////////////////////////////////////////
    //  Compare the derivatives
    /////////////////////////////////////////////////////////////////

    auto riemannianDifference = riemannianInterpolationGradient;
    riemannianDifference -= riemannianInterpolationGradientFD;

    if (std::isnan(riemannianDifference.infinity_norm()) || riemannianDifference.infinity_norm() > 1e-6)
    {
      printmatrix(std::cout, riemannianInterpolationGradient, "riemannianInterpolationGradient", "--");
      printmatrix(std::cout, riemannianInterpolationGradientFD, "riemannianInterpolationGradientFD", "--");
    }

    auto euclideanDifference = euclideanInterpolationGradient;
    euclideanDifference -= euclideanInterpolationGradientFD;

    if (std::isnan(euclideanDifference.infinity_norm()) || euclideanDifference.infinity_norm() > 1e-6)
    {
      printmatrix(std::cout, euclideanInterpolationGradient, "euclideanInterpolationGradient", "--");
      printmatrix(std::cout, euclideanInterpolationGradientFD, "euclideanInterpolationGradientFD", "--");
    }

    auto hessianDifference = interpolationHessian;
    hessianDifference -= interpolationHessianFD;

    if (std::isnan(hessianDifference.infinity_norm()) || hessianDifference.infinity_norm() > 1e-5)
    {
      printmatrix(std::cout, interpolationHessian, "interpolationHessian", "--");
      printmatrix(std::cout, interpolationHessianFD, "interpolationHessianFD", "--");
    }
  }
  return test;
}


int main (int argc, char *argv[])
{
  // Set up MPI, if available
  MPIHelper::instance(argc, argv);

  TestSuite test;

  // Test the UnitSphere class and geodesic interpolation.
  // This uses the default derivatives implementation (using ADOL-C)
  test.subTest(checkDerivatives<UnitVector<double,3>, InterpolationType::Geodesic >());

  // Test the RealTuple class, both with geodesic and projection-based interpolation
  // Both are specialized
  test.subTest(checkDerivatives<RealTuple<double,3>, InterpolationType::Geodesic>());
  test.subTest(checkDerivatives<RealTuple<double,3>, InterpolationType::ProjectionBased>());

  // Test the UnitVector class with projection-based interpolation
  // This is also specialized.
  test.subTest(checkDerivatives<UnitVector<double,3>, InterpolationType::ProjectionBased>());

  return test.exit();
}
