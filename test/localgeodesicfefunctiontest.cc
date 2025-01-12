#include <config.h>

#include <fenv.h>
#include <iostream>

#include <dune/common/fvector.hh>

#include <dune/geometry/quadraturerules.hh>
#include <dune/geometry/type.hh>
#include <dune/geometry/referenceelements.hh>

#include <dune/localfunctions/lagrange/lagrangelfecache.hh>

#include <dune/gfe/functions/localgeodesicfefunction.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>
#include <dune/gfe/spaces/unitvector.hh>

#include "multiindex.hh"
#include "valuefactory.hh"

const double eps = 1e-6;

using namespace Dune;

/** \brief Computes the diameter of a set */
template <class TargetSpace>
double diameter(const std::vector<TargetSpace>& v)
{
  double d = 0;
  for (size_t i=0; i<v.size(); i++)
    for (size_t j=0; j<v.size(); j++)
      d = std::max(d, TargetSpace::distance(v[i],v[j]));
  return d;
}

template <int dim, class ctype, class LocalFunction>
auto
evaluateDerivativeFD(const LocalFunction& f, const Dune::FieldVector<ctype, dim>& local)
-> decltype(f.evaluateDerivative(local))
{
  double eps = 1e-8;
  //static const int embeddedDim = LocalFunction::TargetSpace::embeddedDim;
  decltype(f.evaluateDerivative(local)) result;

  for (int i=0; i<dim; i++) {

    Dune::FieldVector<ctype, dim> forward  = local;
    Dune::FieldVector<ctype, dim> backward = local;

    forward[i]  += eps;
    backward[i] -= eps;

    auto fdDer = f.evaluate(forward).globalCoordinates() - f.evaluate(backward).globalCoordinates();
    fdDer /= 2*eps;

    for (size_t j=0; j<result.N(); j++)
      result[j][i] = fdDer[j];

  }

  return result;
}


/** \brief Test whether interpolation is invariant under permutation of the simplex vertices
 * \todo Implement this for all dimensions
 */
template <int domainDim, class TargetSpace>
void testPermutationInvariance(const std::vector<TargetSpace>& corners)
{
  // works only for 2d domains
  if (domainDim!=2)
    return;

  LagrangeLocalFiniteElementCache<double,double,domainDim,1> feCache;
  typedef typename LagrangeLocalFiniteElementCache<double,double,domainDim,1>::FiniteElementType LocalFiniteElement;

  GeometryType simplex = GeometryTypes::simplex(domainDim);

  //
  std::vector<TargetSpace> cornersRotated1(domainDim+1);
  std::vector<TargetSpace> cornersRotated2(domainDim+1);

  cornersRotated1[0] = cornersRotated2[2] = corners[1];
  cornersRotated1[1] = cornersRotated2[0] = corners[2];
  cornersRotated1[2] = cornersRotated2[1] = corners[0];

  LocalGeodesicFEFunction<2,double,LocalFiniteElement,TargetSpace> f0(feCache.get(simplex), corners);
  LocalGeodesicFEFunction<2,double,LocalFiniteElement,TargetSpace> f1(feCache.get(simplex), cornersRotated1);
  LocalGeodesicFEFunction<2,double,LocalFiniteElement,TargetSpace> f2(feCache.get(simplex), cornersRotated2);

  // A quadrature rule as a set of test points
  int quadOrder = 3;

  const Dune::QuadratureRule<double, domainDim>& quad
    = Dune::QuadratureRules<double, domainDim>::rule(simplex, quadOrder);

  for (size_t pt=0; pt<quad.size(); pt++) {

    const Dune::FieldVector<double,domainDim>& quadPos = quad[pt].position();

    Dune::FieldVector<double,domainDim> l0 = quadPos;
    Dune::FieldVector<double,domainDim> l1, l2;

    l1[0] = quadPos[1];
    l1[1] = 1-quadPos[0]-quadPos[1];

    l2[0] = 1-quadPos[0]-quadPos[1];
    l2[1] = quadPos[0];

    // evaluate the three functions
    TargetSpace v0 = f0.evaluate(l0);
    TargetSpace v1 = f1.evaluate(l1);
    TargetSpace v2 = f2.evaluate(l2);

    // Check that they are all equal
    assert(TargetSpace::distance(v0,v1) < eps);
    assert(TargetSpace::distance(v0,v2) < eps);

  }

}

template <int domainDim, class TargetSpace>
void testDerivative(const LocalGeodesicFEFunction<domainDim,double,typename LagrangeLocalFiniteElementCache<double,double,domainDim,1>::FiniteElementType, TargetSpace>& f)
{
  static const int embeddedDim = TargetSpace::EmbeddedTangentVector::dimension;

  // A quadrature rule as a set of test points
  int quadOrder = 3;

  const Dune::QuadratureRule<double, domainDim>& quad
    = Dune::QuadratureRules<double, domainDim>::rule(f.type(), quadOrder);

  for (size_t pt=0; pt<quad.size(); pt++) {

    const Dune::FieldVector<double,domainDim>& quadPos = quad[pt].position();

    // evaluate actual derivative
    Dune::FieldMatrix<double, embeddedDim, domainDim> derivative = f.evaluateDerivative(quadPos);

    // evaluate fd approximation of derivative
    Dune::FieldMatrix<double, embeddedDim, domainDim> fdDerivative = evaluateDerivativeFD(f,quadPos);

    Dune::FieldMatrix<double, embeddedDim, domainDim> diff = derivative;
    diff -= fdDerivative;

    if ( diff.infinity_norm() > 100*eps ) {
      std::cout << className<TargetSpace>() << ": Analytical gradient does not match fd approximation." << std::endl;
      std::cout << "Analytical: " << derivative << std::endl;
      std::cout << "FD        : " << fdDerivative << std::endl;
    }

  }
}


template <int domainDim, class TargetSpace>
void testDerivativeOfValueWRTCoefficients(const LocalGeodesicFEFunction<domainDim,double,typename LagrangeLocalFiniteElementCache<double,double,domainDim,1>::FiniteElementType, TargetSpace>& f)
{
  static const int embeddedDim = TargetSpace::EmbeddedTangentVector::dimension;

  // A quadrature rule as a set of test points
  int quadOrder = 3;

  const Dune::QuadratureRule<double, domainDim>& quad
    = Dune::QuadratureRules<double, domainDim>::rule(f.type(), quadOrder);

  for (size_t pt=0; pt<quad.size(); pt++) {

    Dune::FieldVector<double,domainDim> quadPos = quad[pt].position();

    // loop over the coefficients
    for (size_t i=0; i<f.size(); i++) {

      // evaluate actual derivative
      FieldMatrix<double, embeddedDim, embeddedDim> derivative;
      f.evaluateDerivativeOfValueWRTCoefficient(quadPos, i, derivative);

      // evaluate fd approximation of derivative
      FieldMatrix<double, embeddedDim, embeddedDim> fdDerivative;
      f.evaluateFDDerivativeOfValueWRTCoefficient(quadPos, i, fdDerivative);

      if ( (derivative - fdDerivative).infinity_norm() > eps ) {
        std::cout << className<TargetSpace>() << ": Analytical derivative of value does not match fd approximation." << std::endl;
        std::cout << "coefficient: " << i << std::endl;
        std::cout << "quad pos: " << quadPos << std::endl;
        std::cout << "gfe: ";
        for (size_t j=0; j<f.size(); j++)
          std::cout << ",   " << f.coefficient(j);
        std::cout << std::endl;
        std::cout << "Analytical:\n " << derivative << std::endl;
        std::cout << "FD        :\n " << fdDerivative << std::endl;
        assert(false);
      }

    }

  }
}

template <int domainDim, class TargetSpace>
void testDerivativeOfGradientWRTCoefficients(const LocalGeodesicFEFunction<domainDim,double,typename LagrangeLocalFiniteElementCache<double,double,domainDim,1>::FiniteElementType, TargetSpace>& f)
{
  static const int embeddedDim = TargetSpace::EmbeddedTangentVector::dimension;

  // A quadrature rule as a set of test points
  int quadOrder = 3;

  const Dune::QuadratureRule<double, domainDim>& quad
    = Dune::QuadratureRules<double, domainDim>::rule(f.type(), quadOrder);

  for (size_t pt=0; pt<quad.size(); pt++) {

    const Dune::FieldVector<double,domainDim>& quadPos = quad[pt].position();

    // loop over the coefficients
    for (size_t i=0; i<f.size(); i++) {

      // evaluate actual derivative
      Tensor3<double, embeddedDim, embeddedDim, domainDim> derivative;
      f.evaluateDerivativeOfGradientWRTCoefficient(quadPos, i, derivative);

      // evaluate fd approximation of derivative
      Tensor3<double, embeddedDim, embeddedDim, domainDim> fdDerivative;
      f.evaluateFDDerivativeOfGradientWRTCoefficient(quadPos, i, fdDerivative);

      if ( (derivative - fdDerivative).infinity_norm() > eps ) {
        std::cout << className<TargetSpace>() << ": Analytical derivative of gradient does not match fd approximation." << std::endl;
        std::cout << "coefficient: " << i << std::endl;
        std::cout << "quad pos: " << quadPos << std::endl;
        std::cout << "gfe: ";
        for (size_t j=0; j<f.size(); j++)
          std::cout << ",   " << f.coefficient(j);
        std::cout << std::endl;
        std::cout << "Analytical:\n " << derivative << std::endl;
        std::cout << "FD        :\n " << fdDerivative << std::endl;
        assert(false);
      }

    }

  }
}


template <class TargetSpace, int domainDim>
void test(const GeometryType& element)
{
  std::cout << " --- Testing " << className<TargetSpace>() << ", domain dimension: " << element.dim() << " ---" << std::endl;

  std::vector<TargetSpace> testPoints;
  ValueFactory<TargetSpace>::get(testPoints);

  int nTestPoints = testPoints.size();
  size_t nVertices = Dune::ReferenceElements<double,domainDim>::general(element).size(domainDim);

  // Set up elements of the target space
  std::vector<TargetSpace> corners(nVertices);

  MultiIndex index(nVertices, nTestPoints);
  int numIndices = index.cycle();

  for (int i=0; i<numIndices; i++, ++index) {

    for (size_t j=0; j<nVertices; j++)
      corners[j] = testPoints[index[j]];

    if (diameter(corners) > 0.5*M_PI)
      continue;

    // Make local gfe function to be tested
    LagrangeLocalFiniteElementCache<double,double,domainDim,1> feCache;
    typedef typename LagrangeLocalFiniteElementCache<double,double,domainDim,1>::FiniteElementType LocalFiniteElement;

    LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement,TargetSpace> f(feCache.get(element),corners);

    //testPermutationInvariance(corners);
    testDerivative<domainDim>(f);
    testDerivativeOfValueWRTCoefficients<domainDim>(f);
    testDerivativeOfGradientWRTCoefficients<domainDim>(f);

  }

}


int main()
{
  // choke on NaN -- don't enable this by default, as there are
  // a few harmless NaN in the loopsolver
  //feenableexcept(FE_INVALID);

  std::cout << std::setw(15) << std::setprecision(12);

  ////////////////////////////////////////////////////////////////
  //  Test functions on 1d elements
  ////////////////////////////////////////////////////////////////

  test<RealTuple<double,1>,1>(GeometryTypes::simplex(1));
  test<UnitVector<double,2>,1>(GeometryTypes::simplex(1));
  test<UnitVector<double,3>,1>(GeometryTypes::simplex(1));
  test<Rotation<double,3>,1>(GeometryTypes::simplex(1));
  typedef Dune::GFE::ProductManifold<RealTuple<double,1>,Rotation<double,3>,UnitVector<double,2> > CrazyManifold;
  test<CrazyManifold,1>(GeometryTypes::simplex(1));

  ////////////////////////////////////////////////////////////////
  //  Test functions on 2d simplex elements
  ////////////////////////////////////////////////////////////////

  test<RealTuple<double,1>,2>(GeometryTypes::simplex(2));
  test<UnitVector<double,2>,2>(GeometryTypes::simplex(2));
  test<UnitVector<double,3>,2>(GeometryTypes::simplex(2));
  test<Rotation<double,3>,2>(GeometryTypes::simplex(2));
  typedef Dune::GFE::ProductManifold<RealTuple<double,1>,Rotation<double,3>,UnitVector<double,2> > CrazyManifold;
  test<CrazyManifold,2>(GeometryTypes::simplex(2));

  ////////////////////////////////////////////////////////////////
  //  Test functions on 2d quadrilateral elements
  ////////////////////////////////////////////////////////////////

  test<RealTuple<double,1>,2>(GeometryTypes::cube(2));
  test<UnitVector<double,2>,2>(GeometryTypes::cube(2));
  test<UnitVector<double,3>,2>(GeometryTypes::cube(2));
  test<Rotation<double,3>,2>(GeometryTypes::cube(2));
  typedef Dune::GFE::ProductManifold<RealTuple<double,1>,Rotation<double,3>,UnitVector<double,2> > CrazyManifold;
  test<CrazyManifold,2>(GeometryTypes::cube(2));

}
