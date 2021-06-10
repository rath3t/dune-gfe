#include <config.h>

#include <fenv.h>
#include <iostream>
#include <iomanip>

#include <dune/common/fvector.hh>

#include <dune/geometry/quadraturerules.hh>
#include <dune/geometry/type.hh>
#include <dune/geometry/referenceelements.hh>

#include <dune/localfunctions/lagrange/pqkfactory.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/unitvector.hh>

#include <dune/gfe/localprojectedfefunction.hh>
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
    static const int embeddedDim = LocalFunction::TargetSpace::embeddedDim;
    Dune::FieldMatrix<ctype, embeddedDim, dim> result;

    for (int i=0; i<dim; i++) {

        Dune::FieldVector<ctype, dim> forward  = local;
        Dune::FieldVector<ctype, dim> backward = local;

        forward[i]  += eps;
        backward[i] -= eps;

        auto fdDer = f.evaluate(forward).globalCoordinates() - f.evaluate(backward).globalCoordinates();
        fdDer /= 2*eps;

        for (int j=0; j<embeddedDim; j++)
            result[j][i] = fdDer[j];

    }

    return result;
}


template <int domainDim, int dim>
void testDerivativeTangentiality(const RealTuple<double,dim>& x,
                                 const FieldMatrix<double,dim,domainDim>& derivative)
{
    // By construction, derivatives of RealTuples are always tangent
}

// the columns of the derivative must be tangential to the manifold
template <int domainDim, int vectorDim>
void testDerivativeTangentiality(const UnitVector<double,vectorDim>& x,
                                 const FieldMatrix<double,vectorDim,domainDim>& derivative)
{
    for (int i=0; i<domainDim; i++) {

        // The i-th column is a tangent vector if its scalar product with the global coordinates
        // of x vanishes.
        double sp = 0;
        for (int j=0; j<vectorDim; j++)
            sp += x.globalCoordinates()[j] * derivative[j][i];

        if (std::fabs(sp) > 1e-8)
            DUNE_THROW(Dune::Exception, "Derivative is not tangential: Column: " << i << ",  product: " << sp);

    }

}

// the columns of the derivative must be tangential to the manifold
template <int domainDim, int vectorDim>
void testDerivativeTangentiality(const Rotation<double,vectorDim-1>& x,
                                 const FieldMatrix<double,vectorDim,domainDim>& derivative)
{
}

// the columns of the derivative must be tangential to the manifold
template <int domainDim, int vectorDim>
void testDerivativeTangentiality(const RigidBodyMotion<double,3>& x,
                                 const FieldMatrix<double,vectorDim,domainDim>& derivative)
{
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

    PQkLocalFiniteElementCache<double,double,domainDim,1> feCache;
    typedef typename PQkLocalFiniteElementCache<double,double,domainDim,1>::FiniteElementType LocalFiniteElement;

    GeometryType simplex = GeometryTypes::simplex(domainDim);

    //
    std::vector<TargetSpace> cornersRotated1(domainDim+1);
    std::vector<TargetSpace> cornersRotated2(domainDim+1);

    cornersRotated1[0] = cornersRotated2[2] = corners[1];
    cornersRotated1[1] = cornersRotated2[0] = corners[2];
    cornersRotated1[2] = cornersRotated2[1] = corners[0];

    GFE::LocalProjectedFEFunction<2,double,LocalFiniteElement,TargetSpace> f0(feCache.get(simplex), corners);
    GFE::LocalProjectedFEFunction<2,double,LocalFiniteElement,TargetSpace> f1(feCache.get(simplex), cornersRotated1);
    GFE::LocalProjectedFEFunction<2,double,LocalFiniteElement,TargetSpace> f2(feCache.get(simplex), cornersRotated2);

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
void testDerivative(const GFE::LocalProjectedFEFunction<domainDim,double,typename PQkLocalFiniteElementCache<double,double,domainDim,1>::FiniteElementType, TargetSpace>& f)
{
    static const int embeddedDim = TargetSpace::EmbeddedTangentVector::dimension;

    // A quadrature rule as a set of test points
    int quadOrder = 3;

    const auto& quad = Dune::QuadratureRules<double, domainDim>::rule(f.type(), quadOrder);

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
            assert(false);
        }

        testDerivativeTangentiality(f.evaluate(quadPos), derivative);

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
        PQkLocalFiniteElementCache<double,double,domainDim,1> feCache;
        typedef typename PQkLocalFiniteElementCache<double,double,domainDim,1>::FiniteElementType LocalFiniteElement;

        GFE::LocalProjectedFEFunction<domainDim,double,LocalFiniteElement,TargetSpace> f(feCache.get(element),corners);

        //testPermutationInvariance(corners);
        testDerivative<domainDim>(f);

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

    test<RealTuple<double,1>,1>(GeometryTypes::line);
    test<UnitVector<double,2>,1>(GeometryTypes::line);
    test<UnitVector<double,3>,1>(GeometryTypes::line);
    test<Rotation<double,3>,1>(GeometryTypes::line);
    test<RigidBodyMotion<double,3>,1>(GeometryTypes::line);

    ////////////////////////////////////////////////////////////////
    //  Test functions on 2d simplex elements
    ////////////////////////////////////////////////////////////////

    test<RealTuple<double,1>,2>(GeometryTypes::triangle);
    test<UnitVector<double,2>,2>(GeometryTypes::triangle);
    test<RealTuple<double,3>,2>(GeometryTypes::triangle);
    test<UnitVector<double,3>,2>(GeometryTypes::triangle);
    test<Rotation<double,3>,2>(GeometryTypes::triangle);
    test<RigidBodyMotion<double,3>,2>(GeometryTypes::triangle);

    ////////////////////////////////////////////////////////////////
    //  Test functions on 2d quadrilateral elements
    ////////////////////////////////////////////////////////////////

    test<RealTuple<double,1>,2>(GeometryTypes::quadrilateral);
    test<UnitVector<double,2>,2>(GeometryTypes::quadrilateral);
    test<UnitVector<double,3>,2>(GeometryTypes::quadrilateral);
    test<Rotation<double,3>,2>(GeometryTypes::quadrilateral);
    test<RigidBodyMotion<double,3>,2>(GeometryTypes::quadrilateral);

}
