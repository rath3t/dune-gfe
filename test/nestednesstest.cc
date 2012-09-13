#include <config.h>

#include <fenv.h>
#include <iostream>

#include <dune/common/fvector.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/localfunctions/lagrange/pqkfactory.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/unitvector.hh>

#include <dune/gfe/localgeodesicfefunction.hh>
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

template <int domainDim, int elementOrder>
std::vector<FieldVector<double,domainDim> > lagrangeNodes(const GeometryType& type)
{
    std::vector<FieldVector<double,domainDim> > result;
    
    // Special case d=1, q=anything, because that one is easy
    if (domainDim==1) {
     
        result.resize(elementOrder+1);
        
        for (size_t i=0; i<result.size(); i++)
            result[i] = double(i)/double(elementOrder);
        
        return result;
    }
    
    PQkLocalFiniteElementCache<double,double,domainDim,elementOrder> feCache;
    result.resize(feCache.get(type).localBasis().size());
    
    // evaluate loF at the Lagrange points of the second-order function
    const Dune::GenericReferenceElement<double,domainDim>& refElement
        = Dune::GenericReferenceElements<double,domainDim>::general(type);
        
    for (size_t i=0; i<feCache.get(type).localBasis().size(); i++) {
        
        result[i] = refElement.position(feCache.get(type).localCoefficients().localKey(i).subEntity(),
                                        feCache.get(type).localCoefficients().localKey(i).codim());
        
    }
    
    return result;
}


template <int domainDim, class TargetSpace, int highElementOrder>
void testNestedness(const LocalGeodesicFEFunction<domainDim,double,typename PQkLocalFiniteElementCache<double,double,domainDim,1>::FiniteElementType, TargetSpace>& loF)
{
    // Make higher order local gfe function 
    PQkLocalFiniteElementCache<double,double,domainDim,highElementOrder> feCache;
    typedef typename PQkLocalFiniteElementCache<double,double,domainDim,highElementOrder>::FiniteElementType LocalFiniteElement;
    
    // Get the Lagrange nodes of the high-order function
    std::vector<FieldVector<double,domainDim> > lNodes = lagrangeNodes<domainDim,highElementOrder>(loF.type());
    
    // Evaluate low-order function at the high-order nodal values
    std::vector<TargetSpace> nodalValues(lNodes.size());
    for (size_t i=0; i<lNodes.size(); i++)
        nodalValues[i] = loF.evaluate(lNodes[i]);
    
    LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement,TargetSpace> hoF(feCache.get(loF.type()),nodalValues);

    
    // A quadrature rule as a set of test points
    int quadOrder = 10;  // high, I want many many points
    
    const Dune::QuadratureRule<double, domainDim>& quad 
        = Dune::QuadratureRules<double, domainDim>::rule(loF.type(), quadOrder);
    
    for (size_t pt=0; pt<quad.size(); pt++) {
        
        const Dune::FieldVector<double,domainDim>& quadPos = quad[pt].position();

        // evaluate value of low-order function
        TargetSpace loValue = loF.evaluate(quadPos);

        // evaluate value of high-order function
        TargetSpace hoValue = hoF.evaluate(quadPos);

        typename TargetSpace::CoordinateType diff = loValue.globalCoordinates();
        diff -= hoValue.globalCoordinates();
        
        static double maxDiff = 0;
        maxDiff = std::max(maxDiff, diff.infinity_norm());
        
        if (maxDiff > 0.2)
            assert(false);
        
        if ( diff.infinity_norm() > eps ) {
            std::cout << className<TargetSpace>() << ": Values doe not match." << std::endl;
            std::cout << "Low order : " << loValue << std::endl;
            std::cout << "High order: " << hoValue << std::endl;
            assert(false);
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
    size_t nVertices = Dune::GenericReferenceElements<double,domainDim>::general(element).size(domainDim);

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
    
        LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement,TargetSpace> f(feCache.get(element),corners);

        static const int highElementOrder = 2;
        testNestedness<domainDim,TargetSpace,highElementOrder>(f);
                
    }

}


int main()
{
    // choke on NaN -- don't enable this by default, as there are
    // a few harmless NaN in the loopsolver
    //feenableexcept(FE_INVALID);

    std::cout << std::setw(15) << std::setprecision(12);
    
    GeometryType element;

    ////////////////////////////////////////////////////////////////
    //  Test functions on 1d elements
    ////////////////////////////////////////////////////////////////
    element.makeSimplex(1);

    test<RealTuple<double,1>,1>(element);
    test<UnitVector<double,2>,1>(element);
    test<UnitVector<double,3>,1>(element);
    test<Rotation<double,3>,1>(element);
    test<RigidBodyMotion<double,3>,1>(element);

    ////////////////////////////////////////////////////////////////
    //  Test functions on 2d simplex elements
    ////////////////////////////////////////////////////////////////
    element.makeSimplex(2);

    test<RealTuple<double,1>,2>(element);
    test<UnitVector<double,2>,2>(element);
    test<UnitVector<double,3>,2>(element);
    test<Rotation<double,3>,2>(element);
    test<RigidBodyMotion<double,3>,2>(element);

    ////////////////////////////////////////////////////////////////
    //  Test functions on 2d quadrilateral elements
    ////////////////////////////////////////////////////////////////
    element.makeCube(2);

    test<RealTuple<double,1>,2>(element);
    test<UnitVector<double,2>,2>(element);
    test<UnitVector<double,3>,2>(element);
    test<Rotation<double,3>,2>(element);
    test<RigidBodyMotion<double,3>,2>(element);

}
