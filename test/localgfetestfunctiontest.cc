#include <config.h>

#include <fenv.h>
#include <iostream>
#include <array>

#include <dune/common/fvector.hh>
#include <dune/localfunctions/lagrange/pqkfactory.hh>

#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>
#include <dune/gfe/spaces/unitvector.hh>

#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/localgfetestfunctionbasis.hh>

#include "multiindex.hh"
#include "valuefactory.hh"

const double eps = 1e-6;

using namespace Dune;

template <class TargetSpace, int domainDim>
void test()
{
    std::cout << " --- Testing " << className<TargetSpace>() << ", domain dimension: " << domainDim << " ---" << std::endl;

    std::vector<TargetSpace> testPoints;
    ValueFactory<TargetSpace>::get(testPoints);
    
    int nTestPoints = testPoints.size();

    // Set up elements of SO(3)
    std::vector<TargetSpace> coefficients(domainDim+1);

    MultiIndex index(domainDim+1, nTestPoints);
    int numIndices = index.cycle();
    
    PQkLocalFiniteElementCache<double,double,domainDim,1> feCache;
    typedef typename PQkLocalFiniteElementCache<double,double,domainDim,1>::FiniteElementType LocalFiniteElement;
    
    for (int i=0; i<numIndices; i++, ++index) {
        
        for (int j=0; j<domainDim+1; j++)
            coefficients[j] = testPoints[index[j]];

        LocalGfeTestFunctionFiniteElement<LocalFiniteElement,TargetSpace> testFunctionSet(feCache.get(GeometryTypes::simplex(domainDim)),coefficients);
        
        FieldVector<double,domainDim> stupidTestPoint(0);
        
        // test whether evaluation of the shape functions works
        std::vector<std::array<typename TargetSpace::EmbeddedTangentVector, TargetSpace::TangentVector::dimension> > values;
        testFunctionSet.localBasis().evaluateFunction(stupidTestPoint, values);

        // test whether evaluation of the shape function derivatives works
        std::vector<std::array<FieldMatrix<double, TargetSpace::EmbeddedTangentVector::dimension, domainDim>, TargetSpace::TangentVector::dimension> > derivatives;
        testFunctionSet.localBasis().evaluateJacobian(stupidTestPoint, derivatives);

    }

}


int main() try
{
    // choke on NaN -- don't enable this by default, as there are
    // a few harmless NaN in the loopsolver
    //feenableexcept(FE_INVALID);

    std::cout << std::setw(15) << std::setprecision(12);
    
    test<RealTuple<double,1>, 1>();
    
    test<UnitVector<double,2>, 1>();
    test<UnitVector<double,3>, 1>();
    test<UnitVector<double,2>, 2>();
    test<UnitVector<double,3>, 2>();
        
    test<Rotation<double,3>, 1>();
    test<Rotation<double,3>, 2>();
    typedef Dune::GFE::ProductManifold<RealTuple<double,1>,Rotation<double,3>,UnitVector<double,2>> CrazyManifold;
    test<CrazyManifold, 2>();

} catch (Exception& e) {
    std::cout << e.what() << std::endl;
}

