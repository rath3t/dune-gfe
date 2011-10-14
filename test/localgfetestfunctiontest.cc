#include <config.h>

#include <fenv.h>
#include <iostream>

#include <dune/common/fvector.hh>
#include <dune/localfunctions/lagrange/pqkfactory.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/unitvector.hh>

#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/localgfetestfunction.hh>

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

    MultiIndex<domainDim+1> index(nTestPoints);
    int numIndices = index.cycle();
    
    PQkLocalFiniteElementCache<double,double,domainDim,1> feCache;
    typedef typename PQkLocalFiniteElementCache<double,double,domainDim,1>::FiniteElementType LocalFiniteElement;
    
    GeometryType simplex;
    simplex.makeSimplex(domainDim);

    for (int i=0; i<numIndices; i++, ++index) {
        
        for (int j=0; j<domainDim+1; j++)
            coefficients[j] = testPoints[index[j]];

        LocalGFETestFunction<domainDim,double,LocalFiniteElement,TargetSpace> testFunctionSet(feCache.get(simplex),coefficients);
        
        FieldVector<double,domainDim> stupidTestPoint(0);
        std::vector<typename TargetSpace::EmbeddedTangentVector> values;
        testFunctionSet.evaluateFunction(stupidTestPoint, values);
    }

}


int main()
{
    // choke on NaN -- don't enable this by default, as there are
    // a few harmless NaN in the loopsolver
    //feenableexcept(FE_INVALID);

    std::cout << std::setw(15) << std::setprecision(12);
    
    test<RealTuple<1>, 1>();
    
    test<UnitVector<2>, 1>();
    test<UnitVector<3>, 1>();
    test<UnitVector<2>, 2>();
    test<UnitVector<3>, 2>();
        
    test<Rotation<3,double>, 1>();
    test<Rotation<3,double>, 2>();
}
