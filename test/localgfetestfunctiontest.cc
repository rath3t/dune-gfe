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



template <int domainDim>
void testRealTuples()
{
    std::cout << " --- Testing RealTuple<1>, domain dimension: " << domainDim << " ---" << std::endl;

    typedef RealTuple<1> TargetSpace;

    std::vector<TargetSpace> corners = {TargetSpace(1),
                                        TargetSpace(2),
                                        TargetSpace(3)};

}

template <int domainDim>
void testUnitVector2d()
{
    std::cout << " --- Testing UnitVector<2>, domain dimension: " << domainDim << " ---" << std::endl;

    typedef UnitVector<2> TargetSpace;

    std::vector<TargetSpace> testPoints;
    ValueFactory<TargetSpace>::get(testPoints);
    
    int nTestPoints = testPoints.size();
    
    // Set up elements of S^1
    std::vector<TargetSpace> corners(domainDim+1);
    
    MultiIndex<domainDim+1> index(nTestPoints);
    int numIndices = index.cycle();
    
    for (int i=0; i<numIndices; i++, ++index) {
        
        for (int j=0; j<domainDim+1; j++)
            corners[j] = testPoints[index[j]];

        bool spreadOut = false;
        for (int j=0; j<domainDim+1; j++)
            for (int k=0; k<domainDim+1; k++)
                if (UnitVector<2>::distance(corners[j],corners[k]) > M_PI*0.98)
                    spreadOut = true;
                    
        if (spreadOut)
            continue;
        
    }

}

template <int domainDim>
void testUnitVector3d()
{
    std::cout << " --- Testing UnitVector<3>, domain dimension: " << domainDim << " ---" << std::endl;

    typedef UnitVector<3> TargetSpace;

    std::vector<TargetSpace> testPoints;
    ValueFactory<TargetSpace>::get(testPoints);
    
    int nTestPoints = testPoints.size();
    
    // Set up elements of S^2
    std::vector<TargetSpace> corners(domainDim+1);

    MultiIndex<domainDim+1> index(nTestPoints);
    int numIndices = index.cycle();

    for (int i=0; i<numIndices; i++, ++index) {
        
        for (int j=0; j<domainDim+1; j++)
            corners[j] = testPoints[index[j]];

                  
    }

}

template <int domainDim>
void testRotation()
{
    std::cout << " --- Testing Rotation<3>, domain dimension: " << domainDim << " ---" << std::endl;

    typedef Rotation<3,double> TargetSpace;

    std::vector<Rotation<3,double> > testPoints;
    ValueFactory<Rotation<3,double> >::get(testPoints);
    
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

        LocalGFETestFunction<3,double,LocalFiniteElement,TargetSpace>(feCache.get(simplex),coefficients);
        
    }

}


int main()
{
    // choke on NaN -- don't enable this by default, as there are
    // a few harmless NaN in the loopsolver
    //feenableexcept(FE_INVALID);

    std::cout << std::setw(15) << std::setprecision(12);
    
    testRealTuples<1>();
    testUnitVector2d<1>();
    testUnitVector3d<1>();
    testUnitVector2d<2>();
    testUnitVector3d<2>();
    
    testRotation<1>();
    testRotation<2>();
}
