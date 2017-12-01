#include <config.h>

#include <fenv.h>
#include <array>
#include <iostream>

#include <dune/common/fvector.hh>

#include <dune/fufem/functionspacebases/p1nodalbasis.hh>

#include <dune/grid/onedgrid.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/unitvector.hh>

#include <dune/gfe/globalgfetestfunctionbasis.hh>

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

    // make a 1-D grid
    OneDGrid grid(nTestPoints,-2.0,2.0);

    // make global basis
    typedef P1NodalBasis<typename OneDGrid::LeafGridView> P1Basis;
    P1Basis p1Basis(grid.leafGridView());
    typedef GlobalGFETestFunctionBasis<P1Basis,TargetSpace> GlobalBasis;
    GlobalBasis basis(p1Basis,testPoints);

    typedef typename OneDGrid::Codim<0>::LeafIterator ElementIterator; 
    ElementIterator eIt = grid.leafbegin<0>();
    ElementIterator eEndIt = grid.leafend<0>();
    
    for (; eIt != eEndIt; ++eIt) {
        
        const typename GlobalBasis::LocalFiniteElement& lfe = basis.getLocalFiniteElement(*eIt);     
    
        FieldVector<double,1> stupidTestPoint(0);
        std::vector<std::array<typename TargetSpace::EmbeddedTangentVector, TargetSpace::TangentVector::dimension> > values;
        lfe.localBasis().evaluateFunction(stupidTestPoint, values);
        for(size_t i=0;i<values.size();i++) {
            std::cout<<values[i]<<std::endl;
            std::cout<<lfe.localCoefficients().localKey(i)<<std::endl;
        }
        //int i = basis.index(*eIt,1);
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
    //test<UnitVector<double,2>, 2>();
    //test<UnitVector<double,3>, 2>();
        
    test<Rotation<double,3>, 1>();
    //test<Rotation<double,3>, 2>();

} catch (Exception e) {
    std::cout << e << std::endl;
}

