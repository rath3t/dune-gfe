#include <config.h>

#include <fenv.h>
#include <iostream>

#include <dune/common/fvector.hh>
#include <dune/grid/common/quadraturerules.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/unitvector.hh>

#include <dune/gfe/localgeodesicfefunction.hh>

const double eps = 1e-6;

using namespace Dune;

/** \brief N-dimensional multi-index
*/
template <int N>
class MultiIndex
    : public array<unsigned int,N>
{

    // The range of each component
    unsigned int limit_;

public:
    /** \brief Constructor with a given range for each digit */
    MultiIndex(unsigned int limit)
        : limit_(limit)
    {
        std::fill(this->begin(), this->end(), 0);
    }

    /** \brief Increment the MultiIndex */
    MultiIndex& operator++() {

        for (int i=0; i<N; i++) {

            // Augment digit
            (*this)[i]++;

            // If there is no carry-over we can stop here
            if ((*this)[i]<limit_)
                break;

            (*this)[i] = 0;
                    
        }
        return *this;
    }

    /** \brief Compute how many times you can call operator++ before getting to (0,...,0) again */
    size_t cycle() const {
        size_t result = 1;
        for (int i=0; i<N; i++)
            result *= limit_;
        return result;
    }

};


template <int domainDim>
void testDerivativeTangentiality(const RealTuple<1>& x,
                                 const FieldMatrix<double,1,domainDim>& derivative)
{
    // By construction, derivatives of RealTuples are always tangent
}

// the columns of the derivative must be tangential to the manifold
template <int domainDim, int vectorDim>
void testDerivativeTangentiality(const UnitVector<vectorDim>& x,
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

/** \brief Test whether interpolation is invariant under permutation of the simplex vertices
 */
template <int domainDim, class TargetSpace>
void testPermutationInvariance(const std::vector<TargetSpace>& corners)
{
    // works only for 2d domains
    assert(domainDim==2);
    
    std::vector<TargetSpace> cornersRotated1(domainDim+1);
    std::vector<TargetSpace> cornersRotated2(domainDim+1);

    cornersRotated1[0] = cornersRotated2[2] = corners[1];
    cornersRotated1[1] = cornersRotated2[0] = corners[2];
    cornersRotated1[2] = cornersRotated2[1] = corners[0];
    
    LocalGeodesicFEFunction<2,double,TargetSpace> f0(corners);
    LocalGeodesicFEFunction<2,double,TargetSpace> f1(cornersRotated1);
    LocalGeodesicFEFunction<2,double,TargetSpace> f2(cornersRotated2);

    // A quadrature rule as a set of test points
    int quadOrder = 3;

    const Dune::QuadratureRule<double, domainDim>& quad 
        = Dune::QuadratureRules<double, domainDim>::rule(GeometryType(GeometryType::simplex,domainDim), quadOrder);
    
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
void testDerivative(const std::vector<TargetSpace>& corners)
{
    // Make local fe function to be tested
    LocalGeodesicFEFunction<domainDim,double,TargetSpace> f(corners);

    // A quadrature rule as a set of test points
    int quadOrder = 3;
    
    const Dune::QuadratureRule<double, domainDim>& quad 
        = Dune::QuadratureRules<double, domainDim>::rule(GeometryType(GeometryType::simplex,domainDim), quadOrder);
    
    for (size_t pt=0; pt<quad.size(); pt++) {
        
        const Dune::FieldVector<double,domainDim>& quadPos = quad[pt].position();

        // evaluate actual derivative
        Dune::FieldMatrix<double, TargetSpace::EmbeddedTangentVector::size, domainDim> derivative = f.evaluateDerivative(quadPos);

        // evaluate fd approximation of derivative
        Dune::FieldMatrix<double, TargetSpace::EmbeddedTangentVector::size, domainDim> fdDerivative = f.evaluateDerivativeFD(quadPos);

        Dune::FieldMatrix<double, TargetSpace::EmbeddedTangentVector::size, domainDim> diff = derivative;
        diff -= fdDerivative;
        
        if ( diff.infinity_norm() > 100*eps ) {
            std::cout << className(corners[0]) << ": Analytical gradient does not match fd approximation." << std::endl;
            std::cout << "Analytical: " << derivative << std::endl;
            std::cout << "FD        : " << fdDerivative << std::endl;
        }

        testDerivativeTangentiality(f.evaluate(quadPos), derivative);

    }
}

template <int domainDim, class TargetSpace>
void testDerivativeOfGradientWRTCoefficients(const std::vector<TargetSpace>& corners)
{
    // Make local fe function to be tested
    LocalGeodesicFEFunction<domainDim,double,TargetSpace> f(corners);

    // A quadrature rule as a set of test points
    int quadOrder = 3;
    
    const Dune::QuadratureRule<double, domainDim>& quad 
        = Dune::QuadratureRules<double, domainDim>::rule(GeometryType(GeometryType::simplex,domainDim), quadOrder);
    
    for (size_t pt=0; pt<quad.size(); pt++) {
        
        const Dune::FieldVector<double,domainDim>& quadPos = quad[pt].position();
        
        // loop over the coefficients
        for (size_t i=0; i<corners.size(); i++) {

            // evaluate actual derivative
            Tensor3<double, TargetSpace::EmbeddedTangentVector::size, TargetSpace::EmbeddedTangentVector::size, domainDim> derivative;
            f.evaluateDerivativeOfGradientWRTCoefficient(quadPos, i, derivative);

            // evaluate fd approximation of derivative
            Tensor3<double, TargetSpace::EmbeddedTangentVector::size, TargetSpace::EmbeddedTangentVector::size, domainDim> fdDerivative;

            for (int j=0; j<TargetSpace::EmbeddedTangentVector::size; j++) {
                
                std::vector<TargetSpace> cornersPlus  = corners;
                std::vector<TargetSpace> cornersMinus = corners;
                typename TargetSpace::CoordinateType aPlus  = corners[i].globalCoordinates();
                typename TargetSpace::CoordinateType aMinus = corners[i].globalCoordinates();
                aPlus[j]  += eps;
                aMinus[j] -= eps;
                cornersPlus[i]  = TargetSpace(aPlus);
                cornersMinus[i] = TargetSpace(aMinus);
                LocalGeodesicFEFunction<domainDim,double,TargetSpace> fPlus(cornersPlus);
                LocalGeodesicFEFunction<domainDim,double,TargetSpace> fMinus(cornersMinus);
                
                FieldMatrix<double,TargetSpace::EmbeddedTangentVector::size,domainDim> hPlus  = fPlus.evaluateDerivative(quadPos);
                FieldMatrix<double,TargetSpace::EmbeddedTangentVector::size,domainDim> hMinus = fMinus.evaluateDerivative(quadPos);
        
                fdDerivative[j]  = hPlus;
                fdDerivative[j] -= hMinus;
                fdDerivative[j] /= 2*eps;
                
            }
            
            if ( (derivative - fdDerivative).infinity_norm() > eps ) {
                std::cout << className(corners[0]) << ": Analytical derivative of gradient does not match fd approximation." << std::endl;
                std::cout << "gfe: ";
                for (int j=0; j<domainDim+1; j++)
                    std::cout << ",   " << corners[j];
                std::cout << std::endl;
                std::cout << "Analytical:\n " << derivative << std::endl;
                std::cout << "FD        :\n " << fdDerivative << std::endl;
            }

            //testDerivativeTangentiality(f.evaluate(quadPos), derivative);

        }
        
    }
}


template <int domainDim>
void testRealTuples()
{
    std::cout << " --- Testing RealTuple<1> ---" << std::endl;

    typedef RealTuple<1> TargetSpace;

    std::vector<TargetSpace> corners = {TargetSpace(1),
                                        TargetSpace(2),
                                        TargetSpace(3)};

    testPermutationInvariance<domainDim>(corners);
    testDerivative<domainDim>(corners);
}

template <int domainDim>
void testUnitVector2d()
{
    std::cout << " --- Testing UnitVector<2> ---" << std::endl;

    typedef UnitVector<2> TargetSpace;

    int nTestPoints = 10;
    double testPoints[10][2] = {{1,0}, {0.5,0.5}, {0,1}, {-0.5,0.5}, {-1,0}, {-0.5,-0.5}, {0,-1}, {0.5,-0.5}, {0.1,1}, {1,.1}};
    
    // Set up elements of S^1
    std::vector<TargetSpace> corners(domainDim+1);
    
    MultiIndex<domainDim+1> index(nTestPoints);
    int numIndices = index.cycle();

    for (int i=0; i<numIndices; i++, ++index) {
        
        for (int j=0; j<domainDim+1; j++) {
            Dune::array<double,2> w = {testPoints[index[j]][0], testPoints[index[j]][1]};
            corners[j] = UnitVector<2>(w);
        }

        bool spreadOut = false;
        for (int j=0; j<domainDim+1; j++)
            for (int k=0; k<domainDim+1; k++)
                if (UnitVector<2>::distance(corners[j],corners[k]) > M_PI*0.98)
                    spreadOut = true;
                    
        if (spreadOut)
            continue;
        
        //testPermutationInvariance(corners);
        testDerivative<domainDim>(corners);
        testDerivativeOfGradientWRTCoefficients<domainDim>(corners);
                
    }

}

template <int domainDim>
void testUnitVector3d()
{
    std::cout << " --- Testing UnitVector<3> ---" << std::endl;

    typedef UnitVector<3> TargetSpace;

    int nTestPoints = 10;
    double testPoints[10][3] = {{1,0,0}, {0,1,0}, {-0.838114,0.356751,-0.412667},
                               {-0.490946,-0.306456,0.81551},{-0.944506,0.123687,-0.304319},
                               {-0.6,0.1,-0.2},{0.45,0.12,0.517},
                               {-0.1,0.3,-0.1},{-0.444506,0.123687,0.104319},{-0.7,-0.123687,-0.304319}};
    
    // Set up elements of S^1
    std::vector<TargetSpace> corners(domainDim+1);

    MultiIndex<domainDim+1> index(nTestPoints);
    int numIndices = index.cycle();

    for (int i=0; i<numIndices; i++, ++index) {
        
        for (int j=0; j<domainDim+1; j++) {
            Dune::array<double,3> w = {testPoints[index[j]][0], testPoints[index[j]][1], testPoints[index[j]][2]};
            corners[j] = UnitVector<3>(w);
        }

        //testPermutationInvariance(corners);
        testDerivative<domainDim>(corners);
        testDerivativeOfGradientWRTCoefficients<domainDim>(corners);
                
    }

}

template <int domainDim>
void testRotations()
{
    std::cout << " --- Testing Rotation<3> ---" << std::endl;

    typedef Rotation<3,double> TargetSpace;

    FieldVector<double,3> xAxis(0);
    xAxis[0] = 1;
    FieldVector<double,3> yAxis(0);
    yAxis[1] = 1;
    FieldVector<double,3> zAxis(0);
    zAxis[2] = 1;


    std::vector<TargetSpace> corners(domainDim+1);
    corners[0] = Rotation<3,double>(xAxis,0.1);
    corners[1] = Rotation<3,double>(yAxis,0.1);
    corners[2] = Rotation<3,double>(zAxis,0.1);

    testPermutationInvariance<domainDim>(corners);
    //testDerivative(corners);
}


int main()
{
    // choke on NaN
    feenableexcept(FE_INVALID);

    //testRealTuples();
    testUnitVector2d<2>();
    testUnitVector3d<2>();
    //testRotations();
}
