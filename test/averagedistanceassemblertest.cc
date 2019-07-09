#include <config.h>

#include <iostream>

#include <dune/common/fvector.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/geometry/type.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/unitvector.hh>

#include <dune/gfe/localgeodesicfefunction.hh>

// Domain dimension
const int dim = 2;

using namespace Dune;

/** \brief Test whether interpolation is invariant under permutation of the simplex vertices
 */
template <class TargetSpace>
void testPoint(const std::vector<TargetSpace>& corners, 
               const std::vector<double>& weights,
               const TargetSpace& argument)
{
    // create the assembler
    AverageDistanceAssembler<TargetSpace> assembler(corners, weights);

    // test the functional
    double value = assembler.value(argument);
    assert(!std::isnan(value));
    assert(value >= 0);
    
    // test the gradient
    typename TargetSpace::TangentVector gradient;
    assembler.assembleGradient(argument, gradient);

    // test the hessian
    FieldMatrix<double, TargetSpace::TangentVector::dimension, TargetSpace::TangentVector::dimension> hessian;
    FieldMatrix<double, TargetSpace::TangentVector::dimension, TargetSpace::TangentVector::dimension> hessianApproximation(0);

    assembler.assembleHessian(argument, hessian);
    //assembler.assembleHessianApproximation(argument, hessianApproximation);

    for (size_t i=0; i<hessian.N(); i++)
        for (size_t j=0; j<hessian.M(); j++) {
            assert(!std::isnan(hessian[i][j]));
            assert(!std::isnan(hessianApproximation[i][j]));
        }

    std::cout << "WARNING: no approximation of the Hessian available, not testing" << std::endl;
    exit(1);

    FieldMatrix<double, TargetSpace::TangentVector::dimension, TargetSpace::TangentVector::dimension> diff = hessian;
    diff -= hessianApproximation;
    std::cout << "Matrix inf diff: " << diff.infinity_norm() << std::endl;
}


template <class TargetSpace>
void testWeightSet(const std::vector<TargetSpace>& corners, 
                   const TargetSpace& argument)
{
    // A quadrature rule as a set of test points
    int quadOrder = 3;
    
    const Dune::QuadratureRule<double, dim>& quad 
        = Dune::QuadratureRules<double, dim>::rule(GeometryTypes::simplex(dim), quadOrder);
    
    for (size_t pt=0; pt<quad.size(); pt++) {
        
        const Dune::FieldVector<double,dim>& quadPos = quad[pt].position();
        
        // local to barycentric coordinates
        std::vector<double> weights(dim+1);
        weights[0] = 1;
        for (int i=0; i<dim; i++) {
            weights[0] -= quadPos[i];
            weights[i+1] = quadPos[i];
        }

        testPoint(corners, weights, argument);

    }

}


void testRealTuples()
{
    typedef RealTuple<double,1> TargetSpace;

    std::vector<TargetSpace> corners = {TargetSpace(1),
                                        TargetSpace(2),
                                        TargetSpace(3)};

    TargetSpace argument = corners[0];
    testWeightSet(corners, argument);
    argument = corners[1];
    testWeightSet(corners, argument);
    argument = corners[2];
    testWeightSet(corners, argument);
}

void testUnitVectors()
{
    typedef UnitVector<double,3> TargetSpace;

    std::vector<TargetSpace> corners(dim+1);

    FieldVector<double,3> input;
    input[0] = 1;  input[1] = 0;  input[2] = 0;
    corners[0] = input;
    input[0] = 0;  input[1] = 1;  input[2] = 0;
    corners[1] = input;
    input[0] = 0;  input[1] = 0;  input[2] = 1;
    corners[2] = input;

    TargetSpace argument = corners[0];
    testWeightSet(corners, argument);
    argument = corners[1];
    testWeightSet(corners, argument);
    argument = corners[2];
    testWeightSet(corners, argument);
}

void testRotations()
{
    typedef Rotation<double,3> TargetSpace;

    FieldVector<double,3> xAxis(0);
    xAxis[0] = 1;
    FieldVector<double,3> yAxis(0);
    yAxis[1] = 1;
    FieldVector<double,3> zAxis(0);
    zAxis[2] = 1;


    std::vector<TargetSpace> corners(dim+1);
    corners[0] = Rotation<double,3>(xAxis,0.1);
    corners[1] = Rotation<double,3>(yAxis,0.1);
    corners[2] = Rotation<double,3>(zAxis,0.1);

    TargetSpace argument = corners[0];
    testWeightSet(corners, argument);
    argument = corners[1];
    testWeightSet(corners, argument);
    argument = corners[2];
    testWeightSet(corners, argument);
}


int main()
{
    testRealTuples();
    testUnitVectors();
    testRotations();
}
