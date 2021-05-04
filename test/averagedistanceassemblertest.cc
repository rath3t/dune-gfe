#include <config.h>

#include <iostream>

#include <dune/common/fvector.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/geometry/type.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/unitvector.hh>
#include <dune/gfe/productmanifold.hh>

#include <dune/gfe/localgeodesicfefunction.hh>

// Domain dimension
const int dim = 2;

using namespace Dune;

// Compute FD approximations to the gradient and the Hesse matrix
template <class DistanceAssembler, class TargetSpace>
void assembleGradientAndHessianApproximation(const DistanceAssembler& assembler,
                                             const TargetSpace& argument,
                                             typename TargetSpace::TangentVector& gradient,
                                             FieldMatrix<double, TargetSpace::TangentVector::dimension, TargetSpace::TangentVector::dimension>& hesseMatrix)
{
    using field_type = typename TargetSpace::field_type;
    constexpr auto blocksize = TargetSpace::TangentVector::dimension;
    constexpr auto embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension;

    const field_type eps = 1e-4;

    FieldMatrix<double,blocksize,embeddedBlocksize> B = argument.orthonormalFrame();

    // Precompute negative energy at the current configuration
    // (negative because that is how we need it as part of the 2nd-order fd formula)
    field_type centerValue   = -assembler.value(argument);

    // Precompute energy infinitesimal corrections in the directions of the local basis vectors
    std::array<field_type,blocksize> forwardEnergy;
    std::array<field_type,blocksize> backwardEnergy;

    for (size_t i2=0; i2<blocksize; i2++)
    {
        typename TargetSpace::EmbeddedTangentVector epsXi = B[i2];
        epsXi *= eps;
        typename TargetSpace::EmbeddedTangentVector minusEpsXi = epsXi;
        minusEpsXi  *= -1;

        TargetSpace forwardSolution  = argument;
        TargetSpace backwardSolution = argument;

        forwardSolution  = TargetSpace::exp(argument,epsXi);
        backwardSolution = TargetSpace::exp(argument,minusEpsXi);

        forwardEnergy[i2]  = assembler.value(forwardSolution);
        backwardEnergy[i2] = assembler.value(backwardSolution);
    }

    //////////////////////////////////////////////////////////////
    //   Compute gradient by finite-difference approximation
    //////////////////////////////////////////////////////////////

    for (int j=0; j<blocksize; j++)
        gradient[j] = (forwardEnergy[j] - backwardEnergy[j]) / (2*eps);

    ///////////////////////////////////////////////////////////////////////////
    //   Compute Riemannian Hesse matrix by finite-difference approximation.
    //   We loop over the lower left triangular half of the matrix.
    //   The other half follows from symmetry.
    ///////////////////////////////////////////////////////////////////////////
    for (size_t i2=0; i2<blocksize; i2++)
    {
        for (size_t j2=0; j2<i2+1; j2++)
        {
            TargetSpace forwardSolutionXiEta   = argument;
            TargetSpace backwardSolutionXiEta  = argument;

            typename TargetSpace::EmbeddedTangentVector epsXi  = B[i2];    epsXi *= eps;
            typename TargetSpace::EmbeddedTangentVector epsEta = B[j2];   epsEta *= eps;

            typename TargetSpace::EmbeddedTangentVector minusEpsXi  = epsXi;   minusEpsXi  *= -1;
            typename TargetSpace::EmbeddedTangentVector minusEpsEta = epsEta;  minusEpsEta *= -1;

            forwardSolutionXiEta  = TargetSpace::exp(argument, epsXi+epsEta);
            backwardSolutionXiEta = TargetSpace::exp(argument, minusEpsXi+minusEpsEta);

            field_type forwardValue  = assembler.value(forwardSolutionXiEta) - forwardEnergy[i2] - forwardEnergy[j2];
            field_type backwardValue = assembler.value(backwardSolutionXiEta) - backwardEnergy[i2] - backwardEnergy[j2];

            hesseMatrix[i2][j2] = hesseMatrix[j2][i2] = 0.5 * (forwardValue - 2*centerValue + backwardValue) / (eps*eps);
        }
    }
}



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
    typename TargetSpace::TangentVector gradientApproximation;

    // test the hessian
    FieldMatrix<double, TargetSpace::TangentVector::dimension, TargetSpace::TangentVector::dimension> hessian;
    FieldMatrix<double, TargetSpace::TangentVector::dimension, TargetSpace::TangentVector::dimension> hessianApproximation(0);

    assembler.assembleHessian(argument, hessian);
    assembleGradientAndHessianApproximation(assembler, argument,
                                            gradientApproximation, hessianApproximation);

    // Check gradient
    for (size_t i=0; i<gradient.size(); i++)
    {
        if (std::isnan(gradient[i]))
            DUNE_THROW(Dune::Exception, "Gradient contains NaN");
        if (std::isnan(gradientApproximation[i]))
            DUNE_THROW(Dune::Exception, "Gradient approximation contains NaN");
        if (std::abs(gradient[i] - gradientApproximation[i]) > 1e-6)
            DUNE_THROW(Dune::Exception, "Gradient and its approximation do not match");
    }

    // Check Hesse matrix
    for (size_t i=0; i<hessian.N(); i++)
        for (size_t j=0; j<hessian.M(); j++)
        {
            if (std::isnan(hessian[i][j]))
                DUNE_THROW(Dune::Exception, "Hesse matrix contains NaN");
            if (std::isnan(hessianApproximation[i][j]))
                DUNE_THROW(Dune::Exception, "Hesse matrix approximation contains NaN");
            if (std::abs(hessian[i][j] - hessianApproximation[i][j]) > 1e-6)
                DUNE_THROW(Dune::Exception, "Hesse matrix and its approximation do not match");
        }

}


template <class TargetSpace>
void testWeightSet(const std::vector<TargetSpace>& corners, 
                   const TargetSpace& argument)
{
    // A quadrature rule as a set of test points
    int quadOrder = 3;
    
    const auto& quad = QuadratureRules<double, dim>::rule(GeometryTypes::simplex(dim), quadOrder);
    
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

    corners[0] = {1,0,0};
    corners[1] = {0,1,0};
    corners[2] = {0,0,1};

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

    std::vector<TargetSpace> corners(dim+1);
    corners[0] = Rotation<double,3>({1,0,0}, 0.1);
    corners[1] = Rotation<double,3>({0,1,0}, 0.1);
    corners[2] = Rotation<double,3>({0,0,1}, 0.1);

    TargetSpace argument = corners[0];
    testWeightSet(corners, argument);
    argument = corners[1];
    testWeightSet(corners, argument);
    argument = corners[2];
    testWeightSet(corners, argument);
}


void testProductManifold()
{
    typedef Dune::GFE::ProductManifold<RealTuple<double,5>,UnitVector<double,3>, Rotation<double,3>> TargetSpace;

    std::vector<TargetSpace> corners(dim+1);

    std::generate(corners.begin(), corners.end(), []()  {
        return Dune::GFE::randomFieldVector<typename TargetSpace::field_type,TargetSpace::CoordinateType::dimension>(0.9,1.1);
    });

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
    testProductManifold();
}
