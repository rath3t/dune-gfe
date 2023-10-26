#include "config.h"

#include <math.h>

#include <dune/foamgrid/foamgrid.hh>

#include <dune/geometry/type.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>

#include <dune/gfe/cosseratvtkwriter.hh>
#include <dune/gfe/assemblers/nonplanarcosseratshellenergy.hh>
#include <dune/gfe/spaces/rigidbodymotion.hh>

#include "multiindex.hh"
#include "valuefactory.hh"

using namespace Dune;

static const int dim = 2;
static const int dimworld = 3;

using GridType = FoamGrid<dim,dimworld>;
using TargetSpace = RigidBodyMotion<double,dimworld>;

//////////////////////////////////////////////////////////
//   Make a test grid consisting of a single triangle
//////////////////////////////////////////////////////////

template <class GridType>
std::unique_ptr<GridType> makeSingleElementGrid()
{
    constexpr auto triangle = Dune::GeometryTypes::triangle;
    GridFactory<GridType> factory;

    //Create a triangle that is not parallel to the planes formed by the coordinate axes
    FieldVector<double,dimworld> vertex0{0,0,0};
    FieldVector<double,dimworld> vertex1{0,1,1}; 
    FieldVector<double,dimworld> vertex2{1,0,0};
    factory.insertVertex(vertex0);
    factory.insertVertex(vertex1);
    factory.insertVertex(vertex2);

    factory.insertElement(triangle, {0,1,2});

    return std::unique_ptr<GridType>(factory.createGrid());
}


//////////////////////////////////////////////////////////////////////////////////////
//   Test energy computation for the same grid with different refinement levels
//////////////////////////////////////////////////////////////////////////////////////
template <class F1, class F2>
double calculateEnergy(const int numLevels, const F1 referenceConfigurationFunction, const F2 configurationFunction)
{
    ParameterTree materialParameters;
    materialParameters["thickness"] = "0.1";
    materialParameters["mu"] = "3.8462e+05";
    materialParameters["lambda"] = "2.7149e+05";
    materialParameters["mu_c"] = "0";
    materialParameters["L_c"] = "1e-3";
    materialParameters["q"] = "2.5";
    materialParameters["kappa"] = "0.1";
    materialParameters["b1"] = "1";
    materialParameters["b2"] = "1";
    materialParameters["b3"] = "1";

    const std::unique_ptr<GridType> grid = makeSingleElementGrid<GridType>();
    grid->globalRefine(numLevels-1);
    GridType::LeafGridView gridView = grid->leafGridView();

    using FEBasis = Dune::Functions::LagrangeBasis<typename GridType::LeafGridView,2>;
    FEBasis feBasis(gridView);

    using namespace Dune::Functions::BasisFactory;
    using namespace Dune::TypeTree::Indices;

    auto deformationPowerBasis = makeBasis(
        gridView,
        power<dimworld>(
            lagrange<2>()
    ));

    BlockVector<FieldVector<double,3> > helperVector1(feBasis.size());
    Dune::Functions::interpolate(deformationPowerBasis, helperVector1, referenceConfigurationFunction);
    auto stressFreeConfiguration = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,dimworld>>(deformationPowerBasis, helperVector1);

    NonplanarCosseratShellEnergy<FEBasis, 3, double, decltype(stressFreeConfiguration)> nonplanarCosseratShellEnergy(materialParameters,
                                                                                                    &stressFreeConfiguration,
                                                                                                    nullptr,
                                                                                                    nullptr,
                                                                                                    nullptr);
    BlockVector<TargetSpace> sol(feBasis.size());
    TupleVector<std::vector<RealTuple<double,3> >,
                std::vector<Rotation<double,3> > > solTuple;
    solTuple[_0].resize(feBasis.size());
    solTuple[_1].resize(feBasis.size());

    BlockVector<FieldVector<double,3> > helperVector2(feBasis.size());
    Dune::Functions::interpolate(deformationPowerBasis, helperVector2, configurationFunction);
    for (std::size_t i = 0; i < feBasis.size(); i++) {
        for (int j = 0; j < dimworld; j++)
            sol[i].r[j] = helperVector2[i][j]; 

        FieldVector<double,4> idRotation = {0, 0, 0, 1}; //set rotation = Id everywhere
        Rotation<double,dimworld> rotation(idRotation);
        FieldMatrix<double,dimworld,dimworld> rotationMatrix(0);
        rotation.matrix(rotationMatrix);
        sol[i].q.set(rotationMatrix);
        solTuple[_0][i] = sol[i].r;
        solTuple[_1][i] = sol[i].q;
    }
    CosseratVTKWriter<GridType>::write<FEBasis>(feBasis, solTuple, "configuration_l" + std::to_string(numLevels));

    double energy = 0;
    // A view on the FE basis on a single element
    auto localView = feBasis.localView();
    // Loop over all elements
    for (const auto& element : elements(feBasis.gridView(), Dune::Partitions::interior)) {
        localView.bind(element);
        // Number of degrees of freedom on this element
        size_t nDofs = localView.tree().size();
        std::vector<TargetSpace> localSolution(nDofs);
        for (size_t i=0; i<nDofs; i++)
            localSolution[i] = sol[localView.index(i)[0]];
        energy += nonplanarCosseratShellEnergy.energy(localView, localSolution);
    }
    return energy;
}

int main(int argc, char** argv)
{
    MPIHelper::instance(argc, argv);
    auto configurationId = [](FieldVector<double,dimworld> x){ return x; };
    auto configurationStretchY = [](FieldVector<double,dimworld> x){
        auto out = x;
        out[1] *= 2;
        return out;
    };

    auto configurationTwist = [](FieldVector<double,dimworld> x){
        auto out = x;
        out[1] = x[2];
        out[2] = -x[1];
        return out;
    };

    auto configurationCurved = [](FieldVector<double,dimworld> x){
        auto out = x;
        out[1] = x[2];
        out[2] = -x[1];
        return out;
    };
    auto configurationSquare = [](FieldVector<double,dimworld> x){
        auto out = x;
        out[1] += x[1]*x[1];
        return out;
    };

    auto configurationSin = [](FieldVector<double,dimworld> x){
        auto out = x;
        out[2] = sin(x[2]);
        return out;
    };

    double energyFine = calculateEnergy(2, configurationId, configurationStretchY);
    double energyCoarse = calculateEnergy(1, configurationId, configurationStretchY);
    assert(std::abs(energyFine - energyCoarse) < 1e-3);

    double energyForZeroDifference = calculateEnergy(1,configurationId,configurationId);
    assert(std::abs(energyForZeroDifference) < 1e-3);

    double energyForZeroDifference2 = calculateEnergy(1,configurationStretchY,configurationStretchY);
    assert(std::abs(energyForZeroDifference2) < 1e-3);

    double energyForZeroDifference3 = calculateEnergy(1,configurationTwist,configurationTwist);
    assert(std::abs(energyForZeroDifference3) < 1e-3);

    double energyForZeroDifference4 = calculateEnergy(1,configurationSquare,configurationSquare);
    assert(std::abs(energyForZeroDifference4) < 1e-3);

    double energyForZeroDifference5 = calculateEnergy(1,configurationSin,configurationSin);
    assert(std::abs(energyForZeroDifference5) < 1e-3);
}
