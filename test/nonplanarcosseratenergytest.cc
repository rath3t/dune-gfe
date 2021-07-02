#include "config.h"

#include <dune/foamgrid/foamgrid.hh>

#include <dune/geometry/type.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>

#include <dune/gfe/cosseratvtkwriter.hh>
#include <dune/gfe/nonplanarcosseratshellenergy.hh>
#include <dune/gfe/rigidbodymotion.hh>

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

    FieldVector<double,dimworld> vertex0{0,0,0};
    FieldVector<double,dimworld> vertex1{0,1,0};
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

TargetSpace getConfiguration(const FieldVector<double, dimworld>& point) {
    FieldVector<double, dimworld> displacementAtPoint(0);
    FieldVector<double,4> rotationVectorAtPoint(0);

    if (point[0] == 0 and point[1] == 0 and point[2] == 0) {
        //0 0 0
        displacementAtPoint = {0, 0, 0};
        rotationVectorAtPoint = {0, 0, 0, 1};
    } else if (point[0] == 1 and point[1] == 0 and point[2] == 0) {
        //1 0 0
        displacementAtPoint = {0, 0, 1};
        rotationVectorAtPoint = {0, 0, 0, 1};
    } else if (point[0] == 0 and point[1] == 1 and point[2] == 0) {
        //0 1 0
        displacementAtPoint = {0, 0, 1};
        rotationVectorAtPoint = {0, 0, 0, 1};
    } else if (point[0] == 0.5 and point[1] == 0 and point[2] == 0) {
        //0.5 0 0
        displacementAtPoint = {0, 0, 0.5};
        rotationVectorAtPoint = {0, 0, 0, 1};
    } else if (point[0] == 0 and point[1] == 0.5 and point[2] == 0) {
        //0 0.5 0
        displacementAtPoint = {0, 0, 0.5};
        rotationVectorAtPoint = {0, 0, 0, 1};
    } else if (point[0] == 0.5 and point[1] == 0.5 and point[2] == 0) {
        //0.5 0.5 0
        displacementAtPoint = {0, 0, 1};
        rotationVectorAtPoint = {0, 0, 0, 1};
    }

    TargetSpace configuration;
    for (int i = 0; i < dimworld; i++)
        configuration.r[i] = point[i] + displacementAtPoint[i];

    Rotation<double,dimworld> rotation(rotationVectorAtPoint);
    FieldMatrix<double,dimworld,dimworld> rotationMatrix(0);
    rotation.matrix(rotationMatrix);
    configuration.q.set(rotationMatrix);

    return configuration;
}

double calculateEnergy(const int numLevels)
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

    using FEBasis = Dune::Functions::LagrangeBasis<typename GridType::LeafGridView,1>;
    FEBasis feBasis(gridView);

    using namespace Dune::Functions::BasisFactory;
    using namespace Dune::TypeTree::Indices;

    auto deformationPowerBasis = makeBasis(
        gridView,
        power<dimworld>(
            lagrange<1>()
    ));

    BlockVector<FieldVector<double,3> > helperVector;
    Dune::Functions::interpolate(deformationPowerBasis, helperVector, [](FieldVector<double,dimworld> x){
        auto out = x;
        out[2] += x[0];
        return out;
    });
    //Dune::Functions::interpolate(deformationPowerBasis, helperVector, [](FieldVector<double,dimworld> x){ return x; });

    auto stressFreeConfiguration = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,dimworld>>(deformationPowerBasis, helperVector);

    NonplanarCosseratShellEnergy<FEBasis, 3, double, decltype(stressFreeConfiguration)> localCosseratEnergyPlanar(materialParameters,
                                                                                                    &stressFreeConfiguration,
                                                                                                    nullptr,
                                                                                                    nullptr,
                                                                                                    nullptr);

    BlockVector<FieldVector<double,3> > id;
    Dune::Functions::interpolate(deformationPowerBasis, id, [](FieldVector<double,dimworld> x){ return x; });
    BlockVector<TargetSpace> sol(feBasis.size());
    TupleVector<std::vector<RealTuple<double,3> >,
                std::vector<Rotation<double,3> > > solTuple;
    solTuple[_0].resize(feBasis.size());
    solTuple[_1].resize(feBasis.size());
    for (int i = 0; i < feBasis.size(); i++) {
        sol[i] = getConfiguration(id[i]);
        solTuple[_0][i] = sol[i].r;
        solTuple[_1][i] = sol[i].q;
    }

    CosseratVTKWriter<GridType>::write<FEBasis>(feBasis, solTuple, "configuration_l" + std::to_string(numLevels));

    double energy = 0;

    // A view on the FE basis on a single element
    auto localView = feBasis.localView();

    // Loop over all elements
    for (const auto& element : elements(feBasis.gridView(), Dune::Partitions::interior))
    {
        localView.bind(element);

        // Number of degrees of freedom on this element
        size_t nDofs = localView.tree().size();

        std::vector<TargetSpace> localSolution(nDofs);

        for (size_t i=0; i<nDofs; i++)
            localSolution[i] = sol[localView.index(i)[0]];

        energy += localCosseratEnergyPlanar.energy(localView, localSolution);

    }

    return energy;
}

int main(int argc, char** argv)
{
    MPIHelper::instance(argc, argv);

    double energyFine = calculateEnergy(2);
    double energyCoarse = calculateEnergy(1);
    std::cout << "energyFine: " << energyFine << std::endl;
    std::cout << "energyCoarse: " << energyCoarse << std::endl;


    assert(std::abs(energyFine - energyCoarse) < 1e-3);
}
