#include <config.h>

#include <fenv.h>
#include <array>

#include <dune/common/typetraits.hh>
#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>

#include <dune/curvedgeometry/curvedgeometry.hh>
#include <dune/curvedgrid/grid.hh>
#include <dune/curvedgrid/gridfunctions/analyticdiscretefunction.hh>
#include <dune/curvedgrid/geometries/cylinder.hh>

#include <dune/grid/yaspgrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/gfe/linearalgebra.hh>
#include <dune/grid/io/file/vtk/subsamplingvtkwriter.hh>

// grid dimension
const int gridDim = 2;
const int dimWorld = 3;
const int approximationOrderGeometry = 4;
const int approximationOrderAnalytics = 6;
const int elementsCircle = 9;

const auto pi = std::acos(-1.0);

using namespace Dune;

int main (int argc, char *argv[])
{
    MPIHelper::instance(argc,argv);
    // Cylinder
    static const double radius            = 4;
    static const double height            = 2;
    static const double cylinderFraction  = 0.75;

    /////////////////////////////////////////////
    //    Create the grid for the cylinder
    /////////////////////////////////////////////

    struct CylinderCreator
            : public Dune :: AnalyticalCoordFunction< double, 2, 3, CylinderCreator >
    {
        FieldVector<double,3> operator() ( const FieldVector<double, 2> &x ) const
        {
            FieldVector<double,3> y;
            y[0] = radius * std::cos(x[0]);
            y[1] = radius * std::sin(x[0]);
            y[2] = x[1];
            return y;
        }
    };

    using GridType = Dune::GeometryGrid< YaspGrid<gridDim>, CylinderCreator>;
    std::shared_ptr<YaspGrid<gridDim>> hostGrid;
    hostGrid = Dune::StructuredGridFactory<Dune::YaspGrid<2>>::createCubeGrid({0, 0}, {cylinderFraction*2*pi, height}, {elementsCircle,2});
    auto cylinderCreator = std::make_shared<CylinderCreator>();
    auto grid = std::make_shared<GridType>(hostGrid, cylinderCreator);
    auto gridView = grid->leafGridView();

    auto cylinder = CylinderProjection<double>{radius};
    auto polynomialCylinderGF = analyticDiscreteFunction(cylinder, *grid, approximationOrderAnalytics);

    auto analyticCylinderGF = cylinderGridFunction<GridType>(radius);
    auto cylinderLocalFunction = localFunction(analyticCylinderGF);

    auto quadOrder = 10;
    for (const auto& element : elements(gridView, Dune::Partitions::interior)) {
        using DT = decltype(gridView)::ctype;

        Dune::CurvedGeometry<DT, gridDim, dimWorld, Dune::CurvedGeometryTraits<DT, Dune::LagrangeLFECache<DT,DT,gridDim>>>
        polynomialGeometry(Dune::referenceElement(element.geometry()), [element, polynomialCylinderGF](const auto& local) {
            auto localGridFunction = localFunction(polynomialCylinderGF);
            localGridFunction.bind(element);
            return localGridFunction(local);
        }, approximationOrderGeometry);

        cylinderLocalFunction.bind(element);
        auto localGeometry = Dune::DefaultLocalGeometry<double,2,2>{};
        auto analyticGeometry = Dune::LocalFunctionGeometry{element.type(), cylinderLocalFunction, localGeometry};
        Dune::LagrangeLFECache<DT,DT,gridDim> cache(approximationOrderAnalytics);
        auto refEle = referenceElement(element);
        auto lFE = cache.get(refEle.type());

        const auto& quad = Dune::QuadratureRules<DT, gridDim>::rule(element.type(), quadOrder);
        for (size_t pt=0; pt<quad.size(); pt++) {
            // Check if mean curvature is correct
            auto realMeanCurvature = std::abs(cylinder.mean_curvature(polynomialGeometry.global(quad[pt].position())));

            auto normalDerivativeP = polynomialGeometry.normalGradient(quad[pt].position());
            auto approximatedCurvatureP = 0.5*std::abs(Dune::GFE::trace(normalDerivativeP));
            auto relativeDifferenceP = std::abs((realMeanCurvature - approximatedCurvatureP)/realMeanCurvature);

            auto normalDerivativeA = analyticGeometry.normalGradient(quad[pt].position(), lFE);
            auto approximatedCurvatureA = 0.5*std::abs(Dune::GFE::trace(normalDerivativeA));
            std::cout << approximatedCurvatureA << std::endl;
            auto relativeDifferenceA = std::abs((realMeanCurvature - approximatedCurvatureA)/realMeanCurvature);

            if (relativeDifferenceP > 0.005){
                std::cerr << "At point " << polynomialGeometry.global(quad[pt].position()) << " the curvature (approximated using a polynomial) is "
                    << approximatedCurvatureP << " but " << realMeanCurvature << " was expected!" << std::endl;
                return 1;
            }
            if (relativeDifferenceA > 0.005){
                std::cerr << "At point " << polynomialGeometry.global(quad[pt].position()) << " the curvature (approximated using the real analytic cylinder function) is "
                    << approximatedCurvatureA << " but " << realMeanCurvature << " was expected!" << std::endl;
                return 1;
            }
        }
    }
}
