#include <config.h>

#include <dune/grid/onedgrid.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>

#include <dune/gfe/cosseratrodenergy.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/rigidbodymotion.hh>


using namespace Dune;


int main (int argc, char *argv[]) try
{
    // Some types that I need
    typedef std::vector<RigidBodyMotion<double,3> > SolutionType;

    // Problem settings
    const int numRodBaseElements = 100;
    
    // ///////////////////////////////////////
    //    Create the grid
    // ///////////////////////////////////////
    typedef OneDGrid GridType;
    GridType grid(numRodBaseElements, 0, 1);
    using GridView = GridType::LeafGridView;
    GridView gridView = grid.leafGridView();

    using FEBasis = Functions::LagrangeBasis<GridView,1>;
    FEBasis feBasis(gridView);

    SolutionType x(feBasis.size());

    // //////////////////////////
    //   Initial solution
    // //////////////////////////

    for (size_t i=0; i<x.size(); i++)
    {
        double s = double(i)/(x.size()-1);
        x[i].r[0] = 0.1*std::cos(2*M_PI*s);
        x[i].r[1] = 0.1*std::sin(2*M_PI*s);
        x[i].r[2] = s;
        x[i].q = Rotation<double,3>::identity();
        //x[i].q = Quaternion<double>(zAxis, (double(i)*M_PI)/(2*(x.size()-1)) );
    }

    FieldVector<double,3> zAxis(0);  zAxis[2]=1;
    x.back().q = Rotation<double,3>(zAxis, M_PI/4);

    // /////////////////////////////////////////////////////////////////////
    //   Create a second, rotated copy of the configuration
    // /////////////////////////////////////////////////////////////////////

    FieldVector<double,3> displacement {0, 1, 0};

    FieldVector<double,3> axis = {1,0,0};
    Rotation<double,3> rotation(axis,M_PI/2);

    SolutionType rotatedX = x;

    for (size_t i=0; i<rotatedX.size(); i++)
    {
        rotatedX[i].r = rotation.rotate(x[i].r);
        rotatedX[i].r += displacement;

        rotatedX[i].q = rotation.mult(x[i].q);
    }

    using GeodesicInterpolationRule  = LocalGeodesicFEFunction<1, double,
                                                               FEBasis::LocalView::Tree::FiniteElement,
                                                               RigidBodyMotion<double,3> >;

    GFE::CosseratRodEnergy<FEBasis,
                           GeodesicInterpolationRule,
                           double> localRodEnergy(gridView,
                                                  1,1,1,1e6,0.3);

    std::vector<RigidBodyMotion<double,3> > referenceConfiguration(gridView.size(1));

    for (const auto vertex : vertices(gridView))
    {
        auto idx = gridView.indexSet().index(vertex);

        referenceConfiguration[idx].r[0] = 0;
        referenceConfiguration[idx].r[1] = 0;
        referenceConfiguration[idx].r[2] = vertex.geometry().corner(0)[0];
        referenceConfiguration[idx].q = Rotation<double,3>::identity();
    }

    localRodEnergy.setReferenceConfiguration(referenceConfiguration);

    auto localView = feBasis.localView();
    localView.bind(*gridView.begin<0>());

    SolutionType localX = {x[0], x[1]};
    SolutionType localRotatedX = {rotatedX[0], rotatedX[1]};

    if (std::abs(localRodEnergy.energy(localView, localX) - localRodEnergy.energy(localView, localRotatedX)) > 1e-6)
        DUNE_THROW(Dune::Exception, "Rod energy not invariant under rigid body motions!");

 } catch (Exception& e) {

    std::cout << e.what() << std::endl;

 }
