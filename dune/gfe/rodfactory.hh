#ifndef MAKE_STRAIGHT_ROD_HH
#define MAKE_STRAIGHT_ROD_HH

#include <vector>
#include <dune/common/fvector.hh>
#include <dune/fufem/crossproduct.hh>

#include "rigidbodymotion.hh"
#include <dune/gfe/localgeodesicfefunction.hh>

/** \brief A factory class that implements various ways to create rod configurations
 */

template <class GridView>
class RodFactory
{
    dune_static_assert(GridView::dimensionworld==1, "RodFactory is only implemented for grids in a 1d world");
    
public:

    RodFactory(const GridView& gridView)
    : gridView_(gridView)
    {}
    
/** \brief Make a straight, unsheared rod from two given endpoints

\param[out] rod The new rod
\param[in] n The number of vertices
*/
template <int dim>
    static void makeStraightRod(std::vector<RigidBodyMotion<dim> >& rod, int n,
                     const Dune::FieldVector<double,3>& beginning, const Dune::FieldVector<double,3>& end)
{
    // Compute the correct orientation
    Rotation<3,double> orientation = Rotation<3,double>::identity();

    Dune::FieldVector<double,3> zAxis(0);
    zAxis[2] = 1;
    Dune::FieldVector<double,3> axis = crossProduct(Dune::FieldVector<double,3>(end-beginning), zAxis);
    if (axis.two_norm() != 0)
        axis /= -axis.two_norm();

    Dune::FieldVector<double,3> d3 = end-beginning;
    d3 /= d3.two_norm();

    double angle = std::acos(zAxis * d3);

    if (angle != 0)
        orientation = Rotation<3,double>(axis, angle);

    // Set the values
    rod.resize(n);
    for (int i=0; i<n; i++) {

        rod[i].r = beginning;
        rod[i].r.axpy(double(i) / (n-1), end-beginning);
        rod[i].q = orientation;

    }

}


/** \brief Make a rod by interpolating between two end configurations

\param[out] rod The new rod
*/
    template <int spaceDim>
    void create(std::vector<RigidBodyMotion<spaceDim> >& rod,
                     const RigidBodyMotion<3,double>& beginning,
                     const RigidBodyMotion<3,double>& end)
{
    
    static const int dim = GridView::dimension;  // de facto: 1
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    //  Get smallest and largest coordinate, in order to create an arc-length parametrization
    //////////////////////////////////////////////////////////////////////////////////////////////
    
    typename GridView::template Codim<dim>::Iterator vIt    = gridView_.template begin<dim>();
    typename GridView::template Codim<dim>::Iterator vEndIt = gridView_.template end<dim>();
    
    double min =  std::numeric_limits<double>::max();
    double max = -std::numeric_limits<double>::max();
    
    for (; vIt != vEndIt; ++vIt) {
        min = std::min(min, vIt->geometry().corner(0)[0]);
        max = std::max(max, vIt->geometry().corner(0)[0]);
    }
    
    ////////////////////////////////////////////////////////////////////////////////////
    //  Make a 1d geodesic finite element function, which will do the interpolation
    ////////////////////////////////////////////////////////////////////////////////////
    
    std::vector<RigidBodyMotion<3> > coefficients(2);
    coefficients[0] = beginning;
    coefficients[1] = end;

    LocalGeodesicFEFunction<1,double,RigidBodyMotion<3> > localGFEFunction(coefficients);
    
    ////////////////////////////////////////////////////////////////////////////////////
    //  Interpolate according to arc-length
    ////////////////////////////////////////////////////////////////////////////////////

    rod.resize(gridView_.size(dim));
    
    for (vIt = gridView_.template begin<dim>(); vIt != vEndIt; ++vIt) {
        int idx = gridView_.indexSet().index(*vIt);
        Dune::FieldVector<double,1> local = (vIt->geometry().corner(0)[0] - min) / (max - min);
        rod[idx] = localGFEFunction.evaluate(local);
    }
}

private:
    
    const GridView& gridView_;
};

#endif
