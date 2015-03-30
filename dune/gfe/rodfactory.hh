#ifndef MAKE_STRAIGHT_ROD_HH
#define MAKE_STRAIGHT_ROD_HH

#include <vector>
#include <dune/common/fvector.hh>
#include <dune/fufem/arithmetic.hh>

#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/localgeodesicfefunction.hh>

#include <dune/gfe/rodassembler.hh>
#include <dune/gfe/riemanniantrsolver.hh>

/** \brief A factory class that implements various ways to create rod configurations
 */

template <class GridView>
class RodFactory
{
    static_assert(GridView::dimensionworld==1, "RodFactory is only implemented for grids in a 1d world");

public:

    RodFactory(const GridView& gridView)
    : gridView_(gridView)
    {}

/** \brief Make a straight, unsheared rod from two given endpoints

\param[out] rod The new rod
\param[in] n The number of vertices
*/
template <int dim>
    void create(std::vector<RigidBodyMotion<double,dim> >& rod,
                     const Dune::FieldVector<double,3>& beginning, const Dune::FieldVector<double,3>& end)
{
    // Compute the correct orientation
    Rotation<double,dim> orientation = Rotation<double,dim>::identity();

    Dune::FieldVector<double,3> zAxis(0);
    zAxis[2] = 1;
    Dune::FieldVector<double,3> axis = Arithmetic::crossProduct(Dune::FieldVector<double,3>(end-beginning), zAxis);
    if (axis.two_norm() != 0)
        axis /= -axis.two_norm();

    Dune::FieldVector<double,3> d3 = end-beginning;
    d3 /= d3.two_norm();

    double angle = std::acos(zAxis * d3);

    if (angle != 0)
        orientation = Rotation<double,3>(axis, angle);

        // Set the values
        create(rod, RigidBodyMotion<double,dim>(beginning,orientation), RigidBodyMotion<double,dim>(end,orientation));
}


/** \brief Make a rod by interpolating between two end configurations

\param[out] rod The new rod
*/
    template <int spaceDim>
    void create(std::vector<RigidBodyMotion<double,spaceDim> >& rod,
                     const RigidBodyMotion<double,spaceDim>& beginning,
                     const RigidBodyMotion<double,spaceDim>& end)
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
    //  Interpolate according to arc-length
    ////////////////////////////////////////////////////////////////////////////////////

    rod.resize(gridView_.size(dim));

    for (vIt = gridView_.template begin<dim>(); vIt != vEndIt; ++vIt) {
        int idx = gridView_.indexSet().index(*vIt);
        Dune::FieldVector<double,1> local = (vIt->geometry().corner(0)[0] - min) / (max - min);

        for (int i=0; i<3; i++)
            rod[idx].r[i] = (1-local)*beginning.r[i] + local*end.r[i];
        rod[idx].q = Rotation<double,3>::interpolate(beginning.q, end.q, local);
    }
}

    /** \brief Make a rod by setting each entry to the same value

    \param[out] rod The new rod
    */
    template <int spaceDim>
    void create(std::vector<RigidBodyMotion<double,spaceDim> >& rod,
                const RigidBodyMotion<double,spaceDim>& value)
    {
        rod.resize(gridView_.size(1));
        std::fill(rod.begin(), rod.end(), value);
    }

    /** \brief Make a rod by linearly interpolating between the end values

        \note The end values are expected to be in the input container!
        \param[in,out] rod The new rod
    */
    template <int spaceDim>
    void create(std::vector<RigidBodyMotion<double,spaceDim> >& rod)
    {
        static const int dim = GridView::dimension;  // de facto: 1
        assert(gridView_.size(dim)==rod.size());

        //////////////////////////////////////////////////////////////////////////////////////////////
        //  Get smallest and largest coordinate, in order to create an arc-length parametrization
        //////////////////////////////////////////////////////////////////////////////////////////////

        typename GridView::template Codim<dim>::Iterator vIt    = gridView_.template begin<dim>();
        typename GridView::template Codim<dim>::Iterator vEndIt = gridView_.template end<dim>();

        double min =  std::numeric_limits<double>::max();
        double max = -std::numeric_limits<double>::max();
        RigidBodyMotion<double,spaceDim> beginning, end;

        for (; vIt != vEndIt; ++vIt) {
            if (vIt->geometry().corner(0)[0] < min) {
                min = vIt->geometry().corner(0)[0];
                beginning = rod[gridView_.indexSet().index(*vIt)];
            }
            if (vIt->geometry().corner(0)[0] > max) {
                max = vIt->geometry().corner(0)[0];
                end = rod[gridView_.indexSet().index(*vIt)];
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////
        //  Interpolate according to arc-length
        ////////////////////////////////////////////////////////////////////////////////////

        rod.resize(gridView_.size(dim));

        for (vIt = gridView_.template begin<dim>(); vIt != vEndIt; ++vIt) {
            int idx = gridView_.indexSet().index(*vIt);
            Dune::FieldVector<double,1> local = (vIt->geometry().corner(0)[0] - min) / (max - min);

            for (int i=0; i<3; i++)
                rod[idx].r[i] = (1-local)*beginning.r[i] + local*end.r[i];
            rod[idx].q = Rotation<double,3>::interpolate(beginning.q, end.q, local);
        }
    }


/** \brief Make a rod solving a static Dirichlet problem

  \param rod The configuration to be computed
  \param radius The rod's radius
  \param E The rod's elastic modulus
  \param nu The rod's Poission modulus
  \param beginning The prescribed Dirichlet values
  \param end The prescribed Dirichlet values
  \param[out] rod The new rod
 */
template <int spaceDim>
void create(std::vector<RigidBodyMotion<double,spaceDim> >& rod,
        double radius, double E, double nu,
        const RigidBodyMotion<double,spaceDim>& beginning,
        const RigidBodyMotion<double,spaceDim>& end)
{

    // Make Dirichlet bitfields for the rods as well
    Dune::BitSetVector<6> rodDirichletNodes(gridView_.size(GridView::dimension),false);

    for (int j=0; j<6; j++) {
        rodDirichletNodes[0][j] = true;
        rodDirichletNodes.back()[j] = true;
    }

    // Create local assembler for the static elastic problem
    RodLocalStiffness<GridView,double> rodLocalStiffness(gridView_, radius*radius*M_PI,
            std::pow(radius,4) * 0.25* M_PI, std::pow(radius,4) * 0.25* M_PI, E, nu);

    typedef Dune::Functions::PQKNodalBasis<GridView,1> RodP1Basis;
    RodP1Basis p1Basis(gridView_);
    RodAssembler<RodP1Basis,spaceDim> assembler(p1Basis, &rodLocalStiffness);

    // Create initial iterate using the straight rod interpolation method
    create(rod, beginning.r, end.r);

    // Set reference configuration
    rodLocalStiffness.setReferenceConfiguration(rod);

    // Set Dirichlet values
    rod[0] = beginning;
    rod.back() = end;

    // Trust--Region solver
    RiemannianTrustRegionSolver<RodP1Basis, RigidBodyMotion<double,spaceDim> > rodSolver;
    rodSolver.setup(gridView_.grid(), &assembler, rod,
            rodDirichletNodes,
            1e-10, 100, // TR tolerance and iterations
            20, // init TR radius
            200, 1e-00, 1, 3, 3, // Multigrid parameters
            100, 1e-8 , false); // base solver parameters

    rodSolver.verbosity_ = NumProc::QUIET;

    rodSolver.solve();

    rod = rodSolver.getSol();


}

private:

    const GridView gridView_;
};

#endif
