#ifndef DUNE_GFE_NONPLANARCOSSERATSHELLENERGY_HH
#define DUNE_GFE_NONPLANARCOSSERATSHELLENERGY_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/parametertree.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/matrix-vector/crossproduct.hh>

#include <dune/fufem/boundarypatch.hh>

#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>

#if HAVE_DUNE_GMSH4
#include <dune/gmsh4/gridcreators/lagrangegridcreator.hh>
#endif

#include <dune/gfe/assemblers/localenergy.hh>
#include <dune/gfe/densities/cosseratshelldensity.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/tensor3.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>

#if HAVE_DUNE_CURVEDGEOMETRY
#include <dune/curvedgeometry/curvedgeometry.hh>
#include <dune/localfunctions/lagrange/lfecache.hh>
#endif

namespace Dune::GFE
{
  namespace Impl
  {
    /** \brief Get LocalFiniteElements from a localView, for different tree depths of the local view
     *
     * We instantiate the CosseratEnergyLocalStiffness class with two different kinds of Basis:
     * A scalar one and a composite one that combines two scalar ones.  But code for accessing the
     * finite elements in the basis tree only work for one kind of basis, not for the other.
     * To allow both kinds of basis in a single class we need this trickery below.
     */
    template <class Basis, std::size_t i>
    class NonplanarCosseratShellLocalFiniteElementFactory
    {
    public:
      static auto get(const typename Basis::LocalView& localView,
                      std::integral_constant<std::size_t, i> iType)
      -> decltype(localView.tree().child(iType,0).finiteElement())
      {
        return localView.tree().child(iType,0).finiteElement();
      }
    };

    /** \brief Specialize for scalar bases, here we cannot call tree().child() */
    template <class GridView, int order, std::size_t i>
    class NonplanarCosseratShellLocalFiniteElementFactory<Dune::Functions::LagrangeBasis<GridView,order>,i>
    {
    public:
      static auto get(const typename Dune::Functions::LagrangeBasis<GridView,order>::LocalView& localView,
                      std::integral_constant<std::size_t, i> iType)
      -> decltype(localView.tree().finiteElement())
      {
        return localView.tree().finiteElement();
      }
    };
  }
}

/** \brief Assembles the cosserat energy for a single element.
 *
 * \tparam Basis                       Type of the Basis used for assembling
 * \tparam dim                         Dimension of the Targetspace, 3
 * \tparam field_type                  The coordinate type of the TargetSpace
 * \tparam StressFreeStateGridFunction Type of the GridFunction representing the Cosserat shell in a stress free state
 */
template<class Basis, int dim, class field_type, class StressFreeStateGridFunction>
class NonplanarCosseratShellEnergy
  : public Dune::GFE::LocalEnergy<Basis,Dune::GFE::ProductManifold<RealTuple<field_type,dim>,Rotation<field_type,dim> > >
{
  // grid types
  typedef typename Basis::GridView GridView;
  typedef typename GridView::ctype DT;
  typedef Dune::GFE::ProductManifold<RealTuple<field_type,dim>,Rotation<field_type,dim> > TargetSpace;
  typedef typename TargetSpace::ctype RT;
  typedef typename GridView::template Codim<0>::Entity Entity;

  // some other sizes
  constexpr static int gridDim = GridView::dimension;
  constexpr static int dimworld = GridView::dimensionworld;

  using Position = typename GridView::template Codim<0>::Entity::Geometry::GlobalCoordinate;

public:

  /** \brief Constructor with a set of material parameters
   * \param parameters                  The material parameters
   * \param stressFreeStateGridFunction Pointer to a parametrization representing the Cosserat shell in a stress-free state
   */
  NonplanarCosseratShellEnergy(const std::shared_ptr<Dune::GFE::CosseratShellDensity<Position,field_type> >& density,
                               const StressFreeStateGridFunction* stressFreeStateGridFunction)
    : stressFreeStateGridFunction_(stressFreeStateGridFunction),
    density_(density)
  {}

  /** \brief Assemble the energy for a single element */
  RT energy (const typename Basis::LocalView& localView,
             const std::vector<TargetSpace>& localSolution) const override;

  RT energy (const typename Basis::LocalView& localView,
             const typename Dune::GFE::Impl::LocalEnergyTypes<TargetSpace>::CompositeCoefficients& coefficients) const override;

#if HAVE_DUNE_GMSH4
  static int getOrder(const Dune::Gmsh4::LagrangeGridCreator<typename GridView::Grid>* lagrangeGridCreator)
  {
    return lagrangeGridCreator->order();
  }
#endif

  template<typename B, typename V, typename NTREM, typename R>
  static int getOrder(const Dune::Functions::DiscreteGlobalBasisFunction<B,V, NTREM, R>* gridFunction)
  {
    return gridFunction->basis().preBasis().subPreBasis().order();
  }

  /** \brief The geometry of the reference deformation used for assembling */
  const StressFreeStateGridFunction* stressFreeStateGridFunction_;

  /** \brief The energy density of a Cosserat shell with nonplanar reference configuration */
  const std::shared_ptr<Dune::GFE::CosseratShellDensity<Position,field_type> > density_;
};

template <class Basis, int dim, class field_type, class StressFreeStateGridFunction>
typename NonplanarCosseratShellEnergy<Basis, dim, field_type, StressFreeStateGridFunction>::RT
NonplanarCosseratShellEnergy<Basis,dim,field_type, StressFreeStateGridFunction>::
energy(const typename Basis::LocalView& localView,
       const std::vector<Dune::GFE::ProductManifold<RealTuple<field_type,dim>,Rotation<field_type,dim> > >& localSolution) const
{
  using namespace Dune::Indices;

  // The element geometry
  auto element = localView.element();

  // The set of shape functions on this element
  using namespace Dune::Indices;
  const auto& localFiniteElement = Dune::GFE::Impl::NonplanarCosseratShellLocalFiniteElementFactory<Basis,0>::get(localView,_0);

#if HAVE_DUNE_CURVEDGEOMETRY
  // Construct a curved geometry of this element of the Cosserat shell in its stress-free state
  // The variable local holds the local coordinates in the reference element
  // and localGeometry.global maps them to the world coordinates
  Dune::CurvedGeometry<DT, gridDim, dimworld, Dune::CurvedGeometryTraits<DT, Dune::LagrangeLFECache<DT,DT,gridDim> > > geometry(referenceElement(element),
                                                                                                                                [this,element](const auto& local) {
                                                                                                                                auto localGridFunction = localFunction(*stressFreeStateGridFunction_);
                                                                                                                                localGridFunction.bind(element);
                                                                                                                                return localGridFunction(local);
    }, getOrder(stressFreeStateGridFunction_));
#else
  // When using element.geometry(), the geometry of the element is flat
  auto geometry = element.geometry();
#endif

  ////////////////////////////////////////////////////////////////////////////////////
  //  Set up the local nonlinear finite element function
  ////////////////////////////////////////////////////////////////////////////////////
  typedef LocalGeodesicFEFunction<gridDim, DT, decltype(localFiniteElement), TargetSpace> LocalGFEFunctionType;
  LocalGFEFunctionType localGeodesicFEFunction(localFiniteElement,localSolution);

  RT energy = 0;

  auto quadOrder = (element.type().isSimplex()) ? localFiniteElement.localBasis().order()
                                                : localFiniteElement.localBasis().order() * gridDim;

  const auto& quad = Dune::QuadratureRules<DT, gridDim>::rule(element.type(), quadOrder);

  for (size_t pt=0; pt<quad.size(); pt++)
  {
    // Local position of the quadrature point
    const Dune::FieldVector<DT,gridDim>& quadPos = quad[pt].position();

    // Global position of the quadrature point
    auto quadPosGlobal = element.geometry().global(quadPos);

    const DT integrationElement = geometry.integrationElement(quadPos);

    // The value of the local function
    Dune::GFE::ProductManifold<RealTuple<field_type,dim>,Rotation<field_type,dim> > value = localGeodesicFEFunction.evaluate(quadPos);

    // The derivative of the local function w.r.t. the coordinate system of the tangent space
    auto derivative = localGeodesicFEFunction.evaluateDerivative(quadPos,value);

    ////////////////////////////////
    //  First fundamental form
    ////////////////////////////////

    Dune::FieldMatrix<double,3,3> aCovariant;

    // If dimworld==3, then the first two lines of aCovariant are simply the jacobianTransposed
    // of the element.  If dimworld<3 (i.e., ==2), we have to explicitly enters 0.0 in the last column.
    auto jacobianTransposed = geometry.jacobianTransposed(quadPos);

    for (int i=0; i<2; i++)
    {
      for (int j=0; j<dimworld; j++)
        aCovariant[i][j] = jacobianTransposed[i][j];
      for (int j=dimworld; j<3; j++)
        aCovariant[i][j] = 0.0;
    }

    aCovariant[2] = Dune::MatrixVector::crossProduct(aCovariant[0], aCovariant[1]);
    aCovariant[2] /= aCovariant[2].two_norm();

#if HAVE_DUNE_CURVEDGEOMETRY
    const auto normalGradient = geometry.normalGradient(quad[pt].position());
#else
    // Assume that the geometry is flat if DUNE_CURVEDGEOMETRY is not present.
    // TODO: This is not always true!
    Dune::FieldMatrix<double,3,3> normalGradient(0);
#endif

    //////////////////////////////////////////////////////////
    // Add the local energy density
    //////////////////////////////////////////////////////////

    const auto energyDensity = (*density_)(quadPosGlobal,
                                           aCovariant,
                                           normalGradient,
                                           value,
                                           derivative);

    // Add energy density
    energy += quad[pt].weight() * integrationElement * energyDensity;
  }

  return energy;
}

template <class Basis, int dim, class field_type, class StressFreeStateGridFunction>
typename NonplanarCosseratShellEnergy<Basis, dim, field_type, StressFreeStateGridFunction>::RT
NonplanarCosseratShellEnergy<Basis,dim,field_type, StressFreeStateGridFunction>::
energy(const typename Basis::LocalView& localView,
       const typename Dune::GFE::Impl::LocalEnergyTypes<TargetSpace>::CompositeCoefficients& localConfiguration) const
{
  // The element geometry
  auto element = localView.element();

  // The set of shape functions on this element

  using namespace Dune::Indices;
  const auto& deformationLocalFiniteElement = Dune::GFE::Impl::NonplanarCosseratShellLocalFiniteElementFactory<Basis,0>::get(localView,_0);
  const auto& orientationLocalFiniteElement = Dune::GFE::Impl::NonplanarCosseratShellLocalFiniteElementFactory<Basis,1>::get(localView,_1);

#if HAVE_DUNE_CURVEDGEOMETRY
  // Construct a curved geometry of this element of the Cosserat shell in its stress-free state
  // The variable local holds the local coordinates in the reference element
  // and localGeometry.global maps them to the world coordinates
  Dune::CurvedGeometry<DT, gridDim, dimworld, Dune::CurvedGeometryTraits<DT, Dune::LagrangeLFECache<DT,DT,gridDim> > > geometry(referenceElement(element),
                                                                                                                                [this,element](const auto& local) {
                                                                                                                                auto localGridFunction = localFunction(*stressFreeStateGridFunction_);
                                                                                                                                localGridFunction.bind(element);
                                                                                                                                return localGridFunction(local);
    }, getOrder(stressFreeStateGridFunction_));
#else
  // When using element.geometry(), the geometry of the element is flat
  auto geometry = element.geometry();
#endif

  ////////////////////////////////////////////////////////////////////////////////////
  //  Set up the local nonlinear finite element function
  ////////////////////////////////////////////////////////////////////////////////////
  typedef LocalGeodesicFEFunction<gridDim, DT, decltype(deformationLocalFiniteElement), RealTuple<field_type,dim> > LocalDeformationGFEFunctionType;
  typedef LocalGeodesicFEFunction<gridDim, DT, decltype(orientationLocalFiniteElement), Rotation<field_type,dim> > LocalOrientationGFEFunctionType;
  LocalDeformationGFEFunctionType localDeformationGFEFunction(deformationLocalFiniteElement,localConfiguration[_0]);
  LocalOrientationGFEFunctionType localOrientationGFEFunction(orientationLocalFiniteElement,localConfiguration[_1]);

  RT energy = 0;

  auto quadOrder = (deformationLocalFiniteElement.type().isSimplex()) ? deformationLocalFiniteElement.localBasis().order()
                                                : deformationLocalFiniteElement.localBasis().order() * gridDim;

  const auto& quad = Dune::QuadratureRules<DT, gridDim>::rule(element.type(), quadOrder);

  for (size_t pt=0; pt<quad.size(); pt++)
  {
    // Local position of the quadrature point
    const Dune::FieldVector<DT,gridDim>& quadPos = quad[pt].position();

    // Global position of the quadrature point
    auto quadPosGlobal = element.geometry().global(quadPos);

    const DT integrationElement = geometry.integrationElement(quadPos);

    // The value of the local function
    TargetSpace value;
    value[_0] = localDeformationGFEFunction.evaluate(quadPos);
    value[_1] = localOrientationGFEFunction.evaluate(quadPos);

    // The derivative of the local function w.r.t. the coordinate system of the tangent space
    auto deformationDerivative = localDeformationGFEFunction.evaluateDerivative(quadPos,value[_0]);
    auto orientationDerivative = localOrientationGFEFunction.evaluateDerivative(quadPos,value[_1]);

    // Concatenate the two derivative matrices
    Dune::FieldMatrix<RT, TargetSpace::embeddedDim, gridDim> derivative;

    for (int i=0; i<deformationDerivative.rows; ++i)
      derivative[i] = deformationDerivative[i];

    for (int i=0; i<orientationDerivative.rows; ++i)
      derivative[i+deformationDerivative.rows] = orientationDerivative[i];

    //////////////////////////////////////////////////////////
    //  Fundamental forms and curvature
    //////////////////////////////////////////////////////////

    // First fundamental form
    Dune::FieldMatrix<double,3,3> aCovariant;

    // If dimworld==3, then the first two lines of aCovariant are simply the jacobianTransposed
    // of the element.  If dimworld<3 (i.e., ==2), we have to explicitly enters 0.0 in the last column.
    auto jacobianTransposed = geometry.jacobianTransposed(quadPos);
    for (int i=0; i<2; i++)
    {
      for (int j=0; j<dimworld; j++)
        aCovariant[i][j] = jacobianTransposed[i][j];
      for (int j=dimworld; j<3; j++)
        aCovariant[i][j] = 0.0;
    }

    aCovariant[2] = Dune::MatrixVector::crossProduct(aCovariant[0], aCovariant[1]);
    aCovariant[2] /= aCovariant[2].two_norm();

#if HAVE_DUNE_CURVEDGEOMETRY
    const auto normalGradient = geometry.normalGradient(quad[pt].position());
#else
    // Assume that the geometry is flat if DUNE_CURVEDGEOMETRY is not present.
    // TODO: This is not always true!
    Dune::FieldMatrix<double,3,3> normalGradient(0);
#endif

    //////////////////////////////////////////////////////////
    // Add the local energy density
    //////////////////////////////////////////////////////////

    const auto energyDensity = (*density_)(quadPosGlobal,
                                           aCovariant,
                                           normalGradient,
                                           value,
                                           derivative);

    // Add energy density
    energy += quad[pt].weight() * integrationElement * energyDensity;
  }

  return energy;
}

#endif   //#ifndef DUNE_GFE_NONPLANARCOSSERATSHELLENERGY_HH
