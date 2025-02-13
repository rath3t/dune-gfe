#ifndef DUNE_GFE_SURFACECOSSERATENERGY_HH
#define DUNE_GFE_SURFACECOSSERATENERGY_HH

#include <dune/common/indices.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/fufem/boundarypatch.hh>

#include <dune/functions/functionspacebases/subspacebasis.hh>

#include <dune/gfe/assemblers/localenergy.hh>
#include <dune/gfe/densities/cosseratshelldensity.hh>
#include <dune/gfe/functions/localgeodesicfefunction.hh>
#include <dune/gfe/functions/localprojectedfefunction.hh>
#include <dune/gfe/spaces/productmanifold.hh>

#if HAVE_DUNE_CURVEDGEOMETRY
#include <dune/curvedgeometry/curvedgeometry.hh>
#include <dune/localfunctions/lagrange/lfecache.hh>
#endif

namespace Dune::GFE
{
  /** \brief Assembles the cosserat energy on the given boundary for a single element.
   *
   * \tparam CurvedGeometryGridFunction Type of the grid function that gives the geometry of the deformed surface
   * \tparam Basis Type of the Basis used for assembling
   * \tparam TargetSpace Currently this is required to be a ProductManifold with two factors
   */
  template<class CurvedGeometryGridFunction, class Basis, class TargetSpace>
  class SurfaceCosseratEnergy
    : public Dune::GFE::LocalEnergy<Basis, TargetSpace>
  {
    using GridView = typename Basis::GridView;
    using DT = typename GridView::ctype ;
    using RT = typename Dune::GFE::LocalEnergy<Basis, TargetSpace>::RT ;

    constexpr static int dimWorld = GridView::dimensionworld;
    constexpr static int gridDim = GridView::dimension;
    static constexpr int boundaryDim = gridDim - 1;

    using Intersection = typename GridView::Intersection;

  public:

    /** \brief Constructor with a set of material parameters
     * \param density The density that is being integrated over
     * \param shellBoundary The shellBoundary contains the faces where the cosserat energy is assembled
     * \param curvedGeometryGridFunction The curvedGeometryGridFunction gives the geometry of the shell in stress-free state.
              When assembling, we deform the intersections using the curvedGeometryGridFunction and then use the deformed geometries.
     */
    SurfaceCosseratEnergy(const std::shared_ptr<CosseratShellDensity<Intersection,RT> >& density,
                          const BoundaryPatch<GridView>* shellBoundary,
                          const CurvedGeometryGridFunction& curvedGeometryGridFunction)
      : shellBoundary_(shellBoundary),
      curvedGeometryGridFunction_(curvedGeometryGridFunction),
      density_(density)
    {}

    /** \brief Assemble the energy for a single element */
    RT energy(const typename Basis::LocalView& localView,
              const std::vector<TargetSpace>& localSolutions) const override
    {
      DUNE_THROW(NotImplemented, "!");
    }

    RT energy (const typename Basis::LocalView& localView,
               const typename Impl::LocalEnergyTypes<TargetSpace>::CompositeCoefficients& localCoefficients) const override
    {
      static_assert(TargetSpace::size() == 2, "SurfaceCosseratEnergy needs exactly two TargetSpaces!");

      using namespace Dune::Indices;

      const auto element = localView.element();

      ////////////////////////////////////////////////////////////////////////////////////
      //  Set up the local nonlinear finite element function
      ////////////////////////////////////////////////////////////////////////////////////

      using RBM0 = typename std::tuple_element<0,TargetSpace>::type;
      using RBM1 = typename std::tuple_element<1,TargetSpace>::type;

      auto deformationScalarBasis = Functions::subspaceBasis(localView.globalBasis(), _0, 0);
      auto rotationScalarBasis = Functions::subspaceBasis(localView.globalBasis(), _1, 0);

      typedef LocalGeodesicFEFunction<decltype(deformationScalarBasis), RBM0> LocalGFEFunctionType0;
      typedef LocalGeodesicFEFunction<decltype(rotationScalarBasis), RBM1> LocalGFEFunctionType1;
      LocalGFEFunctionType0 localGeodesicFEFunction0(deformationScalarBasis);
      LocalGFEFunctionType1 localGeodesicFEFunction1(rotationScalarBasis);
      localGeodesicFEFunction0.bind(element,localCoefficients[_0]);
      localGeodesicFEFunction1.bind(element,localCoefficients[_1]);

      RT energy = 0;

      for (auto&& it : intersections(shellBoundary_->gridView(), element))
      {
        if (not shellBoundary_->contains(it))
          continue;

        // Bind the density to the current intersection
        density_->bind(it);

        const auto deformationOrder = localView.tree().child(_0,0).finiteElement().localBasis().order();
#if HAVE_DUNE_CURVEDGEOMETRY
        auto localGridFunction = localFunction(curvedGeometryGridFunction_);
        localGridFunction.bind(element);
        auto referenceElement = Dune::referenceElement<DT,boundaryDim>(it.type());

        // Construct the geometry on the boundary using the map lGF(localGeometry.global(local)):
        // The variable local holds the local coordinates in the 2D reference element, localGeometry.global maps them to the 3D reference element.
        // The function lGF is the gridfunction bound to the current element, so lGF(localGeometry.global(local)) is the value of curvedGeometryGridFunction_ at
        // the point on the intersection face.
        using BoundaryGeometry = Dune::CurvedGeometry<DT, boundaryDim, dimWorld, Dune::CurvedGeometryTraits<DT, Dune::LagrangeLFECache<DT,DT,boundaryDim> > >;
        BoundaryGeometry boundaryGeometry(referenceElement,
                                          [localGridFunction, localGeometry=it.geometryInInside()](const auto& local) {
                                          return localGridFunction(localGeometry.global(local));
          }, deformationOrder);
#else
        const auto& boundaryGeometry = it.geometry();
#endif

        auto quadOrder = (it.type().isSimplex()) ? deformationOrder : deformationOrder * boundaryDim;

        const auto& quad = Dune::QuadratureRules<DT, boundaryDim>::rule(it.type(), quadOrder);

        for (auto&& qp : quad)
        {
          // Local position of the quadrature point
          const auto& quadPos = it.geometryInInside().global(qp.position());

          const DT integrationElement = boundaryGeometry.integrationElement(qp.position());

          FieldMatrix<RT, TargetSpace::embeddedDim, boundaryDim> derivative2D;

          // The value of the local functions
          TargetSpace value;
          value[_0] = localGeodesicFEFunction0.evaluate(quadPos);
          value[_1] = localGeodesicFEFunction1.evaluate(quadPos);

          // The derivatives of the local functions
          auto derivative0 = localGeodesicFEFunction0.evaluateDerivative(quadPos,value[_0]);
          auto derivative1 = localGeodesicFEFunction1.evaluateDerivative(quadPos,value[_1]);

          // Put the value and the derivatives together from the separated values
          for (int j = 0; j < dimWorld; j++) {
            for (int i = 0; i < RBM0::embeddedDim; i++) {
              if (j < boundaryDim)
                derivative2D[i][j] = derivative0[i][j];
            }
            for (int i = 0; i < RBM1::embeddedDim; i++) {
              if (j < boundaryDim)
                derivative2D[RBM0::embeddedDim + i][j] = derivative1[i][j];
            }
          }

          //////////////////////////////////////////////////////////
          //  Fundamental forms and curvature
          //////////////////////////////////////////////////////////

          // First fundamental form
          FieldMatrix<double,dimWorld,dimWorld> aCovariant;

          const auto jacobianTransposed = boundaryGeometry.jacobianTransposed(qp.position());

          for (int i=0; i<2; i++)
            aCovariant[i] = jacobianTransposed[i];

          aCovariant[2] = Dune::FMatrixHelp::Impl::crossProduct(aCovariant[0], aCovariant[1]);
          aCovariant[2] /= aCovariant[2].two_norm();

          // Second fundamental form: The derivative of the normal field
#if HAVE_DUNE_CURVEDGEOMETRY
          auto normalGradient = boundaryGeometry.normalGradient(qp.position());
#else
          // If we do not have the dune-curvedgeometry module then we have to assume
          // that the boundary is flat.  This may not be true, but there is currently
          // no way to get this information from the dune-grid grid interface.
          // TODO: Fix this once the grid interface is extended.
          FieldMatrix<DT,dimWorld,dimWorld> normalGradient(0);
#endif

          //////////////////////////////////////////////////////////
          // Add the local energy density
          //////////////////////////////////////////////////////////

          const auto energyDensity = (*density_)(qp.position(),
                                                 aCovariant,
                                                 normalGradient,
                                                 value,
                                                 derivative2D);
          energy += qp.weight() * integrationElement * energyDensity;
        }
      }

      return energy;
    }

  private:
    /** \brief The shell boundary */
    const BoundaryPatch<GridView>* shellBoundary_;

    /** \brief The function used to create the Geometries used for assembling */
    const CurvedGeometryGridFunction curvedGeometryGridFunction_;

    /** \brief The density that is being integrated */
    const std::shared_ptr<CosseratShellDensity<Intersection,RT> > density_;

  };
}  // namespace Dune::GFE

#endif   //#ifndef DUNE_GFE_SURFACECOSSERATENERGY_HH
