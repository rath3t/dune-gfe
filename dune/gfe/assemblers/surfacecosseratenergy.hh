#ifndef DUNE_GFE_SURFACECOSSERATENERGY_HH
#define DUNE_GFE_SURFACECOSSERATENERGY_HH

#include <dune/common/indices.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/fufem/boundarypatch.hh>

#include <dune/gfe/cosseratstrain.hh>
#include <dune/gfe/assemblers/localenergy.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/localprojectedfefunction.hh>
#include <dune/gfe/assemblers/localenergy.hh>
#include <dune/gfe/densities/cosseratshelldensity.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>

#if HAVE_DUNE_CURVEDGEOMETRY
#include <dune/curvedgeometry/curvedgeometry.hh>
#include <dune/localfunctions/lagrange/lfecache.hh>
#endif

namespace Dune::GFE {
  /** \brief Assembles the cosserat energy on the given boundary for a single element.
   *
   * \tparam CurvedGeometryGridFunction Type of the grid function that gives the geometry of the deformed surface
   * \tparam Basis Type of the Basis used for assembling
   * \tparam TargetSpaces The List of TargetSpaces - SurfaceCosseratEnergy needs exactly two TargetSpaces!
   */
  template<class CurvedGeometryGridFunction, class Basis, class ... TargetSpaces>
  class SurfaceCosseratEnergy
    : public Dune::GFE::LocalEnergy<Basis, ProductManifold<TargetSpaces...> >
  {
    using TargetSpace = ProductManifold<TargetSpaces...>;
    using GridView = typename Basis::GridView;
    using DT = typename GridView::ctype ;
    using RT = typename Dune::GFE::LocalEnergy<Basis, TargetSpace>::RT ;
    using RBM0 = RealTuple<RT,GridView::dimensionworld> ;
    using RBM1 = Rotation<RT,GridView::dimensionworld> ;
    using RBM = ProductManifold<RBM0,RBM1> ;

    constexpr static int dimWorld = GridView::dimensionworld;
    constexpr static int gridDim = GridView::dimension;
    static constexpr int boundaryDim = gridDim - 1;

  public:

    /** \brief Constructor with a set of material parameters
     * \param parameters The material parameters
     * \param shellBoundary The shellBoundary contains the faces where the cosserat energy is assembled
     * \param curvedGeometryGridFunction The curvedGeometryGridFunction gives the geometry of the shell in stress-free state.
              When assembling, we deform the intersections using the curvedGeometryGridFunction and then use the deformed geometries.
     * \param thicknessF The shell thickness parameter, given as a function and evaluated at each quadrature point
     * \param lameF The Lame parameters, given as a function and evaluated at each quadrature point
     */
    SurfaceCosseratEnergy(const Dune::ParameterTree& parameters,
                          const BoundaryPatch<GridView>* shellBoundary,
                          const CurvedGeometryGridFunction& curvedGeometryGridFunction,
                          const std::function<double(Dune::FieldVector<double,dimWorld>)> thicknessF,
                          const std::function<Dune::FieldVector<double,2>(Dune::FieldVector<double,dimWorld>)> lameF)
      : shellBoundary_(shellBoundary),
      curvedGeometryGridFunction_(curvedGeometryGridFunction),
      density_(parameters,thicknessF,lameF)
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
      static_assert(sizeof...(TargetSpaces) == 2, "SurfaceCosseratEnergy needs exactly two TargetSpaces!");

      using namespace Dune::Indices;

      ////////////////////////////////////////////////////////////////////////////////////
      //  Set up the local nonlinear finite element function
      ////////////////////////////////////////////////////////////////////////////////////

      // The set of shape functions on this element
      const auto& deformationLocalFiniteElement = localView.tree().child(_0,0).finiteElement();
      const auto& orientationLocalFiniteElement = localView.tree().child(_1,0).finiteElement();

      typedef LocalGeodesicFEFunction<gridDim, DT, decltype(deformationLocalFiniteElement), RBM0> LocalGFEFunctionType0;
      typedef LocalGeodesicFEFunction<gridDim, DT, decltype(orientationLocalFiniteElement), RBM1> LocalGFEFunctionType1;
      LocalGFEFunctionType0 localGeodesicFEFunction0(deformationLocalFiniteElement,localCoefficients[_0]);
      LocalGFEFunctionType1 localGeodesicFEFunction1(orientationLocalFiniteElement,localCoefficients[_1]);

      RT energy = 0;

      const auto element = localView.element();

      for (auto&& it : intersections(shellBoundary_->gridView(), element)) {
        if (not shellBoundary_->contains(it))
          continue;

#if HAVE_DUNE_CURVEDGEOMETRY
        auto localGridFunction = localFunction(curvedGeometryGridFunction_);
        auto curvedGeometryGridFunctionOrder = deformationLocalFiniteElement.localBasis().order();//curvedGeometryGridFunction_.basis().localView().tree().child(0).finiteElement().localBasis().order();
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
          }, curvedGeometryGridFunctionOrder);
#else
        const auto& boundaryGeometry = it.geometry();
#endif

        auto quadOrder = (it.type().isSimplex()) ? deformationLocalFiniteElement.localBasis().order()
                                                  : deformationLocalFiniteElement.localBasis().order() * boundaryDim;

        const auto& quad = Dune::QuadratureRules<DT, boundaryDim>::rule(it.type(), quadOrder);
        for (size_t pt=0; pt<quad.size(); pt++) {

          // Local position of the quadrature point
          const Dune::FieldVector<DT,gridDim>& quadPos = it.geometryInInside().global(quad[pt].position());;

          // Global position of the quadrature point
          auto quadPosGlobal = it.geometry().global(quad[pt].position());

          const DT integrationElement = boundaryGeometry.integrationElement(quad[pt].position());

          RBM value;
          Dune::FieldMatrix<RT, RBM::embeddedDim, dimWorld> derivative;
          Dune::FieldMatrix<RT, RBM::embeddedDim, boundaryDim> derivative2D;

          // The value of the local functions
          value[_0] = localGeodesicFEFunction0.evaluate(quadPos);
          value[_1] = localGeodesicFEFunction1.evaluate(quadPos);

          // The derivatives of the local functions
          auto derivative0 = localGeodesicFEFunction0.evaluateDerivative(quadPos,value[_0]);
          auto derivative1 = localGeodesicFEFunction1.evaluateDerivative(quadPos,value[_1]);

          // Put the value and the derivatives together from the separated values
          for (int j = 0; j < dimWorld; j++) {
            for (int i = 0; i < RBM0::embeddedDim; i++) {
              derivative[i][j] = derivative0[i][j];
              if (j < boundaryDim)
                derivative2D[i][j] = derivative0[i][j];
            }
            for (int i = 0; i < RBM1::embeddedDim; i++) {
              derivative[RBM0::embeddedDim + i][j] = derivative1[i][j];
              if (j < boundaryDim)
                derivative2D[RBM0::embeddedDim + i][j] = derivative1[i][j];
            }
          }

          //////////////////////////////////////////////////////////
          //  Fundamental forms and curvature
          //////////////////////////////////////////////////////////

          // First fundamental form
          FieldMatrix<double,dimWorld,dimWorld> aCovariant;

          const auto jacobianTransposed = boundaryGeometry.jacobianTransposed(quad[pt].position());

          for (int i=0; i<2; i++)
            aCovariant[i] = jacobianTransposed[i];

          aCovariant[2] = Dune::FMatrixHelp::Impl::crossProduct(aCovariant[0], aCovariant[1]);
          aCovariant[2] /= aCovariant[2].two_norm();

          // Second fundamental form: The derivative of the normal field
#if HAVE_DUNE_CURVEDGEOMETRY
          auto normalGradient = boundaryGeometry.normalGradient(quad[pt].position());
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

          const auto energyDensity = density_(quadPosGlobal,
                                              aCovariant,
                                              normalGradient,
                                              value,
                                              derivative2D);
          energy += quad[pt].weight() * integrationElement * energyDensity;
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
    const CosseratShellDensity<FieldVector<DT,3>,RT> density_;

  };
}  // namespace Dune::GFE

#endif   //#ifndef DUNE_GFE_SURFACECOSSERATENERGY_HH
