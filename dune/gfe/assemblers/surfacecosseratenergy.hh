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
#include <dune/gfe/tensor3.hh>
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
    using Entity = typename GridView::template Codim<0>::Entity ;
    using RBM0 = RealTuple<RT,GridView::dimensionworld> ;
    using RBM1 = Rotation<RT,GridView::dimensionworld> ;
    using RBM = ProductManifold<RBM0,RBM1> ;

    constexpr static int dimWorld = GridView::dimensionworld;
    constexpr static int gridDim = GridView::dimension;
    static constexpr int boundaryDim = gridDim - 1;

    /** \brief Compute the derivative of the rotation (with respect to x), but wrt matrix coordinates
        \param value Value of the gfe function at a certain point
        \param derivative First derivative of the gfe function wrt x at that point, in quaternion coordinates
        \param DR First derivative of the gfe function wrt x at that point, in matrix coordinates
     */
    static void computeDR(const RBM& value,
                          const Dune::FieldMatrix<RT,RBM::embeddedDim,boundaryDim>& derivative,
                          Tensor3<RT,dimWorld,dimWorld,boundaryDim>& derivative_rotation)
    {
      // The LocalGFEFunction class gives us the derivatives of the orientation variable,
      // but as a map into quaternion space.  To obtain matrix coordinates we use the
      // chain rule, which means that we have to multiply the given derivative with
      // the derivative of the embedding of the unit quaternion into the space of 3x3 matrices.
      // This second derivative is almost given by the method getFirstDerivativesOfDirectors.
      // However, since the directors of a given unit quaternion are the _columns_ of the
      // corresponding orthogonal matrix, we need to invert the i and j indices
      //
      // So, DR[i][j][k] contains \partial R_ij / \partial k
      Tensor3<RT, dimWorld, dimWorld, 4> dd_dq;
      value[Indices::_1].getFirstDerivativesOfDirectors(dd_dq);

      derivative_rotation = RT(0);
      for (int i=0; i<dimWorld; i++)
        for (int j=0; j<dimWorld; j++)
          for (int k=0; k<boundaryDim; k++)
            for (int l=0; l<4; l++)
              derivative_rotation[i][j][k] += dd_dq[j][i][l] * derivative[l+3][k];
    }



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
      thicknessF_(thicknessF),
      lameF_(lameF)
    {
      // Cosserat couple modulus
      mu_c_ = parameters.template get<double>("mu_c");

      // Length scale parameter
      L_c_ = parameters.template get<double>("L_c");

      // Curvature parameters
      b1_ = parameters.template get<double>("b1");
      b2_ = parameters.template get<double>("b2");
      b3_ = parameters.template get<double>("b3");
    }

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

      // The element geometry
      auto element = localView.element();

      ////////////////////////////////////////////////////////////////////////////////////
      //  Set up the local nonlinear finite element function
      ////////////////////////////////////////////////////////////////////////////////////
      using namespace Dune::TypeTree::Indices;

      // The set of shape functions on this element
      const auto& deformationLocalFiniteElement = localView.tree().child(_0,0).finiteElement();
#if MIXED_SPACE
      const auto& orientationLocalFiniteElement = localView.tree().child(_1,0).finiteElement();

      typedef LocalGeodesicFEFunction<gridDim, DT, decltype(deformationLocalFiniteElement), RBM0> LocalGFEFunctionType0;
      typedef LocalGeodesicFEFunction<gridDim, DT, decltype(orientationLocalFiniteElement), RBM1> LocalGFEFunctionType1;
      LocalGFEFunctionType0 localGeodesicFEFunction0(deformationLocalFiniteElement,localCoefficients[_0]);
      LocalGFEFunctionType1 localGeodesicFEFunction1(orientationLocalFiniteElement,localCoefficients[_1]);
#else
      std::vector<RBM> localSolutionRBM(localCoefficients[_0].size());
      for (std::size_t i = 0; i < localCoefficients[_0].size(); i++) {
        localSolutionRBM[i][_0] = localCoefficients[_0][i];
        localSolutionRBM[i][_1] = localCoefficients[_1][i];
      }
      typedef LocalGeodesicFEFunction<gridDim, DT, decltype(deformationLocalFiniteElement), RBM> LocalGFEFunctionType;
      LocalGFEFunctionType localGeodesicFEFunction(deformationLocalFiniteElement,localSolutionRBM);
#endif

      RT energy = 0;

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
          double thickness = thicknessF_(quadPosGlobal);
          auto lameConstants = lameF_(quadPosGlobal);
          double mu = lameConstants[0];
          double lambda = lameConstants[1];

          const DT integrationElement = boundaryGeometry.integrationElement(quad[pt].position());

          RBM value;
          Dune::FieldMatrix<RT, RBM::embeddedDim, dimWorld> derivative;
          Dune::FieldMatrix<RT, RBM::embeddedDim, boundaryDim> derivative2D;

#if MIXED_SPACE
          // The value of the local function
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
#else
          value = localGeodesicFEFunction.evaluate(quadPos);
          derivative = localGeodesicFEFunction.evaluateDerivative(quadPos,value);
          for (int i = 0; i < RBM::embeddedDim; i++)
            for (int j = 0; j < boundaryDim; j++)
              derivative2D[i][j] = derivative[i][j];
#endif

          //////////////////////////////////////////////////////////
          //  The rotation and its derivative
          //  Note: we need it in matrix coordinates
          //////////////////////////////////////////////////////////

          Dune::FieldMatrix<RT,dimWorld,dimWorld> R;
          value[_1].matrix(R);
          auto rt = Dune::GFE::transpose(R);

          Tensor3<RT,dimWorld,dimWorld,boundaryDim> derivative_rotation;
          computeDR(value, derivative2D, derivative_rotation);

          //////////////////////////////////////////////////////////
          //  Fundamental forms and curvature
          //////////////////////////////////////////////////////////

          // First fundamental form
          Dune::FieldMatrix<double,dimWorld,dimWorld> aCovariant;

          // If dimWorld==3, then the first two lines of aCovariant are simply the jacobianTransposed
          // of the element.  If dimWorld<3 (i.e., ==2), we have to explicitly enters 0.0 in the last column.
          const auto jacobianTransposed = boundaryGeometry.jacobianTransposed(quad[pt].position());

          for (int i=0; i<2; i++)
          {
            for (int j=0; j<dimWorld; j++)
              aCovariant[i][j] = jacobianTransposed[i][j];
            for (int j=dimWorld; j<3; j++)
              aCovariant[i][j] = 0.0;
          }

          aCovariant[2] = Dune::FMatrixHelp::Impl::crossProduct(aCovariant[0], aCovariant[1]);
          aCovariant[2] /= aCovariant[2].two_norm();

          auto aContravariant = aCovariant;
          aContravariant.invert();
          // The contravariant base vectors are the *columns* of the inverse of the covariant matrix
          // To get an easier access to the columns, we use the transpose of the contravariant matrix
          aContravariant = Dune::GFE::transpose(aContravariant);

          Dune::FieldMatrix<double,3,3> a(0);
          for (int alpha=0; alpha<boundaryDim; alpha++)
            a += Dune::GFE::dyadicProduct(aCovariant[alpha], aContravariant[alpha]);

          auto a00 = aCovariant[0] * aCovariant[0];
          auto a01 = aCovariant[0] * aCovariant[1];
          auto a10 = aCovariant[1] * aCovariant[0];
          auto a11 = aCovariant[1] * aCovariant[1];
          auto aScalar = std::sqrt(a00*a11 - a10*a01);

          // Alternator tensor
          Dune::FieldMatrix<int,2,2> eps = {{0,1},{-1,0}};
          Dune::FieldMatrix<double,3,3> c(0);

          for (int alpha=0; alpha<2; alpha++)
            for (int beta=0; beta<2; beta++)
              c += aScalar * eps[alpha][beta] * Dune::GFE::dyadicProduct(aContravariant[alpha], aContravariant[beta]);

          // Second fundamental form: The derivative of the normal field
#if HAVE_DUNE_CURVEDGEOMETRY
          auto b = boundaryGeometry.normalGradient(quad[pt].position());
          b *= (-1);
#else
          // If we do not have the dune-curvedgeometry module then we have to assume
          // that the boundary is flat.  This may not be true, but there is currently
          // no way to get this information from the dune-grid grid interface.
          // TODO: Fix this once the grid interface is extended.
          FieldMatrix<DT,dimWorld,dimWorld> b(0);
#endif

          // Mean curvatue
          auto H = 0.5 * Dune::GFE::trace(b);

          // Gauss curvature, calculated with the normalGradient in the eucidean coordinate system
          // see e.g. formula (3.5) from "Reﬁned dimensional reduction for isotropic elastic Cosserat shells with initial curvature"
          auto bSquared = b*b;
          auto K = 2*H*H - 0.5*Dune::GFE::trace(bSquared);

          //////////////////////////////////////////////////////////
          //  Strain tensors
          //////////////////////////////////////////////////////////

          // Elastic shell strain
          Dune::FieldMatrix<RT,3,3> Ee(0);
          Dune::FieldMatrix<RT,3,3> grad_s_m(0);
          for (int alpha=0; alpha<boundaryDim; alpha++)
          {
            Dune::FieldVector<RT,3> vec;
            for (int i=0; i<3; i++)
              vec[i] = derivative[i][alpha];
            grad_s_m += Dune::GFE::dyadicProduct(vec, aContravariant[alpha]);
          }

          Ee = rt * grad_s_m - a;

          // Elastic shell bending-curvature strain
          Dune::FieldMatrix<RT,3,3> Ke(0);
          for (int alpha=0; alpha<boundaryDim; alpha++)
          {
            Dune::FieldMatrix<RT,3,3> tmp;
            for (int i=0; i<3; i++)
              for (int j=0; j<3; j++)
                tmp[i][j] = derivative_rotation[i][j][alpha];
            auto tmp2 = rt * tmp;
            Ke += Dune::GFE::dyadicProduct(SkewMatrix<RT,3>(tmp2).axial(), aContravariant[alpha]);
          }

          //////////////////////////////////////////////////////////
          // Add the local energy density
          //////////////////////////////////////////////////////////

          // Add the membrane energy density
          auto energyDensity = (thickness - K*Dune::power(thickness,3) / 12.0) * W_m(Ee, mu, lambda);
          energyDensity += (Dune::power(thickness,3) / 12.0 - K * Dune::power(thickness,5) / 80.0)*W_m(Ee*b + c*Ke, mu, lambda);
          energyDensity += Dune::power(thickness,3) / 6.0 * W_mixt(Ee, c*Ke*b - 2*H*c*Ke, mu, lambda);
          energyDensity += Dune::power(thickness,5) / 80.0 * W_mp( (Ee*b + c*Ke)*b, mu, lambda);

          // Add the bending energy density
          energyDensity += (thickness - K*Dune::power(thickness,3) / 12.0) * W_curv(Ke, mu)
                           + (Dune::power(thickness,3) / 12.0 - K * Dune::power(thickness,5) / 80.0)*W_curv(Ke*b, mu)
                           + Dune::power(thickness,5) / 80.0 * W_curv(Ke*b*b, mu);

          // Add energy density
          energy += quad[pt].weight() * integrationElement * energyDensity;
        }
      }

      return energy;
    }

    RT W_m(const Dune::FieldMatrix<RT,3,3>& S, double mu, double lambda) const
    {
      return W_mixt(S,S, mu, lambda);
    }

    RT W_mixt(const Dune::FieldMatrix<RT,3,3>& S, const Dune::FieldMatrix<RT,3,3>& T, double mu, double lambda) const
    {
      return mu * Dune::GFE::frobeniusProduct(Dune::GFE::sym(S), Dune::GFE::sym(T))
             + mu_c_ * Dune::GFE::frobeniusProduct(Dune::GFE::skew(S), Dune::GFE::skew(T))
             + lambda * mu / (lambda + 2*mu) * Dune::GFE::trace(S) * Dune::GFE::trace(T);
    }

    RT W_mp(const Dune::FieldMatrix<RT,3,3>& S, double mu, double lambda) const
    {
      return mu * Dune::GFE::sym(S).frobenius_norm2() + mu_c_ * Dune::GFE::skew(S).frobenius_norm2() + lambda * 0.5 * Dune::GFE::traceSquared(S);
    }

    RT W_curv(const Dune::FieldMatrix<RT,3,3>& S, double mu) const
    {
      return mu * L_c_ * L_c_ * (b1_ * Dune::GFE::dev(Dune::GFE::sym(S)).frobenius_norm2()
                                 + b2_ * Dune::GFE::skew(S).frobenius_norm2() + b3_ * Dune::GFE::traceSquared(S));
    }

  private:
    /** \brief The shell boundary */
    const BoundaryPatch<GridView>* shellBoundary_;

    /** \brief The function used to create the Geometries used for assembling */
    const CurvedGeometryGridFunction curvedGeometryGridFunction_;

    /** \brief The shell thickness as a function*/
    std::function<double(Dune::FieldVector<double,dimWorld>)> thicknessF_;

    /** \brief The Lamé-parameters as a function*/
    std::function<Dune::FieldVector<double,2>(Dune::FieldVector<double,dimWorld>)> lameF_;

    /** \brief Lame constants */
    double mu_, lambda_;

    /** \brief Cosserat couple modulus, preferably 0 */
    double mu_c_;

    /** \brief Length scale parameter */
    double L_c_;

    /** \brief Curvature parameters */
    double b1_, b2_, b3_;

  };
}  // namespace Dune::GFE

#endif   //#ifndef DUNE_GFE_SURFACECOSSERATENERGY_HH
