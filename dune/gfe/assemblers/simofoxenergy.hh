#ifndef DUNE_GFE_SIMOFOX_ENERGY_HH
#define DUNE_GFE_SIMOFOX_ENERGY_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/transpose.hh>
#include <dune/common/tuplevector.hh>
#include <dune/functions/functionspacebases/subspacebasis.hh>
#include <dune/fufem/boundarypatch.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/gfe/linearalgebra.hh>
#include <dune/gfe/assemblers/localenergy.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/unitvector.hh>

namespace Dune::GFE
{

  /** \brief Get LocalFiniteElements from a localView, for different tree depths of the local view
   *
   * Copied from CosseratEnergyLocalStiffness.hh
   */
  template <class Basis, std::size_t i>
  class LocalFiniteElementFactory {
  public:
    static auto get(const typename Basis::LocalView &localView, std::integral_constant<std::size_t, i> iType)
    -> decltype(localView.tree().child(iType, 0).finiteElement()) {
      return localView.tree().child(iType, 0).finiteElement();
    }
  };

  /** \brief Specialize for scalar bases, here we cannot call tree().child() */
  template <class GridView, int order, std::size_t i>
  class LocalFiniteElementFactory<Dune::Functions::LagrangeBasis<GridView, order>, i> {
  public:
    static auto get(const typename Dune::Functions::LagrangeBasis<GridView, order>::LocalView &localView,
                    std::integral_constant<std::size_t, i> iType) -> decltype(localView.tree().finiteElement()) {
      return localView.tree().finiteElement();
    }
  };


  /** \brief Implements the energy of a Simo-Fox shell
   *
   * Described in:
   *  "A consistent finite element formulation of the geometrically non-linear Reissner-Mindlin shell model [Müller,
   * Bischoff]"
   *
   * \tparam LocalFEFunction Decides which interpolation function is used for the interpolation of the midsurface
   * and director field.
   */
  template <class Basis, template <typename, typename> typename LocalFEFunction, typename field_type = double>
  class SimoFoxEnergyLocalStiffness
    : public LocalEnergy<Basis, ProductManifold<RealTuple<field_type, 3>,
          UnitVector<field_type, 3> > >
  {
    using TargetSpace = ProductManifold<RealTuple<field_type, 3>, UnitVector<field_type, 3> >;

    // grid types
    typedef typename Basis::GridView GridView;
    typedef typename GridView::ctype DT;
    typedef field_type RT;
    typedef typename GridView::template Codim<0>::Entity Entity;

    // some other sizes
    static constexpr int gridDim  = GridView::dimension;
    static constexpr int dimworld = GridView::dimensionworld;

  public:
    /** \brief Constructor with a set of material parameters
     * \param parameters The material parameters
     * \param x0 reference configuration
     */
    SimoFoxEnergyLocalStiffness(const Dune::ParameterTree &parameters, const BoundaryPatch<GridView> *neumannBoundary,
                                const std::function<Dune::FieldVector<double, 3>(Dune::FieldVector<double, 2>)> neumannFunction,
                                const std::function<Dune::FieldVector<double, 3>(Dune::FieldVector<double, 2>)> volumeLoad,
                                const Dune::TupleVector<std::vector<RealTuple<double, 3> >, std::vector<UnitVector<double, 3> > > &x0)
      : neumannBoundary_(neumannBoundary),
      neumannFunction_(neumannFunction),
      thickness_{parameters.template get<double>("thickness")},      // The sheeqll thickness
      mu_{parameters.template get<double>("mu")},                    // Lame constant 1
      lambda_{parameters.template get<double>("lambda")},            // Lame constant 2
      kappa_{parameters.template get<double>("kappa")},              // Shear correction factor
      midSurfaceRefConfig{x0[Dune::Indices::_0]},
      directorRefConfig{x0[Dune::Indices::_1]}
    {
      // Calculate the over the thickness preintegrated St.Venant Kirchhoff material matrix, Paper equation 10.1 */
      const double Emodul = mu_ * (3 * lambda_ + 2 * mu_) / (lambda_ + mu_);  // Young's modulus
      const double nu     = lambda_ / (2 * (lambda_ + mu_));                  // Poisson ratio

      // membrane
      const double fac1 = thickness_ * Emodul / (1 - nu * nu);
      CMat_[0][0] = CMat_[1][1] = fac1;
      CMat_[2][2]               = fac1 * (1 - nu) * 0.5;
      CMat_[1][0] = CMat_[0][1] = fac1 * nu;

      // bending
      const double fac2 = thickness_ * thickness_ * thickness_ / 12 * Emodul / (1 - nu * nu);
      CMat_[3][3] = CMat_[4][4] = fac2;
      CMat_[5][5]               = fac2 * (1 - nu) * 0.5;
      CMat_[3][4] = CMat_[4][3] = fac2 * nu;

      // transverse shear
      const double fac3 = kappa_ * thickness_ * Emodul * 0.5 / (1 + nu);
      CMat_[6][6] = CMat_[7][7] = fac3;
    }

    /** \brief Assemble the energy for a single element */
    RT energy(const typename Basis::LocalView &localView,
              const std::vector<TargetSpace> &localConfiguration) const override
    {
      DUNE_THROW(NotImplemented, "!");
    }

    RT energy (const typename Basis::LocalView& localView,
               const typename Impl::LocalEnergyTypes<TargetSpace>::CompositeCoefficients& coefficients) const override;

  private:
    /** \brief A structure that contains all quantities to calculate the Lagrangian strains */
    struct KinematicVariables {
      // current configuration
      FieldMatrix<field_type, 2, 3> a1anda2;    // first and second tangent base vector on the current geometry
      FieldMatrix<field_type, 2, 3> td1Andtd2;  // partial derivative of the director on the current geometry in first and second direction
      FieldMatrix<field_type, 2, 3> ud1andud2;  // partial derivative of the displacement function in first and second direction
      FieldVector<field_type, 3> t;             // unit director on the current geometry

      // reference configuration
      FieldMatrix<double, 2, 3> A1andA2;      // first and second tangent base vector on the reference geometry
      FieldMatrix<double, 2, 3> t0d1Andt0d2;  // partial derivative of the director on the reference geometry in first and second direction
      FieldVector<double, 3> t0;              // unit director on the reference geometry
    };

    auto getReferenceLocalConfigurations(const typename Basis::LocalView &localView) const;

    /** \brief Calculates all kinematic quantities */
    template <typename Element, typename LocalDirectorFunction, typename LocalMidSurfaceFunction, typename LocalDirectorReferenceFunction,
        typename LocalMidSurfaceReferenceFunction, typename IntegrationPointPosition>
    static auto kinematicVariablesFactory(const Element &element, const LocalDirectorFunction &directorFunction,
                                          const LocalDirectorReferenceFunction &directorReferenceFunction,
                                          const LocalMidSurfaceFunction &midSurfaceFunction,
                                          const LocalMidSurfaceReferenceFunction &midSurfaceReferenceFunction,
                                          const LocalMidSurfaceFunction &midSurfaceDisplacementFunction, const IntegrationPointPosition &quadPos);

    /** \brief Calculate the Green-Lagrange strain components */
    static Dune::FieldVector<RT, 8> calculateGreenLagrangianStrains(const KinematicVariables &kin);

    /** \brief Save all tangent base matrices for all nodes in one place*/
    Dune::BlockVector<Dune::FieldMatrix<field_type, 2, 3> > directorTangentSpaces;

    /** \brief The Neumann boundary */
    const BoundaryPatch<GridView> *neumannBoundary_;

    /** \brief The function implementing the Neumann data */
    const std::function<Dune::FieldVector<double, 3>(Dune::FieldVector<double, dimworld>)> neumannFunction_;

    /** \brief The shell thickness */
    double thickness_;

    /** \brief Lame constants */
    double mu_, lambda_;

    /** \brief Shear correction factor */
    double kappa_;

    /** \brief Material tangent matrix */
    Dune::FieldMatrix<double, 8, 8> CMat_;

    /** \brief The function implementing a volume load */
    const std::function<Dune::FieldVector<double, 3>(Dune::FieldVector<double, dimworld>)> volumeLoad_;

    /** \brief Stores the reference configuration of the midsurface and the director field */
    const std::vector<RealTuple<double, 3> > &midSurfaceRefConfig;
    const std::vector<UnitVector<double, 3> > &directorRefConfig;
  };

  /** \brief Calculate the Green-Lagrange strain components for the thickness-integrated setting
   *
   * The components are returned as [a_11, a_22, a_12, b_11, b_22, b_12, gamma_1, gamma_2]
   * where a is the midsurface metric,
   * b is similar to the second fundamental form of the surface, but the unit normal is replaced with the shell director
   * gamma_1 and gamma_2 are the transverse shear in the two parametric directions
   * Paper Equation 4.10 */
  template <class Basis, template <typename, typename> typename LocalFEFunction, typename field_type>
  Dune::FieldVector<field_type, 8> SimoFoxEnergyLocalStiffness<Basis, LocalFEFunction, field_type>::calculateGreenLagrangianStrains(
    const KinematicVariables &kin)
  {
    Dune::FieldVector<RT, 8> egl;
    // membrane
    egl[0] = kin.A1andA2[0] * kin.ud1andud2[0] + 0.5 * kin.ud1andud2[0].two_norm2();
    egl[1] = kin.A1andA2[1] * kin.ud1andud2[1] + 0.5 * kin.ud1andud2[1].two_norm2();
    egl[2] = kin.a1anda2[0] * kin.a1anda2[1];

    // bending
    egl[3] = kin.a1anda2[0] * kin.td1Andtd2[0] - kin.A1andA2[0] * kin.t0d1Andt0d2[0];
    egl[4] = kin.a1anda2[1] * kin.td1Andtd2[1] - kin.A1andA2[1] * kin.t0d1Andt0d2[1];
    egl[5] = kin.a1anda2[0] * kin.td1Andtd2[1] + kin.a1anda2[1] * kin.td1Andtd2[0] - kin.A1andA2[0] * kin.t0d1Andt0d2[1]
             - kin.A1andA2[1] * kin.t0d1Andt0d2[0];

    // transverse shear
    egl[6] = kin.a1anda2[0] * kin.t - kin.A1andA2[0] * kin.t0;
    egl[7] = kin.a1anda2[1] * kin.t - kin.A1andA2[1] * kin.t0;

    return egl;
  }

  /** \brief Calculates all kinematic quantities that are needed for strain calculation
   *
   * Paper Equations 6.4 - 6.8
   */
  template <class Basis, template <typename, typename> typename LocalFEFunction, typename field_type>
  template <typename Element, typename LocalDirectorFunction, typename LocalMidSurfaceFunction, typename LocalDirectorReferenceFunction,
      typename LocalMidSurfaceReferenceFunction, typename IntegrationPointPosition>
  auto SimoFoxEnergyLocalStiffness<Basis, LocalFEFunction, field_type>::kinematicVariablesFactory(
    const Element &element, const LocalDirectorFunction &directorFunction, const LocalDirectorReferenceFunction &directorReferenceFunction,
    const LocalMidSurfaceFunction &midSurfaceFunction, const LocalMidSurfaceReferenceFunction &midSurfaceReferenceFunction,
    const LocalMidSurfaceFunction &midSurfaceDisplacementFunction, const IntegrationPointPosition &quadPos)
  {
    KinematicVariables kin{};

    const auto jInvT = element.geometry().jacobianInverseTransposed(quadPos);

    kin.t           = directorFunction.evaluate(quadPos).globalCoordinates();
    kin.t0          = directorReferenceFunction.evaluate(quadPos).globalCoordinates();
    kin.a1anda2     = transpose(midSurfaceFunction.evaluateDerivative(quadPos) * transpose(jInvT));
    kin.A1andA2     = transpose(midSurfaceReferenceFunction.evaluateDerivative(quadPos) * transpose(jInvT));
    kin.ud1andud2   = transpose(midSurfaceDisplacementFunction.evaluateDerivative(quadPos) * transpose(jInvT));
    kin.t0d1Andt0d2 = transpose(directorReferenceFunction.evaluateDerivative(quadPos) * transpose(jInvT));
    kin.td1Andtd2   = transpose(directorFunction.evaluateDerivative(quadPos) * transpose(jInvT));

    return kin;
  }

  template <class Basis, template <typename, typename> typename LocalFEFunction, typename field_type>
  auto SimoFoxEnergyLocalStiffness<Basis, LocalFEFunction, field_type>::getReferenceLocalConfigurations(
    const typename Basis::LocalView &localView) const {
    using namespace Dune::Indices;
    const int nDofs0 = localView.tree().child(_0, 0).finiteElement().size();
    const int nDofs1 = localView.tree().child(_1, 0).finiteElement().size();

    std::vector<RealTuple<double, 3> > localConfiguration0(nDofs0);
    std::vector<UnitVector<double, 3> > localConfiguration1(nDofs1);

    for (int i = 0; i < nDofs0 + nDofs1; i++) {
      int localIndexI = 0;
      if (i < nDofs0) {
        auto &node  = localView.tree().child(_0, 0);
        localIndexI = node.localIndex(i);
      } else {
        auto &node  = localView.tree().child(_1, 0);
        localIndexI = node.localIndex(i - nDofs0);
      }
      auto multiIndex = localView.index(localIndexI);

      // The CompositeBasis number is contained in multiIndex[0]
      // multiIndex[1] contains the actual index
      if (multiIndex[0] == 0)
        localConfiguration0[i] = midSurfaceRefConfig[multiIndex[1]];
      else if (multiIndex[0] == 1)
        localConfiguration1[i - nDofs0] = directorRefConfig[multiIndex[1]];
    }
    return std::make_tuple(localConfiguration0, localConfiguration1);
  }

  /** \brief Computes the Simo-Fox shell energy
   *
   *  The energy is 0.5 * S * E = 0.5 * transpose(E) * Cmat * E, where
   *  S are the stress resultants [membrane forces, bending moments, transverse shear forces]
   *  E are the components of the Green-Lagrangian strains [membrane strains, bending, transverse shear]
   *  see for details Paper Equation 4.11,4.10 and 10.1
   */
  template <class Basis, template <typename, typename> typename LocalFEFunction, typename field_type>
  typename SimoFoxEnergyLocalStiffness<Basis, LocalFEFunction, field_type>::RT
  SimoFoxEnergyLocalStiffness<Basis, LocalFEFunction, field_type>::energy(
    const typename Basis::LocalView &localView,
    const typename Impl::LocalEnergyTypes<TargetSpace>::CompositeCoefficients &localConfiguration) const
  {
    using namespace Dune::Indices;
    const auto& localMidSurfaceConfiguration = localConfiguration[_0];
    const auto& localDirectorConfiguration = localConfiguration[_1];

    auto element = localView.element();

    const auto &midSurfaceElement = LocalFiniteElementFactory<Basis, 0>::get(localView, _0);
    const auto &directorElement   = LocalFiniteElementFactory<Basis, 1>::get(localView, _1);

    const auto [localRefMidSurfaceConfiguration, localRefDirectorConfiguration] = getReferenceLocalConfigurations(localView);

    std::vector<RealTuple<field_type, 3> > displacements;
    displacements.reserve(localMidSurfaceConfiguration.size());
    for (size_t i = 0; i < localMidSurfaceConfiguration.size(); ++i)
      displacements.emplace_back(localMidSurfaceConfiguration[i].globalCoordinates() - localRefMidSurfaceConfiguration[i].globalCoordinates());

    // The local finite element type used for midsurface position and displacement interpolation
    const auto midSurfaceBasis = Functions::subspaceBasis(localView.globalBasis(),_0,0);
    const auto directorBasis = Functions::subspaceBasis(localView.globalBasis(),_1,0);

    // The local finite element function type to evaluate the midsurface position
    using LocalMidSurfaceFunctionType = LocalFEFunction<decltype(midSurfaceBasis), RealTuple<field_type, 3> >;
    // The local finite element function type to evaluate the director
    using LocalDirectorFunctionType = LocalFEFunction<decltype(directorBasis), UnitVector<field_type, 3> >;

    // Extra function type for reference quantities since they are unconditionally doubles, i.e. no ADOL-C types
    using LocalMidSurfaceReferenceFunctionType = LocalFEFunction<decltype(midSurfaceBasis), RealTuple<double, 3> >;
    using LocalDirectorReferenceFunctionType   = LocalFEFunction<decltype(directorBasis), UnitVector<double, 3> >;

    const LocalMidSurfaceFunctionType localMidSurfaceFunction(midSurfaceElement, localMidSurfaceConfiguration);
    const LocalMidSurfaceFunctionType localMidSurfaceDisplacementFunction(midSurfaceElement, displacements);
    const LocalMidSurfaceReferenceFunctionType localMidSurfaceReferenceFunction(midSurfaceElement, localRefMidSurfaceConfiguration);
    const LocalDirectorFunctionType localDirectorFunction(directorElement, localDirectorConfiguration);
    const LocalDirectorReferenceFunctionType localDirectorReferenceFunction(directorElement, localRefDirectorConfiguration);

    const int quadOrder = (element.type().isSimplex()) ? std::max(midSurfaceElement.localBasis().order(), directorElement.localBasis().order())
                                                       : std::max(midSurfaceElement.localBasis().order(), directorElement.localBasis().order()) + 1;

    const auto &quad = Dune::QuadratureRules<DT, gridDim>::rule(element.type(), quadOrder);

    RT energy = 0;
    for (const auto &curQuad : quad) {
      const Dune::FieldVector<DT, gridDim> &quadPos = curQuad.position();

      const KinematicVariables kin
        = kinematicVariablesFactory(element, localDirectorFunction, localDirectorReferenceFunction, localMidSurfaceFunction,
                                    localMidSurfaceReferenceFunction, localMidSurfaceDisplacementFunction, quadPos);

      const DT integrationElement = element.geometry().integrationElement(quadPos);
      const FieldVector<field_type, 8> Egl = calculateGreenLagrangianStrains(kin);
      energy +=  0.5 * Egl * (CMat_ * Egl) * curQuad.weight() * integrationElement;
    }

    //////////////////////////////////////////////////////////////////////////////
    //   Assemble boundary contributions
    //////////////////////////////////////////////////////////////////////////////

    if (not neumannFunction_) return energy;

    for (auto &&it : intersections(neumannBoundary_->gridView(), element))
    {
      if (not neumannBoundary_ or not neumannBoundary_->contains(it)) continue;

      const auto &quadLine = QuadratureRules<DT, gridDim - 1>::rule(it.type(), quadOrder);

      for (const auto &curQuad : quadLine)
      {
        // Local position of the quadrature point
        const FieldVector<DT, gridDim> &quadPos = it.geometryInInside().global(curQuad.position());

        const DT integrationElement = it.geometry().integrationElement(curQuad.position());

        // The value of the local function
        RealTuple<field_type, 3> deformationValue = localMidSurfaceFunction.evaluate(quadPos);

        // Value of the Neumann data at the current position
        auto neumannValue = neumannFunction_(it.geometry().global(curQuad.position()));

        // Only translational dofs are affected by the Neumann force
        for (size_t i = 0; i < neumannValue.size(); i++)
          energy -= (neumannValue[i] * deformationValue.globalCoordinates()[i]) * curQuad.weight() * integrationElement;
      }
    }
    return energy;
  }

}  // namespace Dune::GFE

#endif  // DUNE_GFE_SIMOFOX_ENERGY_HH
