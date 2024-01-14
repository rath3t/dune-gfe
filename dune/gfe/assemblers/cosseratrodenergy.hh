#ifndef DUNE_GFE_COSSERATRODENERGY_HH
#define DUNE_GFE_COSSERATRODENERGY_HH

#include <array>

#include <dune/common/fmatrix.hh>
#include <dune/common/version.hh>
#if DUNE_VERSION_GTE(DUNE_COMMON, 2, 8)
#include <dune/common/transpose.hh>
#endif

#include <dune/istl/matrix.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>

#include <dune/fufem/boundarypatch.hh>

#include <dune/gfe/assemblers/localenergy.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>

namespace Dune::GFE {

  template<class Basis, class LocalInterpolationRule, class RT>
  class CosseratRodEnergy
    : public LocalEnergy<Basis, ProductManifold<RealTuple<RT,3>,Rotation<RT,3> > >
  {
    using TargetSpace = ProductManifold<RealTuple<RT,3>,Rotation<RT,3> >;

    // grid types
    using GridView = typename Basis::GridView;
    typedef typename GridView::Grid::ctype DT;
    typedef typename GridView::template Codim<0>::Entity Entity;

    // some other sizes
    constexpr static int dim = GridView::dimension;

    // Quadrature order used for the extension and shear energy
    constexpr static int shearQuadOrder = 2;

    // Quadrature order used for the bending and torsion energy
    constexpr static int bendingQuadOrder = 2;

  public:

    /** \brief The stress-free configuration

       The number type cannot be RT, because RT become `adouble` when
       using RodLocalStiffness together with an AD system.
       The referenceConfiguration is not a variable, and we don't
       want to use `adouble` for it.
     */
    std::vector<ProductManifold<RealTuple<double,3>,Rotation<double,3> > > referenceConfiguration_;

  public:

    //! Each block is x, y, theta in 2d, T (R^3 \times SO(3)) in 3d
    static constexpr auto blocksize = TargetSpace::EmbeddedTangentVector::dimension;

    // /////////////////////////////////
    //   The material parameters
    // /////////////////////////////////

    /** \brief Material constants */
    std::array<double,3> K_;
    std::array<double,3> A_;

    GridView gridView_;

    //! Constructor
    CosseratRodEnergy(const GridView& gridView,
                      const std::array<double,3>& K, const std::array<double,3>& A)
      : K_(K),
      A_(A),
      gridView_(gridView)
    {}

    /** \brief Constructor setting shape constants and material parameters
        \param A The rod section area
        \param J1, J2 The geometric moments (Flächenträgheitsmomente)
        \param E Young's modulus
        \param nu Poisson number
     */
    CosseratRodEnergy(const GridView& gridView,
                      double A, double J1, double J2, double E, double nu)
      : gridView_(gridView)
    {
      // shear modulus
      double G = E/(2+2*nu);

      K_[0] = E * J1;
      K_[1] = E * J2;
      K_[2] = G * (J1 + J2);

      A_[0] = G * A;
      A_[1] = G * A;
      A_[2] = E * A;
    }

    /** \brief Set the stress-free configuration
     */
    void setReferenceConfiguration(const std::vector<ProductManifold<RealTuple<double,3>,Rotation<double,3> > >& referenceConfiguration) {
      referenceConfiguration_ = referenceConfiguration;
    }

    /** \brief Compute local element energy */
    virtual RT energy (const typename Basis::LocalView& localView,
                       const std::vector<TargetSpace>& localSolution) const override;

    /** \brief Get the rod strain at one point in the rod
     *
     * \tparam Number This is a member template because the method has to work for double and adouble
     */
    template<class ReboundLocalInterpolationRule>
    auto getStrain(const ReboundLocalInterpolationRule& localSolution,
                   const Entity& element,
                   const FieldVector<double,1>& pos) const;

    /** \brief Get the rod stress at one point in the rod
     *
     * \tparam Number This is a member template because the method has to work for double and adouble
     */
    template<class Number>
    auto getStress(const std::vector<ProductManifold<RealTuple<Number,3>,Rotation<Number,3> > >& localSolution,
                   const Entity& element,
                   const FieldVector<double,1>& pos) const;

    /** \brief Get average strain for each element */
    void getStrain(const std::vector<ProductManifold<RealTuple<double,3>,Rotation<double,3> > >& sol,
                   BlockVector<FieldVector<double, blocksize> >& strain) const;

    /** \brief Get average stress for each element */
    void getStress(const std::vector<ProductManifold<RealTuple<double,3>,Rotation<double,3> > >& sol,
                   BlockVector<FieldVector<double, blocksize> >& stress) const;

    /** \brief Return resultant force across boundary in canonical coordinates

       \note Linear run-time in the size of the grid */
    template <class PatchGridView>
    auto getResultantForce(const BoundaryPatch<PatchGridView>& boundary,
                           const std::vector<ProductManifold<RealTuple<double,3>,Rotation<double,3> > >& sol) const;

  protected:

    std::vector<ProductManifold<RealTuple<double,3>,Rotation<double,3> > > getLocalReferenceConfiguration(const typename Basis::LocalView& localView) const
    {
      unsigned int numOfBaseFct = localView.size();
      std::vector<ProductManifold<RealTuple<double,3>,Rotation<double,3> > > localReferenceConfiguration(numOfBaseFct);

      for (size_t i=0; i<numOfBaseFct; i++)
        localReferenceConfiguration[i] = referenceConfiguration_[localView.index(i)];

      return localReferenceConfiguration;
    }

    template <class T>
    static FieldVector<T,3> darboux(const Rotation<T,3>& q, const FieldVector<T,4>& q_s)
    {
      FieldVector<T,3> u;    // The Darboux vector

      u[0] = 2 * (q.B(0) * q_s);
      u[1] = 2 * (q.B(1) * q_s);
      u[2] = 2 * (q.B(2) * q_s);

      return u;
    }

  };

  template<class Basis, class LocalInterpolationRule, class RT>
  RT CosseratRodEnergy<Basis, LocalInterpolationRule, RT>::
  energy(const typename Basis::LocalView& localView,
         const std::vector<TargetSpace>& localCoefficients) const
  {
    const auto& localFiniteElement = localView.tree().finiteElement();
    LocalInterpolationRule localConfiguration(localFiniteElement, localCoefficients);

    const auto& element = localView.element();

    RT energy = 0;

    std::vector<ProductManifold<RealTuple<double,3>,Rotation<double,3> > > localReferenceCoefficients = getLocalReferenceConfiguration(localView);
    using InactiveLocalInterpolationRule = typename LocalInterpolationRule::template rebind<ProductManifold<RealTuple<double,3>,Rotation<double,3> > >::other;
    InactiveLocalInterpolationRule localReferenceConfiguration(localFiniteElement, localReferenceCoefficients);

    // ///////////////////////////////////////////////////////////////////////////////
    //   The following two loops are a reduced integration scheme.  We integrate
    //   the transverse shear and extensional energy with a first-order quadrature
    //   formula, even though it should be second order.  This prevents shear-locking.
    // ///////////////////////////////////////////////////////////////////////////////

    const auto& shearingQuad = QuadratureRules<double, 1>::rule(element.type(), shearQuadOrder);

    for (size_t pt=0; pt<shearingQuad.size(); pt++) {

      // Local position of the quadrature point
      const auto quadPos = shearingQuad[pt].position();

      const double integrationElement = element.geometry().integrationElement(quadPos);

      double weight = shearingQuad[pt].weight() * integrationElement;

      auto strain = getStrain(localConfiguration, element, quadPos);

      // The reference strain
      auto referenceStrain = getStrain(localReferenceConfiguration, element, quadPos);

      for (int i=0; i<3; i++)
        energy += weight * 0.5 * A_[i] * (strain[i] - referenceStrain[i]) * (strain[i] - referenceStrain[i]);

    }

    // Get quadrature rule
    const auto& bendingQuad = QuadratureRules<double, 1>::rule(element.type(), bendingQuadOrder);

    for (size_t pt=0; pt<bendingQuad.size(); pt++) {

      // Local position of the quadrature point
      const FieldVector<double,1>& quadPos = bendingQuad[pt].position();

      double weight = bendingQuad[pt].weight() * element.geometry().integrationElement(quadPos);

      auto strain = getStrain(localConfiguration, element, quadPos);

      // The reference strain
      auto referenceStrain = getStrain(localReferenceConfiguration, element, quadPos);

      // Part II: the bending and twisting energy
      for (int i=0; i<3; i++)
        energy += weight * 0.5 * K_[i] * (strain[i+3] - referenceStrain[i+3]) * (strain[i+3] - referenceStrain[i+3]);

    }

    return energy;
  }


  template<class Basis, class LocalInterpolationRule, class RT>
  template <class ReboundLocalInterpolationRule>
  auto CosseratRodEnergy<Basis, LocalInterpolationRule, RT>::
  getStrain(const ReboundLocalInterpolationRule& localInterpolation,
            const Entity& element,
            const FieldVector<double,1>& pos) const
  {
    const auto jit = element.geometry().jacobianInverseTransposed(pos);

    auto value = localInterpolation.evaluate(pos);

    auto referenceDerivative = localInterpolation.evaluateDerivative(pos);
#if DUNE_VERSION_GTE(DUNE_COMMON, 2, 8)
    auto derivative = referenceDerivative * transpose(jit);
#else
    auto derivative = referenceDerivative;
    derivative *= jit[0][0];
#endif

    using Number = std::decay_t<decltype(derivative[0][0])>;
    FieldVector<Number,3> r_s = {derivative[0][0], derivative[1][0], derivative[2][0]};

    // Transformation from the reference element
    Quaternion<Number> q_s(derivative[3][0],
                           derivative[4][0],
                           derivative[5][0],
                           derivative[6][0]);

    // /////////////////////////////////////////////
    //   Sum it all up
    // /////////////////////////////////////////////

    // Strain defined on each element
    using namespace Dune::Indices;

    // Part I: the shearing and stretching strain
    FieldVector<Number, 6> strain(0);
    strain[0] = r_s * value[_1].director(0);    // shear strain
    strain[1] = r_s * value[_1].director(1);    // shear strain
    strain[2] = r_s * value[_1].director(2);    // stretching strain

    // Part II: the Darboux vector

    FieldVector<Number,3> u = darboux<Number>(value[_1], q_s);
    strain[3] = u[0];
    strain[4] = u[1];
    strain[5] = u[2];

    return strain;
  }

  template<class Basis, class LocalInterpolationRule, class RT>
  template <class Number>
  auto CosseratRodEnergy<Basis, LocalInterpolationRule, RT>::
  getStress(const std::vector<ProductManifold<RealTuple<Number,3>,Rotation<Number,3> > >& localSolution,
            const Entity& element,
            const FieldVector<double, 1>& pos) const
  {
    const auto& indexSet = gridView_.indexSet();
    std::vector<TargetSpace> localRefConf = {referenceConfiguration_[indexSet.subIndex(element, 0, 1)],
                                             referenceConfiguration_[indexSet.subIndex(element, 1, 1)]};

    auto&& strain = getStrain(localSolution, element, pos);
    auto&& referenceStrain = getStrain(localRefConf, element, pos);

    FieldVector<RT, 6> stress;
    for (int i=0; i < dim; i++)
      stress[i] = (strain[i] - referenceStrain[i]) * A_[i];

    for (int i=0; i < dim; i++)
      stress[i+3] = (strain[i+3] - referenceStrain[i+3]) * K_[i];
    return stress;
  }

  template<class Basis, class LocalInterpolationRule, class RT>
  void CosseratRodEnergy<Basis, LocalInterpolationRule, RT>::
  getStrain(const std::vector<ProductManifold<RealTuple<double,3>,Rotation<double,3> > >& sol,
            BlockVector<FieldVector<double, blocksize> >& strain) const
  {
    const typename GridView::Traits::IndexSet& indexSet = this->basis_.gridView().indexSet();

    if (sol.size()!=this->basis_.size())
      DUNE_THROW(Exception, "Solution vector doesn't match the grid!");

    // Strain defined on each element
    strain.resize(indexSet.size(0));
    strain = 0;

    // Loop over all elements
    for (const auto& element : elements(this->basis_.gridView()))
    {
      int elementIdx = indexSet.index(element);

      // Extract local solution on this element
      Dune::LagrangeSimplexLocalFiniteElement<double, double, 1, 1> localFiniteElement;
      int numOfBaseFct = localFiniteElement.localCoefficients().size();

      std::vector<ProductManifold<RealTuple<double,3>,Rotation<double,3> > > localSolution(2);

      for (int i=0; i<numOfBaseFct; i++)
        localSolution[i] = sol[indexSet.subIndex(element,i,1)];

      // Get quadrature rule
      const int polOrd = 2;
      const auto& quad = QuadratureRules<double, 1>::rule(element.type(), polOrd);

      for (std::size_t pt=0; pt<quad.size(); pt++)
      {
        // Local position of the quadrature point
        const auto quadPos = quad[pt].position();

        double weight = quad[pt].weight() * element.geometry().integrationElement(quadPos);

        auto localStrain = getStrain(localSolution, element, quad[pt].position());

        // Sum it all up
        strain[elementIdx].axpy(weight, localStrain);
      }

      // /////////////////////////////////////////////////////////////////////////
      //   We want the average strain per element.  Therefore we have to divide
      //   the integral we just computed by the element volume.
      // /////////////////////////////////////////////////////////////////////////
      // we know the element is a line, therefore the integration element is the volume
      FieldVector<double,1> dummyPos(0.5);
      strain[elementIdx] /= element.geometry().integrationElement(dummyPos);
    }
  }

  template<class Basis, class LocalInterpolationRule, class RT>
  void CosseratRodEnergy<Basis, LocalInterpolationRule, RT>::
  getStress(const std::vector<ProductManifold<RealTuple<double,3>,Rotation<double,3> > >& sol,
            BlockVector<FieldVector<double, blocksize> >& stress) const
  {
    // Get the strain
    getStrain(sol,stress);

    // Get reference strain
    BlockVector<FieldVector<double, blocksize> > referenceStrain;
    getStrain(referenceConfiguration_, referenceStrain);

    // Linear diagonal constitutive law
    for (size_t i=0; i<stress.size(); i++)
    {
      for (int j=0; j<3; j++)
      {
        stress[i][j]   = (stress[i][j]   - referenceStrain[i][j])   * A_[j];
        stress[i][j+3] = (stress[i][j+3] - referenceStrain[i][j+3]) * K_[j];
      }
    }
  }

  template<class Basis, class LocalInterpolationRule, class RT>
  template <class PatchGridView>
  auto CosseratRodEnergy<Basis, LocalInterpolationRule, RT>::
  getResultantForce(const BoundaryPatch<PatchGridView>& boundary,
                    const std::vector<ProductManifold<RealTuple<double,3>,Rotation<double,3> > >& sol) const
  {
    const typename GridView::Traits::IndexSet& indexSet = this->basis_.gridView().indexSet();

    if (sol.size()!=indexSet.size(1))
      DUNE_THROW(Exception, "Solution vector doesn't match the grid!");

    FieldVector<double,3> canonicalStress(0);
    FieldVector<double,3> canonicalTorque(0);

    // Loop over the given boundary
    for (auto facet : boundary)
    {
      // //////////////////////////////////////////////
      //   Compute force across this boundary face
      // //////////////////////////////////////////////

      double pos = facet.geometryInInside().corner(0);

      std::vector<ProductManifold<RealTuple<double,3>,Rotation<double,3> > > localSolution(2);
      localSolution[0] = sol[indexSet.subIndex(*facet.inside(),0,1)];
      localSolution[1] = sol[indexSet.subIndex(*facet.inside(),1,1)];

      std::vector<ProductManifold<RealTuple<double,3>,Rotation<double,3> > > localRefConf(2);
      localRefConf[0] = referenceConfiguration_[indexSet.subIndex(*facet.inside(),0,1)];
      localRefConf[1] = referenceConfiguration_[indexSet.subIndex(*facet.inside(),1,1)];

      auto strain          = getStrain(localSolution, *facet.inside(), pos);
      auto referenceStrain = getStrain(localRefConf, *facet.inside(), pos);

      FieldVector<double,3> localStress;
      for (int i=0; i<3; i++)
        localStress[i] = (strain[i] - referenceStrain[i]) * A_[i];

      FieldVector<double,3> localTorque;
      for (int i=0; i<3; i++)
        localTorque[i] = (strain[i+3] - referenceStrain[i+3]) * K_[i];

      // Transform stress given with respect to the basis given by the three directors to
      // the canonical basis of R^3

      FieldMatrix<double,3,3> orientationMatrix;
      sol[indexSet.subIndex(*facet.inside(),facet.indexInInside(),1)].q.matrix(orientationMatrix);

      orientationMatrix.umv(localStress, canonicalStress);

      orientationMatrix.umv(localTorque, canonicalTorque);

      // Multiply force times boundary normal to get the transmitted force
      canonicalStress *= facet.unitOuterNormal(FieldVector<double,0>(0))[0];
      canonicalTorque *= facet.unitOuterNormal(FieldVector<double,0>(0))[0];
    }

    FieldVector<double,6> result;
    for (int i=0; i<3; i++)
    {
      result[i] = canonicalStress[i];
      result[i+3] = canonicalTorque[i];
    }

    return result;
  }

}  // namespace Dune::GFE

#endif
