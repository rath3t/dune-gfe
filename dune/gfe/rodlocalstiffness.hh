#ifndef ROD_LOCAL_STIFFNESS_HH
#define ROD_LOCAL_STIFFNESS_HH

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

#include <dune/gfe/localenergy.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include "rigidbodymotion.hh"

template<class GridView, class RT>
class RodLocalStiffness
: public Dune::GFE::LocalEnergy<Dune::Functions::LagrangeBasis<GridView,1>, RigidBodyMotion<RT,3> >
{
    typedef RigidBodyMotion<RT,3> TargetSpace;
    typedef Dune::Functions::LagrangeBasis<GridView,1> Basis;

    // grid types
    typedef typename GridView::Grid::ctype DT;
    typedef typename GridView::template Codim<0>::Entity Entity;

    // some other sizes
    enum {dim=GridView::dimension};

    // Quadrature order used for the extension and shear energy
    enum {shearQuadOrder = 2};

    // Quadrature order used for the bending and torsion energy
    enum {bendingQuadOrder = 2};

public:

    /** \brief The stress-free configuration

      The number type cannot be RT, because RT become `adouble` when
      using RodLocalStiffness together with an AD system.
      The referenceConfiguration is not a variable, and we don't
      want to use `adouble` for it.
     */
    std::vector<RigidBodyMotion<double,3> > referenceConfiguration_;

public:

    //! Each block is x, y, theta in 2d, T (R^3 \times SO(3)) in 3d
    enum { blocksize = 6 };

    // define the number of components of your system, this is used outside
    // to allocate the correct size of (dense) blocks with a FieldMatrix
    enum {m=blocksize};

    // types for matrics, vectors and boundary conditions
    typedef Dune::FieldMatrix<RT,m,m> MBlockType; // one entry in the stiffness matrix
    typedef Dune::FieldVector<RT,m> VBlockType;   // one entry in the global vectors

    // /////////////////////////////////
    //   The material parameters
    // /////////////////////////////////

    /** \brief Material constants */
    std::array<double,3> K_;
    std::array<double,3> A_;

    GridView gridView_;

    //! Constructor
    RodLocalStiffness (const GridView& gridView,
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
    RodLocalStiffness (const GridView& gridView,
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



    void setReferenceConfiguration(const std::vector<RigidBodyMotion<double,3> >& referenceConfiguration) {
        referenceConfiguration_ = referenceConfiguration;
    }

    /** \brief Compute local element energy */
    virtual RT energy (const typename Basis::LocalView& localView,
                       const std::vector<RigidBodyMotion<RT,3> >& localSolution) const override;

    /** \brief Get the rod strain at one point in the rod
     *
     * \tparam Number This is a member template because the method has to work for double and adouble
     */
    template<class Number>
    Dune::FieldVector<Number, 6> getStrain(const std::vector<RigidBodyMotion<Number,3> >& localSolution,
                                           const Entity& element,
                                           const Dune::FieldVector<double,1>& pos) const;

    /** \brief Get the rod stress at one point in the rod
     *
     * \tparam Number This is a member template because the method has to work for double and adouble
     */
    template<class Number>
    Dune::FieldVector<Number, 6> getStress(const std::vector<RigidBodyMotion<Number,3> >& localSolution,
                                           const Entity& element,
                                           const Dune::FieldVector<double,1>& pos) const;

    /** \brief Get average strain for each element */
    void getStrain(const std::vector<RigidBodyMotion<double,3> >& sol,
                   Dune::BlockVector<Dune::FieldVector<double, blocksize> >& strain) const;

    /** \brief Get average stress for each element */
    void getStress(const std::vector<RigidBodyMotion<double,3> >& sol,
                   Dune::BlockVector<Dune::FieldVector<double, blocksize> >& stress) const;

    /** \brief Return resultant force across boundary in canonical coordinates

     \note Linear run-time in the size of the grid */
    template <class PatchGridView>
    Dune::FieldVector<double,6> getResultantForce(const BoundaryPatch<PatchGridView>& boundary,
                                                  const std::vector<RigidBodyMotion<double,3> >& sol) const;

protected:

    void getLocalReferenceConfiguration(const Entity& element,
                                        std::vector<RigidBodyMotion<double,3> >& localReferenceConfiguration) const {

        unsigned int numOfBaseFct = element.subEntities(dim);
        localReferenceConfiguration.resize(numOfBaseFct);

        for (size_t i=0; i<numOfBaseFct; i++)
            localReferenceConfiguration[i] = referenceConfiguration_[gridView_.indexSet().subIndex(element,i,dim)];
    }

      template <class T>
    static Dune::FieldVector<T,3> darboux(const Rotation<T,3>& q, const Dune::FieldVector<T,4>& q_s)
    {
        Dune::FieldVector<T,3> u;  // The Darboux vector

        u[0] = 2 * (q.B(0) * q_s);
        u[1] = 2 * (q.B(1) * q_s);
        u[2] = 2 * (q.B(2) * q_s);

        return u;
    }

};

template <class GridView, class RT>
RT RodLocalStiffness<GridView, RT>::
energy(const typename Basis::LocalView& localView,
       const std::vector<RigidBodyMotion<RT,3> >& localSolution) const
{
    assert(localSolution.size()==2);
    const auto& element = localView.element();

    RT energy = 0;

    std::vector<RigidBodyMotion<double,3> > localReferenceConfiguration;
    getLocalReferenceConfiguration(element, localReferenceConfiguration);

    // ///////////////////////////////////////////////////////////////////////////////
    //   The following two loops are a reduced integration scheme.  We integrate
    //   the transverse shear and extensional energy with a first-order quadrature
    //   formula, even though it should be second order.  This prevents shear-locking.
    // ///////////////////////////////////////////////////////////////////////////////

    const Dune::QuadratureRule<double, 1>& shearingQuad
        = Dune::QuadratureRules<double, 1>::rule(element.type(), shearQuadOrder);

    // hack: convert from std::array to std::vector
    std::vector<RigidBodyMotion<RT,3> > localSolutionVector(localSolution.begin(), localSolution.end());

    for (size_t pt=0; pt<shearingQuad.size(); pt++) {

        // Local position of the quadrature point
        const Dune::FieldVector<double,1>& quadPos = shearingQuad[pt].position();

        const double integrationElement = element.geometry().integrationElement(quadPos);

        double weight = shearingQuad[pt].weight() * integrationElement;

        auto strain = getStrain(localSolutionVector, element, quadPos);

        // The reference strain
        auto referenceStrain = getStrain(localReferenceConfiguration, element, quadPos);

        for (int i=0; i<3; i++)
            energy += weight * 0.5 * A_[i] * (strain[i] - referenceStrain[i]) * (strain[i] - referenceStrain[i]);

    }

    // Get quadrature rule
    const Dune::QuadratureRule<double, 1>& bendingQuad
        = Dune::QuadratureRules<double, 1>::rule(element.type(), bendingQuadOrder);

    for (size_t pt=0; pt<bendingQuad.size(); pt++) {

        // Local position of the quadrature point
        const Dune::FieldVector<double,1>& quadPos = bendingQuad[pt].position();

        double weight = bendingQuad[pt].weight() * element.geometry().integrationElement(quadPos);

        auto strain = getStrain(localSolutionVector, element, quadPos);

        // The reference strain
        auto referenceStrain = getStrain(localReferenceConfiguration, element, quadPos);

        // Part II: the bending and twisting energy
        for (int i=0; i<3; i++)
            energy += weight * 0.5 * K_[i] * (strain[i+3] - referenceStrain[i+3]) * (strain[i+3] - referenceStrain[i+3]);

    }

    return energy;
}


template <class GridView, class RT>
template <class Number>
Dune::FieldVector<Number, 6> RodLocalStiffness<GridView, RT>::
getStrain(const std::vector<RigidBodyMotion<Number,3> >& localSolution,
          const Entity& element,
          const Dune::FieldVector<double,1>& pos) const
{
    if (!element.isLeaf())
        DUNE_THROW(Dune::NotImplemented, "Only for leaf elements");

    assert(localSolution.size() == 2);

    // Extract local solution on this element
    Dune::P1LocalFiniteElement<double,double,1> localFiniteElement;

    const auto jit = element.geometry().jacobianInverseTransposed(pos);
    using LocalInterpolationRule = LocalGeodesicFEFunction<1, typename GridView::ctype,
                                                           decltype(localFiniteElement),
                                                           RigidBodyMotion<Number,3> >;
    LocalInterpolationRule localInterpolationRule(localFiniteElement,localSolution);

    auto value = localInterpolationRule.evaluate(pos);

    auto referenceDerivative = localInterpolationRule.evaluateDerivative(pos);
#if DUNE_VERSION_GTE(DUNE_COMMON, 2, 8)
    auto derivative = referenceDerivative * transpose(jit);
#else
    auto derivative = referenceDerivative;
    derivative *= jit[0][0];
#endif

    Dune::FieldVector<Number,3> r_s = {derivative[0], derivative[1], derivative[2]};

    // Transformation from the reference element
    Quaternion<Number> q_s(derivative[3],
                           derivative[4],
                           derivative[5],
                           derivative[6]);

    // /////////////////////////////////////////////
    //   Sum it all up
    // /////////////////////////////////////////////

    // Strain defined on each element
    // Part I: the shearing and stretching strain
    Dune::FieldVector<Number, 6> strain(0);
    strain[0] = r_s * value.q.director(0);    // shear strain
    strain[1] = r_s * value.q.director(1);    // shear strain
    strain[2] = r_s * value.q.director(2);    // stretching strain

    // Part II: the Darboux vector

    Dune::FieldVector<Number,3> u = darboux(value.q, q_s);
    strain[3] = u[0];
    strain[4] = u[1];
    strain[5] = u[2];

    return strain;
}

template <class GridView, class RT>
template <class Number>
Dune::FieldVector<Number, 6> RodLocalStiffness<GridView, RT>::
getStress(const std::vector<RigidBodyMotion<Number,3> >& localSolution,
              const Entity& element,
                        const Dune::FieldVector<double, 1>& pos) const
{
    const auto& indexSet = gridView_.indexSet();
    std::vector<TargetSpace> localRefConf = {referenceConfiguration_[indexSet.subIndex(element, 0, 1)],
                                             referenceConfiguration_[indexSet.subIndex(element, 1, 1)]};

    auto&& strain = getStrain(localSolution, element, pos);
    auto&& referenceStrain = getStrain(localRefConf, element, pos);

    Dune::FieldVector<RT, 6> stress;
    for (int i=0; i < dim; i++)
        stress[i] = (strain[i] - referenceStrain[i]) * A_[i];

    for (int i=0; i < dim; i++)
        stress[i+3] = (strain[i+3] - referenceStrain[i+3]) * K_[i];
    return stress;
}

template <class GridView, class RT>
void RodLocalStiffness<GridView, RT>::
getStrain(const std::vector<RigidBodyMotion<double,3> >& sol,
          Dune::BlockVector<Dune::FieldVector<double, blocksize> >& strain) const
{
    const typename GridView::Traits::IndexSet& indexSet = this->basis_.gridView().indexSet();

    if (sol.size()!=this->basis_.size())
        DUNE_THROW(Dune::Exception, "Solution vector doesn't match the grid!");

    // Strain defined on each element
    strain.resize(indexSet.size(0));
    strain = 0;

    // Loop over all elements
    for (const auto& element : elements(this->basis_.gridView()))
    {
        int elementIdx = indexSet.index(element);

        // Extract local solution on this element
        Dune::P1LocalFiniteElement<double,double,1> localFiniteElement;
        int numOfBaseFct = localFiniteElement.localCoefficients().size();

        std::vector<RigidBodyMotion<double,3> > localSolution(2);

        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = sol[indexSet.subIndex(element,i,1)];

        // Get quadrature rule
        const int polOrd = 2;
        const auto& quad = Dune::QuadratureRules<double, 1>::rule(element.type(), polOrd);

        for (std::size_t pt=0; pt<quad.size(); pt++)
        {
            // Local position of the quadrature point
            const auto quadPos = quad[pt].position();

            double weight = quad[pt].weight() * element.geometry().integrationElement(quadPos);

            auto localStrain = std::dynamic_pointer_cast<RodLocalStiffness<GridView, double> >(this->localStiffness_)->getStrain(localSolution, element, quad[pt].position());

            // Sum it all up
            strain[elementIdx].axpy(weight, localStrain);
        }

        // /////////////////////////////////////////////////////////////////////////
        //   We want the average strain per element.  Therefore we have to divide
        //   the integral we just computed by the element volume.
        // /////////////////////////////////////////////////////////////////////////
        // we know the element is a line, therefore the integration element is the volume
        Dune::FieldVector<double,1> dummyPos(0.5);
        strain[elementIdx] /= element.geometry().integrationElement(dummyPos);
    }
}

template <class GridView, class RT>
void RodLocalStiffness<GridView, RT>::
getStress(const std::vector<RigidBodyMotion<double,3> >& sol,
          Dune::BlockVector<Dune::FieldVector<double, blocksize> >& stress) const
{
    // Get the strain
    getStrain(sol,stress);

    // Get reference strain
    Dune::BlockVector<Dune::FieldVector<double, blocksize> > referenceStrain;
    getStrain(dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->referenceConfiguration_, referenceStrain);

    // Linear diagonal constitutive law
    for (size_t i=0; i<stress.size(); i++)
    {
        for (int j=0; j<3; j++)
        {
            stress[i][j]   = (stress[i][j]   - referenceStrain[i][j])   * dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->A_[j];
            stress[i][j+3] = (stress[i][j+3] - referenceStrain[i][j+3]) * dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->K_[j];
        }
    }
}

template <class GridView, class RT>
template <class PatchGridView>
Dune::FieldVector<double,6> RodLocalStiffness<GridView, RT>::
getResultantForce(const BoundaryPatch<PatchGridView>& boundary,
                  const std::vector<RigidBodyMotion<double,3> >& sol) const
{
    const typename GridView::Traits::IndexSet& indexSet = this->basis_.gridView().indexSet();

    if (sol.size()!=indexSet.size(1))
        DUNE_THROW(Dune::Exception, "Solution vector doesn't match the grid!");

    Dune::FieldVector<double,3> canonicalStress(0);
    Dune::FieldVector<double,3> canonicalTorque(0);

    // Loop over the given boundary
    for (auto facet : boundary)
    {
        // //////////////////////////////////////////////
        //   Compute force across this boundary face
        // //////////////////////////////////////////////

        double pos = facet.geometryInInside().corner(0);

        std::vector<RigidBodyMotion<double,3> > localSolution(2);
        localSolution[0] = sol[indexSet.subIndex(*facet.inside(),0,1)];
        localSolution[1] = sol[indexSet.subIndex(*facet.inside(),1,1)];

        std::vector<RigidBodyMotion<double,3> > localRefConf(2);
        localRefConf[0] = dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->referenceConfiguration_[indexSet.subIndex(*facet.inside(),0,1)];
        localRefConf[1] = dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->referenceConfiguration_[indexSet.subIndex(*facet.inside(),1,1)];

        auto strain          = dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->getStrain(localSolution, *facet.inside(), pos);
        auto referenceStrain = dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->getStrain(localRefConf, *facet.inside(), pos);

        Dune::FieldVector<double,3> localStress;
        for (int i=0; i<3; i++)
            localStress[i] = (strain[i] - referenceStrain[i]) * dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->A_[i];

        Dune::FieldVector<double,3> localTorque;
        for (int i=0; i<3; i++)
            localTorque[i] = (strain[i+3] - referenceStrain[i+3]) * dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->K_[i];

        // Transform stress given with respect to the basis given by the three directors to
        // the canonical basis of R^3

        Dune::FieldMatrix<double,3,3> orientationMatrix;
        sol[indexSet.subIndex(*facet.inside(),facet.indexInInside(),1)].q.matrix(orientationMatrix);

        orientationMatrix.umv(localStress, canonicalStress);

        orientationMatrix.umv(localTorque, canonicalTorque);

        // Multiply force times boundary normal to get the transmitted force
        canonicalStress *= facet.unitOuterNormal(Dune::FieldVector<double,0>(0))[0];
        canonicalTorque *= facet.unitOuterNormal(Dune::FieldVector<double,0>(0))[0];
    }

    Dune::FieldVector<double,6> result;
    for (int i=0; i<3; i++)
    {
        result[i] = canonicalStress[i];
        result[i+3] = canonicalTorque[i];
    }

    return result;
}

#endif

