#ifndef DUNE_GFE_CHIRALSKYRMIONENERGY_HH
#define DUNE_GFE_CHIRALSKYRMIONENERGY_HH

#include <dune/common/fmatrix.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/gfe/assemblers/localenergy.hh>

namespace Dune {

  namespace GFE {

    /** \brief Energy of certain chiral Skyrmion
     *
     * The energy is discussed in:
     * - Christof Melcher, "Chiral skyrmions in the plane", Proc. of the Royal Society, online DOI DOI: 10.1098/rspa.2014.0394
     */
    template<class Basis, class LocalInterpolationRule, class field_type>
    class ChiralSkyrmionEnergy
      : public GFE::LocalEnergy<Basis,UnitVector<field_type,3> >
    {
      // various useful types
      typedef UnitVector<field_type,3> TargetSpace;
      typedef typename Basis::GridView GridView;
      typedef typename GridView::ctype DT;
      typedef typename TargetSpace::ctype RT;
      typedef typename GridView::template Codim<0>::Entity Entity;

      // some other sizes
      constexpr static int gridDim = GridView::dimension;

    public:

      ChiralSkyrmionEnergy(const Dune::ParameterTree& parameters)
      {
        h_     = parameters.template get<double>("h");
        kappa_ = parameters.template get<double>("kappa");
      }

      //! Dimension of a tangent space
      constexpr static int blocksize = TargetSpace::TangentVector::dimension;

      /** \brief Assemble the energy for a single element */
      RT energy (const typename Basis::LocalView& localView,
                 const std::vector<TargetSpace>& localConfiguration) const override;

      RT energy (const typename Basis::LocalView& localView,
                 const typename Impl::LocalEnergyTypes<TargetSpace>::CompositeCoefficients& coefficients) const override
      {
        DUNE_THROW(NotImplemented, "!");
      }

      field_type h_;
      field_type kappa_;
    };

    template <class Basis, class LocalInterpolationRule, class field_type>
    typename ChiralSkyrmionEnergy<Basis, LocalInterpolationRule, field_type>::RT
    ChiralSkyrmionEnergy<Basis, LocalInterpolationRule, field_type>::
    energy(const typename Basis::LocalView& localView,
           const std::vector<TargetSpace>& localConfiguration) const
    {
      typedef typename GridView::template Codim<0>::Entity::Geometry Geometry;

      RT energy = 0;

      const auto element = localView.element();
      const auto& localFiniteElement = localView.tree().finiteElement();
      LocalInterpolationRule localInterpolationRule(localFiniteElement,localConfiguration);

      int quadOrder = (element.type().isSimplex()) ? (localFiniteElement.localBasis().order()-1) * 2
                                               : localFiniteElement.localBasis().order() * 2 * gridDim;



      const Dune::QuadratureRule<double, gridDim>& quad
        = Dune::QuadratureRules<double, gridDim>::rule(element.type(), quadOrder);

      for (size_t pt=0; pt<quad.size(); pt++) {

        // Local position of the quadrature point
        const Dune::FieldVector<double,gridDim>& quadPos = quad[pt].position();

        const double integrationElement = element.geometry().integrationElement(quadPos);

        const typename Geometry::JacobianInverseTransposed& jacobianInverseTransposed = element.geometry().jacobianInverseTransposed(quadPos);

        double weight = quad[pt].weight() * integrationElement;

        // The value of the function
        auto value = localInterpolationRule.evaluate(quadPos);

        // The derivative of the local function defined on the reference element
        typename LocalInterpolationRule::DerivativeType referenceDerivative = localInterpolationRule.evaluateDerivative(quadPos,value);

        // The derivative of the function defined on the actual element
        typename LocalInterpolationRule::DerivativeType derivative(0);

        for (size_t comp=0; comp<referenceDerivative.N(); comp++)
          jacobianInverseTransposed.umv(referenceDerivative[comp], derivative[comp]);

        //////////////////////////////////////////////////////////////
        //  Exchange energy (aka harmonic energy)
        //////////////////////////////////////////////////////////////
        // The Frobenius norm is the correct norm here if the metric of TargetSpace is the identity.
        // (And if the metric of the domain space is the identity, which it always is here.)
        energy += weight * 0.5 * derivative.frobenius_norm2();

        //////////////////////////////////////////////////////////////
        //  Dzyaloshinskii-Moriya interaction term
        //////////////////////////////////////////////////////////////

        // derivative[a][b] contains the partial derivative of m_a in the direction x_b
        Dune::FieldVector<field_type, 3> curl = {derivative[2][1], -derivative[2][0], derivative[1][0]-derivative[0][1]};

        FieldVector<field_type, 3> v = value.globalCoordinates();

        energy += weight * kappa_ * (v * curl);

        //////////////////////////////////////////////////////////////
        //  Zeeman interaction term
        //////////////////////////////////////////////////////////////
        v[2] -= 1; // subtract e_3
        energy += weight * 0.5 * h_ * v.two_norm2();

      }

      return energy;
    }

  } // namespace GFE

}  // namespace Dune
#endif
