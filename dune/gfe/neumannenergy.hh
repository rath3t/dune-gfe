#ifndef DUNE_GFE_NEUMANNENERGY_HH
#define DUNE_GFE_NEUMANNENERGY_HH

#include <dune/geometry/quadraturerules.hh>

#include <dune/fufem/boundarypatch.hh>

#include <dune/gfe/assemblers/localenergy.hh>
#include <dune/gfe/spaces/productmanifold.hh>

namespace Dune::GFE {
  /** \brief Integrate a density over a part of the domain boundary
   *
   * This is typically used to implement Neumann boundary conditions for Cosserat materials.
   * Essentially it works only for that: It is implicitly assumed that the target space
   * is a product, and that the first factor is a RealTuple.
   */
  template<class Basis, class ... TargetSpaces>
  class NeumannEnergy
    : public Dune::GFE::LocalEnergy<Basis,ProductManifold<TargetSpaces...> >
  {
    using TargetSpace = ProductManifold<TargetSpaces...>;

    using LocalView = typename Basis::LocalView;
    using GridView = typename LocalView::GridView;
    using DT = typename GridView::Grid::ctype;
    using RT = typename Dune::GFE::LocalEnergy<Basis,TargetSpace>::RT;

    constexpr static int dim = GridView::dimension;

    // TODO: Remove the hard-coded first factor space!
    using WorldVector = typename std::tuple_element_t<0, std::tuple<TargetSpaces...> >::EmbeddedTangentVector;

  public:

    /** \brief Constructor with a set of material parameters
     * \param parameters The material parameters
     */
    NeumannEnergy(const std::shared_ptr<BoundaryPatch<GridView> >& neumannBoundary,
                  std::function<WorldVector(Dune::FieldVector<DT,dim>)> neumannFunction)
      : neumannBoundary_(neumannBoundary),
      neumannFunction_(neumannFunction)
    {}

    /** \brief Assemble the energy for a single element */
    RT energy(const typename Basis::LocalView& localView,
              const std::vector<TargetSpace>& localSolutions) const override
    {
      DUNE_THROW(NotImplemented, "!");
    }

    RT energy (const typename Basis::LocalView& localView,
               const typename Impl::LocalEnergyTypes<TargetSpace>::CompositeCoefficients& coefficients) const override
    {
      using namespace Dune::Indices;
      // TODO: Remove the hard-coded first factor space!
      using TargetSpace = typename std::tuple_element<0, std::tuple<TargetSpaces...> >::type;
      const std::vector<TargetSpace>& localSolution = coefficients[_0];

      const auto& localFiniteElement = localView.tree().child(_0,0).finiteElement();
      const auto& element = localView.element();

      RT energy = 0;

      for (auto&& intersection : intersections(neumannBoundary_->gridView(), element)) {

        if (not neumannBoundary_ or not neumannBoundary_->contains(intersection))
          continue;

        int quadOrder = (element.type().isSimplex()) ? localFiniteElement.localBasis().order()
                                                    : localFiniteElement.localBasis().order() * dim;

        const auto& quad = Dune::QuadratureRules<DT, dim-1>::rule(intersection.type(), quadOrder);

        for (size_t pt=0; pt<quad.size(); pt++) {

          // Local position of the quadrature point
          const Dune::FieldVector<DT,dim>& quadPos = intersection.geometryInInside().global(quad[pt].position());

          const auto integrationElement = intersection.geometry().integrationElement(quad[pt].position());

          // The value of the local function
          std::vector<Dune::FieldVector<DT,1> > shapeFunctionValues;
          localFiniteElement.localBasis().evaluateFunction(quadPos, shapeFunctionValues);

          WorldVector value(0);
          for (size_t i=0; i<localFiniteElement.size(); i++)
            for (int j=0; j<WorldVector::dimension; j++)
              value[j] += shapeFunctionValues[i] * localSolution[i][j];

          // Value of the Neumann data at the current position
          auto neumannValue = neumannFunction_( intersection.geometry().global(quad[pt].position()) );

          // Only translational dofs are affected by the Neumann force
          for (size_t i=0; i<neumannValue.size(); i++)
            energy += (neumannValue[i] * value[i]) * quad[pt].weight() * integrationElement;

        }

      }

      return energy;
    }

  private:
    /** \brief The Neumann boundary */
    const std::shared_ptr<BoundaryPatch<GridView> > neumannBoundary_;

    /** \brief The function implementing the Neumann data */
    std::function<WorldVector(Dune::FieldVector<DT,dim>)> neumannFunction_;
  };
}  // namespace Dune::GFE

#endif   //#ifndef DUNE_GFE_NEUMANNENERGY_HH
