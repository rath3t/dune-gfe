#ifndef DUNE_GFE_NEUMANNENERGY_HH
#define DUNE_GFE_NEUMANNENERGY_HH

#include <dune/geometry/quadraturerules.hh>

#include <dune/fufem/boundarypatch.hh>

#include <dune/elasticity/assemblers/localenergy.hh>

namespace Dune::GFE {
  /**
     \brief Assembles the Neumann energy for a single element on the Neumann Boundary using the Neumann Function.

           This class works similarly to the class Dune::Elasticity::NeumannEnergy, where Dune::Elasticity::NeumannEnergy extends
           Dune::Elasticity::LocalEnergy and Dune::GFE::NeumannEnergy extends Dune::GFE::LocalEnergy.
   */
  template<class Basis, class ... TargetSpaces>
  class NeumannEnergy
    : public Dune::GFE::LocalEnergy<Basis,TargetSpaces...>
  {
    using LocalView = typename Basis::LocalView;
    using GridView = typename LocalView::GridView;
    using DT = typename GridView::Grid::ctype;
    using RT = typename Dune::GFE::LocalEnergy<Basis,TargetSpaces...>::RT;

    constexpr static int dim = GridView::dimension;

  public:

    /** \brief Constructor with a set of material parameters
     * \param parameters The material parameters
     */
    NeumannEnergy(const std::shared_ptr<BoundaryPatch<GridView> >& neumannBoundary,
                  std::function<Dune::FieldVector<double,dim>(Dune::FieldVector<DT,dim>)> neumannFunction)
      : neumannBoundary_(neumannBoundary),
      neumannFunction_(neumannFunction)
    {}

    /** \brief Assemble the energy for a single element */
    RT energy(const typename Basis::LocalView& localView,
              const std::vector<TargetSpaces>& ... localSolutions) const
    {
      static_assert(sizeof...(TargetSpaces) > 0, "NeumannEnergy needs at least one TargetSpace!");

      using namespace Dune::Indices;
      using TargetSpace = typename std::tuple_element<0, std::tuple<TargetSpaces...> >::type;
      const std::vector<TargetSpace>& localSolution = std::get<0>(std::forward_as_tuple(localSolutions ...));

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

          Dune::FieldVector<RT,dim> value(0);
          for (size_t i=0; i<localFiniteElement.size(); i++)
            for (int j=0; j<dim; j++)
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
    std::function<Dune::FieldVector<double,dim>(Dune::FieldVector<DT,dim>)> neumannFunction_;
  };
}  // namespace Dune::GFE

#endif   //#ifndef DUNE_GFE_NEUMANNENERGY_HH
