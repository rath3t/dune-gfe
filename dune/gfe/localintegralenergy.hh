#ifndef DUNE_GFE_LOCALINTEGRALENERGY_HH
#define DUNE_GFE_LOCALINTEGRALENERGY_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/fmatrixev.hh>

#include <dune/geometry/quadraturerules.hh>

#include <dune/gfe/localenergy.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rigidbodymotion.hh>

#include <dune/elasticity/materials/localdensity.hh>

namespace Dune::GFE {

  /**
    \brief Assembles the elastic energy for a single element integrating the localdensity over one element.

           This class works similarly to the class Dune::Elasticity::LocalIntegralEnergy, where Dune::Elasticity::LocalIntegralEnergy extends
           Dune::Elasticity::LocalEnergy and Dune::GFE::LocalIntegralEnergy extends Dune::GFE::LocalEnergy.
  */
template<class Basis, class... TargetSpaces>
class LocalIntegralEnergy
  : public Dune::GFE::LocalEnergy<Basis,TargetSpaces...>
{
  using LocalView = typename Basis::LocalView;
  using GridView = typename LocalView::GridView;
  using DT = typename GridView::Grid::ctype;
  using RT = typename Dune::GFE::LocalEnergy<Basis,TargetSpaces...>::RT;

  constexpr static int gridDim = GridView::dimension;

public:

  /** \brief Constructor with a local energy density
    */
  LocalIntegralEnergy(const std::shared_ptr<Dune::Elasticity::LocalDensity<gridDim,RT,DT>>& ld)
  : localDensity_(ld)
  {}

  /** \brief Assemble the energy for a single element */
  RT energy(const typename Basis::LocalView& localView,
            const std::vector<TargetSpaces>&... localSolutions) const
  { 
    static_assert(sizeof...(TargetSpaces) > 0, "LocalIntegralEnergy needs at least one TargetSpace!");

    using TargetSpace = typename std::tuple_element<0, std::tuple<TargetSpaces...> >::type;
    static_assert( (std::is_same<TargetSpace, RigidBodyMotion<RT,gridDim>>::value)
      or (std::is_same<TargetSpace, RealTuple<RT,gridDim>>::value), "The first TargetSpace of LocalIntegralEnergy needs to be RigidBodyMotion or RealTuple!" );
    
    const std::vector<TargetSpace>& localSolution = std::get<0>(std::forward_as_tuple(localSolutions...));

    using namespace Dune::Indices;
    // powerBasis: grab the finite element of the first child
    const auto& localFiniteElement = localView.tree().child(_0,0).finiteElement();
    const auto& element = localView.element();

    RT energy = 0;

    // store gradients of shape functions and base functions
    std::vector<FieldMatrix<DT,1,gridDim> > referenceGradients(localFiniteElement.size());
    std::vector<FieldVector<DT,gridDim> > gradients(localFiniteElement.size());

    int quadOrder = (element.type().isSimplex()) ? localFiniteElement.localBasis().order()
                                                 : localFiniteElement.localBasis().order() * gridDim;

    const auto& quad = Dune::QuadratureRules<DT, gridDim>::rule(element.type(), quadOrder);

    for (const auto& qp : quad)
    {
      const DT integrationElement = element.geometry().integrationElement(qp.position());

      const auto jacobianInverseTransposed = element.geometry().jacobianInverseTransposed(qp.position());

      // Global position
      auto x = element.geometry().global(qp.position());

      // Get gradients of shape functions
      localFiniteElement.localBasis().evaluateJacobian(qp.position(), referenceGradients);

      // compute gradients of base functions
      for (size_t i=0; i<gradients.size(); ++i)
        jacobianInverseTransposed.mv(referenceGradients[i][0], gradients[i]);

      // Deformation gradient
      FieldMatrix<RT,gridDim,gridDim> deformationGradient(0);
      for (size_t i=0; i<gradients.size(); i++)
        for (int j=0; j<gridDim; j++)
          deformationGradient[j].axpy(localSolution[i][j], gradients[i]);
      // Integrate the energy density
      energy += qp.weight() * integrationElement * (*localDensity_)(x, deformationGradient);
    }

    return energy;
  }

protected:
  const std::shared_ptr<Dune::Elasticity::LocalDensity<gridDim,RT,DT>> localDensity_ = nullptr;

};

}  // namespace Dune::GFE

#endif   //#ifndef DUNE_GFE_LOCALINTEGRALENERGY_HH
