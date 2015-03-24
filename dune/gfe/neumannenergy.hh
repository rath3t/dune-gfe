#ifndef DUNE_GFE_NEUMANNENERGY_HH
#define DUNE_GFE_NEUMANNENERGY_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/fmatrixev.hh>

#include <dune/geometry/quadraturerules.hh>

#include <dune/fufem/functions/virtualgridfunction.hh>
#include <dune/fufem/boundarypatch.hh>

#include <dune/gfe/localgeodesicfestiffness.hh>
#include <dune/gfe/localfestiffness.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/realtuple.hh>

namespace Dune {

template<class GridView, class LocalFiniteElement, class field_type=double>
class NeumannEnergy
: public LocalFEStiffness<GridView,LocalFiniteElement,std::vector<Dune::FieldVector<field_type,3> > >
{
 // grid types
  typedef typename GridView::ctype ctype;
  typedef typename GridView::template Codim<0>::Entity Entity;

  enum {dim=GridView::dimension};

public:

  /** \brief Constructor with a set of material parameters
   * \param parameters The material parameters
   */
  NeumannEnergy(const BoundaryPatch<GridView>* neumannBoundary,
                const Dune::VirtualFunction<Dune::FieldVector<ctype,dim>, Dune::FieldVector<double,3> >* neumannFunction)
  : neumannBoundary_(neumannBoundary),
    neumannFunction_(neumannFunction)
  {}

  /** \brief Assemble the energy for a single element */
  field_type energy (const Entity& element,
                     const LocalFiniteElement& localFiniteElement,
                     const std::vector<Dune::FieldVector<field_type,dim> >& localConfiguration) const
  {
    assert(element.type() == localFiniteElement.type());

    field_type energy = 0;

    for (auto&& it : intersections(neumannBoundary_->gridView(), element)) {

      if (not neumannBoundary_ or not neumannBoundary_->contains(it))
        continue;

      int quadOrder = localFiniteElement.localBasis().order();

      const Dune::QuadratureRule<ctype, dim-1>& quad
          = Dune::QuadratureRules<ctype, dim-1>::rule(it.type(), quadOrder);

      for (size_t pt=0; pt<quad.size(); pt++) {

        // Local position of the quadrature point
        const Dune::FieldVector<ctype,dim>& quadPos = it.geometryInInside().global(quad[pt].position());

        const auto integrationElement = it.geometry().integrationElement(quad[pt].position());

        // The value of the local function
        std::vector<Dune::FieldVector<ctype,1> > shapeFunctionValues;
        localFiniteElement.localBasis().evaluateFunction(quadPos, shapeFunctionValues);

        Dune::FieldVector<field_type,dim> value(0);
        for (int i=0; i<localFiniteElement.size(); i++)
          for (int j=0; j<dim; j++)
            value[j] += shapeFunctionValues[i] * localConfiguration[i][j];

        // Value of the Neumann data at the current position
        Dune::FieldVector<double,3> neumannValue;

        if (dynamic_cast<const VirtualGridViewFunction<GridView,Dune::FieldVector<double,3> >*>(neumannFunction_))
            dynamic_cast<const VirtualGridViewFunction<GridView,Dune::FieldVector<double,3> >*>(neumannFunction_)->evaluateLocal(element, quadPos, neumannValue);
        else
            neumannFunction_->evaluate(it.geometry().global(quad[pt].position()), neumannValue);

        // Only translational dofs are affected by the Neumann force
        for (size_t i=0; i<neumannValue.size(); i++)
          energy += (neumannValue[i] * value[i]) * quad[pt].weight() * integrationElement;

      }

    }

    return energy;
  }

private:
  /** \brief The Neumann boundary */
  const BoundaryPatch<GridView>* neumannBoundary_;

  /** \brief The function implementing the Neumann data */
  const Dune::VirtualFunction<Dune::FieldVector<double,dim>, Dune::FieldVector<double,3> >* neumannFunction_;
};

}

#endif   //#ifndef DUNE_GFE_NEUMANNENERGY_HH


