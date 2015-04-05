#ifndef DUNE_GFE_HENCKYENERGY_HH
#define DUNE_GFE_HENCKYENERGY_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/fmatrixev.hh>

#include <dune/geometry/quadraturerules.hh>

#include <dune/gfe/localfestiffness.hh>

namespace Dune {

template<class GridView, class LocalFiniteElement, class field_type=double>
class HenckyEnergy
    : public LocalFEStiffness<GridView,LocalFiniteElement,std::vector<Dune::FieldVector<field_type,3> > >
{
    // grid types
    typedef typename GridView::Grid::ctype DT;
    typedef typename GridView::template Codim<0>::Entity Entity;

    // some other sizes
    enum {gridDim=GridView::dimension};
    enum {dim=GridView::dimension};

public:  // for testing

    /** \brief Constructor with a set of material parameters
     * \param parameters The material parameters
     */
    HenckyEnergy(const Dune::ParameterTree& parameters)
    {
      // Lame constants
      mu_ = parameters.template get<double>("mu");
      lambda_ = parameters.template get<double>("lambda");
    }

    /** \brief Assemble the energy for a single element */
    field_type energy (const Entity& e,
               const LocalFiniteElement& localFiniteElement,
               const std::vector<Dune::FieldVector<field_type,gridDim> >& localConfiguration) const;

    /** \brief Lame constants */
    double mu_, lambda_;
};

template <class GridView, class LocalFiniteElement, class field_type>
field_type
HenckyEnergy<GridView,LocalFiniteElement,field_type>::
energy(const Entity& element,
       const LocalFiniteElement& localFiniteElement,
       const std::vector<Dune::FieldVector<field_type,gridDim> >& localConfiguration) const
{
    assert(element.type() == localFiniteElement.type());
    typedef typename GridView::template Codim<0>::Entity::Geometry Geometry;

    field_type energy = 0;

    // store gradients of shape functions and base functions
    std::vector<Dune::FieldMatrix<DT,1,gridDim> > referenceGradients(localFiniteElement.size());
    std::vector<Dune::FieldVector<DT,gridDim> > gradients(localFiniteElement.size());

    int quadOrder = (element.type().isSimplex()) ? localFiniteElement.localBasis().order()
                                                 : localFiniteElement.localBasis().order() * gridDim;

    const Dune::QuadratureRule<DT, gridDim>& quad
        = Dune::QuadratureRules<DT, gridDim>::rule(element.type(), quadOrder);

    for (size_t pt=0; pt<quad.size(); pt++) {

        // Local position of the quadrature point
        const Dune::FieldVector<DT,gridDim>& quadPos = quad[pt].position();

        const DT integrationElement = element.geometry().integrationElement(quadPos);

        const typename Geometry::JacobianInverseTransposed& jacobianInverseTransposed = element.geometry().jacobianInverseTransposed(quadPos);

        DT weight = quad[pt].weight() * integrationElement;

        // get gradients of shape functions
        localFiniteElement.localBasis().evaluateJacobian(quadPos, referenceGradients);

        // compute gradients of base functions
        for (size_t i=0; i<gradients.size(); ++i)
          jacobianInverseTransposed.mv(referenceGradients[i][0], gradients[i]);

        Dune::FieldMatrix<field_type,gridDim,gridDim> derivative(0);
        for (size_t i=0; i<gradients.size(); i++)
          for (int j=0; j<gridDim; j++)
            derivative[j].axpy(localConfiguration[i][j], gradients[i]);

        /////////////////////////////////////////////////////////
        // compute F^T F
        /////////////////////////////////////////////////////////

        Dune::FieldMatrix<field_type,gridDim,gridDim> FTF(0);
        for (int i=0; i<gridDim; i++)
          for (int j=0; j<gridDim; j++)
            for (int k=0; k<gridDim; k++)
              FTF[i][j] += derivative[k][i] * derivative[k][j];

        //////////////////////////////////////////////////////////
        //  Eigenvalues of FTF
        //////////////////////////////////////////////////////////

        Dune::FieldVector<field_type,dim> lambda;
        FMatrixHelp::eigenValues(FTF, lambda);

        // logarithms of the eigenvalues
        std::array<field_type,dim> ln;
        for (int i=0; i<dim; i++)
          ln[i] = std::log(lambda[i]);

        // Add the local energy density
        for (int i=0; i<dim; i++)
          energy += weight * mu_ * ln[i]*ln[i];

        field_type trace = 0;
        for (int i=0; i<dim; i++)
          trace += ln[i];

        energy += weight * 0.5 * lambda_ * trace * trace;

    }

    return energy;
}

}  // namespace Dune

#endif   //#ifndef DUNE_GFE_HENCKYENERGY_HH


