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
      kappa_  = (2.0 * mu_ + 3.0 * lambda_) / 3.0;
    }

    /** \brief Assemble the energy for a single element */
    field_type energy (const Entity& e,
               const LocalFiniteElement& localFiniteElement,
               const std::vector<Dune::FieldVector<field_type,gridDim> >& localConfiguration) const;

    /** \brief Lame constants */
    double mu_, lambda_, kappa_;
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

    field_type distortionEnergy = 0, dilatationEnergy = 0;
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
        // compute C = F^TF
        /////////////////////////////////////////////////////////

        Dune::FieldMatrix<field_type,gridDim,gridDim> C(0);
        for (int i=0; i<gridDim; i++)
          for (int j=0; j<gridDim; j++)
            for (int k=0; k<gridDim; k++)
              C[i][j] += derivative[k][i] * derivative[k][j];

        //////////////////////////////////////////////////////////
        //  Eigenvalues of C = F^TF
        //////////////////////////////////////////////////////////

        Dune::FieldVector<field_type,dim> sigmaSquared;
        FMatrixHelp::eigenValues(C, sigmaSquared);

        // logarithms of the singular values of F, i.e., eigenvalues of U
        std::array<field_type, dim> lnSigma;
        for (int i = 0; i < dim; i++)
          lnSigma[i] = 0.5 * log(sigmaSquared[i]);

        field_type trLogU = 0;
        for (int i = 0; i < dim; i++)
          trLogU += lnSigma[i];

        field_type normDevLogUSquared = 0;
        for (int i = 0; i < dim; i++)
        {
          // the deviator shifts the spectrum by the trace
          field_type lnDevSigma_i = lnSigma[i] - (1.0 / double(dim)) * trLogU;
          normDevLogUSquared += lnDevSigma_i * lnDevSigma_i;
        }

        // Add the local energy density
        distortionEnergy += weight * mu_ * normDevLogUSquared;
        dilatationEnergy += weight * 0.5 * kappa_ * (trLogU * trLogU);
    }

    return distortionEnergy + dilatationEnergy;
}

}  // namespace Dune

#endif   //#ifndef DUNE_GFE_HENCKYENERGY_HH


