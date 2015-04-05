#ifndef DUNE_GFE_NEOHOOKEENERGY_HH
#define DUNE_GFE_NEOHOOKEENERGY_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/fmatrixev.hh>

#include <dune/geometry/quadraturerules.hh>

#include <dune/gfe/localfestiffness.hh>

namespace Dune {

template<class GridView, class LocalFiniteElement, class field_type=double>
class NeoHookeEnergy
  : public LocalFEStiffness<GridView,
                            LocalFiniteElement,
                            std::vector<Dune::FieldVector<field_type,3> > >
{
  // grid types
  typedef typename GridView::Grid::ctype DT;
  typedef typename GridView::template Codim<0>::Entity Entity;

  // some other sizes
  enum {gridDim=GridView::dimension};
  enum {dim=GridView::dimension};

public:

  /** \brief Constructor with a set of material parameters
   * \param parameters The material parameters
   */
  NeoHookeEnergy(const Dune::ParameterTree& parameters)
  {
    // Lame constants
    mu_     = parameters.get<double>("mu");
    lambda_ = parameters.get<double>("lambda");
    kappa_  = (2.0 * mu_ + 3.0 * lambda_) / 3.0;
  }

  /** \brief Assemble the energy for a single element */
  field_type energy (const Entity& e,
                     const LocalFiniteElement& localFiniteElement,
                     const std::vector<Dune::FieldVector<field_type, gridDim> >& localConfiguration) const;

  /** \brief Lame constants */
  field_type mu_, lambda_, kappa_;
};

template <class GridView, class LocalFiniteElement, class field_type>
field_type
NeoHookeEnergy<GridView, LocalFiniteElement, field_type>::
energy(const Entity& element,
       const LocalFiniteElement& localFiniteElement,
       const std::vector<Dune::FieldVector<field_type, gridDim> >& localConfiguration) const
{
  assert(element.type() == localFiniteElement.type());
  typedef typename GridView::template Codim<0>::Entity::Geometry Geometry;

  field_type distortionEnergy = 0, dilatationEnergy = 0;

  // store gradients of shape functions and base functions
  std::vector<Dune::FieldMatrix<DT,1,gridDim> > referenceGradients(localFiniteElement.size());
  std::vector<Dune::FieldVector<DT,gridDim> > gradients(localFiniteElement.size());

  int quadOrder = (element.type().isSimplex()) ? localFiniteElement.localBasis().order()
                                               : localFiniteElement.localBasis().order() * gridDim;

  const Dune::QuadratureRule<DT, gridDim>& quad
      = Dune::QuadratureRules<DT, gridDim>::rule(element.type(), quadOrder);

  for (size_t pt=0; pt<quad.size(); pt++)
  {
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
    // compute C = F^T F
    /////////////////////////////////////////////////////////

    Dune::FieldMatrix<field_type,gridDim,gridDim> C(0);
    for (int i=0; i<gridDim; i++)
      for (int j=0; j<gridDim; j++)
        for (int k=0; k<gridDim; k++)
          C[i][j] += derivative[k][i] * derivative[k][j];

    //////////////////////////////////////////////////////////
    //  Eigenvalues of FTF
    //////////////////////////////////////////////////////////

    // eigenvalues of C, i.e., squared singular values \sigma_i of F
    Dune::FieldVector<field_type, dim> sigmaSquared;
    FMatrixHelp::eigenValues(C, sigmaSquared);

    // singular values of F, i.e., eigenvalues of U
    std::array<field_type, dim> sigma;
    for (int i = 0; i < dim; i++)
      sigma[i] = std::sqrt(sigmaSquared[i]);

    field_type detC = 1.0;
    for (int i = 0; i < dim; i++)
      detC *= sigmaSquared[i];
    field_type detF = std::sqrt(detC);

    // \tilde{C} = \tilde{F}^T\tilde{F} = \frac{1}{\det{F}^{2/3}}C
    field_type trCTilde = 0;
    for (int i = 0; i < dim; i++)
      trCTilde += sigmaSquared[i];
    trCTilde /= pow(detC, 1.0/3.0);

    // Add the local energy density
    distortionEnergy += weight * (0.5 * mu_)    * (trCTilde - 3);
    dilatationEnergy += weight * (0.5 * kappa_) * ((detF - 1) * (detF - 1));
  }

  return distortionEnergy + dilatationEnergy;
}

}  // namespace Dune

#endif   //#ifndef DUNE_GFE_NEOHOOKEENERGY_HH


