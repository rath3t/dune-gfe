#ifndef DUNE_GFE_NONPLANARCOSSERATSHELLENERGY_HH
#define DUNE_GFE_NONPLANARCOSSERATSHELLENERGY_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/parametertree.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/matrix-vector/crossproduct.hh>

#include <dune/fufem/boundarypatch.hh>

#include <dune/gfe/localenergy.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/unitvector.hh>
#include <dune/gfe/tensor3.hh>
#include <dune/gfe/localprojectedfefunction.hh>


template<class Basis, int dim, class field_type=double>
class NonplanarCosseratShellEnergy
  : public Dune::GFE::LocalEnergy<Basis,RigidBodyMotion<field_type,dim> >
{
  // grid types
  typedef typename Basis::GridView GridView;
  typedef typename Basis::LocalView::Tree::FiniteElement LocalFiniteElement;
  typedef typename GridView::ctype DT;
  typedef RigidBodyMotion<field_type,dim> TargetSpace;
  typedef typename TargetSpace::ctype RT;
  typedef typename GridView::template Codim<0>::Entity Entity;

  // some other sizes
  enum {gridDim=GridView::dimension};
  enum {dimworld=GridView::dimensionworld};

public:

  /** \brief Constructor with a set of material parameters
   * \param parameters The material parameters
   */
  NonplanarCosseratShellEnergy(const Dune::ParameterTree& parameters,
                               const std::vector<UnitVector<double,3> >& vertexNormals,
                               const BoundaryPatch<GridView>* neumannBoundary,
                               const std::function<Dune::FieldVector<double,3>(Dune::FieldVector<double,dimworld>)> neumannFunction,
                               const std::function<Dune::FieldVector<double,3>(Dune::FieldVector<double,dimworld>)> volumeLoad)
  : vertexNormals_(vertexNormals),
    neumannBoundary_(neumannBoundary),
    neumannFunction_(neumannFunction),
    volumeLoad_(volumeLoad)
  {
    // The shell thickness
    thickness_ = parameters.template get<double>("thickness");

    // Lame constants
    mu_ = parameters.template get<double>("mu");
    lambda_ = parameters.template get<double>("lambda");

    // Cosserat couple modulus
    mu_c_ = parameters.template get<double>("mu_c");

    // Length scale parameter
    L_c_ = parameters.template get<double>("L_c");

    // Curvature parameters
    b1_ = parameters.template get<double>("b1");
    b2_ = parameters.template get<double>("b2");
    b3_ = parameters.template get<double>("b3");
  }

  /** \brief Assemble the energy for a single element */
  RT energy (const typename Basis::LocalView& localView,
             const std::vector<TargetSpace>& localSolution) const;

  RT W_m(const Dune::FieldMatrix<field_type,3,3>& S) const
  {
    return W_mixt(S,S);
  }

  RT W_mixt(const Dune::FieldMatrix<field_type,3,3>& S, const Dune::FieldMatrix<field_type,3,3>& T) const
  {
    return mu_ * Dune::GFE::frobeniusProduct(Dune::GFE::sym(S), Dune::GFE::sym(T))
         + mu_c_ * Dune::GFE::frobeniusProduct(Dune::GFE::skew(S), Dune::GFE::skew(T))
         + lambda_ * mu_ / (lambda_ + 2*mu_) * Dune::GFE::trace(S) * Dune::GFE::trace(T);
  }

  RT W_mp(const Dune::FieldMatrix<field_type,3,3>& S) const
  {
    return mu_ * Dune::GFE::sym(S).frobenius_norm2() + mu_c_ * Dune::GFE::skew(S).frobenius_norm2() + lambda_ * 0.5 * Dune::GFE::traceSquared(S);
  }

  RT W_curv(const Dune::FieldMatrix<field_type,3,3>& S) const
  {
    return mu_ * L_c_ * L_c_ * (b1_ * Dune::GFE::dev(Dune::GFE::sym(S)).frobenius_norm2()
         + b2_ * Dune::GFE::skew(S).frobenius_norm2() + b3_ * Dune::GFE::traceSquared(S));
  }

  /** \brief The shell thickness */
  double thickness_;

  /** \brief Lame constants */
  double mu_, lambda_;

  /** \brief Cosserat couple modulus */
  double mu_c_;

  /** \brief Length scale parameter */
  double L_c_;

  /** \brief Curvature parameters */
  double b1_, b2_, b3_;

  /** \brief The normal vectors at the grid vertices.  This are used to compute the reference surface curvature. */
  std::vector<UnitVector<double,3> > vertexNormals_;

  /** \brief The Neumann boundary */
  const BoundaryPatch<GridView>* neumannBoundary_;

  /** \brief The function implementing the Neumann data */
  const std::function<Dune::FieldVector<double,3>(Dune::FieldVector<double,dimworld>)> neumannFunction_;

  /** \brief The function implementing a volume load */
  const std::function<Dune::FieldVector<double,3>(Dune::FieldVector<double,dimworld>)> volumeLoad_;
};

template <class Basis, int dim, class field_type>
typename NonplanarCosseratShellEnergy<Basis,dim,field_type>::RT
NonplanarCosseratShellEnergy<Basis,dim,field_type>::
energy(const typename Basis::LocalView& localView,
       const std::vector<RigidBodyMotion<field_type,dim> >& localSolution) const
{
  // The element geometry
  auto element = localView.element();
  auto geometry = element.geometry();

  // The set of shape functions on this element
  const auto& localFiniteElement = localView.tree().finiteElement();

  ////////////////////////////////////////////////////////////////////////////////////
  //  Construct a linear (i.e., non-constant!) normal field on each element
  ////////////////////////////////////////////////////////////////////////////////////
  auto gridView = localView.globalBasis().gridView();

  assert(vertexNormals_.size() == gridView.indexSet().size(gridDim));
  std::vector<UnitVector<double,3> > cornerNormals(element.subEntities(gridDim));
  for (size_t i=0; i<cornerNormals.size(); i++)
    cornerNormals[i] = vertexNormals_[gridView.indexSet().subIndex(element,i,2)];

  typedef typename Dune::PQkLocalFiniteElementCache<DT, double, gridDim, 1> P1FiniteElementCache;
  typedef typename P1FiniteElementCache::FiniteElementType P1LocalFiniteElement;
  P1FiniteElementCache p1FiniteElementCache;
  const auto& p1LocalFiniteElement = p1FiniteElementCache.get(element.type());
  Dune::GFE::LocalProjectedFEFunction<gridDim, DT, P1LocalFiniteElement, UnitVector<double,3> > unitNormals(p1LocalFiniteElement, cornerNormals);

  ////////////////////////////////////////////////////////////////////////////////////
  //  Set up the local nonlinear finite element function
  ////////////////////////////////////////////////////////////////////////////////////
  typedef LocalGeodesicFEFunction<gridDim, DT, LocalFiniteElement, TargetSpace> LocalGFEFunctionType;
  LocalGFEFunctionType localGeodesicFEFunction(localFiniteElement,localSolution);

  RT energy = 0;

  auto quadOrder = (element.type().isSimplex()) ? localFiniteElement.localBasis().order()
                                                : localFiniteElement.localBasis().order() * gridDim;

  const auto& quad = Dune::QuadratureRules<DT, gridDim>::rule(element.type(), quadOrder);

  for (size_t pt=0; pt<quad.size(); pt++)
  {
    // Local position of the quadrature point
    const Dune::FieldVector<DT,gridDim>& quadPos = quad[pt].position();

    const DT integrationElement = geometry.integrationElement(quadPos);

    // The value of the local function
    RigidBodyMotion<field_type,dim> value = localGeodesicFEFunction.evaluate(quadPos);

    // The derivative of the local function
    auto derivative = localGeodesicFEFunction.evaluateDerivative(quadPos,value);

    //////////////////////////////////////////////////////////
    //  The rotation and its derivative
    //  Note: we need it in matrix coordinates
    //////////////////////////////////////////////////////////

    Dune::FieldMatrix<field_type,dim,dim> R;
    value.q.matrix(R);
    auto RT = Dune::GFE::transpose(R);

    Tensor3<field_type,3,3,gridDim> DR = value.quaternionTangentToMatrixTangent(derivative);

    //////////////////////////////////////////////////////////
    //  Fundamental forms and curvature
    //////////////////////////////////////////////////////////

    // First fundamental form
    Dune::FieldMatrix<double,3,3> aCovariant;

    // If dimworld==3, then the first two lines of aCovariant are simply the jacobianTransposed
    // of the element.  If dimworld<3 (i.e., ==2), we have to explicitly enters 0.0 in the last column.
    auto jacobianTransposed = geometry.jacobianTransposed(quadPos);

    for (int i=0; i<2; i++)
    {
      for (int j=0; j<dimworld; j++)
        aCovariant[i][j] = jacobianTransposed[i][j];
      for (int j=dimworld; j<3; j++)
        aCovariant[i][j] = 0.0;
    }

    aCovariant[2] = Dune::MatrixVector::crossProduct(aCovariant[0], aCovariant[1]);
    aCovariant[2] /= aCovariant[2].two_norm();

    auto aContravariant = aCovariant;
    aContravariant.invert();

    Dune::FieldMatrix<double,3,3> a(0);
    for (int alpha=0; alpha<gridDim; alpha++)
      a += Dune::GFE::dyadicProduct(aCovariant[alpha], aContravariant[alpha]);

    auto a00 = aCovariant[0] * aCovariant[0];
    auto a01 = aCovariant[0] * aCovariant[1];
    auto a10 = aCovariant[1] * aCovariant[0];
    auto a11 = aCovariant[1] * aCovariant[1];
    auto aScalar = std::sqrt(a00*a11 - a10*a01);

    // Alternator tensor
    Dune::FieldMatrix<int,2,2> eps = {{0,1},{-1,0}};
    Dune::FieldMatrix<double,3,3> c(0);

    for (int alpha=0; alpha<2; alpha++)
      for (int beta=0; beta<2; beta++)
        c += aScalar * eps[alpha][beta] * Dune::GFE::dyadicProduct(aContravariant[alpha], aContravariant[beta]);

    // Second fundamental form
    // The derivative of the normal field
    auto normalDerivative = unitNormals.evaluateDerivative(quadPos);

    Dune::FieldMatrix<double,3,3> b(0);
    for (int alpha=0; alpha<gridDim; alpha++)
    {
      Dune::FieldVector<double,3> vec;
      for (int i=0; i<3; i++)
        vec[i] = normalDerivative[i][alpha];
      b -= Dune::GFE::dyadicProduct(vec, aContravariant[alpha]);
    }

    // Gauss curvature
    auto K = b.determinant();

    // Mean curvatue
    auto H = 0.5 * Dune::GFE::trace(b);

    //////////////////////////////////////////////////////////
    //  Strain tensors
    //////////////////////////////////////////////////////////

    // Elastic shell strain
    Dune::FieldMatrix<field_type,3,3> Ee(0);
    Dune::FieldMatrix<field_type,3,3> grad_s_m(0);
    for (int alpha=0; alpha<gridDim; alpha++)
    {
      Dune::FieldVector<field_type,3> vec;
      for (int i=0; i<3; i++)
        vec[i] = derivative[i][alpha];
      grad_s_m += Dune::GFE::dyadicProduct(vec, aContravariant[alpha]);
    }

    Ee = RT * grad_s_m - a;

    // Elastic shell bending-curvature strain
    Dune::FieldMatrix<field_type,3,3> Ke(0);
    for (int alpha=0; alpha<gridDim; alpha++)
    {
      Dune::FieldMatrix<field_type,3,3> tmp;
      for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
          tmp[i][j] = DR[i][j][alpha];
      auto tmp2 = RT * tmp;
      Ke += Dune::GFE::dyadicProduct(SkewMatrix<field_type,3>(tmp2).axial(), aContravariant[alpha]);
    }

    //////////////////////////////////////////////////////////
    // Add the local energy density
    //////////////////////////////////////////////////////////

    // Add the membrane energy density
    auto energyDensity = (thickness_ - K*Dune::Power<3>::eval(thickness_) / 12.0) * W_m(Ee)
                       + (Dune::Power<3>::eval(thickness_) / 12.0 - K * Dune::Power<5>::eval(thickness_) / 80.0)*W_m(Ee*b + c*Ke)
                       + Dune::Power<3>::eval(thickness_) / 6.0 * W_mixt(Ee, c*Ke*b - 2*H*c*Ke)
                       + Dune::Power<5>::eval(thickness_) / 80.0 * W_mp( (Ee*b + c*Ke)*b);

    // Add the bending energy density
    energyDensity += (thickness_ - K*Dune::Power<3>::eval(thickness_) / 12.0) * W_curv(Ke)
                   + (Dune::Power<3>::eval(thickness_) / 12.0 - K * Dune::Power<5>::eval(thickness_) / 80.0)*W_curv(Ke*b)
                   + Dune::Power<5>::eval(thickness_) / 80.0 * W_curv(Ke*b*b);

    // Add energy density
    energy += quad[pt].weight() * integrationElement * energyDensity;

    ///////////////////////////////////////////////////////////
    // Volume load contribution
    ///////////////////////////////////////////////////////////

    if (not volumeLoad_)
      continue;

    // Value of the volume load density at the current position
    Dune::FieldVector<double,3> volumeLoadDensity = volumeLoad_(geometry.global(quad[pt].position()));

    // Only translational dofs are affected by the volume load
    for (size_t i=0; i<volumeLoadDensity.size(); i++)
      energy -= thickness_ * (volumeLoadDensity[i] * value.r[i]) * quad[pt].weight() * integrationElement;
  }


  //////////////////////////////////////////////////////////////////////////////
  //   Assemble boundary contributions
  //////////////////////////////////////////////////////////////////////////////

  if (not neumannFunction_)
    return energy;

  for (auto&& it : intersections(neumannBoundary_->gridView(),element) )
  {
    if (not neumannBoundary_ or not neumannBoundary_->contains(it))
      continue;

    const auto& quad = Dune::QuadratureRules<DT, gridDim-1>::rule(it.type(), quadOrder);

    for (size_t pt=0; pt<quad.size(); pt++)
    {
      // Local position of the quadrature point
      const Dune::FieldVector<DT,gridDim>& quadPos = it.geometryInInside().global(quad[pt].position());

      const DT integrationElement = it.geometry().integrationElement(quad[pt].position());

      // The value of the local function
      RigidBodyMotion<field_type,dim> value = localGeodesicFEFunction.evaluate(quadPos);

      // Value of the Neumann data at the current position
      Dune::FieldVector<double,3> neumannValue = neumannFunction_(it.geometry().global(quad[pt].position()));

      // Only translational dofs are affected by the Neumann force
      for (size_t i=0; i<neumannValue.size(); i++)
        energy -= thickness_ * (neumannValue[i] * value.r[i]) * quad[pt].weight() * integrationElement;
    }

  }

  return energy;
}

#endif   //#ifndef DUNE_GFE_NONPLANARCOSSERATSHELLENERGY_HH

