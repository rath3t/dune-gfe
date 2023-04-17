#ifndef DUNE_GFE_NONPLANARCOSSERATSHELLENERGY_HH
#define DUNE_GFE_NONPLANARCOSSERATSHELLENERGY_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/parametertree.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/matrix-vector/crossproduct.hh>

#include <dune/fufem/boundarypatch.hh>

#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>

#include <dune/gfe/cosseratenergystiffness.hh>
#include <dune/gfe/localenergy.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/unitvector.hh>
#include <dune/gfe/tensor3.hh>
#include <dune/gfe/localprojectedfefunction.hh>
#include <dune/gfe/mixedlocalgfeadolcstiffness.hh>

#if HAVE_DUNE_CURVEDGEOMETRY
#include <dune/curvedgeometry/curvedgeometry.hh>
#include <dune/localfunctions/lagrange/lfecache.hh>
#endif

/** \brief Assembles the cosserat energy for a single element.
 *
 * \tparam Basis                       Type of the Basis used for assembling
 * \tparam dim                         Dimension of the Targetspace, 3
 * \tparam field_type                  The coordinate type of the TargetSpace
 * \tparam StressFreeStateGridFunction Type of the GridFunction representing the Cosserat shell in a stress free state
 */
template<class Basis, int dim, class field_type=double, class StressFreeStateGridFunction = 
  Dune::Functions::DiscreteGlobalBasisFunction<Basis,std::vector<Dune::FieldVector<double, Basis::GridView::dimensionworld>> > >
class NonplanarCosseratShellEnergy
  : public Dune::GFE::LocalEnergy<Basis,RigidBodyMotion<field_type,dim> >,
    public MixedLocalGeodesicFEStiffness<Basis,
                                           RealTuple<field_type,dim>,
                                           Rotation<field_type,dim> >
{
  // grid types
  typedef typename Basis::GridView GridView;
  typedef typename GridView::ctype DT;
  typedef RigidBodyMotion<field_type,dim> TargetSpace;
  typedef typename TargetSpace::ctype RT;
  typedef typename GridView::template Codim<0>::Entity Entity;

  // some other sizes
  constexpr static int gridDim = GridView::dimension;
  constexpr static int dimworld = GridView::dimensionworld;

public:

  /** \brief Constructor with a set of material parameters
   * \param parameters                  The material parameters
   * \param stressFreeStateGridFunction Pointer to a parametrization representing the Cosserat shell in a stress-free state
   */
  NonplanarCosseratShellEnergy(const Dune::ParameterTree& parameters,
                               const StressFreeStateGridFunction* stressFreeStateGridFunction,
                               const BoundaryPatch<GridView>* neumannBoundary,
                               const std::function<Dune::FieldVector<double,3>(Dune::FieldVector<double,dimworld>)> neumannFunction,
                               const std::function<Dune::FieldVector<double,3>(Dune::FieldVector<double,dimworld>)> volumeLoad)
  : stressFreeStateGridFunction_(stressFreeStateGridFunction),
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
    
    // Indicator to use the alternative energy W_Coss from Birsan 2021:
    useAlternativeEnergyWCoss_ = parameters.template get<bool>("useAlternativeEnergyWCoss", false);
  }

  /** \brief Assemble the energy for a single element */
  RT energy (const typename Basis::LocalView& localView,
             const std::vector<TargetSpace>& localSolution) const;
/** \brief Assemble the energy for a single element */
  RT energy (const typename Basis::LocalView& localView,
             const std::vector<RealTuple<field_type,dim> >& localDisplacementConfiguration,
             const std::vector<Rotation<field_type,dim> >& localOrientationConfiguration) const override;

  /*  Sources:
      Birsan 2019: Derivation of a refined six-parameter shell model, equation (111)
      Birsan 2021: Alternative derivation of the higher-order constitudtive model for six-parameter elastic shells, equations (119) and (126)
  */
  RT W_Coss(const Dune::FieldMatrix<field_type,3,3>& S, const Dune::FieldMatrix<double,3,3>& a, const Dune::FieldVector<double,3>& n0) const
  {
    return W_Coss_mixt(S,S,a,n0);
  }

  RT W_Coss_mixt(const Dune::FieldMatrix<field_type,3,3>& S, const Dune::FieldMatrix<field_type,3,3>& T, const Dune::FieldMatrix<double,3,3>& a, const Dune::FieldVector<double,3>& n0) const
  {
    auto planarPart = W_mixt(a*S,a*T);
    Dune::FieldVector<field_type, 3> n0S;
    Dune::FieldVector<field_type, 3> n0T;
    S.mtv(n0, n0S);
    T.mtv(n0, n0T);
    field_type normalPart = 2*mu_*mu_c_* n0S * n0T /(mu_ + mu_c_);
    return planarPart + normalPart;
  }

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

  /** \brief Indicator to use the alternative energy W_Coss from Birsan 2021:
             Alternative derivation of the higher-order constitudtive model for six-parameter elastic shells, equations (119) and (126). */

  bool useAlternativeEnergyWCoss_;
  /** \brief The geometry used for assembling */
  const StressFreeStateGridFunction* stressFreeStateGridFunction_;

  /** \brief The Neumann boundary */
  const BoundaryPatch<GridView>* neumannBoundary_;

  /** \brief The function implementing the Neumann data */
  const std::function<Dune::FieldVector<double,3>(Dune::FieldVector<double,dimworld>)> neumannFunction_;

  /** \brief The function implementing a volume load */
  const std::function<Dune::FieldVector<double,3>(Dune::FieldVector<double,dimworld>)> volumeLoad_;
};

template <class Basis, int dim, class field_type, class StressFreeStateGridFunction>
typename NonplanarCosseratShellEnergy<Basis, dim, field_type, StressFreeStateGridFunction>::RT
NonplanarCosseratShellEnergy<Basis,dim,field_type, StressFreeStateGridFunction>::
energy(const typename Basis::LocalView& localView,
       const std::vector<RigidBodyMotion<field_type,dim> >& localSolution) const
{
  // The element geometry
  auto element = localView.element();

  // The set of shape functions on this element
  using namespace Dune::TypeTree::Indices;
  const auto& localFiniteElement = LocalFiniteElementFactory<Basis,0>::get(localView,_0);

#if HAVE_DUNE_CURVEDGEOMETRY
  // Construct a curved geometry of this element of the Cosserat shell in stress-free state
  // When using element.geometry(), then the curvatures on the element are zero, when using a curved geometry, they are not
  // If a parametrization representing the Cosserat shell in a stress-free state is given,
  // this is used for the curved geometry approximation.
  // The variable local holds the local coordinates in the reference element
  // and localGeometry.global maps them to the world coordinates
  auto curvedGeometryGridFunctionOrder = localFiniteElement.localBasis().order();
  Dune::CurvedGeometry<DT, gridDim, dimworld, Dune::CurvedGeometryTraits<DT, Dune::LagrangeLFECache<DT,DT,gridDim>>> geometry(referenceElement(element),
    [this,element](const auto& local) {
      if (not stressFreeStateGridFunction_) {
        return element.geometry().global(local);
      }
      auto localGridFunction = localFunction(*stressFreeStateGridFunction_);
      localGridFunction.bind(element);
      return localGridFunction(local);
    }, curvedGeometryGridFunctionOrder);
#else
  auto geometry = element.geometry();
#endif

  ////////////////////////////////////////////////////////////////////////////////////
  //  Set up the local nonlinear finite element function
  ////////////////////////////////////////////////////////////////////////////////////
  typedef LocalGeodesicFEFunction<gridDim, DT, decltype(localFiniteElement), TargetSpace> LocalGFEFunctionType;
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

    // The derivative of the local function w.r.t. the coordinate system of the tangent space
    auto derivative = localGeodesicFEFunction.evaluateDerivative(quadPos,value);

    //////////////////////////////////////////////////////////
    //  The rotation and its derivative
    //  Note: we need it in matrix coordinates
    //////////////////////////////////////////////////////////

    Dune::FieldMatrix<field_type,dim,dim> R;
    value.q.matrix(R);
    auto RT = Dune::GFE::transpose(R);

    //Derivative of the rotation w.r.t. the coordinate system of the tangent space
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
    // The contravariant base vectors are the *columns* of the inverse of the covariant matrix
    // To get an easier access to the columns, we use the transpose of the contravariant matrix
    aContravariant = Dune::GFE::transpose(aContravariant);

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

    Dune::FieldMatrix<double,3,3> b(0);
#if HAVE_DUNE_CURVEDGEOMETRY
    // In case dune-curvedgeometry is not installed, we assume the second fundamental form b
    // ( (-1) * the derivative of the normal field, evaluated at each quadrature point) is zero!
    auto grad_s_n = geometry.normalGradient(quad[pt].position());
    grad_s_n *= (-1);
    for (int i = 0; i < dimworld; i++)
      for (int j = 0; j < dimworld; j++)
        b[i][j] = grad_s_n[i][j];
#endif
    
    // Mean curvatue
    auto H = 0.5 * Dune::GFE::trace(b);

    // Gauss curvature, calculated with the normalGradient in the euclidean coordinate system
    // see e.g. formula (3.5) in "Reﬁned dimensional reduction for isotropic elastic Cosserat shells with initial curvature"
    auto bSquared = b*b;
    auto K = 2*H*H - 0.5*Dune::GFE::trace(bSquared);

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

    // Elastic shell bending-curvature strain, e.g. formula (4.56) in "Reﬁned dimensional reduction for isotropic elastic Cosserat shells with initial curvature"
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
    field_type energyDensity = 0;
    if (useAlternativeEnergyWCoss_) {
      energyDensity += (thickness_ - K*Dune::power(thickness_,3) / 12.0) * W_Coss(Ee, a, aContravariant[2])
                    + (Dune::power(thickness_,3) / 12.0 - K * Dune::power(thickness_,5) / 80.0)*W_Coss(Ee*b + c*Ke, a, aContravariant[2])
                    + Dune::power(thickness_,3) / 6.0 * W_Coss_mixt(Ee, c*Ke*b - 2*H*c*Ke, a, aContravariant[2])
                    + Dune::power(thickness_,5) / 80.0 * W_Coss( (Ee*b + c*Ke)*b, a, aContravariant[2]);
    } else {
      energyDensity += (thickness_ - K*Dune::power(thickness_,3) / 12.0) * W_m(Ee)
                    + (Dune::power(thickness_,3) / 12.0 - K * Dune::power(thickness_,5) / 80.0)*W_m(Ee*b + c*Ke)
                    + Dune::power(thickness_,3) / 6.0 * W_mixt(Ee, c*Ke*b - 2*H*c*Ke)
                    + Dune::power(thickness_,5) / 80.0 * W_mp( (Ee*b + c*Ke)*b);
    }

    // Add the bending energy density
    energyDensity += (thickness_ - K*Dune::power(thickness_,3) / 12.0) * W_curv(Ke)
                   + (Dune::power(thickness_,3) / 12.0 - K * Dune::power(thickness_,5) / 80.0)*W_curv(Ke*b)
                   + Dune::power(thickness_,5) / 80.0 * W_curv(Ke*b*b);

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
      energy += (volumeLoadDensity[i] * value.r[i]) * quad[pt].weight() * integrationElement;
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
        energy += (neumannValue[i] * value.r[i]) * quad[pt].weight() * integrationElement;
    }
  }

  return energy;
}

template <class Basis, int dim, class field_type, class StressFreeStateGridFunction>
typename NonplanarCosseratShellEnergy<Basis, dim, field_type, StressFreeStateGridFunction>::RT
NonplanarCosseratShellEnergy<Basis,dim,field_type, StressFreeStateGridFunction>::
energy(const typename Basis::LocalView& localView,
       const std::vector<RealTuple<field_type,dim> >& localDeformationConfiguration,
       const std::vector<Rotation<field_type,dim> >& localOrientationConfiguration) const
{
  // The element geometry
  auto element = localView.element();

  // The set of shape functions on this element

  using namespace Dune::TypeTree::Indices;
  const auto& deformationLocalFiniteElement = LocalFiniteElementFactory<Basis,0>::get(localView,_0);
  const auto& orientationLocalFiniteElement = LocalFiniteElementFactory<Basis,1>::get(localView,_1);

#if HAVE_DUNE_CURVEDGEOMETRY
  // Construct a curved geometry of this element of the Cosserat shell in stress-free state
  // When using element.geometry(), then the curvatures on the element are zero, when using a curved geometry, they are not
  // If a parametrization representing the Cosserat shell in a stress-free state is given,
  // this is used for the curved geometry approximation.
  // The variable local holds the local coordinates in the reference element
  // and localGeometry.global maps them to the world coordinates
  auto curvedGeometryGridFunctionOrder = deformationLocalFiniteElement.localBasis().order();
  Dune::CurvedGeometry<DT, gridDim, dimworld, Dune::CurvedGeometryTraits<DT, Dune::LagrangeLFECache<DT,DT,gridDim>>> geometry(referenceElement(element),
    [this,element](const auto& local) {
      if (not stressFreeStateGridFunction_) {
        return element.geometry().global(local);
      }
      auto localGridFunction = localFunction(*stressFreeStateGridFunction_);
      localGridFunction.bind(element);
      return localGridFunction(local);
    }, curvedGeometryGridFunctionOrder);
#else
  auto geometry = element.geometry();
#endif

  ////////////////////////////////////////////////////////////////////////////////////
  //  Set up the local nonlinear finite element function
  ////////////////////////////////////////////////////////////////////////////////////
  typedef LocalGeodesicFEFunction<gridDim, DT, decltype(deformationLocalFiniteElement), RealTuple<field_type,dim> > LocalDeformationGFEFunctionType;
  typedef LocalGeodesicFEFunction<gridDim, DT, decltype(orientationLocalFiniteElement), Rotation<field_type,dim> > LocalOrientationGFEFunctionType;
  LocalDeformationGFEFunctionType localDeformationGFEFunction(deformationLocalFiniteElement,localDeformationConfiguration);
  LocalOrientationGFEFunctionType localOrientationGFEFunction(orientationLocalFiniteElement,localOrientationConfiguration);

  RT energy = 0;

  auto quadOrder = (deformationLocalFiniteElement.type().isSimplex()) ? deformationLocalFiniteElement.localBasis().order()
                                                : deformationLocalFiniteElement.localBasis().order() * gridDim;

  const auto& quad = Dune::QuadratureRules<DT, gridDim>::rule(element.type(), quadOrder);

  for (size_t pt=0; pt<quad.size(); pt++)
  {
    // Local position of the quadrature point
    const Dune::FieldVector<DT,gridDim>& quadPos = quad[pt].position();

    const DT integrationElement = geometry.integrationElement(quadPos);

    // The value of the local function
    RealTuple<field_type,dim> deformationValue = localDeformationGFEFunction.evaluate(quadPos);
    Rotation<field_type,dim>  orientationValue = localOrientationGFEFunction.evaluate(quadPos);

    // The derivative of the local function w.r.t. the coordinate system of the tangent space
    typename LocalDeformationGFEFunctionType::DerivativeType deformationDerivative = localDeformationGFEFunction.evaluateDerivative(quadPos,deformationValue);
    typename LocalOrientationGFEFunctionType::DerivativeType orientationDerivative = localOrientationGFEFunction.evaluateDerivative(quadPos,orientationValue);

    //////////////////////////////////////////////////////////
    //  The rotation and its derivative
    //  Note: we need it in matrix coordinates
    //////////////////////////////////////////////////////////

    Dune::FieldMatrix<field_type,dim,dim> R;
    orientationValue.matrix(R);
    auto RT = Dune::GFE::transpose(R);

    //Derivative of the rotation w.r.t. the coordinate system of the tangent space
    Tensor3<field_type,3,3,gridDim> DR = orientationValue.quaternionTangentToMatrixTangent(orientationDerivative);

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
    // The contravariant base vectors are the *columns* of the inverse of the covariant matrix
    // To get an easier access to the columns, we use the transpose of the contravariant matrix
    aContravariant = Dune::GFE::transpose(aContravariant);

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

    Dune::FieldMatrix<double,3,3> b(0);
#if HAVE_DUNE_CURVEDGEOMETRY
    // In case dune-curvedgeometry is not installed, we assume the second fundamental form b
    // ( (-1) * the derivative of the normal field, evaluated at each quadrature point) is zero!
    auto grad_s_n = geometry.normalGradient(quad[pt].position());
    for (int i = 0; i < dimworld; i++)
      for (int j = 0; j < dimworld; j++)
        b[i][j] = grad_s_n[i][j];
#endif
    b *= (-1);
    
    // Mean curvatue
    auto H = 0.5 * Dune::GFE::trace(b);

    // Gauss curvature, calculated with the normalGradient in the euclidean coordinate system
    // see e.g. formula (3.5) in "Reﬁned dimensional reduction for isotropic elastic Cosserat shells with initial curvature"
    auto bSquared = b*b;
    auto K = 2*H*H - 0.5*Dune::GFE::trace(bSquared);

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
        vec[i] = deformationDerivative[i][alpha];
      grad_s_m += Dune::GFE::dyadicProduct(vec, aContravariant[alpha]);
    }

    Ee = RT * grad_s_m - a;

    // Elastic shell bending-curvature strain, e.g. formula (4.56) in "Reﬁned dimensional reduction for isotropic elastic Cosserat shells with initial curvature"
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
    field_type energyDensity = 0;
    if (useAlternativeEnergyWCoss_) {
      energyDensity += (thickness_ - K*Dune::power(thickness_,3) / 12.0) * W_Coss(Ee, a, aContravariant[2])
                    + (Dune::power(thickness_,3) / 12.0 - K * Dune::power(thickness_,5) / 80.0)*W_Coss(Ee*b + c*Ke, a, aContravariant[2])
                    + Dune::power(thickness_,3) / 6.0 * W_Coss_mixt(Ee, c*Ke*b - 2*H*c*Ke, a, aContravariant[2])
                    + Dune::power(thickness_,5) / 80.0 * W_Coss( (Ee*b + c*Ke)*b, a, aContravariant[2]);
    } else {
      energyDensity += (thickness_ - K*Dune::power(thickness_,3) / 12.0) * W_m(Ee)
                    + (Dune::power(thickness_,3) / 12.0 - K * Dune::power(thickness_,5) / 80.0)*W_m(Ee*b + c*Ke)
                    + Dune::power(thickness_,3) / 6.0 * W_mixt(Ee, c*Ke*b - 2*H*c*Ke)
                    + Dune::power(thickness_,5) / 80.0 * W_mp( (Ee*b + c*Ke)*b);
    }

    energyDensity += (thickness_ - K*Dune::power(thickness_,3) / 12.0) * W_curv(Ke)
                   + (Dune::power(thickness_,3) / 12.0 - K * Dune::power(thickness_,5) / 80.0)*W_curv(Ke*b)
                   + Dune::power(thickness_,5) / 80.0 * W_curv(Ke*b*b);

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
      energy += (volumeLoadDensity[i] * deformationValue[i]) * quad[pt].weight() * integrationElement;
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
      RealTuple<field_type,dim> deformationValue = localDeformationGFEFunction.evaluate(quadPos);

      // Value of the Neumann data at the current position
      Dune::FieldVector<double,3> neumannValue = neumannFunction_(it.geometry().global(quad[pt].position()));

      // Only translational dofs are affected by the Neumann force
      for (size_t i=0; i<neumannValue.size(); i++)
        energy += (neumannValue[i] * deformationValue[i]) * quad[pt].weight() * integrationElement;
    }
  }

  return energy;
}

#endif   //#ifndef DUNE_GFE_NONPLANARCOSSERATSHELLENERGY_HH

