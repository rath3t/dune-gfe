#ifndef DUNE_GFE_SURFACECOSSERATENERGY_HH
#define DUNE_GFE_SURFACECOSSERATENERGY_HH

#include <dune/geometry/quadraturerules.hh>

#include <dune/fufem/functions/virtualgridfunction.hh>
#include <dune/fufem/boundarypatch.hh>

#include <dune/gfe/cosseratstrain.hh>
#include <dune/gfe/localenergy.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/localprojectedfefunction.hh>
#include <dune/gfe/mixedlocalgeodesicfestiffness.hh>
#include <dune/gfe/orthogonalmatrix.hh>
#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/tensor3.hh>
#include <dune/gfe/vertexnormals.hh>

template <class Basis, std::size_t i>
class LocalFiniteElementFactory
{
public:
  static auto get(const typename Basis::LocalView& localView,
           std::integral_constant<std::size_t, i> iType)
    -> decltype(localView.tree().child(iType).finiteElement())
  {
    return localView.tree().child(iType).finiteElement();
  }
};

/** \brief Specialize for scalar bases, here we cannot call tree().child() */
template <class GridView, int order, std::size_t i>
class LocalFiniteElementFactory<Dune::Functions::LagrangeBasis<GridView,order>,i>
{
public:
  static auto get(const typename Dune::Functions::LagrangeBasis<GridView,order>::LocalView& localView,
           std::integral_constant<std::size_t, i> iType)
    -> decltype(localView.tree().finiteElement())
  {
    return localView.tree().finiteElement();
  }
};

template<class Basis, class TargetSpace, class field_type=double, class GradientRT=double>
class SurfaceCosseratEnergy
: public Dune::GFE::LocalEnergy<Basis,TargetSpace>
{
 // grid types
  typedef typename Basis::GridView GridView;
  typedef typename GridView::ctype ctype;
  typedef typename Basis::LocalView::Tree::FiniteElement LocalFiniteElement;
  typedef typename GridView::ctype DT;
  typedef typename TargetSpace::ctype RT;
  typedef typename GridView::template Codim<0>::Entity Entity;
  typedef typename TargetSpace::template rebind<adouble>::other ATargetSpace;

  enum {dim=GridView::dimension};
  enum {dimworld=GridView::dimensionworld};
  enum {gridDim=GridView::dimension};
  static constexpr int boundaryDim = gridDim - 1;


  /** \brief Compute the symmetric part of a matrix A, i.e. \f$ \frac 12 (A + A^T) \f$ */
  static Dune::FieldMatrix<field_type,dim,dim> sym(const Dune::FieldMatrix<field_type,dim,dim>& A)
  {
    Dune::FieldMatrix<field_type,dim,dim> result;
    for (int i=0; i<dim; i++)
      for (int j=0; j<dim; j++)
        result[i][j] = 0.5 * (A[i][j] + A[j][i]);
    return result;
  }

  /** \brief Compute the antisymmetric part of a matrix A, i.e. \f$ \frac 12 (A - A^T) \f$ */
  static Dune::FieldMatrix<field_type,dim,dim> skew(const Dune::FieldMatrix<field_type,dim,dim>& A)
  {
    Dune::FieldMatrix<field_type,dim,dim> result;
    for (int i=0; i<dim; i++)
      for (int j=0; j<dim; j++)
        result[i][j] = 0.5 * (A[i][j] - A[j][i]);
    return result;
  }

  /** \brief Compute the deviator of a matrix A */
  static Dune::FieldMatrix<field_type,dim,dim> dev(const Dune::FieldMatrix<field_type,dim,dim>& A)
  {
    Dune::FieldMatrix<field_type,dim,dim> result = A;
    auto t = trace(A);
    for (int i=0; i<dim; i++)
      result[i][i] -= t / dim;
    return result;
  }

  /** \brief Return the trace of a matrix */
  template <class T, int N>
  static T trace(const Dune::FieldMatrix<T,N,N>& A)
  {
    T trace = 0;
    for (int i=0; i<N; i++)
      trace += A[i][i];
    return trace;
  }

  /** \brief Return the square of the trace of a matrix */
  template <int N>
  static field_type traceSquared(const Dune::FieldMatrix<field_type,N,N>& A)
  {
    field_type trace = 0;
    for (int i=0; i<N; i++)
      trace += A[i][i];
    return trace*trace;
  }

  template <class T, int N>
  static T frobeniusProduct(const Dune::FieldMatrix<T,N,N>& A, const Dune::FieldMatrix<T,N,N>& B)
  {
    T result(0.0);

    for (int i=0; i<N; i++)
      for (int j=0; j<N; j++)
        result += A[i][j] * B[i][j];

    return result;
  }

  template <int N>
  static auto dyadicProduct(const Dune::FieldVector<adouble,N>& A, const Dune::FieldVector<adouble,N>& B)
    -> Dune::FieldMatrix<adouble,N,N>
  {
    Dune::FieldMatrix<adouble,N,N> result;

    for (int i=0; i<N; i++)
      for (int j=0; j<N; j++)
        result[i][j] = A[i]*B[j];

    return result;
  }

  template <int N>
  static auto dyadicProduct(const Dune::FieldVector<double,N>& A, const Dune::FieldVector<double,N>& B)
    -> Dune::FieldMatrix<double,N,N>
  {
    Dune::FieldMatrix<double,N,N> result;

    for (int i=0; i<N; i++)
      for (int j=0; j<N; j++)
        result[i][j] = A[i]*B[j];

    return result;
  }

  template <int N>
  static auto dyadicProduct(const Dune::FieldVector<adouble,N>& A, const Dune::FieldVector<double,N>& B)
    -> Dune::FieldMatrix<adouble,N,N>
  {
    Dune::FieldMatrix<adouble,N,N> result;

    for (int i=0; i<N; i++)
      for (int j=0; j<N; j++)
        result[i][j] = A[i]*B[j];

    return result;
  }

  template <class T, int N>
  static Dune::FieldMatrix<T,N,N> transpose(const Dune::FieldMatrix<T,N,N>& A)
  {
    Dune::FieldMatrix<T,N,N> result;

    for (int i=0; i<N; i++)
      for (int j=0; j<N; j++)
        result[i][j] = A[j][i];

    return result;
  }


  /** \brief Compute the derivative of the rotation (with respect to x), but wrt matrix coordinates
      \param value Value of the gfe function at a certain point
      \param derivative First derivative of the gfe function wrt x at that point, in quaternion coordinates
      \param DR First derivative of the gfe function wrt x at that point, in matrix coordinates
   */
  static void computeDR(const RigidBodyMotion<field_type,dimworld>& value,
                        const Dune::FieldMatrix<field_type,7,boundaryDim>& derivative,
                        Tensor3<field_type,3,3,boundaryDim>& DR)
  {
    // The LocalGFEFunction class gives us the derivatives of the orientation variable,
    // but as a map into quaternion space.  To obtain matrix coordinates we use the
    // chain rule, which means that we have to multiply the given derivative with
    // the derivative of the embedding of the unit quaternion into the space of 3x3 matrices.
    // This second derivative is almost given by the method getFirstDerivativesOfDirectors.
    // However, since the directors of a given unit quaternion are the _columns_ of the
    // corresponding orthogonal matrix, we need to invert the i and j indices
    //
    // So, DR[i][j][k] contains \partial R_ij / \partial k
    Tensor3<field_type,3 , 3, 4> dd_dq;
    value.q.getFirstDerivativesOfDirectors(dd_dq);

    DR = field_type(0);
    for (int i=0; i<3; i++)
      for (int j=0; j<3; j++)
        for (int k=0; k<boundaryDim; k++)
          for (int l=0; l<4; l++)
            DR[i][j][k] += dd_dq[j][i][l] * derivative[l+3][k];
  }



public:

  /** \brief Constructor with a set of material parameters
   * \param parameters The material parameters
   */
  SurfaceCosseratEnergy(const Dune::ParameterTree& parameters, const std::vector<UnitVector<double,3> >& vertexNormals, const BoundaryPatch<GridView>* shellBoundary)
  : shellBoundary_(shellBoundary),
    vertexNormals_(vertexNormals)
  {
    // The shell thickness
    thickness_ = parameters.template get<double>("thickness");

    // Lame constants
    mu_ = parameters.template get<double>("mu_cosserat");
    lambda_ = parameters.template get<double>("lambda_cosserat");

    // Cosserat couple modulus
    mu_c_ = parameters.template get<double>("mu_c");

    // Length scale parameter
    L_c_ = parameters.template get<double>("L_c");

    // Curvature exponent
    q_ = parameters.template get<double>("q");

    // Shear correction factor
    kappa_ = parameters.template get<double>("kappa");

    b1_ = parameters.template get<double>("b1");
    b2_ = parameters.template get<double>("b2");
    b3_ = parameters.template get<double>("b3");
  }

RT energy(const typename Basis::LocalView& localView,
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
    cornerNormals[i] = vertexNormals_[gridView.indexSet().subIndex(element,i,gridDim)];

  typedef typename Dune::PQkLocalFiniteElementCache<DT, double, gridDim, 1> P1FiniteElementCache;
  typedef typename P1FiniteElementCache::FiniteElementType P1LocalFiniteElement;
  //it.type()?
  P1FiniteElementCache p1FiniteElementCache;
  const auto& p1LocalFiniteElement = p1FiniteElementCache.get(element.type());
  Dune::GFE::LocalProjectedFEFunction<gridDim, DT, P1LocalFiniteElement, UnitVector<double,3> > unitNormals(p1LocalFiniteElement, cornerNormals);

  ////////////////////////////////////////////////////////////////////////////////////
  //  Set up the local nonlinear finite element function
  ////////////////////////////////////////////////////////////////////////////////////
  typedef LocalGeodesicFEFunction<gridDim, DT, LocalFiniteElement, TargetSpace> LocalGFEFunctionType;
  LocalGFEFunctionType localGeodesicFEFunction(localFiniteElement,localSolution);

  RT energy = 0;

  for (auto&& it : intersections(shellBoundary_->gridView(), element)) {
    if (not shellBoundary_->contains(it))
      continue;
    
    auto quadOrder = (it.type().isSimplex()) ? localFiniteElement.localBasis().order()
                                                  : localFiniteElement.localBasis().order() * gridDim;

    const auto& quad = Dune::QuadratureRules<DT, boundaryDim>::rule(it.type(), quadOrder);
    for (size_t pt=0; pt<quad.size(); pt++) {
    
      // Local position of the quadrature point
      const Dune::FieldVector<DT,gridDim>& quadPos = it.geometryInInside().global(quad[pt].position());;

      const DT integrationElement = it.geometry().integrationElement(quad[pt].position());

      // The value of the local function
      RigidBodyMotion<field_type,dim> value = localGeodesicFEFunction.evaluate(quadPos);

      // The derivative of the local function
      auto derivative = localGeodesicFEFunction.evaluateDerivative(quadPos,value);

      Dune::FieldMatrix<RT,7,2> derivative2D;
      for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 2; j++) {
          derivative2D[i][j] = derivative[i][j];
        }
      }
      //////////////////////////////////////////////////////////
      //  The rotation and its derivative
      //  Note: we need it in matrix coordinates
      //////////////////////////////////////////////////////////

      Dune::FieldMatrix<field_type,dim,dim> R;
      value.q.matrix(R);
      auto RT = transpose(R);

      Tensor3<field_type,3,3,boundaryDim> DR;
      computeDR(value, derivative2D, DR);

      //////////////////////////////////////////////////////////
      //  Fundamental forms and curvature
      //////////////////////////////////////////////////////////

      // First fundamental form
      Dune::FieldMatrix<double,3,3> aCovariant;

      // If dimworld==3, then the first two lines of aCovariant are simply the jacobianTransposed
      // of the element.  If dimworld<3 (i.e., ==2), we have to explicitly enters 0.0 in the last column.
      const auto jacobianTransposed = it.geometry().jacobianTransposed(quad[pt].position());
      // auto jacobianTransposed = geometry.jacobianTransposed(quadPos);

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
      for (int alpha=0; alpha<boundaryDim; alpha++)
        a += dyadicProduct(aCovariant[alpha], aContravariant[alpha]);

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
          c += sqrt(aScalar) * eps[alpha][beta] * dyadicProduct(aContravariant[alpha], aContravariant[beta]);

      // Second fundamental form
      // The derivative of the normal field
      auto normalDerivative = unitNormals.evaluateDerivative(quadPos);

      Dune::FieldMatrix<double,3,3> b(0);
      for (int alpha=0; alpha<boundaryDim; alpha++)
      {
        Dune::FieldVector<double,3> vec;
        for (int i=0; i<3; i++)
          vec[i] = normalDerivative[i][alpha];
        b -= dyadicProduct(vec, aContravariant[alpha]);
      }

      // Gauss curvature
      auto K = b.determinant();

      // Mean curvatue
      auto H = 0.5 * trace(b);

      //////////////////////////////////////////////////////////
      //  Strain tensors
      //////////////////////////////////////////////////////////

      // Elastic shell strain
      Dune::FieldMatrix<field_type,3,3> Ee(0);
      Dune::FieldMatrix<field_type,3,3> grad_s_m(0);
      for (int alpha=0; alpha<boundaryDim; alpha++)
      {
        Dune::FieldVector<field_type,3> vec;
        for (int i=0; i<3; i++)
          vec[i] = derivative[i][alpha];
        grad_s_m += dyadicProduct(vec, aContravariant[alpha]);
      }

      Ee = RT * grad_s_m - a;

      // Elastic shell bending-curvature strain
      Dune::FieldMatrix<field_type,3,3> Ke(0);
      for (int alpha=0; alpha<boundaryDim; alpha++)
      {
        Dune::FieldMatrix<field_type,3,3> tmp;
        for (int i=0; i<3; i++)
          for (int j=0; j<3; j++)
            tmp[i][j] = DR[i][j][alpha];
        auto tmp2 = RT * tmp;
        Ke += dyadicProduct(SkewMatrix<field_type,3>(tmp2).axial(), aContravariant[alpha]);
      }

      //////////////////////////////////////////////////////////
      // Add the local energy density
      //////////////////////////////////////////////////////////

      // Add the membrane energy density
      auto energyDensity = (thickness_ - K*Dune::Power<3>::eval(thickness_) / 12.0) * W_m(Ee);
      energyDensity += (Dune::Power<3>::eval(thickness_) / 12.0 - K * Dune::Power<5>::eval(thickness_) / 80.0)*W_m(Ee*b + c*Ke);
      energyDensity += Dune::Power<3>::eval(thickness_) / 6.0 * W_mixt(Ee, c*Ke*b - 2*H*c*Ke);
      energyDensity += Dune::Power<5>::eval(thickness_) / 80.0 * W_mp( (Ee*b + c*Ke)*b);

      // Add the bending energy density
      energyDensity += (thickness_ - K*Dune::Power<3>::eval(thickness_) / 12.0) * W_curv(Ke)
                     + (Dune::Power<3>::eval(thickness_) / 12.0 - K * Dune::Power<5>::eval(thickness_) / 80.0)*W_curv(Ke*b)
                     + Dune::Power<5>::eval(thickness_) / 80.0 * W_curv(Ke*b*b);

      // Add energy density
      energy += quad[pt].weight() * integrationElement * energyDensity;
    }
  }

  return energy;
}

  RT W_m(const Dune::FieldMatrix<field_type,3,3>& S) const
  {
    return W_mixt(S,S);
  }

  RT W_mixt(const Dune::FieldMatrix<field_type,3,3>& S, const Dune::FieldMatrix<field_type,3,3>& T) const
  {
    return mu_ * frobeniusProduct(sym(S), sym(T))
         + mu_c_ * frobeniusProduct(skew(S), skew(T))
         + lambda_ * mu_ / (lambda_ + 2*mu_) * trace(S) * trace(T);
  }

  RT W_mp(const Dune::FieldMatrix<field_type,3,3>& S) const
  {
    return mu_ * sym(S).frobenius_norm2() + mu_c_ * skew(S).frobenius_norm2() + lambda_ * 0.5 * traceSquared(S);
  }

  RT W_curv(const Dune::FieldMatrix<field_type,3,3>& S) const
  {
    return mu_ * L_c_ * L_c_ * (b1_ * dev(sym(S)).frobenius_norm2() + b2_ * skew(S).frobenius_norm2() + b3_ * traceSquared(S));
  }

private:
  /** \brief The Neumann boundary */
  const BoundaryPatch<GridView>* shellBoundary_;

  /** \brief The normal vectors at the grid vertices.  This are used to compute the reference surface curvature. */
  std::vector<UnitVector<double,3> > vertexNormals_;

  /** \brief The shell thickness */
  double thickness_;

  /** \brief Lame constants */
  double mu_, lambda_;

  /** \brief Cosserat couple modulus, preferably 0 */
  double mu_c_;

  /** \brief Length scale parameter */
  double L_c_;

  /** \brief Curvature exponent */
  double q_;

  /** \brief Shear correction factor */
  double kappa_;

  /** \brief Curvature parameters */
  double b1_, b2_, b3_;

};

#endif   //#ifndef DUNE_GFE_SURFACECOSSERATENERGY_HH
