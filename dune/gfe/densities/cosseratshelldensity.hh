#ifndef DUNE_GFE_DENSITIES_COSSERATSHELLDENSITY_HH
#define DUNE_GFE_DENSITIES_COSSERATSHELLDENSITY_HH

#include <adolc/adouble.h>

#include <dune/common/fmatrix.hh>
#include <dune/common/parametertree.hh>

#if HAVE_DUNE_CURVEDGRID
#include <dune/curvedgrid/curvedgrid.hh>
#endif

#include <dune/matrix-vector/crossproduct.hh>

#include <dune/gfe/linearalgebra.hh>
#include <dune/gfe/tensor3.hh>
#include <dune/gfe/densities/localdensity.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>

namespace Dune::GFE
{

  /** \brief The energy of a Cosserat shell with nonplanar stress-free configuration
   *
   * This is the model that is described in:
   *
   *   Nebel, Sander, Bîrsan, Neff:
   *   A geometrically nonlinear Cosserat shell model for orientable and non-orientable surfaces:
   *   Discretization with geometric finite elements (2023),
   *   Computer Methods in Applied Mechanics and Engineering. 416,
   *   https://doi.org/10.1016/j.cma.2023.116309
   *
   * (and earlier papers by Neff and Bîrsan).
   *
   * \tparam ElementOrIntersection The domain of the density.
   *   Can be either a grid element (i.e., a codimension-0 Entity)
   *   or an Intersection.
   * \tparam field_type Type used for numbers
   */
  template<class ElementOrIntersection, class field_type>
  class CosseratShellDensity final
    : public LocalDensity<ElementOrIntersection,
          ProductManifold<RealTuple<field_type,3>, Rotation<field_type,3> > >
  {
    constexpr static int dimWorld = 3;
    constexpr static int domainDim = 2;

    using Geometry = typename ElementOrIntersection::Geometry;
    using ctype = typename Geometry::ctype;
    using LocalCoordinate = typename Geometry::LocalCoordinate;

    using RigidBodyMotion = ProductManifold<RealTuple<field_type,dimWorld>, Rotation<field_type,dimWorld> >;
    // The target space with 'adouble' as the number type
    using ARigidBodyMotion = ProductManifold<RealTuple<adouble,3>,Rotation<adouble,3> >;

    // The type used for derivatives of geometric FE functions
    using Derivative = FieldMatrix<field_type,RigidBodyMotion::embeddedDim,domainDim>;

#if HAVE_DUNE_CURVEDGEOMETRY
    /** \brief Get the derivative of the element normal vector
     *
     * Currently only CurvedGrid provides this information, and therefore
     * we have this specialization.
     *
     * The first argument is a geometry implementation rather than
     * the geometry itself because it makes the template specialization simpler.
     */
    template <class LF, class LG, bool useInterpolation>
    static auto normalGradient(const Dune::Curved::Geometry<ctype,domainDim,dimWorld,LF,LG,useInterpolation>& geometryImpl,
                               const typename Geometry::LocalCoordinate& pos)
    {
      return geometryImpl.normalGradient(pos);
    }
#endif

    /** \brief Derivative of the element normal if the geometry is not Dune::Curved::Geometry
     *
     * In this case, the element is assumed to be affine, and the derivative
     * of the normal vector is simply the zero matrix.
     *
     * \todo This may lead to wrong results, for example when using
     * bilinear quadrilateral elements in a 3d world.
     */
    template <typename GeometryImpl>
    static auto normalGradient(const GeometryImpl& geometryImpl,
                               const typename Geometry::LocalCoordinate& pos)
    {
      return Dune::FieldMatrix<ctype,3,3>(0);
    }

    /** \brief Compute the derivative of the rotation (with respect to x), but wrt matrix coordinates
        \param value Value of the gfe function at a certain point
        \param derivative First derivative of the gfe function wrt x at that point, in quaternion coordinates
        \param DR First derivative of the gfe function wrt x at that point, in matrix coordinates
     */
    static void computeDR(const RigidBodyMotion& value,
                          const FieldMatrix<field_type,RigidBodyMotion::embeddedDim,domainDim>& derivative,
                          Tensor3<field_type,dimWorld,dimWorld,domainDim>& derivativeRotation)
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
      Tensor3<field_type, dimWorld, dimWorld, 4> dd_dq;
      value[Indices::_1].getFirstDerivativesOfDirectors(dd_dq);

      derivativeRotation = field_type(0);
      for (int i=0; i<dimWorld; i++)
        for (int j=0; j<dimWorld; j++)
          for (int k=0; k<domainDim; k++)
            for (int l=0; l<4; l++)
              derivativeRotation[i][j][k] += dd_dq[j][i][l] * derivative[l+3][k];
    }

    field_type W_mixt(const FieldMatrix<field_type,dimWorld,dimWorld>& S,
                      const FieldMatrix<field_type,dimWorld,dimWorld>& T,
                      double mu, double lambda) const
    {
      return mu * GFE::frobeniusProduct(GFE::sym(S), GFE::sym(T))
             + mu_c_ * GFE::frobeniusProduct(GFE::skew(S), GFE::skew(T))
             + lambda * mu / (lambda + 2*mu) * GFE::trace(S) * GFE::trace(T);
    }

    field_type W_m(const FieldMatrix<field_type,dimWorld,dimWorld>& S, double mu, double lambda) const
    {
      return W_mixt(S,S, mu, lambda);
    }

    field_type W_mp(const FieldMatrix<field_type,3,3>& S, double mu, double lambda) const
    {
      return mu * GFE::sym(S).frobenius_norm2() + mu_c_ * GFE::skew(S).frobenius_norm2() + lambda * 0.5 * GFE::traceSquared(S);
    }

    // For b1 = b2 = 1 and b3 = 1/3, this reduces to S.frobenius_norm2()
    field_type W_curv(const Dune::FieldMatrix<field_type,3,3>& S, double mu) const
    {
      return mu * L_c_ * L_c_ * (b1_ * GFE::dev(Dune::GFE::sym(S)).frobenius_norm2()
                                 + b2_ * GFE::skew(S).frobenius_norm2() + b3_ * GFE::traceSquared(S));
    }

    /*  Sources:
        Birsan 2019: Derivation of a refined six-parameter shell model, equation (111)
        Birsan 2021: Alternative derivation of the higher-order constitudtive model for six-parameter elastic shells, equations (119) and (126)
     */
    field_type W_Coss(const FieldMatrix<field_type,3,3>& S,
                      const FieldMatrix<double,3,3>& a,
                      const Dune::FieldVector<double,3>& n0,
                      double mu, double lambda) const
    {
      return W_Coss_mixt(S,S,a,n0,mu,lambda);
    }

    field_type W_Coss_mixt(const FieldMatrix<field_type,3,3>& S,
                           const FieldMatrix<field_type,3,3>& T,
                           const FieldMatrix<double,3,3>& a,
                           const FieldVector<double,3>& n0,
                           double mu, double lambda) const
    {
      auto planarPart = W_mixt(a*S,a*T,mu,lambda);
      FieldVector<field_type, 3> n0S;
      FieldVector<field_type, 3> n0T;
      S.mtv(n0, n0S);
      T.mtv(n0, n0T);
      field_type normalPart = 2*mu*mu_c_* n0S * n0T /(mu + mu_c_);
      return planarPart + normalPart;
    }


  public:

    /** \brief Constructor with a set of material parameters
     * \param parameters The material parameters, including the shell thickness
     */
    CosseratShellDensity(const Dune::ParameterTree& parameters)
    {
      // Make a constant function of the given scalar value for the thickness
      auto thicknessValue = parameters.template get<double>("thickness");

      thicknessF_ = [thicknessValue](const FieldVector<double,dimWorld>& x){return thicknessValue;};

      // Same for the Lamé parameters
      double mu = parameters.template get<double>("mu");
      double lambda = parameters.template get<double>("lambda");

      lameF_ = [mu,lambda](const FieldVector<double,dimWorld>& x)
               -> FieldVector<double,2>
               {
                 return {mu,lambda};
               };

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

    /** \brief Constructor with space-dependent thickness and Lamé parameters
     * \param parameters The constant-in-space material parameters
     * \param thickness The shell thickness parameter, as a function of 3d space
     * \param lame The Lamé parameters, as a function of 3d space
     */
    CosseratShellDensity(const Dune::ParameterTree& parameters,
                         const std::function<double(Dune::FieldVector<double,dimWorld>)> thickness,
                         const std::function<Dune::FieldVector<double,2>(Dune::FieldVector<double,dimWorld>)> lame)
      : thicknessF_(thickness),
      lameF_(lame)
    {
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

    /** \brief Constructor with space-dependent thickness and Lamé parameters
     * \param thickness The shell thickness parameter, as a function of 3d space
     * \param lame The Lamé parameters, as a function of 3d space
     */
    CosseratShellDensity(double mu_c,
                         double L_c,
                         double b1, double b2, double b3,
                         bool useAlternativeEnergyWCoss,
                         const std::function<double(Dune::FieldVector<double,dimWorld>)> thickness,
                         const std::function<Dune::FieldVector<double,2>(Dune::FieldVector<double,dimWorld>)> lame)
      : thicknessF_(thickness),
      lameF_(lame),
      mu_c_(mu_c), L_c_(L_c), b1_(b1), b2_(b2), b3_(b3),
      useAlternativeEnergyWCoss_(useAlternativeEnergyWCoss)
    {}

    /** \brief Bind the density to a `ElementOrIntersection` object
     *
     * Stores a copy of the `ElementOrIntersection`'s geometry.
     **/
    void bind(const ElementOrIntersection& elementOrIntersection)
    {
      geometry_.emplace(elementOrIntersection.geometry());
    }

    /** \brief Evaluate the density for a given value and first derivative
     *
     * \param x The current position
     * \param value The value of the integrand at x
     * \param derivative The derivative of the integrand at x
     */
    field_type operator() (const LocalCoordinate& x,
                           const typename RigidBodyMotion::CoordinateType& value,
                           const Derivative& derivative) const override
    {
      // TODO: Using this method via LocalIntegralEnergy will not produce
      // correct results: LocalIntegralEnergy transforms the 'derivative'
      // argument from the reference element to the grid element, while
      // this density expects the derivatives from the reference element.
      // At least that's how I think it is.  Until I have found the time to check,
      // let's abort here.
      DUNE_THROW(NotImplemented, "Check whether this method produces the correct results!");

      ////////////////////////////////
      //  First fundamental form
      ////////////////////////////////

      FieldMatrix<ctype,3,3> aCovariant;

      // If dimworld==3, then the first two lines of aCovariant are simply the jacobianTransposed
      // of the element.  If dimworld<3 (i.e., ==2), we have to explicitly enters 0.0 in the last column.
      const auto jacobianTransposed = geometry_->jacobianTransposed(x);

      for (int i=0; i<2; i++)
      {
        for (int j=0; j<dimWorld; j++)
          aCovariant[i][j] = jacobianTransposed[i][j];
        for (int j=dimWorld; j<3; j++)
          aCovariant[i][j] = 0.0;
      }

      aCovariant[2] = MatrixVector::crossProduct(aCovariant[0], aCovariant[1]);
      aCovariant[2] /= aCovariant[2].two_norm();

      //////////////////////////////////////////////////////////
      // Add the local energy density
      //////////////////////////////////////////////////////////

      return operator()(x,
                        aCovariant,
                        normalGradient(geometry_->impl(), x),
                        RigidBodyMotion(value),
                        derivative);
    }

    /** \brief Evaluation with the current position, the deformation function, the deformation gradient, the rotation and the rotation gradient
     *
     * \param x The current position
     * \param aCovariant The covariant basis at x
     * \param value The deformation at the current position
     * \param derivative The derivative of the deformation at the current position
     */
    field_type operator() (const LocalCoordinate& x,
                           const FieldMatrix<double,dimWorld,dimWorld>& aCovariant,
                           // TODO: Fix the following type
                           const FieldMatrix<double,dimWorld,dimWorld>& normalGradient,
                           const RigidBodyMotion& value,
                           const Derivative& derivative) const
    {
      auto xGlobal = geometry_->global(x);
      double thickness = thicknessF_(xGlobal);
      auto lameConstants = lameF_(xGlobal);
      auto mu = lameConstants[0];
      auto lambda = lameConstants[1];
      // TODO: Use structured binding here
      //const auto& [mu, lambda] = lameConstants;

      //////////////////////////////////////////////////////////
      //  The rotation and its derivative
      //  Note: we need it in matrix coordinates
      //////////////////////////////////////////////////////////

      FieldMatrix<field_type,dimWorld,dimWorld> R;
      value[Indices::_1].matrix(R);
      auto rt = Dune::GFE::transpose(R);

      Tensor3<field_type,dimWorld,dimWorld,domainDim> derivativeRotation;
      computeDR(value, derivative, derivativeRotation);

      //////////////////////////////////////////////////////////
      //  Fundamental forms and curvature
      //////////////////////////////////////////////////////////

      auto aContravariant = aCovariant;
      aContravariant.invert();

      // The contravariant base vectors are the *columns* of the inverse of the covariant matrix
      // To get an easier access to the columns, we use the transpose of the contravariant matrix
      aContravariant = GFE::transpose(aContravariant);

      // First fundamental tensor
      FieldMatrix<double,3,3> a(0);
      for (int alpha=0; alpha<domainDim; alpha++)
        a += GFE::dyadicProduct(aCovariant[alpha], aContravariant[alpha]);

      // Second fundamental tensor: The derivative of the normal field
      auto b = (-1) * normalGradient;

      // Area element of the domain
      // TODO: Write this as determinant of a 2x2 matrix
      auto a00 = aCovariant[0] * aCovariant[0];
      auto a01 = aCovariant[0] * aCovariant[1];
      auto a10 = aCovariant[1] * aCovariant[0];
      auto a11 = aCovariant[1] * aCovariant[1];
      auto areaElement = std::sqrt(a00*a11 - a10*a01);

      // Surface alternating pseudo-tensor
      FieldMatrix<int,2,2> eps = {{0,1},{-1,0}};
      FieldMatrix<double,3,3> c(0);

      // TODO: Fold the eps.  Currently we multiply with zeros.  Also, the paper doesn't have it either.
      for (int alpha=0; alpha<2; alpha++)
        for (int beta=0; beta<2; beta++)
          c += areaElement * eps[alpha][beta] * Dune::GFE::dyadicProduct(aContravariant[alpha], aContravariant[beta]);

      // Mean curvatue
      auto H = 0.5 * GFE::trace(b);

      // Gauss curvature, calculated with the normalGradient in the Euclidean coordinate system
      // see e.g. formula (3.5) from "Reﬁned dimensional reduction for isotropic elastic Cosserat shells with initial curvature"
      // TODO: No need to compute the full square of b -- all we need is its trace
      auto bSquared = b*b;
      auto K = 2*H*H - 0.5*GFE::trace(bSquared);

      //////////////////////////////////////////////////////////
      //  Strain tensors
      //////////////////////////////////////////////////////////

      // Elastic shell strain
      FieldMatrix<field_type,3,3> grad_s_m(0);
      for (int alpha=0; alpha<domainDim; alpha++)
      {
        FieldVector<field_type,3> vec;
        for (int i=0; i<3; i++)
          vec[i] = derivative[i][alpha];
        grad_s_m += GFE::dyadicProduct(vec, aContravariant[alpha]);
      }

      FieldMatrix<field_type,3,3> Ee = rt * grad_s_m - a;

      // Elastic shell bending-curvature strain
      FieldMatrix<field_type,3,3> Ke(0);
      for (int alpha=0; alpha<domainDim; alpha++)
      {
        FieldMatrix<field_type,3,3> tmp;
        for (int i=0; i<3; i++)
          for (int j=0; j<3; j++)
            tmp[i][j] = derivativeRotation[i][j][alpha];
        auto tmp2 = rt * tmp;
        Ke += GFE::dyadicProduct(SkewMatrix<field_type,3>(tmp2).axial(), aContravariant[alpha]);
      }

      //////////////////////////////////////////////////////////
      //   Compute the actual density
      //////////////////////////////////////////////////////////

      // The membrane density
      field_type density = 0;
      if (useAlternativeEnergyWCoss_)
      {
        density += (thickness - K*Dune::power(thickness,3) / 12.0) * W_Coss(Ee, a, aContravariant[2], mu, lambda)
                   + (Dune::power(thickness,3) / 12.0 - K * Dune::power(thickness,5) / 80.0)*W_Coss(Ee*b + c*Ke, a, aContravariant[2], mu, lambda)
                   + Dune::power(thickness,3) / 6.0 * W_Coss_mixt(Ee, c*Ke*b - 2*H*c*Ke, a, aContravariant[2], mu, lambda)
                   + Dune::power(thickness,5) / 80.0 * W_Coss( (Ee*b + c*Ke)*b, a, aContravariant[2], mu, lambda);
      }
      else
      {
        density += (thickness - K*Dune::power(thickness,3) / 12.0) * W_m(Ee, mu, lambda);
        density += (Dune::power(thickness,3) / 12.0 - K * Dune::power(thickness,5) / 80.0)*W_m(Ee*b + c*Ke, mu, lambda);
        density += Dune::power(thickness,3) / 6.0 * W_mixt(Ee, c*Ke*b - 2*H*c*Ke, mu, lambda);
        density += Dune::power(thickness,5) / 80.0 * W_mp( (Ee*b + c*Ke)*b, mu, lambda);
      }

      // The bending density
      density += (thickness - K*Dune::power(thickness,3) / 12.0) * W_curv(Ke, mu)
                 + (Dune::power(thickness,3) / 12.0 - K * Dune::power(thickness,5) / 80.0)*W_curv(Ke*b, mu)
                 + Dune::power(thickness,5) / 80.0 * W_curv(Ke*b*b, mu);

      return density;
    }

    // Construct a copy of this density but using 'adouble' as the number type
    virtual std::unique_ptr<LocalDensity<ElementOrIntersection,ARigidBodyMotion> > makeActiveDensity() const override
    {
      // TODO: Return a density that is bound to the same object as we are
      return std::make_unique<CosseratShellDensity<ElementOrIntersection,adouble> >(mu_c_,
                                                                                    L_c_,
                                                                                    b1_, b2_, b3_,
                                                                                    useAlternativeEnergyWCoss_,
                                                                                    thicknessF_,
                                                                                    lameF_);
    }

    /** \brief Whether the density depends on the 'value' argument
     *
     * \return The Cosserat shell density depends on the microrotation value,
     * but not on the deformation (only on the deformation gradient).
     */
    virtual bool dependsOnValue(int factor=-1) const override
    {
      return factor != 0;
    }

    /** \brief Whether the density depends on the 'derivative' argument
     *
     * \return Always `true` because the Cosserat shell density depends
     * on the derivatives of both factors: The deformation and the microrotation.
     */
    virtual bool dependsOnDerivative([[maybe_unused]] int factor=-1) const override
    {
      return true;
    }

  private:
    /** \brief The shell thickness as a function*/
    std::function<double(Dune::FieldVector<double,dimWorld>)> thicknessF_;

    /** \brief The Lamé-parameters as a function*/
    std::function<Dune::FieldVector<double,2>(Dune::FieldVector<double,dimWorld>)> lameF_;

    /** \brief Cosserat couple modulus */
    double mu_c_;

    /** \brief Length scale parameter */
    double L_c_;

    /** \brief Curvature parameters */
    double b1_, b2_, b3_;


    /** \brief Whether to use the alternative energy W_Coss from Birsan 2021:
               "Alternative derivation of the higher-order constitutive model for six-parameter elastic shells", equations (119) and (126). */
    bool useAlternativeEnergyWCoss_;

    /** \brief The geometry that this density is defined on */
    std::optional<Geometry> geometry_ = std::nullopt;
  };

}  // namespace Dune::GFE

#endif   //#ifndef DUNE_GFE_DENSITIES_COSSERATSHELLDENSITY_HH
