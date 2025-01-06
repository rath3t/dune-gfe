#ifndef DUNE_GFE_DENSITIES_BULKCOSSERATDENSITY_HH
#define DUNE_GFE_DENSITIES_BULKCOSSERATDENSITY_HH

#include <dune/common/fmatrix.hh>

#include <dune/gfe/cosseratstrain.hh>
#include <dune/gfe/linearalgebra.hh>
#include <dune/gfe/densities/localdensity.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>

namespace Dune::GFE {

  template<class Element, class field_type>
  class BulkCosseratDensity final
    : public GFE::LocalDensity<Element, GFE::ProductManifold<RealTuple<field_type,3>,Rotation<field_type,3> > >
  {
  private:
    using LocalCoordinate = typename Element::Geometry::LocalCoordinate;

    // BulkCosseratDensity only works for 3d->3d problems
    static_assert(LocalCoordinate::size()==3);
    static const int gridDim = LocalCoordinate::size();
    static const int embeddedDim = Rotation<field_type,gridDim>::embeddedDim;

    // The target space with 'adouble' as the number type
    using ATargetSpace = GFE::ProductManifold<RealTuple<adouble,3>,Rotation<adouble,3> >;

    static FieldMatrix<field_type,gridDim,gridDim> curl(const Tensor3<field_type,gridDim,gridDim,gridDim>& DR)
    {
      FieldMatrix<field_type,gridDim,gridDim> result;

      for (int i=0; i<gridDim; i++) {
        result[i][0] = DR[i][2][1] - DR[i][1][2];
        result[i][1] = DR[i][0][2] - DR[i][2][0];
        result[i][2] = DR[i][1][0] - DR[i][0][1];
      }

      return result;
    }

    /** \brief Compute the wryness tensor from the microrotation and its derivative
     *
     * The wryness tensor implemented here is defined in Eq. (19) in:
     *
     *   M. Bîrsan, I.D. Ghiba, R.J. Martin, P. Neff: "Refined dimensional reduction
     *   for isotropic elastic Cosserat shells with initial curvature",
     *   Mathematics and Mechanics of Solids, 24(12), 2019
     *   DOI: https://doi.org/10.1177/1081286519856061
     *
     */
    static FieldMatrix<field_type,3,3> wryness(const FieldMatrix<field_type,gridDim,gridDim>& R,
                                               const Tensor3<field_type,3,3,3>& DR)
    {
      FieldMatrix<field_type,3,3> dRx1(0);  //Derivative of R with respect to x1
      FieldMatrix<field_type,3,3> dRx2(0);  //Derivative of R with respect to x2
      FieldMatrix<field_type,3,3> dRx3(0);  //Derivative of R with respect to x3
      for (int i=0; i<3; i++)
        for (int j=0; j<3; j++) {
          dRx1[i][j] = DR[i][j][0];
          dRx2[i][j] = DR[i][j][1];
          dRx3[i][j] = DR[i][j][2];
        }

      FieldMatrix<field_type,3,3> wrynessTensor(0);

      auto axialVectorx1 = SkewMatrix<field_type,3>(transpose(R)*dRx1).axial();
      auto axialVectorx2 = SkewMatrix<field_type,3>(transpose(R)*dRx2).axial();
      auto axialVectorx3 = SkewMatrix<field_type,3>(transpose(R)*dRx3).axial();
      for (int i=0; i<3; i++) {
        wrynessTensor[i][0] = axialVectorx1[i];
        wrynessTensor[i][1] = axialVectorx2[i];
        wrynessTensor[i][2] = axialVectorx3[i];
      }

      return wrynessTensor;
    }

  public:

    /** \brief The different types of curvature tensors */
    enum CurvatureType {Norm, Curl, Wryness};

    /** \brief Constructor with a set of material parameters
     * \param parameters The material parameters
     */
    BulkCosseratDensity(const ParameterTree& parameters)
    {
      mu_ = parameters.template get<double>("mu");
      lambda_ = parameters.template get<double>("lambda");

      // Cosserat couple modulus
      mu_c_ = parameters.template get<double>("mu_c");

      // Length scale parameter
      L_c_ = parameters.template get<double>("L_c");

      // Curvature type
      const auto& curvatureString = parameters.template get<std::string>("curvatureType");
      if (curvatureString=="norm")
        curvatureType_ = CurvatureType::Norm;
      else if (curvatureString=="curl")
        curvatureType_ = CurvatureType::Curl;
      else if (curvatureString=="wryness")
        curvatureType_ = CurvatureType::Wryness;
      else
        DUNE_THROW(NotImplemented, "Curvature type '" << curvatureString << "' is not implemented!");

      // Curvature exponent
      q_ = parameters.template get<double>("q");

      // Curvature parameters
      b1_ = parameters.template get<double>("b1");
      b2_ = parameters.template get<double>("b2");
      b3_ = parameters.template get<double>("b3");
    }

    /** \brief Constructor with a set of material parameters
     */
    BulkCosseratDensity(double mu, double lambda, double mu_c, double L_c, CurvatureType curvatureType, double q, const std::array<double,3>& b)
      : mu_(mu), lambda_(lambda), mu_c_(mu_c), L_c_(L_c), curvatureType_(curvatureType), q_(q), b1_(b[0]), b2_(b[1]), b3_(b[2])
    {}

    /** \brief The energy \f$ W_{mp}(\overline{U}) \f$, as written in
     * the first equation of (4.4) in Neff's paper from 2006: A geometrically exact planar Cosserat shell model with microstructure: Existence of minimizers for zero Cosserat couple modulus
     * OR: the equation (2.27) of 2019: Reﬁned dimensional reduction for isotropic elastic Cosserat shells with initial curvature
     */
    field_type quadraticEnergy(const GFE::CosseratStrain<field_type,3,gridDim>& U) const
    {
      FieldMatrix<field_type,3,3> UMinus1 = U.matrix();
      for (int i=0; i<gridDim; i++)
        UMinus1[i][i] -= 1;

#ifdef QUADRATIC_2006
      field_type materialFactor = (mu_*lambda_)/(2*mu_ + lambda_);
#else
      field_type materialFactor = lambda_/2;
#endif

      // In various papers the last term is traceSquared(sym(UMinus1)),
      // but UMinus1 is symmetric and hence
      //   traceSquared(UMinus1) = traceSquared(sym(UMinus1)).
      return mu_ * GFE::sym(UMinus1).frobenius_norm2()
             + mu_c_ * GFE::skew(UMinus1).frobenius_norm2()
             + materialFactor * GFE::traceSquared(UMinus1);
    }

    /** \brief Evaluate the density
     */
    virtual field_type operator() (const LocalCoordinate& x,
                                   const typename GFE::ProductManifold<RealTuple<field_type,3>,Rotation<field_type,3> >::CoordinateType& value,
                                   const FieldMatrix<field_type,7,gridDim>& derivative) const override
    {
      field_type strainEnergyDensity = 0;

      /////////////////////////////////////////////////////////
      //  Extract value and derivatives of the factor spaces
      /////////////////////////////////////////////////////////

      Rotation<field_type,3> rotationValue(FieldVector<field_type,4>{value[3], value[4], value[5], value[6]});

      FieldMatrix<field_type,3,gridDim> deformationDerivative = {derivative[0], derivative[1], derivative[2]};
      FieldMatrix<field_type,4,gridDim> orientationDerivative = {derivative[3], derivative[4], derivative[5], derivative[6]};

      /////////////////////////////////////////////////////////
      // compute U, the Cosserat strain
      /////////////////////////////////////////////////////////

      FieldMatrix<field_type,3,3> R;
      rotationValue.matrix(R);

      GFE::CosseratStrain<field_type,3,gridDim> U(deformationDerivative,R);

      /////////////////////////////////////////////////////////////////////
      //  Transfer the derivative of the rotation into matrix coordinates
      ////////////////////////////////////////////////////////////////////

      Tensor3<field_type,3,3,gridDim> DR = rotationValue.quaternionTangentToMatrixTangent(orientationDerivative);

      // Compute the bending energy density
      strainEnergyDensity = quadraticEnergy(U);

      // Compute the curvature energy density
      field_type norm = 0;

      switch (curvatureType_)
      {
      case CurvatureType::Norm :
      {
        /* This is a simplified version of the curvature energy that appears in
         *
         *   P. Neff: "A geometrically exact Cosserat shell-model including size effects,
         *   avoiding degeneracy in the thin shell limit. Part I: Formal dimensional reduction
         *   for elastic plates and existence of minimizers for positive Cosserat couple modulus"
         *   Continuum Mech. Thermodyn. (2004) 16: 577–628
         *   DOI: 10.1007/s00161-004-0182-4
         *
         * The curvature energy given there is much more complicated, but it
         * does use the full norm of DR to define the curvature.
         */
        norm = DR.frobenius_norm2();
        break;
      }
      case CurvatureType::Curl :
      {
        auto curvatureTensor = curl(DR);

        // The choice for the Frobenius norm here is b1=b2=1 and b3 = 1/3
        norm = (b1_ * GFE::dev(GFE::sym(curvatureTensor)).frobenius_norm2()
                + b2_ * GFE::skew(curvatureTensor).frobenius_norm2()
                + b3_ * GFE::traceSquared(curvatureTensor));
        break;
      }
      case CurvatureType::Wryness :
      {
        auto curvatureTensor = wryness(R,DR);

        // The choice for the Frobenius norm here is b1=b2=1 and b3 = 1/3
        norm = (b1_ * GFE::dev(GFE::sym(curvatureTensor)).frobenius_norm2()
                + b2_ * GFE::skew(curvatureTensor).frobenius_norm2()
                + b3_ * GFE::traceSquared(curvatureTensor));
        break;
      }
      }

      strainEnergyDensity += mu_ * std::pow(L_c_ * L_c_ * norm,q_/2.0);

      return strainEnergyDensity;
    }

    // Construct a copy of this density but using 'adouble' as the number type
    virtual std::unique_ptr<LocalDensity<Element,ATargetSpace> > makeActiveDensity() const
    {
      // curvatureType_ is a local enum type, and therefore its type changes
      // together with the type of the surrounding class.
      auto activeCurvatureType = static_cast<typename BulkCosseratDensity<Element,adouble>::CurvatureType>(curvatureType_);

      auto result = std::make_unique<BulkCosseratDensity<Element,adouble> >(mu_, lambda_, mu_c_, L_c_, activeCurvatureType, q_, std::array<double,3>{b1_, b2_, b3_});

      if (this->elementOrIntersection_)
        result->bind(*this->elementOrIntersection_);
      return result;
    }

    /** \brief The density depends on the microrotation value, but not on the deformation
     */
    virtual bool dependsOnValue(int factor=-1) const override
    {
      return factor==-1 || factor==1;
    }

    /** \brief The density depends on the deformation gradient and on the microrotation gradient
     */
    virtual bool dependsOnDerivative([[maybe_unused]] int factor=-1) const override
    {
      return true;
    }

    /** \brief Lamé constants */
    double mu_, lambda_;

    /** \brief Cosserat couple modulus */
    double mu_c_;

    /** \brief Length scale parameter */
    double L_c_;

    /** \brief The precise way to compute the curvature tensor */
    enum CurvatureType curvatureType_;

    /** \brief Curvature exponent */
    double q_;

    /** \brief Curvature parameters */
    double b1_, b2_, b3_;

  };

} // namespace GFE

#endif   //#ifndef DUNE_GFE_BULKCOSSERATDENSITY_HH
