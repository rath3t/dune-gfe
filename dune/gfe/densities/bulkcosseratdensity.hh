#ifndef DUNE_GFE_DENSITIES_BULKCOSSERATDENSITY_HH
#define DUNE_GFE_DENSITIES_BULKCOSSERATDENSITY_HH

#define DONT_USE_CURL
#define CURVATURE_WITH_WRYNESS

#include <dune/common/fmatrix.hh>

#include <dune/gfe/cosseratstrain.hh>
#include <dune/gfe/linearalgebra.hh>
#include <dune/gfe/densities/localdensity.hh>

namespace Dune::GFE {

  template<class field_type = double, class ctype = double>
  class BulkCosseratDensity final
    : public GFE::LocalDensity<3,field_type,ctype>
  {
  private:
    // gridDim = 3 - this can be hardwired here, because BulkCosseratDensity only works for 3d->3d problems
    static const int gridDim = 3;
    static const int embeddedDim = Rotation<field_type,gridDim>::embeddedDim;
    using OrientationDerivativeType = FieldMatrix<field_type, embeddedDim, gridDim>;

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

  public:

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

      // Curvature exponent
      q_ = parameters.template get<double>("q");

      // Curvature parameters
      b1_ = parameters.template get<double>("b1");
      b2_ = parameters.template get<double>("b2");
      b3_ = parameters.template get<double>("b3");
    }

    /** \brief The energy \f$ W_{mp}(\overline{U}) \f$, as written in
     * the first equation of (4.4) in Neff's paper from 2006: A geometrically exact planar Cosserat shell model with microstructure: Existence of minimizers for zero Cosserat couple modulus
     * OR: the equation (2.27) of 2019: Reﬁned dimensional reduction for isotropic elastic Cosserat shells with initial curvature
     */
    field_type quadraticEnergy(const GFE::CosseratStrain<field_type,3,gridDim>& U) const
    {
      FieldMatrix<field_type,3,3> UMinus1 = U.matrix();
      for (int i=0; i<gridDim; i++)
        UMinus1[i][i] -= 1;

      return mu_ * GFE::sym(UMinus1).frobenius_norm2()
             + mu_c_ * GFE::skew(UMinus1).frobenius_norm2()
#ifdef QUADRATIC_2006
             + (mu_*lambda_)/(2*mu_ + lambda_) * GFE::traceSquared(UMinus1);  // GFE::traceSquared(UMinus1) = GFE::traceSquared(GFE::sym(UMinus1))
#else
             + lambda_/2 * GFE::traceSquared(UMinus1);  // GFE::traceSquared(UMinus1) = GFE::traceSquared(GFE::sym(UMinus1))
#endif
    }

    field_type curvatureWithWryness(const FieldMatrix<field_type,gridDim,gridDim>& R, const Tensor3<field_type,3,3,3>& DR) const
    {
      // construct Wryness tensor \Gamma as in "Refined dimensional reduction for isotropic elastic Cosserat shells with initial curvature"
      FieldMatrix<field_type,3,3> dRx1(0); //Derivative of R with respect to x1
      FieldMatrix<field_type,3,3> dRx2(0); //Derivative of R with respect to x2
      FieldMatrix<field_type,3,3> dRx3(0); //Derivative of R with respect to x3
      for (int i=0; i<3; i++)
        for (int j=0; j<3; j++) {
          dRx1[i][j] = DR[i][j][0];
          dRx2[i][j] = DR[i][j][1];
          dRx3[i][j] = DR[i][j][2];
        }

      FieldMatrix<field_type,3,3> wryness(0);

      auto axialVectorx1 = SkewMatrix<field_type,3>(transpose(R)*dRx1).axial();
      auto axialVectorx2 = SkewMatrix<field_type,3>(transpose(R)*dRx2).axial();
      auto axialVectorx3 = SkewMatrix<field_type,3>(transpose(R)*dRx3).axial();
      for (int i=0; i<3; i++) {
        wryness[i][0] = axialVectorx1[i];
        wryness[i][1] = axialVectorx2[i];
        wryness[i][2] = axialVectorx3[i];
      }

      // The choice for the Frobenius norm here is b1=b2=1 and b3 = 1/3
      return mu_ * L_c_ * L_c_ * (b1_ * GFE::dev(GFE::sym(wryness)).frobenius_norm2()
                                  + b2_ * GFE::skew(wryness).frobenius_norm2() + b3_ * GFE::traceSquared(wryness));
    }

    /** \brief Evaluation with the current position, the deformation value and its derivative, the orientation value and its derivative
     *
     * \param x The current position
     * \param deformationValue The deformation at the current position
     * \param deformationDerivative The derivative of the deformation at the current position
     * \param orientationValue The orientation at the current position
     * \param orientationDerivative The derivative of the orientation at the current position
     */
    field_type operator() (const FieldVector<ctype,gridDim>& x,
                           const RealTuple<field_type,gridDim>& deformationValue,
                           const FieldMatrix<field_type,gridDim,gridDim>& deformationDerivative,
                           const Rotation<field_type,gridDim>& orientationValue,
                           const OrientationDerivativeType& orientationDerivative) const
    {
      field_type strainEnergyDensity = 0;
      /////////////////////////////////////////////////////////
      // compute U, the Cosserat strain
      /////////////////////////////////////////////////////////

      FieldMatrix<field_type,gridDim,gridDim> R;
      orientationValue.matrix(R);

      GFE::CosseratStrain<field_type,gridDim,gridDim> U(deformationDerivative,R);

      /////////////////////////////////////////////////////////////////////
      //  Transfer the derivative of the rotation into matrix coordinates
      ////////////////////////////////////////////////////////////////////

      Tensor3<field_type,3,3,gridDim> DR = orientationValue.quaternionTangentToMatrixTangent(orientationDerivative);
      strainEnergyDensity = quadraticEnergy(U);

    #ifdef CURVATURE_WITH_WRYNESS
      strainEnergyDensity += curvatureWithWryness(R,DR);
    #else

    #ifdef DONT_USE_CURL
      auto argument = DR;
    #else
      auto argument = curl(DR);
    #endif

      auto norm = (b1_ * GFE::dev(GFE::sym(argument)).frobenius_norm2()
                   + b2_ * GFE::skew(argument).frobenius_norm2() + b3_ * GFE::traceSquared(argument));

      strainEnergyDensity += mu_ * std::pow(L_c_ * L_c_ * norm,q_/2.0);
    #endif

      return strainEnergyDensity;
    }

    /** \brief Lame constants */
    double mu_, lambda_;

    /** \brief Cosserat couple modulus, preferably 0 */
    double mu_c_;

    /** \brief Length scale parameter */
    double L_c_;

    /** \brief Curvature exponent */
    double q_;

    /** \brief Curvature parameters */
    double b1_, b2_, b3_;

  };

} // namespace GFE

#endif   //#ifndef DUNE_GFE_BULKCOSSERATDENSITY_HH
