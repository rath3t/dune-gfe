#ifndef DUNE_GFE_DENSITIES_PLANARCOSSERATSHELLDENSITY_HH
#define DUNE_GFE_DENSITIES_PLANARCOSSERATSHELLDENSITY_HH

#define DONT_USE_CURL

//#define QUADRATIC_2006 //only relevant for gridDim == 3
// Use the formula for the quadratic membrane energy (first equation of (4.4)) in Neff's paper from 2006: A geometrically exact planar Cosserat shell model with microstructure: Existence of minimizers for zero Cosserat couple modulus
// instead of the equation (2.27) of 2019: Reﬁned dimensional reduction for isotropic elastic Cosserat shells with initial curvature

//#define QUADRATIC_MEMBRANE_ENERGY //only relevant for gridDim == 2

#include <dune/common/fmatrix.hh>
#include <dune/common/parametertree.hh>

#include <dune/gfe/tensor3.hh>
#include <dune/gfe/cosseratstrain.hh>
#include <dune/gfe/spaces/orthogonalmatrix.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>

namespace Dune::GFE
{
  template<class Position, class field_type>
  class PlanarCosseratShellDensity final
    : public GFE::LocalDensity<Position, GFE::ProductManifold<RealTuple<field_type,3>,Rotation<field_type,3> > >
  {
  private:
    // PlanarCosseratShellDensity only works for 2d grids
    static_assert(Position::size()==2);
    static constexpr int gridDim = Position::size();

    // The target space with 'adouble' as the number type
    using ATargetSpace = GFE::ProductManifold<RealTuple<adouble,3>,Rotation<adouble,3> >;

#if !defined DONT_USE_CURL
    /** \brief Compute the (row-wise) curl of a matrix R \f$
        \param DR The partial derivatives of the matrix R
     */
    static Dune::FieldMatrix<field_type,dim,dim> curl(const Tensor3<field_type,dim,dim,dim>& DR)
    {
      Dune::FieldMatrix<field_type,dim,dim> result;

      for (int i=0; i<dim; i++) {
        result[i][0] = DR[i][2][1] - DR[i][1][2];
        result[i][1] = DR[i][0][2] - DR[i][2][0];
        result[i][2] = DR[i][1][0] - DR[i][0][1];
      }

      return result;
    }
#endif

    /** \brief The energy \f$ W_{mp}(\overline{U}) \f$, as written in
     * the first equation of (4.4) in Neff's paper from 2006: A geometrically exact planar Cosserat shell model with microstructure: Existence of minimizers for zero Cosserat couple modulus
     * OR: the equation (2.27) of 2019: Reﬁned dimensional reduction for isotropic elastic Cosserat shells with initial curvature
     */
    field_type quadraticMembraneEnergy(const GFE::CosseratStrain<field_type,3,gridDim>& U) const
    {
      FieldMatrix<field_type,3,3> UMinus1 = U.matrix();
      for (int i=0; i<3; i++)
        UMinus1[i][i] -= 1;

      return mu_ * GFE::sym(UMinus1).frobenius_norm2()
             + mu_c_ * GFE::skew(UMinus1).frobenius_norm2()
#ifdef QUADRATIC_2006
             + (mu_*lambda_)/(2*mu_ + lambda_) * Dune::GFE::traceSquared(UMinus1);    // Dune::GFE::traceSquared(UMinus1) = Dune::GFE::traceSquared(Dune::GFE::sym(UMinus1))
#else
             + lambda_/2 * Dune::GFE::traceSquared(UMinus1);    // Dune::GFE::traceSquared(UMinus1) = Dune::GFE::traceSquared(Dune::GFE::sym(UMinus1))
#endif
    }

    /** \brief The energy \f$ W_{mp}(\overline{U}) \f$, as written in
     * the second equation of (4.4) in Neff's paper
     */
    field_type longQuadraticMembraneEnergy(const Dune::GFE::CosseratStrain<field_type,3,gridDim>& U) const
    {
      field_type result = 0;

      // shear-stretch energy
      Dune::FieldMatrix<field_type,2,2> sym2x2;
      for (int i=0; i<2; i++)
        for (int j=0; j<2; j++)
          sym2x2[i][j] = 0.5 * (U.matrix()[i][j] + U.matrix()[j][i]) - (i==j);

      result += mu_ * sym2x2.frobenius_norm2();

      // first order drill energy
      FieldMatrix<field_type,2,2> skew2x2;
      for (int i=0; i<2; i++)
        for (int j=0; j<2; j++)
          skew2x2[i][j] = 0.5 * (U.matrix()[i][j] - U.matrix()[j][i]);

      result += mu_c_ * skew2x2.frobenius_norm2();

      // classical transverse shear energy
      result += kappa_ * (mu_ + mu_c_)/2 * (U.matrix()[2][0]*U.matrix()[2][0] + U.matrix()[2][1]*U.matrix()[2][1]);

      // elongational stretch energy
      result += mu_*lambda_ / (2*mu_ + lambda_) * traceSquared(sym2x2);

      return result;
    }

    /** \brief Energy for large-deformation problems (private communication by Patrizio Neff)
     */
    field_type nonquadraticMembraneEnergy(const Dune::GFE::CosseratStrain<field_type,3,gridDim>& U) const
    {
      Dune::FieldMatrix<field_type,3,3> UMinus1 = U.matrix();
      for (int i=0; i<3; i++)
        UMinus1[i][i] -= 1;

      field_type detU = U.determinant();

      return mu_ * GFE::sym(UMinus1).frobenius_norm2() + mu_c_ * GFE::skew(UMinus1).frobenius_norm2()
             + (mu_*lambda_)/(2*mu_ + lambda_) * 0.5 * ((detU-1)*(detU-1) + (1.0/detU -1)*(1.0/detU -1));
    }

    /** \brief The energy \f$ W_{mp}(\overline{U}) \f$, as written in
     * the second equation of (4.4) in Neff's paper
     */
    field_type longNonquadraticMembraneEnergy(const Dune::GFE::CosseratStrain<field_type,3,gridDim>& U) const
    {
      field_type result = 0;

      // shear-stretch energy
      FieldMatrix<field_type,2,2> sym2x2;
      for (int i=0; i<2; i++)
        for (int j=0; j<2; j++)
          sym2x2[i][j] = 0.5 * (U.matrix()[i][j] + U.matrix()[j][i]) - (i==j);

      result += mu_ * sym2x2.frobenius_norm2();

      // first order drill energy
      Dune::FieldMatrix<field_type,2,2> skew2x2;
      for (int i=0; i<2; i++)
        for (int j=0; j<2; j++)
          skew2x2[i][j] = 0.5 * (U.matrix()[i][j] - U.matrix()[j][i]);

      result += mu_c_ * skew2x2.frobenius_norm2();


      // classical transverse shear energy
      result += kappa_ * (mu_ + mu_c_)/2 * (U.matrix()[2][0]*U.matrix()[2][0] + U.matrix()[2][1]*U.matrix()[2][1]);

      // elongational stretch energy
      field_type detU = U.determinant();
      result += (mu_*lambda_)/(2*mu_ + lambda_) * 0.5 * ((detU-1)*(detU-1) + (1.0/detU -1)*(1.0/detU -1));

      return result;
    }

    field_type curvatureEnergy(const Tensor3<field_type,3,3,gridDim>& DR) const
    {
      using std::pow;
#ifdef DONT_USE_CURL
      return mu_ * pow(L_c_ * L_c_ * DR.frobenius_norm2(),q_/2.0);
#else
      return mu_ * pow(L_c_ * L_c_ * curl(DR).frobenius_norm2(),q_/2.0);
#endif
    }

    field_type bendingEnergy(const Dune::FieldMatrix<field_type,3,3>& R, const Tensor3<field_type,3,3,gridDim>& DR) const
    {
      // left-multiply the derivative of the third director (in DR[][2][]) with R^T
      FieldMatrix<field_type,3,3> RT_DR3(0);
      for (int i=0; i<3; i++)
        for (int j=0; j<gridDim; j++)
          for (int k=0; k<3; k++)
            RT_DR3[i][j] += R[k][i] * DR[k][2][j];

      return mu_ * GFE::sym(RT_DR3).frobenius_norm2()
             + mu_c_ * GFE::skew(RT_DR3).frobenius_norm2()
             + mu_*lambda_/(2*mu_+lambda_) * GFE::traceSquared(RT_DR3);
    }

  public:

    /** \brief Constructor with a set of material parameters
     * \param parameters The material parameters
     */
    PlanarCosseratShellDensity(const Dune::ParameterTree& parameters)
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

      // Curvature exponent
      q_ = parameters.template get<double>("q");

      // Shear correction factor
      kappa_ = parameters.template get<double>("kappa");
    }

    /** \brief Constructor with a set of material parameters
     */
    PlanarCosseratShellDensity(double thickness, double mu, double lambda, double mu_c, double L_c, double q, const double kappa)
      : thickness_(thickness), mu_(mu), lambda_(lambda), mu_c_(mu_c), L_c_(L_c), q_(q), kappa_(kappa)

    {}

    /** \brief Evaluate the density
     */
    virtual field_type operator() (const Position& x,
                                   const typename GFE::ProductManifold<RealTuple<field_type,3>,Rotation<field_type,3> >::CoordinateType& value,
                                   const FieldMatrix<field_type,7,gridDim>& derivative) const override
    {
      field_type density = 0;

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

#ifdef QUADRATIC_MEMBRANE_ENERGY
      //density += thickness_ * quadraticMembraneEnergy(U);
      density += thickness_ * longQuadraticMembraneEnergy(U);
#else
      density += thickness_ * nonquadraticMembraneEnergy(U);
#endif
      density += thickness_ * curvatureEnergy(DR);
      density += std::pow(thickness_,3) / 12.0 * bendingEnergy(R,DR);

      return density;
    }

    // Construct a copy of this density but using 'adouble' as the number type
    virtual std::unique_ptr<LocalDensity<Position,ATargetSpace> > makeActiveDensity() const override
    {
      return std::make_unique<PlanarCosseratShellDensity<Position,adouble> >(thickness_, mu_, lambda_, mu_c_, L_c_, q_, kappa_);
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
  };

}

#endif   //#ifndef DUNE_GFE_DENSITIES_PLANARCOSSERATSHELLDENSITY_HH
