#ifndef DUNE_GFE_DENSITIES_LOCALDENSITY_HH
#define DUNE_GFE_DENSITIES_LOCALDENSITY_HH

#include <optional>

#include <dune/common/fmatrix.hh>

#include <adolc/adalloc.h>
#include <adolc/adolc.h>

#include <dune/istl/matrix.hh>

namespace Dune::GFE
{

  /** \brief A base class for energy densities to be evaluated in an integral energy
   *
   * \tparam ElementOrIntersection The domain of the density.
   *   Can be either a grid element (i.e., a codimension-0 Entity)
   *   or an Intersection.
   * \tparam TargetSpace Type for the function value
   */
  template<class ElementOrIntersection, class TargetSpace>
  class LocalDensity
  {
    using Geometry = typename ElementOrIntersection::Geometry;
    using LocalCoordinate = typename Geometry::LocalCoordinate;

    using field_type = typename TargetSpace::field_type;
    using ATargetSpace = typename TargetSpace::template rebind<adouble>::other;
    using DerivativeType = FieldMatrix<field_type,TargetSpace::EmbeddedTangentVector::dimension,LocalCoordinate::size ()>;

    // Number of independent variables
    static constexpr auto m = TargetSpace::EmbeddedTangentVector::dimension + DerivativeType::rows*DerivativeType::cols;

    // Number of directions for the automatic differentiation
    static constexpr auto q = m;

    // Dense matrix containing the directions that the derivatives of the density
    // will be computed in.
    double*** densityTangent_;

    double*** Yppp_;   /* results of hov_wk_forward  */
    double*** Zppp_;   /* result of hos_ov_reverse */

  protected:
    /** \brief The element or intersection that this density is defined on */
    std::optional<ElementOrIntersection> elementOrIntersection_ = std::nullopt;

  public:

    LocalDensity()
    {
      densityTangent_ = myalloc3(m,m,1);

      // Initialize directions field
      for (int j=0; j<m; j++)
        for (int i=0; i<m; i++)
          densityTangent_[i][j][0] = i == j ? 1.0 : 0.0;

      Yppp_ = myalloc3(1,q,1);   /* results of hov_wk_forward  */
      Zppp_ = myalloc3(q,m,2);   /* result of hos_ov_reverse */

    }

    ~LocalDensity()
    {
      myfree3(densityTangent_);
      myfree3(Yppp_);
      myfree3(Zppp_);
    }

    /** \brief Bind the density to a `ElementOrIntersection` object
     *
     * Stores a copy of the `ElementOrIntersection`'s geometry.
     **/
    virtual void bind(const ElementOrIntersection& elementOrIntersection)
    {
      elementOrIntersection_.emplace(elementOrIntersection);
    }

    /** \brief Evaluate the density for a given value and first derivative
     *
     * \param x The current position
     * \param value The value of the integrand at x
     * \param derivative The derivative of the integrand at x
     */
    virtual field_type operator() (const LocalCoordinate& x,
                                   const typename TargetSpace::CoordinateType& value,
                                   const DerivativeType& derivative) const = 0;

    /** \brief Compute value, first and second derivatives of the density
     *
     * The default implementation here uses ADOL-C for this.
     */
    virtual void derivatives(const LocalCoordinate& x,
                             const TargetSpace& value,
                             const DerivativeType& derivative,
                             field_type& densityValue,
                             std::vector<double>& densityGradient,
                             Matrix<double>& densityHessian) const
    {
      // TODO: Pick the tape number in a smarter way!
      short tapeNumber = 17;

      // The 'derivatives' method shall never be called if field_type is adouble.  However, we cannot avoid
      // that it is instantiated for field_type==adouble.  Therefore, handle this case.
      if constexpr (std::is_same_v<field_type,adouble>)
      {
        std::terminate();
      }
      else
      {
        //////////////////////////////////////////////////////////////////////////////////
        //  Tape the evaluation of the density
        //////////////////////////////////////////////////////////////////////////////////

        std::vector<double> xp(m);

        typename ATargetSpace::CoordinateType aInterpolationValueGlobalCoordinates;

        using ATargetSpaceDerivativeType = FieldMatrix<adouble, DerivativeType::rows, DerivativeType::cols>;
        ATargetSpaceDerivativeType aInterpolationDerivative;

        trace_on(tapeNumber);

        int idx=0;

        if constexpr (not Impl::LocalEnergyTypes<TargetSpace>::isProductManifold)
        {
          for (size_t i = 0; i<TargetSpace::EmbeddedTangentVector::dimension; i++)
          {
            aInterpolationValueGlobalCoordinates[i] <<= value.globalCoordinates()[i];
            xp[idx++] = value.globalCoordinates()[i];
          }

          for (size_t i = 0; i<derivative.rows; i++)
            for (size_t j = 0; j<derivative.cols; j++) {
              aInterpolationDerivative[i][j] <<= derivative[i][j];
              xp[idx++] = derivative[i][j];
            }

          auto activeLocalDensity = makeActiveDensity();

          adouble density = (*activeLocalDensity)(x, aInterpolationValueGlobalCoordinates, aInterpolationDerivative);

          density >>= densityValue;
        }
        else
        {
          using namespace Dune::Indices;

          // Attention: Keep the order of the 4 for-loops below as they are!
          // This is the order of the dependent variables, the derivatives are ordered in the same way!
          typename std::tuple_element_t<0,ATargetSpace>::CoordinateType aDeformationValueGlobalCoordinates;
          typename std::tuple_element_t<1,ATargetSpace>::CoordinateType aOrientationValueGlobalCoordinates;

          constexpr auto embeddedBlocksize0 = std::tuple_element_t<0,ATargetSpace>::CoordinateType::dimension;
          constexpr auto embeddedBlocksize1 = std::tuple_element_t<1,ATargetSpace>::CoordinateType::dimension;

          ATargetSpace aInterpolationValue;

          for (size_t i = 0; i<embeddedBlocksize0; i++)
          {
            aDeformationValueGlobalCoordinates[i] <<= value[_0].globalCoordinates()[i];
            xp[idx++] = value[_0].globalCoordinates()[i];
          }
          aInterpolationValue[_0] = aDeformationValueGlobalCoordinates;

          for (size_t i = 0; i<embeddedBlocksize0; i++)
            for (size_t j = 0; j<derivative.cols; j++) {
              aInterpolationDerivative[i][j] <<= derivative[i][j];
              xp[idx++] = derivative[i][j];
            }

          for (size_t i = 0; i<embeddedBlocksize1; i++)
          {
            aOrientationValueGlobalCoordinates[i] <<= value[_1].globalCoordinates()[i];
            xp[idx++] = value[_1].globalCoordinates()[i];
          }
          aInterpolationValue[_1] = aOrientationValueGlobalCoordinates;

          for (size_t i = 0; i<embeddedBlocksize1; i++)
            for (size_t j = 0; j<derivative.cols; j++) {
              aInterpolationDerivative[embeddedBlocksize0+i][j] <<= derivative[embeddedBlocksize0+i][j];
              xp[idx++] = derivative[embeddedBlocksize0+i][j];
            }

          auto activeLocalDensity = makeActiveDensity();

          adouble density = (*activeLocalDensity)(x, aInterpolationValue.globalCoordinates(), aInterpolationDerivative);

          density >>= densityValue;
        }

        trace_off();


        //////////////////////////////////////////////////////////////////////////////////
        //  Compute the derivatives
        //////////////////////////////////////////////////////////////////////////////////

        double Upp[2] = {1.0, 0.0};
        double* UppPtr[2] = {Upp, Upp+1};

        // Compute first derivatives in forward mode
        //double value;   // The density value, as computed by hov_wk_forward
        hov_wk_forward(tapeNumber,
                       1,
                       m,
                       1,
                       2,
                       q,
                       xp.data(),
                       densityTangent_,
                       &densityValue,          // The function value obtained by evaluating the tape
                       Yppp_);

        for (int i=0; i<m; i++)
          densityGradient[i] = Yppp_[0][i][0];

        // Compute second derivatives in reverse mode
        hos_ov_reverse(tapeNumber, 1, m, 1, q, UppPtr, Zppp_);

        for (int i = 0; i < q; ++i)
          for (int j = 0; j < m; ++j)
            densityHessian[j][i] = Zppp_[i][j][1];
      }
    }

    // Construct a copy of this density but using 'adouble' as the number type
    virtual std::unique_ptr<LocalDensity<ElementOrIntersection,ATargetSpace> > makeActiveDensity() const = 0;

    /** \brief Whether the density depends on the 'value' parameter
     *
     * If TargetSpace is a ProductManifold, then this method returns the information
     * for one factor space only.
     *
     * \param factor The factor space that is being asked about.
     *   The default value -1 means: Does the density depend on the value
     *   of any of the factors?
     */
    virtual bool dependsOnValue(int factor=-1) const = 0;

    /** \brief Whether the density depends on the 'derivative' parameter
     *
     * If TargetSpace is a ProductManifold, then this method returns the information
     * for one factor space only.
     *
     * \param factor The factor space that is being asked about
     *   The default value -1 means: Does the density depend on the derivative
     *   of any of the factors?
     */
    virtual bool dependsOnDerivative(int factor=-1) const = 0;
  };

}  // namespace Dune::GFE

#endif  // DUNE_GFE_DENSITIES_LOCALDENSITY_HH
