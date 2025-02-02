#ifndef DUNE_GFE_FUNCTIONS_INTERPOLATIONDERIVATIVES_HH
#define DUNE_GFE_FUNCTIONS_INTERPOLATIONDERIVATIVES_HH

// Includes for the ADOL-C automatic differentiation library
#include <adolc/adolc.h>

#include <dune/matrix-vector/transpose.hh>

#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/gfe/functions/localgeodesicfefunction.hh>
#include <dune/gfe/functions/localprojectedfefunction.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/unitvector.hh>


namespace Dune::GFE
{
  /** \brief Compute derivatives of GFE interpolation with respect to the coefficients
   *
   * \tparam LocalInterpolationRule The class that implements the interpolation from a set of coefficients
   *
   */
  template <typename LocalInterpolationRule>
  class InterpolationDerivatives
  {
    using TargetSpace = typename LocalInterpolationRule::TargetSpace;

    constexpr static auto blocksize = TargetSpace::TangentVector::dimension;
    constexpr static auto embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension;

    //////////////////////////////////////////////////////////////////////
    //  Data members
    //////////////////////////////////////////////////////////////////////

    const LocalInterpolationRule& localInterpolationRule_;

    // Whether derivatives of the interpolation value are to be computed
    const bool doValue_;

    // Whether derivatives of the derivative of the interpolation
    // with respect to space are to be computed
    const bool doDerivative_;

    // TODO: Don't hardcode FieldMatrix
    std::vector<FieldMatrix<double,blocksize,embeddedBlocksize> > orthonormalFrames_;

    // The coefficient values where we are evaluating the derivatives.
    // In flattened format, because ADOL-C can accept only that.
    std::vector<double> localConfigurationFlat_;

    // The tangent vectors
    double*** Xppp_;

    // Results of hov_wk_forward
    double*   y_;      // Function values
    double*** Yppp_;   // First derivatives
    double*** Zppp_;   // result of Up x H x XPPP
    double**  Upp_;    // Vector on left-hand side

    size_t numberOfTangents() const
    {
      return blocksize * localInterpolationRule_.size();
    }

    /** \brief Expose a two-index window from a three-index object
     *
     * \tparam rows Number of rows of the window
     * \tparam cols Number of cols of the window
     * \tparam thirdIndex The value of the third index
     */
    template <size_t rows, size_t cols, size_t thirdIndex>
    class SubmatrixView
    {
    public:
      SubmatrixView(double*** data, size_t blockRow, size_t blockCol)
        : data_(data), blockRow_(blockRow), blockCol_(blockCol)
      {}

      /** \brief Access to scalar entries */
      double& operator()(size_t row, size_t col)
      {
        return data_[blockRow_*rows+row][blockCol_*cols+col][thirdIndex];
      }

      /** \brief Const access to scalar entries */
      const double& operator()(size_t row, size_t col) const
      {
        return data_[blockRow_*rows+row][blockCol_*cols+col][thirdIndex];
      }

      /** \brief Assignment from a transposed matrix */
      void transposedAssign(FieldMatrix<double,cols,rows> other)
      {
        for (size_t i=0; i<rows; ++i)
          for (size_t j=0; j<cols; ++j)
            (*this)(i,j) = other[j][i];
      }

      /** \brief Matrix multiplication from the right with a transposed matrix
       */
      FieldMatrix<double,rows,rows> multiplyTransposed(const FieldMatrix<double,rows,cols>& other)
      {
        FieldMatrix<double,rows,rows> result;

        for (size_t i=0; i<rows; ++i)
          for (size_t j=0; j<rows; ++j)
          {
            result[i][j] = 0;
            for (size_t k=0; k<cols; ++k)
              result[i][j] += (*this)(i,k) * other[j][k];
          }

        return result;
      }

    private:
      double*** data_;
      const size_t blockRow_;
      const size_t blockCol_;
    };

  public:

    InterpolationDerivatives(const LocalInterpolationRule& localInterpolationRule,
                             bool doValue,
                             bool doDerivative)
      : localInterpolationRule_(localInterpolationRule)
      , doValue_(doValue)
      , doDerivative_(doDerivative)
    {
      // Precompute the orthonormal frames
      orthonormalFrames_.resize(localInterpolationRule_.size());
      for (size_t i=0; i<localInterpolationRule_.size(); ++i)
        orthonormalFrames_[i] = localInterpolationRule_.coefficient(i).orthonormalFrame();

      // Construct vector containing the configuration
      localConfigurationFlat_.resize(localInterpolationRule_.size()*embeddedBlocksize);

      for (size_t i=0; i<localInterpolationRule_.size(); i++)
        for (size_t j=0; j<embeddedBlocksize; j++)
          localConfigurationFlat_[i*embeddedBlocksize+j] = localInterpolationRule_.coefficient(i).globalCoordinates()[j];


      // Various arrays for ADOL-C
      const int d = 1;   // TODO: What is this?  (ADOL-C calls this "highest derivative degree")

      // Number of dependent variables of GFE interpolation
      const size_t m = embeddedBlocksize + LocalInterpolationRule::DerivativeType::rows * LocalInterpolationRule::DerivativeType::cols;

      // Number of independent variables of GFE interpolation
      const size_t n = localInterpolationRule_.size() * embeddedBlocksize;

      // Set up the tangent vectors for ADOL-C
      // They are given by tangent vectors of TargetSpace.
      Xppp_ = myalloc3(n,numberOfTangents(),1);   // The tangent vectors

      for (size_t i=0; i<n; i++)
        for (size_t j=0; j<numberOfTangents(); j++)
          Xppp_[i][j][0] = 0;

      for (size_t i=0; i<orthonormalFrames_.size(); ++i)
      {
        SubmatrixView<embeddedBlocksize,blocksize,0> view(Xppp_,i,i);
        view.transposedAssign(orthonormalFrames_[i]);
      }

      // Results of hov_wk_forward
      y_ = myalloc1(m);                               // Function values
      Yppp_ = myalloc3(m,numberOfTangents(),1);   // First derivatives

      Zppp_ = myalloc3(numberOfTangents(),n,d+1);   /* result of Up x H x XPPP */
      Upp_  = myalloc2(m,d+1);     /* vector on left-hand side */
    }

    ~InterpolationDerivatives()
    {
      // Free allocated memory again
      myfree3(Yppp_);
      myfree1(y_);
      myfree3(Xppp_);
      myfree2(Upp_);
      myfree3(Zppp_);
    }

    /** \brief Bind the objects to a particular evaluation point
     *
     * In particular, this computes the value of the interpolation function at that point,
     * and the derivative at that point with respect to space.  The default implementation
     * uses ADOL-C to tape these evaluations.  That is required for the evaluateDerivatives
     * method below to be able to compute the derivatives with respect to the coefficients.
     *
     *  \param[in]  tapeNumber      Number of the ADOL-C tape to be used
     *  \param[in]  localPos        Local position where the FE function is evaluated
     *  \param[out] value           The function value at the local configuration
     *  \param[out] derivative      The derivative of the interpolation function
     *                              with respect to the evaluation point
     */
    template <typename Element>
    void bind(short tapeNumber,
              const Element& element,
              const typename Element::Geometry::LocalCoordinate& localPos,
              typename TargetSpace::CoordinateType& valueGlobalCoordinates,
              typename LocalInterpolationRule::DerivativeType& derivative)
    {
      using ATargetSpace = typename TargetSpace::template rebind<adouble>::other;
      using ALocalInterpolationRule = typename LocalInterpolationRule::template rebind<ATargetSpace>::other;

      const auto geometryJacobianInverse = element.geometry().jacobianInverse(localPos);

      ////////////////////////////////////////////////////////////////////////////////////////
      //  Tape the FE interpolation and its derivative with respect to the evaluation point.
      ////////////////////////////////////////////////////////////////////////////////////////
      trace_on(tapeNumber);

      std::vector<ATargetSpace> localAConfiguration(localInterpolationRule_.size());
      std::vector<typename ATargetSpace::CoordinateType> aRaw(localInterpolationRule_.size());
      for (size_t i=0; i<localInterpolationRule_.size(); i++) {
        typename TargetSpace::CoordinateType raw = localInterpolationRule_.coefficient(i).globalCoordinates();
        for (size_t j=0; j<raw.size(); j++)
          aRaw[i][j] <<= raw[j];
        localAConfiguration[i] = aRaw[i]; // may contain a projection onto M -- needs to be done in adouble
      }

      // Create the functions, we want to tape the function evaluation and the evaluation of the derivatives
      const auto& scalarFiniteElement = localInterpolationRule_.localFiniteElement();
      ALocalInterpolationRule localGFEFunction;
      localGFEFunction.bind(scalarFiniteElement,localAConfiguration);

      if (doValue_)
      {
        if (doDerivative_)
        {
          // Evaluate the function and its derivative with respect to space
          auto [aValue, aReferenceDerivative] = localGFEFunction.evaluateValueAndDerivative(localPos);

          //... and transfer the function values to global coordinates
          auto aValueGlobalCoordinates = aValue.globalCoordinates();

          // Tell ADOL-C that the value coordinates are dependent variables
          for (size_t i = 0; i<valueGlobalCoordinates.size(); i++)
            aValueGlobalCoordinates[i] >>= valueGlobalCoordinates[i];

          // Evaluate the derivative of the function defined on the actual element - these are in global coordinates already
          auto aDerivative = aReferenceDerivative * geometryJacobianInverse;

          for (size_t i = 0; i<derivative.rows; i++)
            for (size_t j = 0; j<derivative.cols; j++)
              aDerivative[i][j] >>= derivative[i][j];
        }
        else
        {
          // Evaluate the function
          auto aValue = localGFEFunction.evaluate(localPos);

          //... and transfer the function values to global coordinates
          auto aValueGlobalCoordinates = aValue.globalCoordinates();

          // Tell ADOL-C that the value coordinates are dependent variables
          for (size_t i = 0; i<valueGlobalCoordinates.size(); i++)
            aValueGlobalCoordinates[i] >>= valueGlobalCoordinates[i];
        }
      }
      else
      {
        if (doDerivative_)
        {
          // Evaluate the derivative of the local function defined on the reference element
          const auto aReferenceDerivative = localGFEFunction.evaluateDerivative(localPos);

          // Evaluate the derivative of the function defined on the actual element - these are in global coordinates already
          auto aDerivative = aReferenceDerivative * geometryJacobianInverse;

          for (size_t i = 0; i<derivative.rows; i++)
            for (size_t j = 0; j<derivative.cols; j++)
              aDerivative[i][j] >>= derivative[i][j];
        }
        else
        {
          // Do nothing
        }
      }

      trace_off();
    }

    /** \brief Compute first and second derivatives of the FE interpolation
     *
     * This code assumes that `bind` has been called before.
     *
     *  \param[in]  tapeNumber            The tape number to be used by ADOL-C.  Must be the same
     *                                    that was given to the `bind` method.
     *  \param[in]  weights               Vector of weights that the second derivative is contracted with
     *  \param[out] embeddedFirstDerivative       Derivative of the FE interpolation wrt the coefficients
     *  \param[out] firstDerivative       Derivative of the FE interpolation wrt the coefficients
     *  \param[out] secondDerivative      Second derivative of the FE interpolation,
     *                                    contracted with the weight vector
     */
    void evaluateDerivatives(short tapeNumber,
                             const double* weights,
                             Matrix<double>& euclideanFirstDerivative,
                             Matrix<double>& firstDerivative,
                             Matrix<FieldMatrix<double,blocksize,blocksize> >& secondDerivative) const
    {
      const size_t nDofs = localInterpolationRule_.size();

      // Number of dependent variables
      constexpr auto valueSize = embeddedBlocksize;
      constexpr auto derivativeSize = LocalInterpolationRule::DerivativeType::rows * LocalInterpolationRule::DerivativeType::cols;
      const size_t m = ((doValue_) ? embeddedBlocksize : 0)
                       + ((doDerivative_) ? derivativeSize : 0);

      const size_t n = nDofs * embeddedBlocksize;

      // Compute the Jacobian of the interpolation map in coordinates of the embedding space.
      // This is a single reverse sweep, and hence should relatively cheap.

      // However, first we need to wrap the euclideanFirstDerivative matrix by something ADOL-C can understand.
      double* embeddedFirstDerivative[euclideanFirstDerivative.N()];

      std::size_t counter = 0;
      if (doValue_)
      {
        for (std::size_t i=0; i<valueSize; i++)
          embeddedFirstDerivative[counter++] = euclideanFirstDerivative[i].data();
      }
      if (doDerivative_)
      {
        for (std::size_t i=0; i<derivativeSize; i++)
          embeddedFirstDerivative[counter++] = euclideanFirstDerivative[valueSize+i].data();
      }

      // Here is the actual AD reverse sweep
      jacobian(tapeNumber,
               m,
               n,
               localConfigurationFlat_.data(),
               embeddedFirstDerivative);

      ////////////////////////////////////////////////////////////////////////////////////////
      //  Do one forward ADOL-C sweep, using the vector in 'orthonormalFrames' as tangents.
      //  This achieves two things:
      //   a) It computes the Jacobian of the interpolation in the coordinates system
      //      spanned by the orthonormalFrames bases.
      //   b) It is the first of two steps to compute the second derivatives below.
      ////////////////////////////////////////////////////////////////////////////////////////

      const int d = 1; // TODO: What is this?  (ADOL-C calls this "highest derivative degree")

      // Vector-mode forward sweep
      // Disregard the return value.  Apparently it is not an error code.
      hov_wk_forward(tapeNumber,
                     m,         // Dimension of the function range space
                     n,         // Number of independent variables
                     d,         // ???
                     2,         // Keep all computed Taylor coefficients for later up to this order
                     numberOfTangents(),
                     localConfigurationFlat_.data(), // Where to evaluate the derivative
                     Xppp_,
                     y_, // [out] The computed value
                     Yppp_);    // [out] The computed Jacobian

      if (doValue_)
      {
        for (size_t i=0; i<m; i++)
          for (size_t j=0; j<numberOfTangents(); j++)
            firstDerivative[i][j] = Yppp_[i][j][0];
      }
      else
      {
        for (size_t i=0; i<m; i++)
          for (size_t j=0; j<numberOfTangents(); j++)
            firstDerivative[i+valueSize][j] = Yppp_[i][j][0];
      }

      ///////////////////////////////////////////////////////////////////////////
      //  Do a reverse sweep to compute the second derivative
      ///////////////////////////////////////////////////////////////////////////

      if (doValue_)
      {
        for (size_t i=0; i<m; i++)
        {
          Upp_[i][0] = weights[i];
          Upp_[i][1] = 0;
        }
      }
      else
      {
        for (size_t i=0; i<m; i++)
        {
          Upp_[i][0] = weights[i+valueSize];
          Upp_[i][1] = 0;
        }
      }

      // Scalar-mode reverse sweep
      // Scalar-mode is sufficient, because we have only one vector of weights.
      hos_ov_reverse(tapeNumber,
                     m,   // Number of dependent variables
                     n,   // Number of independent variables
                     d,   // d?  Highest derivative degree?
                     numberOfTangents(),   // Number of tangent vectors used in the previous forward sweep
                     Upp_,
                     Zppp_);

      ////////////////////////////////////////////////////////////////////////////////////
      //  Multiply from the right with the transposed orthonormal frames.
      //  ADOL-C doesn't do this for us, we have to do it by hand.
      ////////////////////////////////////////////////////////////////////////////////////

      for (size_t col=0; col<nDofs; col++)
      {
        for (size_t row=0; row<nDofs; row++)
        {
          SubmatrixView<blocksize,embeddedBlocksize,1> view(Zppp_,row,col);
          secondDerivative[row][col] = view.multiplyTransposed(orthonormalFrames_[col]);
        }
      }
    }

  };

  /** \brief Compute derivatives of GFE interpolation to RealTuple with respect to the coefficients
   *
   * This is the specialization of the InterpolationDerivatives class for the RealTuple target space.
   * Since RealTuple models the standard Euclidean space, geodesic FE interpolation reduces to
   * standard FE interpolation, and the derivatives with respect to the coefficients can be
   * computed much simpler and faster than for the general case.
   */
  template <typename Basis, typename field_type,int dim>
  class InterpolationDerivatives<LocalGeodesicFEFunction<Basis, RealTuple<field_type,dim> > >
  {
    using LocalInterpolationRule = LocalGeodesicFEFunction<Basis, RealTuple<field_type,dim> >;

    using Element = typename Basis::GridView::template Codim<0>::Entity;
    using LocalCoordinate = typename Element::Geometry::LocalCoordinate;
    static constexpr auto gridDim = LocalCoordinate::size();

    using LocalFiniteElement = typename Basis::LocalView::Tree::FiniteElement;
    using LocalBasis = typename LocalFiniteElement::Traits::LocalBasisType;
    using WeightType = typename LocalBasis::Traits::RangeType;
    using WeightJacobianType = typename LocalBasis::Traits::JacobianType;

    using TargetSpace = typename LocalInterpolationRule::TargetSpace;

    constexpr static auto blocksize = TargetSpace::TangentVector::dimension;

    //////////////////////////////////////////////////////////////////////
    //  Data members
    //////////////////////////////////////////////////////////////////////

    const LocalInterpolationRule& localInterpolationRule_;

    // Whether derivatives of the interpolation value are to be computed
    const bool doValue_;

    // Whether derivatives of the derivative of the interpolation
    // with respect to space are to be computed
    const bool doDerivative_;

    // Values of all scalar shape functions at the point we are bound to
    std::vector<WeightType> shapeFunctionValues_;

    // Gradients of all scalar shape functions at the point we are bound to
    std::vector<WeightJacobianType> shapeFunctionGradients_;


  public:

    InterpolationDerivatives(const LocalInterpolationRule& localInterpolationRule,
                             bool doValue,
                             bool doDerivative)
      : localInterpolationRule_(localInterpolationRule)
      , doValue_(doValue)
      , doDerivative_(doDerivative)
    {}

    /** \brief Bind the objects to a particular evaluation point
     *
     * In particular, this computes the value of the interpolation function at that point,
     * and the derivative at that point with respect to space.  The default implementation
     * uses ADOL-C to tape these evaluations.  That is required for the evaluateDerivatives
     * method below to be able to compute the derivatives with respect to the coefficients.
     *
     *  \param[in]  tapeNumber      Number of the ADOL-C tape, not used by this specialization
     *  \param[in]  localPos        Local position where the FE function is evaluated
     *  \param[out] value           The function value at the local configuration
     *  \param[out] derivative      The derivative of the interpolation function
     *                              with respect to the evaluation point
     */
    void bind(short tapeNumber,
              const Element& element,
              const LocalCoordinate& localPos,
              typename TargetSpace::CoordinateType& valueGlobalCoordinates,
              typename LocalInterpolationRule::DerivativeType& derivative)
    {
      const auto geometryJacobianInverse = element.geometry().jacobianInverse(localPos);

      const auto& scalarFiniteElement = localInterpolationRule_.localFiniteElement();
      const auto& localBasis = scalarFiniteElement.localBasis();

      // Get shape function values
      localBasis.evaluateFunction(localPos, shapeFunctionValues_);

      // Get shape function Jacobians
      localBasis.evaluateJacobian(localPos, shapeFunctionGradients_);

      for (auto& gradient : shapeFunctionGradients_)
        gradient = gradient * geometryJacobianInverse;

      std::fill(valueGlobalCoordinates.begin(), valueGlobalCoordinates.end(), 0.0);
      for (size_t i=0; i<shapeFunctionValues_.size(); i++)
        valueGlobalCoordinates.axpy(shapeFunctionValues_[i][0],
                                    localInterpolationRule_.coefficient(i).globalCoordinates());

      // Derivatives
      for (size_t i=0; i<localInterpolationRule_.size(); i++)
        for (int j=0; j<dim; j++)
          derivative[j].axpy(localInterpolationRule_.coefficient(i).globalCoordinates()[j],
                             shapeFunctionGradients_[i][0]);
    }

    /** \brief Compute first and second derivatives of the FE interpolation
     *
     * This code assumes that `bind` has been called before.
     *
     *  \param[in]  tapeNumber            The tape number to be used by ADOL-C.  Must be the same
     *                                    that was given to the `bind` method.
     *  \param[in]  weights               Vector of weights that the second derivative is contracted with
     *  \param[out] embeddedFirstDerivative       Derivative of the FE interpolation wrt the coefficients
     *  \param[out] firstDerivative       Derivative of the FE interpolation wrt the coefficients
     *  \param[out] secondDerivative      Second derivative of the FE interpolation,
     *                                    contracted with the weight vector
     */
    void evaluateDerivatives(short tapeNumber,
                             const double* weights,
                             Matrix<double>& embeddedFirstDerivative,
                             Matrix<double>& firstDerivative,
                             Matrix<FieldMatrix<double,blocksize,blocksize> >& secondDerivative) const
    {
      const size_t nDofs = localInterpolationRule_.size();

      ////////////////////////////////////////////////////////////////////
      //  The first derivative of the finite element interpolation
      ////////////////////////////////////////////////////////////////////

      firstDerivative = 0.0;

      // First derivatives of the function value wrt to the FE coefficients
      for (size_t i=0; i<nDofs; ++i)
        for (int j=0; j<blocksize; ++j)
          firstDerivative[j][i*blocksize+j] = shapeFunctionValues_[i][0];

      // First derivatives of the function gradient wrt to the FE coefficients
      for (size_t i=0; i<nDofs; ++i)
        for (int j=0; j<blocksize; ++j)
          for (std::size_t k=0; k<gridDim; ++k)
            firstDerivative[blocksize + j*gridDim + k][i*blocksize+j] = shapeFunctionGradients_[i][0][k];

      // For RealTuple, firstDerivative and embeddedFirstDerivative coincide
      embeddedFirstDerivative = firstDerivative;

      ////////////////////////////////////////////////////////////////////
      //  The second derivative of the finite element interpolation
      //  For RealTuple objects, all second derivatives are zero
      ////////////////////////////////////////////////////////////////////
      secondDerivative = 0;
    }
  };

  /** \brief Compute derivatives of GFE interpolation to RealTuple with respect to the coefficients
   *
   * This is the specialization of the InterpolationDerivatives class for the RealTuple target space
   * and the LocalProjectedGFEFunction interpolation.
   * Since RealTuple models the standard Euclidean space, projection-based FE interpolation reduces to
   * standard FE interpolation, and the derivatives with respect to the coefficients can be
   * computed much simpler and faster than for the general case.
   */
  template <typename Basis, typename field_type, int dim>
  class InterpolationDerivatives<LocalProjectedFEFunction<Basis, RealTuple<field_type,dim> > >
  {
    // TODO: The implementation here would be identical to the geodesic FE case
    using LocalInterpolationRule = LocalProjectedFEFunction<Basis, RealTuple<field_type,dim> >;

    using Element = typename Basis::GridView::template Codim<0>::Entity;
    using LocalCoordinate = typename Element::Geometry::LocalCoordinate;
    static constexpr auto gridDim = LocalCoordinate::size();

    using TargetSpace = typename LocalInterpolationRule::TargetSpace;

    constexpr static auto blocksize = TargetSpace::TangentVector::dimension;

    //////////////////////////////////////////////////////////////////////
    //  Data members
    //////////////////////////////////////////////////////////////////////

    const LocalInterpolationRule& localInterpolationRule_;

    // Whether derivatives of the interpolation value are to be computed
    const bool doValue_;

    // Whether derivatives of the derivative of the interpolation
    // with respect to space are to be computed
    const bool doDerivative_;

    // Values of all scalar shape functions at the point we are bound to
    std::vector<FieldVector<double,1> > shapeFunctionValues_;

    // Gradients of all scalar shape functions at the point we are bound to
    // TODO: The second dimension must be WorldDim
    std::vector<FieldMatrix<double,1,gridDim> > shapeFunctionGradients_;

  public:
    InterpolationDerivatives(const LocalInterpolationRule& localInterpolationRule,
                             bool doValue, bool doDerivative)
      : localInterpolationRule_(localInterpolationRule)
      , doValue_(doValue)
      , doDerivative_(doDerivative)
    {}

    /** \brief Bind the objects to a particular evaluation point
     *
     * In particular, this computes the value of the interpolation function at that point,
     * and the derivative at that point with respect to space.  The default implementation
     * uses ADOL-C to tape these evaluations.  That is required for the evaluateDerivatives
     * method below to be able to compute the derivatives with respect to the coefficients.
     *
     *  \param[in]  tapeNumber      Number of the ADOL-C tape, not used by this specialization
     *  \param[in]  localPos        Local position where the FE function is evaluated
     *  \param[out] value           The function value at the local configuration
     *  \param[out] derivative      The derivative of the interpolation function
     *                              with respect to the evaluation point
     */
    void bind(short tapeNumber,
              const Element& element,
              const LocalCoordinate& localPos,
              typename TargetSpace::CoordinateType& valueGlobalCoordinates,
              typename LocalInterpolationRule::DerivativeType& derivative)
    {
      const auto geometryJacobianInverse = element.geometry().jacobianInverse(localPos);

      const auto& scalarFiniteElement = localInterpolationRule_.localFiniteElement();
      const auto& localBasis = scalarFiniteElement.localBasis();

      // Get shape function values
      localBasis.evaluateFunction(localPos, shapeFunctionValues_);

      // Get shape function Jacobians
      localBasis.evaluateJacobian(localPos, shapeFunctionGradients_);

      for (auto& gradient : shapeFunctionGradients_)
        gradient = gradient * geometryJacobianInverse;

      std::fill(valueGlobalCoordinates.begin(), valueGlobalCoordinates.end(), 0.0);
      for (size_t i=0; i<shapeFunctionValues_.size(); i++)
        valueGlobalCoordinates.axpy(shapeFunctionValues_[i][0],
                                    localInterpolationRule_.coefficient(i).globalCoordinates());

      // Derivatives
      for (size_t i=0; i<localInterpolationRule_.size(); i++)
        for (int j=0; j<dim; j++)
          derivative[j].axpy(localInterpolationRule_.coefficient(i).globalCoordinates()[j],
                             shapeFunctionGradients_[i][0]);
    }

    /** \brief Compute first and second derivatives of the FE interpolation
     *
     * This code assumes that `bind` has been called before.
     *
     *  \param[in]  tapeNumber            The tape number to be used by ADOL-C. Not used by this specialization
     *  \param[in]  weights               Vector of weights that the second derivative is contracted with
     *  \param[out] embeddedFirstDerivative       Derivative of the FE interpolation wrt the coefficients
     *  \param[out] firstDerivative       Derivative of the FE interpolation wrt the coefficients
     *  \param[out] secondDerivative      Second derivative of the FE interpolation,
     *                                    contracted with the weight vector
     */
    void evaluateDerivatives(short tapeNumber,
                             const double* weights,
                             Matrix<double>& embeddedFirstDerivative,
                             Matrix<double>& firstDerivative,
                             Matrix<FieldMatrix<double,blocksize,blocksize> >& secondDerivative) const
    {
      const size_t nDofs = localInterpolationRule_.size();

      ////////////////////////////////////////////////////////////////////
      //  The first derivative of the finite element interpolation
      ////////////////////////////////////////////////////////////////////

      firstDerivative = 0.0;

      // First derivatives of the function value wrt to the FE coefficients
      for (size_t i=0; i<nDofs; ++i)
        for (int j=0; j<blocksize; ++j)
          firstDerivative[j][i*blocksize+j] = shapeFunctionValues_[i][0];

      // First derivatives of the function gradient wrt to the FE coefficients
      for (size_t i=0; i<nDofs; ++i)
        for (int j=0; j<blocksize; ++j)
          for (std::size_t k=0; k<gridDim; ++k)
            firstDerivative[blocksize + j*gridDim + k][i*blocksize+j] = shapeFunctionGradients_[i][0][k];

      // For RealTuple, firstDerivative and embeddedFirstDerivative coincide
      embeddedFirstDerivative = firstDerivative;

      ////////////////////////////////////////////////////////////////////
      //  The second derivative of the finite element interpolation
      //  For RealTuple objects, all second derivatives are zero
      ////////////////////////////////////////////////////////////////////
      secondDerivative = 0;
    }
  };

  /** \brief Compute derivatives of nonconforming interpolation with respect to the coefficients
   *
   * This is the specialization of the InterpolationDerivatives class for the nonconforming
   * interpolation.  No matter what the target space is, the interpolation is always
   * Euclidean in the surrounding space.
   */
  template <typename Basis, typename TargetSpace>
  class InterpolationDerivatives<LocalProjectedFEFunction<Basis, TargetSpace, false> >
  {
    using Element = typename Basis::GridView::template Codim<0>::Entity;
    using LocalCoordinate = typename Element::Geometry::LocalCoordinate;
    static constexpr auto gridDim = LocalCoordinate::size();

    using LocalFiniteElement = typename Basis::LocalView::Tree::FiniteElement;
    using LocalBasis = typename LocalFiniteElement::Traits::LocalBasisType;
    using FERangeType = typename LocalBasis::Traits::RangeType;
    using FEJacobianType = typename LocalBasis::Traits::JacobianType;

    using LocalInterpolationRule = LocalProjectedFEFunction<Basis, TargetSpace, false>;
    using CoordinateType = typename TargetSpace::CoordinateType;

    constexpr static auto blocksize = TargetSpace::TangentVector::dimension;
    constexpr static auto embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension;

    //////////////////////////////////////////////////////////////////////
    //  Data members
    //////////////////////////////////////////////////////////////////////

    const LocalInterpolationRule& localInterpolationRule_;

    // Whether derivatives of the interpolation value are to be computed
    const bool doValue_;

    // Whether derivatives of the derivative of the interpolation
    // with respect to space are to be computed
    const bool doDerivative_;

    // Values of all scalar shape functions at the point we are bound to
    std::vector<FERangeType> shapeFunctionValues_;

    // Gradients of all scalar shape functions at the point we are bound to
    std::vector<FEJacobianType> shapeFunctionGradients_;

    // TODO: Don't hardcode FieldMatrix
    std::vector<FieldMatrix<double,blocksize,embeddedBlocksize> > orthonormalFrames_;

  public:
    InterpolationDerivatives(const LocalInterpolationRule& localInterpolationRule,
                             bool doValue, bool doDerivative)
      : localInterpolationRule_(localInterpolationRule)
      , doValue_(doValue)
      , doDerivative_(doDerivative)
    {
      // Precompute the orthonormal frames
      orthonormalFrames_.resize(localInterpolationRule_.size());

      for (size_t i=0; i<localInterpolationRule_.size(); ++i)
        orthonormalFrames_[i] = localInterpolationRule_.coefficient(i).orthonormalFrame();
    }

    /** \brief Bind the objects to a particular evaluation point
     *
     * In particular, this computes the value of the interpolation function at that point,
     * and the derivative at that point with respect to space.  The default implementation
     * uses ADOL-C to tape these evaluations.  That is required for the evaluateDerivatives
     * method below to be able to compute the derivatives with respect to the coefficients.
     *
     *  \param[in]  tapeNumber      Number of the ADOL-C tape, not used by this specialization
     *  \param[in]  localPos        Local position where the FE function is evaluated
     *  \param[out] value           The function value at the local configuration
     *  \param[out] derivative      The derivative of the interpolation function
     *                              with respect to the evaluation point
     */
    void bind(short tapeNumber,
              const Element& element,
              const LocalCoordinate& localPos,
              typename TargetSpace::CoordinateType& valueGlobalCoordinates,
              typename LocalInterpolationRule::DerivativeType& derivative)
    {
      const auto geometryJacobianInverse = element.geometry().jacobianInverse(localPos);

      const auto& scalarFiniteElement = localInterpolationRule_.localFiniteElement();
      const auto& localBasis = scalarFiniteElement.localBasis();

      // Get shape function values
      localBasis.evaluateFunction(localPos, shapeFunctionValues_);

      // Get shape function Jacobians
      localBasis.evaluateJacobian(localPos, shapeFunctionGradients_);

      for (auto& gradient : shapeFunctionGradients_)
        gradient = gradient * geometryJacobianInverse;

      std::fill(valueGlobalCoordinates.begin(), valueGlobalCoordinates.end(), 0.0);
      for (size_t i=0; i<shapeFunctionValues_.size(); i++)
        valueGlobalCoordinates.axpy(shapeFunctionValues_[i][0],
                                    localInterpolationRule_.coefficient(i).globalCoordinates());

      // Derivatives
      for (size_t i=0; i<localInterpolationRule_.size(); i++)
        for (int j=0; j<TargetSpace::CoordinateType::dimension; j++)
          derivative[j].axpy(localInterpolationRule_.coefficient(i).globalCoordinates()[j],
                             shapeFunctionGradients_[i][0]);
    }

    /** \brief Compute first and second derivatives of the FE interpolation
     *
     * This code assumes that `bind` has been called before.
     *
     *  \param[in]  tapeNumber            The tape number to be used by ADOL-C. Not used by this specialization
     *  \param[in]  weights               Vector of weights that the second derivative is contracted with
     *  \param[out] embeddedFirstDerivative       Derivative of the FE interpolation wrt the coefficients
     *  \param[out] firstDerivative       Derivative of the FE interpolation wrt the coefficients
     *  \param[out] secondDerivative      Second derivative of the FE interpolation,
     *                                    contracted with the weight vector
     */
    void evaluateDerivatives(short tapeNumber,
                             const double* weights,
                             Matrix<double>& embeddedFirstDerivative,
                             Matrix<double>& firstDerivative,
                             Matrix<FieldMatrix<double,blocksize,blocksize> >& secondDerivative) const
    {
      constexpr size_t valueSize = TargetSpace::CoordinateType::dimension;
      constexpr size_t derivativeSize = TargetSpace::CoordinateType::dimension * gridDim;

      const size_t nDofs = localInterpolationRule_.size();

      ////////////////////////////////////////////////////////////////////
      //  The first derivative of the finite element interpolation
      ////////////////////////////////////////////////////////////////////

      Matrix<double> partialDerivative(embeddedFirstDerivative.N(), embeddedFirstDerivative.M());
      partialDerivative = 0.0;

      // First derivatives of the function value wrt to the FE coefficients
      for (size_t i=0; i<nDofs; ++i)
        for (int j=0; j<embeddedBlocksize; ++j)
          partialDerivative[j][i*embeddedBlocksize+j] = shapeFunctionValues_[i][0];

      // First derivatives of the function gradient wrt to the FE coefficients
      for (size_t i=0; i<nDofs; ++i)
        for (int j=0; j<embeddedBlocksize; ++j)
          for (std::size_t k=0; k<gridDim; ++k)
            partialDerivative[embeddedBlocksize + j*gridDim + k][i*embeddedBlocksize+j] = shapeFunctionGradients_[i][0][k];

      // Euclidean derivative: Derivatives in the direction of the projected canonical vectors
      for (std::size_t j=0; j<nDofs; ++j)
      {
        for (int k=0; k<embeddedBlocksize; k++)
        {
          // k-th canonical unit vector
          FieldVector<double,embeddedBlocksize> direction(0);
          direction[k] = 1.0;

          // Project it onto tangent space at the j-th coefficient
          auto projectedDirection = localInterpolationRule_.coefficient(j).projectOntoTangentSpace(direction);

          // The interpolation value
          if (doValue_)
          {
            for (size_t i=0; i<CoordinateType::size(); ++i)
            {
              // Alias name, for shorter notation
              auto& entry = embeddedFirstDerivative[i][j*embeddedBlocksize+k];

              entry = 0;

              for (int l=0; l<embeddedBlocksize; l++)
                entry += projectedDirection[l] * partialDerivative[i][j*embeddedBlocksize+l];
            }
          }

          // The interpolation derivative with respect to space
          if (doDerivative_)
          {
            for (size_t alpha=0; alpha<CoordinateType::size(); alpha++)
            {
              for (size_t beta=0; beta<gridDim; beta++)
              {
                // Alias name, for shorter notation
                auto& entry = embeddedFirstDerivative[CoordinateType::size() + alpha*gridDim+beta][j*embeddedBlocksize+k];

                entry = 0;

                for (int l=0; l<embeddedBlocksize; l++)
                  entry += projectedDirection[l] * partialDerivative[CoordinateType::size() + alpha*gridDim+beta][j*embeddedBlocksize+l];
              }
            }
          }
        }
      }

      // Riemannian derivative: Derivatives in the directions of the orthonormal frame vectors
      for (std::size_t j=0; j<nDofs; ++j)
      {
        for (int k=0; k<blocksize; k++)
        {
          // k-th canonical unit tangent vector
          FieldVector<double,embeddedBlocksize> direction = orthonormalFrames_[j][k];

          // The interpolation value
          if (doValue_)
          {
            for (size_t i=0; i<CoordinateType::size(); ++i)
            {
              // Alias name, for shorter notation
              auto& entry = firstDerivative[i][j*blocksize+k];

              entry = 0;

              for (int l=0; l<embeddedBlocksize; l++)
                entry += direction[l] * partialDerivative[i][j*embeddedBlocksize+l];
            }
          }

          // The interpolation derivative with respect to space
          if (doDerivative_)
          {
            for (size_t alpha=0; alpha<CoordinateType::size(); alpha++)
            {
              for (size_t beta=0; beta<gridDim; beta++)
              {
                // Alias name, for shorter notation
                auto& entry = firstDerivative[CoordinateType::size() + alpha*gridDim+beta][j*blocksize+k];

                entry = 0;

                for (int l=0; l<embeddedBlocksize; l++)
                  entry += direction[l] * partialDerivative[CoordinateType::size() + alpha*gridDim+beta][j*embeddedBlocksize+l];
              }
            }
          }
        }
      }

      ////////////////////////////////////////////////////////////////////
      //  The second derivative of the finite element interpolation
      ////////////////////////////////////////////////////////////////////

      // From this, compute the Hessian with respect to the manifold (which we assume here is embedded
      // isometrically in a Euclidean space.
      // For the detailed explanation of the following see: Absil, Mahoney, Trumpf, "An extrinsic look
      // at the Riemannian Hessian".

      if (!doDerivative_)
        return;

      secondDerivative = 0;


      //////////////////////////////////////////////////////////////////////////
      //  The projection of the Euclidean first derivative of the finite element interpolation
      //  onto the normal space of the unit sphere.
      //////////////////////////////////////////////////////////////////////////

      Matrix<double> normalFirstDerivative(embeddedFirstDerivative.N(), nDofs);

      for (std::size_t j=0; j<nDofs; ++j)
      {
        // The space is a sphere.  Therefore the normal at a point x is x itself.
        const auto& direction = localInterpolationRule_.coefficient(j).globalCoordinates();

        // The interpolation value
        if (doValue_)
        {
          for (size_t i=0; i<CoordinateType::size(); ++i)
          {
            // Alias name, for shorter notation
            auto& entry = normalFirstDerivative[i][j];

            entry = 0;

            for (int l=0; l<embeddedBlocksize; l++)
              entry += direction[l] * partialDerivative[i][j*embeddedBlocksize+l];
          }
        }

        // The interpolation derivative with respect to space
        if (doDerivative_)
        {
          for (size_t alpha=0; alpha<CoordinateType::size(); alpha++)
          {
            for (size_t beta=0; beta<gridDim; beta++)
            {
              // Alias name, for shorter notation
              auto& entry = normalFirstDerivative[CoordinateType::size() + alpha*gridDim+beta][j];

              entry = 0;

              for (int l=0; l<embeddedBlocksize; l++)
                entry += direction[l] * partialDerivative[CoordinateType::size() + alpha*gridDim+beta][j*embeddedBlocksize+l];
            }
          }
        }
      }

      // Project Euclidean gradient onto the normal space
      // The range of input variables that the density depends on
      const size_t begin = (doValue_) ? 0 : valueSize;
      const size_t end = (doDerivative_) ? valueSize + derivativeSize : valueSize;

      Matrix<FieldMatrix<double,1,embeddedBlocksize> > projectedGradient(embeddedFirstDerivative.N(),nDofs);

      for (size_t i=begin; i<end; i++)
        for (size_t j=0; j<nDofs; j++)
          projectedGradient[i][j][0] = normalFirstDerivative[i][j] * localInterpolationRule_.coefficient(j).globalCoordinates();

      //////////////////////////////////////////////////////////////////////////////////////
      //  Further correction due to non-planar configuration space
      //  + \mathfrak{A}_x(z,P^\orth_x \partial f)
      //////////////////////////////////////////////////////////////////////////////////////

      // The configuration space is a product of spheres.  Therefore the Weingarten map
      // has only block-diagonal entries.
      for (size_t row=0; row<nDofs; row++)
      {
        for (size_t subRow=0; subRow<blocksize; subRow++)
        {
          typename TargetSpace::EmbeddedTangentVector z = orthonormalFrames_[row][subRow];

          for (size_t i=begin; i<end; i++)
          {
            auto tmp1 = localInterpolationRule_.coefficient(row).weingarten(z,projectedGradient[i][row][0]);

            typename TargetSpace::TangentVector tmp2;
            orthonormalFrames_[row].mv(tmp1,tmp2);

            for (size_t subCol=0; subCol<blocksize; subCol++)
              secondDerivative[row][row][subRow][subCol] += weights[i]* tmp2[subCol];
          }
        }
      }
    }
  };

  /** \brief Compute derivatives of projection-based interpolation to UnitVector with respect to the coefficients
   *
   * This is the specialization of the InterpolationDerivatives class for the UnitVector target space
   * and the LocalProjectedGFEFunction interpolation.  In this situation, the derivatives can be computed
   * by hand with reasonable effort.  The corresponding implementation is faster than the one
   * based on ADOL-C.
   */
  template <typename Basis, typename field_type,int dim>
  class InterpolationDerivatives<LocalProjectedFEFunction<Basis, UnitVector<field_type,dim> > >
  {
    using Element = typename Basis::GridView::template Codim<0>::Entity;
    using LocalCoordinate = typename Element::Geometry::LocalCoordinate;
    static constexpr auto gridDim = LocalCoordinate::size();

    using LocalFiniteElement = typename Basis::LocalView::Tree::FiniteElement;
    using LocalBasis = typename LocalFiniteElement::Traits::LocalBasisType;
    using FERangeType = typename LocalBasis::Traits::RangeType;
    using FEJacobianType = typename LocalBasis::Traits::JacobianType;

    using TargetSpace = UnitVector<field_type,dim>;
    using CoordinateType = typename TargetSpace::CoordinateType;

    constexpr static auto blocksize = TargetSpace::TangentVector::dimension;
    constexpr static auto embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension;

    using LocalInterpolationRule = LocalProjectedFEFunction<Basis, TargetSpace>;

    /** \brief A vector that can be contracted with a derivative of the projection onto the unit sphere
     *
     * Any vector can be contracted with these derivatives, but this contraction always involves
     * computing the scalar product of the vector and the unprojected point.  For extra efficiency
     * we precompute these scalar products and store the result here.
     */
    struct VectorToContractWith
    {
      // TODO: I'd like this to be const, but how do we then construct arrays of VectorToContractWith?
      typename TargetSpace::EmbeddedTangentVector vector_;

      field_type scalarProduct_;

      VectorToContractWith() = default;

      explicit VectorToContractWith(const typename TargetSpace::EmbeddedTangentVector& vector)
        : vector_(vector)
      {}

      const field_type& operator[](std::size_t i) const
      {
        return vector_[i];
      }

      void bind(const CoordinateType& point)
      {
        scalarProduct_ = point * vector_;
      }
    };

    /** \brief The projection onto the unit sphere
     */
    class Projection
    {
    public:
      CoordinateType x_;

      // Different powers of the norm of x
      std::array<double,8> normPower_;

    public:
      void bind(const CoordinateType& x)
      {
        x_ = x;
        normPower_[0] = 1.0;
        normPower_[1] = x.two_norm();
        for (std::size_t i=2; i<normPower_.size(); i++)
          normPower_[i] = normPower_[i-1] * normPower_[1];
      }

      CoordinateType value() const
      {
        return x_ / normPower_[1];
      }

      /** \brief First partial derivative
       */
#if 0
      double derivative1(std::size_t i, std::size_t j) const
      {
        return (i==j)/normPower_[1] - x_[i]*x_[j] / normPower_[3];
      }
#endif

      /** \brief First directional derivative
       */
      double derivative1(std::size_t i, const typename TargetSpace::EmbeddedTangentVector& d) const
      {
        const auto sp = x_*d;
        return d[i]/normPower_[1] - x_[i]*sp / normPower_[3];
      }

      /** \brief First directional derivative
       */
      double derivative1(std::size_t i, const VectorToContractWith& d) const
      {
        return d[i]/normPower_[1] - x_[i]*d.scalarProduct_ / normPower_[3];
      }

      /** \brief Second partial derivative
       */
#if 0
      double derivative2(std::size_t i, std::size_t j, std::size_t k) const
      {
        return -((i==j)*x_[k] + (i==k)*x_[j] + (j==k)*x_[i]) / normPower_[3]
               + 3*x_[i]*x_[j]*x_[k] / normPower_[5];
      }
#endif

      /** \brief Second mixed-partial-directional derivative
       */
      double derivative2(std::size_t i, std::size_t j, const VectorToContractWith& dk) const
      {
        return -((i==j)*dk.scalarProduct_ + dk[i]*x_[j] + dk[j]*x_[i]) / normPower_[3]
               + 3*x_[i]*x_[j]*dk.scalarProduct_ / normPower_[5];
      }

      /** \brief Second mixed-partial-directional derivative
       */
      double derivative2(std::size_t i, std::size_t j, const typename TargetSpace::EmbeddedTangentVector& dk) const
      {
        const auto spk = x_*dk;
        return -((i==j)*spk + dk[i]*x_[j] + dk[j]*x_[i]) / normPower_[3]
               + 3*x_[i]*x_[j]*spk / normPower_[5];
      }

      /** \brief Second mixed-partial-directional derivative
       */
      double derivative2(std::size_t i, const VectorToContractWith& dj, const typename TargetSpace::EmbeddedTangentVector& dk) const
      {
        const auto spk = x_*dk;
        return -(dj[i]*spk + dk[i]*dj.scalarProduct_ + dk*dj.vector_*x_[i]) / normPower_[3]
               + 3*x_[i]*dj.scalarProduct_*spk / normPower_[5];
      }

      /** \brief Second directional derivative
       */
      double derivative2(std::size_t i, const VectorToContractWith& dj, const VectorToContractWith& dk) const
      {
        return -(dj[i]*dk.scalarProduct_ + dk[i]*dj.scalarProduct_ + (dj.vector_*dk.vector_)*x_[i]) / normPower_[3]
               + 3*x_[i]*dj.scalarProduct_*dk.scalarProduct_ / normPower_[5];
      }

      /** \brief Second directional derivative
       */
      double derivative2(const VectorToContractWith& di, const VectorToContractWith& dj, const VectorToContractWith& dk) const
      {
        return -((di.vector_*dj.vector_)*dk.scalarProduct_ + (di.vector_*dk.vector_)*dj.scalarProduct_ + (dj.vector_*dk.vector_)*di.scalarProduct_) / normPower_[3]
               + 3*di.scalarProduct_*dj.scalarProduct_*dk.scalarProduct_ / normPower_[5];
      }

      /** \brief Third derivative
       */
      double derivative3(std::size_t i,
                         std::size_t j,
                         const VectorToContractWith& dk,
                         const VectorToContractWith& dl) const
      {
        return (-1) * ((i==j)*(dk.vector_*dl.vector_) + dk[i]*dl[j] + dk[j]*dl[i]) / normPower_[3]
               +3 * ((i==j)*dk.scalarProduct_*dl.scalarProduct_ + dk[i]*x_[j]*dl.scalarProduct_ + dk[j]*x_[i]*dl.scalarProduct_ + dl[i]*x_[j]*dk.scalarProduct_ + dl[j]*x_[i]*dk.scalarProduct_ + (dk.vector_*dl.vector_)*x_[i]*x_[j]) / normPower_[5]
               -15 * x_[i] * x_[j] * dk.scalarProduct_ * dl.scalarProduct_ / normPower_[7];
      }

      /** \brief Third derivative
       */
      double derivative3(const VectorToContractWith& di,
                         const VectorToContractWith& dj,
                         const VectorToContractWith& dk,
                         const VectorToContractWith& dl) const
      {
        double sp[4][4];  // Scalar products of the input vectors

        sp[0][1] = di.vector_*dj.vector_;
        sp[0][2] = di.vector_*dk.vector_;
        sp[0][3] = di.vector_*dl.vector_;
        sp[1][2] = dj.vector_*dk.vector_;
        sp[1][3] = dj.vector_*dl.vector_;
        sp[2][3] = dk.vector_*dl.vector_;

        return (-1) * (sp[0][1]*sp[2][3] + sp[0][2]*sp[1][3] + sp[1][2]*sp[0][3]) / normPower_[3]
               +3 * (sp[0][1]*dk.scalarProduct_*dl.scalarProduct_ + sp[0][2]*dj.scalarProduct_*dl.scalarProduct_
                     + sp[1][2]*di.scalarProduct_*dl.scalarProduct_ + sp[0][3]*dj.scalarProduct_*dk.scalarProduct_
                     + sp[1][3]*di.scalarProduct_*dk.scalarProduct_ + sp[2][3]*di.scalarProduct_*dj.scalarProduct_) / normPower_[5]
               -15 * di.scalarProduct_ * dj.scalarProduct_ * dk.scalarProduct_ * dl.scalarProduct_ / normPower_[7];
      }

    };

    //////////////////////////////////////////////////////////////////////
    //  Data members
    //////////////////////////////////////////////////////////////////////

    const LocalInterpolationRule& localInterpolationRule_;

    // Whether derivatives of the interpolation value are to be computed
    const bool doValue_;

    // Whether derivatives of the derivative of the interpolation
    // with respect to space are to be computed
    const bool doDerivative_;

    // Values of all scalar shape functions at the point we are bound to
    std::vector<FERangeType> shapeFunctionValues_;

    // Gradients of all scalar shape functions at the point we are bound to
    std::vector<FEJacobianType> shapeFunctionGradients_;

    // TODO: Is this needed at all?
    typename LocalInterpolationRule::DerivativeType euclideanDerivative_;

    // Transposed derivative, because we need simple access to the columns
    using EuclideanDerivativeTransposed = decltype(transpose(euclideanDerivative_));
    EuclideanDerivativeTransposed euclideanDerivativeTransposed_;

    // TODO: Don't hardcode FieldMatrix
    std::vector<FieldMatrix<double,blocksize,embeddedBlocksize> > orthonormalFrames_;

    // TODO It would be nicer to have this as a std::vector of std::array of VectorToContractWith,
    // but I don't currently see how to set this up without sacrificing constness.
    std::vector<VectorToContractWith> tangentVectors_;

    // An object that implements the projection onto the unit sphere, and its derivatives
    Projection projection_;

  public:
    InterpolationDerivatives(const LocalInterpolationRule& localInterpolationRule,
                             bool doValue,
                             bool doDerivative)
      : localInterpolationRule_(localInterpolationRule)
      , doValue_(doValue)
      , doDerivative_(doDerivative)
    {
      // Precompute the orthonormal frames
      orthonormalFrames_.resize(localInterpolationRule_.size());
      tangentVectors_.reserve(localInterpolationRule_.size()*blocksize);

      for (size_t i=0; i<localInterpolationRule_.size(); ++i)
      {
        orthonormalFrames_[i] = localInterpolationRule_.coefficient(i).orthonormalFrame();

        for (size_t j=0; j<blocksize; j++)
          tangentVectors_.emplace_back(orthonormalFrames_[i][j]);
      }
    }

    /** \brief Bind the objects to a particular evaluation point
     *
     * In particular, this computes the value of the interpolation function at that point,
     * and the derivative at that point with respect to space.  The default implementation
     * uses ADOL-C to tape these evaluations.  That is required for the evaluateDerivatives
     * method below to be able to compute the derivatives with respect to the coefficients.
     *
     *  \param[in]  tapeNumber      Number of the ADOL-C tape, not used by this specialization
     *  \param[in]  localPos        Local position where the FE function is evaluated
     *  \param[out] value           The function value at the local configuration
     *  \param[out] derivative      The derivative of the interpolation function
     *                              with respect to the evaluation point
     */
    void bind(short tapeNumber,
              const Element& element,
              const LocalCoordinate& localPos,
              typename TargetSpace::CoordinateType& valueGlobalCoordinates,
              typename LocalInterpolationRule::DerivativeType& derivative)
    {
      const auto geometryJacobianInverse = element.geometry().jacobianInverse(localPos);

      const auto& scalarFiniteElement = localInterpolationRule_.localFiniteElement();
      const auto& localBasis = scalarFiniteElement.localBasis();

      // Get shape function values
      localBasis.evaluateFunction(localPos, shapeFunctionValues_);

      // Get shape function Jacobians
      localBasis.evaluateJacobian(localPos, shapeFunctionGradients_);

      for (auto& gradient : shapeFunctionGradients_)
        gradient = gradient * geometryJacobianInverse;

      FieldVector<field_type,dim> euclideanInterpolation(0.0);
      for (size_t i=0; i<shapeFunctionValues_.size(); i++)
        euclideanInterpolation.axpy(shapeFunctionValues_[i][0],
                                    localInterpolationRule_.coefficient(i).globalCoordinates());

      // The nonconforming case has its own specialization
      auto value = TargetSpace::projectOnto(euclideanInterpolation);
      valueGlobalCoordinates = value.globalCoordinates();

      // Derivatives
      euclideanDerivative_ = 0;
      for (size_t i=0; i<localInterpolationRule_.size(); i++)
        for (int j=0; j<dim; j++)
          euclideanDerivative_[j].axpy(localInterpolationRule_.coefficient(i).globalCoordinates()[j],
                                       shapeFunctionGradients_[i][0]);

      auto derivativeOfProjection = TargetSpace::derivativeOfProjection(euclideanInterpolation);

      derivative = derivativeOfProjection * euclideanDerivative_;

      euclideanDerivativeTransposed_ = transpose(euclideanDerivative_);

      projection_.bind(euclideanInterpolation);

      for (auto& v : tangentVectors_)
        v.bind(euclideanInterpolation);
    }

    /** \brief Compute first and second derivatives of the FE interpolation
     *
     * This code assumes that `bind` has been called before.
     *
     *  \param[in]  tapeNumber            The tape number to be used by ADOL-C.  Must be the same
     *                                    that was given to the `bind` method.
     *  \param[in]  weights               Vector of weights that the second derivative is contracted with
     *  \param[out] embeddedFirstDerivative       Derivative of the FE interpolation wrt the coefficients
     *  \param[out] firstDerivative       Derivative of the FE interpolation wrt the coefficients
     *  \param[out] secondDerivative      Second derivative of the FE interpolation,
     *                                    contracted with the weight vector
     */
    void evaluateDerivatives(short tapeNumber,
                             const double* weights,
                             Matrix<double>& embeddedFirstDerivative,
                             Matrix<double>& firstDerivative,
                             Matrix<FieldMatrix<double,blocksize,blocksize> >& secondDerivative) const
    {
      const size_t nDofs = localInterpolationRule_.size();

      std::array<VectorToContractWith, gridDim> euclidDer;

      for (std::size_t beta=0; beta<gridDim; beta++)
      {
        euclidDer[beta].vector_ = euclideanDerivativeTransposed_[beta];
        euclidDer[beta].bind(projection_.x_);
      }

      //////////////////////////////////////////////////////////////////////////
      //  The Euclidean first derivative of the finite element interpolation
      //////////////////////////////////////////////////////////////////////////

      // The interpolation value
      if (doValue_)
      {
        for (size_t i=0; i<CoordinateType::size(); ++i)
          for (std::size_t j=0; j<nDofs; ++j)
            for (int k=0; k<embeddedBlocksize; k++)
            {
              // k-th canonical unit vector
              FieldVector<double,embeddedBlocksize> direction(0);
              direction[k] = 1.0;

              // Project it onto tangent space at the j-th coefficient
              auto projectedDirection = localInterpolationRule_.coefficient(j).projectOntoTangentSpace(direction);

              embeddedFirstDerivative[i][j*embeddedBlocksize+k] = projection_.derivative1(i,projectedDirection) * shapeFunctionValues_[j][0];
            }
      }

      // The interpolation derivative with respect to space
      if (doDerivative_)
      {
        for (size_t alpha=0; alpha<CoordinateType::size(); alpha++)
          for (size_t beta=0; beta<gridDim; beta++)
            for (std::size_t j=0; j<nDofs; ++j)
              for (int k=0; k<embeddedBlocksize; k++)
              {
                // Alias name, for shorter notation
                auto& entry = embeddedFirstDerivative[CoordinateType::size() + alpha*gridDim+beta][j*embeddedBlocksize+k];

                entry = 0;

                // k-th canonical unit vector
                FieldVector<double,embeddedBlocksize> direction(0);
                direction[k] = 1.0;

                // Project it onto the tangent space at the j-th coefficient
                auto projectedDirection = localInterpolationRule_.coefficient(j).projectOntoTangentSpace(direction);

                VectorToContractWith euclidDer(euclideanDerivativeTransposed_[beta]);
                euclidDer.bind(projection_.x_);

                entry += projection_.derivative2(alpha,euclidDer,projectedDirection)*shapeFunctionValues_[j];
                entry += projection_.derivative1(alpha,projectedDirection) * shapeFunctionGradients_[j][0][beta];
              }
      }

      //////////////////////////////////////////////////////////////////////////
      //  The Riemannian first derivative of the finite element interpolation
      //////////////////////////////////////////////////////////////////////////

      // The interpolation value
      if (doValue_)
      {
        for (size_t i=0; i<CoordinateType::size(); ++i)
          for (std::size_t j=0; j<nDofs; ++j)
            for (std::size_t k=0; k<orthonormalFrames_[j].size(); k++)
              firstDerivative[i][j*blocksize+k] = projection_.derivative1(i,tangentVectors_[j*blocksize+k]) * shapeFunctionValues_[j][0];
      }

      // The interpolation derivative with respect to space
      if (doDerivative_)
      {
        for (size_t alpha=0; alpha<CoordinateType::size(); alpha++)
          for (size_t beta=0; beta<gridDim; beta++)
            for (size_t j=0; j<nDofs; ++j)
              for (std::size_t k=0; k<orthonormalFrames_[j].size(); k++)
              {
                // Alias name, for shorter notation
                auto& entry = firstDerivative[CoordinateType::size() + alpha*gridDim+beta][j*blocksize+k];

                entry = 0;

                entry += projection_.derivative2(alpha,euclidDer[beta],tangentVectors_[j*blocksize+k])*shapeFunctionValues_[j];
                entry += projection_.derivative1(alpha,tangentVectors_[j*blocksize+k]) * shapeFunctionGradients_[j][0][beta];
              }
      }

      ////////////////////////////////////////////////////////////////////
      //  The second derivative of the finite element interpolation
      ////////////////////////////////////////////////////////////////////

      CoordinateType valueWeights;
      std::copy(weights, weights+CoordinateType::dimension, valueWeights.begin());

      // Extract the transpose because we need easy access to the columns
      using DerivativeWeightsTransposed = MatrixVector::Transposed<typename LocalInterpolationRule::DerivativeType>;
      DerivativeWeightsTransposed derivativeWeightsTransposed;

      for (int i=0; i<derivativeWeightsTransposed.rows; i++)
        for (int j=0; j<derivativeWeightsTransposed.cols; j++)
          derivativeWeightsTransposed[i][j] = weights[j*gridDim+i + CoordinateType::size()];

      for (std::size_t i=0; i<nDofs; ++i)
        for (std::size_t j=0; j<orthonormalFrames_[i].size(); j++)
          for (std::size_t k=0; k<nDofs; ++k)
            for (std::size_t l=0; l<orthonormalFrames_[k].size(); l++)
            {
              // Alias name, for shorter notation
              auto& entry = secondDerivative[i][k][j][l];

              entry = 0;

              if (doValue_)
              {
                for (std::size_t alpha=0; alpha<CoordinateType::size(); alpha++)
                {
                  entry += valueWeights[alpha]
                           * projection_.derivative2(alpha, tangentVectors_[i*blocksize+j], tangentVectors_[k*blocksize+l])
                           * shapeFunctionValues_[i][0] * shapeFunctionValues_[k][0];
                }
              }

              if (doDerivative_)
              {
                for (std::size_t beta=0; beta<gridDim; beta++)
                {
                  VectorToContractWith derivativeWeightsBeta(derivativeWeightsTransposed[beta]);
                  derivativeWeightsBeta.bind(projection_.x_);

                  entry += projection_.derivative3(derivativeWeightsBeta,euclidDer[beta],tangentVectors_[i*blocksize+j],tangentVectors_[k*blocksize+l]) * shapeFunctionValues_[i][0] * shapeFunctionValues_[k][0];

                  entry += projection_.derivative2(derivativeWeightsBeta,tangentVectors_[i*blocksize+j],tangentVectors_[k*blocksize+l])
                           * (shapeFunctionValues_[i][0] * shapeFunctionGradients_[k][0][beta] + shapeFunctionValues_[k][0] * shapeFunctionGradients_[i][0][beta]);
                }
              }
            }

      // From this, compute the Hessian with respect to the manifold (which we assume here is embedded
      // isometrically in a Euclidean space.
      // For the detailed explanation of the following see: Absil, Mahoney, Trumpf, "An extrinsic look
      // at the Riemannian Hessian".

      if (!doDerivative_)
        return;

      //////////////////////////////////////////////////////////////////////////
      //  The projection of the Euclidean first derivative of the finite element interpolation
      //  onto the normal space of the unit sphere.
      //////////////////////////////////////////////////////////////////////////

      Matrix<double> normalFirstDerivative(embeddedFirstDerivative.N(), nDofs);

      // The interpolation value
      for (size_t i=0; i<CoordinateType::size(); ++i)
        for (std::size_t j=0; j<nDofs; ++j)
        {
          // The space is a sphere.  Therefore the normal at a point x is x itself.
          const auto& normal = localInterpolationRule_.coefficient(j).globalCoordinates();
          normalFirstDerivative[i][j] = projection_.derivative1(i,normal) * shapeFunctionValues_[j][0];
        }

      // The interpolation derivative with respect to space
      for (size_t alpha=0; alpha<CoordinateType::size(); alpha++)
        for (size_t beta=0; beta<gridDim; beta++)
          for (std::size_t j=0; j<nDofs; ++j)
          {
            // The space is a sphere.  Therefore the normal at a point x is x itself.
            const auto& normal = localInterpolationRule_.coefficient(j).globalCoordinates();

            // Alias name, for shorter notation
            auto& entry = normalFirstDerivative[CoordinateType::size() + alpha*gridDim+beta][j];

            entry = 0;

            for (std::size_t m=0; m<CoordinateType::size(); m++)
              entry += projection_.derivative2(alpha,m,normal)*shapeFunctionValues_[j] * euclideanDerivative_[m][beta];
            entry += projection_.derivative1(alpha,normal) * shapeFunctionGradients_[j][0][beta];
          }

      // Project Euclidean gradient onto the normal space
      Matrix<FieldMatrix<double,1,embeddedBlocksize> > projectedGradient(embeddedFirstDerivative.N(),nDofs);

      for (size_t i=0; i<embeddedFirstDerivative.N(); i++)
        for (size_t j=0; j<nDofs; j++)
          projectedGradient[i][j][0] = normalFirstDerivative[i][j] * localInterpolationRule_.coefficient(j).globalCoordinates();

      //////////////////////////////////////////////////////////////////////////////////////
      //  Further correction due to non-planar configuration space
      //  + \mathfrak{A}_x(z,P^\orth_x \partial f)
      //////////////////////////////////////////////////////////////////////////////////////

      // The configuration space is a product of spheres.  Therefore the Weingarten map
      // has only block-diagonal entries.
      for (size_t row=0; row<nDofs; row++)
      {
        for (size_t subRow=0; subRow<blocksize; subRow++)
        {
          typename TargetSpace::EmbeddedTangentVector z = orthonormalFrames_[row][subRow];

          for (size_t i=0; i<embeddedFirstDerivative.N(); i++)
          {
            auto tmp1 = localInterpolationRule_.coefficient(row).weingarten(z,projectedGradient[i][row][0]);

            typename TargetSpace::TangentVector tmp2;
            orthonormalFrames_[row].mv(tmp1,tmp2);

            for (size_t subCol=0; subCol<blocksize; subCol++)
              secondDerivative[row][row][subRow][subCol] += weights[i]* tmp2[subCol];
          }
        }

      }
    }
  };
}  // namespace Dune::GFE

#endif
