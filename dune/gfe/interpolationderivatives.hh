#ifndef DUNE_GFE_INTERPOLATIONDERIVATIVES_HH
#define DUNE_GFE_INTERPOLATIONDERIVATIVES_HH

// Includes for the ADOL-C automatic differentiation library
#include <adolc/adolc.h>

#include <dune/matrix-vector/transpose.hh>

#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/localprojectedfefunction.hh>
#include <dune/gfe/spaces/realtuple.hh>

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
      ALocalInterpolationRule localGFEFunction(scalarFiniteElement,localAConfiguration);

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
  template <int gridDim, typename field_type, typename LocalFiniteElement,int dim>
  class InterpolationDerivatives<LocalGeodesicFEFunction<gridDim, field_type, LocalFiniteElement, RealTuple<field_type,dim> > >
  {
    using LocalInterpolationRule = LocalGeodesicFEFunction<gridDim, field_type, LocalFiniteElement, RealTuple<field_type,dim> >;
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
    template <typename Element>
    void bind(short tapeNumber,
              const Element& element,
              const typename Element::Geometry::LocalCoordinate& localPos,
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
          for (int k=0; k<gridDim; ++k)
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
  template <int gridDim, typename field_type, typename LocalFiniteElement,int dim>
  class InterpolationDerivatives<LocalProjectedFEFunction<gridDim, field_type, LocalFiniteElement, RealTuple<field_type,dim> > >
  {
    // TODO: The implementation here would be identical to the geodesic FE case
    using LocalInterpolationRule = LocalProjectedFEFunction<gridDim, field_type, LocalFiniteElement, RealTuple<field_type,dim> >;
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
    template <typename Element>
    void bind(short tapeNumber,
              const Element& element,
              const typename Element::Geometry::LocalCoordinate& localPos,
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
          for (int k=0; k<gridDim; ++k)
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
}  // namespace Dune::GFE

#endif
