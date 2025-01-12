#ifndef DUNE_GFE_ASSEMBLERS_LOCALINTEGRALSTIFFNESS_HH
#define DUNE_GFE_ASSEMBLERS_LOCALINTEGRALSTIFFNESS_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/tuplevector.hh>

#include <dune/istl/matrix.hh>

#include <dune/geometry/quadraturerules.hh>

#include <dune/gfe/assemblers/localgeodesicfestiffness.hh>
#include <dune/gfe/assemblers/localintegralenergy.hh>
#include <dune/gfe/densities/localdensity.hh>
#include <dune/gfe/functions/interpolationderivatives.hh>


namespace Dune::GFE
{
  /** \brief This class assembles the energy gradient and Hessian for an integral over a given density function.
   *
   * Using the chain rule, gradient and Hesse matrix are split into derivatives of the
   * density and derivatives of the geometric FE interpolation functions.
   *
   * These derivatives are computed by dedicated classes, and LocalIntegralStiffness
   * combines the results.
   *
   * \tparam Basis The scalar finite element basis used to construct the interpolation rule
   * \tparam LocalInterpolationRule The rule that turns coefficients into functions
   * \tparam TargetSpace The space that the geometric finite element function maps into
   */
  template<class Basis, class LocalInterpolationRule, class TargetSpace>
  class LocalIntegralStiffness
    : public LocalGeodesicFEStiffness<Basis,TargetSpace>
  {
    using Element = typename Basis::GridView::template Codim<0>::Entity;
  public:
    constexpr static int gridDim = Basis::GridView::dimension;

    //! Dimension of the tangent spaces
    constexpr static int blocksize = TargetSpace::TangentVector::dimension;

    //! Dimension of the embedding spaces
    constexpr static int embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension;

    //! Number of the independent variables for the function evaluation
    // 1 is for the function evaluation, gridDim is for the evaluation of the derivative
    constexpr static int m = (1 + gridDim) * embeddedBlocksize;

    using GridView                    = typename Basis::GridView;
    using DT                          = typename GridView::ctype;
    using RT                          = typename TargetSpace::ctype;
    using TargetSpaceCoordinate      = typename TargetSpace::CoordinateType;
    // TODO: Take this from the interpolation rule
    using TargetSpaceDerivativeType  = FieldMatrix<double, embeddedBlocksize, gridDim>;

    using LocalCoordinate = typename GridView::template Codim<0>::Geometry::LocalCoordinate;

    LocalIntegralStiffness(const std::shared_ptr<GFE::LocalDensity<Element,TargetSpace> >& ld)
      : localDensity_(ld)
    {}

    virtual RT
    energy(const typename Basis::LocalView& localView,
           const typename Impl::LocalEnergyTypes<TargetSpace>::Coefficients& localCoefficients) const override
    {
      RT energy = 0;

      if constexpr (not Impl::LocalEnergyTypes<TargetSpace>::isProductManifold)
      {
        const auto& localFiniteElement = localView.tree().finiteElement();
        LocalInterpolationRule localInterpolationRule(localFiniteElement,localCoefficients);

        // Bind density to the element
        const auto& element = localView.element();
        localDensity_->bind(element);

        // Get a suitable quadrature rule
        int quadOrder = (element.type().isSimplex())
           ? (localFiniteElement.localBasis().order()-1) * 2
           : (localFiniteElement.localBasis().order() * gridDim - 1) * 2;

        const auto& quad = QuadratureRules<double, gridDim>::rule(localFiniteElement.type(), quadOrder);

        for (auto&& qp : quad)
        {
          // Local position of the quadrature point
          const auto& quadPos = qp.position();

          const auto integrationElement = element.geometry().integrationElement(quadPos);

          const auto geometryJacobianInverse = element.geometry().jacobianInverse(quadPos);

          if (localDensity_->dependsOnValue())
          {
            if (localDensity_->dependsOnDerivative())
            {
              auto [value, derivative] = localInterpolationRule.evaluateValueAndDerivative(quadPos);
              derivative = derivative * geometryJacobianInverse;
              energy += qp.weight() * integrationElement * (*localDensity_)(quadPos,value.globalCoordinates(),derivative);
            }
            else
            {
              auto value = localInterpolationRule.evaluate(quadPos);
              typename LocalInterpolationRule::DerivativeType dummyDerivative;
              energy += qp.weight() * integrationElement * (*localDensity_)(quadPos,value.globalCoordinates(),dummyDerivative);
            }
          }
          else
          {
            if (localDensity_->dependsOnDerivative())
            {
              typename TargetSpace::CoordinateType dummyValue;
              auto derivative = localInterpolationRule.evaluateDerivative(quadPos);
              derivative = derivative * geometryJacobianInverse;
              energy += qp.weight() * integrationElement * (*localDensity_)(quadPos,dummyValue,derivative);
            }
            else
            {
              // Density does not depend on anything.  That's rather pointless, but not an error.
            }
          }
        }
      }

      return energy;
    }

    /** \brief ProductManifolds: Compute the energy from coefficients in separate containers
     * for each factor
     */
    virtual RT
    energy(const typename Basis::LocalView& localView,
           const typename Impl::LocalEnergyTypes<TargetSpace>::CompositeCoefficients& coefficients) const override
    {
      DUNE_THROW(NotImplemented,"Energy method not implemented!");
      return 0;
    }

    /** \brief Assemble the element gradient of the energy functional */
    virtual void assembleGradient(const typename Basis::LocalView& localView,
                                  const typename Impl::LocalEnergyTypes<TargetSpace>::Coefficients& coefficients,
                                  std::vector<double>& gradient) const override
    {
      DUNE_THROW(NotImplemented,"assembleGradient method not implemented!");
    }

    /** \brief Assemble the local gradient and stiffness matrix at the current position

     */
    virtual void assembleGradientAndHessian(const typename Basis::LocalView& localView,
                                            const typename GFE::Impl::LocalStiffnessTypes<TargetSpace>::Coefficients& localCoefficients,
                                            std::vector<double>& localGradient,
                                            typename GFE::Impl::LocalStiffnessTypes<TargetSpace>::Hessian& localHessian) const override
    {
      LocalInterpolationRule localGFEFunction(localView.tree().finiteElement(),localCoefficients);

      InterpolationDerivatives<LocalInterpolationRule> interpolationDerivatives(localGFEFunction,
                                                                                localDensity_->dependsOnValue(),
                                                                                localDensity_->dependsOnDerivative());

      const size_t nDofs = localCoefficients.size();
      const size_t n = nDofs * embeddedBlocksize;

      // Precompute the orthonormal frames
      std::vector<FieldMatrix<double,blocksize,embeddedBlocksize> > orthonormalFrames;

      orthonormalFrames.resize(localCoefficients.size());
      for (size_t i=0; i<localCoefficients.size(); ++i)
        orthonormalFrames[i] = localCoefficients[i].orthonormalFrame();

      localGradient.resize(nDofs*blocksize);
      std::fill(localGradient.begin(), localGradient.end(), 0.0);

      std::vector<typename TargetSpace::EmbeddedTangentVector> localEmbeddedGradient(nDofs);
      std::fill(localEmbeddedGradient.begin(), localEmbeddedGradient.end(), 0.0);

      localHessian.setSize(nDofs*blocksize, nDofs*blocksize);
      localHessian = 0.0;

      const auto& localFiniteElement = localView.tree().finiteElement();

      // The range of input variables that the density depends on
      const size_t begin = (localDensity_->dependsOnValue()) ? 0 : TargetSpace::CoordinateType::dimension;
      const size_t end = (localDensity_->dependsOnDerivative()) ? m : TargetSpace::CoordinateType::dimension;

      // Bind density to the element
      const auto& element = localView.element();
      localDensity_->bind(element);

      // The quadrature rule
      int quadOrder = (element.type().isSimplex())
           ? (localFiniteElement.localBasis().order()-1) * 2
           : (localFiniteElement.localBasis().order() * gridDim - 1) * 2;

      const auto& quad = QuadratureRules<double, gridDim>::rule(element.type(), quadOrder);

      Matrix<double> interpolationGradient(m,n);
      Matrix<double> interpolationGradientShort(m,nDofs*blocksize);

      Matrix<FieldMatrix<double,blocksize,blocksize> > interpolationHessian(nDofs,nDofs);

      Matrix<double> hessianDensity(m,m);

      for (const auto& qp : quad)
      {
        typename TargetSpace::CoordinateType interpolationValueGlobalCoordinates;
        TargetSpaceDerivativeType interpolationDerivative;

        // We use even numbers for the FE interpolation, and odd numbers for the integral density.
        // TODO: This comment is wrong in the other specialization.
        int densityTapeNumber       = 2*MPIHelper::getCommunication().rank();
        int interpolationTapeNumber = 2*MPIHelper::getCommunication().rank()+1;

        // Evaluate the FE-function and its derivative with respect to the evaluation point
        // at the current quadrature point
        interpolationDerivatives.bind(interpolationTapeNumber,
                                      localView.element(),
                                      qp.position(),
                                      interpolationValueGlobalCoordinates,
                                      interpolationDerivative);

        // Evaluate the density function and its derivatives at the current quadrature point
        std::vector<double> densityGradient(m);
        evaluateDensity(densityTapeNumber,
                        localView,
                        qp,
                        interpolationValueGlobalCoordinates,
                        interpolationDerivative,
                        densityGradient,
                        hessianDensity);

        // Multiply the gradient and Hesse matrix of the density by the quadrature weight
        // and the integration element.  This is the cheapest place to put this multiplication.
        const auto integrationElement = element.geometry().integrationElement(qp.position());

        for (auto& g : densityGradient)
          g *= qp.weight() * integrationElement;
        hessianDensity *= qp.weight() * integrationElement;

        // Compute the derivatives of the GFE interpolation function
        interpolationDerivatives.evaluateDerivatives(interpolationTapeNumber,
                                                     densityGradient.data(),
                                                     interpolationGradient,
                                                     interpolationGradientShort,
                                                     interpolationHessian);

        // Chain rule: Multiply the derivative of the density with the derivative of the evaluation to get the total gradient, embedded
        // Store the Euclidean gradient for the conversion from the Euclidean to the Riemannian Hesse matrix.
        for (size_t i = 0; i < nDofs; i++)
          for (size_t ii=0; ii<embeddedBlocksize; ++ii)
            for (size_t j = begin; j < end; j++)
              localEmbeddedGradient[i][ii] += densityGradient[j] * interpolationGradient[j][i*embeddedBlocksize+ii];

        for (size_t i = 0; i < nDofs*blocksize; i++)
          for (size_t j = begin; j < end; j++)
            localGradient[i] += densityGradient[j] * interpolationGradientShort[j][i];

        ///////////////////////////////////////////////////////////////////////
        //  Chain rule to construct Hessian
        ///////////////////////////////////////////////////////////////////////

        // tmp = hessianDensity * interpolationGradientShort
        Matrix<double> tmp(m,nDofs*blocksize);
        tmp = 0;

        for (size_t k = begin; k < end; k++)
          for (size_t j = 0; j < nDofs*blocksize; j++)
            for (size_t kk = begin; kk < end; kk++)
              tmp[k][j] += hessianDensity[k][kk] * interpolationGradientShort[kk][j];

        // localHessian += interpolationGradientShort^T * tmp
        // (lower left triangle only)
        for (size_t k = begin; k < end; k++)
          for (size_t i = 0; i < nDofs*blocksize; i++)
            for (size_t j = 0; j <= i; j++)
              localHessian[i][j] += interpolationGradientShort[k][i] * tmp[k][j];

        // Add the contribution from the second derivative of the interpolation
        // (lower left triangle only)
        for (size_t j = 0; j < nDofs; j++)
          for (size_t jj = 0; jj <=j; jj++)
            for (size_t i = 0; i < blocksize; i++)
              for (size_t ii = 0; ii < blocksize; ii++)
                localHessian[j*blocksize+i][jj*blocksize+ii] += interpolationHessian[j][jj][i][ii];

      } // Loop over the current quadrature point

      // Copy the lower-left triangle to the upper-right triangle
      for (size_t i = 0; i < nDofs*blocksize; i++)
        for (size_t j = 0; j <= i; j++)
          localHessian[j][i] = localHessian[i][j];

      //////////////////////////////////////////////////////////////////////////////////////
      //  From this, compute the Hessian with respect to the manifold
      //  (which we assume here is embedded isometrically in a Euclidean space.
      //  We add
      //
      //       \mathfrak{A}_x(z,P^\orth_x \partial f)
      //
      //  For the detailed explanation of the following see:
      //
      //     Absil, Mahoney, Trumpf, "An extrinsic look at the Riemannian Hessian".
      //////////////////////////////////////////////////////////////////////////////////////

      // Project embedded gradient onto normal space
      std::vector<typename TargetSpace::EmbeddedTangentVector> projectedGradient(nDofs);
      for (size_t i=0; i<nDofs; i++)
        projectedGradient[i] = localCoefficients[i].projectOntoNormalSpace(localEmbeddedGradient[i]);

      // The Weingarten map has only diagonal entries
      for (size_t row=0; row<nDofs; row++)
      {
        for (size_t subRow=0; subRow<blocksize; subRow++)
        {
          typename TargetSpace::EmbeddedTangentVector z = orthonormalFrames[row][subRow];
          auto tmp1 = localCoefficients[row].weingarten(z,projectedGradient[row]);

          typename TargetSpace::TangentVector tmp2;
          orthonormalFrames[row].mv(tmp1,tmp2);

          for (size_t subCol=0; subCol<blocksize; subCol++)
            localHessian[row*blocksize+subRow][row*blocksize+subCol] += tmp2[subCol];
        }
      }
    }

    /** \brief Assemble the local gradient and stiffness matrix at the current position -- Composite version
     */
    virtual void assembleGradientAndHessian(const typename Basis::LocalView& localView,
                                            const typename GFE::Impl::LocalStiffnessTypes<TargetSpace>::CompositeCoefficients& localCoefficients,
                                            std::vector<double>& localGradient,
                                            typename GFE::Impl::LocalStiffnessTypes<TargetSpace>::CompositeHessian& localHessian) const override
    {
      DUNE_THROW(NotImplemented, "Method not implemented!");
    }

  private:

    /** \brief Trace the call to the density function, evaluated at the current quadrature point
     *         and the current function value and derivative
     *  \param[in] localView                         The local view
     *  \param[in] qp                                The quadrature point
     *  \param[in] interpolationValueGlobalCoordinates The current function value
     *  \param[in] interpolationDerivative             The current derivative
     *  \param[out] derivativeDensity                The first derivative of the density function
     *  \param[out] hessianDensity                   The second derivative of the density function
     */
    void evaluateDensity(short tapeNumber,
                         const typename Basis::LocalView& localView,
                         const QuadraturePoint<double,gridDim>& qp,
                         const TargetSpaceCoordinate& interpolationValueGlobalCoordinates,
                         const TargetSpaceDerivativeType& interpolationDerivative,
                         std::vector<double>& derivativeDensity,
                         Matrix<double>& hessianDensity) const
    {
      auto x = localView.element().geometry().global(qp.position());

      double valueDensity; // Not used
      TargetSpace interpolationValue = interpolationValueGlobalCoordinates;
      localDensity_->derivatives(x,
                                 interpolationValue,
                                 interpolationDerivative,
                                 valueDensity,
                                 derivativeDensity,
                                 hessianDensity);
    }

    // The density that is being integrated over
    const std::shared_ptr<GFE::LocalDensity<Element,TargetSpace> > localDensity_ = nullptr;
  };


  /** \brief Specialization of the LocalIntegralStiffness class for product manifolds
   *
   * \tparam FactorSpaces The factors of a ProductManifold type
   */
  template<class Basis, class LocalInterpolationRule, class ... FactorSpaces>
  class LocalIntegralStiffness<Basis,LocalInterpolationRule,ProductManifold<FactorSpaces...> >
    : public LocalGeodesicFEStiffness<Basis,ProductManifold<FactorSpaces...> >
  {
    using Element = typename Basis::GridView::template Codim<0>::Entity;
  public:

    using TargetSpace = ProductManifold<FactorSpaces...>;

    constexpr static int gridDim = Basis::GridView::dimension;

    using TargetSpace0 = typename std::tuple_element<0,TargetSpace>::type;
    using TargetSpace1 = typename std::tuple_element<1,TargetSpace>::type;

    //! Dimension of the tangent spaces
    constexpr static int blocksize0 = TargetSpace0::TangentVector::dimension;
    constexpr static int blocksize1 = TargetSpace1::TangentVector::dimension;

    //! Dimension of the embedding spaces
    constexpr static int embeddedBlocksize0 = TargetSpace0::EmbeddedTangentVector::dimension;
    constexpr static int embeddedBlocksize1 = TargetSpace1::EmbeddedTangentVector::dimension;

    //! Number of the independent variables for the function evaluation
    // 1 is for the function evaluation, gridDim is for the evaluation of the derivative
    constexpr static int m0 = (1 + gridDim) * embeddedBlocksize0;
    constexpr static int m1 = (1 + gridDim) * embeddedBlocksize1;
    constexpr static int m = m0 + m1;

    using GridView                    = typename Basis::GridView;
    using DT                          = typename GridView::ctype;
    using RT                          = typename TargetSpace0::ctype;
    using TargetSpace0Coordinate      = typename TargetSpace0::CoordinateType;
    using TargetSpace1Coordinate      = typename TargetSpace1::CoordinateType;
    using TargetSpace0DerivativeType  = FieldMatrix<double, embeddedBlocksize0, gridDim>;
    using TargetSpace1DerivativeType  = FieldMatrix<double, embeddedBlocksize1, gridDim>;

    using LocalCoordinate = typename GridView::template Codim<0>::Geometry::LocalCoordinate;

    LocalIntegralStiffness(const std::shared_ptr<GFE::LocalDensity<Element,TargetSpace> >& ld)
      : localDensity_(ld)
    {}

    virtual ~LocalIntegralStiffness() {}

    virtual RT
    energy(const typename Basis::LocalView& localView,
           const typename Impl::LocalEnergyTypes<TargetSpace>::Coefficients& coefficients) const override
    {
      DUNE_THROW(NotImplemented,"Energy method not implemented!");
      return 0;
    }

    /** \brief ProductManifolds: Compute the energy from coefficients in separate containers
     * for each factor
     */
    virtual RT
    energy(const typename Basis::LocalView& localView,
           const typename Impl::LocalEnergyTypes<TargetSpace>::CompositeCoefficients& coefficients) const override
    {
      DUNE_THROW(NotImplemented,"Energy method not implemented!");
      return 0;
    }

    /** \brief Assemble the element gradient of the energy functional */
    virtual void assembleGradient(const typename Basis::LocalView& localView,
                                  const typename Impl::LocalEnergyTypes<TargetSpace>::Coefficients& coefficients,
                                  std::vector<double>& gradient) const override
    {
      DUNE_THROW(NotImplemented,"assembleGradient method not implemented!");
    }

    /** \brief Assemble the local gradient and stiffness matrix at the current position

     */
    virtual void assembleGradientAndHessian(const typename Basis::LocalView& localView,
                                            const typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::Coefficients& coefficients,
                                            std::vector<double>& localGradient,
                                            typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::Hessian& localHessian) const override
    {
      DUNE_THROW(NotImplemented, "Method not implemented!");
    }

    /** \brief Assemble the local gradient and stiffness matrix at the current position -- Composite version
     */
    virtual void assembleGradientAndHessian(const typename Basis::LocalView& localView,
                                            const typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::CompositeCoefficients& coefficients,
                                            std::vector<double>& localGradient,
                                            typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::CompositeHessian& localHessian) const override;

  protected:
    const std::shared_ptr<GFE::LocalDensity<Element,TargetSpace> > localDensity_ = nullptr;

  private:

    /** \brief Trace the call to the density function, evaluated at the current quadrature point
     *         and the current function value and derivative
     *  \param[in] localView                         The local view
     *  \param[in] qp                                The quadrature point
     *  \param[in] deformationValueGlobalCoordinates The current function value - part 0
     *  \param[in] orientationValueGlobalCoordinates The current function value - part 1
     *  \param[in] deformationDerivative             The current derivative - part 0
     *  \param[in] orientationDerivative             The current derivative - part 1
     *  \param[out] derivativeDensity                The first derivative of the density function
     *  \param[out] hessianDensity                   The second derivative of the density function
     */
    void evaluateDensity(short tapeNumber,
                         const typename Basis::LocalView& localView,
                         const QuadraturePoint<double,gridDim>& qp,
                         const TargetSpace0Coordinate& deformationValueGlobalCoordinates,
                         const TargetSpace1Coordinate& orientationValueGlobalCoordinates,
                         const TargetSpace0DerivativeType& deformationDerivative,
                         const TargetSpace1DerivativeType& orientationDerivative,
                         std::vector<double>& derivativeDensity,
                         Matrix<double>& hessianDensity) const
    {
      using namespace Dune::Indices;
      auto x = localView.element().geometry().global(qp.position());

      double valueDensity; // Not used

      TargetSpace interpolationValue;
      interpolationValue[_0] = deformationValueGlobalCoordinates;
      interpolationValue[_1] = orientationValueGlobalCoordinates;

      using TargetSpaceDerivative = FieldMatrix<double, embeddedBlocksize0 + embeddedBlocksize1, gridDim>;

      TargetSpaceDerivative interpolationDerivative;

      std::size_t i = 0;

      for (auto && row : deformationDerivative)
        interpolationDerivative[i++] = row;

      for (auto && row : orientationDerivative)
        interpolationDerivative[i++] = row;

      localDensity_->derivatives(x,
                                 interpolationValue,
                                 interpolationDerivative,
                                 valueDensity,
                                 derivativeDensity,
                                 hessianDensity);
    }

  };

  template<class Basis, class LocalInterpolationRule, class ... FactorSpaces>
  void LocalIntegralStiffness<Basis, LocalInterpolationRule, ProductManifold<FactorSpaces...> >::
  assembleGradientAndHessian(const typename Basis::LocalView& localView,
                             const typename Impl::LocalStiffnessTypes<TargetSpace>::CompositeCoefficients& localCoefficients,
                             std::vector<double>& localGradient,
                             typename Impl::LocalStiffnessTypes<TargetSpace>::CompositeHessian& localHessian) const
  {
    using namespace Dune::Indices;

    using DeformationLocalInterpolationRule = typename std::tuple_element<0,LocalInterpolationRule>::type;
    using OrientationLocalInterpolationRule = typename std::tuple_element<1,LocalInterpolationRule>::type;

    DeformationLocalInterpolationRule localDeformationGFEFunction(localView.tree().child(_0,0).finiteElement(),localCoefficients[_0]);
    OrientationLocalInterpolationRule localOrientationGFEFunction(localView.tree().child(_1,0).finiteElement(),localCoefficients[_1]);

    InterpolationDerivatives<DeformationLocalInterpolationRule> deformationInterpolationDerivatives(localDeformationGFEFunction,
                                                                                                    localDensity_->dependsOnValue(0),
                                                                                                    localDensity_->dependsOnDerivative(0));
    InterpolationDerivatives<OrientationLocalInterpolationRule> orientationInterpolationDerivatives(localOrientationGFEFunction,
                                                                                                    localDensity_->dependsOnValue(1),
                                                                                                    localDensity_->dependsOnDerivative(1));

    size_t nDofs0 = localCoefficients[_0].size();
    size_t nDofs1 = localCoefficients[_1].size();
    const size_t n0 = nDofs0 * embeddedBlocksize0;
    const size_t n1 = nDofs1 * embeddedBlocksize1;

    // Precompute the orthonormal frames
    TupleVector<std::vector<FieldMatrix<double,blocksize0,embeddedBlocksize0> >,
        std::vector<FieldMatrix<double,blocksize1,embeddedBlocksize1> > > orthonormalFrames;

    orthonormalFrames[_0].resize(localCoefficients[_0].size());
    for (size_t i=0; i<localCoefficients[_0].size(); ++i)
      orthonormalFrames[_0][i] = localCoefficients[_0][i].orthonormalFrame();

    orthonormalFrames[_1].resize(localCoefficients[_1].size());
    for (size_t i=0; i<localCoefficients[_1].size(); ++i)
      orthonormalFrames[_1][i] = localCoefficients[_1][i].orthonormalFrame();

    localGradient.resize(nDofs0*blocksize0 + nDofs1*blocksize1);
    std::fill(localGradient.begin(), localGradient.end(), 0.0);

    std::vector<typename TargetSpace0::EmbeddedTangentVector> localEmbeddedGradient0(nDofs0);
    std::vector<typename TargetSpace1::EmbeddedTangentVector> localEmbeddedGradient1(nDofs1);
    std::fill(localEmbeddedGradient0.begin(), localEmbeddedGradient0.end(), 0.0);
    std::fill(localEmbeddedGradient1.begin(), localEmbeddedGradient1.end(), 0.0);

    localHessian[0][0].setSize(nDofs0*blocksize0, nDofs0*blocksize0);
    localHessian[0][1].setSize(nDofs0*blocksize0, nDofs1*blocksize1);
    localHessian[1][0].setSize(nDofs1*blocksize1, nDofs0*blocksize0);
    localHessian[1][1].setSize(nDofs1*blocksize1, nDofs1*blocksize1);
    for (std::size_t i=0; i<localHessian.N(); i++)
      for (std::size_t j=0; j<localHessian.M(); j++)
        localHessian[i][j] = 0.0;

    const auto& deformationLocalFiniteElement = localView.tree().child(_0,0).finiteElement();

    // Bind density to the element
    const auto& element = localView.element();
    localDensity_->bind(element);

    // Get a suitable quadrature rule
    int quadOrder = (element.type().isSimplex()) ? deformationLocalFiniteElement.localBasis().order()
                                                 : deformationLocalFiniteElement.localBasis().order() * gridDim;

    const auto& quad = QuadratureRules<double, gridDim>::rule(element.type(), quadOrder);

    Matrix<double> deformationInterpolationGradient(m0, n0);
    Matrix<double> deformationInterpolationGradientShort(m0, nDofs0*blocksize0);

    Matrix<FieldMatrix<double,blocksize0,blocksize0> > deformationInterpolationHessian(nDofs0,nDofs0);

    // TODO: Rename this to riemannianSomething
    Matrix<double> orientationInterpolationGradient(m1, n1);
    Matrix<double> orientationInterpolationGradientShort(m1, nDofs1*blocksize1);

    Matrix<FieldMatrix<double,blocksize1,blocksize1> > orientationInterpolationHessian(nDofs1,nDofs1);

    Matrix<double> hessianDensity(m,m);

    for (const auto& qp : quad)
    {
      typename TargetSpace0::CoordinateType deformationValueGlobalCoordinates;
      typename TargetSpace1::CoordinateType orientationValueGlobalCoordinates;

      TargetSpace0DerivativeType deformationDerivative;
      TargetSpace1DerivativeType orientationDerivative;

      // We use even numbers for the FE interpolation, and odd numbers for the integral density.
      const std::size_t numberOfFactorSpaces = 2;
      int densityTapeNumber     = (numberOfFactorSpaces+1)*MPIHelper::getCommunication().rank();
      int deformationTapeNumber = (numberOfFactorSpaces+1)*MPIHelper::getCommunication().rank()+1;
      int orientationTapeNumber = (numberOfFactorSpaces+1)*MPIHelper::getCommunication().rank()+2;

      // Evaluate the FE-functions and their derivatives with respect to the evaluation point
      // at the current quadrature point
      deformationInterpolationDerivatives.bind(deformationTapeNumber,
                                               localView.element(),
                                               qp.position(),
                                               deformationValueGlobalCoordinates,
                                               deformationDerivative);

      orientationInterpolationDerivatives.bind(orientationTapeNumber,
                                               localView.element(),
                                               qp.position(),
                                               orientationValueGlobalCoordinates,
                                               orientationDerivative);

      // Second trace - evaluate the density function for the FE-function value and their derivative at the current QP
      std::vector<double> densityGradient(m);
      evaluateDensity(densityTapeNumber,
                      localView,
                      qp,
                      deformationValueGlobalCoordinates,
                      orientationValueGlobalCoordinates,
                      deformationDerivative,
                      orientationDerivative,
                      densityGradient,
                      hessianDensity);

      // Multiply the gradient and Hesse matrix of the density by the quadrature weight
      // and the integration element.  This is the cheapest place to put this multiplication.
      const auto integrationElement = element.geometry().integrationElement(qp.position());

      for (auto& g : densityGradient)
        g *= qp.weight() * integrationElement;
      hessianDensity *= qp.weight() * integrationElement;

      // Compute the derivatives of the GFE interpolation function
      const int deformationOffset = 0;
      const int orientationOffset = m0;

      deformationInterpolationDerivatives.evaluateDerivatives(deformationTapeNumber,
                                                              densityGradient.data() + deformationOffset,
                                                              deformationInterpolationGradient,
                                                              deformationInterpolationGradientShort,
                                                              deformationInterpolationHessian);

      orientationInterpolationDerivatives.evaluateDerivatives(orientationTapeNumber,
                                                              densityGradient.data() + orientationOffset,
                                                              orientationInterpolationGradient,
                                                              orientationInterpolationGradientShort,
                                                              orientationInterpolationHessian);

      // Chain rule: Multiply the derivative of the density with the derivative of the evaluation to get the total gradient, embedded
      // Store the Euclidean gradient for the conversion from Euclidean Hesse matrix to Riemannian Hesse matrix
      for (size_t i = 0; i < nDofs0; i++)
        for (size_t ii=0; ii<embeddedBlocksize0; ++ii)
          for (size_t j = 0; j < m0; j++)
            localEmbeddedGradient0[i][ii] += densityGradient[j] * deformationInterpolationGradient[j][i*embeddedBlocksize0+ii];

      for (size_t i = 0; i < nDofs1; i++)
        for (size_t ii=0; ii<embeddedBlocksize1; ++ii)
          for (size_t j = 0; j < m1; j++)
            localEmbeddedGradient1[i][ii] += densityGradient[m0 + j] * orientationInterpolationGradient[j][i*embeddedBlocksize1+ii];

      for (size_t i = 0; i < nDofs0*blocksize0; i++)
        for (size_t j = 0; j < m0; j++)
          localGradient[i] += densityGradient[j] * deformationInterpolationGradientShort[j][i];

      for (size_t i = 0; i < nDofs1*blocksize1; i++)
        for (size_t j = 0; j < m1; j++)
          localGradient[nDofs0*blocksize0 + i] += densityGradient[m0 + j] * orientationInterpolationGradientShort[j][i];

      ///////////////////////////////////////////////////////////////////////
      //  Chain rule to construct Hessian
      ///////////////////////////////////////////////////////////////////////

      // Part one: derivative of the FE interpolation times Hessian of the density times derivative of the FE interpolation
      // ------------------------------------------------------------------------------------------------------------------

      // The range of input variables that the density depends on
      const size_t begin0 = (localDensity_->dependsOnValue(0)) ? 0 : TargetSpace0::CoordinateType::dimension;
      const size_t end0 = (localDensity_->dependsOnDerivative(0)) ? m0 : TargetSpace0::CoordinateType::dimension;

      const size_t begin1 = (localDensity_->dependsOnValue(1)) ? 0 : TargetSpace1::CoordinateType::dimension;
      const size_t end1 = (localDensity_->dependsOnDerivative(1)) ? m1 : TargetSpace0::CoordinateType::dimension;

      // tmp00 = hessianDensity[_0][_0] * deformationInterpolationGradientShort
      Matrix<double> tmp00(m0,nDofs0*blocksize0);
      tmp00 = 0;

      for (size_t k = begin0; k < end0; k++)
        for (size_t j = 0; j < nDofs0*blocksize0; j++)
          for (size_t kk = begin0; kk < end0; kk++)
            tmp00[k][j] += hessianDensity[k][kk] * deformationInterpolationGradientShort[kk][j];

      // localHessian[_0][_0] += deformationInterpolationGradientShort^T * tmp00
      for (size_t k = begin0; k < end0; k++)
        for (size_t i = 0; i < nDofs0*blocksize0; i++)
          for (size_t j = 0; j <= i; j++)
            localHessian[_0][_0][i][j] += deformationInterpolationGradientShort[k][i] * tmp00[k][j];

      // tmp10 = hessianDensity[_1][_0] * deformationInterpolationGradientShort
      Matrix<double> tmp10(m1,nDofs0*blocksize0);
      tmp10 = 0;

      for (size_t k = begin1; k < end1; k++)
        for (size_t j = 0; j < nDofs0*blocksize0; j++)
          for (size_t kk = begin0; kk < end0; kk++)
            tmp10[k][j] += hessianDensity[k+m0][kk] * deformationInterpolationGradientShort[kk][j];

      // localHessian[_1][_0] += orientationInterpolationGradientShort^T * tmp10
      for (size_t k = begin1; k < end1; k++)
        for (size_t i = 0; i < nDofs1*blocksize1; i++)
          for (size_t j = 0; j < nDofs0*blocksize0; j++)
            localHessian[_1][_0][i][j] += orientationInterpolationGradientShort[k][i] * tmp10[k][j];

      // tmp11 = hessianDensity[_1][_1] * orientationInterpolationGradientShort
      Matrix<double> tmp11(m1,nDofs1*blocksize1);
      tmp11 = 0;

      for (size_t k = begin1; k < end1; k++)
        for (size_t j = 0; j < nDofs1*blocksize1; j++)
          for (size_t kk = begin1; kk < end1; kk++)
            tmp11[k][j] += hessianDensity[k+m0][kk+m0] * orientationInterpolationGradientShort[kk][j];

      // localHessian[_1][_1] += orientationInterpolationGradientShort^T * tmp11
      // (lower-left triangle only)
      for (size_t k = begin1; k < end1; k++)
        for (size_t i = 0; i < nDofs1*blocksize1; i++)
          for (size_t j = 0; j <= i; j++)
            localHessian[_1][_1][i][j] += orientationInterpolationGradientShort[k][i] * tmp11[k][j];

      // Part two: Gradient of the density times second derivative of the FE interpolation
      // ---------------------------------------------------------------------------------

      for (std::size_t i=0; i<nDofs0; i++)
        for (std::size_t j=0; j<nDofs0; j++)
          for (std::size_t ii=0; ii<blocksize0; ii++)
            for (std::size_t jj=0; jj<blocksize0; jj++)
              localHessian[0][0][i*blocksize0+ii][j*blocksize0+jj] += deformationInterpolationHessian[i][j][ii][jj];

      for (std::size_t i=0; i<nDofs1; i++)
        for (std::size_t j=0; j<nDofs1; j++)
          for (std::size_t ii=0; ii<blocksize1; ii++)
            for (std::size_t jj=0; jj<blocksize1; jj++)
              localHessian[1][1][i*blocksize1+ii][j*blocksize1+jj] += orientationInterpolationHessian[i][j][ii][jj];

    } // Loop over the current quadrature point

    // Copy the lower-left triangle to the upper-right triangle
    for (size_t i = 0; i < nDofs0*blocksize0; i++)
      for (size_t j = 0; j <= i; j++)
        localHessian[_0][_0][j][i] = localHessian[_0][_0][i][j];

    for (size_t i = 0; i < nDofs1*blocksize1; i++)
      for (size_t j = 0; j < nDofs0*blocksize0; j++)
        localHessian[_0][_1][j][i] = localHessian[_1][_0][i][j];

    for (size_t i = 0; i < nDofs1*blocksize1; i++)
      for (size_t j = 0; j <= i; j++)
        localHessian[_1][_1][j][i] = localHessian[_1][_1][i][j];

    //////////////////////////////////////////////////////////////////////////////////////
    //  From this, compute the Hessian with respect to the manifold
    //  (which we assume here is embedded isometrically in a Euclidean space.
    //  We add
    //
    //       \mathfrak{A}_x(z,P^\orth_x \partial f)
    //
    //  For the detailed explanation of the following see:
    //
    //     Absil, Mahoney, Trumpf, "An extrinsic look at the Riemannian Hessian".
    //////////////////////////////////////////////////////////////////////////////////////

    // Project embedded gradient onto normal space
    std::vector<typename TargetSpace0::EmbeddedTangentVector> projectedGradient0(nDofs0);
    for (size_t i=0; i<nDofs0; i++)
      projectedGradient0[i] = localCoefficients[_0][i].projectOntoNormalSpace(localEmbeddedGradient0[i]);

    std::vector<typename TargetSpace1::EmbeddedTangentVector> projectedGradient1(nDofs1);
    for (size_t i=0; i<nDofs1; i++)
      projectedGradient1[i] = localCoefficients[_1][i].projectOntoNormalSpace(localEmbeddedGradient1[i]);

    // The Weingarten map has only diagonal entries
    // TODO: Can we make this quicker if TargetSpace is a RealTuple?
    for (size_t row=0; row<nDofs0; row++) {

      for (size_t subRow=0; subRow<blocksize0; subRow++) {

        typename TargetSpace0::EmbeddedTangentVector z = orthonormalFrames[_0][row][subRow];
        typename TargetSpace0::EmbeddedTangentVector tmp1 = localCoefficients[_0][row].weingarten(z,projectedGradient0[row]);

        typename TargetSpace0::TangentVector tmp2;
        orthonormalFrames[_0][row].mv(tmp1,tmp2);

        for (size_t subCol=0; subCol<blocksize0; subCol++)
          localHessian[0][0][row*blocksize0+subRow][row*blocksize0+subCol] += tmp2[subCol];
      }

    }

    for (size_t row=0; row<nDofs1; row++) {

      for (size_t subRow=0; subRow<blocksize1; subRow++) {

        typename TargetSpace1::EmbeddedTangentVector z = orthonormalFrames[_1][row][subRow];
        typename TargetSpace1::EmbeddedTangentVector tmp1 = localCoefficients[_1][row].weingarten(z,projectedGradient1[row]);

        typename TargetSpace1::TangentVector tmp2;
        orthonormalFrames[_1][row].mv(tmp1,tmp2);

        for (size_t subCol=0; subCol<blocksize1; subCol++)
          localHessian[1][1][row*blocksize1+subRow][row*blocksize1+subCol] += tmp2[subCol];
      }

    }

  }

} // namespace Dune::GFE

#endif
