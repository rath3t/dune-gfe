#ifndef DUNE_GFE_FUNCTIONS_LOCALPROJECTEDFEFUNCTION_HH
#define DUNE_GFE_FUNCTIONS_LOCALPROJECTEDFEFUNCTION_HH

#include <vector>

#include <dune/common/fvector.hh>

#include <dune/geometry/type.hh>

#include <dune/gfe/linearalgebra.hh>
#include <dune/gfe/polardecomposition.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/rotation.hh>

namespace Dune::GFE
{

  /** \brief Interpolate in an embedding Euclidean space, and project back onto the Riemannian manifold
   *
   * \tparam Basis The basis used for the interpolation in the embedding space.
   * \tparam TS The manifold that the function takes its values in
   * \tparam conforming Geometrical conformity of the functions (omits projections when set to false)
   *
   * \note Even though the embedding space is typically more than one-dimensional,
   * we allow this basis to be scalar-valued.
   */
  template <class Basis, class TS, bool conforming=true>
  class LocalProjectedFEFunction
  {
  public:
    using TargetSpace=TS;
  private:
    using Element = typename Basis::GridView::template Codim<0>::Entity;
    using LocalCoordinate = typename Element::Geometry::LocalCoordinate;
    static constexpr auto dim = LocalCoordinate::size();

    using LocalFiniteElement = typename Basis::LocalView::Tree::FiniteElement;
    using LocalBasis = typename LocalFiniteElement::Traits::LocalBasisType;
    using ScalarFERange = typename LocalBasis::Traits::RangeType;
    using ScalarFEJacobian = typename LocalBasis::Traits::JacobianType;

    typedef typename TargetSpace::ctype RT;

    typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;
    static const int embeddedDim = EmbeddedTangentVector::dimension;

    /** \brief Short-cut to the local basis
     */
    const auto& localBasis() const
    {
      return localBasisView_.tree().finiteElement().localBasis();
    }

  public:

    /** \brief The type used for derivatives */
    typedef Dune::FieldMatrix<RT, embeddedDim, dim> DerivativeType;

    /** \brief Constructor with a given scalar basis
     *
     * \param basis The basis that implements the weights for the Riemannian
     * center of mass
     */
    LocalProjectedFEFunction(const Basis& basis)
      : basis_(basis)
      , localBasisView_(basis_)
    {}

    /** \brief Copy constructor
     */
    LocalProjectedFEFunction(const LocalProjectedFEFunction& other)
      : basis_(other.basis_)
      // Do NOT copy the localView: It contains a reference to other.basis_.
      , localBasisView_(basis_.localView())
      , coefficients_(other.coefficients_)
    {
      if (other.localBasisView_.bound())
        localBasisView_.bind(other.localBasisView_.element());
    }

    /** \brief Move constructor
     */
    LocalProjectedFEFunction(LocalProjectedFEFunction&& other)
      : basis_(std::move(other.basis_))
      // Do NOT copy the localView: It contains a reference to other.basis_.
      , localBasisView_(basis_.localView())
      , coefficients_(std::move(other.coefficients_))
    {
      if (other.localBasisView_.bound())
        localBasisView_.bind(other.localBasisView_.element());
    }

    /** \brief Copy assignment
     */
    LocalProjectedFEFunction& operator=(const LocalProjectedFEFunction& other)
    {
      basis_ = other.basis_;
      // Do NOT copy the localView: It contains a reference to other.basis_.
      localBasisView_ = basis_.localView();
      coefficients_ = other.coefficients_;
      if (other.localBasisView_.bound())
        localBasisView_.bind(other.localBasisView_.element());
    }

    /** \brief Move assignment
     */
    LocalProjectedFEFunction& operator=(LocalProjectedFEFunction&& other)
    {
      basis_ = std::move(other.basis_);
      // Do NOT move the localView: It contains a reference to other.basis_.
      localBasisView_ = basis_.localView();
      coefficients_ = std::move(other.coefficients_);
      if (other.localBasisView_.bound())
        localBasisView_.bind(other.localBasisView_.element());
    }

    /** \brief Bind the local function to a grid element and a set of coefficients
     *
     * \param element The grid element
     * \param coefficients Values to be interpolated
     */
    void bind(const Element& element,
              const std::vector<TargetSpace>& coefficients)
    {
      localBasisView_.bind(element);
      coefficients_ = coefficients;

      assert(localBasisView_.tree().finiteElement().size() == coefficients.size());
    }

    /** \brief Rebind the FEFunction to another TargetSpace */
    template<class U>
    struct rebind
    {
      using other = LocalProjectedFEFunction<Basis,U,conforming>;
    };

    /** \brief The number of Lagrange points */
    unsigned int size() const
    {
      return localBasisView_.size();
    }

    /** \brief The type of the reference element */
    Dune::GeometryType type() const
    {
      return localBasisView_.element().type();
    }

    /** \brief The finite element basis used to interpolate in the embedding space
     */
    const Basis& globalBasis() const
    {
      return basis_;
    }

    /** \brief The scalar finite element used as the interpolation weights
     *
     * \note This method was added for InterpolationDerivatives, which needs it
     * to construct a copy of a LocalGeodesicFEFunction with ADOL-C's adouble
     * number type.  This is not optimal, because the localFiniteElement
     * really is an implementation detail of LocalGeodesicFEFunction and
     * should not be needed just to copy an entire object.  Other non-Euclidean
     * interpolation rules may not have such a finite element at all.
     * Therefore, this method may disappear again eventually.
     */
    const LocalFiniteElement& localFiniteElement() const
    {
      return localBasisView_.tree().finiteElement();
    }

    /** \brief Evaluate the function */
    auto evaluate(const LocalCoordinate& local) const;

    /** \brief Evaluate the derivative of the function */
    DerivativeType evaluateDerivative(const LocalCoordinate& local) const;

    /** \brief Evaluate the derivative of the function, if you happen to know the function value (much faster!)
     *        \param local Local coordinates in the reference element where to evaluate the derivative
     *        \param q Value of the local gfe function at 'local'.  If you provide something wrong here the result will be wrong, too!
     *
     * \note This method is only usable in the conforming setting, because it requires the caller
     * to hand over the interpolation value as a TargetSpace object.
     */
    DerivativeType evaluateDerivative(const LocalCoordinate& local,
                                      const TargetSpace& q) const;

    /** \brief Evaluate the value and the derivative of the interpolation function
     *
     * \return A std::pair containing the value and the first derivative of the interpolation function.
     * If the interpolation is conforming then the first member of the pair will be a TargetSpace.
     * Otherwise it will be a RealTuple.
     */
    auto evaluateValueAndDerivative(const LocalCoordinate& local) const;

    /** \brief Get the i'th base coefficient. */
    const TargetSpace& coefficient(int i) const
    {
      return coefficients_[i];
    }
  private:

    /** \brief The finite element basis for the interpolation in the embedding space
     */
    const Basis basis_;

    /** \brief The local finite element used for the interpolation in the embedding space
     */
    typename Basis::LocalView localBasisView_;

    /** \brief The coefficient vector */
    std::vector<TargetSpace> coefficients_;

  };

  template <class Basis, class TargetSpace, bool conforming>
  auto LocalProjectedFEFunction<Basis,TargetSpace,conforming>::
  evaluate(const LocalCoordinate& local) const
  {
    // Compute value of FE function in embedding space
    std::vector<ScalarFERange> w;
    localBasis().evaluateFunction(local,w);

    typename TargetSpace::CoordinateType c(0);
    for (size_t i=0; i<coefficients_.size(); i++)
      c.axpy(w[i][0], coefficients_[i].globalCoordinates());

    // Possibly project onto target space
    if constexpr (conforming)
      return TargetSpace::projectOnto(c);
    else
      return (RealTuple<RT, TargetSpace::CoordinateType::dimension>)c;

  }

  template <class Basis, class TargetSpace,bool conforming>
  typename LocalProjectedFEFunction<Basis,TargetSpace,conforming>::DerivativeType
  LocalProjectedFEFunction<Basis,TargetSpace,conforming>::
  evaluateDerivative(const LocalCoordinate& local) const
  {
    if constexpr(conforming)
    {
      // the function value at the point where we are evaluating the derivative
      TargetSpace q = evaluate(local);

      // Actually compute the derivative
      return evaluateDerivative(local, q);
    }
    else {
      std::vector<ScalarFEJacobian> wDer;
      localBasis().evaluateJacobian(local, wDer);

      Dune::FieldMatrix<RT, embeddedDim, dim> derivative(0);
      for (size_t i = 0; i < embeddedDim; i++)
        for (size_t j = 0; j < dim; j++)
          for (size_t k = 0; k < coefficients_.size(); k++)
            derivative[i][j] += wDer[k][0][j] * coefficients_[k].globalCoordinates()[i];

      return derivative;
    }
  }

  template <class Basis, class TargetSpace,bool conforming>
  typename LocalProjectedFEFunction<Basis,TargetSpace,conforming>::DerivativeType
  LocalProjectedFEFunction<Basis,TargetSpace,conforming>::
  evaluateDerivative(const LocalCoordinate& local, const TargetSpace& q) const
  {
    // Compute value and derivative of FE approximation
    std::vector<ScalarFERange> w;
    localBasis().evaluateFunction(local,w);

    std::vector<ScalarFEJacobian> wDer;
    localBasis().evaluateJacobian(local,wDer);

    typename TargetSpace::CoordinateType embeddedInterpolation(0);
    for (size_t i=0; i<coefficients_.size(); i++)
      embeddedInterpolation.axpy(w[i][0], coefficients_[i].globalCoordinates());

    Dune::FieldMatrix<RT,embeddedDim,dim> derivative(0);
    for (size_t i=0; i<embeddedDim; i++)
      for (size_t j=0; j<dim; j++)
        for (size_t k=0; k<coefficients_.size(); k++)
          derivative[i][j] += wDer[k][0][j] * coefficients_[k].globalCoordinates()[i];

    auto derivativeOfProjection = TargetSpace::derivativeOfProjection(embeddedInterpolation);

    return derivativeOfProjection*derivative;
  }

  template <class Basis, class TargetSpace,bool conforming>
  auto
  LocalProjectedFEFunction<Basis,TargetSpace,conforming>::
  evaluateValueAndDerivative(const LocalCoordinate& local) const
  {
    // Construct the type of the result -- it depends on whether the interpolation
    // is conforming or not.
    using Value = std::conditional_t<conforming,TargetSpace,RealTuple<RT,embeddedDim> >;

    std::pair<Value,DerivativeType> result;

    ///////////////////////////////////////////////////////////
    //  Compute the value of the interpolation function
    ///////////////////////////////////////////////////////////

    std::vector<ScalarFERange> w;
    localBasis().evaluateFunction(local,w);

    typename TargetSpace::CoordinateType embeddedInterpolation(0);
    for (size_t i=0; i<coefficients_.size(); i++)
      embeddedInterpolation.axpy(w[i][0], coefficients_[i].globalCoordinates());

    if constexpr (conforming)
      result.first = TargetSpace::projectOnto(embeddedInterpolation);
    else
      result.first = (RealTuple<RT, TargetSpace::CoordinateType::dimension>)embeddedInterpolation;

    ///////////////////////////////////////////////////////////
    //  Compute the derivative of the interpolation function
    ///////////////////////////////////////////////////////////

    // Compute the interpolation in the surrounding space
    std::vector<ScalarFEJacobian> wDer;
    localBasis().evaluateJacobian(local,wDer);

    result.second = 0.0;
    for (size_t i=0; i<embeddedDim; i++)
      for (size_t j=0; j<dim; j++)
        for (size_t k=0; k<coefficients_.size(); k++)
          result.second[i][j] += wDer[k][0][j] * coefficients_[k].globalCoordinates()[i];

    // The derivative of the projection onto the manifold
    if constexpr(conforming)
      result.second = TargetSpace::derivativeOfProjection(embeddedInterpolation) * result.second;

    return result;
  }


  /** \brief Interpolate in an embedding Euclidean space, and project back onto the Riemannian manifold -- specialization for SO(3)
   *
   * \tparam Basis The basis used for the interpolation in the embedding space.
   * \tparam field_type The number type used for function values
   * \tparam conforming Geometrical conformity of the functions (omits projections when set to false)
   *
   * \note Even though the embedding space is 9-dimensional, we allow this basis
   * to be scalar-valued.
   */
  template <class Basis, class field_type, bool conforming>
  class LocalProjectedFEFunction<Basis,Rotation<field_type,3>,conforming>
  {
  public:
    typedef Rotation<field_type,3> TargetSpace;
  private:
    using Element = typename Basis::GridView::template Codim<0>::Entity;
    using LocalCoordinate = typename Element::Geometry::LocalCoordinate;
    static constexpr auto dim = LocalCoordinate::size();

    using LocalFiniteElement = typename Basis::LocalView::Tree::FiniteElement;
    using LocalBasis = typename LocalFiniteElement::Traits::LocalBasisType;
    using ScalarFERange = typename LocalBasis::Traits::RangeType;
    using ScalarFEJacobian = typename LocalBasis::Traits::JacobianType;

    typedef typename TargetSpace::ctype RT;

    typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;
    static const int embeddedDim = EmbeddedTangentVector::dimension;

    /**
     * \param A The argument of the projection
     * \param polar The image of the projection, i.e., the polar factor of A
     */
    static std::array<std::array<FieldMatrix<field_type,3,3>, 3>, 3> derivativeOfProjection(const FieldMatrix<field_type,3,3>& A,
                                                                                            FieldMatrix<field_type,3,3>& polar)
    {
      std::array<std::array<FieldMatrix<field_type,3,3>, 3>, 3> result;

      for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
          for (int k=0; k<3; k++)
            for (int l=0; l<3; l++)
              result[i][j][k][l] = (i==k) and (j==l);

      polar = A;

      // Use Heron's method
      const size_t maxIterations = 100;
      double tol = 0.001;
      for (size_t iteration=0; iteration<maxIterations; iteration++)
      {
        auto oldPolar = polar;
        auto polarInvert = polar;
        polarInvert.invert();
        for (size_t i=0; i<polar.N(); i++)
          for (size_t j=0; j<polar.M(); j++)
            polar[i][j] = 0.5 * (polar[i][j] + polarInvert[j][i]);

        // Alternative name to align the code better with a description in a math text
        const auto& dQT = result;

        // Multiply from the right with Q^{-T}
        decltype(result) tmp2;
        for (int i=0; i<3; i++)
          for (int j=0; j<3; j++)
            for (int k=0; k<3; k++)
              for (int l=0; l<3; l++)
                tmp2[i][j][k][l] = 0.0;

        for (int i=0; i<3; i++)
          for (int j=0; j<3; j++)
            for (int k=0; k<3; k++)
              for (int l=0; l<3; l++)
                for (int m=0; m<3; m++)
                  for (int o=0; o<3; o++)
                    tmp2[i][j][k][l] += polarInvert[m][i] * dQT[o][m][k][l] * polarInvert[j][o];

        for (int i=0; i<3; i++)
          for (int j=0; j<3; j++)
            for (int k=0; k<3; k++)
              for (int l=0; l<3; l++)
                result[i][j][k][l] = 0.5 * (result[i][j][k][l] - tmp2[i][j][k][l]);

        oldPolar -= polar;
        if (oldPolar.frobenius_norm() < tol) {
          break;
        }
      }

      return result;
    }

    /** \brief Short-cut to the local basis
     */
    const auto& localBasis() const
    {
      return localBasisView_.tree().finiteElement().localBasis();
    }

  public:

    /** \brief The type used for derivatives */
    typedef Dune::FieldMatrix<RT, embeddedDim, dim> DerivativeType;

    /** \brief Constructor with a given scalar basis
     *
     * \param basis The basis that implements the weights for the Riemannian
     * center of mass
     */
    LocalProjectedFEFunction(const Basis basis)
      : basis_(basis)
      , localBasisView_(basis_)
    {}

    /** \brief Copy constructor
     */
    LocalProjectedFEFunction(const LocalProjectedFEFunction& other)
      : basis_(other.basis_)
      // Do NOT copy the localView: It contains a reference to other.basis_.
      , localBasisView_(basis_.localView())
      , coefficients_(other.coefficients_)
    {
      if (other.localBasisView_.bound())
        localBasisView_.bind(other.localBasisView_.element());
    }

    /** \brief Move constructor
     */
    LocalProjectedFEFunction(LocalProjectedFEFunction&& other)
      : basis_(std::move(other.basis_))
      // Do NOT copy the localView: It contains a reference to other.basis_.
      , localBasisView_(basis_.localView())
      , coefficients_(std::move(other.coefficients_))
    {
      if (other.localBasisView_.bound())
        localBasisView_.bind(other.localBasisView_.element());
    }

    /** \brief Copy assignment
     */
    LocalProjectedFEFunction& operator=(const LocalProjectedFEFunction& other)
    {
      basis_ = other.basis_;
      // Do NOT copy the localView: It contains a reference to other.basis_.
      localBasisView_ = basis_.localView();
      coefficients_ = other.coefficients_;
      if (other.localBasisView_.bound())
        localBasisView_.bind(other.localBasisView_.element());
    }

    /** \brief Move assignment
     */
    LocalProjectedFEFunction& operator=(LocalProjectedFEFunction&& other)
    {
      basis_ = std::move(other.basis_);
      // Do NOT move the localView: It contains a reference to other.basis_.
      localBasisView_ = basis_.localView();
      coefficients_ = std::move(other.coefficients_);
      if (other.localBasisView_.bound())
        localBasisView_.bind(other.localBasisView_.element());
    }

    /** \brief Bind the function to a specific finite element interpolation space and set of coefficients
     *
     * \param element The grid element to bind to
     * \param coefficients Coefficients of the finite element function on that element
     */
    void bind(const Element& element,
              const std::vector<TargetSpace>& coefficients)
    {
      localBasisView_.bind(element);
      coefficients_ = coefficients;

      assert(localBasisView_.size() == coefficients.size());
    }

    /** \brief Rebind the FEFunction to another TargetSpace */
    template<class U>
    struct rebind
    {
      using other = LocalProjectedFEFunction<Basis,U>;
    };

    /** \brief The number of Lagrange points */
    unsigned int size() const
    {
      return localBasisView_.tree().finiteElement().size();
    }

    /** \brief The type of the reference element */
    Dune::GeometryType type() const
    {
      return localBasisView_.element().type();
    }

    /** \brief The finite element basis used to interpolate in the embedding space
     */
    const Basis& globalBasis() const
    {
      return basis_;
    }

    /** \brief The scalar finite element used as the interpolation weights
     *
     * \note This method was added for InterpolationDerivatives, which needs it
     * to construct a copy of a LocalGeodesicFEFunction with ADOL-C's adouble
     * number type.  This is not optimal, because the localFiniteElement
     * really is an implementation detail of LocalGeodesicFEFunction and
     * should not be needed just to copy an entire object.  Other non-Euclidean
     * interpolation rules may not have such a finite element at all.
     * Therefore, this method may disappear again eventually.
     */
    const LocalFiniteElement& localFiniteElement() const
    {
      return localBasisView_.tree().finiteElement();
    }

    /** \brief Evaluate the function */
    TargetSpace evaluate(const LocalCoordinate& local) const
    {
      Rotation<field_type,3> result;

      // Evaluate the weighting factors---these are the Lagrangian shape function values at 'local'
      std::vector<ScalarFERange> w;
      localBasis().evaluateFunction(local,w);

      // Interpolate in R^{3x3}
      FieldMatrix<field_type,3,3> interpolatedMatrix(0);
      for (size_t i=0; i<coefficients_.size(); i++)
      {
        FieldMatrix<field_type,3,3> coefficientAsMatrix;
        coefficients_[i].matrix(coefficientAsMatrix);
        interpolatedMatrix.axpy(w[i][0], coefficientAsMatrix);
      }

      // Project back onto SO(3)
      result.set(Dune::GFE::PolarDecomposition()(interpolatedMatrix));

      return result;
    }

    /** \brief Evaluate the derivative of the function */
    DerivativeType evaluateDerivative(const LocalCoordinate& local) const
    {
      // the function value at the point where we are evaluating the derivative
      TargetSpace q = evaluate(local);

      // Actually compute the derivative
      return evaluateDerivative(local,q);
    }

    /** \brief Evaluate the derivative of the function, if you happen to know the function value (much faster!)
     *        \param local Local coordinates in the reference element where to evaluate the derivative
     *        \param q Value of the local function at 'local'.  If you provide something wrong here the result will be wrong, too!
     */
    DerivativeType evaluateDerivative(const LocalCoordinate& local,
                                      const TargetSpace& q) const
    {
      // Evaluate the weighting factors---these are the Lagrangian shape function values at 'local'
      std::vector<ScalarFERange> w;
      localBasis().evaluateFunction(local,w);

      std::vector<ScalarFEJacobian> wDer;
      localBasis().evaluateJacobian(local,wDer);

      // Compute matrix representations for all coefficients (we only have them in quaternion representation)
      std::vector<Dune::FieldMatrix<field_type,3,3> > coefficientsAsMatrix(coefficients_.size());
      for (size_t i=0; i<coefficients_.size(); i++)
        coefficients_[i].matrix(coefficientsAsMatrix[i]);

      // Interpolate in R^{3x3}
      FieldMatrix<field_type,3,3> interpolatedMatrix(0);
      for (size_t i=0; i<coefficients_.size(); i++)
        interpolatedMatrix.axpy(w[i][0], coefficientsAsMatrix[i]);

      Tensor3<RT,dim,3,3> derivative(0);

      for (size_t dir=0; dir<dim; dir++)
        for (size_t i=0; i<3; i++)
          for (size_t j=0; j<3; j++)
            for (size_t k=0; k<coefficients_.size(); k++)
              derivative[dir][i][j] += wDer[k][0][dir] * coefficientsAsMatrix[k][i][j];

      FieldMatrix<field_type,3,3> polarFactor;
      auto derivativeOfProjection = this->derivativeOfProjection(interpolatedMatrix,polarFactor);

      Tensor3<field_type,dim,3,3> intermediateResult(0);

      for (size_t dir=0; dir<dim; dir++)
        for (size_t i=0; i<3; i++)
          for (size_t j=0; j<3; j++)
            for (size_t k=0; k<3; k++)
              for (size_t l=0; l<3; l++)
                intermediateResult[dir][i][j] += derivativeOfProjection[i][j][k][l]*derivative[dir][k][l];

      // One more application of the chain rule: we need to go from orthogonal matrices to quaternions
      Tensor3<field_type,4,3,3> derivativeOfMatrixToQuaternion = Rotation<field_type,3>::derivativeOfMatrixToQuaternion(polarFactor);

      DerivativeType result(0);

      for (size_t dir0=0; dir0<4; dir0++)
        for (size_t dir1=0; dir1<dim; dir1++)
          for (size_t i=0; i<3; i++)
            for (size_t j=0; j<3; j++)
              result[dir0][dir1] += derivativeOfMatrixToQuaternion[dir0][i][j] * intermediateResult[dir1][i][j];

      return result;
    }

    /** \brief Evaluate the value and the derivative of the interpolation function
     *
     * \return A std::pair containing the value and the first derivative of the interpolation function.
     * If the interpolation is conforming then the first member of the pair will be a TargetSpace.
     * Otherwise it will be a RealTuple.
     */
    auto evaluateValueAndDerivative(const LocalCoordinate& local) const
    {
      // Construct the type of the result -- it depends on whether the interpolation
      // is conforming or not.
      using Value = std::conditional_t<conforming,TargetSpace,RealTuple<RT,embeddedDim> >;

      std::pair<Value,DerivativeType> result;

      // Evaluate the weighting factors---these are the Lagrangian shape function values at 'local'
      std::vector<ScalarFERange> w;
      localBasis().evaluateFunction(local,w);

      // Interpolate in R^{3x3}
      FieldMatrix<field_type,3,3> interpolatedMatrix(0);
      for (size_t i=0; i<coefficients_.size(); i++)
      {
        FieldMatrix<field_type,3,3> coefficientAsMatrix;
        coefficients_[i].matrix(coefficientAsMatrix);
        interpolatedMatrix.axpy(w[i][0], coefficientAsMatrix);
      }

      // Project back onto SO(3)
      static_assert(conforming, "Nonconforming interpolation into SO(3) is not implemented!");
      result.first.set(Dune::GFE::PolarDecomposition()(interpolatedMatrix));

      result.second = evaluateDerivative(local, result.first);

      return result;
    }

    /** \brief Get the i'th base coefficient. */
    const TargetSpace& coefficient(int i) const
    {
      return coefficients_[i];
    }
  private:

    /** \brief The finite element basis for the interpolation in the embedding space
     */
    const Basis basis_;

    /** \brief The scalar local finite element, for interpolating in the embedding space
     */
    typename Basis::LocalView localBasisView_;

    /** \brief The coefficient vector */
    std::vector<TargetSpace> coefficients_;

  };


  /** \brief Interpolate in an embedding Euclidean space, and project back onto the Riemannian manifold -- specialization for R^3 x SO(3)
   *
   * \tparam Basis The basis used for the interpolation in the embedding space.
   * \tparam field_type The number type used for function values
   *
   * \note Even though the embedding space is 12-dimensional, we allow this basis
   * to be scalar-valued.
   */
  template <class Basis, class field_type>
  class LocalProjectedFEFunction<Basis,ProductManifold<RealTuple<field_type,3>,Rotation<field_type,3> > >
  {
  public:
    using TargetSpace = ProductManifold<RealTuple<field_type,3>,Rotation<field_type,3> >;
  private:
    using Element = typename Basis::GridView::template Codim<0>::Entity;
    using LocalCoordinate = typename Element::Geometry::LocalCoordinate;
    static constexpr auto dim = LocalCoordinate::size();

    using LocalFiniteElement = typename Basis::LocalView::Tree::FiniteElement;
    using LocalBasis = typename LocalFiniteElement::Traits::LocalBasisType;
    using ScalarFERange = typename LocalBasis::Traits::RangeType;
    using ScalarFEJacobian = typename LocalBasis::Traits::JacobianType;

    typedef typename TargetSpace::ctype RT;

    typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;
    static const int embeddedDim = EmbeddedTangentVector::dimension;

    /** \brief Short-cut to the local basis
     */
    const auto& localBasis() const
    {
      return localBasisView_.tree().finiteElement().localBasis();
    }

  public:

    /** \brief The type used for derivatives */
    typedef Dune::FieldMatrix<RT, embeddedDim, dim> DerivativeType;

    /** \brief Constructor with a given scalar basis
     *
     * \param basis The basis that implements the weights for the Riemannian
     * center of mass
     */
    LocalProjectedFEFunction(const Basis basis)
      : basis_(basis)
      , localBasisView_(basis_)
      , orientationFunction_(basis)
    {}

    /** \brief Copy constructor
     */
    LocalProjectedFEFunction(const LocalProjectedFEFunction& other)
      : basis_(other.basis_)
      // Do NOT copy the localView: It contains a reference to other.basis_.
      , localBasisView_(basis_.localView())
      , coefficients_(other.coefficients_)
      , translationCoefficients_(other.translationCoefficients_)
      , orientationFunction_(other.orientationFunction_)
    {
      if (other.localBasisView_.bound())
        localBasisView_.bind(other.localBasisView_.element());
    }

    /** \brief Move constructor
     */
    LocalProjectedFEFunction(LocalProjectedFEFunction&& other)
      : basis_(std::move(other.basis_))
      // Do NOT copy the localView: It contains a reference to other.basis_.
      , localBasisView_(basis_.localView())
      , coefficients_(std::move(other.coefficients_))
      , translationCoefficients_(std::move(other.translationCoefficients_))
      , orientationFunction_(std::move(other.orientationFunction_))
    {
      if (other.localBasisView_.bound())
        localBasisView_.bind(other.localBasisView_.element());
    }

    /** \brief Copy assignment
     */
    LocalProjectedFEFunction& operator=(const LocalProjectedFEFunction& other)
    {
      basis_ = other.basis_;
      // Do NOT copy the localView: It contains a reference to other.basis_.
      localBasisView_ = basis_.localView();
      coefficients_ = other.coefficients_;
      if (other.localBasisView_.bound())
        localBasisView_.bind(other.localBasisView_.element());
      translationCoefficients_ = other.translationCoefficients_;
      orientationFunction_ = other.orientationFunction_;
    }

    /** \brief Move assignment
     */
    LocalProjectedFEFunction& operator=(LocalProjectedFEFunction&& other)
    {
      basis_ = std::move(other.basis_);
      // Do NOT move the localView: It contains a reference to other.basis_.
      localBasisView_ = basis_.localView();
      coefficients_ = std::move(other.coefficients_);
      if (other.localBasisView_.bound())
        localBasisView_.bind(other.localBasisView_.element());
      translationCoefficients_ = std::move(other.translationCoefficients_);
      orientationFunction_ = std::move(other.orientationFunction_);
    }

    /** \brief Bind the function to a specific finite element interpolation space and set of coefficients
     *
     * \param element The grid element to bind to
     * \param coefficients Coefficients of the finite element function
     */
    void bind(const Element& element,
              const std::vector<TargetSpace>& coefficients)
    {
      using namespace Dune::Indices;

      localBasisView_.bind(element);
      coefficients_ = coefficients;

      assert(localBasis().size() == coefficients.size());

      translationCoefficients_.resize(coefficients.size());
      for (size_t i=0; i<coefficients.size(); i++)
        translationCoefficients_[i] = coefficients[i][_0].globalCoordinates();

      std::vector<Rotation<field_type,3> > orientationCoefficients(coefficients.size());
      for (size_t i=0; i<coefficients.size(); i++)
        orientationCoefficients[i] = coefficients[i][_1];

      orientationFunction_.bind(element,orientationCoefficients);
    }

    /** \brief Rebind the FEFunction to another TargetSpace */
    template<class U>
    struct rebind
    {
      using other = LocalProjectedFEFunction<Basis,U>;
    };

    /** \brief The number of Lagrange points */
    unsigned int size() const
    {
      return localBasisView_.size();
    }

    /** \brief The type of the reference element */
    Dune::GeometryType type() const
    {
      return localBasisView_.element().type();
    }

    /** \brief The finite element basis used to interpolate in the embedding space
     */
    const Basis& globalBasis() const
    {
      return basis_;
    }

    /** \brief Evaluate the function */
    TargetSpace evaluate(const LocalCoordinate& local) const
    {
      using namespace Dune::Indices;

      TargetSpace result;

      // Evaluate the weighting factors---these are the Lagrangian shape function values at 'local'
      std::vector<ScalarFERange> w;
      localBasis().evaluateFunction(local,w);

      result[_0] = FieldVector<field_type,3>(0.0);
      for (size_t i=0; i<w.size(); i++)
        result[_0].globalCoordinates().axpy(w[i][0], translationCoefficients_[i]);

      result[_1] = orientationFunction_.evaluate(local);

      return result;
    }

    /** \brief Evaluate the derivative of the function */
    DerivativeType evaluateDerivative(const LocalCoordinate& local) const
    {
      // the function value at the point where we are evaluating the derivative
      TargetSpace q = evaluate(local);

      // Actually compute the derivative
      return evaluateDerivative(local,q);
    }

    /** \brief Evaluate the derivative of the function, if you happen to know the function value (much faster!)
     *        \param local Local coordinates in the reference element where to evaluate the derivative
     *        \param q Value of the local function at 'local'.  If you provide something wrong here the result will be wrong, too!
     */
    DerivativeType evaluateDerivative(const LocalCoordinate& local,
                                      const TargetSpace& q) const
    {
      using namespace Dune::Indices;

      DerivativeType result(0);

      // get translation part
      std::vector<ScalarFEJacobian> sfDer(translationCoefficients_.size());
      localBasis().evaluateJacobian(local, sfDer);

      for (size_t i=0; i<translationCoefficients_.size(); i++)
        for (int j=0; j<3; j++)
          result[j].axpy(translationCoefficients_[i][j], sfDer[i][0]);

      // get orientation part
      Dune::FieldMatrix<field_type,4,dim> qResult = orientationFunction_.evaluateDerivative(local,q[_1]);

      for (int i=0; i<4; i++)
        for (std::size_t j=0; j<dim; j++)
          result[3+i][j] = qResult[i][j];

      return result;
    }

    /** \brief Get the i'th base coefficient. */
    const TargetSpace& coefficient(int i) const
    {
      return coefficients_[i];
    }
  private:

    /** \brief The finite element basis for the interpolation in the embedding space
     */
    const Basis basis_;

    /** \brief The local finite element, for the interpolation in the embedding space
     */
    typename Basis::LocalView localBasisView_;

    // The coefficients of this interpolation rule
    std::vector<TargetSpace> coefficients_;

    // The coefficients again.  Yes, we store it twice:  To evaluate the interpolation rule efficiently
    // we need access to the coefficients for the various factor spaces separately.
    std::vector<Dune::FieldVector<field_type,3> > translationCoefficients_;

    LocalProjectedFEFunction<Basis,Rotation<field_type, 3> > orientationFunction_;


  };

}  // namespace Dune::GFE

#endif
