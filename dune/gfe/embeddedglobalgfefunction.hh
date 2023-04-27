// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_GFE_EMBEDDEDGLOBALGFEFUNCTION_HH
#define DUNE_GFE_EMBEDDEDGLOBALGFEFUNCTION_HH

#include <memory>
#include <optional>
#include <vector>

#include <dune/common/typetraits.hh>

#include <dune/grid/utility/hierarchicsearch.hh>

#include <dune/functions/gridfunctions/gridviewentityset.hh>
#include <dune/functions/gridfunctions/gridfunction.hh>
#include <dune/functions/backends/concepts.hh>

#include <dune/gfe/globalgfefunction.hh>

namespace Dune::GFE {

template<typename EGGF>
class EmbeddedGlobalGFEFunctionDerivative;

/**
 * \brief A geometric finite element function with an embedding into Euclidean space
 *
 * The `GlobalGFEFunction` implements a geometric finite element function.
 * The values of that function implement the `TargetSpace` model.
 * In contrast, the values of the `EmbeddedGlobalGFEFunction` implemented here
 * are the corresponding values in Euclidean space.  The precise type is
 * `TargetSpace::CoordinateType`, which is typically a vector or matrix type.
 *
 * \tparam B Type of global scalar(!) basis
 * \tparam LIR Local interpolation rule for manifold-valued data
 * \tparam TargetSpace Range type of this function
 */
template<typename B, typename LIR, typename TargetSpace>
class EmbeddedGlobalGFEFunction
// There is no separate base class for EmbeddedGlobalGFEFunction, because the base class
// only handles coefficients and indices.  It is independent of the type of function values.
  : public Impl::GlobalGFEFunctionBase<B, std::vector<TargetSpace>, LIR>
{
  using Base = Impl::GlobalGFEFunctionBase<B, std::vector<TargetSpace>, LIR>;
  using Data = typename Base::Data;

public:
  using Basis = typename Base::Basis;
  using Vector = typename Base::Vector;
  using LocalInterpolationRule = LIR;

  using Domain = typename Base::Domain;
  using Range = typename TargetSpace::CoordinateType;

  using Traits = Functions::Imp::GridFunctionTraits<Range(Domain), typename Base::EntitySet, Functions::DefaultDerivativeTraits, 16>;

  class LocalFunction
    : public Base::LocalFunctionBase
  {
    using LocalBase = typename Base::LocalFunctionBase;
    using size_type = typename Base::Tree::size_type;

  public:

    using GlobalFunction = EmbeddedGlobalGFEFunction;
    using Domain = typename LocalBase::Domain;
    using Range = GlobalFunction::Range;
    using Element = typename LocalBase::Element;

    //! Create a local-function from the associated grid-function
    LocalFunction(const EmbeddedGlobalGFEFunction& globalFunction)
      : LocalBase(globalFunction.data_)
    {
      /* Nothing. */
    }

    /**
     * \brief Evaluate this local-function in coordinates `x` in the bound element.
     *
     * The result of this method is undefined if you did
     * not call bind() beforehand or changed the coefficient
     * vector after the last call to bind(). In the latter case
     * you have to call bind() again in order to make operator()
     * usable.
     */
    Range operator()(const Domain& x) const
    {
      return this->localInterpolationRule_->evaluate(x).globalCoordinates();
    }

    //! Local function of the derivative
    friend typename EmbeddedGlobalGFEFunctionDerivative<EmbeddedGlobalGFEFunction>::LocalFunction derivative(const LocalFunction& lf)
    {
      auto dlf = localFunction(EmbeddedGlobalGFEFunctionDerivative<EmbeddedGlobalGFEFunction>(lf.data_));
      if (lf.bound())
        dlf.bind(lf.localContext());
      return dlf;
    }
  };

  //! Create a grid-function, by wrapping the arguments in `std::shared_ptr`.
  template<class B_T, class V_T>
  EmbeddedGlobalGFEFunction(B_T && basis, V_T && coefficients)
    : Base(std::make_shared<Data>(Data{{basis.gridView()}, wrap_or_move(std::forward<B_T>(basis)), wrap_or_move(std::forward<V_T>(coefficients))}))
  {}

  //! Create a grid-function, by moving the arguments in `std::shared_ptr`.
  EmbeddedGlobalGFEFunction(std::shared_ptr<const Basis> basis, std::shared_ptr<const Vector> coefficients)
    : Base(std::make_shared<Data>(Data{{basis->gridView()}, basis, coefficients}))
  {}

  /** \brief Evaluate at a point given in world coordinates
   *
   * \warning This has to find the element that the evaluation point is in.
   *   It is therefore very slow.
   */
  Range operator() (const Domain& x) const
  {
    HierarchicSearch search(this->data_->basis->gridView().grid(), this->data_->basis->gridView().indexSet());

    const auto e = search.findEntity(x);
    auto localThis = localFunction(*this);
    localThis.bind(e);
    return localThis(e.geometry().local(x));
  }

  //! Derivative of the `EmbeddedGlobalGFEFunction`
  friend EmbeddedGlobalGFEFunctionDerivative<EmbeddedGlobalGFEFunction> derivative(const EmbeddedGlobalGFEFunction& f)
  {
    return EmbeddedGlobalGFEFunctionDerivative<EmbeddedGlobalGFEFunction>(f.data_);
  }

  /**
   * \brief Construct local function from a EmbeddedGlobalGFEFunction.
   *
   * The obtained local function satisfies the concept
   * `Dune::Functions::Concept::LocalFunction`. It must be bound
   * to an entity from the entity set of the EmbeddedGlobalGFEFunction
   * before it can be used.
   */
  friend LocalFunction localFunction(const EmbeddedGlobalGFEFunction& t)
  {
    return LocalFunction(t);
  }
};


/**
 * \brief Derivative of a `EmbeddedGlobalGFEFunction`
 *
 * Function returning the derivative of the given `EmbeddedGlobalGFEFunction`
 * with respect to global coordinates.
 *
 * \tparam EGGF instance of the `EmbeddedGlobalGFEFunction` this is a derivative of
 */
template<typename EGGF>
class EmbeddedGlobalGFEFunctionDerivative
// There is no separate base class for EmbeddedGlobalGFEFunction, because the base class
// only handles coefficients and indices.  It is independent of the type of function values.
  : public Impl::GlobalGFEFunctionBase<typename EGGF::Basis, typename EGGF::Vector, typename EGGF::LocalInterpolationRule>
{
  using Base = Impl::GlobalGFEFunctionBase<typename EGGF::Basis, typename EGGF::Vector, typename EGGF::LocalInterpolationRule>;
  using Data = typename Base::Data;

public:
  using EmbeddedGlobalGFEFunction = EGGF;

  using Basis = typename Base::Basis;
  using Vector = typename Base::Vector;

  using Domain = typename Base::Domain;
  using Range = typename Functions::SignatureTraits<typename EmbeddedGlobalGFEFunction::Traits::DerivativeInterface>::Range;

  using Traits = Functions::Imp::GridFunctionTraits<Range(Domain), typename Base::EntitySet, Functions::DefaultDerivativeTraits, 16>;

  /**
   * \brief local function evaluating the derivative in reference coordinates
   *
   * Note that the function returns the derivative with respect to global
   * coordinates even when the point is given in reference coordinates on
   * an element.
   */
  class LocalFunction
    : public Base::LocalFunctionBase
  {
    using LocalBase = typename Base::LocalFunctionBase;
    using size_type = typename Base::Tree::size_type;

  public:
    using GlobalFunction = EmbeddedGlobalGFEFunctionDerivative;
    using Domain = typename LocalBase::Domain;
    using Range = GlobalFunction::Range;
    using Element = typename LocalBase::Element;

    //! Create a local function from the associated grid function
    LocalFunction(const GlobalFunction& globalFunction)
      : LocalBase(globalFunction.data_)
    {
      /* Nothing. */
    }

    /**
     * \brief Bind LocalFunction to grid element.
     *
     * You must call this method before `operator()`
     * and after changes to the coefficient vector.
     */
    void bind(const Element& element)
    {
      LocalBase::bind(element);
      geometry_.emplace(element.geometry());
    }

    //! Unbind the local-function.
    void unbind()
    {
      geometry_.reset();
      LocalBase::unbind();
    }

    /**
     * \brief Evaluate this local-function in coordinates `x` in the bound element.
     *
     * The result of this method is undefined if you did
     * not call bind() beforehand or changed the coefficient
     * vector after the last call to bind(). In the latter case
     * you have to call bind() again in order to make operator()
     * usable.
     *
     * Note that the function returns the derivative with respect to global
     * coordinates even though the evaluation point is given in reference coordinates
     * on the current element.
     */
    Range operator()(const Domain& x) const
    {
        // Jacobian with respect to local coordinates
        auto refJac = this->localInterpolationRule_->evaluateDerivative(x);

        // Transform to world coordinates
        return refJac * geometry_->jacobianInverse(x);
    }

    //! Not implemented
    friend typename Traits::LocalFunctionTraits::DerivativeInterface derivative(const LocalFunction&)
    {
      DUNE_THROW(NotImplemented, "derivative of derivative is not implemented");
    }

  private:
    std::optional<typename Element::Geometry> geometry_;
  };

  /**
   * \brief create object from `EmbeddedGlobalGFEFunction` data
   *
   * Please call `derivative(embeddedGlobalGFEFunction)` to create an instance
   * of this class.
   */
  EmbeddedGlobalGFEFunctionDerivative(const std::shared_ptr<const Data>& data)
    : Base(data)
  {
    /* Nothing. */
  }

  /** \brief Evaluate the discrete grid-function derivative in global coordinates
   *
   * \warning This has to find the element that the evaluation point is in.
   *   It is therefore very slow.
   */
  Range operator()(const Domain& x) const
  {
    HierarchicSearch search(this->data_->basis->gridView().grid(), this->data_->basis->gridView().indexSet());

    const auto e = search.findEntity(x);
    auto localThis = localFunction(*this);
    localThis.bind(e);
    return localThis(e.geometry().local(x));
  }

  friend typename Traits::DerivativeInterface derivative(const EmbeddedGlobalGFEFunctionDerivative& f)
  {
    DUNE_THROW(NotImplemented, "derivative of derivative is not implemented");
  }

  //! Construct local function from a `EmbeddedGlobalGFEFunctionDerivative`
  friend LocalFunction localFunction(const EmbeddedGlobalGFEFunctionDerivative& f)
  {
    return LocalFunction(f);
  }
};

} // namespace Dune::GFE

#endif // DUNE_GFE_EMBEDDEDGLOBALGFEFUNCTION_HH
