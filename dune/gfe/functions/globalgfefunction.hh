// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_GFE_FUNCTIONS_GLOBALGFEFUNCTION_HH
#define DUNE_GFE_FUNCTIONS_GLOBALGFEFUNCTION_HH

#include <memory>
#include <optional>
#include <vector>

#include <dune/common/typetraits.hh>
#include <dune/common/version.hh>

#include <dune/grid/utility/hierarchicsearch.hh>

#include <dune/functions/gridfunctions/gridviewentityset.hh>
#include <dune/functions/gridfunctions/gridfunction.hh>
#include <dune/functions/backends/concepts.hh>

#if DUNE_VERSION_LTE(DUNE_FUFEM, 2, 9)
#include <dune/fufem/functions/virtualgridfunction.hh>
#endif

namespace Dune::GFE
{
  namespace Impl {

#if DUNE_VERSION_LTE(DUNE_FUFEM, 2, 9)
    // This collects all data that is shared by all related
    // global and local functions.
    template <typename Basis, typename Vector>
    struct Data
    {
      using GridView = typename Basis::GridView;
      using EntitySet = Functions::GridViewEntitySet<GridView, 0>;
      EntitySet entitySet;
      std::shared_ptr<const Basis> basis;
      std::shared_ptr<const Vector> coefficients;
    };
#endif

    /** \brief Common base class for GlobalGFEFunction and its derivative
     *
     * \tparam B Scalar(!) function-space basis
     * \tparam V Container of coefficients
     * \tparam LocalInterpolationRule How to interpolate manifold-valued data
     */
#if DUNE_VERSION_LTE(DUNE_FUFEM, 2, 9)
    template<typename B, typename V, typename LocalInterpolationRule, typename Range>
    class GlobalGFEFunctionBase
      : public VirtualGridViewFunction<typename B::GridView, Range>
#else
    template<typename B, typename V, typename LocalInterpolationRule>
    class GlobalGFEFunctionBase
#endif
    {
    public:
      using Basis = B;
      using Vector = V;

      // In order to make the cache work for proxy-references
      // we have to use AutonomousValue<T> instead of std::decay_t<T>
      using Coefficient = Dune::AutonomousValue<decltype(std::declval<Vector>()[std::declval<typename Basis::MultiIndex>()])>;

      using GridView = typename Basis::GridView;
      using EntitySet = Functions::GridViewEntitySet<GridView, 0>;
      using Tree = typename Basis::LocalView::Tree;

      using Domain = typename EntitySet::GlobalCoordinate;

      using LocalDomain = typename EntitySet::LocalCoordinate;
      using Element = typename EntitySet::Element;

    protected:

#if DUNE_VERSION_GT(DUNE_FUFEM, 2, 9)
      // This collects all data that is shared by all related
      // global and local functions. This way we don't need to
      // keep track of it individually.
      struct Data
      {
        EntitySet entitySet;
        std::shared_ptr<const Basis> basis;
        std::shared_ptr<const Vector> coefficients;
      };
#endif

    public:
      class LocalFunctionBase
      {
        using LocalView = typename Basis::LocalView;
        using size_type = typename Tree::size_type;

      public:
        using Domain = LocalDomain;
        using Element = typename EntitySet::Element;

      protected:
#if DUNE_VERSION_LTE(DUNE_FUFEM, 2, 9)
        LocalFunctionBase(const std::shared_ptr<const Data<Basis,Vector> >& data)
#else
        LocalFunctionBase(const std::shared_ptr<const Data>& data)
#endif
          : data_(data)
          , localView_(data_->basis->localView())
        {
          localDoFs_.reserve(localView_.maxSize());
        }

        /**
         * \brief Copy-construct the local-function.
         *
         * This copy-constructor copies the cached local DOFs only
         * if the `other` local-function is bound to an element.
         **/
        LocalFunctionBase(const LocalFunctionBase& other)
          : data_(other.data_)
          , localView_(other.localView_)
        {
          localDoFs_.reserve(localView_.maxSize());
          if (bound())
            localDoFs_ = other.localDoFs_;
        }

        /**
         * \brief Copy-assignment of the local-function.
         *
         * Assign all members from `other` to `this`, except the
         * local DOFs. Those are copied only if the `other`
         * local-function is bound to an element.
         **/
        LocalFunctionBase& operator=(const LocalFunctionBase& other)
        {
          data_ = other.data_;
          localView_ = other.localView_;
          if (bound())
            localDoFs_ = other.localDoFs_;
          return *this;
        }

      public:
        /**
         * \brief Bind LocalFunction to grid element.
         *
         * You must call this method before `operator()`
         * and after changes to the coefficient vector.
         */
        void bind(const Element& element)
        {
          localView_.bind(element);

          localDoFs_.resize(localView_.size());
          const auto& dofs = *data_->coefficients;
          for (size_type i = 0; i < localView_.tree().size(); ++i)
          {
            // For a subspace basis the index-within-tree i
            // is not the same as the localIndex within the
            // full local view.
            size_t localIndex = localView_.tree().localIndex(i);
            localDoFs_[localIndex] = dofs[localView_.index(localIndex)];
          }

          // create local GFE function
          // TODO Store this object by value
          localInterpolationRule_ = std::make_unique<LocalInterpolationRule>(this->localView_.tree().finiteElement(),localDoFs_);
        }

        //! Unbind the local-function.
        void unbind()
        {
          localView_.unbind();
        }

        //! Check if LocalFunction is already bound to an element.
        bool bound() const
        {
          return localView_.bound();
        }

        //! Return the element the local-function is bound to.
        const Element& localContext() const
        {
          return localView_.element();
        }

      protected:

#if DUNE_VERSION_LTE(DUNE_FUFEM, 2, 9)
        std::shared_ptr<const Data<Basis,Vector> > data_;
#else
        std::shared_ptr<const Data> data_;
#endif
        LocalView localView_;
        std::vector<Coefficient> localDoFs_;
        std::unique_ptr<LocalInterpolationRule> localInterpolationRule_;
      };

    protected:
#if DUNE_VERSION_LTE(DUNE_FUFEM, 2, 9)
      GlobalGFEFunctionBase(const std::shared_ptr<const Data<Basis,Vector> >& data)
        : VirtualGridViewFunction<typename B::GridView, Range>(data->basis->gridView())
        , data_(data)
#else
      GlobalGFEFunctionBase(const std::shared_ptr<const Data>& data)
        : data_(data)
#endif
      {
        /* Nothing. */
      }

    public:

      //! Return a const reference to the stored basis.
      const Basis& basis() const
      {
        return *data_->basis;
      }

      //! Return the coefficients of this discrete function by reference.
      const Vector& dofs() const
      {
        return *data_->coefficients;
      }

      //! Get associated set of entities the local-function can be bound to.
      const EntitySet& entitySet() const
      {
        return data_->entitySet;
      }

    protected:
#if DUNE_VERSION_LTE(DUNE_FUFEM, 2, 9)
      std::shared_ptr<const Data<Basis, Vector> > data_;
#else
      std::shared_ptr<const Data> data_;
#endif
    };

  } // namespace Impl



  template<typename GGF>
  class GlobalGFEFunctionDerivative;

  /**
   * \brief A global geometric finite element function
   *
   * \tparam B Type of global basis
   * \tparam LIR Local interpolation rule for manifold-valued data
   * \tparam TargetSpace Range type of this function
   */
  template<typename B, typename LIR, typename TargetSpace>
  class GlobalGFEFunction
#if DUNE_VERSION_LTE(DUNE_FUFEM, 2, 9)
    : public Impl::GlobalGFEFunctionBase<B, std::vector<TargetSpace>, LIR, typename TargetSpace::CoordinateType>
#else
    : public Impl::GlobalGFEFunctionBase<B, std::vector<TargetSpace>, LIR>
#endif
  {
#if DUNE_VERSION_LTE(DUNE_FUFEM, 2, 9)
    using Base = Impl::GlobalGFEFunctionBase<B, std::vector<TargetSpace>, LIR, typename TargetSpace::CoordinateType>;
#else
    using Base = Impl::GlobalGFEFunctionBase<B, std::vector<TargetSpace>, LIR>;
    using Data = typename Base::Data;
#endif

  public:
    using Basis = typename Base::Basis;
    using Vector = typename Base::Vector;
#if DUNE_VERSION_LTE(DUNE_FUFEM, 2, 9)
    using Data = typename Impl::Data<Basis,Vector>;
#endif
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

      using GlobalFunction = GlobalGFEFunction;
      using Domain = typename LocalBase::Domain;
      using Range = GlobalFunction::Range;
      using Element = typename LocalBase::Element;

      //! Create a local-function from the associated grid-function
      LocalFunction(const GlobalGFEFunction& globalFunction)
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
      friend typename GlobalGFEFunctionDerivative<GlobalGFEFunction>::LocalFunction derivative(const LocalFunction& lf)
      {
        auto dlf = localFunction(GlobalGFEFunctionDerivative<GlobalGFEFunction>(lf.data_));
        if (lf.bound())
          dlf.bind(lf.localContext());
        return dlf;
      }
    };

    //! Create a grid-function, by wrapping the arguments in `std::shared_ptr`.
    template<class B_T, class V_T>
    GlobalGFEFunction(B_T && basis, V_T && coefficients)
      : Base(std::make_shared<Data>(Data{{basis.gridView()}, wrap_or_move(std::forward<B_T>(basis)), wrap_or_move(std::forward<V_T>(coefficients))}))
    {}

    //! Create a grid-function, by moving the arguments in `std::shared_ptr`.
    GlobalGFEFunction(std::shared_ptr<const Basis> basis, std::shared_ptr<const Vector> coefficients)
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

    //! Derivative of the `GlobalGFEFunction`
    friend GlobalGFEFunctionDerivative<GlobalGFEFunction> derivative(const GlobalGFEFunction& f)
    {
      return GlobalGFEFunctionDerivative<GlobalGFEFunction>(f.data_);
    }

    /**
     * \brief Construct local function from a GlobalGFEFunction.
     *
     * The obtained local function satisfies the concept
     * `Dune::Functions::Concept::LocalFunction`. It must be bound
     * to an entity from the entity set of the GlobalGFEFunction
     * before it can be used.
     */
    friend LocalFunction localFunction(const GlobalGFEFunction& t)
    {
      return LocalFunction(t);
    }

#if DUNE_VERSION_LTE(DUNE_FUFEM, 2, 9)
    using Element = typename Basis::GridView::template Codim<0>::Entity;
    /** \brief Evaluate the function at local coordinates. */
    void evaluateLocal(const Element& element, const Domain& local, typename TargetSpace::CoordinateType& out) const
    {
      DUNE_THROW(NotImplemented, "!");
    }
#endif
  };


  /**
   * \brief Derivative of a `GlobalGFEFunction`
   *
   * Function returning the derivative of the given `GlobalGFEFunction`
   * with respect to global coordinates.
   *
   * \tparam GGF instance of the `GlobalGFEFunction` this is a derivative of
   */
  template<typename GGF>
  class GlobalGFEFunctionDerivative
#if DUNE_VERSION_LTE(DUNE_FUFEM, 2, 9)
    : public Impl::GlobalGFEFunctionBase<typename GGF::Basis, typename GGF::Vector, typename GGF::LocalInterpolationRule,
          Dune::FieldMatrix<double, GGF::Vector::value_type::EmbeddedTangentVector::dimension, GGF::Basis::GridView::dimensionworld> >
#else
    : public Impl::GlobalGFEFunctionBase<typename GGF::Basis, typename GGF::Vector, typename GGF::LocalInterpolationRule>
#endif
  {
#if DUNE_VERSION_LTE(DUNE_FUFEM, 2, 9)
    using Base = Impl::GlobalGFEFunctionBase<typename GGF::Basis, typename GGF::Vector, typename GGF::LocalInterpolationRule,
        Dune::FieldMatrix<double, GGF::Vector::value_type::EmbeddedTangentVector::dimension, GGF::Basis::GridView::dimensionworld> >;
#else
    using Base = Impl::GlobalGFEFunctionBase<typename GGF::Basis, typename GGF::Vector, typename GGF::LocalInterpolationRule>;
    using Data = typename Base::Data;
#endif

  public:
    using GlobalGFEFunction = GGF;

    using Basis = typename Base::Basis;
    using Vector = typename Base::Vector;
#if DUNE_VERSION_LTE(DUNE_FUFEM, 2, 9)
    using Data = typename Impl::Data<Basis,Vector>;
#endif

    using Domain = typename Base::Domain;
    using Range = typename Functions::SignatureTraits<typename GlobalGFEFunction::Traits::DerivativeInterface>::Range;

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
      using GlobalFunction = GlobalGFEFunctionDerivative;
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
     * \brief create object from `GlobalGFEFunction` data
     *
     * Please call `derivative(globalGFEFunction)` to create an instance
     * of this class.
     */
    GlobalGFEFunctionDerivative(const std::shared_ptr<const Data>& data)
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

    friend typename Traits::DerivativeInterface derivative(const GlobalGFEFunctionDerivative& f)
    {
      DUNE_THROW(NotImplemented, "derivative of derivative is not implemented");
    }

    //! Construct local function from a `GlobalGFEFunctionDerivative`
    friend LocalFunction localFunction(const GlobalGFEFunctionDerivative& f)
    {
      return LocalFunction(f);
    }

#if DUNE_VERSION_LTE(DUNE_FUFEM, 2, 9)
    using Element = typename Basis::GridView::template Codim<0>::Entity;
    /** \brief Evaluate the function at local coordinates. */
    void evaluateLocal(const Element& element, const Domain& local, Range& out) const override
    {
      // This method will never be called.
    }
#endif

  };

} // namespace Dune::GFE

#endif // DUNE_GFE_FUNCTIONS_GLOBALGFEFUNCTION_HH
