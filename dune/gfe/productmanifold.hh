#ifndef DUNE_GFE_PRODUCTMANIFOLD_HH
#define DUNE_GFE_PRODUCTMANIFOLD_HH

#include <iostream>
#include <tuple>

#include <dune/common/fvector.hh>
#include <dune/common/math.hh>
#include <dune/common/hybridutilities.hh>
#include <dune/common/tuplevector.hh>

#include <dune/gfe/linearalgebra.hh>
#include <dune/gfe/symmetricmatrix.hh>
#include <dune/gfe/tensor3.hh>

namespace Dune::GFE
{
  template <typename ... TargetSpaces>
  class ProductManifold;

  namespace Impl
  {
    template<typename T, typename ... Ts>
    constexpr auto variadicConvexityRadiusMin()
    {
      if constexpr (sizeof...(Ts)!=0)
        return std::min(T::convexityRadius ,  variadicConvexityRadiusMin<Ts ...>());
      else
        return T::convexityRadius;
    }

    template<class U,typename Tfirst,typename ... TargetSpaces2>
    struct rebindHelper
    {
        typedef ProductManifold<typename Tfirst::template rebind<U>::other ,typename rebindHelper<U,TargetSpaces2...>::other> other;
    };

    template<class U,typename Tlast>
    struct rebindHelper<U,Tlast>
    {
        typedef  typename Tlast::template rebind<U>::other other;
    };
  }

  /** \brief A Product manifold */
  template <typename ... TargetSpaces>
  class ProductManifold
  {
  public:
    template<std::size_t I>
    using IC = Dune::index_constant<I>;
    /** \brief The type used for coordinates. */
    using ctype = typename std::common_type<typename TargetSpaces::ctype...>::type ;
    using field_type =  typename std::common_type<typename TargetSpaces::ctype...>::type ;

    /** \brief Number of factors */
    static constexpr int numTS = sizeof...(TargetSpaces);

    /** \brief Dimension of manifold */
    static constexpr int dim = (0 + ... + TargetSpaces::dim);

    /** \brief Dimension of the embedding space */
    static constexpr int embeddedDim = (0 + ... + TargetSpaces::embeddedDim);

    /** \brief Type of a tangent of the ProductManifold with inner dimensions*/
    typedef Dune::FieldVector<field_type, dim> TangentVector;

    /** \brief Type of a tangent of the ProductManifold represented in the embedding space*/
    typedef Dune::FieldVector<field_type, embeddedDim> EmbeddedTangentVector;

    /** \brief The global convexity radius of the Product space */
    static constexpr double convexityRadius = Impl::variadicConvexityRadiusMin<TargetSpaces ...>();

    /** \brief The type used for global coordinates */
    typedef Dune::FieldVector<field_type ,embeddedDim> CoordinateType;

    /** \brief Default constructor */
    ProductManifold()    = default;

    /** \brief Constructor from another ProductManifold */
    ProductManifold(const ProductManifold& productManifold)
      : data_(productManifold.data_)
    {}

    /** \brief Constructor from a coordinates vector of the embedding space */
    explicit ProductManifold(const CoordinateType& globalCoordinates)
    {
      DUNE_ASSERT_BOUNDS(globalCoordinates.size()== sumEmbeddedDim)
      auto constructorFunctor =[]  (auto& argsTuple, std::array<std::size_t,2>& posHelper, const auto& manifoldInt)
      {
          auto& res = std::get<0>(argsTuple)[manifoldInt];
          const auto& globalCoords =std::get<1>(argsTuple);
          using Manifold = std::remove_const_t<typename std::remove_reference_t<decltype(res)> >;
          res = Manifold(Dune::GFE::segmentAt<Manifold::embeddedDim>(globalCoords,posHelper[0]));
          posHelper[0] += Manifold::embeddedDim;
      };
      foreachManifold(constructorFunctor,*this,globalCoordinates);
    }

    /** \brief Assignment from a coordinates vector of the embedding space */
    ProductManifold& operator=(const CoordinateType& globalCoordinates)
    {
      ProductManifold res(globalCoordinates);
      data_ = res.data_;
      return *this;
    }

    /** \brief Assignment from ProductManifold with different type -- used for automatic differentiation with ADOL-C */
    template <typename ... TargetSpaces2>
    ProductManifold<TargetSpaces ...>& operator <<=(const ProductManifold<TargetSpaces2 ...>& other)
    {
      forEach(integralRange(IC<numTS>()), [&](auto&& i) {
        (*this)[i] <<= other[i];
      });
      return *this;
    }

    /** \brief Rebind the ProductManifold to another coordinate type using an embedded tangent vector*/
    template<class U,typename ... TargetSpaces2>
    struct rebind
    {
      typedef  typename Impl::rebindHelper<U,TargetSpaces...>::other other;
    };

    /** \brief The exponential map from a given point. */
    static ProductManifold<TargetSpaces ...> exp(const ProductManifold& p, const EmbeddedTangentVector& v)
    {
      ProductManifold res;
      auto expFunctor =[]  (auto& argsTuple, std::array<std::size_t,2>& posHelper, const auto& manifoldInt)
      {
          auto& res        = std::get<0>(argsTuple)[manifoldInt];
          const auto& p    = std::get<1>(argsTuple)[manifoldInt];
          const auto& tang = std::get<2>(argsTuple);
          using Manifold = std::remove_const_t<typename std::remove_reference_t<decltype(res)> >;
          auto currentEmbeddedTangentVector = Dune::GFE::segmentAt<Manifold::embeddedDim>(tang,posHelper[0]);
          res =  Manifold::exp(p,currentEmbeddedTangentVector);
          posHelper[0] += Manifold::embeddedDim;
      };
      foreachManifold(expFunctor,res,p,v);
      return res;
    }

    /** \brief The exponential map from a given point using an intrinsic tangent vector*/
    static ProductManifold exp(const ProductManifold& p, const TangentVector& v)
    {
      auto basis = p.orthonormalFrame();
      EmbeddedTangentVector embeddedTangent;
      basis.mtv(v, embeddedTangent);
      return exp(p,embeddedTangent);
    }

    /** \brief Compute difference vector from a to b on the tangent space of a */
    static EmbeddedTangentVector log(const ProductManifold& a, const ProductManifold& b)
    {
      EmbeddedTangentVector diff;
      auto logFunctor =[] (auto& argsTuple, std::array<std::size_t,2>& posHelper, const auto& manifoldInt)
      {
          auto& res     = std::get<0>(argsTuple);
          const auto& a = std::get<1>(argsTuple)[manifoldInt];
          const auto& b = std::get<2>(argsTuple)[manifoldInt];
          using Manifold = std::remove_const_t<typename std::remove_reference_t<decltype(a)> >;
          const auto diffLoc =  Manifold::log(a,b);
          std::copy(diffLoc.begin(),diffLoc.end(),res.begin()+posHelper[0]);
          posHelper[0] += Manifold::embeddedDim;
      };
      foreachManifold(logFunctor,diff,a, b);
      return diff;
    }

    /** \brief Compute geodesic distance from a to b */
    static field_type distance(const ProductManifold& a, const ProductManifold& b)
    {
      field_type dist=0.0;
      auto distanceFunctor =[] (auto& argsTuple, std::array<std::size_t,2>& posHelper, const auto& manifoldInt)
      {
          auto& res     = std::get<0>(argsTuple);
          const auto& a = std::get<1>(argsTuple)[manifoldInt];
          const auto& b = std::get<2>(argsTuple)[manifoldInt];
          using Manifold = std::remove_const_t<typename std::remove_reference_t<decltype(a)> >;
          res += power(Manifold::distance(a,b),2);
      };
      foreachManifold(distanceFunctor,dist,a, b);
      return sqrt(dist);
    }

    /** \brief Compute the gradient of the squared distance function keeping the first argument fixed */
    static EmbeddedTangentVector derivativeOfDistanceSquaredWRTSecondArgument(const ProductManifold& a,
                                                                              const ProductManifold& b)
    {
      EmbeddedTangentVector derivative;
      derivative= 0.0;

      auto derivOfDistSqdWRTSecArgFunctor = [] (auto& argsTuple, std::array<std::size_t,2>& posHelper, const auto& manifoldInt)
      {
          auto& derivative = std::get<0>(argsTuple);
          const auto& a    = std::get<1>(argsTuple)[manifoldInt];
          const auto& b    = std::get<2>(argsTuple)[manifoldInt];
          using Manifold = std::remove_const_t<typename std::remove_reference_t<decltype(a)> >;
          const auto diffLoc = Manifold::derivativeOfDistanceSquaredWRTSecondArgument(a,b);
          std::copy(diffLoc.begin(),diffLoc.end(),derivative.begin()+posHelper[0]);
          posHelper[0] += Manifold::embeddedDim;
      };
      foreachManifold(derivOfDistSqdWRTSecArgFunctor, derivative,a, b);
      return derivative;
    }

    /** \brief Compute the Hessian of the squared distance function keeping the first argument fixed */
    static auto secondDerivativeOfDistanceSquaredWRTSecondArgument(const ProductManifold& a,
                                                                   const ProductManifold& b)
    {
      Dune::SymmetricMatrix<field_type,embeddedDim> result;
      auto secDerivOfDistSqWRTSecArgFunctor = [] (auto& argsTuple, std::array<std::size_t,2>& posHelper, const auto& manifoldInt)
      {
          auto& deriv   = std::get<0>(argsTuple);
          const auto& a = std::get<1>(argsTuple)[manifoldInt];
          const auto& b = std::get<2>(argsTuple)[manifoldInt];
          using Manifold = std::remove_const_t<typename std::remove_reference_t<decltype(a)> >;
          const auto diffLoc = Manifold::secondDerivativeOfDistanceSquaredWRTSecondArgument(a,b);
          for(size_t i=posHelper[0]; i<posHelper[0]+Manifold::embeddedDim; ++i )
            for(size_t j=posHelper[0]; j<=i; ++j )
                deriv(i,j)  = diffLoc(i - posHelper[0], j - posHelper[0]);
          posHelper[0] += Manifold::embeddedDim;
      };
      foreachManifold(secDerivOfDistSqWRTSecArgFunctor,result,a, b);
      return result;
    }

    /** \brief Compute the mixed second derivate \partial d^2 / \partial da db     */
    static Dune::FieldMatrix<field_type ,embeddedDim,embeddedDim>
    secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(const ProductManifold& a,
                                                               const ProductManifold& b)
    {
      Dune::FieldMatrix<field_type,embeddedDim,embeddedDim> result(0);
      auto secDerivOfDistSqWRTFirstAndSecArgFunctor = [] (auto& argsTuple, std::array<std::size_t,2>& posHelper, const auto& manifoldInt)
      {
          auto&   deriv = std::get<0>(argsTuple);
          const auto& a = std::get<1>(argsTuple)[manifoldInt];
          const auto& b = std::get<2>(argsTuple)[manifoldInt];
          using Manifold = std::remove_const_t<typename std::remove_reference_t<decltype(a)> >;
          const auto diffLoc = Manifold::secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(a,b);
          for(size_t i=posHelper[0]; i<posHelper[0]+Manifold::embeddedDim; ++i )
            for(size_t j=posHelper[0]; j<posHelper[0]+Manifold::embeddedDim; ++j )
                deriv[i][j] = diffLoc[i - posHelper[0]][ j - posHelper[0]];
          posHelper[0] += Manifold::embeddedDim;
      };
      foreachManifold(secDerivOfDistSqWRTFirstAndSecArgFunctor,result,a, b);
      return result;
    }

    /** \brief Compute the third derivative \partial d^3 / \partial dq^3  */
    static Tensor3<field_type ,embeddedDim,embeddedDim,embeddedDim>
    thirdDerivativeOfDistanceSquaredWRTSecondArgument(const ProductManifold& a,
                                                      const ProductManifold& b)
    {
      Tensor3<field_type,embeddedDim,embeddedDim,embeddedDim> result(0);
      auto thirdDerivOfDistSqWRTSecArgFunctor =[](auto& argsTuple, std::array<std::size_t,2>& posHelper, const auto& manifoldInt)
      {
          auto& deriv   = std::get<0>(argsTuple);
          const auto& a = std::get<1>(argsTuple)[manifoldInt];
          const auto& b = std::get<2>(argsTuple)[manifoldInt];
          using Manifold = std::remove_const_t<typename std::remove_reference_t<decltype(a)>>;
          const auto diffLoc = Manifold::thirdDerivativeOfDistanceSquaredWRTSecondArgument(a,b);
          for(size_t i=posHelper[0]; i<posHelper[0]+Manifold::embeddedDim; ++i )
              for(size_t j=posHelper[0]; j<posHelper[0]+Manifold::embeddedDim; ++j )
                  for(size_t k=posHelper[0]; k<posHelper[0]+Manifold::embeddedDim; ++k )
                      deriv[i][j][k] = diffLoc[i - posHelper[0]][ j - posHelper[0]][k - posHelper[0]];
          posHelper[0] += Manifold::embeddedDim;
      };
      foreachManifold(thirdDerivOfDistSqWRTSecArgFunctor,result,a, b);
      return result;
    }

    /** \brief Compute the mixed third derivative \partial d^3 / \partial da db^2  */
    static Tensor3<field_type ,embeddedDim,embeddedDim,embeddedDim>
    thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(const ProductManifold& a,
                                                                const ProductManifold& b)
    {
      Tensor3<field_type,embeddedDim,embeddedDim,embeddedDim> result(0);
      auto thirdDerivOfDistSqWRT1And2ArgFunctor =[] (auto& argsTuple, std::array<std::size_t,2>& posHelper, const auto& manifoldInt)
      {
          auto& deriv   = std::get<0>(argsTuple);
          const auto& a = std::get<1>(argsTuple)[manifoldInt];
          const auto& b = std::get<2>(argsTuple)[manifoldInt];
          using Manifold = std::remove_const_t<typename std::remove_reference_t<decltype(a)>>;
          const auto diffLoc =  Manifold::thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(a,b);
          for(size_t i=posHelper[0]; i<posHelper[0]+Manifold::embeddedDim; ++i )
              for(size_t j=posHelper[0]; j<posHelper[0]+Manifold::embeddedDim; ++j )
                  for(size_t k=posHelper[0]; k<posHelper[0]+Manifold::embeddedDim; ++k )
                      deriv[i][j][k] = diffLoc[i - posHelper[0]][ j - posHelper[0]][k - posHelper[0]];
          posHelper[0] += Manifold::embeddedDim;
      };
      foreachManifold(thirdDerivOfDistSqWRT1And2ArgFunctor,result,a, b);
      return result;
    }

    /** \brief Project tangent vector of R^n onto the tangent space */
    EmbeddedTangentVector projectOntoTangentSpace(const EmbeddedTangentVector& v) const
    {
      EmbeddedTangentVector result {v};
      auto projectOntoTangentSpaceFunctor =[] (auto& argsTuple, std::array<std::size_t,2>& posHelper, const auto& manifoldInt)
      {
          auto& v       = std::get<0>(argsTuple);
          const auto& a = std::get<1>(argsTuple)[manifoldInt];
          using Manifold = std::remove_const_t<typename std::remove_reference_t<decltype(a)> >;
          const auto vLoc = a.projectOntoTangentSpace(Dune::GFE::segmentAt<Manifold::embeddedDim>(v,posHelper[0]));
          std::copy(vLoc.begin(),vLoc.end(),v.begin()+posHelper[0]);
          posHelper[0] += Manifold::embeddedDim;
      };
      foreachManifold(projectOntoTangentSpaceFunctor,result,*this);
      return result;
    }

    /** \brief Project tangent vector of R^n onto the normal space space */
    EmbeddedTangentVector projectOntoNormalSpace(const EmbeddedTangentVector& v) const
    {
      EmbeddedTangentVector result {v};
      auto projectOntoNormalSpaceFunctor =[] (auto& argsTuple, std::array<std::size_t,2>& posHelper, const auto& manifoldInt)
      {
          auto& v       = std::get<0>(argsTuple);
          const auto& a = std::get<1>(argsTuple)[manifoldInt];
          using Manifold = std::remove_const_t<typename std::remove_reference_t<decltype(a)> >;
          const auto vLoc = a.projectOntoNormalSpace(Dune::GFE::segmentAt<Manifold::embeddedDim>(v,posHelper[0]));
          std::copy(vLoc.begin(),vLoc.end(),v.begin()+posHelper[0]);
          posHelper[0] += Manifold::embeddedDim;
      };
      foreachManifold(projectOntoNormalSpaceFunctor,result,*this);
      return result;
    }

    /** \brief Project tangent vector of R^n onto the normal space space */
    static ProductManifold projectOnto(const CoordinateType& v)
    {
      ProductManifold<TargetSpaces ...> result {v};
      auto projectOntoFunctor =[] (auto& argsTuple, std::array<std::size_t,2>& posHelper, const auto& manifoldInt)
      {
          auto& res = std::get<0>(argsTuple)[manifoldInt];
          auto& v   = std::get<1>(argsTuple);
          using Manifold = std::remove_const_t<typename std::remove_reference_t<decltype(res)> >;
          res =  Manifold::projectOnto(Dune::GFE::segmentAt<Manifold::embeddedDim>(v,posHelper[0]));
          posHelper[0] += Manifold::embeddedDim;
      };
      foreachManifold(projectOntoFunctor,result, v);
      return result;
    }

    /** \brief Project tangent vector of R^n onto the normal space space */
    static auto derivativeOfProjection(const CoordinateType& v)
    {
      Dune::FieldMatrix<typename CoordinateType::value_type, embeddedDim, embeddedDim> result;
      auto derivativeOfProjectionFunctor =[] (auto& argsTuple, std::array<std::size_t,2>& posHelper, const auto& manifoldInt)
      {
          auto& res     = std::get<0>(argsTuple);
          const auto& v = std::get<1>(argsTuple);
          const Dune::TupleVector<TargetSpaces...> ManifoldTuple;
          using Manifold = std::remove_const_t<typename std::remove_reference_t<decltype(ManifoldTuple[manifoldInt])> >;
          const auto vLoc =  Manifold::derivativeOfProjection(Dune::GFE::segmentAt<Manifold::embeddedDim>(v,posHelper[0]));
          for(size_t k=posHelper[0]; k<posHelper[0]+Manifold::embeddedDim; ++k )
              for(size_t j=posHelper[0]; j<posHelper[0]+Manifold::embeddedDim; ++j )
                  res[k][j] = vLoc[k - posHelper[0]][j-posHelper[0]];
          posHelper[0] += Manifold::embeddedDim;
      };
      foreachManifold(derivativeOfProjectionFunctor,result, v);
      return result;
    }

    /** \brief The Weingarten map */
    EmbeddedTangentVector weingarten(const EmbeddedTangentVector& z, const EmbeddedTangentVector& v) const
    {
      EmbeddedTangentVector result;
      auto weingartenFunctor =[] (auto& argsTuple, std::array<std::size_t,2>& posHelper, const auto& manifoldInt)
      {
          auto& res     = std::get<0>(argsTuple);
          const auto& a = std::get<1>(argsTuple)[manifoldInt];
          const auto& z = std::get<2>(argsTuple);
          const auto& v = std::get<3>(argsTuple);
          using Manifold = std::remove_const_t<typename std::remove_reference_t<decltype(a)> >;
          const auto resLoc =  a.weingarten(Dune::GFE::segmentAt<Manifold::embeddedDim>(z,posHelper[0]),
                                            Dune::GFE::segmentAt<Manifold::embeddedDim>(v,posHelper[0]));
          std::copy(resLoc.begin(),resLoc.end(),res.begin()+posHelper[0]);
          posHelper[0] +=Manifold::embeddedDim;
      };
      foreachManifold(weingartenFunctor,result, *this,z, v);
      return result;
    }

    /** \brief Compute an orthonormal basis of the tangent space of the ProductManifold
       This basis may not be globally continuous.     */
    Dune::FieldMatrix<field_type ,dim,embeddedDim> orthonormalFrame() const
    {
      Dune::FieldMatrix<field_type,dim,embeddedDim> result(0);
      auto orthonormalFrameFunctor =[] (auto& argsTuple, std::array<std::size_t,2>& posHelper, const auto& manifoldInt)
      {
          auto& res     = std::get<0>(argsTuple);
          const auto& a = std::get<1>(argsTuple)[manifoldInt];
          using Manifold = std::remove_const_t<typename std::remove_reference_t<decltype(a)> >;
          const auto resLoc =  a.orthonormalFrame();
          for(size_t i=posHelper[0]; i<posHelper[0]+Manifold::dim; ++i )
              for(size_t j=posHelper[1]; j<posHelper[1]+Manifold::embeddedDim; ++j )
                  res[i][j] = resLoc[i - posHelper[0]][j-posHelper[1]];
          posHelper[0] += Manifold::dim;
          posHelper[1] += Manifold::embeddedDim;
      };
      foreachManifold(orthonormalFrameFunctor,result,*this);
      return result;
    }

    /** \brief The global coordinates */
    CoordinateType globalCoordinates() const
    {
      CoordinateType returnValue;
      auto globalCoordinatesFunctor =[] (auto& argsTuple, std::array<std::size_t,2>& posHelper, const auto& i)
      {
          auto& res       = std::get<0>(argsTuple);
          const auto& p   = std::get<1>(argsTuple)[i];
          const auto vLoc =  p.globalCoordinates();
          using Manifold = std::remove_const_t<typename std::remove_reference_t<decltype(p)> >;
          std::copy(vLoc.begin(),vLoc.end(),res.begin()+posHelper[0]);
          posHelper[0] += Manifold::embeddedDim;
      };
      foreachManifold(globalCoordinatesFunctor,returnValue,*this);
      return returnValue;
    }

    /** \brief Const access to the tuple elements  */
    template<std::size_t i>
    constexpr decltype(auto) operator[] (const Dune::index_constant<i>&) const
    {
      return std::get<i>(data_);
    }

    /** \brief Non-const access to the tuple elements     */
    template<std::size_t i>
    decltype(auto) operator[] (const Dune::index_constant<i>&)
    {
      return std::get<i>(data_);
    }

    /** \brief Number of elements of the tuple */
    static constexpr std::size_t size()
    {
      return numTS;
    }

    template<class... TargetSpaces2>
    friend std::ostream& operator<<(std::ostream& s, const ProductManifold<TargetSpaces2 ...>& c);

  private:
    /**
     * \brief Range based for loop over all Manifolds in ProductManifold
     *
     * \tparam FunctorType the functor which should be applied for all manifolds
     * \tparam Args... The arguments which are passed to the functor
     *
     * The functor has to extract the current manifold from its arguments.
     * Therefore, the current index i is passed to the functor.
     *
     * Furthermore, the functor gets two integers to deal with FieldVectors
     * or FieldMatrices which span the whole (embedding) coordinate range of the ProductManifold.
     *
     * Example:
     * If a argument of the functor is
     * ProductManifold<Realtuple<double,3>,UnitVector<double,3>>::EmbeddedTangentVector,
     * the embedded coordinate vector has 6 entries. The functor needs to know where
     * to insert the next subvector of the submanifold. Therefore, posHelper[0] should be 0 at the first
     * iteration and 3 at the second one. Then first indices [0..2] stores the result of
     * Realtuple<double,3> and [3..5] stores
     * the result of UnitVector<double,3>.
     * See e.g. the globalCoordinates() function
     */
    template<typename FunctorType, typename ... Args>
    static void foreachManifold( FunctorType&& functor,Args&&... args)
    {
      auto argsTuple = std::forward_as_tuple(args ...);
      std::array<std::size_t,2> posHelper({0,0});
      Dune::Hybrid::forEach(Dune::Hybrid::integralRange(IC<numTS>()),[&](auto&& i) {
          functor(argsTuple,posHelper,i);
      });
    }

    std::tuple<TargetSpaces ...> data_;
};

  template<typename ... TargetSpaces>
  std::ostream& operator<<(std::ostream& s, const ProductManifold<TargetSpaces ...>& c)
  {
    Dune::Hybrid::forEach(Dune::Hybrid::integralRange(Dune::index_constant<ProductManifold<TargetSpaces ...>::numTS>()), [&](auto&& i) {
        s<<Dune::className<decltype(c[i])>()<<" "<< c[i]<<"\n";
    });
    return s;
  }
}
#endif
