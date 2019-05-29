// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_GFE_PERIODIC_1D_PQ1NODALBASIS_HH
#define DUNE_GFE_PERIODIC_1D_PQ1NODALBASIS_HH

#include <dune/common/exceptions.hh>

#include <dune/localfunctions/lagrange/pqkfactory.hh>

#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
#include <dune/typetree/leafnode.hh>
#endif

#include <dune/functions/functionspacebases/nodes.hh>
#include <dune/functions/functionspacebases/flatmultiindex.hh>
#include <dune/functions/functionspacebases/defaultglobalbasis.hh>


namespace Dune {
namespace Functions {

// *****************************************************************************
// This is the reusable part of the basis. It contains
//
//   PQ1PreBasis
//   PQ1NodeIndexSet
//   PQ1Node
//
// The pre-basis allows to create the others and is the owner of possible shared
// state. These three components do _not_ depend on the global basis or index
// set and can be used without a global basis.
// *****************************************************************************

#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
template<typename GV, typename ST, typename TP>
#else
template<typename GV>
#endif
class Periodic1DPQ1Node;

#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
template<typename GV, class MI, class TP, class ST>
#else
template<typename GV, class MI>
#endif
class Periodic1DPQ1NodeIndexSet;

#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
template<typename GV, class MI, class ST>
class Periodic1DPQ1NodeFactory
#else
template<typename GV, class MI>
class Periodic1DPQ1PreBasis
#endif
{
  static const int dim = GV::dimension;

public:

  //! The grid view that the FE basis is defined on
  using GridView = GV;
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
  using size_type = ST;

  template<class TP>
  using Node = Periodic1DPQ1Node<GV, size_type, TP>;

  template<class TP>
  using IndexSet = Periodic1DPQ1NodeIndexSet<GV, MI, TP, ST>;
#else
  //! Type used for indices and size information
  using size_type = std::size_t;

  using Node = Periodic1DPQ1Node<GV>;

  using IndexSet = Periodic1DPQ1NodeIndexSet<GV, MI>;
#endif

  /** \brief Type used for global numbering of the basis vectors */
  using MultiIndex = MI;

  //! Type used for prefixes handed to the size() method
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
  using SizePrefix = Dune::ReservedVector<size_type, 2>;
#else
  using SizePrefix = Dune::ReservedVector<size_type, 1>;
#endif

  //! Constructor for a given grid view object
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
  Periodic1DPQ1NodeFactory(const GridView& gv) :
#else
  Periodic1DPQ1PreBasis(const GridView& gv) :
#endif
    gridView_(gv)
  {}

  void initializeIndices()
  {}

  /** \brief Obtain the grid view that the basis is defined on
   */
  const GridView& gridView() const
  {
    return gridView_;
  }

  //! Update the stored grid view, to be called if the grid has changed
  void update (const GridView& gv)
  {
    gridView_ = gv;
  }

#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
  template<class TP>
  Node<TP> node(const TP& tp) const
  {
    return Node<TP>{tp};
  }

  template<class TP>
  IndexSet<TP> indexSet() const
  {
    return IndexSet<TP>{*this};
  }
#else
  Node makeNode() const
  {
    return Node{};
  }

  IndexSet makeIndexSet() const
  {
    return IndexSet{*this};
  }
#endif

  size_type size() const
  {
    return gridView_.size(dim)-1;
  }

  //! Return number possible values for next position in multi index
  size_type size(const SizePrefix prefix) const
  {
    if (prefix.size() == 0)
      return size();
    if (prefix.size() == 1)
      return 0;
    DUNE_THROW(RangeError, "Method size() can only be called for prefixes of length up to one");
  }

  //! Get the total dimension of the space spanned by this basis
  size_type dimension() const
  {
    return size()-1;
  }

  size_type maxNodeSize() const
  {
    return StaticPower<2,GV::dimension>::power;
  }

//protected:
  const GridView gridView_;
};



#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
template<typename GV, typename ST, typename TP>
class Periodic1DPQ1Node :
  public LeafBasisNode<ST, TP>
#else
template<typename GV>
class Periodic1DPQ1Node :
  public LeafBasisNode
#endif
{
  static const int dim = GV::dimension;
  static const int maxSize = StaticPower<2,GV::dimension>::power;

#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
  using Base = LeafBasisNode<ST,TP>;
#endif
  using FiniteElementCache = typename Dune::PQkLocalFiniteElementCache<typename GV::ctype, double, dim, 1>;

public:

#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
  using size_type = ST;
  using TreePath = TP;
#else
  using size_type = std::size_t;
#endif
  using Element = typename GV::template Codim<0>::Entity;
  using FiniteElement = typename FiniteElementCache::FiniteElementType;

#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
  Periodic1DPQ1Node(const TreePath& treePath) :
    Base(treePath),
#else
  Periodic1DPQ1Node() :
#endif
    finiteElement_(nullptr),
    element_(nullptr)
  {}

  //! Return current element, throw if unbound
  const Element& element() const
  {
    return *element_;
  }

  /** \brief Return the LocalFiniteElement for the element we are bound to
   *
   * The LocalFiniteElement implements the corresponding interfaces of the dune-localfunctions module
   */
  const FiniteElement& finiteElement() const
  {
    return *finiteElement_;
  }

  //! Bind to element.
  void bind(const Element& e)
  {
    element_ = &e;
    finiteElement_ = &(cache_.get(element_->type()));
    this->setSize(finiteElement_->size());
  }

protected:

  FiniteElementCache cache_;
  const FiniteElement* finiteElement_;
  const Element* element_;
};



#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
template<typename GV, class MI, class TP, class ST>
#else
template<typename GV, class MI>
#endif
class Periodic1DPQ1NodeIndexSet
{
  enum {dim = GV::dimension};

public:

#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
  using size_type = ST;
#else
  using size_type = std::size_t;
#endif

  /** \brief Type used for global numbering of the basis vectors */
  using MultiIndex = MI;

#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
  using NodeFactory = Periodic1DPQ1NodeFactory<GV, MI, ST>;

  using Node = typename NodeFactory::template Node<TP>;
#else
  using PreBasis = Periodic1DPQ1PreBasis<GV, MI>;

  using Node = Periodic1DPQ1Node<GV>;
#endif


#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
  Periodic1DPQ1NodeIndexSet(const NodeFactory& nodeFactory) :
    nodeFactory_(&nodeFactory)
#else
  Periodic1DPQ1NodeIndexSet(const PreBasis& preBasis) :
    preBasis_(&preBasis),
    node_(nullptr)
#endif
  {}

  /** \brief Bind the view to a grid element
   *
   * Having to bind the view to an element before being able to actually access any of its data members
   * offers to centralize some expensive setup code in the 'bind' method, which can save a lot of run-time.
   */
  void bind(const Node& node)
  {
    node_ = &node;
  }

  /** \brief Unbind the view
   */
  void unbind()
  {
    node_ = nullptr;
  }

  /** \brief Size of subtree rooted in this node (element-local)
   */
  size_type size() const
  {
    assert(node_ != nullptr);
    return node_->finiteElement().size();
  }

  //! Maps from subtree index set [0..size-1] to a globally unique multi index in global basis
  MultiIndex index(size_type i) const
  {
    Dune::LocalKey localKey = node_->finiteElement().localCoefficients().localKey(i);
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
    const auto& gridIndexSet = nodeFactory_->gridView().indexSet();
#else
    const auto& gridIndexSet = preBasis_->gridView().indexSet();
#endif
    const auto& element = node_->element();

    //return {{ gridIndexSet.subIndex(element,localKey.subEntity(),dim) }};

    MultiIndex idx{{gridIndexSet.subIndex(element,localKey.subEntity(),dim) }};

    // make periodic
    if (idx == gridIndexSet.size(dim)-1)
      idx = {{0}};

    return idx;
  }

protected:
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
  const NodeFactory* nodeFactory_;
#else
  const PreBasis* preBasis_;
#endif

  const Node* node_;
};

/** \brief Nodal basis of a scalar first-order Lagrangian finite element space
 *   on a one-dimensional domain with periodic boundary conditions
 *
 * \tparam GV The GridView that the space is defined on
 */
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
template<typename GV, class ST = std::size_t>
using Periodic1DPQ1NodalBasis = DefaultGlobalBasis<Periodic1DPQ1NodeFactory<GV, FlatMultiIndex<ST>, ST> >;
#else
template<typename GV>
using Periodic1DPQ1NodalBasis = DefaultGlobalBasis<Periodic1DPQ1PreBasis<GV, FlatMultiIndex<std::size_t> > >;
#endif

} // end namespace Functions
} // end namespace Dune

#endif // DUNE_FUNCTIONS_FUNCTIONSPACEBASES_PQ1NODALBASIS_HH
