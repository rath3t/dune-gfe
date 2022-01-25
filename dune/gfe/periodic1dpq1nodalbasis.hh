// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_GFE_PERIODIC_1D_PQ1NODALBASIS_HH
#define DUNE_GFE_PERIODIC_1D_PQ1NODALBASIS_HH

#include <dune/common/exceptions.hh>
#include <dune/common/math.hh>

#include <dune/localfunctions/lagrange/pqkfactory.hh>

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

template<typename GV>
class Periodic1DPQ1Node;

template<typename GV, class MI>
class Periodic1DPQ1NodeIndexSet;

template<typename GV, class MI>
class Periodic1DPQ1PreBasis
{
  static const int dim = GV::dimension;

public:

  //! The grid view that the FE basis is defined on
  using GridView = GV;
  //! Type used for indices and size information
  using size_type = std::size_t;

  using Node = Periodic1DPQ1Node<GV>;

  using IndexSet = Periodic1DPQ1NodeIndexSet<GV, MI>;

  /** \brief Type used for global numbering of the basis vectors */
  using MultiIndex = MI;

  //! Type used for prefixes handed to the size() method
  using SizePrefix = Dune::ReservedVector<size_type, 1>;

  //! Constructor for a given grid view object
  Periodic1DPQ1PreBasis(const GridView& gv) :
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

  Node makeNode() const
  {
    return Node{};
  }

  IndexSet makeIndexSet() const
  {
    return IndexSet{*this};
  }

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
    return Dune::power(2,GV::dimension);
  }

//protected:
  const GridView gridView_;
};



template<typename GV>
class Periodic1DPQ1Node :
  public LeafBasisNode
{
  static const int dim = GV::dimension;
  static const int maxSize = Dune::power(2,GV::dimension);

  using FiniteElementCache = typename Dune::PQkLocalFiniteElementCache<typename GV::ctype, double, dim, 1>;

public:

  using size_type = std::size_t;
  using Element = typename GV::template Codim<0>::Entity;
  using FiniteElement = typename FiniteElementCache::FiniteElementType;

  Periodic1DPQ1Node() :
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



template<typename GV, class MI>
class Periodic1DPQ1NodeIndexSet
{
  enum {dim = GV::dimension};

public:

  using size_type = std::size_t;

  /** \brief Type used for global numbering of the basis vectors */
  using MultiIndex = MI;
  using PreBasis = Periodic1DPQ1PreBasis<GV, MI>;
  using Node = Periodic1DPQ1Node<GV>;

  Periodic1DPQ1NodeIndexSet(const PreBasis& preBasis) :
    preBasis_(&preBasis),
    node_(nullptr)
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
    const auto& gridIndexSet = preBasis_->gridView().indexSet();
    const auto& element = node_->element();

    //return {{ gridIndexSet.subIndex(element,localKey.subEntity(),dim) }};

    MultiIndex idx{{gridIndexSet.subIndex(element,localKey.subEntity(),dim) }};

    // make periodic
    if (idx == gridIndexSet.size(dim)-1)
      idx = {{0}};

    return idx;
  }

protected:
  const PreBasis* preBasis_;

  const Node* node_;
};

/** \brief Nodal basis of a scalar first-order Lagrangian finite element space
 *   on a one-dimensional domain with periodic boundary conditions
 *
 * \tparam GV The GridView that the space is defined on
 */
template<typename GV>
using Periodic1DPQ1NodalBasis = DefaultGlobalBasis<Periodic1DPQ1PreBasis<GV, FlatMultiIndex<std::size_t> > >;

} // end namespace Functions
} // end namespace Dune

#endif // DUNE_FUNCTIONS_FUNCTIONSPACEBASES_PQ1NODALBASIS_HH
