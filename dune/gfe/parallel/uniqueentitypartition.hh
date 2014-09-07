/** \file
 *  \brief implement universally usable, unique partitioning of entities (vertex, edge, face, element)
 *   in a Dune grid; it is a general problem in parallel calculation to assign entities in a unique
 *   way to processors; thus, we need a unique criterion; here, we will use the rule that an entity
 *   is always assigned to the process with the lowest rank.
 *
 *
 *   In particular, it is the objective to provide this functionality in a unique way so that it can
 *   be used in a wide range of an application's needs: e.g., it can be used to assign faces,
 *   situated on the inter processor boundary, in a unique way, i.e. the face is always assigned to
 *   the process with the lower rank. A concrete application of this mechanism is the implementation
 *   of the transparent integral boundary condition to remove redundant triangular faces of the
 *   fictitious boundary surface. Indeed, the implementation of the UniqueEntityPartition class has
 *   been motivated by this need. However, its usage is much wider. It will also be the underlying
 *   infrastructure to calculate global, consecutive, zero-starting indices for grid entities over all
 *   the processes over which the grid is partitioned.
 *
 *
 *  \author    Benedikt Oswald, Patrick Leidenberger
 *
 *  \warning   None
 *
 *  \attention
 *
 *  \bug
 *
 *  \todo
 */

#ifndef ENTITY_PARTITONING_HH
#define ENTITY_PARTITONING_HH


/** \brief Include standard header files. */
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <utility>

#include <dune/common/version.hh>

/** Include base class functionality for the communication interface */
#include <dune/grid/common/datahandleif.hh>


/*********************************************************************************************/
/* calculate unique partitioning for all entities of a given codim in a given GridView,      */
/* assuming they all have the same geometry, i.e. codim, type                                */
/*********************************************************************************************/
template<class GridView, int CODIM>
class UniqueEntityPartition
{
private:
  /* A DataHandle class to cacluate the minimum of a std::vector which is accompanied by an index set */
  template<class IS, class V> // mapper type and vector type
  class MinimumExchange		:	public Dune::CommDataHandleIF<MinimumExchange<IS,V>,typename V::value_type>
  {
	  public:
		//! export type of data for message buffer
		typedef typename V::value_type DataType;

		//! returns true if data for this codim should be communicated
		bool contains (int dim, int codim) const
		{
		  return(codim==CODIM);
		}

		//! returns true if size per entity of given dim and codim is a constant
		bool fixedsize (int dim, int codim) const
		{
		  return(true);
		}

		/*! how many objects of type DataType have to be sent for a given entity
		 *
		 *  Note: Only the sender side needs to know this size. */
		template<class EntityType>
		size_t size (EntityType& e) const
		{
		  return(1);
		}

		/*! pack data from user to message buffer */
		template<class MessageBuffer, class EntityType>
		void gather (MessageBuffer& buff, const EntityType& e) const
		{
		  buff.write(v_[indexset_.index(e)]);
		}

		/*! unpack data from message buffer to user

		  n is the number of objects sent by the sender
		*/
		template<class MessageBuffer, class EntityType>
		void scatter (MessageBuffer& buff, const EntityType& e, size_t n)
		{
		  DataType x; buff.read(x);
		  if (x>=0) // other is -1 means, he does not want it
		  {
			v_[indexset_.index(e)] = std::min(x,v_[indexset_.index(e)]);
		  }
		}

		//! constructor
		MinimumExchange (const IS& indexset, V& v)
		: indexset_(indexset),
		  v_(v)
		{
			;
		}

	  private:
		const IS& indexset_;
		V& v_;
	  };

public:
  /** declare often uses types */
  typedef typename GridView::Traits::template Codim<CODIM>::Iterator Iterator;
  typedef typename GridView::Traits::template Codim<CODIM>::Entity Entity;

  /*! \brief Constructor needs to know the grid function space
   */
  UniqueEntityPartition (const GridView& gridview)
  :	gridview_(gridview),
   	assignment_(gridview.size(CODIM)),
   	rank_(gridview_.comm().rank())
  {
	/** extract types from the GridView data type */
	typedef typename GridView::IndexSet IndexSet;

	// assign own rank to entities that I might have
        for(auto it = gridview_.template begin<0>();it!=gridview_.template end<0>(); ++it)
#if DUNE_VERSION_NEWER(DUNE_GRID,2,4)
          for (int i=0; i<it->subEntities(CODIM); i++)
#else
          for (int i=0; i<it->template count<CODIM>(); i++)
#endif
          {
            assignment_[gridview_.indexSet().template subIndex(*it,i,CODIM)]
              = ( (it->template subEntity<CODIM>(i)->partitionType()==Dune::InteriorEntity) || (it->template subEntity<CODIM>(i)->partitionType()==Dune::BorderEntity) )
              ? rank_  // set to own rank
              : - 1;   // it is a ghost entity, I will not possibly own it.
          }

        /** exchange entity index through communication */
	MinimumExchange<IndexSet,std::vector<int> > dh(gridview_.indexSet(),assignment_);

	gridview_.communicate(dh,Dune::All_All_Interface,Dune::ForwardCommunication);

	/* convert vector of minimum ranks to assignment vector */
        for (size_t i=0; i<assignment_.size(); i++)
          assignment_[i] = (assignment_[i] == rank_) ? 1 : 0;
  }

  /** answer question if entity belongs to me, to this process */
  bool owner(size_t i)
  {
    return assignment_[i];
  }

  size_t numOwners() const
  {
    return std::accumulate(assignment_.begin(), assignment_.end(), 0);
  }

  /** answer question if entity belongs to me, to this process */
  bool owner(const Entity& entity)
  {
	return(assignment_[gridview_.indexSet().template index(entity)]);
  }

private:
  /** declare private data members */
  const GridView& gridview_;
  std::vector<int> assignment_;
  int rank_;
};

#endif /** ENTITY_PARTITONING_HH */
