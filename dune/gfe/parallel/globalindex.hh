/** \file
*
*  \brief Implement the functionality that provides a globally unique index for
*         all kinds of entities of a Dune grid. Such functionality is relevant
*         for a number of cases:
*         e.g. to map a degree of freedom associated wit an entity to its
*         location in a global matrix or global vector; e.g. to associate data
*         with entities in an adaptive mesh refinement scheme.
*
*  Nota bene: the functionality to provide a globally unique index for an entity, over all
*             processes in a distributed memory parallel compute environment is at present
*  not available in the Dune framework; therefore, we implement such functionality in a class
*  that is templatized by the GridView Type and the codimension of the entity for which the
*  globaly unique index shall be computed.
*
*  Method: (1) we use the UniqueEntityPartition class that reports if a specific entity belongs
*              to a specific process;
*
*          (2) we compute the number of entities that are owned by the specific process;
*
*          (3) we communicate the index of entities that are owned by the process to processes
*              that also contain these entities but do not own them, so that on a non-owner process
*              we have information on the index of the entity that it got from the owner-process;
*
*  Thus, we can access the same element in a global, distributed matrix that is managed
*  by all processes; a typical case would be using matrix and vector routines from the PETSc
*  or trilinos parallel linear algebra packages for distributed memory parallel computers.
*
*  Copyright by Benedikt Oswald and Patrick Leidenberger, 2002-2010. All rights reserved.
*
*  \author    Benedikt Oswald, Patrick Leidenberger
*
*  \warning   None
*
*  \attention globally unique indices are ONLY provided for entities of the
*             InteriorBorder_Partition type, NOT for the Ghost_Partition type !!!
*/

#ifndef GLOBALUNIQUEINDEX_HH_
#define GLOBALUNIQUEINDEX_HH_

/** \brief Include standard header files. */
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <utility>

/** include base class functionality for the communication interface */
#include <dune/grid/common/datahandleif.hh>

// Include proprietary header files.
#include "uniqueentitypartition.hh"

#include <dune/common/version.hh>

/** include parallel capability */
#if HAVE_MPI
  #include <dune/common/parallel/mpihelper.hh>
#endif


/*********************************************************************************************/
/* calculate globally unique index over all processes in a Dune grid that is distributed     */
/* over all the processes in a distributed memory parallel environment, for all entities     */
/* of a given codim in a given GridView, assuming they all have the same geometry,           */
/* i.e. codim, type                                                                          */
/*********************************************************************************************/

namespace Dune {

template<class GridView, int CODIM>
class GlobalIndexSet
{
private:
  /** define data types */
  typedef typename GridView::Grid Grid;

  typedef typename GridView::Grid::GlobalIdSet GlobalIdSet;
  typedef typename GridView::Grid::GlobalIdSet::IdType IdType;
  typedef typename GridView::Traits::template Codim<0>::Iterator Iterator;
  typedef typename GridView::Traits::template Codim<CODIM>::Entity Entity;


#if HAVE_MPI
  /** define data type related to collective communication */
  typedef typename Dune::template CollectiveCommunication<MPI_Comm> CollectiveCommunication;
#else
  typedef typename Grid::CollectiveCommunication CollectiveCommunication;
#endif

  typedef std::map<IdType,int> MapId2Index;
  typedef std::map<int,int>    IndexMap;

  typedef UniqueEntityPartition<GridView,CODIM> UniqueEntityPartitionType;

private:
  /* A DataHandle class to communicate the global index from the
   * owning to the non-owning entity; the class is based on the MinimumExchange
   * class in the parallelsolver.hh header file.
   */
  template<class GlobalIdSet, class IdType, class Map, typename ValueType>
  class IndexExchange           :       public Dune::CommDataHandleIF<IndexExchange<GlobalIdSet,IdType,Map,ValueType>,ValueType>
  {
  public:
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
      IdType id=globalidset_.id(e);

      buff.write((*mapid2entity_.find(id)).second);
    }

    /*! unpack data from message buffer to user

      n is the number of objects sent by the sender
    */
    template<class MessageBuffer, class EntityType>
    void scatter (MessageBuffer& buff, const EntityType& entity, size_t n)
    {
      ValueType x;
      buff.read(x);

      /** only if the incoming index is a valid one,
	  i.e. if it is greater than zero, will it be
	  inserted as the global index; it is made
	  sure in the upper class, i.e. GlobalUniqueIndex,
	  that non-owning processes use -1 to mark an entity
	  that they do not own.
      */
      if(x >= 0)
	{
	  IdType id = globalidset_.id(entity);
	  mapid2entity_.erase(id);
	  mapid2entity_.insert(std::make_pair(id,x));

	  const int lindex = indexSet_.index(entity);
	  localGlobalMap_[lindex] = x;
	  globalLocalMap_[x]      = lindex;
	}
    }

    //! constructor
    IndexExchange (const GlobalIdSet& globalidset, Map& mapid2entity, int &rank,
		   const typename GridView::IndexSet& localIndexSet, IndexMap& localGlobal, IndexMap& globalLocal)
      : globalidset_(globalidset),
	mapid2entity_(mapid2entity),
	rank_(rank),
	indexSet_(localIndexSet),
	localGlobalMap_(localGlobal),
	globalLocalMap_(globalLocal)
    {
    }

  private:
    const GlobalIdSet& globalidset_;
    Map& mapid2entity_;
    int& rank_;

    const typename GridView::IndexSet& indexSet_;
    IndexMap& localGlobalMap_;
    IndexMap& globalLocalMap_;
  };

public:
  /**********************************************************************************************************************/
  /* \brief class constructor - when the class is instantiated by passing a const reference to a GridView object,       */
  /*                      we calculate the complete set of global unique indices so that we can then                    */
  /*  later query the global index, by directly passing the entity in question: then the respective global              */
  /*  index is returned.                                                                                                */
  /**********************************************************************************************************************/
  GlobalIndexSet(const GridView& gridview)
    :       gridview_(gridview),
	    uniqueEntityPartition_(gridview)
  {
    int rank = gridview.comm().rank();
    int size = gridview.comm().size();

    nLocalEntity_=0;
    nGlobalEntity_=0;

    nLocalEntity_ = uniqueEntityPartition_.numOwners();


    /** compute the global, non-redundant number of entities, i.e. the number of entities in the set
     *  without double, aka. redundant entities, on the interprocessor boundary via global reduce. */

    const CollectiveCommunication& collective = gridview_.comm();
    nGlobalEntity_ = collective.template sum<int>(nLocalEntity_);


    /** communicate the number of locally owned entities to all other processes so that the respective offset
     *  can be calculated on the respective processor; we use the Dune mpi collective communication facility
     *  for this; first, we gather the number of locally owned entities on the root process and, second, we
     *  broadcast the array to all processes where the respective offset can be calculated. */

    std::vector<int> offset(size);
    std::fill(offset.begin(), offset.end(), 0);

    /** gather number of locally owned entities on root process */
    collective.template gather<int>(&nLocalEntity_,offset.data(),1,0);

    /** broadcast the array containing the number of locally owned entities to all processes */
    collective.template broadcast<int>(offset.data(),size,0);

    indexOffset_.clear();
    indexOffset_.resize(size,0);

    for(unsigned int ii=0; ii<indexOffset_.size(); ++ii)
	for(unsigned int jj=0; jj < ii; ++jj)
	    indexOffset_[ii] += offset[jj];

    /** compute globally unique index over all processes; the idea of the algorithm is as follows: if
     *  an entity is owned by the process, it is assigned an index that is the addition of the offset
     *  specific for this process and a consecutively incremented counter; if the entity is not owned
     *  by the process, it is assigned -1, which signals that this specific entity will get its global
     *  unique index through communication afterwards;
     *
     *  thus, the calculation of the globally unique index is divided into 2 stages:
     *
     *  (1) we calculate the global index independently;
     *
     *  (2) we achieve parallel adjustment by communicating the index
     *      from the owning entity to the non-owning entity.
     *
     */

    /** 1st stage of global index calculation: calculate global index for owned entities */
     /** intialize map that stores an entity's global index via it's globally unique id as key */
    MapId2Index globalIndex;

    const typename GridView::IndexSet& indexSet = gridview.indexSet();

    const GlobalIdSet &globalIdSet=gridview_.grid().globalIdSet();  /** retrieve globally unique Id set */
    int myoffset=indexOffset_[rank];

    int globalcontrib=0;    /** initialize contribution for the global index */

    std::vector<bool> firstTime(gridview_.size(CODIM));
    std::fill(firstTime.begin(), firstTime.end(), true);

    for(Iterator iter = gridview_.template begin<0>();iter!=gridview_.template end<0>(); ++iter)
    {
#if DUNE_VERSION_NEWER(DUNE_GRID,2,4)
      for (size_t i=0; i<iter->subEntities(CODIM); i++)
#else
      for (size_t i=0; i<iter->template count<CODIM>(); i++)
#endif
      {
        IdType id=globalIdSet.subId(*iter,i,CODIM);                 /** retrieve the entity's id */

        int idx = gridview_.indexSet().subIndex(*iter,i,CODIM);

        if (! firstTime[idx] )
          continue;

        firstTime[idx] = false;

        if(uniqueEntityPartition_.owner(idx) == true)  /** if the entity is owned by the process, go ahead with computing the global index */
        {
          const int gindex = myoffset + globalcontrib;    /** compute global index */
          globalIndex.insert(std::make_pair(id,gindex)); /** insert pair (key, value) into the map */

          const int lindex = idx;
          localGlobalMap_[lindex] = gindex;
          globalLocalMap_[gindex] = lindex;

          globalcontrib++;                                /** increment contribution to global index */
        }
        else /** if entity is not owned, insert -1 to signal not yet calculated global index */
        {
          globalIndex.insert(std::make_pair(id,-1));
        }
      }

    }

    /** 2nd stage of global index calculation: communicate global index for non-owned entities */

    // Create the data handle and communicate.
    IndexExchange<GlobalIdSet,IdType,MapId2Index,int> dataHandle(globalIdSet,globalIndex,rank,indexSet,localGlobalMap_,globalLocalMap_);
    gridview_.communicate(dataHandle, Dune::All_All_Interface, Dune::ForwardCommunication);
  }


  /** \brief Given a local index, retrieve its index globally unique over all processes. */
  int globalIndex(const int& localIndex) const {
    return localGlobalMap_.find(localIndex)->second;
  }

  int localIndex(const int& globalIndex) const {
    return globalLocalMap_.find(globalIndex)->second;
  }

  int globalIndex(const typename GridView::template Codim<CODIM>::Entity& entity) const
  {
    return localGlobalMap_.find(gridview_.indexSet().index(entity))->second;
  }

  int localIndex(const typename GridView::template Codim<CODIM>::Entity& entity) const {
    return gridview_.indexSet().index(entity);
  }

  inline unsigned int nGlobalEntity() const
  {
    return(nGlobalEntity_);
  }

  inline unsigned int nOwnedLocalEntity() const
  {
    return(nLocalEntity_);
  }

  const GridView& getGridView() const {
    return gridview_;
  }

protected:
  /** store data members */
  const GridView gridview_;                                                       /** store a const reference to a gridview */
  UniqueEntityPartitionType uniqueEntityPartition_;
  int nLocalEntity_;                                                      /** store number of entities that are owned by the local */
  int nGlobalEntity_;                                             /** store global number of entities, i.e. number of entities without rendundant entities on interprocessor boundaries */
  std::vector<int> indexOffset_;                          /** store offset of entity index on every process */

  IndexMap localGlobalMap_;
  IndexMap globalLocalMap_;
};

}  // namespace Dune

#endif /* GLOBALUNIQUEINDEX_HH_ */
