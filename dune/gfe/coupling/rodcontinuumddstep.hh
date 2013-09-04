#ifndef ROD_CONTINUUM_DD_STEP_HH
#define ROD_CONTINUUM_DD_STEP_HH

#warning This file is deprecated.  Use the one in dune-gfe-coupling instead!

/** \file
 *  \brief Base class for rod-continuum domain decomposition algorithms
 */

#include <vector>

#include <dune/fufem/functionspacebases/p1nodalbasis.hh>
#include <dune/fufem/assemblers/boundaryfunctionalassembler.hh>
#include <dune/fufem/assemblers/localassemblers/neumannboundaryassembler.hh>

#include <dune/gfe/coupling/rodcontinuumcomplex.hh>


/** \brief Base class for rod-continua domain decomposition problems
 * \tparam RodGridType Dune grid type used for rod grids (must be 1d)
 * \tparam ContinuumGridType Dune grid type used for the continua grids
 */
template <class RodGridType, class ContinuumGridType>
class RodContinuumDDStep
{
    static const int dim = ContinuumGridType::dimension;
    
    // The type used for rod configurations
    typedef std::vector<RigidBodyMotion<double,dim> > RodConfigurationType;

    // The type used for continuum configurations
    typedef Dune::BlockVector<Dune::FieldVector<double,dim> > VectorType;
    typedef Dune::BlockVector<Dune::FieldVector<double,dim> > ContinuumConfigurationType;
    
    typedef Dune::BlockVector<Dune::FieldVector<double,6> >  RodCorrectionType;
    
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,3,3> > MatrixType;
                
    typedef typename P1NodalBasis<typename ContinuumGridType::LeafGridView,double>::LocalFiniteElement ContinuumLocalFiniteElement;
    
public:

    /** \brief Constructor for a complex with one rod and one continuum */
    RodContinuumDDStep(const RodContinuumComplex<RodGridType,ContinuumGridType>& complex/*,
                                    RodAssembler<typename RodGridType::LeafGridView,3>* rodAssembler,
                                    RiemannianTrustRegionSolver<RodGridType,RigidBodyMotion<double,3> >* rodSolver,
                                    const MatrixType* stiffnessMatrix3d,
                                    const Dune::shared_ptr< ::LoopSolver<VectorType> > solver*/)
      : complex_(complex)
    {
#if 0
        rods_["rod"].assembler_      = rodAssembler;
        rods_["rod"].solver_         = rodSolver;

        continua_["continuum"].stiffnessMatrix_ = stiffnessMatrix3d;
        continua_["continuum"].solver_          = solver;

        mergeRodDirichletAndCouplingBoundaries();
        mergeContinuumDirichletAndCouplingBoundaries();
#endif
    }
    
    /** \brief Do one domain decomposition step
     * \param[in,out] lambda The old and new iterate
     */
    virtual void iterate(std::map<std::pair<std::string,std::string>, RigidBodyMotion<double,3> >& lambda) = 0;

protected:

    std::set<std::string> rodsPerContinuum(const std::string& name) const;
    
    std::set<std::string> continuaPerRod(const std::string& name) const;

    /** \brief Add the content of one map to another, aborting rather than overwriting stuff
     */
    template <class X, class Y>
    static void insert(std::map<X,Y>& map1, const std::map<X,Y>& map2)
    {
        int oldSize = map1.size();
        map1.insert(map2.begin(), map2.end());
        assert(map1.size() == oldSize + map2.size());
    }

    //////////////////////////////////////////////////////////////////
    //  Data members related to the coupled problem
    //////////////////////////////////////////////////////////////////
    const RodContinuumComplex<RodGridType,ContinuumGridType>& complex_;
#if 0
protected:
    
    //////////////////////////////////////////////////////////////////
    //  Data members related to the rod problems
    //////////////////////////////////////////////////////////////////
    
    struct RodData
    {
        Dune::BitSetVector<6> dirichletAndCouplingNodes_;
    
        RodAssembler<typename RodGridType::LeafGridView,3>* assembler_;
    
        RodLocalStiffness<typename RodGridType::LeafGridView,double>* localStiffness_;
    
        RiemannianTrustRegionSolver<RodGridType,RigidBodyMotion<double,3> >* solver_;
    };
    
    /** \brief Simple const access to rods */
    const RodData& rod(const std::string& name) const
    {
        assert(rods_.find(name) != rods_.end());
        return rods_.find(name)->second;
    }

    std::map<std::string, RodData> rods_;
    
    typedef typename std::map<std::string, RodData>::iterator RodIterator;
#endif
public:    
    /** \todo Should be part of RodData, too */
    mutable std::map<std::string, RodConfigurationType> rodSubdomainSolutions_;
#if 0
protected:
    //////////////////////////////////////////////////////////////////
    //  Data members related to the continuum problems
    //////////////////////////////////////////////////////////////////

    struct ContinuumData
    {
        const MatrixType* stiffnessMatrix_;
    
        Dune::shared_ptr< ::LoopSolver<VectorType> > solver_;
    
        Dune::BitSetVector<dim> dirichletAndCouplingNodes_;
    
        LocalOperatorAssembler<ContinuumGridType, 
                             ContinuumLocalFiniteElement, 
                             ContinuumLocalFiniteElement,
                             Dune::FieldMatrix<double,dim,dim> >* localAssembler_;
    };
    
    /** \brief Simple const access to continua */
    const ContinuumData& continuum(const std::string& name) const
    {
        assert(continua_.find(name) != continua_.end());
        return continua_.find(name)->second;
    }

    std::map<std::string, ContinuumData> continua_;

    typedef typename std::map<std::string, ContinuumData>::iterator ContinuumIterator;
#endif    
public:
    /** \todo Should be part of ContinuumData, too */
    mutable std::map<std::string, ContinuumConfigurationType> continuumSubdomainSolutions_;

};


template <class RodGridType, class ContinuumGridType>
std::set<std::string> RodContinuumDDStep<RodGridType,ContinuumGridType>::
rodsPerContinuum(const std::string& name) const
{
    std::set<std::string> result;
    
    for (typename RodContinuumComplex<RodGridType,ContinuumGridType>::ConstCouplingIterator it = complex_.couplings_.begin(); 
         it!=complex_.couplings_.end(); ++it)
        if (it->first.second == name)
            result.insert(it->first.first);
    
    return result;
}

template <class RodGridType, class ContinuumGridType>
std::set<std::string> RodContinuumDDStep<RodGridType,ContinuumGridType>::
continuaPerRod(const std::string& name) const
{
    std::set<std::string> result;
    
    for (typename RodContinuumComplex<RodGridType,ContinuumGridType>::ConstCouplingIterator it = complex_.couplings_.begin(); 
         it!=complex_.couplings_.end(); ++it)
        if (it->first.first == name)
            result.insert(it->first.second);
    
    return result;
}

#endif
