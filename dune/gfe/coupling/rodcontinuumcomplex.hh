#ifndef ROD_CONTINUUM_COMPLEX_HH
#define ROD_CONTINUUM_COMPLEX_HH

#include <map>
#include <string>

#include <dune/common/shared_ptr.hh>

#include <dune/istl/bvector.hh>

#include <dune/fufem/boundarypatch.hh>
#include <dune/gfe/rigidbodymotion.hh>

/** \brief A set of rods and a set of continua, all coupled to each other
 */
template <class RodGrid, class ContinuumGrid>
class RodContinuumComplex
{
    dune_static_assert(RodGrid::dimension==1, "The RodGrid has to be one-dimensional!");
    
    static const int dim = ContinuumGrid::dimension;

    typedef std::vector<RigidBodyMotion<3> > RodConfiguration;
    
    typedef Dune::BlockVector<Dune::FieldVector<double,3> > ContinuumConfiguration;
    
    /** \brief Holds all data for a rod/continuum coupling */
    struct Coupling
    {
        LeafBoundaryPatch<RodGrid> rodInterfaceBoundary_;

        LeafBoundaryPatch<ContinuumGrid> continuumInterfaceBoundary_;
        
        /** \brief The orientation of the interface in the reference configuration */
        RigidBodyMotion<dim> referenceInterface_;
    };
    
    /** \brief Holds all data for a rod subproblem */
    struct RodData
    {
        Dune::shared_ptr<RodGrid> grid_;
        
        LeafBoundaryPatch<RodGrid> dirichletBoundary_;
        
        RodConfiguration dirichletValues_;
    };
    
    /** \brief Holds all data for a continuum subproblem */
    struct ContinuumData
    {
        Dune::shared_ptr<ContinuumGrid> grid_;
        
        LeafBoundaryPatch<ContinuumGrid> dirichletBoundary_;
        
        ContinuumConfiguration dirichletValues_;
    };
    
public:
    
    /** \brief Iterator over the rods */
    typedef typename std::map<std::string, RodData>::iterator       RodIterator;
    typedef typename std::map<std::string, RodData>::const_iterator ConstRodIterator;
    
    /** \brief Iterator over the continua */
    typedef typename std::map<std::string, ContinuumData>::iterator       ContinuumIterator;
    typedef typename std::map<std::string, ContinuumData>::const_iterator ConstContinuumIterator;
    
    /** \brief Iterator over the couplings */
    typedef typename std::map<std::pair<std::string,std::string>, Coupling>::const_iterator ConstCouplingIterator;
    
    /** \brief Simple const access to rod grids */
    const Dune::shared_ptr<RodGrid> rodGrid(const std::string& name) const
    {
        assert(rods_.find(name) != rods_.end());
        return rods_.find(name)->second.grid_;
    }

    /** \brief Simple const access to continuum grids */
    const Dune::shared_ptr<ContinuumGrid> continuumGrid(const std::string& name) const
    {
        assert(continua_.find(name) != continua_.end());
        return continua_.find(name)->second.grid_;
    }
    
    /** \brief Simple const access to continua */
    const ContinuumData continuum(const std::string& name) const
    {
        assert(continua_.find(name) != continua_.end());
        return continua_.find(name)->second;
    }
    
    /** \brief Simple const access to rods */
    const RodData rod(const std::string& name) const
    {
        assert(rods_.find(name) != rods_.end());
        return rods_.find(name)->second;
    }
    
    /** \brief Simple const access to couplings */
    const Coupling& coupling(const std::pair<std::string,std::string>& name) const
    {
        assert(couplings_.find(name) != couplings_.end());
        return couplings_.find(name)->second;
    }

    /////////////////////////////////////////////////////////////////////
    //  Data concerning the individual rod problems
    /////////////////////////////////////////////////////////////////////

    /** \brief The set of rods, accessible by name (string) */
    std::map<std::string, RodData > rods_;
    
    /////////////////////////////////////////////////////////////////////
    //  Data concerning the individual continuum problems
    /////////////////////////////////////////////////////////////////////

    /** \brief The set of continua, accessible by name (string) */
    std::map<std::string, ContinuumData> continua_;
    
    /////////////////////////////////////////////////////////////////////
    //   Data about the couplings
    /////////////////////////////////////////////////////////////////////

    std::map<std::pair<std::string,std::string>, Coupling> couplings_;
    
};

#endif    // ROD_CONTINUUM_COMPLEX_HH