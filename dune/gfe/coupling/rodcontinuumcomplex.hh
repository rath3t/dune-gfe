#ifndef ROD_CONTINUUM_COMPLEX_HH
#define ROD_CONTINUUM_COMPLEX_HH

#include <map>
#include <string>

#include <dune/common/shared_ptr.hh>

#include <dune/istl/bvector.hh>

#include <dune/gfe/rigidbodymotion.hh>

/** \brief A set of rods and a set of continua, all coupled to each other
 */
template <class RodGrid, class ContinuumGrid>
class RodContinuumComplex
{
    dune_static_assert(RodGrid::dimension==1, "The RodGrid has to be one-dimensional!");

    typedef std::vector<RigidBodyMotion<3> > RodConfiguration;
    
    typedef Dune::BlockVector<Dune::FieldVector<double,3> > ContinuumConfiguration;
    
    struct Coupling
    {
        LeafBoundaryPatch<RodGrid> rodInterfaceBoundary_;
        
        LeafBoundaryPatch<ContinuumGrid> continuumInterfaceBoundary_;
    };
    
public:
    
    /** \brief Simple const access to rod grids */
    const Dune::shared_ptr<RodGrid> rodGrid(const std::string& name) const
    {
        assert(rodGrids_.find(name) != rodGrids_.end());
        return rodGrids_.find(name)->second;
    }

    /** \brief Simple const access to continuum grids */
    const Dune::shared_ptr<ContinuumGrid> continuumGrid(const std::string& name) const
    {
        assert(continuumGrids_.find(name) != continuumGrids_.end());
        return continuumGrids_.find(name)->second;
    }
    
    const LeafBoundaryPatch<ContinuumGrid> continuumDirichletBoundary(const std::string& name) const
    {
        assert(continuumDirichletBoundaries_.find(name) != continuumDirichletBoundaries_.end());
        return continuumDirichletBoundaries_.find(name)->second;
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
    std::map<std::string, Dune::shared_ptr<RodGrid> > rodGrids_;
    
    /** \brief A Dirichlet boundary for each rod */
    std::map<std::string, LeafBoundaryPatch<RodGrid> > rodDirichletBoundaries_;
    
    /** \brief The Dirichlet values for each rod */
    std::map<std::string, RodConfiguration> rodDirichletValues_;
    
    /////////////////////////////////////////////////////////////////////
    //  Data concerning the individual continuum problems
    /////////////////////////////////////////////////////////////////////

    /** \brief The set of continua, accessible by name (string) */
    std::map<std::string, Dune::shared_ptr<ContinuumGrid> > continuumGrids_;
    
    /** \brief A Dirichlet boundary for each continuum */
    std::map<std::string, LeafBoundaryPatch<ContinuumGrid> > continuumDirichletBoundaries_;

    /** \brief The Dirichlet values for each continuum */
    std::map<std::string, ContinuumConfiguration> continuumDirichletValues_;
    
    /////////////////////////////////////////////////////////////////////
    //   Data about the couplings
    /////////////////////////////////////////////////////////////////////

    std::map<std::pair<std::string,std::string>, Coupling> couplings_;
    
};

#endif    // ROD_CONTINUUM_COMPLEX_HH