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