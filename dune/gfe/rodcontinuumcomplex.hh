#ifndef ROD_CONTINUUM_COMPLEX_HH
#define ROD_CONTINUUM_COMPLEX_HH

#include <map>
#include <string>

#include <dune/common/shared_ptr.hh>

/** \brief A set of rods and a set of continua, all coupled to each other
 */
template <class RodGrid, class ContinuumGrid>
class RodContinuumComplex
{
public:

    /** \brief The set of rods, accessible by name (string) */
    std::map<std::string, Dune::shared_ptr<RodGrid> > rodGrids_;
    
    /** \brief The set of rods, accessible by name (string) */
    std::map<std::string, Dune::shared_ptr<ContinuumGrid> > continuumGrids_;
    
};

#endif    // ROD_CONTINUUM_COMPLEX_HH