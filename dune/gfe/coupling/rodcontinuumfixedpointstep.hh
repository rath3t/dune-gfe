#ifndef ROD_CONTINUUM_FIXED_POINT_STEP_HH
#define ROD_CONTINUUM_FIXED_POINT_STEP_HH

#include <vector>

// #include <dune/common/shared_ptr.hh>
// #include <dune/common/fvector.hh>
// #include <dune/common/fmatrix.hh>
// #include <dune/common/bitsetvector.hh>
// 
// #include <dune/istl/bcrsmatrix.hh>
// #include <dune/istl/bvector.hh>
// 
#include <dune/fufem/functionspacebases/p1nodalbasis.hh>
#include <dune/fufem/assemblers/boundaryfunctionalassembler.hh>
#include <dune/fufem/assemblers/localassemblers/neumannboundaryassembler.hh>

#include <dune/gfe/coupling/rodcontinuumcomplex.hh>
#include <dune/gfe/coupling/rodcontinuumddstep.hh>


template <class RodGridType, class ContinuumGridType>
class RodContinuumFixedPointStep
: public RodContinuumDDStep<RodGridType,ContinuumGridType>
{
    static const int dim = ContinuumGridType::dimension;
    
    // The type used for rod configurations
    typedef std::vector<RigidBodyMotion<dim> > RodConfigurationType;

    // The type used for continuum configurations
    typedef Dune::BlockVector<Dune::FieldVector<double,dim> > VectorType;
    typedef Dune::BlockVector<Dune::FieldVector<double,dim> > ContinuumConfigurationType;
    
    typedef Dune::BlockVector<Dune::FieldVector<double,6> >  RodCorrectionType;
    
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,3,3> > MatrixType;
                
    typedef typename P1NodalBasis<typename ContinuumGridType::LeafGridView,double>::LocalFiniteElement ContinuumLocalFiniteElement;
    
public:
    
    /** \brief Constructor for a complex with one rod and one continuum */
    RodContinuumFixedPointStep(const RodContinuumComplex<RodGridType,ContinuumGridType>& complex,
                                    double damping,
                                    RodAssembler<typename RodGridType::LeafGridView,3>* rodAssembler,
                                    RiemannianTrustRegionSolver<RodGridType,RigidBodyMotion<3> >* rodSolver,
                                    const MatrixType* stiffnessMatrix3d,
                                    const Dune::shared_ptr< ::LoopSolver<VectorType> > solver)
      : RodContinuumDDStep<RodGridType,ContinuumGridType>(complex),
        damping_(damping)
    {
        rods_["rod"].assembler_      = rodAssembler;
        rods_["rod"].solver_         = rodSolver;

        continua_["continuum"].stiffnessMatrix_ = stiffnessMatrix3d;
        continua_["continuum"].solver_          = solver;

        mergeRodDirichletAndCouplingBoundaries();
        mergeContinuumDirichletAndCouplingBoundaries();
    }

    /** \brief Constructor for a general complex */
    RodContinuumFixedPointStep(const RodContinuumComplex<RodGridType,ContinuumGridType>& complex,
                               double damping,
                               const std::map<std::string,RodAssembler<typename RodGridType::LeafGridView,3>*>& rodAssembler,
                               const std::map<std::string,RiemannianTrustRegionSolver<RodGridType,RigidBodyMotion<3> >*>& rodSolver,
                               const std::map<std::string,const MatrixType*>& stiffnessMatrix3d,
                               const std::map<std::string, const Dune::shared_ptr< ::LoopSolver<VectorType> > >& solver)
      : RodContinuumDDStep<RodGridType,ContinuumGridType>(complex),
        damping_(damping)
    {
        ///////////////////////////////////////////////////////////////////////////////////
        //  Rod-related data
        ///////////////////////////////////////////////////////////////////////////////////
        for (typename std::map<std::string,RodAssembler<typename RodGridType::LeafGridView,3>*>::const_iterator it = rodAssembler.begin();
             it != rodAssembler.end();
             ++it)
            rods_[it->first].assembler_ = it->second;
        
        for (typename std::map<std::string,RiemannianTrustRegionSolver<RodGridType,RigidBodyMotion<3> >*>::const_iterator it = rodSolver.begin();
             it != rodSolver.end();
             ++it)
            rods_[it->first].solver_ = it->second;

        ///////////////////////////////////////////////////////////////////////////////////
        //  Continuum-related data
        ///////////////////////////////////////////////////////////////////////////////////
        for (typename std::map<std::string,const MatrixType*>::const_iterator it = stiffnessMatrix3d.begin();
             it != stiffnessMatrix3d.end();
             ++it)
            continua_[it->first].stiffnessMatrix_ = it->second;
        
        for (typename std::map<std::string,const Dune::shared_ptr< ::LoopSolver<VectorType> > >::const_iterator it = solver.begin();
             it != solver.end();
             ++it)
            continua_[it->first].solver_ = it->second;
        
        mergeRodDirichletAndCouplingBoundaries();
        mergeContinuumDirichletAndCouplingBoundaries();
    }

    void mergeRodDirichletAndCouplingBoundaries();

    void mergeContinuumDirichletAndCouplingBoundaries();

    
    /** \brief Do one Steklov-Poincare step
     * \param[in,out] lambda The old and new iterate
     */
    void iterate(std::map<std::pair<std::string,std::string>, RigidBodyMotion<3> >& lambda);

protected:
    
    std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector> 
        rodDirichletToNeumannMap(const std::string& rodName, 
                                 const std::map<std::pair<std::string,std::string>,RigidBodyMotion<3> >& lambda) const;
    
    std::map<std::pair<std::string,std::string>,RigidBodyMotion<3> >
    continuumNeumannToDirichletMap(const std::string& continuumName,
                               const std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector>& forceTorque,
                               const std::map<std::pair<std::string,std::string>,RigidBodyMotion<3> >& centerOfTorque) const;

    //////////////////////////////////////////////////////////////////
    //  Data members related to the coupled problem
    //////////////////////////////////////////////////////////////////
    
    /** \brief Damping factor */
    double damping_;
    
protected:
    
    //////////////////////////////////////////////////////////////////
    //  Data members related to the rod problems
    //////////////////////////////////////////////////////////////////
    
    struct RodData
    {
        Dune::BitSetVector<6> dirichletAndCouplingNodes_;
    
        RodAssembler<typename RodGridType::LeafGridView,3>* assembler_;
    
        RodLocalStiffness<typename RodGridType::LeafGridView,double>* localStiffness_;
    
        RiemannianTrustRegionSolver<RodGridType,RigidBodyMotion<3> >* solver_;
    };
    
    /** \brief Simple const access to rods */
    const RodData& rod(const std::string& name) const
    {
        assert(rods_.find(name) != rods_.end());
        return rods_.find(name)->second;
    }

    std::map<std::string, RodData> rods_;
    
    typedef typename std::map<std::string, RodData>::iterator RodIterator;
    
protected:
    //////////////////////////////////////////////////////////////////
    //  Data members related to the continuum problems
    //////////////////////////////////////////////////////////////////

    struct ContinuumData
    {
        const MatrixType* stiffnessMatrix_;
    
        Dune::shared_ptr< ::LoopSolver<VectorType> > solver_;
    
        Dune::BitSetVector<dim> dirichletAndCouplingNodes_;
    
        LinearLocalAssembler<ContinuumGridType, 
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
    
};


template <class RodGridType, class ContinuumGridType>
void RodContinuumFixedPointStep<RodGridType,ContinuumGridType>::
mergeRodDirichletAndCouplingBoundaries()
{
    ////////////////////////////////////////////////////////////////////////////////////
    //  For each rod, merge the Dirichlet boundary with all interface boundaries
    //
    //  Currently, we can really only solve rod problems with complete Dirichlet
    //  boundary.  Hence there are more direct ways to construct the
    //  dirichletAndCouplingNodes field.  Yet like to keep the analogy to the continuum
    //  problem.  And maybe one day we have a more flexible rod solver, too.
    ////////////////////////////////////////////////////////////////////////////////////
        
    for (RodIterator rIt = rods_.begin(); rIt != rods_.end(); ++rIt) {
            
        // name of the current rod
        const std::string& name = rIt->first;
            
        // short-cut to avoid frequent map look-up
        Dune::BitSetVector<6>& dirichletAndCouplingNodes = rods_[name].dirichletAndCouplingNodes_;
        
        dirichletAndCouplingNodes.resize(this->complex_.rodGrid(name)->size(1));
        
        // first copy the true Dirichlet boundary
        const LeafBoundaryPatch<RodGridType>& dirichletBoundary = this->complex_.rods_.find(name)->second.dirichletBoundary_;

        for (int i=0; i<dirichletAndCouplingNodes.size(); i++)
            dirichletAndCouplingNodes[i] = dirichletBoundary.containsVertex(i);
        
        // get the names of all the continua that we couple with
        std::set<std::string> continuumNames = this->continuaPerRod(name);
        
        for (std::set<std::string>::const_iterator cIt = continuumNames.begin();
             cIt != continuumNames.end();
             ++cIt) {

            const LeafBoundaryPatch<RodGridType>& rodInterfaceBoundary 
                    = this->complex_.coupling(std::make_pair(name,*cIt)).rodInterfaceBoundary_;

            /** \todo Use the BoundaryPatch iterator here, for increased efficiency */
            for (int i=0; i<dirichletAndCouplingNodes.size(); i++) {
                bool v = rodInterfaceBoundary.containsVertex(i);
                for (int j=0; j<6; j++)
                    dirichletAndCouplingNodes[i][j] = dirichletAndCouplingNodes[i][j] or v;
            }
        
        }
        
        // We can only handle rod problems with a full Dirichlet boundary
        assert(dirichletAndCouplingNodes.count()==12);
        
    }
        
}


template <class RodGridType, class ContinuumGridType>
void RodContinuumFixedPointStep<RodGridType,ContinuumGridType>::
mergeContinuumDirichletAndCouplingBoundaries()
{
    ////////////////////////////////////////////////////////////////////////////////////
    //  For each continuum, merge the Dirichlet boundary with all interface boundaries
    ////////////////////////////////////////////////////////////////////////////////////
        
    for (ContinuumIterator cIt = continua_.begin(); cIt != continua_.end(); ++cIt) {
            
        // name of the current continuum
        const std::string& name = cIt->first;
            
        // short-cut to avoid frequent map look-up
        Dune::BitSetVector<dim>& dirichletAndCouplingNodes = continua_[name].dirichletAndCouplingNodes_;
        
        dirichletAndCouplingNodes.resize(this->complex_.continuumGrid(name)->size(dim));
        
        // first copy the true Dirichlet boundary
        const LeafBoundaryPatch<ContinuumGridType>& dirichletBoundary = this->complex_.continua_.find(name)->second.dirichletBoundary_;

        for (int i=0; i<dirichletAndCouplingNodes.size(); i++)
            dirichletAndCouplingNodes[i] = dirichletBoundary.containsVertex(i);
        
        // get the names of all the rods that we couple with
        std::set<std::string> rodNames = this->rodsPerContinuum(name);
        
        for (std::set<std::string>::const_iterator rIt = rodNames.begin();
             rIt != rodNames.end();
             ++rIt) {

            const LeafBoundaryPatch<ContinuumGridType>& continuumInterfaceBoundary 
                    = this->complex_.coupling(std::make_pair(*rIt,name)).continuumInterfaceBoundary_;

            /** \todo Use the BoundaryPatch iterator here, for increased efficiency */
            for (int i=0; i<dirichletAndCouplingNodes.size(); i++) {
                bool v = continuumInterfaceBoundary.containsVertex(i);
                for (int j=0; j<dim; j++)
                    dirichletAndCouplingNodes[i][j] = dirichletAndCouplingNodes[i][j] or v;
            }
            
        }
        
    }
        
}


template <class RodGridType, class ContinuumGridType>
std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector> RodContinuumFixedPointStep<RodGridType,ContinuumGridType>::
rodDirichletToNeumannMap(const std::string& rodName, 
                               const std::map<std::pair<std::string,std::string>,RigidBodyMotion<3> >& lambda) const
{
    // container for the subdomain solution
    RodConfigurationType& rodX = this->rodSubdomainSolutions_[rodName];
    rodX.resize(this->complex_.rodGrid(rodName)->size(1));
    
    ///////////////////////////////////////////////////////////
    //  Set the complete set of Dirichlet values
    ///////////////////////////////////////////////////////////
    const LeafBoundaryPatch<RodGridType>& dirichletBoundary = this->complex_.rod(rodName).dirichletBoundary_;
    const RodConfigurationType& dirichletValues = this->complex_.rod(rodName).dirichletValues_;
    
    for (size_t i=0; i<rodX.size(); i++)
        if (dirichletBoundary.containsVertex(i))
            rodX[i] = dirichletValues[i];
    
    typename std::map<std::pair<std::string,std::string>,RigidBodyMotion<3> >::const_iterator it = lambda.begin();
    for (; it!=lambda.end(); ++it) {
        
        const std::pair<std::string,std::string>& couplingName = it->first;
        
        if (couplingName.first != rodName)
            continue;
    
        // Use \lambda as a Dirichlet value for the rod
        const LeafBoundaryPatch<RodGridType>& interfaceBoundary = this->complex_.coupling(couplingName).rodInterfaceBoundary_;

        /** \todo Use the BoundaryPatch iterator, which will be a lot faster
         * once we use EntitySeed for its implementation
         */
        for (size_t i=0; i<rodX.size(); i++)
            if (interfaceBoundary.containsVertex(i))
                rodX[i] = it->second;
    }
    
    // Set the correct Dirichlet nodes
    rod(rodName).solver_->setIgnoreNodes(rod(rodName).dirichletAndCouplingNodes_);

    ////////////////////////////////////////////////////////////////////////////////
    //  Solve the Dirichlet problem
    ////////////////////////////////////////////////////////////////////////////////
           
    // Set initial iterate by interpolating between the Dirichlet values
    RodFactory<typename RodGridType::LeafGridView> rodFactory(this->complex_.rodGrid(rodName)->leafView());
    rodFactory.create(rodX);
    
    rod(rodName).solver_->setInitialSolution(rodX);
    
    // Solve the Dirichlet problem
    rod(rodName).solver_->solve();

    rodX = rod(rodName).solver_->getSol();

    //   Extract Neumann values
    std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector > result;

    for (it = lambda.begin(); it!=lambda.end(); ++it) {
        
        const std::pair<std::string,std::string>& couplingName = it->first;
        
        if (couplingName.first != rodName)
            continue;
        
        const LeafBoundaryPatch<RodGridType>& couplingBoundary = this->complex_.coupling(couplingName).rodInterfaceBoundary_;

        result[couplingName] = rod(rodName).assembler_->getResultantForce(couplingBoundary, rodX);
    }
    
    return result;
}

template <class RodGridType, class ContinuumGridType>
std::map<std::pair<std::string,std::string>,RigidBodyMotion<3> > 
RodContinuumFixedPointStep<RodGridType,ContinuumGridType>::
continuumNeumannToDirichletMap(const std::string& continuumName,
                               const std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector>& forceTorque,
                               const std::map<std::pair<std::string,std::string>,RigidBodyMotion<3> >& centerOfTorque) const
{
    ////////////////////////////////////////////////////
    //  Assemble the problem
    ////////////////////////////////////////////////////

    typedef P1NodalBasis<typename ContinuumGridType::LeafGridView,double> P1Basis;
    const LeafBoundaryPatch<ContinuumGridType>& dirichletBoundary = this->complex_.continuum(continuumName).dirichletBoundary_;
    P1Basis basis(dirichletBoundary.gridView());

    const MatrixType& stiffnessMatrix = *continuum(continuumName).stiffnessMatrix_;
    
    /////////////////////////////////////////////////////////////////////
    //  Turn the input force and torque into a Neumann boundary field
    /////////////////////////////////////////////////////////////////////
    
    // The weak form of the volume and Neumann data
    /** \brief Not implemented yet! */
    VectorType rhs(this->complex_.continuumGrid(continuumName)->size(dim));
    rhs = 0;

    typedef typename std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector>::const_iterator ForceIterator;
    
    for (ForceIterator it = forceTorque.begin(); it!=forceTorque.end(); ++it) {
        
        const std::pair<std::string,std::string>& couplingName = it->first;

        if (couplingName.second != continuumName)
            continue;
        
        // Use 'forceTorque' as a Neumann value for the rod
        const LeafBoundaryPatch<ContinuumGridType>& interfaceBoundary = this->complex_.coupling(couplingName).continuumInterfaceBoundary_;
        
        // 
        VectorType localNeumannValues;
        computeAveragePressure<typename ContinuumGridType::LeafGridView>(forceTorque.find(couplingName)->second, 
                                        interfaceBoundary, 
                                        centerOfTorque.find(couplingName)->second.r,
                                        localNeumannValues);
        
        BoundaryFunctionalAssembler<P1Basis> boundaryFunctionalAssembler(basis, interfaceBoundary);
        BasisGridFunction<P1Basis,VectorType> neumannValuesFunction(basis,localNeumannValues);
        NeumannBoundaryAssembler<ContinuumGridType, Dune::FieldVector<double,3> > localNeumannAssembler(neumannValuesFunction);
        boundaryFunctionalAssembler.assemble(localNeumannAssembler, rhs, false);

    }
    
    // Set the correct Dirichlet nodes
    /** \brief Don't write this BitSetVector at each iteration */
    Dune::BitSetVector<dim> dirichletNodes(rhs.size(),false);
    for (size_t i=0; i<dirichletNodes.size(); i++)
        dirichletNodes[i] = dirichletBoundary.containsVertex(i);
    dynamic_cast<IterationStep<VectorType>* >(continuum(continuumName).solver_->iterationStep_)->ignoreNodes_ 
           = &dirichletNodes;

    //  Initial iterate is 0,  all Dirichlet values are 0 (because we solve a correction problem
    VectorType x(dirichletNodes.size());
    x = 0;
    
    assert( (dynamic_cast<LinearIterationStep<MatrixType,VectorType>* >(continuum(continuumName).solver_->iterationStep_)) );
    dynamic_cast<LinearIterationStep<MatrixType,VectorType>* >(continuum(continuumName).solver_->iterationStep_)->setProblem(stiffnessMatrix, x, rhs);

    //solver.preprocess();
        
    continuum(continuumName).solver_->solve();
        
    x = continuum(continuumName).solver_->iterationStep_->getSol();

    /////////////////////////////////////////////////////////////////////////////////
    //  Average the continuum displacement on the coupling boundary
    /////////////////////////////////////////////////////////////////////////////////
    std::map<std::pair<std::string,std::string>,RigidBodyMotion<3> > averageInterface;

    for (ForceIterator it = forceTorque.begin(); it!=forceTorque.end(); ++it) {
        
        const std::pair<std::string,std::string>& couplingName = it->first;

        if (couplingName.second != continuumName)
            continue;
        
        // Use 'forceTorque' as a Neumann value for the rod
        const LeafBoundaryPatch<ContinuumGridType>& interfaceBoundary = this->complex_.coupling(couplingName).continuumInterfaceBoundary_;
        
        computeAverageInterface(interfaceBoundary, x, averageInterface[couplingName]);
        
    }
    
    return averageInterface;
}


/** \brief One preconditioned Richardson step
*/
template <class RodGridType, class ContinuumGridType>
void RodContinuumFixedPointStep<RodGridType,ContinuumGridType>::
iterate(std::map<std::pair<std::string,std::string>, RigidBodyMotion<3> >& lambda)
{
    ///////////////////////////////////////////////////////////////////
    //  Evaluate the Dirichlet-to-Neumann maps for the rods
    ///////////////////////////////////////////////////////////////////

    std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector> rodForceTorque;
    
    for (RodIterator it = rods_.begin(); it != rods_.end(); ++it) {
        
        const std::string& rodName = it->first;
    
        std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector> forceTorque = rodDirichletToNeumannMap(rodName, lambda);

        this->insert(rodForceTorque, forceTorque);
        
    }

    std::cout << "resultant rod forces and torques: "  << std::endl;
    typedef typename std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector>::iterator ForceIterator;
    for (ForceIterator it = rodForceTorque.begin(); it != rodForceTorque.end(); ++it)
        std::cout << "    [" << it->first.first << ", " << it->first.second << "] -- "
                  << it->second << std::endl;

    ///////////////////////////////////////////////////////////////
    //  Flip orientation of all rod forces, to account for opposing normals.
    ///////////////////////////////////////////////////////////////

    for (ForceIterator it = rodForceTorque.begin(); it != rodForceTorque.end(); ++it)
        it->second *= -1;


    ///////////////////////////////////////////////////////////////
    //  Solve the Neumann problems for the continua
    ///////////////////////////////////////////////////////////////
            
    std::map<std::pair<std::string,std::string>, RigidBodyMotion<3> > averageInterface;
            
    for (ContinuumIterator it = continua_.begin(); it != continua_.end(); ++it) {
        
        const std::string& continuumName = it->first;
    
        std::map<std::pair<std::string,std::string>, RigidBodyMotion<3> > localAverageInterface
                = continuumNeumannToDirichletMap(continuumName,
                                                rodForceTorque,
                                                lambda);

        this->insert(averageInterface, localAverageInterface);
        
    }

    std::cout << "averaged interfaces: " << std::endl;
    for (typename std::map<std::pair<std::string,std::string>, RigidBodyMotion<3> >::const_iterator it = averageInterface.begin(); it != averageInterface.end(); ++it)
        std::cout << "    [" << it->first.first << ", " << it->first.second << "] -- "
                  << it->second << std::endl;

    //////////////////////////////////////////////////////////////
    //   Compute new damped interface value
    //////////////////////////////////////////////////////////////
    
    typename std::map<std::pair<std::string,std::string>,RigidBodyMotion<3> >::iterator it = lambda.begin();
    typename std::map<std::pair<std::string,std::string>, RigidBodyMotion<3> >::const_iterator aIIt = averageInterface.begin();

    for (; it!=lambda.end(); ++it, ++aIIt) {

        assert(it->first == aIIt->first);
        
        const std::pair<std::string,std::string>& interfaceName = it->first;
        
        const RigidBodyMotion<dim>& referenceInterface = this->complex_.coupling(interfaceName).referenceInterface_;

        for (int j=0; j<dim; j++)
            it->second.r[j] = (1-damping_) * it->second.r[j] 
                                       + damping_ * (referenceInterface.r[j] + averageInterface[interfaceName].r[j]);

        lambda[interfaceName].q = Rotation<3,double>::interpolate(lambda[interfaceName].q, 
                                                       referenceInterface.q.mult(averageInterface[interfaceName].q), 
                                                       damping_);
        
    }

}

#endif
