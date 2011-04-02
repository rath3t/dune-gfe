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
#include <dune/fufem/assemblers/boundaryfunctionalassembler.hh>
#include <dune/fufem/assemblers/localassemblers/neumannboundaryassembler.hh>

#include <dune/gfe/coupling/rodcontinuumcomplex.hh>


template <class RodGridType, class ContinuumGridType>
class RodContinuumFixedPointStep
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
      : complex_(complex),
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
      : complex_(complex),
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
    
    std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector>
    continuumNeumannToDirichletMap(const std::string& continuumName,
                               const std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector>& forceTorque,
                               const std::map<std::pair<std::string,std::string>,RigidBodyMotion<3> >& centerOfTorque) const;

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
    
public:    
    /** \todo Should be part of RodData, too */
    mutable std::map<std::string, RodConfigurationType> rodSubdomainSolutions_;
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
    
public:
    /** \todo Should be part of ContinuumData, too */
    mutable std::map<std::string, ContinuumConfigurationType> continuumSubdomainSolutions_;

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
        
        dirichletAndCouplingNodes.resize(complex_.rodGrid(name)->size(1));
        
        // first copy the true Dirichlet boundary
        const LeafBoundaryPatch<RodGridType>& dirichletBoundary = complex_.rods_.find(name)->second.dirichletBoundary_;

        for (int i=0; i<dirichletAndCouplingNodes.size(); i++)
            dirichletAndCouplingNodes[i] = dirichletBoundary.containsVertex(i);
        
        // get the names of all the continua that we couple with
        std::set<std::string> continuumNames = continuaPerRod(name);
        
        for (std::set<std::string>::const_iterator cIt = continuumNames.begin();
             cIt != continuumNames.end();
             ++cIt) {

            const LeafBoundaryPatch<RodGridType>& rodInterfaceBoundary 
                    = complex_.coupling(std::make_pair(name,*cIt)).rodInterfaceBoundary_;

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
        
        dirichletAndCouplingNodes.resize(complex_.continuumGrid(name)->size(dim));
        
        // first copy the true Dirichlet boundary
        const LeafBoundaryPatch<ContinuumGridType>& dirichletBoundary = complex_.continua_.find(name)->second.dirichletBoundary_;

        for (int i=0; i<dirichletAndCouplingNodes.size(); i++)
            dirichletAndCouplingNodes[i] = dirichletBoundary.containsVertex(i);
        
        // get the names of all the rods that we couple with
        std::set<std::string> rodNames = rodsPerContinuum(name);
        
        for (std::set<std::string>::const_iterator rIt = rodNames.begin();
             rIt != rodNames.end();
             ++rIt) {

            const LeafBoundaryPatch<ContinuumGridType>& continuumInterfaceBoundary 
                    = complex_.coupling(std::make_pair(*rIt,name)).continuumInterfaceBoundary_;

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
std::set<std::string> RodContinuumFixedPointStep<RodGridType,ContinuumGridType>::
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
std::set<std::string> RodContinuumFixedPointStep<RodGridType,ContinuumGridType>::
continuaPerRod(const std::string& name) const
{
    std::set<std::string> result;
    
    for (typename RodContinuumComplex<RodGridType,ContinuumGridType>::ConstCouplingIterator it = complex_.couplings_.begin(); 
         it!=complex_.couplings_.end(); ++it)
        if (it->first.first == name)
            result.insert(it->first.second);
    
    return result;
}

template <class RodGridType, class ContinuumGridType>
std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector> RodContinuumFixedPointStep<RodGridType,ContinuumGridType>::
rodDirichletToNeumannMap(const std::string& rodName, 
                               const std::map<std::pair<std::string,std::string>,RigidBodyMotion<3> >& lambda) const
{
    // container for the subdomain solution
    RodConfigurationType& rodX = rodSubdomainSolutions_[rodName];
    rodX.resize(complex_.rodGrid(rodName)->size(1));
    
    ///////////////////////////////////////////////////////////
    //  Set the complete set of Dirichlet values
    ///////////////////////////////////////////////////////////
    const LeafBoundaryPatch<RodGridType>& dirichletBoundary = complex_.rod(rodName).dirichletBoundary_;
    const RodConfigurationType& dirichletValues = complex_.rod(rodName).dirichletValues_;
    
    for (size_t i=0; i<rodX.size(); i++)
        if (dirichletBoundary.containsVertex(i))
            rodX[i] = dirichletValues[i];
    
    typename std::map<std::pair<std::string,std::string>,RigidBodyMotion<3> >::const_iterator it = lambda.begin();
    for (; it!=lambda.end(); ++it) {
        
        const std::pair<std::string,std::string>& couplingName = it->first;
        
        if (couplingName.first != rodName)
            continue;
    
        // Use \lambda as a Dirichlet value for the rod
        const LeafBoundaryPatch<RodGridType>& interfaceBoundary = complex_.coupling(couplingName).rodInterfaceBoundary_;

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
    RodFactory<typename RodGridType::LeafGridView> rodFactory(complex_.rodGrid(rodName)->leafView());
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
        
        const LeafBoundaryPatch<RodGridType>& couplingBoundary = complex_.coupling(couplingName).rodInterfaceBoundary_;

        result[couplingName] = rod(rodName).assembler_->getResultantForce(couplingBoundary, rodX);
    }
    
    return result;
}

template <class RodGridType, class ContinuumGridType>
std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector> 
RodContinuumFixedPointStep<RodGridType,ContinuumGridType>::
continuumNeumannToDirichletMap(const std::string& continuumName,
                               const std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector>& forceTorque,
                               const std::map<std::pair<std::string,std::string>,RigidBodyMotion<3> >& centerOfTorque) const
{
    ////////////////////////////////////////////////////
    //  Assemble the linearized problem
    ////////////////////////////////////////////////////

    /** \todo We are actually assembling standard linear elasticity,
    * i.e. the linearization at zero
    */
    typedef P1NodalBasis<typename ContinuumGridType::LeafGridView,double> P1Basis;
    const LeafBoundaryPatch<ContinuumGridType>& dirichletBoundary = complex_.continuum(continuumName).dirichletBoundary_;
    P1Basis basis(dirichletBoundary.gridView());
    OperatorAssembler<P1Basis,P1Basis> assembler(basis, basis);

    MatrixType stiffnessMatrix;
    assembler.assemble(*continuum(continuumName).localAssembler_, stiffnessMatrix);
    
    
    /////////////////////////////////////////////////////////////////////
    //  Turn the input force and torque into a Neumann boundary field
    /////////////////////////////////////////////////////////////////////
    
    // The weak form of the volume and Neumann data
    /** \brief Not implemented yet! */
    VectorType rhs(complex_.continuumGrid(continuumName)->size(dim));
    rhs = 0;

    typedef typename std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector>::const_iterator ForceIterator;
    
    for (ForceIterator it = forceTorque.begin(); it!=forceTorque.end(); ++it) {
        
        const std::pair<std::string,std::string>& couplingName = it->first;

        if (couplingName.second != continuumName)
            continue;
        
        // Use 'forceTorque' as a Neumann value for the rod
        const LeafBoundaryPatch<ContinuumGridType>& interfaceBoundary = complex_.coupling(couplingName).continuumInterfaceBoundary_;
        
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
    std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector> interfaceCorrection;

    for (ForceIterator it = forceTorque.begin(); it!=forceTorque.end(); ++it) {
        
        const std::pair<std::string,std::string>& couplingName = it->first;

        if (couplingName.second != continuumName)
            continue;
        
        // Use 'forceTorque' as a Neumann value for the rod
        const LeafBoundaryPatch<ContinuumGridType>& interfaceBoundary = complex_.coupling(couplingName).continuumInterfaceBoundary_;
        
        RigidBodyMotion<3> averageInterface;
        computeAverageInterface(interfaceBoundary, x, averageInterface);
        
        // Compute the linearization
        /** \todo We could loop directly over interfaceCorrection (and save the name lookup)
         * if we could be sure that interfaceCorrection contains all possible entries already
         */
        interfaceCorrection[couplingName][0] = averageInterface.r[0];
        interfaceCorrection[couplingName][1] = averageInterface.r[1];
        interfaceCorrection[couplingName][2] = averageInterface.r[2];
        
        Dune::FieldVector<double,3> infinitesimalRotation = Rotation<3,double>::expInv(averageInterface.q);
        interfaceCorrection[couplingName][3] = infinitesimalRotation[0];
        interfaceCorrection[couplingName][4] = infinitesimalRotation[1];
        interfaceCorrection[couplingName][5] = infinitesimalRotation[2];
        
    }
    
    return interfaceCorrection;
}


/** \brief One preconditioned Richardson step
*/
template <class RodGridType, class ContinuumGridType>
void RodContinuumFixedPointStep<RodGridType,ContinuumGridType>::
iterate(std::map<std::pair<std::string,std::string>, RigidBodyMotion<3> >& lambda)
{
    std::pair<std::string,std::string> interfaceName = std::make_pair("rod","continuum");
    
    ///////////////////////////////////////////////////////////////////
    //  Evaluate the Dirichlet-to-Neumann maps for the rods
    ///////////////////////////////////////////////////////////////////

    std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector> rodForceTorque;
    
    for (RodIterator it = rods_.begin(); it != rods_.end(); ++it) {
        
        const std::string& rodName = it->first;
    
        std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector> forceTorque = rodDirichletToNeumannMap(rodName, lambda);

        insert(rodForceTorque, forceTorque);
        
    }

    std::cout << "resultant rod forces and torques: "  << std::endl;
    typedef typename std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector>::iterator ForceIterator;
    for (ForceIterator it = rodForceTorque.begin(); it != rodForceTorque.end(); ++it)
        std::cout << "    [" << it->first.first << ", " << it->first.second << "] -- "
                  << it->second << std::endl;

    RodConfigurationType rodX = rods_["rod"].solver_->getSol();

    ///////////////////////////////////////////////////////////////
    //  Flip orientation of all rod forces, to account for opposing normals.
    ///////////////////////////////////////////////////////////////

    for (ForceIterator it = rodForceTorque.begin(); it != rodForceTorque.end(); ++it)
        it->second *= -1;

    
    typedef P1NodalBasis<typename ContinuumGridType::LeafGridView,double> P1Basis;
    const LeafBoundaryPatch<ContinuumGridType>& dirichletBoundary = complex_.continuum("continuum").dirichletBoundary_;
    P1Basis basis(dirichletBoundary.gridView());

    VectorType neumannValues(basis.size());
    VectorType rhs3d;
    
    // Using that index 0 is always the left boundary for a uniformly refined OneDGrid
    computeAveragePressure<typename ContinuumGridType::LeafGridView>(rodForceTorque.begin()->second, 
                                            complex_.coupling(interfaceName).continuumInterfaceBoundary_, 
                                            rodX[0].r,
                                            neumannValues);

    BoundaryFunctionalAssembler<P1Basis> boundaryFunctionalAssembler(basis, complex_.coupling(interfaceName).continuumInterfaceBoundary_);
    BasisGridFunction<P1Basis, VectorType> neumannValuesFunction(basis, neumannValues);
    NeumannBoundaryAssembler<ContinuumGridType, Dune::FieldVector<double,dim> > localNeumannAssembler(neumannValuesFunction);
    boundaryFunctionalAssembler.assemble(localNeumannAssembler, rhs3d, true);

    // ///////////////////////////////////////////////////////////
    //   Solve the Neumann problem for the continuum
    // ///////////////////////////////////////////////////////////
    VectorType& x3d = continuumSubdomainSolutions_["continuum"];

    assert( (dynamic_cast<LinearIterationStep<MatrixType,VectorType>* >(continuum("continuum").solver_->iterationStep_)) );
    dynamic_cast<LinearIterationStep<MatrixType,VectorType>* >(continuum("continuum").solver_->iterationStep_)->setProblem(*continuum("continuum").stiffnessMatrix_, x3d, rhs3d);
    //multigridStep.setProblem(*continua_["continuum"].stiffnessMatrix_, x3d, rhs3d, complex_.continua_["continuum"].grid_->maxLevel()+1);
        
    continua_["continuum"].solver_->preprocess();
        
    continua_["continuum"].solver_->solve();
        
    x3d = dynamic_cast<IterationStep<VectorType>* >(continuum("continuum").solver_->iterationStep_)->getSol();

    // ///////////////////////////////////////////////////////////
    //   Extract new interface position and orientation
    // ///////////////////////////////////////////////////////////

    RigidBodyMotion<3> averageInterface;

    computeAverageInterface(complex_.coupling(interfaceName).continuumInterfaceBoundary_, x3d, averageInterface);

    std::cout << "average interface: " << averageInterface << std::endl;

    std::cout << "director 0:  " << averageInterface.q.director(0) << std::endl;
    std::cout << "director 1:  " << averageInterface.q.director(1) << std::endl;
    std::cout << "director 2:  " << averageInterface.q.director(2) << std::endl;
    std::cout << std::endl;

    //////////////////////////////////////////////////////////////
    //   Compute new damped interface value
    //////////////////////////////////////////////////////////////
    
    const RigidBodyMotion<dim>& referenceInterface = complex_.coupling(interfaceName).referenceInterface_;

    for (int j=0; j<dim; j++)
        lambda[interfaceName].r[j] = (1-damping_) * lambda[interfaceName].r[j] 
                                   + damping_ * (referenceInterface.r[j] + averageInterface.r[j]);

        lambda[interfaceName].q = Rotation<3,double>::interpolate(lambda[interfaceName].q, 
                                                       referenceInterface.q.mult(averageInterface.q), 
                                                       damping_);

#if 0

            
    ///////////////////////////////////////////////////////////////
    //  Apply the preconditioner
    ///////////////////////////////////////////////////////////////
            
    std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector> continuumCorrection;
    std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector> rodCorrection;
            
    if (preconditioner_=="DirichletNeumann" or preconditioner_=="NeumannNeumann") {
                
        ////////////////////////////////////////////////////////////////////
        //  Preconditioner is the linearized Neumann-to-Dirichlet map
        //  of the continuum.
        ////////////////////////////////////////////////////////////////////
                
        for (ContinuumIterator it = continua_.begin(); it != continua_.end(); ++it) {
        
            const std::string& continuumName = it->first;
    
            std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector> continuumInterfaceCorrection
                    = linearizedContinuumNeumannToDirichletMap(continuumName,
                                                               continuumSubdomainSolutions_[continuumName], 
                                                               residualForceTorque,
                                                               lambda);

            insert(continuumCorrection, continuumInterfaceCorrection);
        
        }

        std::cout << "resultant continuum interface corrections: " << std::endl;
        for (ForceIterator it = continuumCorrection.begin(); it != continuumCorrection.end(); ++it)
        std::cout << "    [" << it->first.first << ", " << it->first.second << "] -- "
                  << it->second << std::endl;


    } 
    
#endif
}

#endif
