#ifndef ROD_CONTINUUM_STEKLOV_POINCARE_STEP_HH
#define ROD_CONTINUUM_STEKLOV_POINCARE_STEP_HH

#include <vector>

#include <dune/common/shared_ptr.hh>
#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/bitsetvector.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>

#include <dune/fufem/assemblers/boundaryfunctionalassembler.hh>
#include <dune/fufem/assemblers/localassemblers/neumannboundaryassembler.hh>

#include <dune/gfe/coupling/rodcontinuumcomplex.hh>


template <class RodGridType, class ContinuumGridType>
class RodContinuumSteklovPoincareStep
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
    
    /** \brief Constructor */
    RodContinuumSteklovPoincareStep(const RodContinuumComplex<RodGridType,ContinuumGridType>& complex,
                                    const std::string& preconditioner,
                                    const Dune::array<double,2>& alpha,
                                    double richardsonDamping,
                                    RodAssembler<typename RodGridType::LeafGridView,3>* rodAssembler,
                                    RodLocalStiffness<typename RodGridType::LeafGridView,double>* rodLocalStiffness,
                                    RiemannianTrustRegionSolver<RodGridType,RigidBodyMotion<3> >* rodSolver,
                                    const MatrixType* stiffnessMatrix3d,
                                    const Dune::shared_ptr< ::LoopSolver<VectorType> > solver,
                                    LinearLocalAssembler<ContinuumGridType, 
                                                         ContinuumLocalFiniteElement, 
                                                         ContinuumLocalFiniteElement,
                                                         Dune::FieldMatrix<double,dim,dim> >* localAssembler)
      : complex_(complex),
        preconditioner_(preconditioner),
        alpha_(alpha),
        richardsonDamping_(richardsonDamping)
    {
        rods_["rod"].assembler_      = rodAssembler;
        rods_["rod"].localStiffness_ = rodLocalStiffness;
        rods_["rod"].solver_         = rodSolver;

        continua_["continuum"].stiffnessMatrix_ = stiffnessMatrix3d;
        continua_["continuum"].solver_          = solver;
        continua_["continuum"].localAssembler_  = localAssembler;

        mergeRodDirichletAndCouplingBoundaries();
        mergeContinuumDirichletAndCouplingBoundaries();
    }
    
    void mergeRodDirichletAndCouplingBoundaries();

    void mergeContinuumDirichletAndCouplingBoundaries();

    
    /** \brief Do one Steklov-Poincare step
     * \param[in,out] lambda The old and new iterate
     */
    void iterate(std::map<std::pair<std::string,std::string>, RigidBodyMotion<3> >& lambda);
    
private:
    
    std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector> 
        rodDirichletToNeumannMap(const std::string& rodName, 
                                 const std::map<std::pair<std::string,std::string>,RigidBodyMotion<3> >& lambda) const;
    
    std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector> 
        continuumDirichletToNeumannMap(const std::string& continuumName, 
                                       const std::map<std::pair<std::string,std::string>,RigidBodyMotion<3> >& lambda) const;
                                       
    /** \brief Map a Neumann value to a Dirichlet value
     * 
     * \param currentIterate The rod configuration that we are linearizing about
     * 
     * \return The Dirichlet value
     */
    std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector>
                linearizedRodNeumannToDirichletMap(const RodConfigurationType& currentIterate,
                                                                   const RigidBodyMotion<3>::TangentVector& forceTorque,
                                                                   const Dune::FieldVector<double,3>& centerOfTorque) const;
    
    /** \brief Map a Neumann value to a Dirichlet value
     * 
     * \param currentIterate The continuum configuration that we are linearizing about
     * 
     * \return The infinitesimal movement of the interface
     */
    std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector>
                linearizedContinuumNeumannToDirichletMap(const VectorType& currentIterate,
                                                                         const RigidBodyMotion<3>::TangentVector& forceTorque,
                                                                         const Dune::FieldVector<double,3>& centerOfTorque) const;
                                       
    std::set<std::string> rodsPerContinuum(const std::string& name) const;
    
    std::set<std::string> continuaPerRod(const std::string& name) const;
    
    //////////////////////////////////////////////////////////////////
    //  Data members related to the coupled problem
    //////////////////////////////////////////////////////////////////
    const RodContinuumComplex<RodGridType,ContinuumGridType>& complex_;
    
    /** \brief Decides which preconditioner is used */
    std::string preconditioner_;
    
    /** \brief Neumann-Neumann damping */
    Dune::array<double,2> alpha_;

    /** \brief Damping factor for the Richardson iteration */
    double richardsonDamping_;
    
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
private:
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
private:
};


template <class RodGridType, class ContinuumGridType>
void RodContinuumSteklovPoincareStep<RodGridType,ContinuumGridType>::
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
void RodContinuumSteklovPoincareStep<RodGridType,ContinuumGridType>::
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
std::set<std::string> RodContinuumSteklovPoincareStep<RodGridType,ContinuumGridType>::
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
std::set<std::string> RodContinuumSteklovPoincareStep<RodGridType,ContinuumGridType>::
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
std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector> RodContinuumSteklovPoincareStep<RodGridType,ContinuumGridType>::
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
        
        const LeafBoundaryPatch<RodGridType>& couplingBoundary = complex_.coupling(couplingName).rodInterfaceBoundary_;

        result[couplingName] = rod(rodName).assembler_->getResultantForce(couplingBoundary, rodX);
    }
    
    return result;
}


template <class RodGridType, class ContinuumGridType>
std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector> RodContinuumSteklovPoincareStep<RodGridType,ContinuumGridType>::
continuumDirichletToNeumannMap(const std::string& continuumName, 
                               const std::map<std::pair<std::string,std::string>,RigidBodyMotion<3> >& lambda) const
{
    // Set initial iterate
    VectorType& x3d = continuumSubdomainSolutions_[continuumName];
    x3d.resize(complex_.continuumGrid(continuumName)->size(dim));
    x3d = 0;

    // Copy the true Dirichlet values into it
    const LeafBoundaryPatch<ContinuumGridType>& dirichletBoundary = complex_.continuum(continuumName).dirichletBoundary_;
    const VectorType& dirichletValues = complex_.continuum(continuumName).dirichletValues_;
    
    for (size_t i=0; i<x3d.size(); i++)
        if (dirichletBoundary.containsVertex(i))
            x3d[i] = dirichletValues[i];
    
    typename std::map<std::pair<std::string,std::string>,RigidBodyMotion<3> >::const_iterator it = lambda.begin();
    for (; it!=lambda.end(); ++it) {
        
        const std::pair<std::string,std::string>& couplingName = it->first;
    
        // Turn \lambda \in TSE(3) into a Dirichlet value for the continuum
        const LeafBoundaryPatch<ContinuumGridType>& interfaceBoundary = complex_.coupling(couplingName).continuumInterfaceBoundary_;
        const RigidBodyMotion<dim>& referenceInterface = complex_.coupling(couplingName).referenceInterface_;
        
        setRotation(interfaceBoundary, x3d, referenceInterface, it->second);
    
    }
    
    // Set the correct Dirichlet nodes
    dynamic_cast<IterationStep<VectorType>* >(continuum(continuumName).solver_->iterationStep_)->ignoreNodes_ 
           = &continuum(continuumName).dirichletAndCouplingNodes_;
    
    // Right hand side vector: currently without Neumann and volume terms
    VectorType rhs3d(x3d.size());
    rhs3d = 0;
    
    // Solve the Dirichlet problem
    assert( (dynamic_cast<LinearIterationStep<MatrixType,VectorType>* >(continuum(continuumName).solver_->iterationStep_)) );
    dynamic_cast<LinearIterationStep<MatrixType,VectorType>* >(continuum(continuumName).solver_->iterationStep_)->setProblem(*continuum(continuumName).stiffnessMatrix_, x3d, rhs3d);

    continuum(continuumName).solver_->preprocess();
    dynamic_cast<IterationStep<VectorType>* >(continuum(continuumName).solver_->iterationStep_)->preprocess();
        
    continuum(continuumName).solver_->solve();
        
    x3d = dynamic_cast<IterationStep<VectorType>* >(continuum(continuumName).solver_->iterationStep_)->getSol();
            
    // Integrate over the residual on the coupling boundary to obtain
    // an element of T^*SE.
    Dune::FieldVector<double,3> continuumForce, continuumTorque;
            
    VectorType residual = rhs3d;
    continuum(continuumName).stiffnessMatrix_->mmv(x3d,residual);
    
    //////////////////////////////////////////////////////////////////////////////
    //  Extract the residual stresses
    //////////////////////////////////////////////////////////////////////////////
            
    std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector > result;

    for (it = lambda.begin(); it!=lambda.end(); ++it) {
        
        const std::pair<std::string,std::string>& couplingName = it->first;
        
        /** \todo Is referenceInterface.r the correct center of rotation? */
        const RigidBodyMotion<dim>& referenceInterface = complex_.coupling(couplingName).referenceInterface_;

        computeTotalForceAndTorque(complex_.coupling(couplingName).continuumInterfaceBoundary_, 
                                   residual, 
                                   referenceInterface.r,
                                   continuumForce, continuumTorque);

        result[couplingName][0] = continuumForce[0];
        result[couplingName][1] = continuumForce[1];
        result[couplingName][2] = continuumForce[2];
        result[couplingName][3] = continuumTorque[0];
        result[couplingName][4] = continuumTorque[1];
        result[couplingName][5] = continuumTorque[2];
    
    }
    
    return result;
}

/** \brief Map a Neumann value to a Dirichlet value
*  
* \param currentIterate The rod configuration that we are linearizing about
* 
* \return The Dirichlet value
*/
template <class RodGridType, class ContinuumGridType>
std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector> RodContinuumSteklovPoincareStep<RodGridType,ContinuumGridType>::
linearizedRodNeumannToDirichletMap(const RodConfigurationType& currentIterate,
                                   const RigidBodyMotion<3>::TangentVector& forceTorque,
                                   const Dune::FieldVector<double,3>& centerOfTorque) const
{
    std::pair<std::string,std::string> interfaceName = std::make_pair("rod","continuum");
        
    ////////////////////////////////////////////////////
    //  Assemble the linearized rod problem
    ////////////////////////////////////////////////////

    const LeafBoundaryPatch<RodGridType>& interface = complex_.coupling(interfaceName).rodInterfaceBoundary_;

    typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,6,6> > MatrixType;
    GeodesicFEAssembler<typename RodGridType::LeafGridView, RigidBodyMotion<3> > assembler(interface.gridView(),
                                                                    rod("rod").localStiffness_);
    MatrixType stiffnessMatrix;
    assembler.assembleMatrix(currentIterate, 
                             stiffnessMatrix,
                             true   // assemble occupation pattern
                            );
    
    RodCorrectionType rhs(currentIterate.size());
    rhs = 0;
    assembler.assembleGradient(currentIterate, rhs);
        
    // The right hand side is the _negative_ gradient
    rhs *= -1;

    /////////////////////////////////////////////////////////////////////
    //  Turn the input force and torque into a Neumann boundary field
    /////////////////////////////////////////////////////////////////////

    // The weak form of the Neumann data
    rhs[0] += forceTorque;
        
    ///////////////////////////////////////////////////////////
    // Modify matrix and rhs to contain one Dirichlet node
    ///////////////////////////////////////////////////////////

    int dIdx = rhs.size()-1;   // hardwired index of the single Dirichlet node
    rhs[dIdx] = 0;
    stiffnessMatrix[dIdx] = 0;
    stiffnessMatrix[dIdx][dIdx] = Dune::ScaledIdentityMatrix<double,6>(1);
        
    //////////////////////////////////////////////////
    //   Solve the Neumann problem
    //////////////////////////////////////////////////

    RodCorrectionType x(rhs.size());
    x = 0;  // initial iterate

    // Technicality:  turn the matrix into a linear operator
    Dune::MatrixAdapter<MatrixType,RodCorrectionType,RodCorrectionType> op(stiffnessMatrix);

    // A preconditioner
    Dune::SeqILU0<MatrixType,RodCorrectionType,RodCorrectionType> ilu0(stiffnessMatrix,1.0);

    // A preconditioned conjugate-gradient solver
    Dune::CGSolver<RodCorrectionType> cg(op,ilu0,1E-4,100,2);

    // Object storing some statistics about the solving process
    Dune::InverseOperatorResult statistics;

    // Solve!
    cg.apply(x, rhs, statistics);

    std::cout << "Linear rod interface correction: " << x[0] << std::endl;
        
    std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector> interfaceCorrection;
    interfaceCorrection[interfaceName] = x[0];
    
    return interfaceCorrection;
}


template <class RodGridType, class ContinuumGridType>
std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector> RodContinuumSteklovPoincareStep<RodGridType,ContinuumGridType>::
linearizedContinuumNeumannToDirichletMap(const VectorType& currentIterate,
                                         const RigidBodyMotion<3>::TangentVector& forceTorque,
                                         const Dune::FieldVector<double,3>& centerOfTorque) const
{
    std::pair<std::string,std::string> interfaceName = std::make_pair("rod","continuum");
        
    ////////////////////////////////////////////////////
    //  Assemble the linearized problem
    ////////////////////////////////////////////////////

    /** \todo We are actually assembling standard linear elasticity,
    * i.e. the linearization at zero
    */
    typedef P1NodalBasis<typename ContinuumGridType::LeafGridView,double> P1Basis;
    const LeafBoundaryPatch<ContinuumGridType>& interface = complex_.coupling(interfaceName).continuumInterfaceBoundary_;
    P1Basis basis(interface.gridView());
    OperatorAssembler<P1Basis,P1Basis> assembler(basis, basis);

    MatrixType stiffnessMatrix;
    assembler.assemble(*continuum("continuum").localAssembler_, stiffnessMatrix);
    
    
    /////////////////////////////////////////////////////////////////////
    //  Turn the input force and torque into a Neumann boundary field
    /////////////////////////////////////////////////////////////////////
    VectorType neumannValues(stiffnessMatrix.N());
    neumannValues = 0;

    // 
    computeAveragePressure<typename ContinuumGridType::LeafGridView>(forceTorque, 
                                     interface, 
                                     centerOfTorque,
                                     neumannValues);

    // The weak form of the volume and Neumann data
    /** \brief Not implemented yet! */
    VectorType rhs(complex_.continuumGrid("continuum")->size(dim));
    rhs = 0;

    BoundaryFunctionalAssembler<P1Basis> boundaryFunctionalAssembler(basis, interface);
    BasisGridFunction<P1Basis,VectorType> neumannValuesFunction(basis,neumannValues);
    NeumannBoundaryAssembler<ContinuumGridType, Dune::FieldVector<double,3> > localNeumannAssembler(neumannValuesFunction);
    boundaryFunctionalAssembler.assemble(localNeumannAssembler, rhs, false);

    //   Solve the Neumann problem for the continuum
    VectorType x = complex_.continuum("continuum").dirichletValues_;
    assert( (dynamic_cast<LinearIterationStep<MatrixType,VectorType>* >(continuum("continuum").solver_->iterationStep_)) );
    dynamic_cast<LinearIterationStep<MatrixType,VectorType>* >(continuum("continuum").solver_->iterationStep_)->setProblem(stiffnessMatrix, x, rhs);

    //solver.preprocess();
    continuum("continuum").solver_->iterationStep_->preprocess();
        
    continuum("continuum").solver_->solve();
        
    x = continuum("continuum").solver_->iterationStep_->getSol();
        
    //  Average the continuum displacement on the coupling boundary
    RigidBodyMotion<3> averageInterface;
    computeAverageInterface(interface, x, averageInterface);
        
    std::cout << "Average interface: " << averageInterface << std::endl;
        
    // Compute the linearization
    std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector> interfaceCorrection;
    interfaceCorrection[interfaceName][0] = averageInterface.r[0];
    interfaceCorrection[interfaceName][1] = averageInterface.r[1];
    interfaceCorrection[interfaceName][2] = averageInterface.r[2];
        
    Dune::FieldVector<double,3> infinitesimalRotation = Rotation<3,double>::expInv(averageInterface.q);
    interfaceCorrection[interfaceName][3] = infinitesimalRotation[0];
    interfaceCorrection[interfaceName][4] = infinitesimalRotation[1];
    interfaceCorrection[interfaceName][5] = infinitesimalRotation[2];
        
    return interfaceCorrection;
}


/** \brief One preconditioned Richardson step
*/
template <class RodGridType, class ContinuumGridType>
void RodContinuumSteklovPoincareStep<RodGridType,ContinuumGridType>::
iterate(std::map<std::pair<std::string,std::string>, RigidBodyMotion<3> >& lambda)
{
    ///////////////////////////////////////////////////////////////////
    //  Evaluate the Dirichlet-to-Neumann maps for the rods
    ///////////////////////////////////////////////////////////////////

    std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector> rodForceTorque;
    
    for (RodIterator it = rods_.begin(); it != rods_.end(); ++it) {
        
        const std::string& rodName = it->first;
    
        std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector> forceTorque = rodDirichletToNeumannMap(rodName, lambda);

        int oldSize = rodForceTorque.size();  // for debugging
        rodForceTorque.insert(forceTorque.begin(), forceTorque.end());
        assert(rodForceTorque.size() == oldSize + forceTorque.size());
        
    }

    std::cout << "resultant rod forces and torques: "  << std::endl;
    typedef typename std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector>::iterator ForceIterator;
    for (ForceIterator it = rodForceTorque.begin(); it != rodForceTorque.end(); ++it)
        std::cout << "    [" << it->first.first << ", " << it->first.second << "] -- "
                  << it->second << std::endl;

    ///////////////////////////////////////////////////////////////////
    //  Evaluate the Dirichlet-to-Neumann map for the continuum
    ///////////////////////////////////////////////////////////////////
    
    std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector> continuumForceTorque; 

    for (ContinuumIterator it = continua_.begin(); it != continua_.end(); ++it) {
        
        const std::string& continuumName = it->first;
    
        std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector> forceTorque 
                = continuumDirichletToNeumannMap(continuumName, lambda);

        int oldSize = continuumForceTorque.size();  // for debugging
        continuumForceTorque.insert(forceTorque.begin(), forceTorque.end());
        assert(continuumForceTorque.size() == oldSize + forceTorque.size());
        
    }

    std::cout << "resultant continuum force and torque: " << std::endl;
    for (ForceIterator it = continuumForceTorque.begin(); it != continuumForceTorque.end(); ++it)
        std::cout << "    [" << it->first.first << ", " << it->first.second << "] -- "
                  << it->second << std::endl;

    ///////////////////////////////////////////////////////////////
    //  Compute the overall Steklov-Poincare residual
    ///////////////////////////////////////////////////////////////

    // Flip orientation of all rod forces, to account for opposing normals.
    for (ForceIterator it = rodForceTorque.begin(); it != rodForceTorque.end(); ++it)
        it->second *= -1;

    std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector> residualForceTorque = rodForceTorque;
    
    for (ForceIterator it = residualForceTorque.begin(), it2 = continuumForceTorque.begin(); 
         it != residualForceTorque.end(); 
         ++it, ++it2) {
        assert(it->first == it2->first);
        it->second += it2->second;
    }
            
    ///////////////////////////////////////////////////////////////
    //  Apply the preconditioner
    ///////////////////////////////////////////////////////////////
            
    // temporary
    std::pair<std::string,std::string> interfaceName = std::make_pair("rod","continuum");
    
    std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector> interfaceCorrection;
            
    if (preconditioner_=="DirichletNeumann") {
                
        ////////////////////////////////////////////////////////////////////
        //  Preconditioner is the linearized Neumann-to-Dirichlet map
        //  of the continuum.
        ////////////////////////////////////////////////////////////////////
                
        interfaceCorrection = linearizedContinuumNeumannToDirichletMap(continuumSubdomainSolutions_["continuum"], 
                                                  residualForceTorque[interfaceName], 
                                                  lambda[interfaceName].r);
                
    } else if (preconditioner_=="NeumannDirichlet") {
            
        ////////////////////////////////////////////////////////////////////
        //  Preconditioner is the linearized Neumann-to-Dirichlet map
        //  of the rod.
        ////////////////////////////////////////////////////////////////////
        
        interfaceCorrection = linearizedRodNeumannToDirichletMap(rodSubdomainSolutions_["rod"], 
                                                 residualForceTorque[interfaceName], 
                                                 lambda[interfaceName].r);


    } else if (preconditioner_=="NeumannNeumann") {
            
        ////////////////////////////////////////////////////////////////////
        //  Preconditioner is a convex combination of the linearized
        //  Neumann-to-Dirichlet map of the continuum and the linearized
        //  Neumann-to-Dirichlet map of the rod.
        ////////////////////////////////////////////////////////////////////

        std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector> continuumCorrection = linearizedContinuumNeumannToDirichletMap(continuumSubdomainSolutions_["continuum"], 
                                                                              residualForceTorque[interfaceName], 
                                                                              lambda[interfaceName].r);
        std::map<std::pair<std::string,std::string>,RigidBodyMotion<3>::TangentVector> rodCorrection       = linearizedRodNeumannToDirichletMap(rodSubdomainSolutions_["rod"],
                                                                             residualForceTorque[interfaceName],
                                                                             lambda[interfaceName].r);
                
        for (int j=0; j<6; j++)
            interfaceCorrection[interfaceName][j] = (alpha_[0] * continuumCorrection[interfaceName][j] + alpha_[1] * rodCorrection[interfaceName][j])
                                        / alpha_[0] + alpha_[1];
                
    } else if (preconditioner_=="RobinRobin") {
            
        DUNE_THROW(Dune::NotImplemented, "Robin-Robin preconditioner not implemented yet");
                
    } else
        DUNE_THROW(Dune::NotImplemented, preconditioner_ << " is not a known preconditioner");
            
    ///////////////////////////////////////////////////////////////////////////////
    //  Apply the damped correction to the current interface value
    ///////////////////////////////////////////////////////////////////////////////
                
    interfaceCorrection[interfaceName] *= richardsonDamping_;
    typename std::map<std::pair<std::string,std::string>,RigidBodyMotion<3> >::iterator it = lambda.begin();
    for (; it!=lambda.end(); ++it) {
        it->second = RigidBodyMotion<3>::exp(it->second, interfaceCorrection[interfaceName]);
    }
}

#endif