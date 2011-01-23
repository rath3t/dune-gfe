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

template <class GridView, class MatrixType, class VectorType>
class LinearizedContinuumNeumannToDirichletMap
{
public:
    
    /** \brief Constructor 
     * 
     */
    LinearizedContinuumNeumannToDirichletMap(const BoundaryPatchBase<GridView>& interface,
                                             const VectorType& weakVolumeAndNeumannTerm,
                                             const VectorType& dirichletValues,
                                             const LinearLocalAssembler<typename GridView::Grid,
                                                                  typename P1NodalBasis<GridView,double>::LocalFiniteElement,
                                                                  typename P1NodalBasis<GridView,double>::LocalFiniteElement,
                                                                  Dune::FieldMatrix<double,3,3> >* localAssembler,
                                             const Dune::shared_ptr< ::LoopSolver<VectorType> >& solver
                                            )
    : interface_(interface),
      weakVolumeAndNeumannTerm_(weakVolumeAndNeumannTerm),
      dirichletValues_(dirichletValues),
      solver_(solver),
      localAssembler_(localAssembler)
    {}
    
    /** \brief Map a Neumann value to a Dirichlet value
     * 
     * \param currentIterate The continuum configuration that we are linearizing about
     * 
     * \return The infinitesimal movement of the interface
     */
    Dune::FieldVector<double,6> apply(const VectorType& currentIterate,
                                      const RigidBodyMotion<3>::TangentVector& forceTorque,
                             const Dune::FieldVector<double,3>& centerOfTorque
                            )
    {
        Dune::FieldVector<double,3> force, torque;
        for (int i=0; i<3; i++) {
            force[i]  = forceTorque[i];
            torque[i] = forceTorque[i+3];
        }
        
        ////////////////////////////////////////////////////
        //  Assemble the linearized problem
        ////////////////////////////////////////////////////

        /** \todo We are actually assembling standard linear elasticity,
         * i.e. the linearization at zero
         */
        typedef P1NodalBasis<GridView,double> P1Basis;
        P1Basis basis(interface_.gridView());
        OperatorAssembler<P1Basis,P1Basis> assembler(basis, basis);

        MatrixType stiffnessMatrix;
        assembler.assemble(*localAssembler_, stiffnessMatrix);
    
    
        /////////////////////////////////////////////////////////////////////
        //  Turn the input force and torque into a Neumann boundary field
        /////////////////////////////////////////////////////////////////////
        VectorType neumannValues(stiffnessMatrix.N());
        neumannValues = 0;

        // 
        computeAveragePressure<GridView>(force, torque, 
                                         interface_, 
                                         centerOfTorque,
                                         neumannValues);

        // The weak form of the Neumann data
        VectorType rhs = weakVolumeAndNeumannTerm_;

        BoundaryFunctionalAssembler<P1Basis> boundaryFunctionalAssembler(basis, interface_);
        BasisGridFunction<P1Basis,VectorType> neumannValuesFunction(basis,neumannValues);
        NeumannBoundaryAssembler<typename GridView::Grid, Dune::FieldVector<double,3> > localNeumannAssembler(neumannValuesFunction);
        boundaryFunctionalAssembler.assemble(localNeumannAssembler, rhs, false);

        //   Solve the Neumann problem for the continuum
        VectorType x = dirichletValues_;
        assert( (dynamic_cast<LinearIterationStep<MatrixType,VectorType>* >(solver_->iterationStep_)) );
        dynamic_cast<LinearIterationStep<MatrixType,VectorType>* >(solver_->iterationStep_)->setProblem(stiffnessMatrix, x, rhs);
        
        //solver.preprocess();
        solver_->iterationStep_->preprocess();
        
        solver_->solve();
        
        x = solver_->iterationStep_->getSol();
        
        std::cout << "x:\n" << x << std::endl;

        //  Average the continuum displacement on the coupling boundary
        RigidBodyMotion<3> averageInterface;
        computeAverageInterface(interface_, x, averageInterface);
        
        std::cout << "Average interface: " << averageInterface << std::endl;
        
        // Compute the linearization
        Dune::FieldVector<double,6> interfaceCorrection;
        interfaceCorrection[0] = averageInterface.r[0];
        interfaceCorrection[1] = averageInterface.r[1];
        interfaceCorrection[2] = averageInterface.r[2];
        
        Dune::FieldVector<double,3> infinitesimalRotation = Rotation<3,double>::expInv(averageInterface.q);
        interfaceCorrection[3] = infinitesimalRotation[0];
        interfaceCorrection[4] = infinitesimalRotation[1];
        interfaceCorrection[5] = infinitesimalRotation[2];
        
        return interfaceCorrection;
     }
     
private:
    
    const VectorType& weakVolumeAndNeumannTerm_;
    
    const VectorType& dirichletValues_;
    
    const Dune::shared_ptr< ::LoopSolver<VectorType> > solver_;
    
    const BoundaryPatchBase<GridView>& interface_;
    
    const LinearLocalAssembler<typename GridView::Grid,
                         typename P1NodalBasis<GridView,double>::LocalFiniteElement,
                         typename P1NodalBasis<GridView,double>::LocalFiniteElement,
                         Dune::FieldMatrix<double,3,3> >* localAssembler_;
};


template <class GridView, class VectorType>
class LinearizedRodNeumannToDirichletMap
{
    static const int dim = 3;
    
    typedef std::vector<RigidBodyMotion<dim> > RodSolutionType;

public:
    
    /** \brief Constructor 
     * 
     */
    LinearizedRodNeumannToDirichletMap(const BoundaryPatchBase<GridView>& interface,
                                       LocalGeodesicFEStiffness<GridView, RigidBodyMotion<3> >* localAssembler)
    : interface_(interface),
      localAssembler_(localAssembler)
    {}
    
    /** \brief Map a Neumann value to a Dirichlet value
     * 
     * \param currentIterate The rod configuration that we are linearizing about
     * 
     * \return The Dirichlet value
     */
    Dune::FieldVector<double,6> apply(const RodSolutionType& currentIterate,
                             const RigidBodyMotion<3>::TangentVector& forceTorque,
                             const Dune::FieldVector<double,3>& centerOfTorque
                            )
    {
        Dune::FieldVector<double,3> force, torque;
        for (int i=0; i<3; i++) {
            force[i]  = forceTorque[i];
            torque[i] = forceTorque[i+3];
        }
        
        ////////////////////////////////////////////////////
        //  Assemble the linearized rod problem
        ////////////////////////////////////////////////////

        typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,6,6> > MatrixType;
        GeodesicFEAssembler<GridView, RigidBodyMotion<3> > assembler(interface_.gridView(),
                                                                     localAssembler_);
        MatrixType stiffnessMatrix;
        assembler.assembleMatrix(currentIterate, 
                                 stiffnessMatrix,
                                 true   // assemble occupation pattern
                                );
    
        VectorType rhs(currentIterate.size());
        rhs = 0;
        assembler.assembleGradient(currentIterate, rhs);
        
        // The right hand side is the _negative_ gradient
        rhs *= -1;

        /////////////////////////////////////////////////////////////////////
        //  Turn the input force and torque into a Neumann boundary field
        /////////////////////////////////////////////////////////////////////

        // The weak form of the Neumann data
        rhs[0][0] += force[0];
        rhs[0][1] += force[1];
        rhs[0][2] += force[2];
        rhs[0][3] += torque[0];
        rhs[0][4] += torque[1];
        rhs[0][5] += torque[2];
        
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

        VectorType x(rhs.size());
        x = 0;  // initial iterate

        // Technicality:  turn the matrix into a linear operator
        Dune::MatrixAdapter<MatrixType,VectorType,VectorType> op(stiffnessMatrix);

        // A preconditioner
        Dune::SeqILU0<MatrixType,VectorType,VectorType> ilu0(stiffnessMatrix,1.0);

        // A preconditioned conjugate-gradient solver
        Dune::CGSolver<VectorType> cg(op,ilu0,1E-4,100,2);

        // Object storing some statistics about the solving process
        Dune::InverseOperatorResult statistics;

        // Solve!
        cg.apply(x, rhs, statistics);

        std::cout << "Linear rod interface correction: " << x[0] << std::endl;
        
        return x[0];
     }
     
private:
    
    const BoundaryPatchBase<GridView>& interface_;
    
    LocalGeodesicFEStiffness<GridView, RigidBodyMotion<3> >* localAssembler_;
};


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

        /** \todo Hack: this should be a tangent vector right away */
        Dune::FieldVector<double,dim> rodForce, rodTorque;
        rodForce = rod(rodName).assembler_->getResultantForce(couplingBoundary, rodX, rodTorque);
    
        dune_static_assert(RigidBodyMotion<3>::TangentVector::size == 2*dim, "TangentVector does not have appropriate size");
        result[couplingName][0] = rodForce[0];
        result[couplingName][1] = rodForce[1];
        result[couplingName][2] = rodForce[2];
        result[couplingName][3] = rodTorque[0];
        result[couplingName][4] = rodTorque[1];
        result[couplingName][5] = rodTorque[2];
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

/** \brief One preconditioned Richardson step
*/
template <class RodGridType, class ContinuumGridType>
void RodContinuumSteklovPoincareStep<RodGridType,ContinuumGridType>::
iterate(std::map<std::pair<std::string,std::string>, RigidBodyMotion<3> >& lambda)
{
    // temporary
    std::pair<std::string,std::string> interfaceName = std::make_pair("rod","continuum");
    
    ///////////////////////////////////////////////////////////////////
    //  Evaluate the Dirichlet-to-Neumann map for the rod
    ///////////////////////////////////////////////////////////////////

    std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector> rodForceTorque 
            = rodDirichletToNeumannMap("rod", lambda);

    std::cout << "resultant rod force and torque: "  << rodForceTorque[interfaceName] << std::endl;

    ///////////////////////////////////////////////////////////////////
    //  Evaluate the Dirichlet-to-Neumann map for the continuum
    ///////////////////////////////////////////////////////////////////
    
    std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector> continuumForceTorque 
            = continuumDirichletToNeumannMap("continuum", lambda);

    std::cout << "resultant continuum force and torque: "  << continuumForceTorque[interfaceName]  << std::endl;

    ///////////////////////////////////////////////////////////////
    //  Compute the overall Steklov-Poincare residual
    ///////////////////////////////////////////////////////////////

    // Flip orientation to account for opposing normals
    rodForceTorque[interfaceName] *= -1;

    std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector> residualForceTorque;
    residualForceTorque[interfaceName] = rodForceTorque[interfaceName] + continuumForceTorque[interfaceName];
            
            
    ///////////////////////////////////////////////////////////////
    //  Apply the preconditioner
    ///////////////////////////////////////////////////////////////
            
    std::map<std::pair<std::string,std::string>, RigidBodyMotion<3>::TangentVector> interfaceCorrection;
            
    if (preconditioner_=="DirichletNeumann") {
                
        ////////////////////////////////////////////////////////////////////
        //  Preconditioner is the linearized Neumann-to-Dirichlet map
        //  of the continuum.
        ////////////////////////////////////////////////////////////////////
                
        // Right hand side vector: currently without Neumann and volume terms
        VectorType rhs3d(complex_.continuumGrid("continuum")->size(dim));
        rhs3d = 0;

        LinearizedContinuumNeumannToDirichletMap<typename ContinuumGridType::LeafGridView,MatrixType,VectorType>
                linContNtDMap(complex_.coupling(interfaceName).continuumInterfaceBoundary_,
                              rhs3d,
                              complex_.continuum("continuum").dirichletValues_,
                              continuum("continuum").localAssembler_,
                              continuum("continuum").solver_);
                        
        interfaceCorrection[interfaceName] = linContNtDMap.apply(continuumSubdomainSolutions_["continuum"], 
                                                  residualForceTorque[interfaceName], 
                                                  lambda[interfaceName].r);
                
    } else if (preconditioner_=="NeumannDirichlet") {
            
        ////////////////////////////////////////////////////////////////////
        //  Preconditioner is the linearized Neumann-to-Dirichlet map
        //  of the rod.
        ////////////////////////////////////////////////////////////////////
        
        LinearizedRodNeumannToDirichletMap<typename RodGridType::LeafGridView,RodCorrectionType> linRodNtDMap(complex_.coupling(interfaceName).rodInterfaceBoundary_,
                                                                                                         rods_["rod"].localStiffness_);

        interfaceCorrection[interfaceName] = linRodNtDMap.apply(rodSubdomainSolutions_["rod"], 
                                                 residualForceTorque[interfaceName], 
                                                 lambda[interfaceName].r);


    } else if (preconditioner_=="NeumannNeumann") {
            
        ////////////////////////////////////////////////////////////////////
        //  Preconditioner is a convex combination of the linearized
        //  Neumann-to-Dirichlet map of the continuum and the linearized
        //  Neumann-to-Dirichlet map of the rod.
        ////////////////////////////////////////////////////////////////////

        // Right hand side vector: currently without Neumann and volume terms
        VectorType rhs3d(complex_.continuumGrid("continuum")->size(dim));
        rhs3d = 0;

        LinearizedContinuumNeumannToDirichletMap<typename ContinuumGridType::LeafGridView,MatrixType,VectorType>
                linContNtDMap(complex_.coupling(interfaceName).continuumInterfaceBoundary_,
                              rhs3d,
                              complex_.continuum("continuum").dirichletValues_,
                              continuum("continuum").localAssembler_,
                              continuum("continuum").solver_);

        LinearizedRodNeumannToDirichletMap<typename RodGridType::LeafGridView,RodCorrectionType> linRodNtDMap(complex_.coupling(interfaceName).rodInterfaceBoundary_,
                                                                                                              rods_["rod"].localStiffness_);

        Dune::FieldVector<double,6> continuumCorrection = linContNtDMap.apply(continuumSubdomainSolutions_["continuum"], 
                                                                              residualForceTorque[interfaceName], 
                                                                              lambda[interfaceName].r);
        Dune::FieldVector<double,6> rodCorrection       = linRodNtDMap.apply(rodSubdomainSolutions_["rod"],
                                                                             residualForceTorque[interfaceName],
                                                                             lambda[interfaceName].r);
                
        for (int j=0; j<6; j++)
            interfaceCorrection[interfaceName][j] = (alpha_[0] * continuumCorrection[j] + alpha_[1] * rodCorrection[j])
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