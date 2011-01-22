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
                
    typedef P1NodalBasis<typename ContinuumGridType::LeafGridView,double> ContinuumFEBasis;
    
public:
    
    /** \brief Constructor */
    RodContinuumSteklovPoincareStep(const RodContinuumComplex<RodGridType,ContinuumGridType>& complex,
                                    const std::string& preconditioner,
                                    const Dune::array<double,2>& alpha,
                                    double richardsonDamping,
                                    const RigidBodyMotion<3>& referenceInterface,
                                    RodAssembler<typename RodGridType::LeafGridView,3>* rodAssembler,
                                    RodLocalStiffness<typename RodGridType::LeafGridView,double>* rodLocalStiffness,
                                    RiemannianTrustRegionSolver<RodGridType,RigidBodyMotion<3> >* rodSolver,
                                    const MatrixType* stiffnessMatrix3d,
                                    const VectorType* dirichletValues,
                                    const Dune::shared_ptr< ::LoopSolver<VectorType> > solver,
                                    StVenantKirchhoffAssembler<ContinuumGridType, 
                                                               typename ContinuumFEBasis::LocalFiniteElement, 
                                                               typename ContinuumFEBasis::LocalFiniteElement>* localAssembler)
      : complex_(complex),
        preconditioner_(preconditioner),
        alpha_(alpha),
        richardsonDamping_(richardsonDamping),
        referenceInterface_(referenceInterface),
        rodAssembler_(rodAssembler),
        rodLocalStiffness_(rodLocalStiffness),
        rodSolver_(rodSolver),
        stiffnessMatrix3d_(stiffnessMatrix3d),
        dirichletValues_(dirichletValues),
        solver_(solver),
        localAssembler_(localAssembler)
    {
        dirichletAndCouplingNodes_.resize(complex.continuumGrid("continuum")->size(dim));

        const LeafBoundaryPatch<ContinuumGridType>& dirichletBoundary = complex.continuumDirichletBoundary("continuum");
        
        for (int i=0; i<dirichletAndCouplingNodes_.size(); i++)
            dirichletAndCouplingNodes_[i] = dirichletBoundary.containsVertex(i);

        const LeafBoundaryPatch<ContinuumGridType>& continuumInterfaceBoundary = complex.coupling(std::make_pair("rod","continuum")).continuumInterfaceBoundary_;

        for (int i=0; i<dirichletAndCouplingNodes_.size(); i++) {
            bool v = continuumInterfaceBoundary.containsVertex(i);
            for (int j=0; j<dim; j++)
                dirichletAndCouplingNodes_[i][j] = dirichletAndCouplingNodes_[i][j] or v;
        }
    }
    
    
    
    
    
    
    
        
        
    /** \brief Do one Steklov-Poincare step
     * \param[in,out] lambda The old and new iterate
     */
    void iterate(RigidBodyMotion<3>& lambda);
    
private:
    
    RigidBodyMotion<3>::TangentVector rodDirichletToNeumannMap(const RigidBodyMotion<3>& lambda) const;
    
    RigidBodyMotion<3>::TangentVector continuumDirichletToNeumannMap(const RigidBodyMotion<3>& lambda) const;
    
    //////////////////////////////////////////////////////////////////
    //  Data members related to the coupled problem
    //////////////////////////////////////////////////////////////////
    const RodContinuumComplex<RodGridType,ContinuumGridType>& complex_;
    
    std::string preconditioner_;
    
    /** \brief Neumann-Neumann damping */
    Dune::array<double,2> alpha_;

    double richardsonDamping_;
    
    //////////////////////////////////////////////////////////////////
    //  Data members related to the rod problems
    //////////////////////////////////////////////////////////////////
    RigidBodyMotion<dim> referenceInterface_;
    
    RodAssembler<typename RodGridType::LeafGridView,3>* rodAssembler_;
    
    RodLocalStiffness<typename RodGridType::LeafGridView,double>* rodLocalStiffness_;
    
    RiemannianTrustRegionSolver<RodGridType,RigidBodyMotion<3> >* rodSolver_;
public:    
    mutable std::map<std::string, RodConfigurationType> rodSubdomainSolutions_;
private:
    //////////////////////////////////////////////////////////////////
    //  Data members related to the continuum problems
    //////////////////////////////////////////////////////////////////

    const MatrixType* stiffnessMatrix3d_;
    
    const VectorType* dirichletValues_;
    
    const Dune::shared_ptr< ::LoopSolver<VectorType> > solver_;
    
    Dune::BitSetVector<dim> dirichletAndCouplingNodes_;
    
    /** \todo Hack:
     * - we actually need a base class
     * - we don't need the global ContinuumFEBasis
     */
    StVenantKirchhoffAssembler<ContinuumGridType, 
                               typename ContinuumFEBasis::LocalFiniteElement, 
                               typename ContinuumFEBasis::LocalFiniteElement>* localAssembler_;
    
public:
    mutable std::map<std::string, ContinuumConfigurationType> continuumSubdomainSolutions_;
private:
};

template <class RodGridType, class ContinuumGridType>
RigidBodyMotion<3>::TangentVector RodContinuumSteklovPoincareStep<RodGridType,ContinuumGridType>::
rodDirichletToNeumannMap(const RigidBodyMotion<3>& lambda) const
{
    // Create an initial iterate by interpolating between lambda and the Dirichlet value
    /** \todo Using that the coupling boundary is the one with the lower coordinate */
    RigidBodyMotion<3> rodDirichletValue = complex_.rodDirichletValues_.find("rod")->second.back();
    
    // Set initial iterate
    RodConfigurationType& rodX = rodSubdomainSolutions_["rod"];
    RodFactory<typename RodGridType::LeafGridView> rodFactory(complex_.rodGrid("rod")->leafView());
    rodFactory.create(rodX,lambda,rodDirichletValue);
    
    rodSolver_->setInitialSolution(rodX);
    
    // Solve the Dirichlet problem
    rodSolver_->solve();

    rodX = rodSolver_->getSol();

    //   Extract Neumann values

    Dune::BitSetVector<1> couplingBitfield(rodX.size(),false);
    /** \todo Using that index 0 is always the left boundary for a uniformly refined OneDGrid */
    couplingBitfield[0] = true;
    LeafBoundaryPatch<RodGridType> couplingBoundary(*complex_.rodGrid("rod"), couplingBitfield);

    /** \todo Hack: this should be a tangent vector right away */
    Dune::FieldVector<double,dim> rodForce, rodTorque;
    rodForce = rodAssembler_->getResultantForce(couplingBoundary, rodX, rodTorque);
    
    dune_static_assert(RigidBodyMotion<3>::TangentVector::size == 2*dim, "TangentVector does not have appropriate size");
    RigidBodyMotion<3>::TangentVector result;
    result[0] = rodForce[0];
    result[1] = rodForce[1];
    result[2] = rodForce[2];
    result[3] = rodTorque[0];
    result[4] = rodTorque[1];
    result[5] = rodTorque[2];
    
    return result;
}


template <class RodGridType, class ContinuumGridType>
RigidBodyMotion<3>::TangentVector RodContinuumSteklovPoincareStep<RodGridType,ContinuumGridType>::
continuumDirichletToNeumannMap(const RigidBodyMotion<3>& lambda) const
{
    std::pair<std::string,std::string> couplingName = std::make_pair("rod", "continuum");
    
    VectorType& x3d = continuumSubdomainSolutions_["continuum"];
    x3d.resize(complex_.continuumGrid("continuum")->size(dim));
    x3d = 0;

    // Turn \lambda \in TSE(3) into a Dirichlet value for the continuum
    const LeafBoundaryPatch<ContinuumGridType>& foo = complex_.coupling(couplingName).continuumInterfaceBoundary_;
    setRotation(foo, x3d, referenceInterface_, lambda);
    
    // Set the correct Dirichlet nodes
    dynamic_cast<IterationStep<VectorType>* >(solver_->iterationStep_)->ignoreNodes_ = &dirichletAndCouplingNodes_;
    
    // Right hand side vector: currently without Neumann and volume terms
    VectorType rhs3d(x3d.size());
    rhs3d = 0;
    
    // Solve the Dirichlet problem
    assert( (dynamic_cast<LinearIterationStep<MatrixType,VectorType>* >(solver_->iterationStep_)) );
    dynamic_cast<LinearIterationStep<MatrixType,VectorType>* >(solver_->iterationStep_)->setProblem(*stiffnessMatrix3d_, x3d, rhs3d);

    solver_->preprocess();
    dynamic_cast<IterationStep<VectorType>* >(solver_->iterationStep_)->preprocess();
        
    solver_->solve();
        
    x3d = dynamic_cast<IterationStep<VectorType>* >(solver_->iterationStep_)->getSol();
            
    // Integrate over the residual on the coupling boundary to obtain
    // an element of T^*SE.
    Dune::FieldVector<double,3> continuumForce, continuumTorque;
            
    VectorType residual = rhs3d;
    stiffnessMatrix3d_->mmv(x3d,residual);
            
    /** \todo Is referenceInterface.r the correct center of rotation? */
    computeTotalForceAndTorque(complex_.coupling(couplingName).continuumInterfaceBoundary_, residual, referenceInterface_.r,
                               continuumForce, continuumTorque);

    RigidBodyMotion<3>::TangentVector result;
    result[0] = continuumForce[0];
    result[1] = continuumForce[1];
    result[2] = continuumForce[2];
    result[3] = continuumTorque[0];
    result[4] = continuumTorque[1];
    result[5] = continuumTorque[2];
    
    return result;
}

/** \brief One preconditioned Richardson step
*/
template <class RodGridType, class ContinuumGridType>
void RodContinuumSteklovPoincareStep<RodGridType,ContinuumGridType>::iterate(RigidBodyMotion<3>& lambda)
{
    ///////////////////////////////////////////////////////////////////
    //  Evaluate the Dirichlet-to-Neumann map for the rod
    ///////////////////////////////////////////////////////////////////

    RigidBodyMotion<3>::TangentVector rodForceTorque = rodDirichletToNeumannMap(lambda);

    std::cout << "resultant rod force and torque: "  << rodForceTorque << std::endl;
            
    ///////////////////////////////////////////////////////////////////
    //  Evaluate the Dirichlet-to-Neumann map for the continuum
    ///////////////////////////////////////////////////////////////////
              
    RigidBodyMotion<3>::TangentVector continuumForceTorque = continuumDirichletToNeumannMap(lambda);

    std::cout << "resultant continuum force and torque: "  << continuumForceTorque  << std::endl;

    ///////////////////////////////////////////////////////////////
    //  Compute the overall Steklov-Poincare residual
    ///////////////////////////////////////////////////////////////

    // Flip orientation to account for opposing normals
    rodForceTorque *= -1;

    RigidBodyMotion<3>::TangentVector residualForceTorque = rodForceTorque + continuumForceTorque;
            
            
    ///////////////////////////////////////////////////////////////
    //  Apply the preconditioner
    ///////////////////////////////////////////////////////////////
            
    Dune::FieldVector<double,6> interfaceCorrection;
    std::pair<std::string,std::string> couplingName = std::make_pair("rod","continuum");
            
    if (preconditioner_=="DirichletNeumann") {
                
        ////////////////////////////////////////////////////////////////////
        //  Preconditioner is the linearized Neumann-to-Dirichlet map
        //  of the continuum.
        ////////////////////////////////////////////////////////////////////
                
        // Right hand side vector: currently without Neumann and volume terms
        VectorType rhs3d(complex_.continuumGrid("continuum")->size(dim));
        rhs3d = 0;

        LinearizedContinuumNeumannToDirichletMap<typename ContinuumGridType::LeafGridView,MatrixType,VectorType>
                linContNtDMap(complex_.coupling(couplingName).continuumInterfaceBoundary_,
                              rhs3d,
                              *dirichletValues_,
                              localAssembler_,
                              solver_);
                        
        interfaceCorrection = linContNtDMap.apply(continuumSubdomainSolutions_["continuum"], residualForceTorque, lambda.r);
                
    } else if (preconditioner_=="NeumannDirichlet") {
            
        ////////////////////////////////////////////////////////////////////
        //  Preconditioner is the linearized Neumann-to-Dirichlet map
        //  of the rod.
        ////////////////////////////////////////////////////////////////////
        
        LinearizedRodNeumannToDirichletMap<typename RodGridType::LeafGridView,RodCorrectionType> linRodNtDMap(complex_.coupling(couplingName).rodInterfaceBoundary_,
                                                                                                         rodLocalStiffness_);

        interfaceCorrection = linRodNtDMap.apply(rodSubdomainSolutions_["rod"], residualForceTorque, lambda.r);


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
                linContNtDMap(complex_.coupling(couplingName).continuumInterfaceBoundary_,
                              rhs3d,
                              *dirichletValues_,
                              localAssembler_,
                              solver_);

        LinearizedRodNeumannToDirichletMap<typename RodGridType::LeafGridView,RodCorrectionType> linRodNtDMap(complex_.coupling(couplingName).rodInterfaceBoundary_,
                                                                                                              rodLocalStiffness_);

        Dune::FieldVector<double,6> continuumCorrection = linContNtDMap.apply(continuumSubdomainSolutions_["continuum"], residualForceTorque, lambda.r);
        Dune::FieldVector<double,6> rodCorrection       = linRodNtDMap.apply(rodSubdomainSolutions_["rod"],
                                                                             residualForceTorque, lambda.r);
                
        for (int j=0; j<6; j++)
            interfaceCorrection[j] = (alpha_[0] * continuumCorrection[j] + alpha_[1] * rodCorrection[j])
                                        / alpha_[0] + alpha_[1];
                
    } else if (preconditioner_=="RobinRobin") {
            
        DUNE_THROW(Dune::NotImplemented, "Robin-Robin preconditioner not implemented yet");
                
    } else
        DUNE_THROW(Dune::NotImplemented, preconditioner_ << " is not a known preconditioner");
            
    ///////////////////////////////////////////////////////////////////////////////
    //  Apply the damped correction to the current interface value
    ///////////////////////////////////////////////////////////////////////////////
                
    interfaceCorrection *= richardsonDamping_;
    lambda = RigidBodyMotion<3>::exp(lambda, interfaceCorrection);
    
}

#endif