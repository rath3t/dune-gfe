#ifndef DUNE_GFE_BENDINGISOMETRYHELPER_HH
#define DUNE_GFE_BENDINGISOMETRYHELPER_HH

#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/geometry/quadraturerules.hh>

#include <dune/functions/functionspacebases/cubichermitebasis.hh>
#include <dune/functions/common/differentiablefunction.hh>
#include <dune/functions/common/differentiablefunctionfromcallables.hh>
#include <dune/functions/common/functionconcepts.hh>


#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>
#include <dune/gfe/functions/embeddedglobalgfefunction.hh>
#include <dune/gfe/functions/localprojectedfefunction.hh>

#include <dune/matrix-vector/crossproduct.hh>

/** \file
 * \brief Various helper functions related to bending isometries.
 */


namespace Dune::GFE::Impl
{


  /** \brief The identity function in R^d, and its derivative
   *
   * This is needed to compute the displacement from the deformation (for visualization).
   * Since we want to represent it in a Hermite finite element space, we need
   * the derivative as well.
   *
   * Is there no shorter way to implement this?
   */
  template<class K>
  class IdentityGridEmbedding
  {
  public:
    //! Evaluate identity
    Dune::FieldVector<K,3> operator() (const Dune::FieldVector<K,2>& x) const
    {
      return {x[0], x[1], 0.0};
    }

    /**
     * \brief Obtain derivative of identity function
     */
    friend auto derivative(const IdentityGridEmbedding& p)
    {
      return [](const Dune::FieldVector<K,2>& x) { return Dune::FieldMatrix<K,3,2>({{1,0}, {0,1}, {0,0}}); };
    }
  };


  /**
      Define some Generic lambda expressions (C++ 20)
   */
  auto cancelLastMatrixColumn = []<typename RT>(Dune::FieldMatrix<RT,3,3> matrix) -> Dune::FieldMatrix<RT,3,2>
  {
    return Dune::FieldMatrix<RT,3,2>{{matrix[0][0],matrix[0][1]},
      {matrix[1][0],matrix[1][1]},
      {matrix[2][0],matrix[2][1]}
    };
  };

  auto rotationExtraction = []<typename RT>(Dune::FieldVector<RT,7> vec) -> Rotation<RT,3>
  {
    std::array<RT,4> quaternion {vec[3],vec[4],vec[5],vec[6]};
    return Rotation<RT,3>(quaternion);
  };

  auto deformationExtraction = []<typename RT>(Dune::FieldVector<RT,7> vec) -> Dune::FieldVector<RT,3>
  {
    return Dune::FieldVector<RT,3>{vec[0],vec[1],vec[2]};
  };







  /**
   * @brief Data structure conversion from 'VectorSpaceCoefficients' to 'IsometryCoefficients'.
   */
  template<class DiscreteKirchhoffBasis, class CoefficientBasis, class Coefficients, class IsometryCoefficients>
  void vectorToIsometryCoefficientMap(const DiscreteKirchhoffBasis& discreteKirchhoffBasis,
                                      const CoefficientBasis& coefficientBasis,
                                      const Coefficients& vectorCoefficients,
                                      IsometryCoefficients& isometryCoefficients)
  {
    isometryCoefficients.resize(coefficientBasis.size());
    auto localView = discreteKirchhoffBasis.localView();
    auto localViewCoefficients = coefficientBasis.localView();

    using RT = typename IsometryCoefficients::value_type::ctype;

    for (const auto &element :  elements(coefficientBasis.gridView()))
    {
      localView.bind(element);
      localViewCoefficients.bind(element);

      /**
         Create data structures to store deformation values and the deformation gradient (Dofs)
         on the current element. We store the values in an std::vector where each entry
         contains the deformation vector (resp. x,y derivative vectors) at one node of the element.
       */
      std::vector<Dune::FieldVector<RT,3> > deformationDofs(3);
      std::vector<Dune::FieldVector<RT,3> > xDerivativeDofs(3);
      std::vector<Dune::FieldVector<RT,3> > yDerivativeDofs(3);

      // Loop over components of power basis
      for(std::size_t k=0; k<3 ; k++)
      {
        // This could be placed out of the loop if we assume a power basis.
        const auto& hermiteLFE = localView.tree().child(k).finiteElement();

        for(std::size_t i=0; i<hermiteLFE.size(); i++)
        {
          auto localIdx = localView.tree().child(k).localIndex(i);
          auto globalIdx = localView.index(localIdx);
          // Get current node index (Caution: This assumes that the hermite LocalFiniteElement only contains vertex dofs.)
          auto nodeIdx = hermiteLFE.localCoefficients().localKey(i).subEntity();

          /**
             We use the functionalDescriptor of the (discreteKirchhoffBasis) hermite LocalFiniteELement
             to determine whether a current DOF corresponds to a function evaluation or partial derivative w.r.t
             the first or second variable.
             (Alternatively to the - experimental - FunctionalDescriptor: use localKey(i).index(): 0~value, 1~xDerivative, 2~yDerivative.)
           */
          if(std::array<unsigned int,2> val {0,0} ; hermiteLFE.localInterpolation().functionalDescriptor(i).partialDerivativeOrder() == val  )
          {
            deformationDofs[nodeIdx][k] = vectorCoefficients[globalIdx[0]][globalIdx[1]];
          }
          if(std::array<unsigned int,2> val {1,0} ; hermiteLFE.localInterpolation().functionalDescriptor(i).partialDerivativeOrder() == val  )
          {
            xDerivativeDofs[nodeIdx][k] = vectorCoefficients[globalIdx[0]][globalIdx[1]];
          }
          if(std::array<unsigned int,2> val {0,1} ; hermiteLFE.localInterpolation().functionalDescriptor(i).partialDerivativeOrder() == val  )
          {
            yDerivativeDofs[nodeIdx][k] = vectorCoefficients[globalIdx[0]][globalIdx[1]];
          }
        }
      }

      // Loop over vertices of the current element
      for(std::size_t c=0; c<3 ; c++)
      {
        /** Normalize derivative vectors (this is necessary due to the h-dependent scaling used in the hermite-Basis). */
        xDerivativeDofs[c] /= xDerivativeDofs[c].two_norm();
        yDerivativeDofs[c] /= yDerivativeDofs[c].two_norm();

        /**
            cross product to get last column of rotation matrix.
         */
        Dune::FieldVector<RT,3> cross = Dune::MatrixVector::crossProduct(xDerivativeDofs[c],yDerivativeDofs[c]);

        Dune::FieldMatrix<RT,3,3> rotMatrix(0);

        for(std::size_t k=0; k<3; k++)
        {
          rotMatrix[k][0] = xDerivativeDofs[c][k];
          rotMatrix[k][1] = yDerivativeDofs[c][k];
          rotMatrix[k][2] = cross[k];
        }

        /**
           Check if derivative vectors are orthonormal.
         */
        Dune::FieldMatrix<double,3,3> I = Dune::ScaledIdentityMatrix<double,3>(1);

        if(((rotMatrix.transposed()*rotMatrix) - I).frobenius_norm()>1e-8)
          DUNE_THROW(Dune::Exception, "Error: input rotation is not orthonormal!"<< ((rotMatrix.transposed()*rotMatrix) - I).frobenius_norm());

        size_t localIdxOut = localViewCoefficients.tree().localIndex(c);
        size_t globalIdxOut = localViewCoefficients.index(localIdxOut);

        using namespace Dune::Indices;
        isometryCoefficients[globalIdxOut][_1].set(rotMatrix);
        isometryCoefficients[globalIdxOut][_0] = (RealTuple<RT, 3>)deformationDofs[c];
      }
    }
  }


  /**
   * @brief Data structure conversion from 'IsometryCoefficients' to 'VectorSpaceCoefficients'.
   */
  template<class DiscreteKirchhoffBasis, class CoefficientBasis, class Coefficients, class IsometryCoefficients>
  void isometryToVectorCoefficientMap(const DiscreteKirchhoffBasis& discreteKirchhoffBasis,
                                      const CoefficientBasis& coefficientBasis,
                                      Coefficients& vectorCoefficients,
                                      const IsometryCoefficients& isometryCoefficients)
  {
    auto localView = discreteKirchhoffBasis.localView();
    auto localViewCoefficients = coefficientBasis.localView();

    auto gridView = discreteKirchhoffBasis.gridView();

    /** Create an EmbeddedGlobalGFEFunction that serves as a GridViewFunction later on.
        We need a function to be passed into the interpolation method of the hermite basis
        (although the values in between grid nodes do not matter).

        The range type of EmbeddedGlobalFEFunction is a FieldVector<double,7>
        where the first three entries correspond to the deformation and the
        last four entries correspond to a (quaternion) rotation.
     */
    typedef GFE::LocalProjectedFEFunction<CoefficientBasis, GFE::ProductManifold<RealTuple<double,3>, Rotation<double,3> > > LocalInterpolationRule;
    Dune::GFE::EmbeddedGlobalGFEFunction<CoefficientBasis, LocalInterpolationRule, Dune::GFE::ProductManifold<RealTuple<double,3>, Rotation<double,3> > > embeddedGlobalFunction(coefficientBasis, isometryCoefficients);

    auto productSpaceGridViewFunction = Dune::Functions::makeAnalyticGridViewFunction(embeddedGlobalFunction, gridView);

    /** Extract the deformation and rotation into separate functions. */
    auto deformationGlobalFunction = Dune::Functions::makeComposedGridFunction(deformationExtraction,productSpaceGridViewFunction);



    auto rotationGlobalFunction = Dune::Functions::makeComposedGridFunction(cancelLastMatrixColumn,
                                                                            Dune::Functions::makeComposedGridFunction(Rotation<double,3>::quaternionToMatrix,
                                                                                                                      Dune::Functions::makeComposedGridFunction(rotationExtraction,productSpaceGridViewFunction)));

    using Domain = const Dune::FieldVector<double,2>&;
    using Range = Dune::FieldVector<double,3>;
    auto globalIsometryFunction = makeDifferentiableFunctionFromCallables(Dune::Functions::SignatureTag<Range(Domain)>(),deformationGlobalFunction,rotationGlobalFunction);

    /** Interpolate into the hermite basis to get the coefficient vector. */
    interpolate(discreteKirchhoffBasis, vectorCoefficients, globalIsometryFunction);
  }


  template<class LocalDiscreteKirchhoffFunction, class NormalBasis>
  auto computeDiscreteSurfaceNormal(LocalDiscreteKirchhoffFunction& deformationFunction,
                                    const NormalBasis& normalBasis)
  {
    std::vector<Dune::FieldVector<double,3> > discreteNormalVectorCoefficients(normalBasis.size());

    auto localView = normalBasis.localView();
    for (auto&& element : elements(normalBasis.gridView()))
    {
      auto geometry = element.geometry();

      localView.bind(element);
      const auto nSf = localView.tree().child(0).finiteElement().localBasis().size();

      deformationFunction.bind(element);

      for (int i=0; i<geometry.corners(); i++)
      {
        auto numDir = deformationFunction.evaluateDerivative(geometry.local(geometry.corner(i)));

        /**
            Get the normal vector via cross-product.
         */
        Dune::FieldVector<double,3> normal = {numDir[1][0]*numDir[2][1] - numDir[1][1]*numDir[2][0],
                                              numDir[2][0]*numDir[0][1] - numDir[0][0]*numDir[2][1],
                                              numDir[0][0]*numDir[1][1] - numDir[1][0]*numDir[0][1]};
        // printvector(std::cout, normal, "discrete normal:" , "--");
        for (size_t k=0; k < 3; k++)
          for (size_t i=0; i < nSf; i++)
          {
            size_t localIdx = localView.tree().child(k).localIndex(i); // hier i:leafIdx
            size_t globalIdx = localView.index(localIdx);
            discreteNormalVectorCoefficients[globalIdx] = normal[k];
          }
      }
    }

    return discreteNormalVectorCoefficients;
  }


  /**
   * @brief  Compute Coefficients of the local discrete gradient operator.
   *
   * The discrete Gradient is a special linear combination represented in a [P2]^2  space (locally by a representation local finite element)
   * The coefficients of this linear combination correspond to certain linear combinations of the Gradients of localfunction_ .
   * The coefficients are stored in the form [Basisfunctions x components x gridDim]
   * in a BlockVector<FieldMatrix<RT, 3, gridDim>> .
   */
  template<class Coefficient,class LocalP2Element, class DeformationFunction >
  auto discreteGradientCoefficients(Coefficient& discreteGradientCoefficients,
                                    LocalP2Element& representationLFE,
                                    DeformationFunction& deformationFunction,
                                    auto& element)
  {
    using RT = typename DeformationFunction::RT;

    auto geometry = element.geometry();
    constexpr static int gridDim = DeformationFunction::gridDim;
    auto gridView = deformationFunction.gridView();
    const auto &indexSet = gridView.indexSet();
    discreteGradientCoefficients.resize(representationLFE.size());
    /**
     * @brief On the current element we need to assign the coefficients of the discrete Gradient to the right P2-Lagrange-Basisfunctions.
     *        this is done by iterating over all P2-Lagrange-Basisfunctions and determine the corresponding coefficients.
     *        We distinguish two situations:
     *        - [1] If the Lagrange-basisfunction is assigned to a node: the corresponding coefficient is just the evaluation of the deformation gradient at that specific node.
     *        - [2] If the Lagrange-basisfunction is assigned to an edge center: the corresponding coefficient is decomposed into a normal-& tangential part,
     *              where the normal-part: is given by the affine combination of the deformation gradient values at the edge-vertices
     *              and the tangential-part: is given by the tangential part of the deformation gradient at the edge-center.
     *
     *        * We do this for all (k=3) components of the deformation-function simulatneously.
     */
    for(std::size_t i=0; i<representationLFE.size(); i++)
    {
      // [1] Determine coefficients for basis functions assigned to a vertex
      if (representationLFE.localCoefficients().localKey(i).codim() == 2)
      {
        size_t nodeIndex = representationLFE.localCoefficients().localKey(i).subEntity();     // This gives index on reference Element ?
        // Alternative: use subEntity to get (local) node position

        typename DeformationFunction::DerivativeType whJacobianValue = deformationFunction.evaluateDerivative(geometry.local(geometry.corner(nodeIndex)));
        discreteGradientCoefficients[i] = whJacobianValue;
      }

      // [2] Determine coefficients for basis functions assigned to an edge
      if (representationLFE.localCoefficients().localKey(i).codim() == 1)
      {
        /**
         * @brief On each edge the (componentwise) value of the discete jacobian on the edge midpoint
         * is decomposed into a normal and tangent component.
         * The normal component at the edge center is given as the affine combination of the
         * normal component of the jacobian of localfunctions_ at the two vertices on each edge.
         * The tangential component is the tangent component of the jacobian of localfunctions_ on
         * the edge center.
         */
        size_t edgeIndex =  representationLFE.localCoefficients().localKey(i).subEntity();
        auto edgeGeometry = element.template subEntity<1>(edgeIndex).geometry();

        /**
         * @brief Get the right orientation for tangent and normal vectors on current edge.
         */
        const auto &refElement = referenceElement(element);
        // Local vertex indices within the element
        auto localV0 = refElement.subEntity(edgeIndex, gridDim-1, 0, gridDim);
        auto localV1 = refElement.subEntity(edgeIndex, gridDim-1, 1, gridDim);

        // Global vertex indices within the grid
        auto globalV0 = indexSet.subIndex(element, localV0, gridDim);
        auto globalV1 = indexSet.subIndex(element, localV1, gridDim);

        double edgeOrientation = 1.0;

        if ((localV0<localV1 && globalV0>globalV1) || (localV0>localV1 && globalV0<globalV1))
          edgeOrientation = -1.0;

        // Get tangent vector on current edge
        Dune::FieldVector<RT,2> tmp = (geometry.corner(localV1) - geometry.corner(localV0)) / abs(edgeGeometry.volume());

        //convert to FieldMatrix
        Dune::FieldMatrix<RT,2,1> edgeTangent = {tmp[0], tmp[1]};
        edgeTangent *= edgeOrientation;

        /**
         * @brief Get edge normal by rotating the edge-tangent by by pi/2 clockwise
         */
        Dune::FieldMatrix<RT,2,1> edgeNormal = {edgeTangent[1], -1.0*edgeTangent[0]};

        /**
         * @brief Evaluate Jacobian of localfunctions_ at edge-vertices and edge-midpoints.
         */
        typename DeformationFunction::DerivativeType whJacobianValueCenter = deformationFunction.evaluateDerivative(geometry.local(edgeGeometry.center()));  // Q: Are these transformations correct?
        typename DeformationFunction::DerivativeType whJacobianValueFirstVertex = deformationFunction.evaluateDerivative(geometry.local(geometry.corner(localV0)));
        typename DeformationFunction::DerivativeType whJacobianValueSecondVertex = deformationFunction.evaluateDerivative(geometry.local(geometry.corner(localV1)));

        // @OS: Is there a elementwise/hadamard matrix multiplication in Dune?
        Dune::FieldMatrix<RT,3,gridDim> normalComponent(0);
        Dune::FieldMatrix<RT,3,gridDim> tangentialComponent(0);

        auto normalComponentFactor = 0.5 * (whJacobianValueFirstVertex + whJacobianValueSecondVertex) * edgeNormal;
        auto tangentialComponentFactor = whJacobianValueCenter * edgeTangent;

        /**
         * @brief Compute normal and tangential component on current edge.
         */
        for (int k=0; k<3; k++)
          for (int l=0; l<gridDim; l++)
          {
            normalComponent[k][l]   = normalComponentFactor[k][0]*edgeNormal[l];
            tangentialComponent[k][l] = tangentialComponentFactor[k][0]*edgeTangent[l];
          }
        discreteGradientCoefficients[i] = normalComponent + tangentialComponent;

      }
    } //end of P2-indices
  }



} // end namespace Dune::GFE::Impl

#endif
