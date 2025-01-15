#ifndef VALUE_FACTORY_HH
#define VALUE_FACTORY_HH

#include <vector>

#include <dune/gfe/spaces/hyperbolichalfspacepoint.hh>
#include <dune/gfe/spaces/orthogonalmatrix.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>
#include <dune/gfe/spaces/unitvector.hh>


namespace Dune::GFE
{

/** \brief A class that creates sets of values of various types, to be used in unit tests
 *
 * This is the generic dummy.  The actual work is done in specializations.
 */
template <class T>
class ValueFactory
{
public:
  static void get(std::vector<T>& values);

};


/** \brief A class that creates sets of values of various types, to be used in unit tests
 *
 * This is the specialization for RealTuple<1>
 */
template <>
class ValueFactory<RealTuple<double,1> >
{
public:
  static void get(std::vector<RealTuple<double,1> >& values) {

    std::vector<double> testPoints = {-3, -1, 0, 2, 4};

    values.resize(testPoints.size());

    // Set up elements of S^1
    for (size_t i=0; i<values.size(); i++)
      values[i] = RealTuple<double,1>(testPoints[i]);

  }

};

/** \brief A class that creates sets of values of various types, to be used in unit tests
 *
 * This is the specialization for RealTuple<3>
 */
template <>
class ValueFactory<RealTuple<double,3> >
{
public:
  static void get(std::vector<RealTuple<double,3> >& values) {

    std::vector<Dune::FieldVector<double,3> > testPoints = {{1,0,0}, {0,1,0}, {-0.838114,0.356751,-0.412667},
      {-0.490946,-0.306456,0.81551},{-0.944506,0.123687,-0.304319},
      {-0.6,0.1,-0.2},{0.45,0.12,0.517},
      {-0.1,0.3,-0.1},{-0.444506,0.123687,0.104319},{-0.7,-0.123687,-0.304319}};

    values.resize(testPoints.size());

    // Set up elements of S^1
    for (size_t i=0; i<values.size(); i++)
      values[i] = RealTuple<double,3>(testPoints[i]);

  }

};

/** \brief A class that creates sets of values of various types, to be used in unit tests
 *
 * This is the specialization for UnitVector<2>
 */
template <>
class ValueFactory<UnitVector<double,2> >
{
public:
  static void get(std::vector<UnitVector<double,2> >& values) {

    std::vector<std::array<double,2> > testPoints = {{1,0}, {0.5,0.5}, {0,1}, {-0.5,0.5}, {-1,0}, {-0.5,-0.5}, {0,-1}, {0.5,-0.5}, {0.1,1}, {1,.1}};

    values.resize(testPoints.size());

    // Set up elements of S^1
    for (size_t i=0; i<values.size(); i++)
      values[i] = UnitVector<double,2>(testPoints[i]);

  }

};


/** \brief A class that creates sets of values of various types, to be used in unit tests
 *
 * This is the specialization for UnitVector<3>
 */
template <>
class ValueFactory<UnitVector<double,3> >
{
public:
  static void get(std::vector<UnitVector<double,3> >& values) {

    std::vector<std::array<double,3> > testPoints = {{1,0,0}, {0,1,0}, {-0.838114,0.356751,-0.412667},
      {-0.490946,-0.306456,0.81551},{-0.944506,0.123687,-0.304319},
      {-0.6,0.1,-0.2},{0.45,0.12,0.517},
      {-0.1,0.3,-0.1},{-0.444506,0.123687,0.104319},{-0.7,-0.123687,-0.304319}};

    values.resize(testPoints.size());

    // Set up elements of S^1
    for (size_t i=0; i<values.size(); i++)
      values[i] = UnitVector<double,3>(testPoints[i]);

  }

};


/** \brief A class that creates sets of values of various types, to be used in unit tests
 *
 * This is the specialization for UnitVector<4>
 */
template <>
class ValueFactory<UnitVector<double,4> >
{
public:
  static void get(std::vector<UnitVector<double,4> >& values) {

    std::vector<std::array<double,4> > testPoints = {{1,0,0,0}, {0,1,0,0}, {-0.838114,0.356751,-0.412667,0.5},
      {-0.490946,-0.306456,0.81551,0.23},{-0.944506,0.123687,-0.304319,-0.7},
      {-0.6,0.1,-0.2,0.8},{0.45,0.12,0.517,0},
      {-0.1,0.3,-0.1,0.73},{-0.444506,0.123687,0.104319,-0.23},{-0.7,-0.123687,-0.304319,0.72}};


    values.resize(testPoints.size());

    // Set up elements of S^1
    for (size_t i=0; i<values.size(); i++)
      values[i] = UnitVector<double,4>(testPoints[i]);

  }

};


/** \brief A class that creates sets of values of various types, to be used in unit tests
 *
 * This is the specialization for Rotation<3>
 */
template <>
class ValueFactory<Rotation<double,3> >
{
public:
  static void get(std::vector<Rotation<double,3> >& values) {

    std::vector<std::array<double,4> > testPoints = {{1,0,0,0}, {0,1,0,0}, {-0.838114,0.356751,-0.412667,0.5},
      {-0.490946,-0.306456,0.81551,0.23},{-0.944506,0.123687,-0.304319,-0.7},
      {-0.6,0.1,-0.2,0.8},{0.45,0.12,0.517,0},
      {-0.1,0.3,-0.1,0.73},{-0.444506,0.123687,0.104319,-0.23},{-0.7,-0.123687,-0.304319,0.72},
      {-0.035669, -0.463824, -0.333265, 0.820079}, {0.0178678, 0.916836, 0.367358, 0.155374}};

    values.resize(testPoints.size());

    // Set up elements of S^1
    for (std::size_t i=0; i<values.size(); i++)
      values[i] = Rotation<double,3>(testPoints[i]);

  }

};

/** \brief A class that creates sets of values of various types, to be used in unit tests
 *
 * This is the specialization for a ProductManifold<RealTuple,Rotation>
 */
template <>
class ValueFactory<Dune::GFE::ProductManifold<RealTuple<double,3>,Rotation<double,3> > >
{
public:
  static void get(std::vector<Dune::GFE::ProductManifold<RealTuple<double,3>,Rotation<double,3> > >& values) {

    using namespace Dune::Indices;

    std::vector<RealTuple<double,3> > rValues;
    ValueFactory<RealTuple<double,3> >::get(rValues);

    std::vector<Rotation<double,3> > qValues;
    ValueFactory<Rotation<double,3> >::get(qValues);

    int nTestPoints = std::min(rValues.size(), qValues.size());

    values.resize(nTestPoints);

    // Set up elements of SE(3)
    for (int i=0; i<nTestPoints; i++)
    {
      values[i][_0] = rValues[i];
      values[i][_1] = qValues[i];
    }

  }

};

/** \brief A class that creates sets of values of various types, to be used in unit tests
 *
 * This is the specialization for square FieldMatrices
 */
template <class T, int N>
class ValueFactory<Dune::FieldMatrix<T,N,N> >
{
public:
  static void get(std::vector<Dune::FieldMatrix<T,N,N> >& values) {

    int nTestPoints = 10;
    values.resize(nTestPoints);

    // Set up elements of T^{N \times N}
    for (int i=0; i<nTestPoints; i++)
      for (int j=0; j<N; j++)
        for (int k=0; k<N; k++)
          values[i][j][k] = std::rand()%100 - 50;

  }

};


/** \brief A class that creates sets of values of various types, to be used in unit tests
 *
 * This is the specialization for OrthogonalMatrices
 */
template <class T, int N>
class ValueFactory<OrthogonalMatrix<T,N> >
{

  static Dune::FieldVector<T,N> proj(const Dune::FieldVector<T,N>& u, const Dune::FieldVector<T,N>& v)
  {
    Dune::FieldVector<T,N> result = u;
    result *= (v*u) / (u*u);
    return result;
  }

public:
  static void get(std::vector<OrthogonalMatrix<T,N> >& values) {

    // Get general matrices
    std::vector<Dune::FieldMatrix<T,N,N> > mValues;
    ValueFactory<Dune::FieldMatrix<T,N,N> >::get(mValues);

    values.resize(mValues.size());

    // Do Gram-Schmidt orthogonalization of the rows
    for (size_t m=0; m<mValues.size(); m++) {

      Dune::FieldMatrix<T,N,N>& v = mValues[m];

      if (std::fabs(v.determinant()) < 1e-6)
        continue;

      for (int j=0; j<N; j++) {

        for (int i=0; i<j; i++) {

          // v_j = v_j - proj_{v_i} v_j
          v[j] -= proj(v[i],v[j]);

        }

        // normalize
        v[j] /= v[j].two_norm();
      }

      values[m] = OrthogonalMatrix<T,N>(v);

    }

  }

};

/** \brief A class that creates sets of values of various types, to be used in unit tests
 *
 * This is the specialization for HyperbolicHalfspacePoint<3>
 */
template <>
class ValueFactory<HyperbolicHalfspacePoint<double,2> >
{
public:
  static void get(std::vector<HyperbolicHalfspacePoint<double,2> >& values) {

    std::vector<Dune::FieldVector<double,2> > testPoints = {{0,2}, {0,1}, {0,0.5},
      {-0.490946,0.81551},{-0.944506,0.304319},
      {-0.6,0.2},{0.45,0.517},
      {-0.1,0.1},{-0.444506,0.104319},{-0.7,0.304319}};

    values.resize(testPoints.size());

    // Set up elements of S^1
    for (size_t i=0; i<values.size(); i++)
      values[i] = HyperbolicHalfspacePoint<double,2>(testPoints[i]);

  }

};

/** \brief A class that creates sets of values of various types, to be used in unit tests
 *
 * This is the specialization for HyperbolicHalfspacePoint<3>
 */
template <>
class ValueFactory<HyperbolicHalfspacePoint<double,3> >
{
public:
  static void get(std::vector<HyperbolicHalfspacePoint<double,3> >& values) {

    std::vector<Dune::FieldVector<double,3> > testPoints = {{1,0,0.01}, {0,1,0.01}, {-0.838114,0.356751,0.412667},
      {-0.490946,-0.306456,0.81551},{-0.944506,0.123687,0.304319},
      {-0.6,0.1,0.2},{0.45,0.12,0.517},
      {-0.1,0.3,0.1},{-0.444506,0.123687,0.104319},{-0.7,-0.123687,0.304319}};

    values.resize(testPoints.size());

    // Set up elements of S^1
    for (size_t i=0; i<values.size(); i++)
      values[i] = HyperbolicHalfspacePoint<double,3>(testPoints[i]);

  }

};





/** \brief A class that creates sets of values of various types, to be used in unit tests
 *
 * This is the specialization for ProductManifold<...>
 */
template <typename ... TargetSpaces>
class ValueFactory<Dune::GFE::ProductManifold<TargetSpaces...> >
{
  using TargetSpace = Dune::GFE::ProductManifold<TargetSpaces...>;

public:
  static void get(std::vector<TargetSpace >& values) {

    std::vector<typename TargetSpace::CoordinateType > testPoints(10);

    std::generate(testPoints.begin(), testPoints.end(), [](){
      return Dune::GFE::randomFieldVector<typename TargetSpace::field_type,TargetSpace::CoordinateType::dimension>(0.9,1.1) ;
    });

    values.resize(testPoints.size());

    std::transform(testPoints.cbegin(),testPoints.cend(),values.begin(),[](const auto& testPoint){return TargetSpace(testPoint);});
  }
};

}  // namespace Dune::GFE

#endif
