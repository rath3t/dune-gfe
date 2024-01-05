// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/gfe/polardecomposition.hh>
#include <dune/gfe/spaces/rotation.hh>

#include <chrono>
#include <fstream>
#include <random>


using namespace Dune;
using field_type = double;


class PolarDecompositionTest : public Dune::GFE::HighamNoferiniPolarDecomposition {
public:
  using HighamNoferiniPolarDecomposition::HighamNoferiniPolarDecomposition;
  using HighamNoferiniPolarDecomposition::bilinearMatrix;
  using HighamNoferiniPolarDecomposition::determinantLaplace;
  using HighamNoferiniPolarDecomposition::dominantEVNewton;
  using HighamNoferiniPolarDecomposition::characteristicCoefficients;
  using HighamNoferiniPolarDecomposition::obtainQuaternion;
};

/** \brief Check if the matrix is not orthogonal */
static bool isNotOrthogonal(const FieldMatrix<field_type,3,3>& matrix) {
  bool notOrthogonal = std::abs(1-matrix.determinant()) > 1e-12;
  notOrthogonal = notOrthogonal or (( matrix[0]*matrix[1] ) > 1e-12);
  notOrthogonal = notOrthogonal or (( matrix[0]*matrix[2] ) > 1e-12);
  notOrthogonal = notOrthogonal or (( matrix[1]*matrix[2] ) > 1e-12);
  return notOrthogonal;
}

/** \brief Return the time needed for 1000 polar decompositions of the given matrix using the iterative algorithm
        and the algorithm by Higham. */
static FieldVector<field_type,2> measureTimeForPolarDecomposition(const FieldMatrix<field_type, 3, 3>& matrix, double tol = 10e-3)
{
  FieldMatrix<field_type, 3, 3> Q1;
  FieldMatrix<field_type, 3, 3> Q2;
  FieldVector<field_type,2> actualtime;
  std::chrono::steady_clock::time_point begindecomold = std::chrono::steady_clock::now();
  for (int i = 0; i < 1000; ++i) {
    Q1 = Dune::GFE::PolarDecomposition()(matrix, tol);
  }
  std::chrono::steady_clock::time_point enddecomold = std::chrono::steady_clock::now();
  actualtime[0] = (enddecomold - begindecomold).count();

  std::chrono::steady_clock::time_point begindecom = std::chrono::steady_clock::now();
  for (int i = 0; i < 1000; ++i) {
    Q2 = Dune::GFE::HighamNoferiniPolarDecomposition()(matrix, tol, tol);
  }
  std::chrono::steady_clock::time_point enddecom = std::chrono::steady_clock::now();
  actualtime[1] = (enddecom - begindecom).count();
  return actualtime;
}

/** \brief Returns a 3x3-Matrix with random entries between 0 and upper Bound*/
static FieldMatrix<field_type, 3, 3> randomMatrix3d(double upperBound = 1.0)
{
  const int dim = 3;
  FieldMatrix<field_type, dim, dim> matrix;
  std::random_device rd;    // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd());   // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(0.0, upperBound);   // equally distributed between 0 and upper bound
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j)
      matrix[i][j] = dis(gen);

  return matrix;
}

/** \brief Returns an "almost orthogonal" 3x3-Matrix where a random perturbation
                     between 0 and maxPerturbation is added to each entry.*/
static FieldMatrix<field_type, 3, 3> randomMatrixAlmostOrthogonal(double maxPerturbation = 0.1)
{
  const int dim = 3;
  FieldVector<field_type,4> f;
  std::random_device rd;    // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd());   // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(0.0, 1.0);   // equally distributed between 0 and upper bound
  for (int i = 0; i < 4; ++i)
    f[i] = dis(gen);
  Rotation<field_type,3> q(f);
  q.normalize();

  assert(std::abs(1-q.two_norm()) < 1e-12);

  // Turn it into a matrix
  FieldMatrix<double,3,3> matrix;
  q.matrix(matrix);

  if (isNotOrthogonal(matrix))
    DUNE_THROW(Exception, "Expected the matrix " << matrix << " to be orthogonal, but it is not!");

  //Now perturb this a little bit
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j)
      matrix[i][j] += dis(gen)*maxPerturbation;
  return matrix;
}

static double timeTest(double perturbationFromSO3 = 1.0) {
  // Define matrices we want to use for testing
  int numberOfTests = 100;
  std::vector<double> timeOld(numberOfTests);
  std::vector<double> timeNew(numberOfTests);
  double totalTimeOld = 0;
  double totalTimeNew = 0;

  for (int j = 0; j < numberOfTests; ++j) {   // testing loop
    FieldMatrix<field_type,3,3> N;
    // Only measure the time if the decomposition is unique and if both algorithms will return an orthogonal matrix!
    // Attention: For matrices that are quite far away from an orthogonal matrix, Dune::GFE::PolarDecomposition() might return a matrix with determinant = -1 !
    double normOfDifference = 10;
    FieldMatrix<field_type,3,3> Q1;
    FieldMatrix<field_type,3,3> Q2;
    while(isNotOrthogonal(Q1) or isNotOrthogonal(Q2) or normOfDifference > 0.001) {
      N = randomMatrixAlmostOrthogonal(perturbationFromSO3);
      Q1 = PolarDecompositionTest()(N);
      Q2 = Dune::GFE::PolarDecomposition()(N);
      normOfDifference = (Q1-Q2).frobenius_norm();
    }
    auto actualtime = measureTimeForPolarDecomposition(N);
    timeOld[j] = actualtime[0];
    timeNew[j] = actualtime[1];
  }

  std::sort(timeOld.begin(), timeOld.end());
  std::sort(timeNew.begin(), timeNew.end());

  for (int j = 0; j < numberOfTests; ++j) {
    totalTimeOld += timeOld[j];
    totalTimeNew += timeNew[j];
  }
  std::cout << "Perturbation from an orthogonal matrix: " << perturbationFromSO3 << std::endl;
  std::cout << "Average (old): " << totalTimeOld/numberOfTests << "[ns]" << std::endl;
  std::cout << "Average (new): " << totalTimeNew/numberOfTests << "[ns]" << std::endl;
  std::cout << "Relative: " << totalTimeOld/totalTimeNew << std::endl << std::endl;

  return totalTimeOld/totalTimeNew;
}

static bool testBilinearMatrix() {
  FieldMatrix<field_type,3,3> M = { { 20,    0,  -3 },
    {  3,  2.0, 310 },
    {  5, 0.22,   0 } };
  M /= M.frobenius_norm();
  FieldMatrix<field_type,4,4> B = { { 0.0096632,  0.997822,   0.0257685,  0.00644213 },
    { 0.997822,  -0.00322107, 0.0708635,  0.00644213 },
    { 0.0257685,  0.0708635,  0.00322107, 0.999239   },
    { 0.00644213, 0.00644213, 0.999239,  -0.0096632  } };
  auto Btest = PolarDecompositionTest::bilinearMatrix(M);
  Btest -= B;
  return Btest.frobenius_norm() < 0.001;
}

static bool test4dDeterminant() {
  FieldMatrix<field_type,4,4> B = { { 0.0096632,  0.997822,   0.0257685,  0.00644213 },
    { 0.997822,  -0.00322107, 0.0708635,  0.00644213 },
    { 0.0257685,  0.0708635,  0.00322107, 0.999239   },
    { 0.00644213, 0.00644213, 0.999239,  -0.0096632  } };
  double detBtest = PolarDecompositionTest::determinantLaplace(B);
  detBtest -= B.determinant();
  return std::abs(detBtest) < 0.001;
}

static bool testdominantEVNewton() {
  field_type detB = 0.992928;
  FieldVector<field_type,3>  minpol = {0, -1.99999, -0.00496083};
  double correctDominantEV = 1.05397;

  auto dominantEVNewton = PolarDecompositionTest::dominantEVNewton(minpol, detB);
  return std::abs(correctDominantEV - dominantEVNewton) < 0.001;
}

static bool testCharacteristicPolynomial4D() {
  FieldMatrix<field_type,4,4> B = { { 0.0096632,  0.997822,   0.0257685,  0.00644213 },
    { 0.997822,  -0.00322107, 0.0708635,  0.00644213 },
    { 0.0257685,  0.0708635,  0.00322107, 0.999239   },
    { 0.00644213, 0.00644213, 0.999239,  -0.0096632  } };
  FieldVector<field_type,3>  minpol = {0, -1.99999, -0.00496083};

  auto coefficients = PolarDecompositionTest::characteristicCoefficients(B);
  coefficients -= minpol;
  return coefficients.two_norm() < 0.001;
}

static bool testQuaternionFunction() {
  FieldMatrix<field_type,4,4> BS = { {  1.04431,    -0.997822,   -0.0257685, -0.00644213 },
    { -0.997822,    1.05719,    -0.0708635, -0.00644213 },
    { -0.0257685,  -0.0708635,   1.05075,   -0.999239   },
    { -0.00644213, -0.00644213, -0.999239,   1.06363    } };

  FieldVector<field_type, 4 > v = { 0.508566, 0.516353, 0.499062, 0.475055 };
  auto vtest = PolarDecompositionTest::obtainQuaternion(BS);
  vtest -= v;
  return vtest.two_norm() < 0.001;
}

int main (int argc, char *argv[]) try
{
  //test the correctness of the algorithm
  auto testMatrix = randomMatrix3d();
  auto Q = PolarDecompositionTest()(testMatrix,1e-12,1e-12);
  if (isNotOrthogonal(Q)) {
    std::cerr << "The polar decomposition did not return an orthogonal matrix when decomposing: " << std::endl << testMatrix << std::endl;
    return 1;
  }

  if (testBilinearMatrix()) {
    std::cerr << "The test calculation of the bilinearMatrix is wrong!" << std::endl;
    return 1;
  }
  if (!test4dDeterminant()) {
    std::cerr << "The test calculation of the 4d determinant is wrong!" << std::endl;
    return 1;
  }
  if (!testdominantEVNewton()) {
    std::cerr << "The test calculation of the dominant eigenvalue is wrong!" << std::endl;
    return 1;
  }
  if (!testCharacteristicPolynomial4D()) {
    std::cerr << "The test calculation of the characteristic polynomial is wrong!" << std::endl;
    return 1;
  }
  if (!testQuaternionFunction()) {
    std::cerr << "The test calculation of the quaternion function is wrong!" << std::endl;
    return 1;
  }

  timeTest(.1);
  timeTest(2);
  timeTest(30);

  return 0;

}
catch (Exception& e) {
  std::cout << e.what() << std::endl;
}
