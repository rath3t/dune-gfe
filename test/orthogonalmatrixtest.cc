#include <config.h>

#include <dune/gfe/spaces/orthogonalmatrix.hh>

#include "valuefactory.hh"


/** \file
    \brief Unit tests for the OrthogonalMatrix class
*/

using namespace Dune;

double eps = 1e-6;

/** \brief Test whether orthogonal matrix really is orthogonal
 */
template <class T, int N>
void testOrthogonality(const OrthogonalMatrix<T,N>& p)
{
    Dune::FieldMatrix<T,N,N> prod(0);
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++)
            for (int k=0; k<N; k++)
                prod[i][j] += p.data()[k][i] * p.data()[k][j];
            
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++)
            if ( std::abs(prod[i][j] - (i==j) ) > eps)
                DUNE_THROW(Exception, "Matrix is not orthogonal");
}
    
/** \brief Test orthogonal projection onto the tangent space at p
 */
template <class T, int N>
void testProjectionOntoTangentSpace(const OrthogonalMatrix<T,N>& p)
{
    std::vector<FieldMatrix<T,N,N> > testVectors;
    ValueFactory<FieldMatrix<T,N,N> >::get(testVectors);
    
    // Test each element in the list
    for (size_t i=0; i<testVectors.size(); i++) {
        
        // get projection
        FieldMatrix<T,N,N> projected = p.projectOntoTangentSpace(testVectors[i]);
        
        // Test whether the tangent vector is really tangent.
        // The tangent space at q consist of all matrices of the form q * (a skew matrix).
        // Hence we left-multiply by q^T and check whether the result is skew-symmetric.
        FieldMatrix<T,N,N> skewPart(0);
        for (int j=0; j<N; j++)
            for (int k=0; k<N; k++)
                for (int l=0; l<N; l++)
                    skewPart[j][k] += p.data()[l][j] * projected[l][k];
        
        // is it skew?
        for (int j=0; j<N; j++)
            for (int k=0; k<=j; k++)
                if ( std::abs(skewPart[j][k] + skewPart[k][j]) > eps ) {
                    
                    std::cout << "matrix is not skew-symmetric\n" << skewPart << std::endl;
                    abort();
                    
                }
    }
    
    
}

template <class T, int N>
void test()
{
    std::cout << "Testing class " << className<OrthogonalMatrix<T,N> >() << std::endl;
    
    // Get set of orthogonal test matrices
    std::vector<OrthogonalMatrix<T,N> > testPoints;
    ValueFactory<OrthogonalMatrix<T,N> >::get(testPoints);
    
    int nTestPoints = testPoints.size();
    
    // Test each element in the list
    for (int i=0; i<nTestPoints; i++) {
      
        // Test whether the matrix really is orthogonal
        testOrthogonality(testPoints[i]);
            
        testProjectionOntoTangentSpace(testPoints[i]);
        
    }
    
}


int main() try
{
    test<double,1>();
    test<double,2>();
    test<double,3>();
    
} catch (Exception& e) {

    std::cout << e.what() << std::endl;

}

