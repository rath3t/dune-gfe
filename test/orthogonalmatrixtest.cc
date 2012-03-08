#include <config.h>

#include <dune/gfe/orthogonalmatrix.hh>

#include "valuefactory.hh"


/** \file
    \brief Unit tests for the OrthogonalMatrix class
*/

using namespace Dune;

double eps = 1e-6;

/** \brief Test orthogonal projection onto the tangent space at q
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
        
        std::cout << "skew part\n" << skewPart << std::endl;
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
    
    // Use the ValueFactory for general matrices.
    // I am too lazy to program a factory for orthogonal matrices
    std::vector<FieldMatrix<T,N,N> > testPoints;
    ValueFactory<FieldMatrix<T,N,N> >::get(testPoints);
    
    int nTestPoints = testPoints.size();
    
    // Test each element in the list
    for (int i=0; i<nTestPoints; i++) {
        
        // I cannot use matrices with determinant zero for testing
        if (testPoints[i].determinant() > eps)
            testProjectionOntoTangentSpace(OrthogonalMatrix<T,N>(testPoints[i]));
        
    }
    
}


int main() try
{
    test<double,1>();
    test<double,2>();
    test<double,3>();
    
} catch (Exception e) {

    std::cout << e << std::endl;

}

