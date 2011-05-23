#include <iostream>

#include <dune/gfe/svd.hh>

#include "multiindex.hh"

using namespace Dune;

int main()
{
    const int N=3;
    const int M=3;
    
    FieldMatrix<double,N,M> testMatrix;

    int nTestValues = 5;
    double maxDiff = 0;
    
    MultiIndex<N*M> index(nTestValues);
    int numIndices = index.cycle();
    
    for (int i=0; i<numIndices; i++, ++index) {
        
        for (int j=0; j<N; j++)
            for (int k=0; k<M; k++) {
                testMatrix[j][k] = std::exp(double(2*index[j*M+k]-double(nTestValues))) - std::exp(double(nTestValues));
                //std::cout << std::exp(2*index[j*M+k]-double(nTestValues)) << std::endl;
            }
        //std::cout << "testMatrix:\n" << testMatrix << std::endl;
        //exit(0);
        FieldVector<double,N> w;
        FieldMatrix<double,N,N> v;
        
        FieldMatrix<double,N,M> testMatrixBackup = testMatrix;
        svdcmp(testMatrix,w,v);
        
        // Multiply the three matrices to see whether we get the original one back
        FieldMatrix<double,N,M> product(0);
        
        for (int j=0; j<N; j++)
            for (int k=0; k<M; k++)
                for (int l=0; l<N; l++)
                    product[j][k] += testMatrix[j][l] * w[l] * v[k][l];
        
        product -= testMatrixBackup;
        //std::cout << product << std::endl;
        if (maxDiff < product.infinity_norm()) {
            maxDiff = product.infinity_norm();
            std::cout << "max diff: " << maxDiff << std::endl;
            std::cout << "testMatrix:\n" << testMatrixBackup << std::endl;
            std::cout << "difference:\n" << product << std::endl;
            exit(0);
        }
    
    }
    
}