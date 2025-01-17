/** \file
    \brief Singular value decomposition code taken from 'Numerical Recipes in C'
 */
#ifndef SVD_HH
#define SVD_HH

#include <cmath>

#include <dune/common/fmatrix.hh>

// template <class T>
// int SIGN(const T& a, const T& b) {return 1;}
#define SIGN(a,b)((b)>=0.0 ? fabs(a) : -fabs(a))

namespace Dune::GFE
{

  /** Computes (a^2 + b^2 )1/2 without destructive underflow or overflow. */
  template <class T>
  T pythag(T a, T b)
  {
    T absa=std::fabs(a);
    T absb=std::fabs(b);
    if (absa > absb)
      return absa*std::sqrt(T(1.0)+(absb/absa)*(absb/absa));
    else
      return (absb == 0.0) ? T(0.0) : absb*sqrt(T(1.0)+(absa/absb)*(absa/absb));
  }


  /**
      Given a matrix a[1..m][1..n], this routine computes its singular value decomposition, A =
      U W V^T . The matrix U replaces a on output. The diagonal matrix of singular values W is out-
      put as a vector w[1..n]. The matrix V (not the transpose V T ) is output as v[1..n][1..n].
   */
  template <class T, int m, int n>
  void svdcmp(Dune::FieldMatrix<T,m,n>& a_, Dune::FieldVector<T,n>& w, Dune::FieldMatrix<T,n,n>& v_)
  {
    Dune::FieldMatrix<T,m+1,n+1> a(0);
    Dune::FieldMatrix<T,n+1,n+1> v(0);

    for (int i=0; i<m; i++)
      for (int j=0; j<n; j++)
        a[i+1][j+1] = a_[i][j];

    int flag,i,its,j,jj,k,l,nm;
    T anorm,c,f,g,h,s,scale,x,y,z;
    T rv1[n+1];  // 1 too large to accommodate fortran numbering

    //Householder reduction to bidiagonal form.
    g=scale=anorm=0.0;

    for (i=1; i<=n; i++) {

      l=i+1;
      rv1[i]=scale*g;
      g=s=scale=0.0;

      if (i <= m) {
        for (k=i; k<=m; k++)
          scale += std::abs(a[k][i]);
        if (scale!=0) {
          for (k=i; k<=m; k++) {
            a[k][i] /= scale;
            s += a[k][i]*a[k][i];
          }
          f=a[i][i];
          g = -SIGN(std::sqrt(s),f);
          h=f*g-s;
          a[i][i]=f-g;
          for (j=l; j<=n; j++) {
            for (s=0.0,k=i; k<=m; k++)
              s += a[k][i]*a[k][j];
            f=s/h;
            for (k=i; k<=m; k++)
              a[k][j] += f*a[k][i];
          }
          for (k=i; k<=m; k++)
            a[k][i] *= scale;
        }
      }

      w[i-1]=scale *g;
      g=s=scale=0.0;

      if (i <= m && i != n) {

        for (k=l; k<=n; k++) scale += fabs(a[i][k]);
        if (scale!=0) {

          for (k=l; k<=n; k++) {
            a[i][k] /= scale;
            s += a[i][k]*a[i][k];
          }
          f=a[i][l];
          g = -SIGN(std::sqrt(s),f);
          h=f*g-s;
          a[i][l]=f-g;
          for (k=l; k<=n; k++) rv1[k]=a[i][k]/h;
          for (j=l; j<=m; j++) {
            for (s=0.0,k=l; k<=n; k++)
              s += a[j][k]*a[i][k];
            for (k=l; k<=n; k++)
              a[j][k] += s*rv1[k];
          }
          for (k=l; k<=n; k++)
            a[i][k] *= scale;
        }
      }

      anorm = std::max(anorm,(std::abs(w[i-1])+std::abs(rv1[i])));

    }

    // Accumulation of right-hand transformations.
    for (i=n; i>=1; i--) {
      if (i < n) {
        if (g!=0) {
          // Double division to avoid possible underflow.
          for (j=l; j<=n; j++)
            v[j][i]=(a[i][j]/a[i][l])/g;
          for (j=l; j<=n; j++) {
            for (s=0.0,k=l; k<=n; k++) s += a[i][k]*v[k][j];
            for (k=l; k<=n; k++) v[k][j] += s*v[k][i];
          }
        }
        for (j=l; j<=n; j++) v[i][j]=v[j][i]=0.0;
      }
      v[i][i]=1.0;
      g=rv1[i];
      l=i;
    }

    // Accumulation of left-hand transformations.
    //for (i=IMIN(m,n);i>=1;i--) {
    for (i=std::min(m,n); i>=1; i--) {
      l=i+1;
      g=w[i-1];
      for (j=l; j<=n; j++) a[i][j]=0.0;
      if (g!=0) {
        g=1.0/g;
        for (j=l; j<=n; j++) {
          for (s=0.0,k=l; k<=m; k++) s += a[k][i]*a[k][j];
          f=(s/a[i][i])*g;
          for (k=i; k<=m; k++) a[k][j] += f*a[k][i];
        }
        for (j=i; j<=m; j++) a[j][i] *= g;
      } else for (j=i; j<=m; j++) a[j][i]=0.0;
      ++a[i][i];
    }

    // Diagonalization of the bidiagonal form: Loop over
    //singular values, and over allowed iterations.
    for (k=n; k>=1; k--) {

      for (its=1; its<=30; its++) {
        flag=1;
        // Test for splitting.
        for (l=k; l>=1; l--) {
          // Note that rv1[1] is always zero.
          nm=l-1;
          if ((T)(fabs(rv1[l])+anorm) == anorm) {
            flag=0;
            break;
          }
          if ((T)(fabs(w[nm-1])+anorm) == anorm) break;
        }
        if (flag) {
          // Cancellation of rv1[l], if l > 1.
          c=0.0;
          s=1.0;
          for (i=l; i<=k; i++) {

            f=s*rv1[i];
            rv1[i]=c*rv1[i];
            if ((T)(fabs(f)+anorm) == anorm) break;
            g=w[i-1];
            h=pythag(f,g);
            w[i-1]=h;
            h=1.0/h;
            c=g*h;
            s = -f*h;
            for (j=1; j<=m; j++) {
              y=a[j][nm];
              z=a[j][i];
              a[j][nm]=y*c+z*s;
              a[j][i]=z*c-y*s;
            }
          }
        }
        z=w[k-1];
        //Convergence.
        if (l == k) {
          // Singular value is made nonnegative.
          if (z < 0.0) {
            w[k-1] = -z;
            for (j=1; j<=n; j++) v[j][k] = -v[j][k];
          }
          break;
        }
        if (its == 30)
          std::cerr << "no convergence in 30 svdcmp iterations" << std::endl;
        // Shift from bottom 2-by-2 minor.
        x=w[l-1];
        nm=k-1;
        y=w[nm-1];
        g=rv1[nm];
        h=rv1[k];
        f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
        g=pythag(f,T(1.0));
        f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
        // Next QR transformation:
        c=s=1.0;
        for (j=l; j<=nm; j++) {
          i=j+1;
          g=rv1[i];
          y=w[i-1];
          h=s*g;
          g=c*g;
          z=pythag(f,h);
          rv1[j]=z;
          c=f/z;
          s=h/z;
          f=x*c+g*s;
          g = g*c-x*s;
          h=y*s;
          y *= c;
          for (jj=1; jj<=n; jj++) {
            x=v[jj][j];
            z=v[jj][i];
            v[jj][j]=x*c+z*s;
            v[jj][i]=z*c-x*s;
          }
          z=pythag(f,h);
          // Rotation can be arbitrary if z = 0.
          w[j-1]=z;
          if (z!=0) {
            z=1.0/z;
            c=f*z;
            s=h*z;
          }
          f=c*g+s*y;
          x=c*y-s*g;

          for (jj=1; jj<=m; jj++) {
            y=a[jj][j];
            z=a[jj][i];
            a[jj][j]=y*c+z*s;
            a[jj][i]=z*c-y*s;
          }
        }
        rv1[l]=0.0;
        rv1[k]=f;
        w[k-1]=x;
      }
    }

    for (int i=0; i<m; i++)
      for (int j=0; j<n; j++)
        a_[i][j] = a[i+1][j+1];

    for (int i=0; i<n; i++)
      for (int j=0; j<n; j++)
        v_[i][j] = v[i+1][j+1];
  }

}  // namespace Dune::GFE

#endif
