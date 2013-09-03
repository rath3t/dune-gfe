#ifndef DUNE_GFE_ADOLC_NAMESPACE_INJECTIONS_HH
#define DUNE_GFE_ADOLC_NAMESPACE_INJECTIONS_HH

adouble sqrt_hack(adouble a) {
  return sqrt(a);
}

adouble pow_hack(const adouble& a, const adouble& b) {
    return pow(a,b);
}

adouble pow_hack(const adouble& a, double b) {
    return pow(a,b);
}

namespace std
{
   adouble min(adouble a, adouble b) {
      return fmin(a,b);
   }

   adouble max(adouble a, adouble b) {
      return fmax(a,b);
   }

   adouble sqrt(adouble a) {
     return sqrt_hack(a);
   }

   adouble abs(adouble a) {
     return fabs(a);
   }

//    adouble fabs(adouble a) {
//      return fabs(a);
//    }

   adouble pow(const adouble& a, const adouble& b) {
     return pow_hack(a,b);
   }

   adouble pow(const adouble& a, double b) {
     return pow_hack(a,b);
   }

   adouble sin(adouble a) {
     return sin(a);
   }

   adouble cos(adouble a) {
     return cos(a);
   }

   adouble acos(adouble a) {
     return acos(a);
   }

   bool isnan(adouble a) {
     return std::isnan(a.value());
   }

   bool isinf(adouble a) {
     return std::isinf(a.value());
   }

}

#endif