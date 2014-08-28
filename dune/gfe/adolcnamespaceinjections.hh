#ifndef DUNE_GFE_ADOLC_NAMESPACE_INJECTIONS_HH
#define DUNE_GFE_ADOLC_NAMESPACE_INJECTIONS_HH

#include <limits>

adouble sqrt_hack(adouble a) {
  return sqrt(a);
}

adouble abs_hack(adouble a) {
  return fabs(a);
}

adouble log_hack(adouble a) {
  return log(a);
}

adouble pow_hack(const adouble& a, const adouble& b) {
    return pow(a,b);
}

adouble pow_hack(const adouble& a, double b) {
    return pow(a,b);
}

adouble sin_hack(adouble a) {
  return sin(a);
}

adouble cos_hack(adouble a) {
  return cos(a);
}

adouble acos_hack(adouble a) {
  return acos(a);
}


namespace std
{
   adouble min(adouble a, adouble b) {
     // Do not use 'min', because ADOL-C lacks support for that in vector-mode
     return  (a + b - abs_hack(a - b))*0.5;
   }

   adouble max(adouble a, adouble b) {
     // Do not use 'max', because ADOL-C lacks support for that in vector-mode
     return - min(-a,-b);
   }

   adouble sqrt(adouble a) {
     return sqrt_hack(a);
   }

   adouble abs(adouble a) {
     return abs_hack(a);
   }

   adouble log(adouble a) {
     return log_hack(a);
   }

   adouble pow(const adouble& a, const adouble& b) {
     return pow_hack(a,b);
   }

   adouble pow(const adouble& a, double b) {
     return pow_hack(a,b);
   }

   adouble sin(adouble a) {
     return sin_hack(a);
   }

   adouble cos(adouble a) {
     return cos_hack(a);
   }

   adouble acos(adouble a) {
     return acos_hack(a);
   }

   bool isnan(adouble a) {
     return std::isnan(a.value());
   }

   bool isinf(adouble a) {
     return std::isinf(a.value());
   }

   /** \brief Specialization of the numeric_limits class for adouble */
   template <>
   struct numeric_limits<adouble> {

      static adouble max() {
        return numeric_limits<double>::max();
      }

      static adouble epsilon() {
        return numeric_limits<double>::epsilon();
      }
      static adouble quiet_NaN() {
        return numeric_limits<double>::quiet_NaN();
      }

      static constexpr bool is_integer = false;

   };
}

#endif