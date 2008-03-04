#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.50)
AC_INIT(rods, 1.0, sander@mi.fu-berlin.de)
AM_INIT_AUTOMAKE(rods, 1.0, sander@mi.fu-berlin.de)
AC_CONFIG_SRCDIR([staticrod.cc])
AM_CONFIG_HEADER([config.h])

# we need no more than the standard DUNE-stuff
DUNE_CHECK_ALL_M([dune-common], [dune-grid], [dune-istl], [dune-disc])

# implicitly set the Dune-flags everywhere
AC_SUBST(AM_CPPFLAGS, $DUNE_CPPFLAGS)
AC_SUBST(AM_LDFLAGS, $DUNE_LDFLAGS)
LIBS="$DUNE_LIBS"

AC_CONFIG_FILES([Makefile
                 test/Makefile])
AC_OUTPUT

# finally print the summary information
DUNE_SUMMARY_ALL
