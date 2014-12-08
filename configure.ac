#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.50)
DUNE_AC_INIT
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([rod3d.cc])
AM_CONFIG_HEADER([config.h])

# we need no more than the standard DUNE-stuff
DUNE_CHECK_ALL

# implicitly set the Dune-flags everywhere
AC_SUBST(AM_CPPFLAGS, $DUNE_CPPFLAGS)
AC_SUBST(AM_LDFLAGS, $DUNE_LDFLAGS)
LIBS="$DUNE_LIBS"

AC_CONFIG_FILES([Makefile
                 doc/Makefile
                 doc/doxygen/Makefile
                 doc/doxygen/Doxyfile
                 dune/Makefile
                 dune/gfe/Makefile
                 dune/gfe/coupling/Makefile
                 src/Makefile
                 test/Makefile
                 m4/Makefile
                 dune-gfe.pc
                ])
AC_OUTPUT

# finally print the summary information
DUNE_SUMMARY_ALL
