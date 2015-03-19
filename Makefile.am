# $Id$

# we need the module file to be able to build via dunecontrol
EXTRA_DIST=dune.module

SUBDIRS = doc m4 dune src test

# possible options
LDADD = $(IPOPT_LDFLAGS) $(IPOPT_LIBS)
AM_CPPFLAGS += $(IPOPT_CPPFLAGS)

ADOLC_CPPFLAGS = -I/home/sander/adolc-inst/include
ADOLC_LDFLAGS = -L/home/sander/adolc-inst/lib64
ADOLC_LIBS = -ladolc

noinst_PROGRAMS = finite-strain-elasticity \
                  mixed-cosserat-continuum \
                  rodobstacle rod-eoc

finite_strain_elasticity_SOURCES  = finite-strain-elasticity.cc
finite_strain_elasticity_CXXFLAGS = $(UG_CPPFLAGS) $(IPOPT_CPPFLAGS) \
                              $(ADOLC_CPPFLAGS) $(PYTHON_CPPFLAGS) $(UMFPACK_CPPFLAGS)
finite_strain_elasticity_LDADD    = $(UG_LIBS) $(IPOPT_LIBS) \
                              $(ADOLC_LIBS) $(PYTHON_LIBS) $(UMFPACK_LIBS)
finite_strain_elasticity_LDFLAGS  = $(UG_LDFLAGS) $(IPOPT_LDFLAGS) \
                              $(ADOLC_LDFLAGS) $(PYTHON_LDFLAGS) $(UMFPACK_LDFLAGS)

mixed_cosserat_continuum_SOURCES  = mixed-cosserat-continuum.cc
mixed_cosserat_continuum_CXXFLAGS = $(UG_CPPFLAGS) $(IPOPT_CPPFLAGS) \
                              $(ADOLC_CPPFLAGS) $(PYTHON_CPPFLAGS)
mixed_cosserat_continuum_LDADD    = $(UG_LIBS) $(IPOPT_LIBS) \
                              $(ADOLC_LIBS) $(PYTHON_LIBS)
mixed_cosserat_continuum_LDFLAGS  = $(UG_LDFLAGS) $(IPOPT_LDFLAGS) \
                              $(ADOLC_LDFLAGS) $(PYTHON_LDFLAGS)

rodobstacle_SOURCES = rodobstacle.cc

rod_eoc_SOURCES = rod-eoc.cc

# don't follow the full GNU-standard
# we need automake 1.5
AUTOMAKE_OPTIONS = foreign 1.5

include $(top_srcdir)/am/global-rules
include $(top_srcdir)/am/top-rules
