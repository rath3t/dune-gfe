
SUBDIRS =

noinst_PROGRAMS = rod3d

rod3d_SOURCES = rod3d.cc
rod3d_CXXFLAGS = $(IPOPT_CPPFLAGS)
rod3d_LDADD    = $(IPOPT_LIBS)
rod3d_LDFLAGS  = $(IPOPT_LDFLAGS)

# we need automake 1.5
AUTOMAKE_OPTIONS = foreign 1.5

# pass most important options when "make distcheck" is used
DISTCHECK_CONFIGURE_FLAGS = --with-dune-common=$(DUNE_COMMON_ROOT) --with-dune-geometry=$(DUNE_GEOMETRY_ROOT) --with-dune-grid=$(DUNE_GRID_ROOT) --with-dune-istl=$(DUNE_ISTL_ROOT) --with-dune-localfunctions=$(DUNE_LOCALFUNCTIONS_ROOT)  CXX="$(CXX)" CC="$(CC)"

include $(top_srcdir)/am/global-rules

