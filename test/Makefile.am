# $Id: Makefile.am 1867 2007-12-31 15:45:00Z sander@PCPOOL.MI.FU-BERLIN.DE $

# ADOL-C while it doesn't support pkg-config yet
# ADOLC_CPPFLAGS = -I/usr/include
# ADOLC_LDFLAGS = -L/usr/lib
ADOLC_CPPFLAGS = -I/home/sander/adolc-inst/include
ADOLC_LDFLAGS = -L/home/sander/adolc-inst/lib64
ADOLC_LIBS = -ladolc


# possible options
LDADD =  $(UG_LDFLAGS) $(AMIRAMESH_LDFLAGS) $(UG_LIBS) $(AMIRAMESH_LIBS) $(IPOPT_LDFLAGS) $(IPOPT_LIBS)
AM_CPPFLAGS += $(ADOLC_CPPFLAGS) $(UG_CPPFLAGS) $(AMIRAMESH_CPPFLAGS) -Wall $(IPOPT_CPPFLAGS) $(PYTHON_CPPFLAGS)

check_PROGRAMS = adolctest \
                 averagedistanceassemblertest \
                 cosseratenergytest \
                 frameinvariancetest \
                 globalgfetestfunctionbasistest \
                 harmonicenergytest \
                 interpolationtest \
                 localgeodesicfefunctiontest \
                 localgeodesicfestiffnesstest \
                 localgfetestfunctiontest \
                 nestednesstest \
                 nonconvexitytest \
                 nonconvexitytest_simple \
                 orthogonalmatrixtest \
                 rodassemblertest \
                 rotationtest \
                 svdtest \
                 targetspacetest \
                 true-adolctest

adolctest_SOURCES = adolctest.cc
adolctest_LDFLAGS = $(ADOLC_LDFLAGS) $(PYTHON_LDFLAGS) $(IPOPT_LDFLAGS)
adolctest_LDADD = $(ADOLC_LIBS) $(PYTHON_LIBS) $(IPOPT_LIBS)

true_adolctest_SOURCES = true-adolctest.cc
true_adolctest_LDFLAGS = $(ADOLC_LDFLAGS) $(PYTHON_LDFLAGS) $(IPOPT_LDFLAGS)
true_adolctest_LDADD = $(ADOLC_LIBS) $(PYTHON_LIBS) $(IPOPT_LIBS)

frameinvariancetest_SOURCES = frameinvariancetest.cc

rotationtest_SOURCES = rotationtest.cc

localgeodesicfefunctiontest_SOURCES = localgeodesicfefunctiontest.cc

localgeodesicfestiffnesstest_SOURCES = localgeodesicfestiffnesstest.cc

localgfetestfunctiontest_SOURCES = localgfetestfunctiontest.cc

nestednesstest_SOURCES = nestednesstest.cc

nonconvexitytest_SOURCES = nonconvexitytest.cc
nonconvexitytest_LDADD = -ladolc

nonconvexitytest_simple_SOURCES = nonconvexitytest_simple.cc
nonconvexitytest_simple_LDADD = -ladolc

globalgfetestfunctionbasistest_SOURCES = globalgfetestfunctionbasistest.cc

harmonicenergytest_SOURCES = harmonicenergytest.cc

interpolationtest_SOURCES = interpolationtest.cc

cosseratenergytest_SOURCES = cosseratenergytest.cc

averagedistanceassemblertest_SOURCES = averagedistanceassemblertest.cc

orthogonalmatrixtest_SOURCES = orthogonalmatrixtest.cc

rodassemblertest_SOURCES = rodassemblertest.cc

targetspacetest_SOURCES = targetspacetest.cc

svdtest_SOURCES = svdtest.cc

# don't follow the full GNU-standard
# we need automake 1.5
AUTOMAKE_OPTIONS = foreign 1.5
