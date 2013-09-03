# $Id: Makefile.am 1867 2007-12-31 15:45:00Z sander@PCPOOL.MI.FU-BERLIN.DE $

# possible options
LDADD = $(UG_LDFLAGS) $(AMIRAMESH_LDFLAGS) $(UG_LIBS) $(AMIRAMESH_LIBS)
AM_CPPFLAGS += $(UG_CPPFLAGS) $(AMIRAMESH_CPPFLAGS) -Wall

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
                 orthogonalmatrixtest \
                 rodassemblertest \
                 rotationtest \
                 svdtest \
                 targetspacetest

adolctest_SOURCES = adolctest.cc
adolctest_LDADD = -ladolc

frameinvariancetest_SOURCES = frameinvariancetest.cc

rotationtest_SOURCES = rotationtest.cc

localgeodesicfefunctiontest_SOURCES = localgeodesicfefunctiontest.cc

localgeodesicfestiffnesstest_SOURCES = localgeodesicfestiffnesstest.cc

localgfetestfunctiontest_SOURCES = localgfetestfunctiontest.cc

nestednesstest_SOURCES = nestednesstest.cc

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
