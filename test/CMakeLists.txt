set(TESTS
  adolctest
  averagedistanceassemblertest
  cosseratenergytest
  frameinvariancetest
  harmonicenergytest
  localgeodesicfefunctiontest
  localgfetestfunctiontest
  localprojectedfefunctiontest
  orthogonalmatrixtest
  rigidbodymotiontest
  rodassemblertest
  rotationtest
  svdtest
  targetspacetest
)

foreach(_test ${TESTS})
  dune_add_test(SOURCES ${_test}.cc)
  target_compile_options(${_test} PRIVATE -g)
endforeach()
