set(TESTS
  adolctest
  averagedistanceassemblertest
  cosseratenergytest
  frameinvariancetest
  globalgfetestfunctionbasistest
  harmonicenergytest
  interillustration
  localgeodesicfefunctiontest
  localgeodesicfestiffnesstest
  localgfetestfunctiontest
  localprojectedfefunctiontest
  nestednesstest
  orthogonalmatrixtest
  rodassemblertest
  rotationtest
  svdtest
  targetspacetest
)

add_directory_test_target(_test_target)
add_dependencies(${_test_target} ${TESTS})

foreach(_test ${TESTS})
  add_executable(${_test} ${_test}.cc)
  target_link_libraries(${_test} ${DUNE_LIBS})
  add_dune_ug_flags(${_test})
  add_test(${_test} ${_test})
endforeach()
