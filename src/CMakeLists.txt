
add_executable("cosserat-continuum" cosserat-continuum.cc)
target_link_dune_default_libraries("cosserat-continuum")

if (ADOLC_FOUND)
    add_dune_adolc_flags(cosserat-continuum)
endif()

add_dune_pythonlibs_flags(cosserat-continuum)

add_dune_ipopt_flags(cosserat-continuum)

add_dune_ug_flags(cosserat-continuum)
