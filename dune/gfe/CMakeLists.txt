#install headers
install(FILES gfe.hh DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/gfe)

add_subdirectory("spaces")
add_subdirectory("assemblers")
add_subdirectory("densities")