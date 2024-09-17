#install headers
install(FILES
        bulkcosseratdensity.hh
        chiralskyrmiondensity.hh
        cosseratshelldensity.hh
        cosseratvolumeloaddensity.hh
        duneelasticitydensity.hh
        harmonicdensity.hh
        localdensity.hh
        planarcosseratshelldensity.hh
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/gfe/densities)
