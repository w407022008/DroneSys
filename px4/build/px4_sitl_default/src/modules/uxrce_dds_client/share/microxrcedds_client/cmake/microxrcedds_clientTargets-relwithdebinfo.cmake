#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "microxrcedds_client" for configuration "RelWithDebInfo"
set_property(TARGET microxrcedds_client APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(microxrcedds_client PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/libmicroxrcedds_client.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS microxrcedds_client )
list(APPEND _IMPORT_CHECK_FILES_FOR_microxrcedds_client "${_IMPORT_PREFIX}/lib/libmicroxrcedds_client.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
