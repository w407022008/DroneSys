
cmake_minimum_required(VERSION 3.15)

set(command "/usr/bin/cmake;--build;.;--target;install")
set(log_merged "")
set(log_output_on_failure "true")
set(stdout_log "/home/sique/src/PX4_v1.14.2/build/px4_sitl_default/src/modules/uxrce_dds_client/src/libmicroxrceddsclient_project-stamp/libmicroxrceddsclient_project-install-out.log")
set(stderr_log "/home/sique/src/PX4_v1.14.2/build/px4_sitl_default/src/modules/uxrce_dds_client/src/libmicroxrceddsclient_project-stamp/libmicroxrceddsclient_project-install-err.log")
execute_process(
  COMMAND ${command}
  RESULT_VARIABLE result
  OUTPUT_FILE "${stdout_log}"
  ERROR_FILE "${stderr_log}"
  )
macro(read_up_to_max_size log_file output_var)
  file(SIZE ${log_file} determined_size)
  set(max_size 10240)
  if (determined_size GREATER max_size)
    math(EXPR seek_position "${determined_size} - ${max_size}")
    file(READ ${log_file} ${output_var} OFFSET ${seek_position})
    set(${output_var} "...skipping to end...\n${${output_var}}")
  else()
    file(READ ${log_file} ${output_var})
  endif()
endmacro()
if(result)
  set(msg "Command failed: ${result}\n")
  foreach(arg IN LISTS command)
    set(msg "${msg} '${arg}'")
  endforeach()
  if (${log_merged})
    set(msg "${msg}\nSee also\n  ${stderr_log}")
  else()
    set(msg "${msg}\nSee also\n  /home/sique/src/PX4_v1.14.2/build/px4_sitl_default/src/modules/uxrce_dds_client/src/libmicroxrceddsclient_project-stamp/libmicroxrceddsclient_project-install-*.log")
  endif()
  if (${log_output_on_failure})
    message(SEND_ERROR "${msg}")
    if (${log_merged})
      read_up_to_max_size("${stderr_log}" error_log_contents)
      message(STATUS "Log output is:\n${error_log_contents}")
    else()
      read_up_to_max_size("${stdout_log}" out_log_contents)
      read_up_to_max_size("${stderr_log}" err_log_contents)
      message(STATUS "stdout output is:\n${out_log_contents}")
      message(STATUS "stderr output is:\n${err_log_contents}")
    endif()
    message(FATAL_ERROR "Stopping after outputting logs.")
  else()
    message(FATAL_ERROR "${msg}")
  endif()
else()
  set(msg "libmicroxrceddsclient_project install command succeeded.  See also /home/sique/src/PX4_v1.14.2/build/px4_sitl_default/src/modules/uxrce_dds_client/src/libmicroxrceddsclient_project-stamp/libmicroxrceddsclient_project-install-*.log")
  message(STATUS "${msg}")
endif()
