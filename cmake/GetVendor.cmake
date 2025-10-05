# This function detects the CPU vendor and sets the result
# in an output variable in the parent scope.
#
# Usage:
#   detect_cpu_vendor(MY_VARIABLE_NAME)
#   message("CPU is: ${MY_VARIABLE_NAME}")
#
function(detect_cpu_vendor OUTPUT_VARIABLE)
    set(DETECTED_VENDOR "Unknown")

    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        execute_process(
            COMMAND lscpu
            OUTPUT_VARIABLE LSCPU_OUTPUT
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(LSCPU_OUTPUT MATCHES "GenuineIntel")
            set(DETECTED_VENDOR "Intel")
        elseif(LSCPU_OUTPUT MATCHES "AuthenticAMD")
            set(DETECTED_VENDOR "AMD")
        endif()

    elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
        execute_process(
            COMMAND wmic cpu get manufacturer
            OUTPUT_VARIABLE WMIC_OUTPUT
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(WMIC_OUTPUT MATCHES "GenuineIntel")
            set(DETECTED_VENDOR "Intel")
        elseif(WMIC_OUTPUT MATCHES "Advanced Micro Devices")
            set(DETECTED_VENDOR "AMD")
        endif()

    elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin") # macOS
        execute_process(
            COMMAND sysctl -n machdep.cpu.vendor
            OUTPUT_VARIABLE SYSCTL_OUTPUT
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(SYSCTL_OUTPUT MATCHES "GenuineIntel")
            set(DETECTED_VENDOR "Intel")
        elseif(SYSCTL_OUTPUT MATCHES "AuthenticAMD")
            set(DETECTED_VENDOR "AMD")
        endif()
    endif()

    # This is the key part: set the variable in the scope that CALLED the function.
    set(${OUTPUT_VARIABLE} ${DETECTED_VENDOR} PARENT_SCOPE)

endfunction()
