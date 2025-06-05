# FindXPrec.cmake
# Finds the XPrec library
#
# This will define the following variables:
#
#   XPrec_FOUND        - True if the system has XPrec
#   XPrec_VERSION      - The version of XPrec which was found
#   XPrec_INCLUDE_DIRS - XPrec include directory
#   XPrec_LIBRARIES    - XPrec libraries
#   XPrec_DIR          - The directory containing XPrecConfig.cmake

include(FindPackageHandleStandardArgs)

#message(STATUS "Searching for XPrec in:")
#message(STATUS "  XPREC_DIR = $ENV{XPREC_DIR}")
#message(STATUS "  XPREC_DIR = ${XPREC_DIR}")

# Try to find XPrec in standard locations and user-specified locations
find_path(XPrec_DIR
    NAMES XPrecConfig.cmake
    PATHS
        $ENV{XPREC_DIR}
        $HOME/opt/libxprec/share/cmake
        /usr/local/share/cmake
        /usr/share/cmake
    NO_DEFAULT_PATH
)

message(STATUS "XPrec_DIR = ${XPrec_DIR}")

if(XPrec_DIR)
    message(STATUS "Found XPrec in ${XPrec_DIR}")
    include(${XPrec_DIR}/XPrecConfig.cmake OPTIONAL)
    
    # Get the root directory of XPrec installation
    get_filename_component(XPrec_ROOT_DIR "${XPrec_DIR}/../.." ABSOLUTE)
    
    # Set include directories
    set(XPrec_INCLUDE_DIRS
        ${XPrec_ROOT_DIR}/include
        ${XPrec_ROOT_DIR}
    )
    
    # Set libraries
    if(TARGET XPrec::xprec)
        set(XPrec_LIBRARIES XPrec::xprec)
    endif()
    
    # Extract version from XPrecConfigVersion.cmake if available
    if(EXISTS "${XPrec_DIR}/XPrecConfigVersion.cmake")
        include("${XPrec_DIR}/XPrecConfigVersion.cmake")
        set(XPrec_VERSION ${PACKAGE_VERSION})
    endif()
    
    find_package_handle_standard_args(XPrec
        REQUIRED_VARS XPrec_DIR XPrec_INCLUDE_DIRS
        VERSION_VAR XPrec_VERSION
    )
else()
    message(STATUS "XPrec not found in specified paths")
    set(XPrec_FOUND FALSE)
endif()

mark_as_advanced(XPrec_DIR XPrec_INCLUDE_DIRS XPrec_LIBRARIES)

# Extract version from header if available
if(XPrec_INCLUDE_DIRS)
    message(STATUS "XPrec_INCLUDE_DIRS = ${XPrec_INCLUDE_DIRS}")
    find_file(XPREC_VERSION_HEADER
        NAMES version.h
        PATHS ${XPrec_INCLUDE_DIRS}
        PATH_SUFFIXES xprec
    )
    if(XPREC_VERSION_HEADER)
        message(STATUS "Found version header: ${XPREC_VERSION_HEADER}")
        file(STRINGS ${XPREC_VERSION_HEADER} XPREC_VERSION_LINE
            REGEX "^#define XPREC_VERSION_[A-Z]+ [0-9]+$")
        string(REGEX REPLACE "^#define XPREC_VERSION_MAJOR ([0-9]+).*" "\\1" XPREC_VERSION_MAJOR "${XPREC_VERSION_LINE}")
        string(REGEX REPLACE "^#define XPREC_VERSION_MINOR ([0-9]+).*" "\\1" XPREC_VERSION_MINOR "${XPREC_VERSION_LINE}")
        string(REGEX REPLACE "^#define XPREC_VERSION_PATCH ([0-9]+).*" "\\1" XPREC_VERSION_PATCH "${XPREC_VERSION_LINE}")
        set(XPrec_VERSION "${XPREC_VERSION_MAJOR}.${XPREC_VERSION_MINOR}.${XPREC_VERSION_PATCH}")
        message(STATUS "XPrec version: ${XPrec_VERSION}")
    endif()
endif() 