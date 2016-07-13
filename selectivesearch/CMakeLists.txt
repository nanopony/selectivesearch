cmake_minimum_required(VERSION 2.6)
project(selectivesearch)

if (CMAKE_VERSION VERSION_LESS "3.1")
if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
  set (CMAKE_C_FLAGS "--std=gnu99 ${CMAKE_C_FLAGS}")
endif ()
else ()
    set (CMAKE_C_STANDARD 99)
endif ()

find_package( OpenCV REQUIRED )
FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs)

INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

set(CFLAGS "-O0 -g -std=c99")
set(CXXFLAGS "-O0 -g")

add_library(_selectivesearch SHARED _selectivesearch.cpp selectivesearch.cpp)
set_property(TARGET _selectivesearch  PROPERTY C_STANDARD 99)
set_property(TARGET _selectivesearch PROPERTY PREFIX "")

target_link_libraries( _selectivesearch ${OpenCV_LIBS} )

#install(TARGETS selectivesearch RUNTIME DESTINATION bin)
