# This CMake file is part of PEST++

# pestpp-appl is run by the model's forward run (once per model run), not by the user as a
# manager, so it is kept self-contained: it uses no PEST++ library, only the version string
# from config_os.h, and so starts quickly.
add_executable(pestpp-appl pestpp-appl.cpp)

target_compile_options(pestpp-appl PRIVATE ${PESTPP_CXX_WARN_FLAGS})

target_include_directories(pestpp-appl PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../libs/common")

target_link_libraries(pestpp-appl Threads::Threads)

install(TARGETS pestpp-appl RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
