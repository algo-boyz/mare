cmake_minimum_required(VERSION 3.20)
project(edge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_TLS_VERIFY TRUE)

include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)

# sys
find_package(Threads REQUIRED)

find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs videoio highgui)
if(NOT OpenCV_FOUND)
  message(STATUS "OpenCV not found – edge_cv / edge_client will be skipped")
endif()

find_package(PkgConfig REQUIRED)

# grpc / protobuf
set(_use_grpc_config FALSE)
if(NOT APPLE)
  find_package(grpc CONFIG QUIET)
  if(gRPC_FOUND)
    set(_use_grpc_config TRUE)
  endif()
endif()

if(_use_grpc_config)
  set(_GRPC_GRPCPP grpc::grpc++)
  if(TARGET grpc::grpc_cpp_plugin)
    set(_GRPC_CPP_PLUGIN $<TARGET_FILE:grpc::grpc_cpp_plugin>)
  else()
    find_program(_GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)
  endif()
  if(NOT TARGET protobuf::libprotobuf)
    find_package(Protobuf REQUIRED)
  endif()
  message(STATUS "Using grpc (CONFIG)")
else()
  pkg_check_modules(PROTOBUF REQUIRED IMPORTED_TARGET protobuf)
  pkg_check_modules(GRPC REQUIRED IMPORTED_TARGET grpc++)
  if(NOT TARGET protobuf::libprotobuf)
    add_library(protobuf::libprotobuf ALIAS PkgConfig::PROTOBUF)
  endif()
  if(PROTOBUF_INCLUDE_DIRS)
    set(Protobuf_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIRS})
  else()
    execute_process(COMMAND brew --prefix protobuf
      OUTPUT_VARIABLE _PB_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(_PB_PREFIX)
      set(Protobuf_INCLUDE_DIRS "${_PB_PREFIX}/include")
    endif()
  endif()
  set(_GRPC_GRPCPP PkgConfig::GRPC)
  find_program(_GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)
  find_program(_PROTOBUF_PROTOC protoc REQUIRED)
  message(STATUS "Using grpc + Protobuf via pkg-config")
  message(STATUS "  Protobuf includes: ${Protobuf_INCLUDE_DIRS}")
endif()

if(NOT DEFINED _PROTOBUF_PROTOC)
  if(TARGET protobuf::protoc)
    set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
  else()
    find_program(_PROTOBUF_PROTOC protoc REQUIRED)
  endif()
endif()

# onnx runtime
if(NOT DEFINED ONNXRUNTIME_ROOT)
  if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
  endif()
endif()

if(ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT}/include)
  find_library(ONNXRUNTIME_LIB onnxruntime
    PATHS ${ONNXRUNTIME_ROOT}/lib ${ONNXRUNTIME_ROOT}/lib64
    NO_DEFAULT_PATH)
else()
  find_path(ONNXRUNTIME_INCLUDE_DIRS onnxruntime_cxx_api.h QUIET)
  find_library(ONNXRUNTIME_LIB onnxruntime QUIET)
endif()

if(NOT ONNXRUNTIME_LIB)
  message(STATUS "ONNX Runtime not found – edge_cv client will not build")
endif()

# telemetry
option(WITH_OTEL "Enable OpenTelemetry structured logging via spdlog sink" OFF)
if(WITH_OTEL)
  message(STATUS "OpenTelemetry support: ON")

  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(WITH_OTLP_HTTP OFF CACHE BOOL "" FORCE)
  set(WITH_OTLP_GRPC ON CACHE BOOL "" FORCE)
  set(WITH_LOGS_PREVIEW ON CACHE BOOL "" FORCE)
  set(WITH_STL ON CACHE BOOL "" FORCE)

  FetchContent_Declare(opentelemetry-cpp
    GIT_REPOSITORY https://github.com/open-telemetry/opentelemetry-cpp.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(opentelemetry-cpp)
else()
  message(STATUS "OpenTelemetry support: OFF (enable with -DWITH_OTEL=ON)")
endif()

# spdlog
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
  FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.14.1 GIT_SHALLOW TRUE)
  set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
  set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(spdlog)
endif()

# nats
find_package(NATS QUIET)
if(NOT NATS_FOUND)
  FetchContent_Declare(nats
    GIT_REPOSITORY https://github.com/nats-io/nats.c.git
    GIT_TAG v3.9.1 GIT_SHALLOW TRUE)
  set(NATS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(NATS_BUILD_STREAMING OFF CACHE BOOL "" FORCE)
  set(NATS_BUILD_LIB_SHARED OFF CACHE BOOL "" FORCE)
  set(NATS_BUILD_LIB_STATIC ON CACHE BOOL "" FORCE)
  set(NATS_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(nats)
  set(NATS_LIBRARIES nats_static)
  set(NATS_INCLUDE_DIRS ${nats_SOURCE_DIR}/src)
endif()

# proto compile
set(PROTO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

set(PROTO_FILES
  ${PROTO_ROOT}/audio/v1/transcript.proto
  ${PROTO_ROOT}/audio/v1/live_service.proto
  ${PROTO_ROOT}/common/v1/error.proto
  ${PROTO_ROOT}/detection/v1/detection.proto
  ${PROTO_ROOT}/detection/v1/annotated_video.proto
  ${PROTO_ROOT}/ingest/v1/ingest_service.proto
  ${PROTO_ROOT}/nats/v1/publisher_service.proto
)

set(PROTO_SRCS)
set(PROTO_HDRS)
set(GRPC_SRCS)
set(GRPC_HDRS)

foreach(proto ${PROTO_FILES})
  get_filename_component(proto_name ${proto} NAME_WE)
  get_filename_component(proto_dir ${proto} DIRECTORY)
  file(RELATIVE_PATH rel_dir ${PROTO_ROOT} ${proto_dir})

  set(pb_cc   ${PROTO_GEN_DIR}/${rel_dir}/${proto_name}.pb.cc)
  set(pb_h    ${PROTO_GEN_DIR}/${rel_dir}/${proto_name}.pb.h)
  set(grpc_cc ${PROTO_GEN_DIR}/${rel_dir}/${proto_name}.grpc.pb.cc)
  set(grpc_h  ${PROTO_GEN_DIR}/${rel_dir}/${proto_name}.grpc.pb.h)

  add_custom_command(
    OUTPUT ${pb_cc} ${pb_h} ${grpc_cc} ${grpc_h}
    COMMAND ${_PROTOBUF_PROTOC}
      --proto_path=${PROTO_ROOT}
      --proto_path=${Protobuf_INCLUDE_DIRS}
      --cpp_out=${PROTO_GEN_DIR}
      --grpc_out=${PROTO_GEN_DIR}
      --plugin=protoc-gen-grpc=${_GRPC_CPP_PLUGIN}
      ${proto}
    DEPENDS ${proto}
    COMMENT "Generating ${rel_dir}/${proto_name}"
  )
  list(APPEND PROTO_SRCS ${pb_cc})
  list(APPEND PROTO_HDRS ${pb_h})
  list(APPEND GRPC_SRCS ${grpc_cc})
  list(APPEND GRPC_HDRS ${grpc_h})
endforeach()

add_library(edge_proto ${PROTO_SRCS} ${GRPC_SRCS})
target_include_directories(edge_proto PUBLIC
  ${PROTO_GEN_DIR}
  ${Protobuf_INCLUDE_DIRS}
)
target_link_libraries(edge_proto PUBLIC
  protobuf::libprotobuf
  ${_GRPC_GRPCPP}
)

# common
add_library(edge_common STATIC
  src/common/otel.cpp
)

target_include_directories(edge_common PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${NATS_INCLUDE_DIRS}
)

target_link_libraries(edge_common PUBLIC
  edge_proto
  spdlog::spdlog
  ${NATS_LIBRARIES}
  Threads::Threads
)

if(WITH_OTEL)
  target_compile_definitions(edge_common PUBLIC WITH_OTEL)
  target_link_libraries(edge_common PUBLIC
    opentelemetry_trace
    opentelemetry_logs
    opentelemetry_exporter_otlp_grpc_log
  )
endif()

# sherpa-onnx (STT) + PortAudio (mic capture)
find_path(SHERPA_ONNX_INCLUDE_DIR
  NAMES sherpa-onnx/c-api/cxx-api.h
  PATHS
    /usr/local/include
    /usr/include
    $ENV{SHERPA_ONNX_ROOT}/include
)

find_library(SHERPA_ONNX_LIB
  NAMES sherpa-onnx-c-api sherpa-onnx
  PATHS
    /usr/local/lib
    /usr/lib
    $ENV{SHERPA_ONNX_ROOT}/lib
)

if(NOT SHERPA_ONNX_INCLUDE_DIR OR NOT SHERPA_ONNX_LIB)
  message(STATUS "sherpa-onnx not found on system – using FetchContent")
  include(FetchContent)
  FetchContent_Declare(
    sherpa-onnx
    GIT_REPOSITORY https://github.com/k2-fsa/sherpa-onnx.git
    GIT_TAG        v1.10.46
    GIT_SHALLOW    TRUE
  )
  set(SHERPA_ONNX_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
  set(SHERPA_ONNX_ENABLE_TESTS  OFF CACHE BOOL "" FORCE)
  set(SHERPA_ONNX_ENABLE_C_API  ON  CACHE BOOL "" FORCE)
  set(SHERPA_ONNX_ENABLE_BINARY OFF CACHE BOOL "" FORCE)
  set(SHERPA_ONNX_ENABLE_CHECK  OFF CACHE BOOL "" FORCE)
  set(SHERPA_ONNX_ENABLE_PORTAUDIO OFF CACHE BOOL "" FORCE)
  set(SHERPA_ONNX_ENABLE_WEBSOCKET OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(sherpa-onnx)
  set(SHERPA_ONNX_INCLUDE_DIR ${sherpa-onnx_SOURCE_DIR})
  set(SHERPA_ONNX_LIB sherpa-onnx-c-api)
endif()

# PortAudio
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PORTAUDIO portaudio-2.0)
endif()
if(NOT PORTAUDIO_FOUND)
  find_path(PORTAUDIO_INCLUDE_DIR portaudio.h)
  find_library(PORTAUDIO_LIBRARY NAMES portaudio)
  if(PORTAUDIO_INCLUDE_DIR AND PORTAUDIO_LIBRARY)
    set(PORTAUDIO_FOUND TRUE)
    set(PORTAUDIO_INCLUDE_DIRS ${PORTAUDIO_INCLUDE_DIR})
    set(PORTAUDIO_LIBRARIES ${PORTAUDIO_LIBRARY})
  endif()
endif()

if(NOT PORTAUDIO_FOUND)
  message(STATUS "PortAudio not found – edge_audio / edge_audio_client will be skipped")
  message(STATUS "  Ubuntu: sudo apt install libportaudio2 portaudio19-dev")
  message(STATUS "  macOS:  brew install portaudio")
endif()

# edge_audio lib + edge_audio_client
if(SHERPA_ONNX_LIB AND PORTAUDIO_FOUND)
  add_library(edge_audio STATIC
    src/audio/src/capture.cpp
    src/audio/src/pipeline.cpp
    src/audio/src/transcriber.cpp
  )
  target_include_directories(edge_audio PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/audio/include
    ${SHERPA_ONNX_INCLUDE_DIR}
    ${PORTAUDIO_INCLUDE_DIRS}
  )
  target_link_libraries(edge_audio PUBLIC
    ${SHERPA_ONNX_LIB}
    ${PORTAUDIO_LIBRARIES}
    Threads::Threads
  )

  add_executable(edge_audio_client
    src/audio_client/main.cpp
    src/audio_client/live_service.cpp
  )
  target_include_directories(edge_audio_client PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/audio_client
  )
  target_link_libraries(edge_audio_client PRIVATE
    edge_audio
    edge_common
    edge_proto
  )

  if(UNIX AND NOT APPLE)
    set_target_properties(edge_audio_client PROPERTIES
      BUILD_RPATH   "$ORIGIN"
      INSTALL_RPATH "$ORIGIN"
    )
  endif()

  message(STATUS "  edge_audio / edge_audio_client: ON")
  message(STATUS "    sherpa-onnx: ${SHERPA_ONNX_INCLUDE_DIR}")
  message(STATUS "    PortAudio:   ${PORTAUDIO_LIBRARIES}")
else()
  message(STATUS "  edge_audio / edge_audio_client: OFF (needs sherpa-onnx + PortAudio)")
endif()

if(OpenCV_FOUND AND ONNXRUNTIME_LIB)
  add_library(edge_cv
    src/cv/src/capture.cpp
    src/cv/src/detector.cpp
    src/cv/src/pipeline.cpp
    src/cv/src/postprocess.cpp
  )
  target_include_directories(edge_cv PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cv/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
  )
  target_link_libraries(edge_cv PUBLIC
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    Threads::Threads
  )

  add_executable(edge_client src/client/main.cpp)
  target_link_libraries(edge_client PRIVATE
    edge_cv edge_common edge_proto
  )
else()
  message(STATUS "OpenCV or ONNX Runtime missing - edge_cv and edge_client will NOT be built")
endif()

# apps
add_executable(nats_publisher
  src/nats_publisher/main.cpp
  src/nats_publisher/publisher_service.cpp
)
target_link_libraries(nats_publisher PRIVATE edge_common)

add_executable(ingest_server
  src/ingest/main.cpp
  src/ingest/ingest_service.cpp
)
target_link_libraries(ingest_server PRIVATE edge_common)

add_executable(consumer src/consumer/main.cpp)
target_link_libraries(consumer PRIVATE edge_common)

# clickhouse_consumer
option(BUILD_CLICKHOUSE_CONSUMER "Build clickhouse_consumer binary" ON)
if(BUILD_CLICKHOUSE_CONSUMER)
  set(WITH_OPENSSL ON CACHE BOOL "" FORCE)
  set(CH_USE_ABSEIL_FOR_BIGNUM OFF CACHE BOOL "" FORCE)
  set(CH_MAP_BOOL_TO_UINT8 OFF CACHE BOOL "" FORCE)
  set(CLICKHOUSE_CLIENT_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(CLICKHOUSE_CLIENT_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(WITH_SYSTEM_ABSEIL ON CACHE BOOL "" FORCE)
  set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)

  FetchContent_Declare(clickhouse-cpp
    GIT_REPOSITORY https://github.com/ClickHouse/clickhouse-cpp.git
    GIT_TAG v2.6.2
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(clickhouse-cpp)

  add_executable(clickhouse_consumer src/clickhouse_consumer/main.cpp)
  target_link_libraries(clickhouse_consumer PRIVATE
    edge_common
    clickhouse-cpp-lib
  )
  message(STATUS "  clickhouse_consumer: ON")
else()
  message(STATUS "  clickhouse_consumer: OFF")
endif()

message(STATUS "edge configured (C++${CMAKE_CXX_STANDARD})")
message(STATUS "  OpenCV: ${OpenCV_VERSION}")
if(ONNXRUNTIME_LIB)
  message(STATUS "  ONNX Runtime: ${ONNXRUNTIME_LIB}")
else()
  message(STATUS "  ONNX Runtime: Disabled")
endif()
if(TARGET edge_audio)
  message(STATUS "  Audio (sherpa + PortAudio): ON")
else()
  message(STATUS "  Audio (sherpa + PortAudio): OFF")
endif()

# video_server (needs OpenCV + clickhouse-cpp)
if(OpenCV_FOUND AND BUILD_CLICKHOUSE_CONSUMER)
  add_executable(video_server
    src/video_server/main.cpp
    src/video_server/video_service.cpp
  )
  target_include_directories(video_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
  )
  target_compile_definitions(video_server PRIVATE
    ABSL_ATTRIBUTE_PURE_FUNCTION=
    ABSL_ATTRIBUTE_CONST_FUNCTION=
    ABSL_ATTRIBUTE_WEAK=
    ABSL_HAVE_ATTRIBUTE_WEAK=0
  )
  target_link_libraries(video_server PRIVATE
    edge_common
    edge_proto
    clickhouse-cpp-lib
    ${OpenCV_LIBS}
    grpc++_reflection
  )
  # Ubuntu linkers drop unused shared libs so force keep reflection
  target_link_options(video_server PRIVATE "LINKER:--no-as-needed")
  message(STATUS "  video_server: ON")
else()
  message(STATUS "  video_server: OFF (needs OpenCV + BUILD_CLICKHOUSE_CONSUMER)")
endif()