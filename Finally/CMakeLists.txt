cmake_minimum_required(VERSION 3.15)
project(GameEngine)

set(CMAKE_CXX_STANDARD 17)

# JSON 설정 파일 경로
set(CONFIG_FILE "${CMAKE_SOURCE_DIR}/config.json")

# JSON에서 실행 파일 이름 & 아이콘 경로 읽기
execute_process(
    COMMAND ${CMAKE_SOURCE_DIR}/Tools/json_reader ${CONFIG_FILE} exe_name
    OUTPUT_VARIABLE EXE_NAME OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${CMAKE_SOURCE_DIR}/Tools/json_reader ${CONFIG_FILE} icon_path
    OUTPUT_VARIABLE ICON_PATH OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 소스파일 설정
file(GLOB EngineSrc "Engine/src/*.cpp")

# 헤더파일 설정
include_directories(Engine/include)

# 엔진 실행 파일 생성
add_executable(${EXE_NAME} ${EngineSrc})

# Windows에서 아이콘 적용 (리소스 파일 생성)
if(WIN32)
    set(RC_FILE "${CMAKE_BINARY_DIR}/icon.rc")
    set(ICO_FILE "${CMAKE_SOURCE_DIR}/${ICON_PATH}")

    file(WRITE ${RC_FILE} "IDI_ICON1 ICON \"${ICO_FILE}\"\n")

    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/icon.res
        COMMAND windres ${RC_FILE} -O coff -o ${CMAKE_BINARY_DIR}/icon.res
        DEPENDS ${RC_FILE}
    )

    target_sources(${EXE_NAME} PRIVATE ${CMAKE_BINARY_DIR}/icon.res)
endif()

# 플러그인 자동 감지 및 추가
set(PLUGIN_DIR "${CMAKE_SOURCE_DIR}/Plugin")
file(GLOB PLUGIN_FOLDERS LIST_DIRECTORIES true ${PLUGIN_DIR}/*)

foreach(PLUGIN_PATH ${PLUGIN_FOLDERS})
    if (IS_DIRECTORY ${PLUGIN_PATH})
        get_filename_component(PLUGIN_NAME ${PLUGIN_PATH} NAME)

        # 헤더 추가
        set(PLUGIN_INC_DIR "${PLUGIN_PATH}/inc")
        if (EXISTS ${PLUGIN_INC_DIR})
            include_directories(${PLUGIN_INC_DIR})
            message(STATUS "Adding include directory: ${PLUGIN_INC_DIR}")
        endif()

        # 정적 라이브러리 추가
        set(PLUGIN_LIB_DIR "${PLUGIN_PATH}/lib/")
        file(GLOB PLUGIN_LIB_FILES "${PLUGIN_LIB_DIR}/*.lib")

        if (PLUGIN_LIB_FILES)
            message(STATUS "Linking static library: ${PLUGIN_LIB_FILES}")
            target_link_libraries(${EXE_NAME} PUBLIC ${PLUGIN_LIB_FILES})
        endif()

        # 동적 라이브러리 복사
        set(PLUGIN_DLL_DIR "${PLUGIN_PATH}/dll/")
        file(GLOB PLUGIN_DLL_FILES "${PLUGIN_DLL_DIR}/*.dll")

        if (PLUGIN_DLL_FILES)
            message(STATUS "Copying DLL files to output directory: ${PLUGIN_DLL_FILES}")
            foreach(DLL_FILE ${PLUGIN_DLL_FILES})
                add_custom_command(TARGET ${EXE_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${DLL_FILE} $<TARGET_FILE_DIR:${EXE_NAME}>
                )
            endforeach()
        endif()
    endif()
endforeach()

