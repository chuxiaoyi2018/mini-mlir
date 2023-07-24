file(GLOB _sources *.cpp)

    add_mlir_conversion_library(
        MiniTopToTosa ${_sources}

        ADDITIONAL_HEADER_DIRS ${PROJECT_SOURCE_DIR} /
        include / mini_mlir / Dialect / Top ${PROJECT_SOURCE_DIR} /
        third_party / llvm / include / mlir / Dialect /
        Tosa

            DEPENDS MiniMlirConversionPassIncGen

                LINK_LIBS PUBLIC MiniTop MLIRTosaDialect)
