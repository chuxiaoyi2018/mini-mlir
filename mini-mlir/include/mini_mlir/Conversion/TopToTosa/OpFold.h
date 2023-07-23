#pragma once

#include "mini_mlir/Conversion/TopToTosa/TopLowering.h"

namespace mini_mlir {

void populateTosaFoldDoubleReciprocalPatterns(RewritePatternSet *patterns);

} // namespace mini_mlir
