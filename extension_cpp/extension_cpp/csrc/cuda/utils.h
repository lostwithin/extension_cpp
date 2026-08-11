#pragma once

#include <c10/util/Exception.h>

/// Check tensor dtype at runtime.
#define CHECK_TORCH_TENSOR_DTYPE(t, expected_dtype)                     \
  TORCH_CHECK((t).dtype() == (expected_dtype), "Expected dtype ", (expected_dtype), \
              " but got ", (t).dtype(), " for tensor ", #t)
