/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_TESTS_LITERAL_TEST_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_LITERAL_TEST_UTIL_H_
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTh {
public:
   std::string _s;
   int _indent = 0;
   std::string _functionName;
   bool _isFile = false;
   std::string _fileName;
   std::string _envMHIndent;
   int _lineNumber;
   bool _filtered = false;
   bool _otherThread = false;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
      _functionName = functionName;
      _lineNumber = lineNumber;

      // Check if tracing is enabled
      const char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }
      // Should we trace of filter?
      const char* env_filter = std::getenv("MHTRACER_FILTER");
      if (env_filter != nullptr) {
         std::string sfilter = std::string(env_filter);
         std::string sLineNumber = std::to_string(lineNumber);
         while (true) {
            std::size_t ioE = sfilter.find(";");
            if (sfilter.size() == 0) {
               break;
            }
            std::string cfs = sfilter.substr(0, ioE);
            std::size_t ioFileName = cfs.find("|");
            std::string fFileName  = cfs.substr(0, ioFileName);
            std::size_t ioFunctionName = cfs.find("|", ioFileName+1);
            std::string fFunctionName  = cfs.substr(ioFileName+1, ioFunctionName-ioFileName-1);
            std::string fLineNumber    = cfs.substr(ioFunctionName+1, cfs.size()-ioFunctionName-1);

            if (  (fFileName == "*" || fFileName == fileName)
               && (fFunctionName == "*" || fFunctionName == functionName)
               && (fLineNumber == "*" || fLineNumber == sLineNumber)) {
              _filtered = true;
               return;
            }

            if (ioE == std::string::npos) {
               sfilter = "";
            } else {
               sfilter = sfilter.substr(ioE+1, sfilter.size()-ioE-1);
            }
         }
      }

      // Create log string
      std::string ostr;

      // Assign indent spaces (tied to PID and TID)
      pid_t pid = getpid();
      std::thread::id tid = std::this_thread::get_id();
      std::stringstream pid_dash_tid_ss;
      pid_dash_tid_ss << pid << "-" << tid;
      std::string pid_dash_tid_str = pid_dash_tid_ss.str();
      _envMHIndent = "MHTRACER_INDENT_";
      char* env_indent = std::getenv(_envMHIndent.c_str());
      if (env_indent != nullptr) {
         _indent = std::stoi(std::string(env_indent));
      }
      _s.assign(_indent, ' ');

      // Check that reporting matches pid/tid
      const char* env_pid_dash_tid = std::getenv("MHTRACER_PID_DASH_TID");
      if (env_pid_dash_tid != nullptr) {
         std::string env_pid_dash_tid_str(env_pid_dash_tid);
         if (env_pid_dash_tid_str != pid_dash_tid_str) {
            _otherThread = true;
         }
      }
      else {  // PID-THREAD not set, set it for the first time (starter thread)
         setenv("MHTRACER_PID_DASH_TID", pid_dash_tid_str.c_str(), 1);
      }

      std::string paramStr;
      for (int i=0; i < params.size(); i++) {
         auto e = params[i];
         while (e.find("\n") != std::string::npos) {
            size_t pos = e.find("\n");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<NL>");
         }
         while (e.find("[") != std::string::npos) {
            size_t pos = e.find("[");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<LB>");
         }
         while (e.find("]") != std::string::npos) {
            size_t pos = e.find("]");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<RB>");
         }
         paramStr += e;
         if ((i+1) < params.size()) {
            paramStr += ", ";
         }
      }

      const char* env_dont_print_pid_dash_tid = std::getenv("MHTRACER_DONT_PRINT_PID_DASH_TID");
      if (env_dont_print_pid_dash_tid != nullptr) {
         pid_dash_tid_str = "";
      }
      if (_otherThread) {
         functionName = "MHOT_" + functionName;
      }
      ostr += _s + functionName + 
         + " [1]"
         + " [" + prefix + "]"
         + " [" + paramStr + "]"
         + " [" + pid_dash_tid_str + " "
         +    std::to_string(lineNumber)
         +    " @ " + fileName + "]\n";

      // Log to file
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_USEFILE") != std::string::npos) {
         _isFile = true;
         _fileName = "/tmp/mhtracer_" + pid_dash_tid_str + ".log";
         std::ofstream os;
         os.open(_fileName, std::ofstream::out | std::ofstream::app);
         os << ostr << "";
         os.close();
      }
      // Log to stdout
      else {
         std::cout << ostr << "";
      }

      // Increment indent spaces
      if (_otherThread) {
         return;
      }
      _indent += 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTh() {
      // Check if tracing is enabled
      char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }

      // Don't update indent if tracing was filtered or from another thread
      if (_filtered || _otherThread) {
         return;
      }

      _indent -= 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
};


#include <initializer_list>
#include <memory>
#include <random>
#include <string>

#include "absl/base/attributes.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/error_spec.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/test.h"

namespace xla {

// Utility class for making expectations/assertions related to XLA literals.
class LiteralTestUtil {
 public:
  // Asserts that the given shapes have the same rank, dimension sizes, and
  // primitive types.
  static ::testing::AssertionResult EqualShapes(
      const Shape& expected, const Shape& actual) ABSL_MUST_USE_RESULT;

  // Asserts that the provided shapes are equal as defined in AssertEqualShapes
  // and that they have the same layout.
  static ::testing::AssertionResult EqualShapesAndLayouts(
      const Shape& expected, const Shape& actual) ABSL_MUST_USE_RESULT;

  static ::testing::AssertionResult Equal(const LiteralSlice& expected,
                                          const LiteralSlice& actual)
      ABSL_MUST_USE_RESULT;

  // Asserts the given literal are (bitwise) equal to given expected values.
  template <typename NativeT>
  static void ExpectR0Equal(NativeT expected, const LiteralSlice& actual);

  template <typename NativeT>
  static void ExpectR1Equal(absl::Span<const NativeT> expected,
                            const LiteralSlice& actual);
  template <typename NativeT>
  static void ExpectR2Equal(
      std::initializer_list<std::initializer_list<NativeT>> expected,
      const LiteralSlice& actual);

  template <typename NativeT>
  static void ExpectR3Equal(
      std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>
          expected,
      const LiteralSlice& actual);

  // Asserts the given literal are (bitwise) equal to given array.
  template <typename NativeT>
  static void ExpectR2EqualArray2D(const Array2D<NativeT>& expected,
                                   const LiteralSlice& actual);
  template <typename NativeT>
  static void ExpectR3EqualArray3D(const Array3D<NativeT>& expected,
                                   const LiteralSlice& actual);
  template <typename NativeT>
  static void ExpectR4EqualArray4D(const Array4D<NativeT>& expected,
                                   const LiteralSlice& actual);

  // Decorates literal_comparison::Near() with an AssertionResult return type.
  //
  // See comment on literal_comparison::Near().
  static ::testing::AssertionResult Near(
      const LiteralSlice& expected, const LiteralSlice& actual,
      const ErrorSpec& error_spec,
      absl::optional<bool> detailed_message = absl::nullopt)
      ABSL_MUST_USE_RESULT;

  // Asserts the given literal are within the given error bound of the given
  // expected values. Only supported for floating point values.
  template <typename NativeT>
  static void ExpectR0Near(NativeT expected, const LiteralSlice& actual,
                           const ErrorSpec& error);

  template <typename NativeT>
  static void ExpectR1Near(absl::Span<const NativeT> expected,
                           const LiteralSlice& actual, const ErrorSpec& error);

  template <typename NativeT>
  static void ExpectR2Near(
      std::initializer_list<std::initializer_list<NativeT>> expected,
      const LiteralSlice& actual, const ErrorSpec& error);

  template <typename NativeT>
  static void ExpectR3Near(
      std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>
          expected,
      const LiteralSlice& actual, const ErrorSpec& error);

  template <typename NativeT>
  static void ExpectR4Near(
      std::initializer_list<std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>>
          expected,
      const LiteralSlice& actual, const ErrorSpec& error);

  // Asserts the given literal are within the given error bound to the given
  // array. Only supported for floating point values.
  template <typename NativeT>
  static void ExpectR2NearArray2D(const Array2D<NativeT>& expected,
                                  const LiteralSlice& actual,
                                  const ErrorSpec& error);

  template <typename NativeT>
  static void ExpectR3NearArray3D(const Array3D<NativeT>& expected,
                                  const LiteralSlice& actual,
                                  const ErrorSpec& error);

  template <typename NativeT>
  static void ExpectR4NearArray4D(const Array4D<NativeT>& expected,
                                  const LiteralSlice& actual,
                                  const ErrorSpec& error);

  // If the error spec is given, returns whether the expected and the actual are
  // within the error bound; otherwise, returns whether they are equal. Tuples
  // will be compared recursively.
  static ::testing::AssertionResult NearOrEqual(
      const LiteralSlice& expected, const LiteralSlice& actual,
      const absl::optional<ErrorSpec>& error) ABSL_MUST_USE_RESULT;

 private:
  LiteralTestUtil(const LiteralTestUtil&) = delete;
  LiteralTestUtil& operator=(const LiteralTestUtil&) = delete;
};

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR0Equal(NativeT expected,
                                                 const LiteralSlice& actual) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTh mht_0(mht_0_v, 327, "", "./tensorflow/compiler/xla/tests/literal_test_util.h", "LiteralTestUtil::ExpectR0Equal");

  EXPECT_TRUE(Equal(LiteralUtil::CreateR0<NativeT>(expected), actual));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR1Equal(
    absl::Span<const NativeT> expected, const LiteralSlice& actual) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTh mht_1(mht_1_v, 336, "", "./tensorflow/compiler/xla/tests/literal_test_util.h", "LiteralTestUtil::ExpectR1Equal");

  EXPECT_TRUE(Equal(LiteralUtil::CreateR1<NativeT>(expected), actual));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR2Equal(
    std::initializer_list<std::initializer_list<NativeT>> expected,
    const LiteralSlice& actual) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTh mht_2(mht_2_v, 346, "", "./tensorflow/compiler/xla/tests/literal_test_util.h", "LiteralTestUtil::ExpectR2Equal");

  EXPECT_TRUE(Equal(LiteralUtil::CreateR2<NativeT>(expected), actual));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR3Equal(
    std::initializer_list<std::initializer_list<std::initializer_list<NativeT>>>
        expected,
    const LiteralSlice& actual) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTh mht_3(mht_3_v, 357, "", "./tensorflow/compiler/xla/tests/literal_test_util.h", "LiteralTestUtil::ExpectR3Equal");

  EXPECT_TRUE(Equal(LiteralUtil::CreateR3<NativeT>(expected), actual));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR2EqualArray2D(
    const Array2D<NativeT>& expected, const LiteralSlice& actual) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTh mht_4(mht_4_v, 366, "", "./tensorflow/compiler/xla/tests/literal_test_util.h", "LiteralTestUtil::ExpectR2EqualArray2D");

  EXPECT_TRUE(Equal(LiteralUtil::CreateR2FromArray2D(expected), actual));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR3EqualArray3D(
    const Array3D<NativeT>& expected, const LiteralSlice& actual) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTh mht_5(mht_5_v, 375, "", "./tensorflow/compiler/xla/tests/literal_test_util.h", "LiteralTestUtil::ExpectR3EqualArray3D");

  EXPECT_TRUE(Equal(LiteralUtil::CreateR3FromArray3D(expected), actual));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR4EqualArray4D(
    const Array4D<NativeT>& expected, const LiteralSlice& actual) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTh mht_6(mht_6_v, 384, "", "./tensorflow/compiler/xla/tests/literal_test_util.h", "LiteralTestUtil::ExpectR4EqualArray4D");

  EXPECT_TRUE(Equal(LiteralUtil::CreateR4FromArray4D(expected), actual));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR0Near(NativeT expected,
                                                const LiteralSlice& actual,
                                                const ErrorSpec& error) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTh mht_7(mht_7_v, 394, "", "./tensorflow/compiler/xla/tests/literal_test_util.h", "LiteralTestUtil::ExpectR0Near");

  EXPECT_TRUE(Near(LiteralUtil::CreateR0<NativeT>(expected), actual, error));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR1Near(
    absl::Span<const NativeT> expected, const LiteralSlice& actual,
    const ErrorSpec& error) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTh mht_8(mht_8_v, 404, "", "./tensorflow/compiler/xla/tests/literal_test_util.h", "LiteralTestUtil::ExpectR1Near");

  EXPECT_TRUE(Near(LiteralUtil::CreateR1<NativeT>(expected), actual, error));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR2Near(
    std::initializer_list<std::initializer_list<NativeT>> expected,
    const LiteralSlice& actual, const ErrorSpec& error) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTh mht_9(mht_9_v, 414, "", "./tensorflow/compiler/xla/tests/literal_test_util.h", "LiteralTestUtil::ExpectR2Near");

  EXPECT_TRUE(Near(LiteralUtil::CreateR2<NativeT>(expected), actual, error));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR3Near(
    std::initializer_list<std::initializer_list<std::initializer_list<NativeT>>>
        expected,
    const LiteralSlice& actual, const ErrorSpec& error) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTh mht_10(mht_10_v, 425, "", "./tensorflow/compiler/xla/tests/literal_test_util.h", "LiteralTestUtil::ExpectR3Near");

  EXPECT_TRUE(Near(LiteralUtil::CreateR3<NativeT>(expected), actual, error));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR4Near(
    std::initializer_list<std::initializer_list<
        std::initializer_list<std::initializer_list<NativeT>>>>
        expected,
    const LiteralSlice& actual, const ErrorSpec& error) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTh mht_11(mht_11_v, 437, "", "./tensorflow/compiler/xla/tests/literal_test_util.h", "LiteralTestUtil::ExpectR4Near");

  EXPECT_TRUE(Near(LiteralUtil::CreateR4<NativeT>(expected), actual, error));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR2NearArray2D(
    const Array2D<NativeT>& expected, const LiteralSlice& actual,
    const ErrorSpec& error) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTh mht_12(mht_12_v, 447, "", "./tensorflow/compiler/xla/tests/literal_test_util.h", "LiteralTestUtil::ExpectR2NearArray2D");

  EXPECT_TRUE(Near(LiteralUtil::CreateR2FromArray2D(expected), actual, error));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR3NearArray3D(
    const Array3D<NativeT>& expected, const LiteralSlice& actual,
    const ErrorSpec& error) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTh mht_13(mht_13_v, 457, "", "./tensorflow/compiler/xla/tests/literal_test_util.h", "LiteralTestUtil::ExpectR3NearArray3D");

  EXPECT_TRUE(Near(LiteralUtil::CreateR3FromArray3D(expected), actual, error));
}

template <typename NativeT>
/* static */ void LiteralTestUtil::ExpectR4NearArray4D(
    const Array4D<NativeT>& expected, const LiteralSlice& actual,
    const ErrorSpec& error) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTh mht_14(mht_14_v, 467, "", "./tensorflow/compiler/xla/tests/literal_test_util.h", "LiteralTestUtil::ExpectR4NearArray4D");

  EXPECT_TRUE(Near(LiteralUtil::CreateR4FromArray4D(expected), actual, error));
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_LITERAL_TEST_UTIL_H_
