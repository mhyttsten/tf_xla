/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_BASE_OPS_TEST_H_
#define TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_BASE_OPS_TEST_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPSbase_ops_testDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPSbase_ops_testDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPSbase_ops_testDTh() {
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


#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace test {

template <typename T>
using is_integer = llvm::is_one_of<T, int8_t, int16_t, int32_t, int64_t,
                                   uint8_t, uint16_t, uint32_t, uint64_t>;

/// Helper functions to create or derive inputs of the right type and size.

template <typename T, typename LiteralT>
absl::InlinedVector<T, 10> InputAsVector(
    std::initializer_list<LiteralT> input) {
  absl::InlinedVector<T, 10> result;
  result.reserve(input.size());
  for (const LiteralT& value : input) {
    result.push_back(static_cast<T>(value));
  }
  return result;
}

template <typename T>
absl::InlinedVector<T, 10> RepeatInputToMatchShape(
    absl::InlinedVector<T, 10> input, int size) {
  absl::InlinedVector<T, 10> result;
  for (int i = 0; i < size; i++) {
    auto value = input[i % input.size()];
    result.push_back(value);
  }
  return result;
}

template <typename T>
absl::InlinedVector<T, 10> RepeatElements(absl::InlinedVector<T, 10> input,
                                          int num_repeats) {
  absl::InlinedVector<T, 10> result;
  for (T value : input) {
    for (int i = 0; i < num_repeats; ++i) {
      result.push_back(value);
    }
  }
  return result;
}

/// Helper functions to get default input shapes.

TensorShape DefaultInputShape();

/// Helper functions to configure tests.

struct OpsTestConfig {
  bool add_t = true;
  bool add_tout = false;
  // Only used for gpu_unary_ops_test.
  bool expect_buffer_reuse = true;
  bool expect_strictly_equal = false;
  bool supress_tolerance = false;
  // Negative atol/rtol will make ExpectClose use the default.
  double atol = -1;
  double rtol = -1;
  std::string input_attribute = "T";
  std::string output_attribute = "Tout";
  bool jit_compilation = false;
  OpsTestConfig ExpectStrictlyEqual() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPSbase_ops_testDTh mht_0(mht_0_v, 257, "", "./tensorflow/core/kernels/mlir_generated/base_ops_test.h", "ExpectStrictlyEqual");

    OpsTestConfig config = *this;
    config.expect_strictly_equal = true;
    return config;
  }
  OpsTestConfig SuppressTolerance() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPSbase_ops_testDTh mht_1(mht_1_v, 265, "", "./tensorflow/core/kernels/mlir_generated/base_ops_test.h", "SuppressTolerance");

    OpsTestConfig config = *this;
    config.supress_tolerance = true;
    return config;
  }
  OpsTestConfig NoBufferReuse() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPSbase_ops_testDTh mht_2(mht_2_v, 273, "", "./tensorflow/core/kernels/mlir_generated/base_ops_test.h", "NoBufferReuse");

    OpsTestConfig config = *this;
    config.expect_buffer_reuse = false;
    return config;
  }
  OpsTestConfig AddTout() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPSbase_ops_testDTh mht_3(mht_3_v, 281, "", "./tensorflow/core/kernels/mlir_generated/base_ops_test.h", "AddTout");

    OpsTestConfig config = *this;
    config.add_tout = true;
    return config;
  }
  OpsTestConfig NoT() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPSbase_ops_testDTh mht_4(mht_4_v, 289, "", "./tensorflow/core/kernels/mlir_generated/base_ops_test.h", "NoT");

    OpsTestConfig config = *this;
    config.add_t = false;
    return config;
  }
  OpsTestConfig RTol(double new_rtol) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPSbase_ops_testDTh mht_5(mht_5_v, 297, "", "./tensorflow/core/kernels/mlir_generated/base_ops_test.h", "RTol");

    OpsTestConfig config = *this;
    config.rtol = new_rtol;
    return config;
  }
  OpsTestConfig ATol(double new_atol) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPSbase_ops_testDTh mht_6(mht_6_v, 305, "", "./tensorflow/core/kernels/mlir_generated/base_ops_test.h", "ATol");

    OpsTestConfig config = *this;
    config.atol = new_atol;
    return config;
  }
  OpsTestConfig InputAttribute(const std::string& attr) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("attr: \"" + attr + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPSbase_ops_testDTh mht_7(mht_7_v, 314, "", "./tensorflow/core/kernels/mlir_generated/base_ops_test.h", "InputAttribute");

    OpsTestConfig config = *this;
    config.input_attribute = attr;
    return config;
  }
  OpsTestConfig OutputAttribute(const std::string& attr) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("attr: \"" + attr + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPSbase_ops_testDTh mht_8(mht_8_v, 323, "", "./tensorflow/core/kernels/mlir_generated/base_ops_test.h", "OutputAttribute");

    OpsTestConfig config = *this;
    config.output_attribute = attr;
    return config;
  }
  OpsTestConfig JITCompilation() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPSbase_ops_testDTh mht_9(mht_9_v, 331, "", "./tensorflow/core/kernels/mlir_generated/base_ops_test.h", "JITCompilation");

    OpsTestConfig config = *this;
    config.jit_compilation = true;
    return config;
  }
};

/// Helper functions to get more specific input data.

template <typename T, std::enable_if_t<
                          llvm::is_one_of<T, Eigen::half, float, double>::value,
                          bool> = true>
absl::InlinedVector<T, 10> NearZeroAndExtremeInput() {
  return InputAsVector<T, double>({-std::numeric_limits<double>::infinity(),
                                   -0.1, -0.0, 0.0, 0.1,
                                   std::numeric_limits<double>::infinity()});
}

template <typename T, std::enable_if_t<is_integer<T>::value, bool> = true>
absl::InlinedVector<T, 10> NearZeroAndExtremeInput() {
  return InputAsVector<T, T>({std::numeric_limits<T>::min(),
                              std::numeric_limits<T>::min() + 1, -1, 0, 1,
                              std::numeric_limits<T>::max()});
}

template <typename T, std::enable_if_t<
                          llvm::is_one_of<T, Eigen::half, float, double>::value,
                          bool> = true>
absl::InlinedVector<T, 10> NearZeroInfAndNanInput() {
  return InputAsVector<T, double>({-std::numeric_limits<double>::quiet_NaN(),
                                   -std::numeric_limits<double>::infinity(),
                                   -0.1, -0.0, 0.0, 0.1,
                                   std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::quiet_NaN()});
}

template <typename T, std::enable_if_t<
                          llvm::is_one_of<T, Eigen::half, float, double>::value,
                          bool> = true>
absl::InlinedVector<T, 10> DefaultInputGreaterEqualOne() {
  return test::InputAsVector<T, double>(
      {18.0, 9.0, 1.0, std::numeric_limits<T>::max(), 42.0, 2.0, 1.0,
       std::sqrt(std::numeric_limits<T>::max()), 9.0, 18.0});
}

template <typename T, std::enable_if_t<
                          llvm::is_one_of<T, Eigen::half, float, double>::value,
                          bool> = true>
absl::InlinedVector<T, 10> DefaultInputGreaterThanZero() {
  return test::InputAsVector<T, double>({18.0, 9.0, 1e-6, 1.0, 0.1, 1e-6, 0.1,
                                         0.2, 0.3, 0.5, 0.7, 0.9, 9.0, 18.0});
}

template <typename T, std::enable_if_t<
                          llvm::is_one_of<T, Eigen::half, float, double>::value,
                          bool> = true>
absl::InlinedVector<T, 10> DefaultInputGreaterOrEqualToZero() {
  return test::InputAsVector<T, double>({18.0, 9.0, 1e-6, 0.0, 0.1, 1e-6, 0.1,
                                         0.2, 0.3, 0.5, 0.7, 0.9, 9.0, 18.0});
}

template <typename T, std::enable_if_t<
                          llvm::is_one_of<T, Eigen::half, float, double>::value,
                          bool> = true>
absl::InlinedVector<T, 10> DefaultInputNonZero() {
  return test::InputAsVector<T, double>({18.0, 9.0, 1e-6, -0.1, 0.1, 1e-6, 0.1,
                                         0.2, 0.3, 0.5, 0.7, 0.9, 9.0, 18.0});
}

template <typename T, std::enable_if_t<is_integer<T>::value, bool> = true>
absl::InlinedVector<T, 10> DefaultInputNonZero() {
  return test::InputAsVector<T, int>({-18, -9, -1, 1, 3, 4, 5, 7, 9, 10, 18});
}

template <typename T, std::enable_if_t<
                          llvm::is_one_of<T, Eigen::half, float, double>::value,
                          bool> = true>
absl::InlinedVector<T, 10> DefaultInputBetweenZeroAndOne() {
  return test::InputAsVector<T, double>({-0.999, -0.9, -0.8, -0.5, -0.1, -0.001,
                                         -0, 0, 0.001, 0.1, 0.5, 0.8, 0.9,
                                         0.999});
}

template <typename T, std::enable_if_t<is_integer<T>::value, bool> = true>
absl::InlinedVector<T, 10> DefaultInputLessThanBitwidth() {
  auto max_shift = sizeof(T) * 8 - 1;
  absl::InlinedVector<T, 10> v;
  for (auto i = 0; i < max_shift; ++i) v.push_back(i);
  return v;
}

/// Helper functions to get default input data.

template <typename T, std::enable_if_t<is_integer<T>::value, bool> = true>
absl::InlinedVector<T, 10> DefaultInput() {
  return InputAsVector<T, int>({-18, -9, -1, 0, 0, 1, 1, 2, 3, 5, 7, 9, 9, 18});
}

template <typename T, std::enable_if_t<
                          llvm::is_one_of<T, Eigen::half, float, double>::value,
                          bool> = true>
absl::InlinedVector<T, 10> DefaultInput() {
  return InputAsVector<T, double>({-18.0, -9.0, -0.7, -0.5, -0.3, -0.2, -0.1,
                                   -1e-6, -0.0, 0.0, 1e-6, 0.1, 0.2, 0.3, 0.5,
                                   0.7, 0.9, 18.0});
}

template <typename T,
          std::enable_if_t<llvm::is_one_of<T, std::complex<float>,
                                           std::complex<double>>::value,
                           bool> = true>
absl::InlinedVector<T, 10> DefaultInput() {
  using ElementType = typename T::value_type;
  auto input = test::DefaultInput<ElementType>();
  absl::InlinedVector<T, 10> complex_input;
  for (ElementType value : input) {
    complex_input.emplace_back(value, -value);
  }
  return complex_input;
}

template <typename T,
          std::enable_if_t<llvm::is_one_of<T, std::complex<float>,
                                           std::complex<double>>::value,
                           bool> = true>
absl::InlinedVector<T, 10> ComplexInputFromValues(
    const absl::InlinedVector<typename T::value_type, 10>& real,
    const absl::InlinedVector<typename T::value_type, 10>& imag) {
  using ElementType = typename T::value_type;
  auto input = test::DefaultInput<ElementType>();
  absl::InlinedVector<T, 10> complex_input;
  CHECK_EQ(real.size(), imag.size());
  for (size_t i = 0; i < real.size() && i < imag.size(); ++i) {
    complex_input.emplace_back(real[i], imag[i]);
  }
  return complex_input;
}

template <typename T,
          std::enable_if_t<llvm::is_one_of<T, std::complex<float>,
                                           std::complex<double>>::value,
                           bool> = true>
absl::InlinedVector<T, 10> DefaultInputNonZero() {
  auto real = test::DefaultInputNonZero<typename T::value_type>();
  auto imag = real;
  std::reverse(imag.begin(), imag.end());
  return test::ComplexInputFromValues<T>(real, imag);
}

template <typename T,
          std::enable_if_t<llvm::is_one_of<T, std::complex<float>,
                                           std::complex<double>>::value,
                           bool> = true>
absl::InlinedVector<T, 10> NearZeroInfAndNanInput() {
  using ElementType = typename T::value_type;
  auto input = test::NearZeroInfAndNanInput<ElementType>();
  absl::InlinedVector<ElementType, 10> real;
  absl::InlinedVector<ElementType, 10> imag;
  for (ElementType r : input) {
    for (ElementType i : input) {
      real.push_back(r);
      imag.push_back(i);
    }
  }
  return test::ComplexInputFromValues<T>(real, imag);
}

template <typename T,
          std::enable_if_t<llvm::is_one_of<T, bool>::value, bool> = true>
absl::InlinedVector<T, 10> DefaultInput() {
  return InputAsVector<T, bool>({true, false, true, true, false});
}

}  // namespace test
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_BASE_OPS_TEST_H_
