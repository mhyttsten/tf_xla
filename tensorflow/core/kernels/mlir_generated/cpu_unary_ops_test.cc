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
class MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPScpu_unary_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPScpu_unary_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPScpu_unary_ops_testDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/kernels/mlir_generated/base_ops_test.h"
#include "tensorflow/core/kernels/mlir_generated/base_unary_ops_test.h"

namespace tensorflow {
namespace {

// Test fixture `UnaryOpsTest` that sets the TF device is expected by the TEST
// macros below.
class UnaryOpsTest : public UnaryOpsTestBase {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPScpu_unary_ops_testDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/kernels/mlir_generated/cpu_unary_ops_test.cc", "SetUp");

    std::unique_ptr<tensorflow::Device> device_cpu(
        tensorflow::DeviceFactory::NewDevice("CPU", {},
                                             "/job:a/replica:0/task:0"));
    SetDevice(tensorflow::DEVICE_CPU, std::move(device_cpu));
  }
};

/// Test `tf.Abs`.

// TODO(b/179242253): Re-enable buffer reuse.
GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    Abs, DT_HALF, DT_HALF, DT_HALF, DT_HALF,
    test::NearZeroAndExtremeInput<Eigen::half>(), Eigen::numext::abs,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Abs, DT_FLOAT, DT_FLOAT, test::NearZeroAndExtremeInput<float>(), std::abs,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Abs, DT_DOUBLE, DT_DOUBLE, test::NearZeroAndExtremeInput<double>(),
    std::abs, test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    Abs, DT_INT8, DT_INT32, DT_INT8, DT_INT32,
    test::NearZeroAndExtremeInput<int8_t>(), std::abs,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    Abs, DT_INT16, DT_INT32, DT_INT16, DT_INT32,
    test::NearZeroAndExtremeInput<int16_t>(), std::abs,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Abs, DT_INT32, DT_INT32, test::NearZeroAndExtremeInput<int32_t>(), std::abs,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Abs, DT_INT64, DT_INT64, test::NearZeroAndExtremeInput<int64_t>(), std::abs,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())

/// Test `tf.Angle`.
template <typename T>
typename T::value_type baseline_angle(T x) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPScpu_unary_ops_testDTcc mht_1(mht_1_v, 244, "", "./tensorflow/core/kernels/mlir_generated/cpu_unary_ops_test.cc", "baseline_angle");

  return std::arg(x);
}

GENERATE_DEFAULT_TEST(Angle, DT_COMPLEX64, DT_FLOAT, baseline_angle,
                      test::OpsTestConfig().AddTout().NoBufferReuse())

GENERATE_DEFAULT_TEST(Angle, DT_COMPLEX128, DT_DOUBLE, baseline_angle,
                      test::OpsTestConfig().AddTout().NoBufferReuse())

/// Test `tf.Ceil`.
GENERATE_DEFAULT_TEST(Ceil, DT_HALF, DT_HALF, Eigen::numext::ceil,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Ceil, DT_FLOAT, DT_FLOAT, Eigen::numext::ceil,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Ceil, DT_DOUBLE, DT_DOUBLE, Eigen::numext::ceil,
                      test::OpsTestConfig().NoBufferReuse())

/// Test `tf.Cos`.
GENERATE_DEFAULT_TEST(Cos, DT_HALF, DT_HALF, Eigen::numext::cos,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Cos, DT_FLOAT, DT_FLOAT, Eigen::numext::cos,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Cos, DT_DOUBLE, DT_DOUBLE, Eigen::numext::cos,
                      test::OpsTestConfig().NoBufferReuse())

/// Test `tf.Floor`.
GENERATE_DEFAULT_TEST(Floor, DT_HALF, DT_HALF, Eigen::numext::floor,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Floor, DT_FLOAT, DT_FLOAT, Eigen::numext::floor,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Floor, DT_DOUBLE, DT_DOUBLE, Eigen::numext::floor,
                      test::OpsTestConfig().NoBufferReuse())

/// Test `tf.Invert`.
template <typename T>
T baseline_invert(T x) {
  return ~x;
}
GENERATE_DEFAULT_TEST(
    Invert, DT_INT8, DT_INT8, baseline_invert,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(
    Invert, DT_INT16, DT_INT16, baseline_invert,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(
    Invert, DT_INT32, DT_INT32, baseline_invert,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(
    Invert, DT_INT64, DT_INT64, baseline_invert,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(
    Invert, DT_UINT8, DT_UINT8, baseline_invert,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(
    Invert, DT_UINT16, DT_UINT16, baseline_invert,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(
    Invert, DT_UINT32, DT_UINT32, baseline_invert,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(
    Invert, DT_UINT64, DT_UINT64, baseline_invert,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())

/// Test `tf.Rsqrt`.
GENERATE_DEFAULT_TEST(Rsqrt, DT_HALF, DT_HALF, Eigen::numext::rsqrt,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Rsqrt, DT_FLOAT, DT_FLOAT, Eigen::numext::rsqrt,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Rsqrt, DT_DOUBLE, DT_DOUBLE, Eigen::numext::rsqrt,
                      test::OpsTestConfig().NoBufferReuse())

/// Test `tf.Sin`.
GENERATE_DEFAULT_TEST(Sin, DT_HALF, DT_HALF, Eigen::numext::sin,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Sin, DT_FLOAT, DT_FLOAT, Eigen::numext::sin,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Sin, DT_DOUBLE, DT_DOUBLE, Eigen::numext::sin,
                      test::OpsTestConfig().NoBufferReuse())

/// Test `tf.Sqrt`.
GENERATE_DEFAULT_TEST(Sqrt, DT_HALF, DT_HALF, Eigen::numext::sqrt,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Sqrt, DT_FLOAT, DT_FLOAT, Eigen::numext::sqrt,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Sqrt, DT_DOUBLE, DT_DOUBLE, Eigen::numext::sqrt,
                      test::OpsTestConfig().NoBufferReuse())

/// Test `tf.Square`.
template <typename T>
T baseline_square(T a) {
  return a * a;
}

GENERATE_DEFAULT_TEST(Square, DT_HALF, DT_HALF, baseline_square,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Square, DT_FLOAT, DT_FLOAT, baseline_square,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Square, DT_DOUBLE, DT_DOUBLE, baseline_square,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(
    Square, DT_INT32, DT_INT32, baseline_square,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(
    Square, DT_INT64, DT_INT64, baseline_square,
    test::OpsTestConfig().NoBufferReuse().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(Square, DT_COMPLEX64, DT_COMPLEX64, baseline_square,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Square, DT_COMPLEX128, DT_COMPLEX128, baseline_square,
                      test::OpsTestConfig().NoBufferReuse())

/// Test `tf.Tan`.
GENERATE_DEFAULT_TEST(Tan, DT_HALF, DT_HALF, Eigen::numext::tan,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Tan, DT_FLOAT, DT_FLOAT, Eigen::numext::tan,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Tan, DT_DOUBLE, DT_DOUBLE, Eigen::numext::tan,
                      test::OpsTestConfig().NoBufferReuse())

/// Test `tf.Tanh`.
GENERATE_DEFAULT_TEST(Tanh, DT_FLOAT, DT_FLOAT, Eigen::numext::tanh,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST(Tanh, DT_DOUBLE, DT_DOUBLE, Eigen::numext::tanh,
                      test::OpsTestConfig().NoBufferReuse())
GENERATE_DEFAULT_TEST_2(Tanh, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT,
                        Eigen::numext::tanh,
                        test::OpsTestConfig().NoBufferReuse())

}  // namespace
}  // namespace tensorflow
