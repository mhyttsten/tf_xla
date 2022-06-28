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
class MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPScpu_binary_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPScpu_binary_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPScpu_binary_ops_testDTcc() {
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

#include <complex>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/kernels/mlir_generated/base_binary_ops_test.h"
#include "tensorflow/core/kernels/mlir_generated/base_ops_test.h"

namespace tensorflow {
namespace {

// Test fixture `BinaryOpsTest` that sets the TF device is expected by the TEST
// macros below.
class BinaryOpsTest : public BinaryOpsTestBase {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPScpu_binary_ops_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/kernels/mlir_generated/cpu_binary_ops_test.cc", "SetUp");

    std::unique_ptr<tensorflow::Device> device_cpu(
        tensorflow::DeviceFactory::NewDevice("CPU", {},
                                             "/job:a/replica:0/task:0"));
    SetDevice(tensorflow::DEVICE_CPU, std::move(device_cpu));
  }
};

/// Test `tf.AddV2`.

template <typename T>
T baseline_add(T lhs, T rhs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmlir_generatedPScpu_binary_ops_testDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/kernels/mlir_generated/cpu_binary_ops_test.cc", "baseline_add");

  return lhs + rhs;
}

GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Half, Eigen::half, Eigen::half,
                       baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Float, float, float, baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Double, double, double,
                       baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Int8, int8_t, int8_t, baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Int16, int16_t, int16_t,
                       baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Int32, int32_t, int32_t,
                       baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Int64, int64_t, int64_t,
                       baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/UInt8, uint8_t, uint8_t,
                       baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/UInt16, uint16_t, uint16_t,
                       baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/UInt32, uint32_t, uint32_t,
                       baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/UInt64, uint64_t, uint64_t,
                       baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Complex64, std::complex<float>,
                       std::complex<float>, baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Complex128, std::complex<double>,
                       std::complex<double>, baseline_add,
                       test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.BitwiseAnd`.
template <typename T>
T baseline_bitwise_and(T lhs, T rhs) {
  return lhs & rhs;
}
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/Int8, int8_t, int8_t, baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/Int16, int16_t, int16_t,
                       baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/Int32, int32_t, int32_t,
                       baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/Int64, int64_t, int64_t,
                       baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/UInt8, uint8_t, uint8_t,
                       baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/UInt16, uint16_t, uint16_t,
                       baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/UInt32, uint32_t, uint32_t,
                       baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseAnd,
                       /*test_name=*/UInt64, uint64_t, uint64_t,
                       baseline_bitwise_and,
                       test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.BitwiseOr`.
template <typename T>
T baseline_bitwise_or(T lhs, T rhs) {
  return lhs | rhs;
}
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/Int8, int8_t, int8_t, baseline_bitwise_or,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/Int16, int16_t, int16_t,
                       baseline_bitwise_or,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/Int32, int32_t, int32_t,
                       baseline_bitwise_or,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/Int64, int64_t, int64_t,
                       baseline_bitwise_or,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/UInt8, uint8_t, uint8_t,
                       baseline_bitwise_or,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/UInt16, uint16_t, uint16_t,
                       baseline_bitwise_or,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/UInt32, uint32_t, uint32_t,
                       baseline_bitwise_or,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseOr,
                       /*test_name=*/UInt64, uint64_t, uint64_t,
                       baseline_bitwise_or,
                       test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.BitwiseXor`.
template <typename T>
T baseline_bitwise_xor(T lhs, T rhs) {
  return lhs ^ rhs;
}
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/Int8, int8_t, int8_t, baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/Int16, int16_t, int16_t,
                       baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/Int32, int32_t, int32_t,
                       baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/Int64, int64_t, int64_t,
                       baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/UInt8, uint8_t, uint8_t,
                       baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/UInt16, uint16_t, uint16_t,
                       baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/UInt32, uint32_t, uint32_t,
                       baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS(BitwiseXor,
                       /*test_name=*/UInt64, uint64_t, uint64_t,
                       baseline_bitwise_xor,
                       test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.LeftShift`.
template <typename T>
T baseline_left_shift(T lhs, T rhs) {
  return lhs << rhs;
}
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/Int8, int8_t, int8_t, test::DefaultInput<int8_t>(),
    test::DefaultInputLessThanBitwidth<int8_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/Int16, int16_t, int16_t,
    test::DefaultInput<int16_t>(),
    test::DefaultInputLessThanBitwidth<int16_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/Int32, int32_t, int32_t,
    test::DefaultInput<int32_t>(),
    test::DefaultInputLessThanBitwidth<int32_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/Int64, int64_t, int64_t,
    test::DefaultInput<int64_t>(),
    test::DefaultInputLessThanBitwidth<int64_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/UInt8, uint8_t, uint8_t,
    test::DefaultInput<uint8_t>(),
    test::DefaultInputLessThanBitwidth<uint8_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/UInt16, uint16_t, uint16_t,
    test::DefaultInput<uint16_t>(),
    test::DefaultInputLessThanBitwidth<uint16_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/UInt32, uint32_t, uint32_t,
    test::DefaultInput<uint32_t>(),
    test::DefaultInputLessThanBitwidth<uint32_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    LeftShift, /*test_name=*/UInt64, uint64_t, uint64_t,
    test::DefaultInput<uint64_t>(),
    test::DefaultInputLessThanBitwidth<uint64_t>(), baseline_left_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.RightShift`.
template <typename T>
T baseline_right_shift(T lhs, T rhs) {
  return lhs >> rhs;
}
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/Int8, int8_t, int8_t, test::DefaultInput<int8_t>(),
    test::DefaultInputLessThanBitwidth<int8_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/Int16, int16_t, int16_t, test::DefaultInput<int16_t>(),
    test::DefaultInputLessThanBitwidth<int16_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/Int32, int32_t, int32_t, test::DefaultInput<int32_t>(),
    test::DefaultInputLessThanBitwidth<int32_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/Int64, int64_t, int64_t, test::DefaultInput<int64_t>(),
    test::DefaultInputLessThanBitwidth<int64_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/UInt8, uint8_t, uint8_t, test::DefaultInput<uint8_t>(),
    test::DefaultInputLessThanBitwidth<uint8_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/UInt16, uint16_t, uint16_t, test::DefaultInput<uint16_t>(),
    test::DefaultInputLessThanBitwidth<uint16_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/UInt32, uint32_t, uint32_t, test::DefaultInput<uint32_t>(),
    test::DefaultInputLessThanBitwidth<uint32_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TESTS_WITH_SPECIFIC_INPUT_VALUES(
    RightShift,
    /*test_name=*/UInt64, uint64_t, uint64_t, test::DefaultInput<uint64_t>(),
    test::DefaultInputLessThanBitwidth<uint64_t>(), baseline_right_shift,
    test::OpsTestConfig().ExpectStrictlyEqual())

}  // namespace
}  // namespace tensorflow
