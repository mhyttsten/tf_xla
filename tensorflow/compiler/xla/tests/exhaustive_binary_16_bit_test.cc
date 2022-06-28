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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_16_bit_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_16_bit_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_16_bit_testDTcc() {
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

#include <cmath>

#include "tensorflow/compiler/xla/tests/exhaustive_op_test_utils.h"

#ifdef __FAST_MATH__
#error("Can't be compiled with fast math on");
#endif

namespace xla {
namespace exhaustive_op_test {
namespace {

// Exhaustive test for binary operations for 16 bit floating point types,
// including float16 and bfloat.
//
// Test parameter is a pair of (begin, end) for range under test.
template <PrimitiveType T>
class Exhaustive16BitBinaryTest
    : public ExhaustiveBinaryTest<T>,
      public ::testing::WithParamInterface<std::pair<int64_t, int64_t>> {
 public:
  int64_t GetInputSize() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_16_bit_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_16_bit_test.cc", "GetInputSize");

    int64_t begin, end;
    std::tie(begin, end) = GetParam();
    return end - begin;
  }

  // Given a range of uint64_t representation, uses bits 0..15 and bits 16..31
  // for the values of src0 and src1 for a 16 bit binary operation being tested,
  // and generates the cartesian product of the two sets as the two inputs for
  // the test.
  void FillInput(std::array<Literal, 2>* input_literals) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_16_bit_testDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_16_bit_test.cc", "FillInput");

    int64_t input_size = GetInputSize();
    CHECK_EQ(input_size, (*input_literals)[0].element_count());
    CHECK_EQ(input_size, (*input_literals)[1].element_count());

    int64_t begin, end;
    std::tie(begin, end) = GetParam();
    VLOG(2) << "Checking range [" << begin << ", " << end << "]";

    absl::Span<NativeT> input_arr_0 = (*input_literals)[0].data<NativeT>();
    absl::Span<NativeT> input_arr_1 = (*input_literals)[1].data<NativeT>();
    for (int64_t i = 0; i < input_size; i++) {
      uint32_t input_val = i + begin;
      // Convert the lower 16 bits to the NativeT and replaced known incorrect
      // input values with 0.
      input_arr_0[i] = ConvertAndReplaceKnownIncorrectValueWith(input_val, 0);
      input_arr_1[i] =
          ConvertAndReplaceKnownIncorrectValueWith(input_val >> 16, 0);
    }
  }

 protected:
  using typename ExhaustiveBinaryTest<T>::NativeT;
  using ExhaustiveBinaryTest<T>::ConvertAndReplaceKnownIncorrectValueWith;
};

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16)
using ExhaustiveF16BinaryTest = Exhaustive16BitBinaryTest<F16>;
#define BINARY_TEST_F16(test_name, ...)          \
  XLA_TEST_P(ExhaustiveF16BinaryTest, test_name) \
  __VA_ARGS__
#else
#define BINARY_TEST_F16(test_name, ...)
#endif

#if defined(XLA_BACKEND_SUPPORTS_BFLOAT16)
using ExhaustiveBF16BinaryTest = Exhaustive16BitBinaryTest<BF16>;
#define BINARY_TEST_BF16(test_name, ...)          \
  XLA_TEST_P(ExhaustiveBF16BinaryTest, test_name) \
  __VA_ARGS__
#else
#define BINARY_TEST_BF16(test_name, ...)
#endif

#define BINARY_TEST_16BIT(test_name, ...) \
  BINARY_TEST_F16(test_name, __VA_ARGS__) \
  BINARY_TEST_BF16(test_name, __VA_ARGS__)

BINARY_TEST_16BIT(Add, {
  auto host_add = [](float x, float y) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_16_bit_testDTcc mht_2(mht_2_v, 271, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_16_bit_test.cc", "lambda");
 return x + y; };
  Run(AddEmptyBroadcastDimension(Add), host_add);
})

BINARY_TEST_16BIT(Sub, {
  auto host_sub = [](float x, float y) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_16_bit_testDTcc mht_3(mht_3_v, 279, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_16_bit_test.cc", "lambda");
 return x - y; };
  Run(AddEmptyBroadcastDimension(Sub), host_sub);
})

// TODO(bixia): Mul fails with bfloat16 on CPU.
BINARY_TEST_16BIT(DISABLED_ON_CPU(Mul), {
  auto host_mul = [](float x, float y) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_16_bit_testDTcc mht_4(mht_4_v, 288, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_16_bit_test.cc", "lambda");
 return x * y; };
  Run(AddEmptyBroadcastDimension(Mul), host_mul);
})

// TODO(bixia): Div fails with bfloat16 on CPU.
BINARY_TEST_16BIT(DISABLED_ON_CPU(Div), {
  auto host_div = [](float x, float y) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_16_bit_testDTcc mht_5(mht_5_v, 297, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_16_bit_test.cc", "lambda");
 return x / y; };
  Run(AddEmptyBroadcastDimension(Div), host_div);
})

BINARY_TEST_16BIT(Max, {
  Run(AddEmptyBroadcastDimension(Max), ReferenceMax<float>);
})

BINARY_TEST_16BIT(Min, {
  Run(AddEmptyBroadcastDimension(Min), ReferenceMin<float>);
})

// TODO(bixia): Pow fails with bfloat16 on CPU.
BINARY_TEST_16BIT(DISABLED_ON_GPU(DISABLED_ON_CPU(Pow)), {
  // See b/162664705.
  known_incorrect_fn_ = [](int64_t val) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_binary_16_bit_testDTcc mht_6(mht_6_v, 315, "", "./tensorflow/compiler/xla/tests/exhaustive_binary_16_bit_test.cc", "lambda");

    Eigen::bfloat16 f;
    uint16_t val_16 = val;
    memcpy(&f, &val_16, 2);
    return std::isnan(f);
  };
  Run(AddEmptyBroadcastDimension(Pow), std::pow);
})

// TODO(bixia): Atan2 fails with bfloat16 on CPU.
BINARY_TEST_16BIT(DISABLED_ON_CPU(Atan2),
                  { Run(AddEmptyBroadcastDimension(Atan2), std::atan2); })

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16)
INSTANTIATE_TEST_SUITE_P(F16, ExhaustiveF16BinaryTest,
                         ::testing::ValuesIn(CreateExhaustiveF32Ranges()));
#endif

#if defined(XLA_BACKEND_SUPPORTS_BFLOAT16)
INSTANTIATE_TEST_SUITE_P(BF16, ExhaustiveBF16BinaryTest,
                         ::testing::ValuesIn(CreateExhaustiveF32Ranges()));
#endif

}  // namespace
}  // namespace exhaustive_op_test
}  // namespace xla
