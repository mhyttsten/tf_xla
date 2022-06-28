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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_unary_test_f64DTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_unary_test_f64DTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_unary_test_f64DTcc() {
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

#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/exhaustive_op_test_utils.h"
#include "tensorflow/compiler/xla/util.h"

#ifdef __FAST_MATH__
#error "Can't be compiled with fast math on"
#endif

namespace xla {
namespace exhaustive_op_test {

// Exhaustive test for unary operations for double.
//
// Test parameter is a tuple containing
//   - primitive type under test,
//   - FpValues representing a set of double values.

class ExhaustiveF64UnaryTest : public ExhaustiveUnaryTest<F64>,
                               public ::testing::WithParamInterface<FpValues> {
 private:
  int64_t GetInputSize() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_unary_test_f64DTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/xla/tests/exhaustive_unary_test_f64.cc", "GetInputSize");

    FpValues values = GetParam();
    return values.GetTotalNumValues();
  }

  void FillInput(std::array<Literal, 1>* input_literal) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_unary_test_f64DTcc mht_1(mht_1_v, 213, "", "./tensorflow/compiler/xla/tests/exhaustive_unary_test_f64.cc", "FillInput");

    FpValues fp_values = GetParam();
    int64_t input_size = (*input_literal)[0].element_count();
    LOG(INFO) << "Checking fp values " << fp_values.ToString() << ", "
              << input_size;
    absl::Span<double> input_arr = (*input_literal)[0].data<double>();

    uint64_t i = 0;
    for (auto bits : fp_values) {
      input_arr[i] = this->ConvertAndReplaceKnownIncorrectValueWith(bits, 1);
      ++i;
    }
    CHECK_EQ(i, input_size);
  }
};

#define UNARY_TEST_FLOAT_64(test_name, ...)     \
  XLA_TEST_P(ExhaustiveF64UnaryTest, test_name) \
  __VA_ARGS__

UNARY_TEST_FLOAT_64(Log, { Run(Log, std::log); })

UNARY_TEST_FLOAT_64(Log1p, { Run(Log1p, std::log1p); })

UNARY_TEST_FLOAT_64(Exp, { Run(Exp, std::exp); })

UNARY_TEST_FLOAT_64(Expm1, { Run(Expm1, std::expm1); })

// TODO(b/138385863): Turn on the test for GPU after fixing the bug.
UNARY_TEST_FLOAT_64(DISABLED_ON_GPU(PowOneHalf), {
  Run([](XlaOp x) { return Pow(x, ScalarLike(x, 0.5)); },
      +[](double x) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_unary_test_f64DTcc mht_2(mht_2_v, 247, "", "./tensorflow/compiler/xla/tests/exhaustive_unary_test_f64.cc", "lambda");
 return std::pow(x, 0.5); });
})

UNARY_TEST_FLOAT_64(Rsqrt, {
  Run(
      Rsqrt, +[](double x) { return 1 / std::sqrt(x); });
})

UNARY_TEST_FLOAT_64(Sqrt, { Run(Sqrt, std::sqrt); })

UNARY_TEST_FLOAT_64(Acosh, { Run(Acosh, std::acosh); })

UNARY_TEST_FLOAT_64(Asinh, { Run(Asinh, std::asinh); })

UNARY_TEST_FLOAT_64(Atanh, { Run(Atanh, std::atanh); })

UNARY_TEST_FLOAT_64(Acos, { Run(Acos, std::acos); })

UNARY_TEST_FLOAT_64(Asin, { Run(Asin, std::asin); })

UNARY_TEST_FLOAT_64(Cosh, { Run(Cosh, std::cosh); })

UNARY_TEST_FLOAT_64(Sinh, { Run(Sinh, std::sinh); })

UNARY_TEST_FLOAT_64(Tanh, {
  ErrorSpecGen error_spec_gen = GetDefaultSpecGenerator();
  if (platform_ == "CUDA") {
    error_spec_gen = +[](NativeT x) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSexhaustive_unary_test_f64DTcc mht_3(mht_3_v, 277, "", "./tensorflow/compiler/xla/tests/exhaustive_unary_test_f64.cc", "lambda");

      return x <= static_cast<NativeT>(-20.0) || x >= static_cast<NativeT>(20.0)
                 ? ErrorSpec{0, 0}
                 : GetDefaultSpecGenerator()(x);
    };
  }
  Run(Tanh, std::tanh, error_spec_gen);
})

UNARY_TEST_FLOAT_64(Cos, { Run(Cos, std::cos); })

UNARY_TEST_FLOAT_64(Sin, { Run(Sin, std::sin); })

UNARY_TEST_FLOAT_64(Tan, { Run(Tan, std::tan); })

UNARY_TEST_FLOAT_64(Round, { Run(Round, std::round); })

UNARY_TEST_FLOAT_64(Erf, {
  Run(Erf, std::erf, [](NativeT x) { return ErrorSpec{1e-20, 1e-20}; });
})

UNARY_TEST_FLOAT_64(Erfc, {
  Run(Erfc, std::erfc, [](NativeT x) { return ErrorSpec{1e-20, 1e-20}; });
})

INSTANTIATE_TEST_SUITE_P(
    SpecialValues, ExhaustiveF64UnaryTest,
    ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>()));

INSTANTIATE_TEST_SUITE_P(NormalValues, ExhaustiveF64UnaryTest,
                         ::testing::Values(GetNormals<double>(1000)));

// Tests a total of 4000000000 inputs, with 16000000 inputs in each sub-test, to
// keep the peak memory usage low.
INSTANTIATE_TEST_SUITE_P(
    LargeAndSmallMagnitudeNormalValues, ExhaustiveF64UnaryTest,
    ::testing::ValuesIn(GetFpValuesForMagnitudeExtremeNormals<double>(
        4000000000ull, 16000000)));

}  // namespace exhaustive_op_test
}  // namespace xla
