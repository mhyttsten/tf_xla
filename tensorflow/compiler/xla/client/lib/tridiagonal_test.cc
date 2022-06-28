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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPStridiagonal_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPStridiagonal_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPStridiagonal_testDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/lib/tridiagonal.h"

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/error_spec.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace tridiagonal {
namespace {

class TridiagonalTest
    : public ClientLibraryTestBase,
      public ::testing::WithParamInterface<std::tuple<int, int, int>> {};

XLA_TEST_P(TridiagonalTest, Solves) {
  const auto& spec = GetParam();
  xla::XlaBuilder builder(TestName());

  // TODO(belletti): parametrize num_rhs.
  const int64_t batch_size = std::get<0>(spec);
  const int64_t num_eqs = std::get<1>(spec);
  const int64_t num_rhs = std::get<2>(spec);

  Array3D<float> lower_diagonal(batch_size, 1, num_eqs);
  Array3D<float> main_diagonal(batch_size, 1, num_eqs);
  Array3D<float> upper_diagonal(batch_size, 1, num_eqs);
  Array3D<float> rhs(batch_size, num_rhs, num_eqs);

  lower_diagonal.FillRandom(1.0, /*mean=*/0.0, /*seed=*/0);
  main_diagonal.FillRandom(0.05, /*mean=*/1.0,
                           /*seed=*/batch_size * num_eqs);
  upper_diagonal.FillRandom(1.0, /*mean=*/0.0,
                            /*seed=*/2 * batch_size * num_eqs);
  rhs.FillRandom(1.0, /*mean=*/0.0, /*seed=*/3 * batch_size * num_eqs);

  XlaOp lower_diagonal_xla;
  XlaOp main_diagonal_xla;
  XlaOp upper_diagonal_xla;
  XlaOp rhs_xla;

  auto lower_diagonal_data = CreateR3Parameter<float>(
      lower_diagonal, 0, "lower_diagonal", &builder, &lower_diagonal_xla);
  auto main_diagonal_data = CreateR3Parameter<float>(
      main_diagonal, 1, "main_diagonal", &builder, &main_diagonal_xla);
  auto upper_diagonal_data = CreateR3Parameter<float>(
      upper_diagonal, 2, "upper_diagonal", &builder, &upper_diagonal_xla);
  auto rhs_data = CreateR3Parameter<float>(rhs, 3, "rhs", &builder, &rhs_xla);

  TF_ASSERT_OK_AND_ASSIGN(
      XlaOp x, TridiagonalSolver(kThomas, lower_diagonal_xla, main_diagonal_xla,
                                 upper_diagonal_xla, rhs_xla));

  auto Coefficient = [](auto operand, auto i) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPStridiagonal_testDTcc mht_0(mht_0_v, 245, "", "./tensorflow/compiler/xla/client/lib/tridiagonal_test.cc", "lambda");

    return SliceInMinorDims(operand, /*start=*/{i}, /*end=*/{i + 1});
  };

  std::vector<XlaOp> relative_errors(num_eqs);

  for (int64_t i = 0; i < num_eqs; i++) {
    auto a_i = Coefficient(lower_diagonal_xla, i);
    auto b_i = Coefficient(main_diagonal_xla, i);
    auto c_i = Coefficient(upper_diagonal_xla, i);
    auto d_i = Coefficient(rhs_xla, i);

    if (i == 0) {
      relative_errors[i] =
          (b_i * Coefficient(x, i) + c_i * Coefficient(x, i + 1) - d_i) / d_i;
    } else if (i == num_eqs - 1) {
      relative_errors[i] =
          (a_i * Coefficient(x, i - 1) + b_i * Coefficient(x, i) - d_i) / d_i;
    } else {
      relative_errors[i] =
          (a_i * Coefficient(x, i - 1) + b_i * Coefficient(x, i) +
           c_i * Coefficient(x, i + 1) - d_i) /
          d_i;
    }
  }
  Abs(ConcatInDim(&builder, relative_errors, 2));

  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      ComputeAndTransfer(&builder,
                         {lower_diagonal_data.get(), main_diagonal_data.get(),
                          upper_diagonal_data.get(), rhs_data.get()}));

  auto result_data = result.data<float>({});
  for (auto result_component : result_data) {
    EXPECT_TRUE(result_component < 5e-3);
  }
}

INSTANTIATE_TEST_CASE_P(TridiagonalTestInstantiation, TridiagonalTest,
                        ::testing::Combine(::testing::Values(1, 12),
                                           ::testing::Values(4, 8),
                                           ::testing::Values(1, 12)));

}  // namespace
}  // namespace tridiagonal
}  // namespace xla
