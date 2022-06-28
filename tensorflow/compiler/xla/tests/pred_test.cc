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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSpred_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSpred_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSpred_testDTcc() {
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

// Miscellaneous tests with the PRED type that don't fit anywhere else.
#include <memory>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class PredTest : public ClientLibraryTestBase {
 protected:
  void TestCompare(bool lhs, bool rhs, bool expected,
                   std::function<XlaOp(const xla::XlaOp&, const xla::XlaOp&,
                                       absl::Span<const int64_t>)>
                       op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSpred_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/xla/tests/pred_test.cc", "TestCompare");

    XlaBuilder builder(TestName());
    XlaOp lhs_op = ConstantR0<bool>(&builder, lhs);
    XlaOp rhs_op = ConstantR0<bool>(&builder, rhs);
    op(lhs_op, rhs_op, {});
    ComputeAndCompareR0<bool>(&builder, expected, {});
  }
};

TEST_F(PredTest, ConstantR0PredTrue) {
  XlaBuilder builder(TestName());
  ConstantR0<bool>(&builder, true);
  ComputeAndCompareR0<bool>(&builder, true, {});
}

TEST_F(PredTest, ConstantR0PredFalse) {
  XlaBuilder builder(TestName());
  ConstantR0<bool>(&builder, false);
  ComputeAndCompareR0<bool>(&builder, false, {});
}

TEST_F(PredTest, ConstantR0PredCompareEq) {
  TestCompare(true, false, false, &Eq);
}

TEST_F(PredTest, ConstantR0PredCompareNe) {
  TestCompare(true, false, true, &Ne);
}

TEST_F(PredTest, ConstantR0PredCompareLe) {
  TestCompare(true, false, false, &Le);
}

TEST_F(PredTest, ConstantR0PredCompareLt) {
  TestCompare(true, false, false, &Lt);
}

TEST_F(PredTest, ConstantR0PredCompareGe) {
  TestCompare(true, false, true, &Ge);
}

TEST_F(PredTest, ConstantR0PredCompareGt) {
  TestCompare(true, false, true, &Gt);
}

TEST_F(PredTest, ConstantR1Pred) {
  XlaBuilder builder(TestName());
  ConstantR1<bool>(&builder, {true, false, false, true});
  ComputeAndCompareR1<bool>(&builder, {true, false, false, true}, {});
}

TEST_F(PredTest, ConstantR2Pred) {
  XlaBuilder builder(TestName());
  ConstantR2<bool>(&builder, {{false, true, true}, {true, false, false}});
  const std::string expected = R"(pred[2,3] {
  { 0, 1, 1 },
  { 1, 0, 0 }
})";
  EXPECT_EQ(expected, ExecuteToString(&builder, {}));
}

TEST_F(PredTest, AnyR1True) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<bool>(&builder, {true, false});
  Any(a);
  ComputeAndCompareR0<bool>(&builder, true, {});
}

TEST_F(PredTest, AnyR1False) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<bool>(&builder, {false, false});
  Any(a);
  ComputeAndCompareR0<bool>(&builder, false, {});
}

TEST_F(PredTest, AnyR1VacuouslyFalse) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<bool>(&builder, {});
  Any(a);
  ComputeAndCompareR0<bool>(&builder, false, {});
}

TEST_F(PredTest, AnyR2True) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<bool>(&builder, {
                                          {false, false, false},
                                          {false, false, false},
                                          {false, false, true},
                                      });
  Any(a);
  ComputeAndCompareR0<bool>(&builder, true, {});
}

TEST_F(PredTest, AnyR2False) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<bool>(&builder, {
                                          {false, false, false},
                                          {false, false, false},
                                          {false, false, false},
                                      });
  Any(a);
  ComputeAndCompareR0<bool>(&builder, false, {});
}

}  // namespace
}  // namespace xla
