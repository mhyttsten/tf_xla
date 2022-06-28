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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTcc() {
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

#include "tensorflow/compiler/xla/tests/literal_test_util.h"

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/literal_comparison.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"

namespace xla {

namespace {

// Writes the given literal to a file in the test temporary directory.
void WriteLiteralToTempFile(const LiteralSlice& literal,
                            const std::string& name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/xla/tests/literal_test_util.cc", "WriteLiteralToTempFile");

  // Bazel likes for tests to write "debugging outputs" like these to
  // TEST_UNDECLARED_OUTPUTS_DIR.  This plays well with tools that inspect test
  // results, especially when they're run on remote machines.
  std::string outdir;
  if (!tensorflow::io::GetTestUndeclaredOutputsDir(&outdir)) {
    outdir = tensorflow::testing::TmpDir();
  }

  auto* env = tensorflow::Env::Default();
  std::string filename = tensorflow::io::JoinPath(
      outdir, absl::StrFormat("tempfile-%d-%s", env->NowMicros(), name));
  TF_CHECK_OK(tensorflow::WriteBinaryProto(env, absl::StrCat(filename, ".pb"),
                                           literal.ToProto()));
  TF_CHECK_OK(tensorflow::WriteStringToFile(env, absl::StrCat(filename, ".txt"),
                                            literal.ToString()));
  LOG(ERROR) << "wrote Literal to " << name << " file: " << filename
             << ".{pb,txt}";
}

// Callback helper that dumps literals to temporary files in the event of a
// miscomparison.
void OnMiscompare(const LiteralSlice& expected, const LiteralSlice& actual,
                  const LiteralSlice& mismatches,
                  const ShapeIndex& /*shape_index*/) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTcc mht_1(mht_1_v, 227, "", "./tensorflow/compiler/xla/tests/literal_test_util.cc", "OnMiscompare");

  LOG(INFO) << "expected: " << ShapeUtil::HumanString(expected.shape()) << " "
            << literal_comparison::ToStringTruncated(expected);
  LOG(INFO) << "actual:   " << ShapeUtil::HumanString(actual.shape()) << " "
            << literal_comparison::ToStringTruncated(actual);
  LOG(INFO) << "Dumping literals to temp files...";
  WriteLiteralToTempFile(expected, "expected");
  WriteLiteralToTempFile(actual, "actual");
  WriteLiteralToTempFile(mismatches, "mismatches");
}

::testing::AssertionResult StatusToAssertion(const Status& s) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTcc mht_2(mht_2_v, 241, "", "./tensorflow/compiler/xla/tests/literal_test_util.cc", "StatusToAssertion");

  if (s.ok()) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << s.error_message();
}

}  // namespace

/* static */ ::testing::AssertionResult LiteralTestUtil::EqualShapes(
    const Shape& expected, const Shape& actual) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTcc mht_3(mht_3_v, 254, "", "./tensorflow/compiler/xla/tests/literal_test_util.cc", "LiteralTestUtil::EqualShapes");

  return StatusToAssertion(literal_comparison::EqualShapes(expected, actual));
}

/* static */ ::testing::AssertionResult LiteralTestUtil::EqualShapesAndLayouts(
    const Shape& expected, const Shape& actual) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTcc mht_4(mht_4_v, 262, "", "./tensorflow/compiler/xla/tests/literal_test_util.cc", "LiteralTestUtil::EqualShapesAndLayouts");

  if (expected.ShortDebugString() != actual.ShortDebugString()) {
    return ::testing::AssertionFailure()
           << "want: " << expected.ShortDebugString()
           << " got: " << actual.ShortDebugString();
  }
  return ::testing::AssertionSuccess();
}

/* static */ ::testing::AssertionResult LiteralTestUtil::Equal(
    const LiteralSlice& expected, const LiteralSlice& actual) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTcc mht_5(mht_5_v, 275, "", "./tensorflow/compiler/xla/tests/literal_test_util.cc", "LiteralTestUtil::Equal");

  return StatusToAssertion(literal_comparison::Equal(expected, actual));
}

/* static */ ::testing::AssertionResult LiteralTestUtil::Near(
    const LiteralSlice& expected, const LiteralSlice& actual,
    const ErrorSpec& error_spec, absl::optional<bool> detailed_message) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTcc mht_6(mht_6_v, 284, "", "./tensorflow/compiler/xla/tests/literal_test_util.cc", "LiteralTestUtil::Near");

  return StatusToAssertion(literal_comparison::Near(
      expected, actual, error_spec, detailed_message, &OnMiscompare));
}

/* static */ ::testing::AssertionResult LiteralTestUtil::NearOrEqual(
    const LiteralSlice& expected, const LiteralSlice& actual,
    const absl::optional<ErrorSpec>& error) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSliteral_test_utilDTcc mht_7(mht_7_v, 294, "", "./tensorflow/compiler/xla/tests/literal_test_util.cc", "LiteralTestUtil::NearOrEqual");

  if (error.has_value()) {
    VLOG(1) << "Expects near";
    return StatusToAssertion(literal_comparison::Near(
        expected, actual, *error, /*detailed_message=*/absl::nullopt,
        &OnMiscompare));
  }
  VLOG(1) << "Expects equal";
  return StatusToAssertion(literal_comparison::Equal(expected, actual));
}

}  // namespace xla
