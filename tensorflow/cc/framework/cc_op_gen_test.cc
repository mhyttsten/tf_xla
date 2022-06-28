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
class MHTracer_DTPStensorflowPSccPSframeworkPScc_op_gen_testDTcc {
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
   MHTracer_DTPStensorflowPSccPSframeworkPScc_op_gen_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSframeworkPScc_op_gen_testDTcc() {
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

#include "tensorflow/cc/framework/cc_op_gen.h"

#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

constexpr char kBaseOpDef[] = R"(
op {
  name: "Foo"
  input_arg {
    name: "images"
    description: "Images to process."
  }
  input_arg {
    name: "dim"
    description: "Description for dim."
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    description: "Description for output."
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
    description: "Type for images"
    allowed_values {
      list {
        type: DT_UINT8
        type: DT_INT8
      }
    }
    default_value {
      i: 1
    }
  }
  summary: "Summary for op Foo."
  description: "Description for op Foo."
}
)";

void ExpectHasSubstr(StringPiece s, StringPiece expected) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSframeworkPScc_op_gen_testDTcc mht_0(mht_0_v, 233, "", "./tensorflow/cc/framework/cc_op_gen_test.cc", "ExpectHasSubstr");

  EXPECT_TRUE(absl::StrContains(s, expected))
      << "'" << s << "' does not contain '" << expected << "'";
}

void ExpectDoesNotHaveSubstr(StringPiece s, StringPiece expected) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSframeworkPScc_op_gen_testDTcc mht_1(mht_1_v, 241, "", "./tensorflow/cc/framework/cc_op_gen_test.cc", "ExpectDoesNotHaveSubstr");

  EXPECT_FALSE(absl::StrContains(s, expected))
      << "'" << s << "' contains '" << expected << "'";
}

void ExpectSubstrOrder(const string& s, const string& before,
                       const string& after) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("s: \"" + s + "\"");
   mht_2_v.push_back("before: \"" + before + "\"");
   mht_2_v.push_back("after: \"" + after + "\"");
   MHTracer_DTPStensorflowPSccPSframeworkPScc_op_gen_testDTcc mht_2(mht_2_v, 253, "", "./tensorflow/cc/framework/cc_op_gen_test.cc", "ExpectSubstrOrder");

  int before_pos = s.find(before);
  int after_pos = s.find(after);
  ASSERT_NE(std::string::npos, before_pos);
  ASSERT_NE(std::string::npos, after_pos);
  EXPECT_LT(before_pos, after_pos)
      << before << " is not before " << after << " in " << s;
}

// Runs WriteCCOps and stores output in (internal_)cc_file_path and
// (internal_)h_file_path.
void GenerateCcOpFiles(Env* env, const OpList& ops,
                       const ApiDefMap& api_def_map, string* h_file_text,
                       string* internal_h_file_text) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSframeworkPScc_op_gen_testDTcc mht_3(mht_3_v, 269, "", "./tensorflow/cc/framework/cc_op_gen_test.cc", "GenerateCcOpFiles");

  const string& tmpdir = testing::TmpDir();

  const auto h_file_path = io::JoinPath(tmpdir, "test.h");
  const auto cc_file_path = io::JoinPath(tmpdir, "test.cc");
  const auto internal_h_file_path = io::JoinPath(tmpdir, "test_internal.h");
  const auto internal_cc_file_path = io::JoinPath(tmpdir, "test_internal.cc");

  WriteCCOps(ops, api_def_map, h_file_path, cc_file_path);

  TF_ASSERT_OK(ReadFileToString(env, h_file_path, h_file_text));
  TF_ASSERT_OK(
      ReadFileToString(env, internal_h_file_path, internal_h_file_text));
}

TEST(CcOpGenTest, TestVisibilityChangedToHidden) {
  const string api_def = R"(
op {
  graph_op_name: "Foo"
  visibility: HIDDEN
}
)";
  Env* env = Env::Default();
  OpList op_defs;
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);  // NOLINT
  ApiDefMap api_def_map(op_defs);

  string h_file_text, internal_h_file_text;
  // Without ApiDef
  GenerateCcOpFiles(env, op_defs, api_def_map, &h_file_text,
                    &internal_h_file_text);
  ExpectHasSubstr(h_file_text, "class Foo");
  ExpectDoesNotHaveSubstr(internal_h_file_text, "class Foo");

  // With ApiDef
  TF_ASSERT_OK(api_def_map.LoadApiDef(api_def));
  GenerateCcOpFiles(env, op_defs, api_def_map, &h_file_text,
                    &internal_h_file_text);
  ExpectHasSubstr(internal_h_file_text, "class Foo");
  ExpectDoesNotHaveSubstr(h_file_text, "class Foo");
}

TEST(CcOpGenTest, TestArgNameChanges) {
  const string api_def = R"(
op {
  graph_op_name: "Foo"
  arg_order: "dim"
  arg_order: "images"
}
)";
  Env* env = Env::Default();
  OpList op_defs;
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);  // NOLINT

  ApiDefMap api_def_map(op_defs);
  string cc_file_text, h_file_text;
  string internal_cc_file_text, internal_h_file_text;
  // Without ApiDef
  GenerateCcOpFiles(env, op_defs, api_def_map, &h_file_text,
                    &internal_h_file_text);
  ExpectSubstrOrder(h_file_text, "Input images", "Input dim");

  // With ApiDef
  TF_ASSERT_OK(api_def_map.LoadApiDef(api_def));
  GenerateCcOpFiles(env, op_defs, api_def_map, &h_file_text,
                    &internal_h_file_text);
  ExpectSubstrOrder(h_file_text, "Input dim", "Input images");
}

TEST(CcOpGenTest, TestEndpoints) {
  const string api_def = R"(
op {
  graph_op_name: "Foo"
  endpoint {
    name: "Foo1"
  }
  endpoint {
    name: "Foo2"
  }
}
)";
  Env* env = Env::Default();
  OpList op_defs;
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);  // NOLINT

  ApiDefMap api_def_map(op_defs);
  string cc_file_text, h_file_text;
  string internal_cc_file_text, internal_h_file_text;
  // Without ApiDef
  GenerateCcOpFiles(env, op_defs, api_def_map, &h_file_text,
                    &internal_h_file_text);
  ExpectHasSubstr(h_file_text, "class Foo {");
  ExpectDoesNotHaveSubstr(h_file_text, "class Foo1");
  ExpectDoesNotHaveSubstr(h_file_text, "class Foo2");

  // With ApiDef
  TF_ASSERT_OK(api_def_map.LoadApiDef(api_def));
  GenerateCcOpFiles(env, op_defs, api_def_map, &h_file_text,
                    &internal_h_file_text);
  ExpectHasSubstr(h_file_text, "class Foo1");
  ExpectHasSubstr(h_file_text, "typedef Foo1 Foo2");
  ExpectDoesNotHaveSubstr(h_file_text, "class Foo {");
}
}  // namespace
}  // namespace tensorflow
