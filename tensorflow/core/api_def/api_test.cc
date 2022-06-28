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
class MHTracer_DTPStensorflowPScorePSapi_defPSapi_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSapi_defPSapi_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSapi_defPSapi_testDTcc() {
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

// Test that validates tensorflow/core/api_def/base_api/api_def*.pbtxt files.

#include <ctype.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/api_def/excluded_ops.h"
#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

constexpr char kApiDefFilePattern[] = "api_def_*.pbtxt";

string DefaultApiDefDir() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSapi_defPSapi_testDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/api_def/api_test.cc", "DefaultApiDefDir");

  return GetDataDependencyFilepath(
      io::JoinPath("tensorflow", "core", "api_def", "base_api"));
}

string PythonApiDefDir() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSapi_defPSapi_testDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/api_def/api_test.cc", "PythonApiDefDir");

  return GetDataDependencyFilepath(
      io::JoinPath("tensorflow", "core", "api_def", "python_api"));
}

// Reads golden ApiDef files and returns a map from file name to ApiDef file
// contents.
void GetGoldenApiDefs(Env* env, const string& api_files_dir,
                      std::unordered_map<string, ApiDef>* name_to_api_def) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("api_files_dir: \"" + api_files_dir + "\"");
   MHTracer_DTPStensorflowPScorePSapi_defPSapi_testDTcc mht_2(mht_2_v, 235, "", "./tensorflow/core/api_def/api_test.cc", "GetGoldenApiDefs");

  std::vector<string> matching_paths;
  TF_CHECK_OK(env->GetMatchingPaths(
      io::JoinPath(api_files_dir, kApiDefFilePattern), &matching_paths));

  for (auto& file_path : matching_paths) {
    string file_contents;
    TF_CHECK_OK(ReadFileToString(env, file_path, &file_contents));
    file_contents = PBTxtFromMultiline(file_contents);

    ApiDefs api_defs;
    QCHECK(tensorflow::protobuf::TextFormat::ParseFromString(file_contents,
                                                             &api_defs))
        << "Failed to load " << file_path;
    CHECK_EQ(api_defs.op_size(), 1);
    (*name_to_api_def)[api_defs.op(0).graph_op_name()] = api_defs.op(0);
  }
}

void TestAllApiDefsHaveCorrespondingOp(
    const OpList& ops, const std::unordered_map<string, ApiDef>& api_defs_map) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSapi_defPSapi_testDTcc mht_3(mht_3_v, 258, "", "./tensorflow/core/api_def/api_test.cc", "TestAllApiDefsHaveCorrespondingOp");

  std::unordered_set<string> op_names;
  for (const auto& op : ops.op()) {
    op_names.insert(op.name());
  }
  for (const auto& name_and_api_def : api_defs_map) {
    ASSERT_TRUE(op_names.find(name_and_api_def.first) != op_names.end())
        << name_and_api_def.first << " op has ApiDef but missing from ops. "
        << "Does api_def_" << name_and_api_def.first << " need to be deleted?";
  }
}

void TestAllApiDefInputArgsAreValid(
    const OpList& ops, const std::unordered_map<string, ApiDef>& api_defs_map) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSapi_defPSapi_testDTcc mht_4(mht_4_v, 274, "", "./tensorflow/core/api_def/api_test.cc", "TestAllApiDefInputArgsAreValid");

  for (const auto& op : ops.op()) {
    const auto api_def_iter = api_defs_map.find(op.name());
    if (api_def_iter == api_defs_map.end()) {
      continue;
    }
    const auto& api_def = api_def_iter->second;
    for (const auto& api_def_arg : api_def.in_arg()) {
      bool found_arg = false;
      for (const auto& op_arg : op.input_arg()) {
        if (api_def_arg.name() == op_arg.name()) {
          found_arg = true;
          break;
        }
      }
      ASSERT_TRUE(found_arg)
          << "Input argument " << api_def_arg.name()
          << " (overwritten in api_def_" << op.name()
          << ".pbtxt) is not defined in OpDef for " << op.name();
    }
  }
}

void TestAllApiDefOutputArgsAreValid(
    const OpList& ops, const std::unordered_map<string, ApiDef>& api_defs_map) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSapi_defPSapi_testDTcc mht_5(mht_5_v, 301, "", "./tensorflow/core/api_def/api_test.cc", "TestAllApiDefOutputArgsAreValid");

  for (const auto& op : ops.op()) {
    const auto api_def_iter = api_defs_map.find(op.name());
    if (api_def_iter == api_defs_map.end()) {
      continue;
    }
    const auto& api_def = api_def_iter->second;
    for (const auto& api_def_arg : api_def.out_arg()) {
      bool found_arg = false;
      for (const auto& op_arg : op.output_arg()) {
        if (api_def_arg.name() == op_arg.name()) {
          found_arg = true;
          break;
        }
      }
      ASSERT_TRUE(found_arg)
          << "Output argument " << api_def_arg.name()
          << " (overwritten in api_def_" << op.name()
          << ".pbtxt) is not defined in OpDef for " << op.name();
    }
  }
}

void TestAllApiDefAttributeNamesAreValid(
    const OpList& ops, const std::unordered_map<string, ApiDef>& api_defs_map) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSapi_defPSapi_testDTcc mht_6(mht_6_v, 328, "", "./tensorflow/core/api_def/api_test.cc", "TestAllApiDefAttributeNamesAreValid");

  for (const auto& op : ops.op()) {
    const auto api_def_iter = api_defs_map.find(op.name());
    if (api_def_iter == api_defs_map.end()) {
      continue;
    }
    const auto& api_def = api_def_iter->second;
    for (const auto& api_def_attr : api_def.attr()) {
      bool found_attr = false;
      for (const auto& op_attr : op.attr()) {
        if (api_def_attr.name() == op_attr.name()) {
          found_attr = true;
        }
      }
      ASSERT_TRUE(found_attr)
          << "Attribute " << api_def_attr.name() << " (overwritten in api_def_"
          << op.name() << ".pbtxt) is not defined in OpDef for " << op.name();
    }
  }
}

void TestDeprecatedAttributesSetCorrectly(
    const std::unordered_map<string, ApiDef>& api_defs_map) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSapi_defPSapi_testDTcc mht_7(mht_7_v, 353, "", "./tensorflow/core/api_def/api_test.cc", "TestDeprecatedAttributesSetCorrectly");

  for (const auto& name_and_api_def : api_defs_map) {
    int num_deprecated_endpoints = 0;
    const auto& api_def = name_and_api_def.second;
    for (const auto& endpoint : api_def.endpoint()) {
      if (endpoint.deprecated()) {
        ++num_deprecated_endpoints;
      }
    }

    const auto& name = name_and_api_def.first;
    ASSERT_TRUE(api_def.deprecation_message().empty() ||
                num_deprecated_endpoints == 0)
        << "Endpoints are set to 'deprecated' for deprecated op " << name
        << ". If an op is deprecated (i.e. deprecation_message is set), "
        << "all the endpoints are deprecated implicitly and 'deprecated' "
        << "field should not be set.";
    if (num_deprecated_endpoints > 0) {
      ASSERT_NE(num_deprecated_endpoints, api_def.endpoint_size())
          << "All " << name << " endpoints are deprecated. Please, set "
          << "deprecation_message in api_def_" << name << ".pbtxt instead. "
          << "to indicate that the op is deprecated.";
    }
  }
}

void TestDeprecationVersionSetCorrectly(
    const std::unordered_map<string, ApiDef>& api_defs_map) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSapi_defPSapi_testDTcc mht_8(mht_8_v, 383, "", "./tensorflow/core/api_def/api_test.cc", "TestDeprecationVersionSetCorrectly");

  for (const auto& name_and_api_def : api_defs_map) {
    const auto& name = name_and_api_def.first;
    const auto& api_def = name_and_api_def.second;
    if (api_def.deprecation_version() != 0) {
      ASSERT_TRUE(api_def.deprecation_version() > 0)
          << "Found ApiDef with negative deprecation_version";
      ASSERT_FALSE(api_def.deprecation_message().empty())
          << "ApiDef that includes deprecation_version > 0 must also specify "
          << "a deprecation_message. Op " << name
          << " has deprecation_version > 0 but deprecation_message is not set.";
    }
  }
}

class BaseApiTest : public ::testing::Test {
 protected:
  BaseApiTest() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSapi_defPSapi_testDTcc mht_9(mht_9_v, 403, "", "./tensorflow/core/api_def/api_test.cc", "BaseApiTest");

    OpRegistry::Global()->Export(false, &ops_);
    const std::vector<string> multi_line_fields = {"description"};

    Env* env = Env::Default();
    GetGoldenApiDefs(env, DefaultApiDefDir(), &api_defs_map_);
  }
  OpList ops_;
  std::unordered_map<string, ApiDef> api_defs_map_;
};

// Check that all ops have an ApiDef.
TEST_F(BaseApiTest, AllOpsAreInApiDef) {
  auto* excluded_ops = GetExcludedOps();
  for (const auto& op : ops_.op()) {
    if (excluded_ops->find(op.name()) != excluded_ops->end()) {
      continue;
    }
    EXPECT_TRUE(api_defs_map_.find(op.name()) != api_defs_map_.end())
        << op.name() << " op does not have api_def_*.pbtxt file. "
        << "Please add api_def_" << op.name() << ".pbtxt file "
        << "under tensorflow/core/api_def/base_api/ directory.";
  }
}

// Check that ApiDefs have a corresponding op.
TEST_F(BaseApiTest, AllApiDefsHaveCorrespondingOp) {
  TestAllApiDefsHaveCorrespondingOp(ops_, api_defs_map_);
}

string GetOpDefHasDocStringError(const string& op_name) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScorePSapi_defPSapi_testDTcc mht_10(mht_10_v, 437, "", "./tensorflow/core/api_def/api_test.cc", "GetOpDefHasDocStringError");

  return strings::Printf(
      "OpDef for %s has a doc string. "
      "Doc strings must be defined in ApiDef instead of OpDef. "
      "Please, add summary and descriptions in api_def_%s"
      ".pbtxt file instead",
      op_name.c_str(), op_name.c_str());
}

// Check that OpDef's do not have descriptions and summaries.
// Descriptions and summaries must be in corresponding ApiDefs.
TEST_F(BaseApiTest, OpDefsShouldNotHaveDocs) {
  auto* excluded_ops = GetExcludedOps();
  for (const auto& op : ops_.op()) {
    if (excluded_ops->find(op.name()) != excluded_ops->end()) {
      continue;
    }
    ASSERT_TRUE(op.summary().empty()) << GetOpDefHasDocStringError(op.name());
    ASSERT_TRUE(op.description().empty())
        << GetOpDefHasDocStringError(op.name());
    for (const auto& arg : op.input_arg()) {
      ASSERT_TRUE(arg.description().empty())
          << GetOpDefHasDocStringError(op.name());
    }
    for (const auto& arg : op.output_arg()) {
      ASSERT_TRUE(arg.description().empty())
          << GetOpDefHasDocStringError(op.name());
    }
    for (const auto& attr : op.attr()) {
      ASSERT_TRUE(attr.description().empty())
          << GetOpDefHasDocStringError(op.name());
    }
  }
}

// Checks that input arg names in an ApiDef match input
// arg names in corresponding OpDef.
TEST_F(BaseApiTest, AllApiDefInputArgsAreValid) {
  TestAllApiDefInputArgsAreValid(ops_, api_defs_map_);
}

// Checks that output arg names in an ApiDef match output
// arg names in corresponding OpDef.
TEST_F(BaseApiTest, AllApiDefOutputArgsAreValid) {
  TestAllApiDefOutputArgsAreValid(ops_, api_defs_map_);
}

// Checks that attribute names in an ApiDef match attribute
// names in corresponding OpDef.
TEST_F(BaseApiTest, AllApiDefAttributeNamesAreValid) {
  TestAllApiDefAttributeNamesAreValid(ops_, api_defs_map_);
}

// Checks that deprecation is set correctly.
TEST_F(BaseApiTest, DeprecationSetCorrectly) {
  TestDeprecatedAttributesSetCorrectly(api_defs_map_);
}

// Checks that deprecation_version is set for entire op only if
// deprecation_message is set.
TEST_F(BaseApiTest, DeprecationVersionSetCorrectly) {
  TestDeprecationVersionSetCorrectly(api_defs_map_);
}

class PythonApiTest : public ::testing::Test {
 protected:
  PythonApiTest() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSapi_defPSapi_testDTcc mht_11(mht_11_v, 506, "", "./tensorflow/core/api_def/api_test.cc", "PythonApiTest");

    OpRegistry::Global()->Export(false, &ops_);
    const std::vector<string> multi_line_fields = {"description"};

    Env* env = Env::Default();
    GetGoldenApiDefs(env, PythonApiDefDir(), &api_defs_map_);
  }
  OpList ops_;
  std::unordered_map<string, ApiDef> api_defs_map_;
};

// Check that ApiDefs have a corresponding op.
TEST_F(PythonApiTest, AllApiDefsHaveCorrespondingOp) {
  TestAllApiDefsHaveCorrespondingOp(ops_, api_defs_map_);
}

// Checks that input arg names in an ApiDef match input
// arg names in corresponding OpDef.
TEST_F(PythonApiTest, AllApiDefInputArgsAreValid) {
  TestAllApiDefInputArgsAreValid(ops_, api_defs_map_);
}

// Checks that output arg names in an ApiDef match output
// arg names in corresponding OpDef.
TEST_F(PythonApiTest, AllApiDefOutputArgsAreValid) {
  TestAllApiDefOutputArgsAreValid(ops_, api_defs_map_);
}

// Checks that attribute names in an ApiDef match attribute
// names in corresponding OpDef.
TEST_F(PythonApiTest, AllApiDefAttributeNamesAreValid) {
  TestAllApiDefAttributeNamesAreValid(ops_, api_defs_map_);
}

// Checks that deprecation is set correctly.
TEST_F(PythonApiTest, DeprecationSetCorrectly) {
  TestDeprecatedAttributesSetCorrectly(api_defs_map_);
}

// Checks that deprecation_version is set for entire op only if
// deprecation_message is set.
TEST_F(PythonApiTest, DeprecationVersionSetCorrectly) {
  TestDeprecationVersionSetCorrectly(api_defs_map_);
}

}  // namespace
}  // namespace tensorflow
