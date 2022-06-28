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
class MHTracer_DTPStensorflowPSccPSsaved_modelPSreader_testDTcc {
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
   MHTracer_DTPStensorflowPSccPSsaved_modelPSreader_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSsaved_modelPSreader_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/saved_model/reader.h"

#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/metrics.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

string TestDataPbTxt() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSreader_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/cc/saved_model/reader_test.cc", "TestDataPbTxt");

  return io::JoinPath("tensorflow", "cc", "saved_model", "testdata",
                      "half_plus_two_pbtxt", "00000123");
}

string TestDataSharded() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSreader_testDTcc mht_1(mht_1_v, 209, "", "./tensorflow/cc/saved_model/reader_test.cc", "TestDataSharded");

  return io::JoinPath("tensorflow", "cc", "saved_model", "testdata",
                      "half_plus_two", "00000123");
}

class ReaderTest : public ::testing::Test {
 protected:
  ReaderTest() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSreader_testDTcc mht_2(mht_2_v, 219, "", "./tensorflow/cc/saved_model/reader_test.cc", "ReaderTest");
}

  void CheckMetaGraphDef(const MetaGraphDef& meta_graph_def) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSreader_testDTcc mht_3(mht_3_v, 224, "", "./tensorflow/cc/saved_model/reader_test.cc", "CheckMetaGraphDef");

    const auto& tags = meta_graph_def.meta_info_def().tags();
    EXPECT_TRUE(std::find(tags.begin(), tags.end(), kSavedModelTagServe) !=
                tags.end());
    EXPECT_NE(meta_graph_def.meta_info_def().tensorflow_version(), "");
    EXPECT_EQ(
        meta_graph_def.signature_def().at("serving_default").method_name(),
        "tensorflow/serving/predict");
  }
};

TEST_F(ReaderTest, TagMatch) {
  MetaGraphDef meta_graph_def;

  const string export_dir = GetDataDependencyFilepath(TestDataSharded());
  TF_ASSERT_OK(ReadMetaGraphDefFromSavedModel(export_dir, {kSavedModelTagServe},
                                              &meta_graph_def));
  CheckMetaGraphDef(meta_graph_def);
}

TEST_F(ReaderTest, NoTagMatch) {
  MetaGraphDef meta_graph_def;

  const string export_dir = GetDataDependencyFilepath(TestDataSharded());
  Status st = ReadMetaGraphDefFromSavedModel(export_dir, {"missing-tag"},
                                             &meta_graph_def);
  EXPECT_FALSE(st.ok());
  EXPECT_TRUE(absl::StrContains(
      st.error_message(),
      "Could not find meta graph def matching supplied tags: { missing-tag }"))
      << st.error_message();
}

TEST_F(ReaderTest, NoTagMatchMultiple) {
  MetaGraphDef meta_graph_def;

  const string export_dir = GetDataDependencyFilepath(TestDataSharded());
  Status st = ReadMetaGraphDefFromSavedModel(
      export_dir, {kSavedModelTagServe, "missing-tag"}, &meta_graph_def);
  EXPECT_FALSE(st.ok());
  EXPECT_TRUE(absl::StrContains(
      st.error_message(),
      "Could not find meta graph def matching supplied tags: "))
      << st.error_message();
}

TEST_F(ReaderTest, PbtxtFormat) {
  MetaGraphDef meta_graph_def;

  const string export_dir = GetDataDependencyFilepath(TestDataPbTxt());
  TF_ASSERT_OK(ReadMetaGraphDefFromSavedModel(export_dir, {kSavedModelTagServe},
                                              &meta_graph_def));
  CheckMetaGraphDef(meta_graph_def);
}

TEST_F(ReaderTest, InvalidExportPath) {
  MetaGraphDef meta_graph_def;

  const string export_dir = GetDataDependencyFilepath("missing-path");
  Status st = ReadMetaGraphDefFromSavedModel(export_dir, {kSavedModelTagServe},
                                             &meta_graph_def);
  EXPECT_FALSE(st.ok());
}

TEST_F(ReaderTest, ReadSavedModelDebugInfoIfPresent) {
  const string export_dir = GetDataDependencyFilepath(TestDataSharded());
  std::unique_ptr<GraphDebugInfo> debug_info_proto;
  TF_ASSERT_OK(ReadSavedModelDebugInfoIfPresent(export_dir, &debug_info_proto));
}

TEST_F(ReaderTest, MetricsNotUpdatedFailedRead) {
  MetaGraphDef meta_graph_def;
  const int read_count_v1 = metrics::SavedModelRead("1").value();
  const int read_count_v2 = metrics::SavedModelRead("2").value();

  const string export_dir = GetDataDependencyFilepath("missing-path");
  Status st =
      ReadMetaGraphDefFromSavedModel(export_dir, {"serve"}, &meta_graph_def);

  EXPECT_FALSE(st.ok());
  EXPECT_EQ(metrics::SavedModelRead("1").value(), read_count_v1);
  EXPECT_EQ(metrics::SavedModelRead("2").value(), read_count_v2);
}

TEST_F(ReaderTest, MetricsUpdatedSuccessfulRead) {
  MetaGraphDef meta_graph_def;
  const int read_count_v1 = metrics::SavedModelRead("1").value();

  const string export_dir = GetDataDependencyFilepath(TestDataSharded());
  Status st =
      ReadMetaGraphDefFromSavedModel(export_dir, {"serve"}, &meta_graph_def);
  EXPECT_EQ(metrics::SavedModelRead("1").value(), read_count_v1 + 1);
}

}  // namespace
}  // namespace tensorflow
