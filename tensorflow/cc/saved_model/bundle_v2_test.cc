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
class MHTracer_DTPStensorflowPSccPSsaved_modelPSbundle_v2_testDTcc {
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
   MHTracer_DTPStensorflowPSccPSsaved_modelPSbundle_v2_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSsaved_modelPSbundle_v2_testDTcc() {
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

#include "tensorflow/cc/saved_model/bundle_v2.h"

#include "tensorflow/cc/saved_model/metrics.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

constexpr char kTestData[] = "cc/saved_model/testdata";

class BundleV2Test : public ::testing::Test {
 protected:
  BundleV2Test() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSbundle_v2_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/cc/saved_model/bundle_v2_test.cc", "BundleV2Test");
}

  void RestoreVarsAndVerify(SavedModelV2Bundle* bundle,
                            std::vector<std::string> expected_names) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSbundle_v2_testDTcc mht_1(mht_1_v, 205, "", "./tensorflow/cc/saved_model/bundle_v2_test.cc", "RestoreVarsAndVerify");

    // Collect saved_node_id, full_name, checkpoint_key into a vector.
    using RestoredVarType = std::tuple<int, std::string, std::string>;
    std::vector<RestoredVarType> restored_vars;
    TF_ASSERT_OK(bundle->VisitObjectsToRestore(
        [&](int saved_node_id,
            const TrackableObjectGraph::TrackableObject& trackable_object)
            -> Status {
          for (const auto& attr : trackable_object.attributes()) {
            if (attr.name() == "VARIABLE_VALUE") {
              restored_vars.emplace_back(saved_node_id, attr.full_name(),
                                         attr.checkpoint_key());
            }
          }
          return Status::OK();
        }));

    // Should be one of each var name restored.
    for (const auto& expected_name : expected_names) {
      EXPECT_EQ(1, std::count_if(restored_vars.begin(), restored_vars.end(),
                                 [&](RestoredVarType t) {
                                   return std::get<1>(t) == expected_name;
                                 }));
    }

    for (const auto& restored_var : restored_vars) {
      // Each restored var should match a SavedObjectGraph node with the same
      // variable name.
      const auto& saved_node =
          bundle->saved_object_graph().nodes(std::get<0>(restored_var));
      EXPECT_EQ(std::get<1>(restored_var), saved_node.variable().name());

      // And should be able to load it from the tensor_bundle.
      Tensor value;
      TF_ASSERT_OK(
          bundle->variable_reader()->Lookup(std::get<2>(restored_var), &value));
    }
  }
};

TEST_F(BundleV2Test, LoadsVarsAndArithmeticObjectGraph) {
  const string export_dir = io::JoinPath(
      testing::TensorFlowSrcRoot(), kTestData, "VarsAndArithmeticObjectGraph");

  SavedModelV2Bundle bundle;
  TF_ASSERT_OK(SavedModelV2Bundle::Load(export_dir, &bundle));

  // Ensure that there are nodes in the trackable_object_graph.
  EXPECT_GT(bundle.trackable_object_graph().nodes_size(), 0);

  RestoreVarsAndVerify(&bundle, {"variable_x", "variable_y", "child_variable"});
}

TEST_F(BundleV2Test, LoadsCyclicModule) {
  const string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), kTestData, "CyclicModule");

  SavedModelV2Bundle bundle;
  TF_ASSERT_OK(SavedModelV2Bundle::Load(export_dir, &bundle));

  // Ensure that there are nodes in the trackable_object_graph.
  EXPECT_GT(bundle.trackable_object_graph().nodes_size(), 0);

  RestoreVarsAndVerify(&bundle, {"MyVariable"});
}

TEST_F(BundleV2Test, UpdatesMetrics) {
  const string kCCLoadBundleV2Label = "cc_load_bundle_v2";
  const int read_count = metrics::SavedModelRead("2").value();
  const int api_count =
      metrics::SavedModelReadApi(kCCLoadBundleV2Label).value();
  const string export_dir = io::JoinPath(
      testing::TensorFlowSrcRoot(), kTestData, "VarsAndArithmeticObjectGraph");

  SavedModelV2Bundle bundle;
  TF_ASSERT_OK(SavedModelV2Bundle::Load(export_dir, &bundle));

  EXPECT_EQ(metrics::SavedModelRead("2").value(), read_count + 1);
  EXPECT_EQ(metrics::SavedModelReadApi(kCCLoadBundleV2Label).value(),
            api_count + 1);
}

}  // namespace
}  // namespace tensorflow
