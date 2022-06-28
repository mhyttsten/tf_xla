/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Helpers for loading the persistent representation of a SavedModelV2.
// Please note that this is depended on by code that does not make use of
// the full runtime and its dependencies should be restricted.

#ifndef TENSORFLOW_CC_SAVED_MODEL_BUNDLE_V2_H_
#define TENSORFLOW_CC_SAVED_MODEL_BUNDLE_V2_H_
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
class MHTracer_DTPStensorflowPSccPSsaved_modelPSbundle_v2DTh {
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
   MHTracer_DTPStensorflowPSccPSsaved_modelPSbundle_v2DTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSsaved_modelPSbundle_v2DTh() {
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


#include <string>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/protobuf/trackable_object_graph.pb.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {

/// Represents a version 2 SavedModel that is loaded from storage (but not yet
/// loaded into an executable in-memory representation).
class SavedModelV2Bundle {
 public:
  using RestoreObjectsCallback =
      std::function<Status(int, const TrackableObjectGraph::TrackableObject&)>;

  /// Loads persistent representations for a SavedModelV2 from the specified
  /// export directory.
  static Status Load(const std::string& export_dir, SavedModelV2Bundle* bundle);

  /// MetaGraphDef from the loaded SavedModel.
  MetaGraphDef& meta_graph_def() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSbundle_v2DTh mht_0(mht_0_v, 216, "", "./tensorflow/cc/saved_model/bundle_v2.h", "meta_graph_def");
 return meta_graph_def_; }

  /// SavedObjectGraph from the MetaGraphDef.
  const SavedObjectGraph& saved_object_graph() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSbundle_v2DTh mht_1(mht_1_v, 222, "", "./tensorflow/cc/saved_model/bundle_v2.h", "saved_object_graph");

    return meta_graph_def().object_graph_def();
  }

  /// TrackableObjectGraph loaded from the variable_reader() checkpoint.
  TrackableObjectGraph& trackable_object_graph() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSbundle_v2DTh mht_2(mht_2_v, 230, "", "./tensorflow/cc/saved_model/bundle_v2.h", "trackable_object_graph");

    return trackable_object_graph_;
  }

  /// BundleReader for accessing the variables bundle.
  BundleReader* variable_reader() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSbundle_v2DTh mht_3(mht_3_v, 238, "", "./tensorflow/cc/saved_model/bundle_v2.h", "variable_reader");
 return variable_reader_.get(); }

  /// The GraphDebugInfo (or nullptr if none).
  GraphDebugInfo* debug_info() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSbundle_v2DTh mht_4(mht_4_v, 244, "", "./tensorflow/cc/saved_model/bundle_v2.h", "debug_info");
 return debug_info_.get(); }

  /// Restores objects, invoking the callback with the node id in the
  /// saved_object_graph() and the corresponding TrackableObject from the
  /// trackable_object_graph(). The callback may use the variable_reader() but
  /// must not modify the underlying saved_object_graph().
  Status VisitObjectsToRestore(RestoreObjectsCallback callback);

 private:
  Status RecurseObjectsToRestore(
      const SavedObject* saved_object, int saved_object_node_id,
      const TrackableObjectGraph::TrackableObject* trackable_object,
      std::string object_name,
      absl::flat_hash_set<int>* seen_trackable_node_ids,
      RestoreObjectsCallback callback);

  MetaGraphDef meta_graph_def_;
  TrackableObjectGraph trackable_object_graph_;
  std::unique_ptr<BundleReader> variable_reader_;
  std::unique_ptr<GraphDebugInfo> debug_info_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_BUNDLE_V2_H_
