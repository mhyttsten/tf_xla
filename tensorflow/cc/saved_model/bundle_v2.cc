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
class MHTracer_DTPStensorflowPSccPSsaved_modelPSbundle_v2DTcc {
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
   MHTracer_DTPStensorflowPSccPSsaved_modelPSbundle_v2DTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSsaved_modelPSbundle_v2DTcc() {
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

#include "tensorflow/cc/saved_model/bundle_v2.h"

#include <string>
#include <utility>

#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/metrics.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/cc/saved_model/util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/protobuf/trackable_object_graph.pb.h"
#include "tensorflow/core/util/tensor_bundle/byte_swap.h"

namespace tensorflow {
namespace {

// `tensorflow::SavedModelV2Bundle::Load` API label.
constexpr char kCCLoadBundleV2Label[] = "cc_load_bundle_v2";

Status ReadSavedModelProto(const string& export_dir,
                           SavedModel* saved_model_proto) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("export_dir: \"" + export_dir + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSbundle_v2DTcc mht_0(mht_0_v, 210, "", "./tensorflow/cc/saved_model/bundle_v2.cc", "ReadSavedModelProto");

  LOG(INFO) << "Reading SavedModel from: " << export_dir;

  const string saved_model_pb_path =
      io::JoinPath(export_dir, kSavedModelFilenamePb);

  if (Env::Default()->FileExists(saved_model_pb_path).ok()) {
    Status result =
        ReadBinaryProto(Env::Default(), saved_model_pb_path, saved_model_proto);
    if (result.ok()) {
      metrics::SavedModelRead(saved_model::GetWriteVersion(*saved_model_proto))
          .IncrementBy(1);
    }
    return result;
  }
  const string saved_model_pbtxt_path =
      io::JoinPath(export_dir, kSavedModelFilenamePbTxt);
  if (Env::Default()->FileExists(saved_model_pbtxt_path).ok()) {
    Status result = ReadTextProto(Env::Default(), saved_model_pbtxt_path,
                                  saved_model_proto);
    if (result.ok()) {
      metrics::SavedModelRead(saved_model::GetWriteVersion(*saved_model_proto))
          .IncrementBy(1);
    }
    return result;
  }

  return Status(error::Code::NOT_FOUND,
                "Could not find SavedModel .pb or .pbtxt at supplied export "
                "directory path: " +
                    export_dir);
}

Status ReadCheckpointObjectGraph(BundleReader* bundle_reader,
                                 TrackableObjectGraph* object_graph) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSbundle_v2DTcc mht_1(mht_1_v, 247, "", "./tensorflow/cc/saved_model/bundle_v2.cc", "ReadCheckpointObjectGraph");

  Tensor object_graph_tensor;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      bundle_reader->Lookup(kObjectGraphProtoKey, &object_graph_tensor),
      "SavedModel checkpoint does not contain object graph.");
  if (object_graph_tensor.dtype() != DT_STRING ||
      object_graph_tensor.dims() != 0 ||
      object_graph_tensor.NumElements() != 1) {
    return Status(
        error::Code::FAILED_PRECONDITION,
        "SavedModel checkpoint object graph was not the correct type.");
  }

  const tstring* object_graph_string = reinterpret_cast<const tstring*>(
      object_graph_tensor.tensor_data().data());
  if (!object_graph->ParseFromString(*object_graph_string)) {
    return Status(
        error::Code::FAILED_PRECONDITION,
        "SavedModel checkpoint object graph could not be deserialized.");
  }
  return Status::OK();
}

}  // namespace

Status SavedModelV2Bundle::Load(const std::string& export_dir,
                                SavedModelV2Bundle* const bundle) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("export_dir: \"" + export_dir + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSbundle_v2DTcc mht_2(mht_2_v, 277, "", "./tensorflow/cc/saved_model/bundle_v2.cc", "SavedModelV2Bundle::Load");

  metrics::SavedModelReadApi(kCCLoadBundleV2Label).IncrementBy(1);
  SavedModel saved_model_proto;
  TF_RETURN_IF_ERROR(ReadSavedModelProto(export_dir, &saved_model_proto));

  // Load MetaGraphDef.
  // In version 2 SavedModels, there is only one MetaGraphDef.
  if (saved_model_proto.meta_graphs_size() != 1) {
    return Status(
        error::Code::INVALID_ARGUMENT,
        strings::StrCat(
            "SavedModelV2 should have exactly one MetaGraphDef but actually ",
            "contains ", saved_model_proto.meta_graphs_size()));
  }
  bundle->meta_graph_def_ =
      std::move(*saved_model_proto.mutable_meta_graphs(0));

  // Correct the endiness of Tensor content on big-endian system
  if (!port::kLittleEndian) {
    TF_RETURN_IF_ERROR(ByteSwapTensorContent(&(bundle->meta_graph_def_)));
  }

  // Load GraphDebugInfo.
  TF_RETURN_IF_ERROR(
      ReadSavedModelDebugInfoIfPresent(export_dir, &bundle->debug_info_));

  const std::string variables_dir =
      io::JoinPath(export_dir, kSavedModelVariablesDirectory);
  if (!Env::Default()->FileExists(variables_dir).ok()) {
    LOG(INFO)
        << "No checkpoint found, assuming this is a program-only SavedModel";
  } else {
    // Load the variables checkpoint reader.
    const std::string variables_prefix =
        io::JoinPath(variables_dir, kSavedModelVariablesFilename);
    bundle->variable_reader_.reset(
        new BundleReader(Env::Default(), variables_prefix));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        bundle->variable_reader_->status(),
        "Unable to load SavedModel variables checkpoint from ",
        variables_prefix);

    // Deserialize the object graph proto from the tensor bundle.
    TF_RETURN_IF_ERROR(ReadCheckpointObjectGraph(
        bundle->variable_reader_.get(), &bundle->trackable_object_graph_));
  }
  return Status::OK();
}

Status SavedModelV2Bundle::VisitObjectsToRestore(
    RestoreObjectsCallback callback) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSbundle_v2DTcc mht_3(mht_3_v, 330, "", "./tensorflow/cc/saved_model/bundle_v2.cc", "SavedModelV2Bundle::VisitObjectsToRestore");

  if (saved_object_graph().nodes_size() == 0 ||
      trackable_object_graph().nodes_size() == 0) {
    return Status::OK();
  }

  // Start from root nodes of both the SavedObjectGraph and TrackableObjectGraph
  // and descend to leaves. Note that the TrackableObjectGraph can have cycles
  // (as can the SavedObjectGraph).
  // This is detected and cycle edges are skipped.
  const SavedObject* root_saved_object = &saved_object_graph().nodes(0);
  const TrackableObjectGraph::TrackableObject* root_trackable_object =
      &trackable_object_graph().nodes(0);
  absl::flat_hash_set<int> trackable_node_ids;
  return RecurseObjectsToRestore(root_saved_object, 0, root_trackable_object,
                                 std::string(), &trackable_node_ids,
                                 std::move(callback));
}

Status SavedModelV2Bundle::RecurseObjectsToRestore(
    const SavedObject* saved_object, int saved_object_node_id,
    const TrackableObjectGraph::TrackableObject* trackable_object,
    std::string object_name, absl::flat_hash_set<int>* seen_trackable_node_ids,
    RestoreObjectsCallback callback) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("object_name: \"" + object_name + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSbundle_v2DTcc mht_4(mht_4_v, 357, "", "./tensorflow/cc/saved_model/bundle_v2.cc", "SavedModelV2Bundle::RecurseObjectsToRestore");

  // Callback if any attributes or slot variables.
  // Note that the root is always excluded from the search (it can never
  // be a restorable object). This matches some logic on the Python side.
  if (saved_object_node_id != 0 &&
      (trackable_object->attributes_size() > 0 ||
       trackable_object->slot_variables_size() > 0)) {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        callback(saved_object_node_id, *trackable_object), "Unable to restore ",
        object_name);
  }

  for (const auto& trackable_child_ref : trackable_object->children()) {
    const auto& local_name = trackable_child_ref.local_name();

    // Compute the full child name.
    std::string child_name;
    if (object_name.empty()) {
      child_name = local_name;
    } else {
      child_name = strings::StrCat(object_name, ".", local_name);
    }

    // Descend down the trackable graph.
    int trackable_child_node_id = trackable_child_ref.node_id();
    if (!seen_trackable_node_ids->insert(trackable_child_node_id).second) {
      // Cycle or duplicate detected - ignore this branch.
      continue;
    }
    if (trackable_child_node_id < 0 ||
        trackable_child_node_id >= trackable_object_graph().nodes_size()) {
      return Status(
          errors::Code::FAILED_PRECONDITION,
          strings::StrCat("Illegal trackable child node id for ", child_name));
    }
    const auto* trackable_child =
        &trackable_object_graph().nodes(trackable_child_node_id);

    // Descend down the saved object graph.
    int saved_child_node_id = -1;
    const SavedObject* saved_child = nullptr;
    for (const auto& saved_child_ref : saved_object->children()) {
      if (saved_child_ref.local_name() == local_name) {
        // Found.
        saved_child_node_id = saved_child_ref.node_id();
        if (saved_child_node_id >= 0 &&
            saved_child_node_id < saved_object_graph().nodes_size()) {
          saved_child = &saved_object_graph().nodes(saved_child_node_id);
        }
        break;
      }
    }

    if (!saved_child) {
      return Status(
          errors::Code::FAILED_PRECONDITION,
          strings::StrCat("Could not find saved object to restore for ",
                          child_name));
    }

    TF_RETURN_IF_ERROR(RecurseObjectsToRestore(
        saved_child, saved_child_node_id, trackable_child, child_name,
        seen_trackable_node_ids, callback));
  }
  return Status::OK();
}

}  // namespace tensorflow
