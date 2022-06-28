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
class MHTracer_DTPStensorflowPSccPSexperimentalPSlibexportPSloadDTcc {
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
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibexportPSloadDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSexperimentalPSlibexportPSloadDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/cc/experimental/libexport/load.h"

#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

#define RETURN_IF_ERROR(s) \
  {                        \
    auto c = (s);          \
    if (!c.ok()) return c; \
  }

namespace tensorflow {
namespace libexport {

using protobuf::RepeatedPtrField;

tensorflow::StatusOr<TFPackage> TFPackage::Load(const std::string& path) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibexportPSloadDTcc mht_0(mht_0_v, 206, "", "./tensorflow/cc/experimental/libexport/load.cc", "TFPackage::Load");

  // Load the proto
  TFPackage tf_package;
  const string saved_model_pb_path = io::JoinPath(path, kSavedModelFilenamePb);
  const string saved_model_pbtxt_path =
      io::JoinPath(path, kSavedModelFilenamePbTxt);
  if (Env::Default()->FileExists(saved_model_pb_path).ok()) {
    RETURN_IF_ERROR(ReadBinaryProto(Env::Default(), saved_model_pb_path,
                                    &tf_package.saved_model_proto_));
  } else if (Env::Default()->FileExists(saved_model_pbtxt_path).ok()) {
    RETURN_IF_ERROR(ReadTextProto(Env::Default(), saved_model_pbtxt_path,
                                  &tf_package.saved_model_proto_));
  } else {
    return Status(error::Code::NOT_FOUND,
                  "Could not find SavedModel .pb or .pbtxt at supplied export "
                  "directory path: " +
                      path);
  }

  // Load the trackable object graph for restoring checkpoint values
  const std::string variables_dir =
      tensorflow::io::JoinPath(path, tensorflow::kSavedModelVariablesDirectory);
  const std::string variables_prefix = tensorflow::io::JoinPath(
      variables_dir, tensorflow::kSavedModelVariablesFilename);
  tf_package.variable_reader_ = std::make_unique<tensorflow::BundleReader>(
      tensorflow::Env::Default(), variables_prefix);
  tensorflow::Tensor object_graph_tensor;
  RETURN_IF_ERROR(tf_package.variable_reader_->Lookup(
      tensorflow::kObjectGraphProtoKey, &object_graph_tensor));
  const auto* object_graph_string =
      reinterpret_cast<const tensorflow::tstring*>(
          object_graph_tensor.tensor_data().data());
  // TODO(danielellis): make sure parse was successful
  tf_package.trackable_object_graph_.ParseFromString(*object_graph_string);
  return tf_package;
}

tensorflow::StatusOr<std::string> TFPackage::GetVariableCheckpointKey(
    int index) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibexportPSloadDTcc mht_1(mht_1_v, 247, "", "./tensorflow/cc/experimental/libexport/load.cc", "TFPackage::GetVariableCheckpointKey");

  // TODO(danielellis): make sure valid index
  const auto& trackable_object = trackable_object_graph_.nodes(index);
  const TrackableObjectGraph::TrackableObject::SerializedTensor*
      serialized_tensor = nullptr;
  for (auto& maybe_serialized_tensor : trackable_object.attributes()) {
    if (maybe_serialized_tensor.name() == "VARIABLE_VALUE") {
      serialized_tensor = &maybe_serialized_tensor;
    }
  }
  if (serialized_tensor == nullptr) {
    return tensorflow::Status(error::INTERNAL,
                              "Failed to find variable value field.");
  }
  return serialized_tensor->checkpoint_key();
}

const SavedObjectGraph& TFPackage::GetObjectGraph() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibexportPSloadDTcc mht_2(mht_2_v, 267, "", "./tensorflow/cc/experimental/libexport/load.cc", "TFPackage::GetObjectGraph");

  return saved_model_proto_.mutable_meta_graphs(0)->object_graph_def();
}

const RepeatedPtrField<FunctionDef>& TFPackage::GetFunctionDefs() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibexportPSloadDTcc mht_3(mht_3_v, 274, "", "./tensorflow/cc/experimental/libexport/load.cc", "TFPackage::GetFunctionDefs");

  auto& function_library =
      saved_model_proto_.mutable_meta_graphs(0)->graph_def().library();
  return function_library.function();
}

}  // namespace libexport
}  // namespace tensorflow
