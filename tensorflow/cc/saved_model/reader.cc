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
class MHTracer_DTPStensorflowPSccPSsaved_modelPSreaderDTcc {
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
   MHTracer_DTPStensorflowPSccPSsaved_modelPSreaderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSsaved_modelPSreaderDTcc() {
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

#include <unordered_set>

#include "absl/memory/memory.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/metrics.h"
#include "tensorflow/cc/saved_model/util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/util/tensor_bundle/byte_swap.h"

namespace tensorflow {
namespace {

// Reads the SavedModel proto from saved_model.pb in `export_dir`.
// Returns a failure status when the SavedModel file does not exist.
Status ReadSavedModel(absl::string_view export_dir,
                      SavedModel* saved_model_proto) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("export_dir: \"" + std::string(export_dir.data(), export_dir.size()) + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSreaderDTcc mht_0(mht_0_v, 212, "", "./tensorflow/cc/saved_model/reader.cc", "ReadSavedModel");

  LOG(INFO) << "Reading SavedModel from: " << export_dir;

  const std::string saved_model_pb_path =
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
  const std::string saved_model_pbtxt_path =
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
  return Status(
      error::Code::NOT_FOUND,
      strings::StrCat("Could not find SavedModel .pb or .pbtxt at supplied "
                      "export directory path: ",
                      export_dir,
                      ". Check that "
                      "the directory exists and that you have the right "
                      "permissions for accessing it."));
}

Status FindMetaGraphDef(const std::unordered_set<string>& tags,
                        SavedModel* saved_model_proto,
                        MetaGraphDef* meta_graph_def) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSreaderDTcc mht_1(mht_1_v, 253, "", "./tensorflow/cc/saved_model/reader.cc", "FindMetaGraphDef");

  LOG(INFO) << "Reading meta graph with tags { " << absl::StrJoin(tags, " ")
            << " }";
  for (MetaGraphDef& graph_def : *saved_model_proto->mutable_meta_graphs()) {
    // Get tags from the graph_def.
    std::unordered_set<string> graph_tags;
    for (const string& tag : graph_def.meta_info_def().tags()) {
      graph_tags.insert(tag);
    }
    // Match with the set of tags provided.
    if (graph_tags == tags) {
      *meta_graph_def = std::move(graph_def);
      // Correct the endiness of Tensor content on big-endian system
      if (!port::kLittleEndian) {
        TF_RETURN_IF_ERROR(ByteSwapTensorContent(meta_graph_def));
      }
      return Status::OK();
    }
  }
  return Status(
      error::Code::NOT_FOUND,
      strings::StrCat(
          "Could not find meta graph def matching supplied tags: { ",
          absl::StrJoin(tags, " "),
          " }. To inspect available tag-sets in the SavedModel, please "
          "use the SavedModel CLI: `saved_model_cli`"));
}
}  // namespace

Status ReadMetaGraphDefFromSavedModel(const string& export_dir,
                                      const std::unordered_set<string>& tags,
                                      MetaGraphDef* const meta_graph_def) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("export_dir: \"" + export_dir + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSreaderDTcc mht_2(mht_2_v, 288, "", "./tensorflow/cc/saved_model/reader.cc", "ReadMetaGraphDefFromSavedModel");

  SavedModel saved_model_proto;
  TF_RETURN_IF_ERROR(ReadSavedModel(export_dir, &saved_model_proto));
  TF_RETURN_IF_ERROR(
      FindMetaGraphDef(tags, &saved_model_proto, meta_graph_def));
  return Status::OK();
}

Status ReadSavedModelDebugInfoIfPresent(
    const string& export_dir,
    std::unique_ptr<GraphDebugInfo>* debug_info_proto) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("export_dir: \"" + export_dir + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSreaderDTcc mht_3(mht_3_v, 302, "", "./tensorflow/cc/saved_model/reader.cc", "ReadSavedModelDebugInfoIfPresent");

  LOG(INFO) << "Reading SavedModel debug info (if present) from: "
            << export_dir;

  const string debug_info_pb_path =
      io::JoinPath(export_dir, "debug", "saved_model_debug_info.pb");
  if (Env::Default()->FileExists(debug_info_pb_path).ok()) {
    GraphDebugInfo debug_info;
    TF_RETURN_IF_ERROR(
        ReadBinaryProto(Env::Default(), debug_info_pb_path, &debug_info));
    *debug_info_proto =
        absl::make_unique<GraphDebugInfo>(std::move(debug_info));
  }
  return Status::OK();
}

}  // namespace tensorflow
