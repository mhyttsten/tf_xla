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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_dataDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_dataDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_dataDTcc() {
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
#include "tensorflow/lite/delegates/flex/delegate_data.h"

#include <set>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/delegates/flex/util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace flex {

namespace {

// Builds a `FunctionDef` proto that contains two nodes:
// The first node is a constant node which has the value of the resource key,
// the second node is a `TfLiteSubgraphExecute` node which will take the
// resource key, and the subgraph's inputs as arguments. The function's return
// value is the return value of `TfLiteSubgraphExecute`.
void BuildFunctionDefProto(const std::string& function_name,
                           const Subgraph& subgraph,
                           tensorflow::FunctionDef& fdef) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("function_name: \"" + function_name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_dataDTcc mht_0(mht_0_v, 223, "", "./tensorflow/lite/delegates/flex/delegate_data.cc", "BuildFunctionDefProto");

  // Map inputs/outputs to types.
  std::vector<std::string> inputs, outputs;
  inputs.reserve(subgraph.inputs().size());
  outputs.reserve(subgraph.outputs().size());
  for (int i = 0; i < subgraph.inputs().size(); ++i) {
    inputs.push_back(absl::StrCat(
        "args_", i, ": ",
        TfLiteTypeToTfTypeName(subgraph.tensor(subgraph.inputs()[i])->type)));
  }
  for (int i = 0; i < subgraph.outputs().size(); ++i) {
    outputs.push_back(absl::StrCat(
        "res_", i, ": ",
        TfLiteTypeToTfTypeName(subgraph.tensor(subgraph.outputs()[i])->type)));
  }
  std::vector<tensorflow::FunctionDefHelper::Node> nodes;
  // The first node is a constant node containing the string value for the
  // resource name.
  nodes.push_back(tensorflow::FunctionDefHelper::Const<tensorflow::tstring>(
      "SubgraphResourceKey", function_name));
  // Builds the `TfLiteSubgraphExecute` node.
  tensorflow::FunctionDefHelper::Node execute_node;
  execute_node.ret.push_back("InvokeTfLite");
  execute_node.op = "TfLiteSubgraphExecute";
  execute_node.arg.push_back("SubgraphResourceKey:output:0");
  for (int i = 0; i < subgraph.inputs().size(); ++i) {
    execute_node.arg.push_back(absl::StrCat("args_", i));
  }
  nodes.push_back(execute_node);

  std::vector<std::pair<std::string, std::string>> ret_def;
  ret_def.reserve(subgraph.outputs().size());
  for (int i = 0; i < subgraph.outputs().size(); ++i) {
    ret_def.emplace_back(absl::StrCat("res_", i),
                         absl::StrCat("InvokeTfLite:output:", i));
  }
  fdef = tensorflow::FunctionDefHelper::Create(function_name, inputs, outputs,
                                               /*attr_def=*/{}, nodes, ret_def);
  // Insert input/output type attrs.
  tensorflow::AttrValue tin_attrs, tout_attrs;
  for (int i = 0; i < subgraph.inputs().size(); ++i) {
    TF_DataType dtype = tflite::flex::GetTensorFlowDataType(
        subgraph.tensor(subgraph.inputs()[i])->type);
    tin_attrs.mutable_list()->add_type(tensorflow::DataType(dtype));
  }
  for (int i = 0; i < subgraph.outputs().size(); ++i) {
    TF_DataType dtype = tflite::flex::GetTensorFlowDataType(
        subgraph.tensor(subgraph.outputs()[i])->type);
    tout_attrs.mutable_list()->add_type(tensorflow::DataType(dtype));
  }
  fdef.mutable_node_def(1)->mutable_attr()->insert({"Tin", tin_attrs});
  fdef.mutable_node_def(1)->mutable_attr()->insert({"Tout", tout_attrs});
}

// Returns a list of subgraph names which have associated function attributes.
tensorflow::Status GetSubgraphNamesForFunctionExecution(
    const std::vector<std::unique_ptr<Subgraph>>& subgraphs,
    std::set<std::string>* result) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_dataDTcc mht_1(mht_1_v, 283, "", "./tensorflow/lite/delegates/flex/delegate_data.cc", "GetSubgraphNamesForFunctionExecution");

  tensorflow::NodeDef node_def;
  for (const auto& subgraph : subgraphs) {
    for (const auto& node_and_reg : subgraph->nodes_and_registration()) {
      if (node_and_reg.second.builtin_code != tflite::BuiltinOperator_CUSTOM) {
        // If this isn't a custom op, skip.
        continue;
      }
      const std::string custom_name = node_and_reg.second.custom_name;
      if (custom_name.substr(0, strlen(tflite::kFlexCustomCodePrefix)) !=
          tflite::kFlexCustomCodePrefix) {
        // Skip if this is not a flex op.
        continue;
      }
      // The flexbuffer contains a vector where the first elements is the
      // op name and the second is a serialized NodeDef.
      const flexbuffers::Vector& v =
          flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(
                                   node_and_reg.first.custom_initial_data),
                               node_and_reg.first.custom_initial_data_size)
              .AsVector();
      // TODO(b/181352924): Use proto arena if we see performance regression.
      if (!node_def.ParseFromString(v[1].AsString().str())) {
        return tensorflow::Status(tensorflow::error::INTERNAL,
                                  "could not parse NodeDef");
      }
      // Loop through all the attributes in this node to check if it has
      // function attribute.
      for (const auto& attr : node_def.attr()) {
        if (attr.second.has_func()) {
          result->insert(attr.second.func().name());
        }
      }
    }
  }
  return tensorflow::Status::OK();
}

}  // namespace

tensorflow::Status RegisterFunctionDefForSubgraphs(
    Subgraph& main_subgraph,
    const std::function<tensorflow::Status(
        const std::vector<std::unique_ptr<Subgraph>>&, std::set<std::string>*)>&
        select_subgraphs_to_register,
    tensorflow::ResourceMgr* resource_mgr,
    tensorflow::EagerContext* eager_context, TfLiteDelegate* flex_delegate) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_dataDTcc mht_2(mht_2_v, 332, "", "./tensorflow/lite/delegates/flex/delegate_data.cc", "RegisterFunctionDefForSubgraphs");

  std::vector<std::unique_ptr<Subgraph>>* subgraphs =
      main_subgraph.GetSubgraphs();
  if (!subgraphs) {
    // If there are no subgraphs associated with the main subgraph, we will
    // return ok status because no FunctionDef needs to be registered.
    return tensorflow::Status::OK();
  }
  std::set<std::string> function_subgraphs;
  TF_RETURN_IF_ERROR(
      select_subgraphs_to_register(*subgraphs, &function_subgraphs));
  for (int i = 0; i < subgraphs->size(); ++i) {
    if (subgraphs->at(i)->GetName() == "main") {
      continue;
    }
    const std::string subgraph_name = subgraphs->at(i)->GetName();
    if (!function_subgraphs.count(subgraph_name)) {
      continue;
    }
    // This is to ensure that we only register FunctionDefs for subgraphs that
    // are used by TF ops to invoke functions.
    auto* subgraph_resource =
        new TFLiteSubgraphResource(*(subgraphs->at(i)), flex_delegate);
    TF_RETURN_IF_ERROR(resource_mgr->Create<TFLiteSubgraphResource>(
        "flex", subgraph_name, subgraph_resource));
    tensorflow::FunctionDef fdef;
    BuildFunctionDefProto(subgraph_name, *(subgraphs->at(i)), fdef);
    TF_RETURN_IF_ERROR(eager_context->AddFunctionDef(fdef));
  }
  return tensorflow::Status::OK();
}

DelegateData::DelegateData() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_dataDTcc mht_3(mht_3_v, 367, "", "./tensorflow/lite/delegates/flex/delegate_data.cc", "DelegateData::DelegateData");
}

DelegateData::~DelegateData() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_dataDTcc mht_4(mht_4_v, 372, "", "./tensorflow/lite/delegates/flex/delegate_data.cc", "DelegateData::~DelegateData");

  if (eager_context_) {
    // Notify the eager context to clean up the resource being held before
    // destructing the `DelegateData`.
    eager_context_->HostCPU()->ClearResourceMgr();
    eager_context_->Unref();
  }
}

tensorflow::Status DelegateData::Prepare(
    const tensorflow::SessionOptions& session_options, Subgraph* main_subgraph,
    TfLiteDelegate* flex_delegate) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_dataDTcc mht_5(mht_5_v, 386, "", "./tensorflow/lite/delegates/flex/delegate_data.cc", "DelegateData::Prepare");

  if (eager_context_) {
    return tensorflow::Status();
  }
  if (flex_delegate == nullptr && main_subgraph != nullptr) {
    return tensorflow::Status(
        tensorflow::error::FAILED_PRECONDITION,
        "flex_delegate must be non-null when main_subgraph is provided.");
  }

  std::vector<std::unique_ptr<tensorflow::Device>> devices;

  TF_RETURN_IF_ERROR(tensorflow::DeviceFactory::AddDevices(
      session_options, "/job:localhost/replica:0/task:0", &devices));

  auto device_mgr =
      absl::make_unique<tensorflow::StaticDeviceMgr>(std::move(devices));
  // Note that Rendezvous is ref-counted so it will be automatically deleted.
  tensorflow::Rendezvous* rendezvous =
      new tensorflow::IntraProcessRendezvous(device_mgr.get());
  eager_context_ = new tensorflow::EagerContext(
      session_options,
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
      /*async=*/false, device_mgr.release(), /*device_mgr_owned*/ true,
      rendezvous, nullptr);

  if (main_subgraph) {
    TF_RETURN_IF_ERROR(RegisterFunctionDefForSubgraphs(
        *main_subgraph, GetSubgraphNamesForFunctionExecution,
        eager_context_->HostCPU()->resource_manager(), eager_context_,
        flex_delegate));
  }
  return tensorflow::Status();
}

}  // namespace flex
}  // namespace tflite
