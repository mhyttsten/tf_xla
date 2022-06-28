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
class MHTracer_DTPStensorflowPScorePStfrtPSeagerPStransform_graph_functionDTcc {
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
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStransform_graph_functionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSeagerPStransform_graph_functionDTcc() {
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

#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/function_optimization_registry.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tfrt/host_context/device.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime

namespace tensorflow {

namespace {
constexpr char kDefaultCpuDeviceName[] = "CPU:0";
}  // namespace

Status TransformGraphFunction(const std::string& func_name,
                              const FunctionDef& fdef,
                              const std::string& device_name,
                              const tensorflow::DeviceSet& device_set,
                              EagerContext* eager_ctx, bool enable_grappler,
                              std::unique_ptr<FunctionBody>* fbody,
                              std::unique_ptr<Graph> graph,
                              tfrt::ArrayRef<const tfrt::Device*> input_devices,
                              FunctionLibraryDefinition* func_lib_def) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("func_name: \"" + func_name + "\"");
   mht_0_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPStransform_graph_functionDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/tfrt/eager/transform_graph_function.cc", "TransformGraphFunction");

  const DeviceMgr* device_mgr = eager_ctx->local_device_mgr();
  if (device_mgr == nullptr)
    return errors::Internal("Cannot find device manager");
  DumpGraph("Input function graph", graph.get());

  std::vector<string> ret_node_names;
  std::vector<string> control_ret_node_names;
  // Mapping from a function body node name to the control output name.
  std::unordered_map<string, string> node_name_to_control_ret;
  std::vector<Node*> arg_nodes, ret_nodes;
  DataTypeVector ret_types;
  auto attrs = AttrSlice(&fdef.attr());
  TF_RETURN_IF_ERROR(GetGraphAndArgRets(
      func_name, attrs, &fdef, func_lib_def, &graph, &arg_nodes, &ret_nodes,
      &ret_node_names, &ret_types, &control_ret_node_names));
  for (const auto& control_ret : fdef.control_ret()) {
    node_name_to_control_ret.emplace(control_ret.second, control_ret.first);
  }
  for (Node* node : arg_nodes) {
    const AttrValue* attr_value;
    TF_RETURN_IF_ERROR(node->attrs().Find("index", &attr_value));
    int64_t index = attr_value->i();
    node->set_assigned_device_name(input_devices[index]->name().str());
  }

  std::vector<string> input_device_names;
  int input_size = input_devices.size();
  input_device_names.reserve(input_size);
  for (int i = 0; i < input_size; ++i) {
    input_device_names.push_back(input_devices[i]->name().str());
  }

  std::vector<string> output_device_names;
  int output_size = fdef.signature().output_arg_size();
  output_device_names.reserve(output_size);
  for (int i = 0; i < output_size; ++i) {
    output_device_names.push_back(device_name);
  }

  // set default_device for placer.
  Device* default_device = nullptr;
  tensorflow::Status s = device_mgr->LookupDevice(device_name, &default_device);
  if (!s.ok())
    VLOG(1) << "TransformGraphFunction(): " << device_name << " is unknown."
            << " default device for placer is not set.";

  TF_RETURN_IF_ERROR(ProcessFunctionLibraryRuntime::PinArgsAndRets(
      input_device_names, output_device_names, device_set, arg_nodes, ret_nodes,
      func_lib_def,
      eager_ctx->AllowSoftPlacement() ? default_device : nullptr));
  DumpGraph("After running PinArgsAndRets", graph.get());

  ConfigProto config;
  bool control_rets_updated = false;
  TF_RETURN_IF_ERROR(FunctionOptimizationPassRegistry::Global().Run(
      device_set, config, &graph, func_lib_def, &control_ret_node_names,
      &control_rets_updated));

  if (control_rets_updated) {
    // Function graph pass may have resulted in different nodes/node names for
    // control rets.
    for (const auto& control_ret : control_ret_node_names) {
      node_name_to_control_ret.emplace(control_ret, control_ret);
    }
  } else {
    for (const auto& control_ret : fdef.control_ret()) {
      node_name_to_control_ret.emplace(control_ret.second, control_ret.first);
    }
  }
  DumpGraph("After running function optimization pass (bridge)", graph.get());

  // Run function inlining so that placer can place ops in nested functions.
  GraphOptimizationPassOptions optimization_options;
  SessionOptions session_options;
  // In TFRT we don't lower v2 control flow to v1.
  session_options.config.mutable_experimental()->set_use_tfrt(true);
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_do_function_inlining(true);
  optimization_options.session_options = &session_options;
  optimization_options.graph = &graph;
  optimization_options.flib_def = func_lib_def;
  optimization_options.device_set = &device_set;
  optimization_options.is_function_graph = true;
  optimization_options.default_function_device = default_device;
  optimization_options.function_def = &fdef;

  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::PRE_PLACEMENT, optimization_options));
  DumpGraph("After running pre placement passes", graph.get());

  // Run placer before importing GraphDef to MLIR.
  Placer placer(graph.get(), func_name, func_lib_def, &device_set,
                default_device, eager_ctx->AllowSoftPlacement(),
                /*log_device_placement=*/false);
  TF_RETURN_IF_ERROR(placer.Run());
  DumpGraph("After running placer", graph.get());

  if (enable_grappler) {
    Device* cpu_device;
    TF_RETURN_IF_ERROR(
        device_mgr->LookupDevice(kDefaultCpuDeviceName, &cpu_device));

    ConfigProto config_proto;
    config_proto.mutable_experimental()->set_use_tfrt(true);
    config_proto.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_do_function_inlining(true);
    // Do not skip grappler optimization even for small graphs.
    config_proto.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_min_graph_nodes(-1);

    grappler::GrapplerItem::OptimizationOptions grappler_options =
        grappler::CreateOptOptionsForEager();
    auto status = grappler::OptimizeGraph(
        std::move(ret_node_names), std::move(control_ret_node_names),
        func_lib_def, device_set, cpu_device, config_proto,
        fdef.signature().name(), grappler_options, &graph);
    if (!status.ok()) {
      LOG(WARNING) << "Ignoring multi-device function optimization failure: "
                   << status.ToString();
    }
    DumpGraph("After grappler optimization", graph.get());
  }

  // We must preserve control returns in each of the function components,
  // otherwise after function inlining we might prune side-effectful nodes.
  const auto control_ret =
      [&node_name_to_control_ret](const Node* n) -> absl::optional<string> {
    const auto it = node_name_to_control_ret.find(n->name());
    if (it != node_name_to_control_ret.end())
      return absl::make_optional<string>(it->second);
    return absl::nullopt;
  };
  FunctionDef new_func;
  TF_RETURN_IF_ERROR(
      GraphToFunctionDef(*graph, func_name, control_ret, &new_func));
  // Refresh `fbody`.
  TF_RETURN_IF_ERROR(
      FunctionDefToBodyHelper(new_func, AttrSlice(), func_lib_def, fbody));
  return Status::OK();
}
}  // namespace tensorflow
