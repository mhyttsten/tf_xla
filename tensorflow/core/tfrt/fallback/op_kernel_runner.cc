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
class MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTcc {
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
   MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTcc() {
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
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"

#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace tfrt_stub {
namespace {

Status CheckOpDefCompatibility(const tensorflow::OpDef& op_def) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTcc mht_0(mht_0_v, 192, "", "./tensorflow/core/tfrt/fallback/op_kernel_runner.cc", "CheckOpDefCompatibility");

  auto check_arg_def = [&](const auto& arg_def) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTcc mht_1(mht_1_v, 196, "", "./tensorflow/core/tfrt/fallback/op_kernel_runner.cc", "lambda");

    if (arg_def.is_ref())
      return tensorflow::errors::Internal(
          "TFRT kernel fallback error: Unsupported ref args in ",
          op_def.name());
    return Status::OK();
  };

  for (const auto& arg_def : op_def.input_arg())
    TF_RETURN_IF_ERROR(check_arg_def(arg_def));
  for (const auto& arg_def : op_def.output_arg())
    TF_RETURN_IF_ERROR(check_arg_def(arg_def));

  return Status::OK();
}

// Create a tensorflow::NodeDef from the tensorflow::OpDef and the attributes.
StatusOr<tensorflow::NodeDef> BuildNodeDef(
    const tensorflow::OpDef& op_def, int num_args,
    const std::function<Status(tensorflow::AttrValueMap*)>& attr_builder) {
  tensorflow::NodeDef node_def;
  node_def.set_name(op_def.name());
  node_def.set_op(op_def.name());
  for (int i = 0; i < num_args; ++i) {
    node_def.add_input("dummy_input");
  }

  auto* attr_value_map = node_def.mutable_attr();
  TF_RETURN_IF_ERROR(attr_builder(attr_value_map));

  // For any attr-value pairs that exist in the op def (from op registry)
  // but not in `attr_value_map`, fill them into `attr_value_map`, so that we
  // can run a TFE_Op without having to specify all the default attr values
  // (e.g. for matmul, the `transpose_a` attr defaults to false).
  for (const auto& attr_def : op_def.attr()) {
    if (attr_def.has_default_value()) {
      // Insertion will fail if this attribute already has a value.
      attr_value_map->insert({attr_def.name(), attr_def.default_value()});
    }
  }
  return node_def;
}

tensorflow::Status CreateOpKernel(
    tensorflow::FunctionLibraryRuntime* flr, tensorflow::NodeDef ndef,
    std::unique_ptr<tensorflow::OpKernel>* result) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTcc mht_2(mht_2_v, 244, "", "./tensorflow/core/tfrt/fallback/op_kernel_runner.cc", "CreateOpKernel");

  std::shared_ptr<const tensorflow::NodeProperties> props;
  TF_RETURN_IF_ERROR(tensorflow::NodeProperties::CreateFromNodeDef(
      ndef, flr->GetFunctionLibraryDefinition(), &props));
  tensorflow::OpKernel* k = nullptr;
  TF_RETURN_IF_ERROR(flr->CreateKernel(props, &k));
  result->reset(k);
  return Status::OK();
}

}  // namespace

StatusOr<OpKernelRunner> OpKernelRunner::Create(
    absl::string_view op_name, absl::string_view device_name, int num_args,
    const std::function<Status(tensorflow::AttrValueMap*)>& attr_builder,
    const tensorflow::DeviceMgr& device_manager,
    const tensorflow::ProcessFunctionLibraryRuntime&
        process_function_library_runtime) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("op_name: \"" + std::string(op_name.data(), op_name.size()) + "\"");
   mht_3_v.push_back("device_name: \"" + std::string(device_name.data(), device_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTcc mht_3(mht_3_v, 266, "", "./tensorflow/core/tfrt/fallback/op_kernel_runner.cc", "OpKernelRunner::Create");

  tensorflow::Device* device = nullptr;
  Status s = device_manager.LookupDevice(device_name, &device);

  // Fall back to host device if it fails to find the specified device.
  if (!s.ok()) {
    LOG(WARNING) << "Failed to find device " << device_name
                 << " when creating OpKernel: " << op_name << ". Error: " << s;
    LOG(WARNING) << "Fallback to host device instead";
    device = device_manager.HostCPU();
  }

  return Create(op_name, num_args, attr_builder,
                process_function_library_runtime, device);
}

StatusOr<OpKernelRunner> OpKernelRunner::Create(
    absl::string_view op_name, int num_args,
    const std::function<Status(tensorflow::AttrValueMap*)>& attr_builder,
    const tensorflow::ProcessFunctionLibraryRuntime&
        process_function_library_runtime,
    tensorflow::Device* device) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("op_name: \"" + std::string(op_name.data(), op_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTcc mht_4(mht_4_v, 291, "", "./tensorflow/core/tfrt/fallback/op_kernel_runner.cc", "OpKernelRunner::Create");

  const OpDef* op_def = nullptr;
  TF_RETURN_IF_ERROR(tensorflow::OpRegistry::Global()->LookUpOpDef(
      std::string(op_name), &op_def));
  TF_RETURN_IF_ERROR(CheckOpDefCompatibility(*op_def));
  VLOG(1) << "KernelFallbackExecuteCompat creating op from OpDef: "
          << op_def->DebugString();

  TF_ASSIGN_OR_RETURN(auto node_def,
                      BuildNodeDef(*op_def, num_args, attr_builder));

  VLOG(1) << "KernelFallbackExecuteCompat created NodeDef: "
          << node_def.DebugString();

  tensorflow::FunctionLibraryRuntime* function_library_runtime = nullptr;

  function_library_runtime =
      process_function_library_runtime.GetFLR(device->name());

  std::unique_ptr<OpKernel> op_kernel;
  TF_RETURN_IF_ERROR(CreateOpKernel(function_library_runtime,
                                    std::move(node_def), &op_kernel));
  return OpKernelRunner(device, function_library_runtime, std::move(op_kernel));
}

OpKernelRunner::OpKernelRunner(
    tensorflow::Device* device,
    tensorflow::FunctionLibraryRuntime* function_library_runtime,
    std::unique_ptr<tensorflow::OpKernel> op_kernel)
    : device_(device),
      function_library_runtime_(function_library_runtime),
      resource_manager_(device->resource_manager()),
      op_kernel_(std::move(op_kernel)),
      is_async_(op_kernel_->AsAsync() != nullptr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTcc mht_5(mht_5_v, 327, "", "./tensorflow/core/tfrt/fallback/op_kernel_runner.cc", "OpKernelRunner::OpKernelRunner");

  DCHECK(device_);
  DCHECK(function_library_runtime_);

  const auto& input_memory_types = op_kernel_->input_memory_types();
  input_alloc_attrs_.resize(op_kernel_->num_inputs());
  for (size_t i = 0, e = op_kernel_->num_inputs(); i < e; ++i) {
    input_alloc_attrs_[i].set_on_host(input_memory_types[i] ==
                                      tensorflow::HOST_MEMORY);
  }
  const auto& output_memory_types = op_kernel_->output_memory_types();
  output_alloc_attrs_.resize(op_kernel_->num_outputs());
  for (size_t i = 0, e = output_alloc_attrs_.size(); i < e; ++i) {
    output_alloc_attrs_[i].set_on_host(output_memory_types[i] ==
                                       tensorflow::HOST_MEMORY);
  }
}

void OpKernelRunner::RunAsync(OpKernelContext* context,
                              AsyncOpKernel::DoneCallback done_callback) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTcc mht_6(mht_6_v, 349, "", "./tensorflow/core/tfrt/fallback/op_kernel_runner.cc", "OpKernelRunner::RunAsync");

  DVLOG(1) << "KernelFallbackExecuteCompat Running Async Op: "
           << op_kernel_->def().DebugString()
           << ", on Device: " << device_->name();

  AsyncOpKernel* async = op_kernel_->AsAsync();
  DCHECK(async);

  async->ComputeAsync(context, std::move(done_callback));
}

}  // namespace tfrt_stub
}  // namespace tensorflow
