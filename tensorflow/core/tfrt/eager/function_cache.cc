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
class MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cacheDTcc {
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
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cacheDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cacheDTcc() {
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
#include "tensorflow/core/tfrt/eager/function_cache.h"

#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/tfrt/eager/transform_graph_function.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime_op.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace tfrt {
namespace tf {

void FunctionCache::RemoveFunction(string_view op_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("op_name: \"" + std::string(op_name.data(), op_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cacheDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/tfrt/eager/function_cache.cc", "FunctionCache::RemoveFunction");

  mutex_lock l(cache_mu_);
  auto iter = cache_.begin();
  while (iter != cache_.end()) {
    if (iter->first.op_name == op_name) {
      iter = cache_.erase(iter);
    } else {
      ++iter;
    }
  }
}

tensorflow::Status FunctionCache::GetOrAddFunction(
    const std::string& op_name, const std::string& device_name,
    const tensorflow::DeviceSet& device_set,
    tensorflow::EagerContext* eager_ctx, tfrt::CoreRuntime* corert,
    RequestCtxBuilder request_ctx_fn, Location loc,
    tensorflow::TfrtFunctionCompileOptions compile_options,
    tfrt::ArrayRef<const Device*> input_devices,
    FunctionCache::FunctionCacheResult* result) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("op_name: \"" + op_name + "\"");
   mht_1_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cacheDTcc mht_1(mht_1_v, 227, "", "./tensorflow/core/tfrt/eager/function_cache.cc", "FunctionCache::GetOrAddFunction");

  const CacheKey cache_key{op_name, device_name};
  {
    mutex_lock l(cache_mu_);
    auto& function_state = cache_[cache_key];
    if (function_state) {
      *result = FunctionCache::FunctionCacheResult{function_state, false};
      return tensorflow::Status::OK();
    }
  }

  tensorflow::FunctionLibraryDefinition* func_lib_def = eager_ctx->FuncLibDef();
  const tensorflow::FunctionDef* fdef = func_lib_def->Find(op_name);
  if (fdef == nullptr)
    return tensorflow::errors::NotFound(
        "Cannot find function from FunctionLibraryDefinition ", op_name);

  // Run graph optimizations using current runtime components before converting
  // the graph to MLIR module.
  std::unique_ptr<tensorflow::FunctionBody> fbody;
  TF_RETURN_IF_ERROR(tensorflow::FunctionDefToBodyHelper(
      *fdef, tensorflow::AttrSlice(), func_lib_def, &fbody));

  // Transferring out the graph ownership from fbody.
  auto graph = std::unique_ptr<tensorflow::Graph>(fbody->graph);
  fbody->graph = nullptr;

  tensorflow::GraphDef graph_def;
  graph->ToGraphDef(&graph_def);
  tensorflow::FunctionLibraryDefinition reachable_lib_def =
      func_lib_def->ReachableDefinitions(graph_def);

  TF_RETURN_IF_ERROR(tensorflow::TransformGraphFunction(
      op_name, *fdef, device_name, device_set, eager_ctx,
      compile_options.enable_grappler, &fbody, std::move(graph), input_devices,
      &reachable_lib_def));

  BefBuffer bef_buffer;

  llvm::SmallVector<tfrt::string_view, 4> device_names;
  device_names.reserve(device_set.devices().size());
  for (auto& d : device_set.devices()) {
    device_names.push_back(d->name());
  }

  // Lower FunctionDef to BEF.
  TF_RETURN_IF_ERROR(tensorflow::ConvertFunctionToBef(
      op_name, fbody.get(), reachable_lib_def, device_names, compile_options,
      &bef_buffer));

  HostContext* host_ctx = corert->GetHostContext();
  auto bef_file =
      tfrt::BEFFile::Open(bef_buffer, host_ctx->GetKernelRegistry(),
                          host_ctx->diag_handler(), host_ctx->allocator());
  if (!bef_file)
    return tensorflow::errors::Internal(
        "Failed to open lowered BEF for function ", op_name, ".");

  const tfrt::Function* function = bef_file->GetFunction(op_name);
  if (!function)
    return tensorflow::errors::Internal(
        "Failed to get function from BEF for function ", op_name, ".");

  auto expected_fn = corert->MakeCompositeOp(function);
  if (!expected_fn)
    return tensorflow::errors::Internal(StrCat("Construct CoreRuntimeOp for ",
                                               op_name.c_str(), " failed. ",
                                               expected_fn.takeError()));

  TfrtDataTypeVector tfrt_arg_types;
  tensorflow::DataTypeVector tf_ret_types;

  for (const auto& arg_type : fbody->arg_types) {
    tfrt_arg_types.push_back(ConvertTfDTypeToTfrtDType(arg_type));
  }

  for (const auto& ret_type : fbody->ret_types) {
    tf_ret_types.push_back(ret_type);
  }

  auto runner_table =
      absl::make_unique<tensorflow::tfrt_stub::OpKernelRunnerTable>();
  RCReference<RequestContext> request_ctx;
  TF_RETURN_IF_ERROR(request_ctx_fn(runner_table.get(), &request_ctx));

  ExecutionContext exec_ctx{std::move(request_ctx), loc};
  TF_RETURN_IF_ERROR(
      RunRuntimeInitializer(exec_ctx, bef_file.get(), "_tfrt_fallback_init"));

  RCReference<FunctionState> entry = FunctionState::CreateFunctionState(
      tfrt_arg_types, tf_ret_types, std::move(bef_buffer), std::move(bef_file),
      std::move(expected_fn.get()), std::move(runner_table));

  mutex_lock l(cache_mu_);
  // Insert the new entry to cache. If an entry with the same key is already
  // present in the cache at this moment due to race condition, overwrites it.
  cache_[cache_key] = entry;
  *result = FunctionCache::FunctionCacheResult{std::move(entry), true};
  return tensorflow::Status::OK();
}

}  // namespace tf
}  // namespace tfrt
