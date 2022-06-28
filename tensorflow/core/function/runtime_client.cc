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
class MHTracer_DTPStensorflowPScorePSfunctionPSruntime_clientDTcc {
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
   MHTracer_DTPStensorflowPScorePSfunctionPSruntime_clientDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSfunctionPSruntime_clientDTcc() {
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

#include "tensorflow/core/function/runtime_client.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/ir/importexport/export.h"
#include "tensorflow/core/ir/importexport/import.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace core {
namespace function {

EagerContext& GlobalEagerContext() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSfunctionPSruntime_clientDTcc mht_0(mht_0_v, 225, "", "./tensorflow/core/function/runtime_client.cc", "GlobalEagerContext");

  static EagerContext* global_ctx = []() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSfunctionPSruntime_clientDTcc mht_1(mht_1_v, 229, "", "./tensorflow/core/function/runtime_client.cc", "lambda");

    SessionOptions opts;
    std::vector<std::unique_ptr<Device>> devices;
    Status&& device_init_status = DeviceFactory::AddDevices(
        opts, "/job:localhost/replica:0/task:0", &devices);
    CHECK(device_init_status.ok());  // Crash OK

    return new EagerContext(
        opts, ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
        /*async=*/false,
        /*device_mgr=*/new DynamicDeviceMgr(std::move(devices)),
        /*device_mgr_owned=*/true,
        /*rendezvous=*/nullptr,
        /*cluster_flr=*/nullptr,
        /*collective_executor_mgr=*/nullptr,
        /*run_eager_op_as_function=*/false);
  }();
  return *global_ctx;
}

EagerContext& GlobalPythonEagerContext() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSfunctionPSruntime_clientDTcc mht_2(mht_2_v, 252, "", "./tensorflow/core/function/runtime_client.cc", "GlobalPythonEagerContext");

  EagerContext* ctx = reinterpret_cast<EagerContext*>(GetCEagerContext());
  DCHECK(ctx) << "The Python eager context must be initialized first.";
  return *ctx;
}

StatusOr<FunctionDef> Runtime::GetFunctionProto(StringPiece name) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSfunctionPSruntime_clientDTcc mht_3(mht_3_v, 261, "", "./tensorflow/core/function/runtime_client.cc", "Runtime::GetFunctionProto");

  EagerContext& ctx = this->eager_ctx_;

  const FunctionDef* f = ctx.FindFunctionDef(std::string(name));
  if (f == nullptr) {
    return Status(error::INVALID_ARGUMENT,
                  absl::StrCat("Could not find an attribute for key ", name));
  }

  return *f;
}

Status Runtime::CreateFunction(const FunctionDef& fdef) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSfunctionPSruntime_clientDTcc mht_4(mht_4_v, 276, "", "./tensorflow/core/function/runtime_client.cc", "Runtime::CreateFunction");

  const auto& fname = fdef.signature().name();
  if (this->eager_ctx_.FindFunctionByName(fname)) {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(this->eager_ctx_.RemoveFunction(fname),
                                    "removing function ", fname);
  }
  return this->eager_ctx_.AddFunctionDef(fdef);
}

Status Runtime::CreateFunction(OpaqueTfgGraphFuncOp* fop) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSfunctionPSruntime_clientDTcc mht_5(mht_5_v, 288, "", "./tensorflow/core/function/runtime_client.cc", "Runtime::CreateFunction");

  mlir::tfg::GraphFuncOp fop_proper =
      *reinterpret_cast<mlir::tfg::GraphFuncOp*>(fop);
  return mlir::tfg::ExportFunction(fop_proper, *this->eager_ctx_.FuncLibDef());
}

Status Runtime::TransformFunction(StringPiece name, StringPiece pipeline_name) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSfunctionPSruntime_clientDTcc mht_6(mht_6_v, 297, "", "./tensorflow/core/function/runtime_client.cc", "Runtime::TransformFunction");

  // TODO(mdan): Use a longer-lived context.
  mlir::MLIRContext ctx;
  mlir::PassManager pm(&ctx);

  std::string error;
  llvm::raw_string_ostream error_stream(error);
  // StringPiece doesn't seem to always be compatible with StringRef.
  if (mlir::failed(mlir::parsePassPipeline(std::string(pipeline_name), pm,
                                           error_stream))) {
    return Status(error::INVALID_ARGUMENT,
                  absl::StrCat("locating pass pipeline ", pipeline_name, ": ",
                               error_stream.str()));
  }

  // For now, we roundtrip from proto. Once we have a permanent MLIR
  // representation, we should be able to use it directly.
  auto fn = GetFunctionProto(name);
  TF_RETURN_WITH_CONTEXT_IF_ERROR(fn.status(), "loading function ", name);

  GraphDef graph;
  *graph.mutable_library()->add_function() = *fn;
  tensorflow::GraphDebugInfo debug_info;
  auto mlir_fn = mlir::tfg::ImportGraphDefToMlir(&ctx, debug_info, graph);
  TF_RETURN_WITH_CONTEXT_IF_ERROR(mlir_fn.status(), "importing function ",
                                  name);

  mlir::StatusScopedDiagnosticHandler diagnostics_handler(&ctx);
  if (failed(pm.run(mlir_fn->get()))) {
    return diagnostics_handler.Combine(
        Status(error::INVALID_ARGUMENT,
               absl::StrCat("running pass pipeline ", pipeline_name, ": ")));
  }

  for (auto fn : mlir_fn->get().getBody()->getOps<mlir::tfg::GraphFuncOp>()) {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        CreateFunction(reinterpret_cast<OpaqueTfgGraphFuncOp*>(&fn)),
        absl::StrCat("updating function ", fn.getName().str()));
  }

  return Status::OK();
}

StatusOr<ReturnValues> Runtime::CallFunction(
    StringPiece name, absl::Span<AbstractTensorHandle* const> args) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSfunctionPSruntime_clientDTcc mht_7(mht_7_v, 344, "", "./tensorflow/core/function/runtime_client.cc", "Runtime::CallFunction");

  EagerContext& ctx = this->eager_ctx_;

  ImmediateOpPtr op(ctx.CreateOperation());
  TF_RETURN_WITH_CONTEXT_IF_ERROR(op->Reset(name.data(), nullptr),
                                  "initializing call op for ", name);

  TF_RETURN_WITH_CONTEXT_IF_ERROR(op->AddInputList(args),
                                  "preparing call args for ", name);

  const FunctionDef* fn_def = ctx.GetFunctionDef(string(name));
  int num_retvals = fn_def->signature().output_arg_size();
  int actual_retvals = num_retvals;
  std::vector<ImmediateExecutionTensorHandle*> retvals(num_retvals);
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      op->Execute(absl::MakeSpan(
                      reinterpret_cast<AbstractTensorHandle**>(retvals.data()),
                      num_retvals),
                  &actual_retvals),
      "executing call op for ", name);
  DCHECK(num_retvals == actual_retvals);

  ReturnValues final_returns;
  for (const auto& r : retvals) {
    final_returns.emplace_back(ImmediateTensorHandlePtr(r));
  }

  return final_returns;
}

}  // namespace function
}  // namespace core
}  // namespace tensorflow
