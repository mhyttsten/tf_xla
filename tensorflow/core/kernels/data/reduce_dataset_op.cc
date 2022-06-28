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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSreduce_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSreduce_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSreduce_dataset_opDTcc() {
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

#include "tensorflow/core/kernels/data/reduce_dataset_op.h"

#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/data/root_dataset.h"
#include "tensorflow/core/platform/resource.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
namespace data {
namespace {

const char kOutputShapes[] = "output_shapes";
const char kOutputTypes[] = "output_types";

}  // namespace

ReduceDatasetOp::ReduceDatasetOp(OpKernelConstruction* ctx)
    : HybridAsyncOpKernel(ctx, "tf_data_reduce_dataset") {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSreduce_dataset_opDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/kernels/data/reduce_dataset_op.cc", "ReduceDatasetOp::ReduceDatasetOp");

  FunctionMetadata::Params params;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("use_inter_op_parallelism",
                                   &params.use_inter_op_parallelism));
  params.use_default_device = false;
  OP_REQUIRES_OK(ctx,
                 FunctionMetadata::Create(ctx, "f", params, &func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

Status ReduceDatasetOp::DoCompute(OpKernelContext* ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSreduce_dataset_opDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/kernels/data/reduce_dataset_op.cc", "ReduceDatasetOp::DoCompute");

  profiler::TraceMe traceme(
      [&] {
        return profiler::TraceMeEncode("ReduceDatasetOp::DoCompute",
                                       {{"id", ctx->step_id()}});
      },
      profiler::kInfo);
  tensorflow::ResourceTagger tag(kTFDataResourceTag,
                                 ctx->op_kernel().type_string());
  DatasetBase* dataset;
  TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(ctx->input(0), &dataset));
  OpInputList inputs;
  TF_RETURN_IF_ERROR(ctx->input_list("initial_state", &inputs));
  std::vector<Tensor> state(inputs.begin(), inputs.end());

  std::unique_ptr<CapturedFunction> captured_func;
  TF_RETURN_IF_ERROR(CapturedFunction::Create(
      ctx, func_metadata_, "other_arguments", &captured_func));

  IteratorContext::Params params(ctx);
  auto function_handle_cache =
      absl::make_unique<FunctionHandleCache>(params.flr);
  params.function_handle_cache = function_handle_cache.get();
  ResourceMgr resource_mgr;
  params.resource_mgr = &resource_mgr;
  CancellationManager cancellation_manager(ctx->cancellation_manager());
  params.cancellation_manager = &cancellation_manager;

  IteratorContext iter_ctx(std::move(params));
  std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func;
  TF_RETURN_IF_ERROR(
      captured_func->Instantiate(&iter_ctx, &instantiated_captured_func));

  std::unique_ptr<IteratorBase> iterator;
  if (ctx->function_library()->device()->device_type() == DEVICE_CPU) {
    DatasetBase* finalized_dataset = nullptr;
    TF_RETURN_IF_ERROR(FinalizeDataset(ctx, dataset, &finalized_dataset));
    core::ScopedUnref unref(finalized_dataset);
    TF_RETURN_IF_ERROR(finalized_dataset->MakeIterator(
        &iter_ctx, /*parent=*/nullptr, "ReduceIterator", &iterator));
  } else {
    TF_RETURN_IF_ERROR(dataset->MakeIterator(&iter_ctx, /*parent=*/nullptr,
                                             "ReduceIterator", &iterator));
  }

  // Iterate through the input dataset.
  while (true) {
    if (ctx->cancellation_manager()->IsCancelled()) {
      return errors::Cancelled("Operation was cancelled");
    }
    std::vector<Tensor> next_input_element;
    bool end_of_input;
    TF_RETURN_IF_ERROR(
        iterator->GetNext(&iter_ctx, &next_input_element, &end_of_input));
    if (end_of_input) {
      break;
    }

    // Run the reduce function to update the current state.
    std::vector<Tensor> args;
    args.reserve(state.size() + next_input_element.size());
    std::copy(state.begin(), state.end(), std::back_inserter(args));
    std::copy(next_input_element.begin(), next_input_element.end(),
              std::back_inserter(args));

    std::vector<Tensor> reduce_func_output;
    TF_RETURN_IF_ERROR(instantiated_captured_func->Run(
        &iter_ctx, std::move(args), &reduce_func_output, /*node=*/nullptr));
    if (reduce_func_output.size() != state.size()) {
      return errors::InvalidArgument(
          "The number of components of the initial state and the "
          "reduce "
          "function output does not match. (initial_state=",
          state.size(), ", output=", reduce_func_output.size(), ").");
    }
    std::swap(reduce_func_output, state);
  }

  TF_RETURN_IF_ERROR(VerifyTypesMatch(output_types_, state));
  TF_RETURN_IF_ERROR(VerifyShapesCompatible(output_shapes_, state));
  for (size_t i = 0; i < state.size(); ++i) {
    ctx->set_output(i, state[i]);
  }
  return Status::OK();
}

namespace {

REGISTER_KERNEL_BUILDER(Name("ReduceDataset").Device(DEVICE_CPU),
                        ReduceDatasetOp);
REGISTER_INPUT_COLOCATION_EXEMPTION("ReduceDataset");

}  // namespace
}  // namespace data
}  // namespace tensorflow
