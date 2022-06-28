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
class MHTracer_DTPStensorflowPScPSeagerPSunified_api_testutilDTcc {
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
   MHTracer_DTPStensorflowPScPSeagerPSunified_api_testutilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPSunified_api_testutilDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/eager/unified_api_testutil.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

AbstractContext* BuildFunction(const char* fn_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("fn_name: \"" + (fn_name == nullptr ? std::string("nullptr") : std::string((char*)fn_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSunified_api_testutilDTcc mht_0(mht_0_v, 200, "", "./tensorflow/c/eager/unified_api_testutil.cc", "BuildFunction");

  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_ExecutionContext* graph_ctx = TF_CreateFunction(fn_name, status.get());
  return unwrap(graph_ctx);
}

Status CreateParamsForInputs(AbstractContext* ctx,
                             absl::Span<AbstractTensorHandle* const> inputs,
                             std::vector<AbstractTensorHandle*>* params) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSeagerPSunified_api_testutilDTcc mht_1(mht_1_v, 212, "", "./tensorflow/c/eager/unified_api_testutil.cc", "CreateParamsForInputs");

  tracing::TracingTensorHandle* handle = nullptr;
  for (auto input : inputs) {
    PartialTensorShape shape;
    TF_RETURN_IF_ERROR(input->Shape(&shape));
    TF_RETURN_IF_ERROR(dyn_cast<tracing::TracingContext>(ctx)->AddParameter(
        input->DataType(), shape, &handle));
    params->emplace_back(handle);
  }
  return Status::OK();
}

// Runs `model` maybe wrapped in a function.
Status RunModel(Model model, AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs, bool use_function) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSeagerPSunified_api_testutilDTcc mht_2(mht_2_v, 230, "", "./tensorflow/c/eager/unified_api_testutil.cc", "RunModel");

  if (use_function) {
    const char* fn_name = "test_fn";
    core::RefCountPtr<AbstractFunction> scoped_func;
    // Returning null tensors from a tf.function is not supported, so we keep
    // track of indices in the model's outputs are nullptr in this set.
    // The FunctionDef only outputs the non-null tensors. We later pad the
    // function op outputs to have nullptrs at the `null_indices`.
    absl::flat_hash_set<int> null_indices;
    {
      AbstractContextPtr func_ctx(BuildFunction(fn_name));
      std::vector<AbstractTensorHandle*> func_inputs;
      func_inputs.reserve(inputs.size());
      TF_RETURN_IF_ERROR(
          CreateParamsForInputs(func_ctx.get(), inputs, &func_inputs));
      std::vector<AbstractTensorHandle*> model_outputs;
      model_outputs.resize(outputs.size());
      TF_RETURN_IF_ERROR(model(func_ctx.get(), absl::MakeSpan(func_inputs),
                               absl::MakeSpan(model_outputs)));
      for (auto func_input : func_inputs) {
        func_input->Unref();
      }
      AbstractFunction* func = nullptr;
      OutputList output_list;
      output_list.expected_num_outputs = 0;
      output_list.outputs.reserve(outputs.size());
      for (int i = 0; i < model_outputs.size(); i++) {
        if (model_outputs[i]) {
          output_list.outputs.emplace_back(model_outputs[i]);
          output_list.expected_num_outputs += 1;
        } else {
          null_indices.insert(i);
        }
      }
      TF_RETURN_IF_ERROR(dyn_cast<tracing::TracingContext>(func_ctx.get())
                             ->Finalize(&output_list, &func));
      scoped_func.reset(func);
      for (auto output : output_list.outputs) {
        output->Unref();
      }
      TF_RETURN_IF_ERROR(ctx->RegisterFunction(func));
    }

    AbstractOperationPtr fn_op(ctx->CreateOperation());
    TF_RETURN_IF_ERROR(fn_op->Reset(fn_name, /*raw_device_name=*/nullptr));
    for (auto input : inputs) {
      TF_RETURN_IF_ERROR(fn_op->AddInput(input));
    }
    int retvals = outputs.size() - null_indices.size();
    std::vector<AbstractTensorHandle*> fn_outputs(retvals);
    TF_RETURN_IF_ERROR(fn_op->Execute(
        absl::Span<AbstractTensorHandle*>(fn_outputs.data(), fn_outputs.size()),
        &retvals));
    int skipped_indices = 0;
    for (int i = 0; i < outputs.size(); i++) {
      if (!null_indices.contains(i)) {
        outputs[i] = fn_outputs[i - skipped_indices];
      } else {
        skipped_indices += 1;
      }
    }
    TF_RETURN_IF_ERROR(ctx->RemoveFunction(fn_name));
    return Status::OK();
  } else {
    return model(ctx, inputs, outputs);
  }
}

Status BuildImmediateExecutionContext(bool use_tfrt, AbstractContext** ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSeagerPSunified_api_testutilDTcc mht_3(mht_3_v, 301, "", "./tensorflow/c/eager/unified_api_testutil.cc", "BuildImmediateExecutionContext");

  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetTfrt(opts, use_tfrt);
  *ctx = unwrap(TF_NewEagerExecutionContext(opts, status.get()));
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status.get()));
  TFE_DeleteContextOptions(opts);
  return Status::OK();
}

Status GetValue(AbstractTensorHandle* t, TF_Tensor** result_tensor) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSeagerPSunified_api_testutilDTcc mht_4(mht_4_v, 315, "", "./tensorflow/c/eager/unified_api_testutil.cc", "GetValue");

  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_TensorHandle* result_t =
      TF_AbstractTensorGetEagerTensor(wrap(t), status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status.get()));
  *result_tensor = TFE_TensorHandleResolve(result_t, status.get());
  return StatusFromTF_Status(status.get());
}

}  // namespace tensorflow
