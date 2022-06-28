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
class MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cache_testDTcc {
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
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cache_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cache_testDTcc() {
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

#include <memory>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/tfrt/eager/c_api_tfrt.h"

namespace tfrt {
namespace tf {
namespace {

using tensorflow::Status;
using tensorflow::StatusFromTF_Status;
using tensorflow::TF_StatusPtr;

constexpr char kCpuName[] = "/job:localhost/replica:0/task:0/device:CPU:0";
constexpr char kFunctionName[] = "test_fn";

class CppTests : public ::testing::TestWithParam<const char*> {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cache_testDTcc mht_0(mht_0_v, 219, "", "./tensorflow/core/tfrt/eager/function_cache_test.cc", "SetUp");

    TF_StatusPtr status(TF_NewStatus());
    TF_SetTracingImplementation(GetParam(), status.get());
    Status s = StatusFromTF_Status(status.get());
    CHECK_EQ(tensorflow::errors::OK, s.code()) << s.error_message();
  }
};

// Computes `inputs[0] + inputs[1]` and records it on the tape.
tensorflow::Status Add(
    tensorflow::AbstractContext* ctx,
    absl::Span<tensorflow::AbstractTensorHandle* const> inputs,
    absl::Span<tensorflow::AbstractTensorHandle*> outputs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cache_testDTcc mht_1(mht_1_v, 234, "", "./tensorflow/core/tfrt/eager/function_cache_test.cc", "Add");

  tensorflow::AbstractOperationPtr add_op(ctx->CreateOperation());

  TF_RETURN_IF_ERROR(add_op.get()->Reset("Add", /*raw_device_name=*/nullptr));

  if (isa<tensorflow::tracing::TracingOperation>(add_op.get())) {
    TF_RETURN_IF_ERROR(
        dyn_cast<tensorflow::tracing::TracingOperation>(add_op.get())
            ->SetOpName("my_add"));
  }

  TF_RETURN_IF_ERROR(add_op.get()->AddInput(inputs[0]));
  TF_RETURN_IF_ERROR(add_op.get()->AddInput(inputs[1]));
  int num_retvals = 1;
  return add_op.get()->Execute(outputs, &num_retvals);
}

// Computes
// return inputs[0] + inputs[1]
tensorflow::Status AddModel(
    tensorflow::AbstractContext* ctx,
    absl::Span<tensorflow::AbstractTensorHandle* const> inputs,
    absl::Span<tensorflow::AbstractTensorHandle*> outputs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cache_testDTcc mht_2(mht_2_v, 259, "", "./tensorflow/core/tfrt/eager/function_cache_test.cc", "AddModel");

  std::vector<tensorflow::AbstractTensorHandle*> add_outputs(1);
  // Compute x+y.
  TF_RETURN_IF_ERROR(Add(ctx, inputs, absl::MakeSpan(add_outputs)));

  outputs[0] = add_outputs[0];
  return tensorflow::Status::OK();
}

tensorflow::AbstractContext* BuildFunction(const char* fn_name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("fn_name: \"" + (fn_name == nullptr ? std::string("nullptr") : std::string((char*)fn_name)) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cache_testDTcc mht_3(mht_3_v, 272, "", "./tensorflow/core/tfrt/eager/function_cache_test.cc", "BuildFunction");

  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_ExecutionContext* graph_ctx = TF_CreateFunction(fn_name, status.get());
  return tensorflow::unwrap(graph_ctx);
}

tensorflow::Status CreateParamsForInputs(
    tensorflow::AbstractContext* ctx,
    absl::Span<tensorflow::AbstractTensorHandle* const> inputs,
    std::vector<tensorflow::AbstractTensorHandle*>* params) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cache_testDTcc mht_4(mht_4_v, 285, "", "./tensorflow/core/tfrt/eager/function_cache_test.cc", "CreateParamsForInputs");

  tensorflow::tracing::TracingTensorHandle* handle = nullptr;
  for (auto input : inputs) {
    tensorflow::PartialTensorShape shape;
    TF_RETURN_IF_ERROR(input->Shape(&shape));
    TF_RETURN_IF_ERROR(
        dyn_cast<tensorflow::tracing::TracingContext>(ctx)->AddParameter(
            input->DataType(), shape, &handle));
    params->emplace_back(handle);
  }
  return tensorflow::Status::OK();
}

using Model = std::function<tensorflow::Status(
    tensorflow::AbstractContext*,
    absl::Span<tensorflow::AbstractTensorHandle* const>,
    absl::Span<tensorflow::AbstractTensorHandle*>)>;

tensorflow::Status PrepareFunction(
    Model model, tensorflow::AbstractContext* ctx,
    absl::Span<tensorflow::AbstractTensorHandle* const> inputs,
    absl::Span<tensorflow::AbstractTensorHandle*> outputs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cache_testDTcc mht_5(mht_5_v, 309, "", "./tensorflow/core/tfrt/eager/function_cache_test.cc", "PrepareFunction");

  tensorflow::core::RefCountPtr<tensorflow::AbstractFunction> scoped_func;

  tensorflow::AbstractContextPtr func_ctx(BuildFunction(kFunctionName));
  std::vector<tensorflow::AbstractTensorHandle*> func_inputs;
  func_inputs.reserve(inputs.size());
  TF_RETURN_IF_ERROR(
      CreateParamsForInputs(func_ctx.get(), inputs, &func_inputs));
  tensorflow::OutputList output_list;
  output_list.expected_num_outputs = outputs.size();
  output_list.outputs.resize(outputs.size());
  TF_RETURN_IF_ERROR(model(func_ctx.get(), absl::MakeSpan(func_inputs),
                           absl::MakeSpan(output_list.outputs)));
  for (auto func_input : func_inputs) {
    func_input->Unref();
  }
  tensorflow::AbstractFunction* func = nullptr;
  TF_RETURN_IF_ERROR(
      dyn_cast<tensorflow::tracing::TracingContext>(func_ctx.get())
          ->Finalize(&output_list, &func));
  scoped_func.reset(func);
  for (auto output : output_list.outputs) {
    output->Unref();
  }
  TF_RETURN_IF_ERROR(ctx->RegisterFunction(func));

  return tensorflow::Status::OK();
}

tensorflow::Status BuildImmediateExecutionContext(
    bool use_tfrt, tensorflow::AbstractContext** ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cache_testDTcc mht_6(mht_6_v, 342, "", "./tensorflow/core/tfrt/eager/function_cache_test.cc", "BuildImmediateExecutionContext");

  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetTfrt(opts, use_tfrt);
  *ctx = tensorflow::unwrap(TF_NewEagerExecutionContext(opts, status.get()));
  TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(status.get()));
  TFE_DeleteContextOptions(opts);
  return tensorflow::Status::OK();
}

tensorflow::Status TestScalarTensorHandle(
    tensorflow::AbstractContext* ctx, float value,
    tensorflow::AbstractTensorHandle** tensor) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cache_testDTcc mht_7(mht_7_v, 358, "", "./tensorflow/core/tfrt/eager/function_cache_test.cc", "TestScalarTensorHandle");

  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_Context* eager_ctx =
      TF_ExecutionContextGetTFEContext(wrap(ctx), status.get());
  TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(status.get()));
  TFE_TensorHandle* input_eager = TestScalarTensorHandle(eager_ctx, value);
  *tensor = tensorflow::unwrap(
      TF_CreateAbstractTensorFromEagerTensor(input_eager, status.get()));
  return tensorflow::Status::OK();
}

TEST_P(CppTests, TestFunctionCacheWithAdd) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  tensorflow::AbstractContextPtr ctx;
  {
    tensorflow::AbstractContext* ctx_raw = nullptr;
    tensorflow::Status s = BuildImmediateExecutionContext(true, &ctx_raw);
    ASSERT_EQ(tensorflow::errors::OK, s.code()) << s.error_message();
    ctx.reset(ctx_raw);
  }

  tensorflow::AbstractTensorHandlePtr x;
  {
    tensorflow::AbstractTensorHandle* x_raw = nullptr;
    tensorflow::Status s = TestScalarTensorHandle(ctx.get(), 2.0f, &x_raw);
    ASSERT_EQ(tensorflow::errors::OK, s.code()) << s.error_message();
    x.reset(x_raw);
  }

  tensorflow::AbstractTensorHandlePtr y;
  {
    tensorflow::AbstractTensorHandle* y_raw = nullptr;
    tensorflow::Status s = TestScalarTensorHandle(ctx.get(), 2.0f, &y_raw);
    ASSERT_EQ(tensorflow::errors::OK, s.code()) << s.error_message();
    y.reset(y_raw);
  }

  // Pseudo-code:
  // outputs = x + y
  tensorflow::Status s;
  std::vector<tensorflow::AbstractTensorHandle*> outputs(1);
  s = PrepareFunction(AddModel, ctx.get(), {x.get(), y.get()},
                      absl::MakeSpan(outputs));

  ::tfrt::tf::FunctionCache cache;
  ::tfrt::tf::ContextInterface* tfrt_ctx =
      static_cast<::tfrt::tf::ContextInterface*>(ctx.get());
  ::tfrt::CoreRuntime* corert = tfrt_ctx->GetCoreRuntime();
  tensorflow::EagerContext* eager_ctx = tfrt_ctx->GetEagerContext();

  // Cache is empty initially.
  ASSERT_EQ(cache.Size(), 0);
  ASSERT_EQ(cache.Contains(kFunctionName, kCpuName), false);

  tensorflow::DeviceSet dev_set;
  const tensorflow::DeviceMgr* device_mgr =
      tfrt_ctx->GetEagerContext()->local_device_mgr();
  for (auto d : device_mgr->ListDevices()) dev_set.AddDevice(d);
  auto& device = corert->GetHostContext()->GetHostDevice();
  const Device* input_devices[2] = {&device, &device};
  auto req_ctx = RequestContextBuilder(corert->GetHostContext(),
                                       /*resource_context=*/nullptr)
                     .build();
  ExecutionContext exec_ctx(std::move(*req_ctx));

  auto request_ctx_fn =
      [host = corert->GetHostContext()](
          tensorflow::tfrt_stub::OpKernelRunnerTable* runner_table,
          RCReference<RequestContext>* request_ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSfunction_cache_testDTcc mht_8(mht_8_v, 431, "", "./tensorflow/core/tfrt/eager/function_cache_test.cc", "lambda");

        *request_ctx =
            std::move(*RequestContextBuilder(host,
                                             /*resource_context=*/nullptr)
                           .build());
        return Status::OK();
      };

  // Inserts a new cache entry.
  FunctionCache::FunctionCacheResult result;
  TF_ASSERT_OK(cache.GetOrAddFunction(
      kFunctionName, kCpuName, dev_set, eager_ctx, corert, request_ctx_fn,
      /*loc=*/{}, tensorflow::TfrtFunctionCompileOptions(), input_devices,
      &result));
  ASSERT_NE(result.function_state.get(), nullptr);
  // Cache contains the inserted entry now.
  ASSERT_EQ(cache.Contains(kFunctionName, kCpuName), true);

  // There's one entry in the cache.
  ASSERT_EQ(cache.Size(), 1);

  // This lookup is a cache hit.
  TF_ASSERT_OK(cache.GetOrAddFunction(
      kFunctionName, kCpuName, dev_set, eager_ctx, corert, request_ctx_fn,
      /*loc=*/{}, tensorflow::TfrtFunctionCompileOptions(), input_devices,
      &result));
  ASSERT_NE(result.function_state.get(), nullptr);
  // Cache hit doesn't create new entry in the cache.
  ASSERT_EQ(cache.Size(), 1);

  // Add another entry with the same function name but different device name.
  // This lookup is a cache miss.
  TF_ASSERT_OK(cache.GetOrAddFunction(
      kFunctionName, "", dev_set, eager_ctx, corert, request_ctx_fn,
      /*loc=*/{}, tensorflow::TfrtFunctionCompileOptions(), input_devices,
      &result));
  ASSERT_NE(result.function_state.get(), nullptr);
  // Cache miss adds a new entry in the cache.
  ASSERT_EQ(cache.Size(), 2);

  cache.RemoveFunction(kFunctionName);

  // RemoveFunction removes all entries in the cache since they have the same
  // function name.
  ASSERT_EQ(cache.Size(), 0);
}

INSTANTIATE_TEST_SUITE_P(UnifiedCAPI, CppTests, ::testing::Values("graphdef"));

}  // namespace
}  // namespace tf
}  // namespace tfrt
