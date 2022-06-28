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
#ifndef TENSORFLOW_CORE_TFRT_EAGER_C_API_TFRT_H_
#define TENSORFLOW_CORE_TFRT_EAGER_C_API_TFRT_H_
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
class MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh {
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
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh() {
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


#include <memory>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/c/eager/abstract_op_attrs.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/saved_model_api.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/tfrt/eager/function_cache.h"
#include "tensorflow/core/tfrt/eager/op_cache.h"
#include "tensorflow/core/tfrt/eager/tfrt_context.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tfrt/bef_converter/bef_attr_encoder.h"  // from @tf_runtime
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime_op.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/value.h"  // from @tf_runtime
#include "tfrt/support/aligned_buffer.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime

namespace tfrt {

class CoreRuntime;
class CoreRuntimeOp;
class DenseHostTensor;
class OpHandler;
class TensorHandle;
class TensorMetadata;

namespace tf {
class EagerOpHandlerSelector;

class ContextInterface : public tensorflow::ImmediateExecutionContext {
 public:
  ContextInterface(
      const tensorflow::SessionOptions& opts,
      tensorflow::ContextDevicePlacementPolicy default_device_placement_policy,
      bool is_async, bool use_tfrt_distributed_runtime);
  ~ContextInterface() override;

  void Release() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_0(mht_0_v, 244, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "Release");
 delete this; }

  tensorflow::AbstractTensorInterface* CreateInt64Scalar(
      int64_t value) override;
  tensorflow::AbstractTensorInterface* CreateUint64Scalar(
      uint64_t value) override;
  tensorflow::AbstractTensorInterface* CreateInt32Scalar(
      int32_t value) override;
  tensorflow::AbstractTensorInterface* CreateFloatScalar(float value) override;
  tensorflow::AbstractTensorInterface* CreateDoubleScalar(
      double value) override;
  tensorflow::AbstractTensorInterface* CreateHalfScalar(
      Eigen::half value) override;
  tensorflow::AbstractTensorInterface* CreateStringScalar(
      tensorflow::tstring value) override;
  tensorflow::AbstractTensorInterface* CreateComplex128Scalar(
      tensorflow::complex128 value) override;
  tensorflow::AbstractTensorInterface* CreateBoolScalar(bool value) override;

  tensorflow::AbstractTensorInterface* CreateTensor(
      tensorflow::DataType dtype, absl::Span<const int64_t> dim_sizes) override;
  tensorflow::AbstractTensorInterface* CreateTensor(
      tensorflow::DataType dtype, const int64_t* dims, int num_dims, void* data,
      size_t len, MemoryReleaser memory_releaser,
      void* memory_releaser_arg) override;

  tensorflow::ImmediateExecutionTensorHandle* CreateLocalHandle(
      tensorflow::AbstractTensorInterface* t) override;
  // Create an abstract tensor handle from tensorflow::Tensor.
  tensorflow::ImmediateExecutionTensorHandle* CreateLocalHandleFromTFTensor(
      tensorflow::Tensor& t, const char* d_name) override;

  // Convert a TFRT TensorHandle to tensorflow::TensorHandle.
  tensorflow::ImmediateExecutionTensorHandle* TFTensorHandleFromInterface(
      tensorflow::ImmediateExecutionTensorHandle* handle) override;

  tensorflow::ImmediateExecutionTensorHandle* CopyTensorHandleToDevice(
      tensorflow::ImmediateExecutionTensorHandle* handle,
      const char* device_name, tensorflow::Status* status) override;

  tensorflow::ImmediateExecutionOperation* CreateOperation() override;
  tensorflow::Status RegisterFunction(tensorflow::AbstractFunction*) override;

  tensorflow::CustomDeviceOpHandler& GetCustomDeviceOpHandler() override;

  tensorflow::Status RegisterCustomDevice(
      const std::string& name,
      std::unique_ptr<tensorflow::CustomDevice> device) override;

  tensorflow::FunctionLibraryDefinition* FuncLibDef() override;

  void SetReuseRendezvousForFunctions(
      bool reuse_rendezvous_for_functions) override;

  void ResetGlobalRendezvousForFunction() override;

  bool UsesTFRT() override;

  void ListDevices(std::vector<tensorflow::DeviceAttributes>* devices) override;

  std::vector<tensorflow::Device*> ListLocalTfDevices() override {
    return context_.GetEagerContext()->local_device_mgr()->ListDevices();
  }

  std::vector<tensorflow::Device*> ListAllTfDevices() override {
    return context_.GetEagerContext()->ListAllTfDevices();
  }

  tensorflow::Status AddDevices(
      std::vector<std::unique_ptr<tensorflow::Device>> devices) override;

  void ClearCachesAndThreadExecutors() override;
  void StartStep() override;
  void EndStep() override;

  tensorflow::Status AsyncWait() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_1(mht_1_v, 322, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "AsyncWait");

    TF_RETURN_IF_ERROR(GetEagerContext()->AsyncWait());
    GetHostContext()->Quiesce();
    return tensorflow::Status::OK();
  }

  tensorflow::Status AddFunctionDef(
      const tensorflow::FunctionDef& fdef) override;
  tensorflow::Status AddFunctionDefWithStackTraces(
      const tensorflow::FunctionDef& fdef,
      const tensorflow::StackTracesMap& stack_traces) override;
  std::vector<std::string> ListFunctionNames() override;
  tensorflow::Status RemoveFunction(const std::string& func) override;
  const tensorflow::FunctionDef* FindFunctionDef(
      const std::string& name) const override;

  const tensorflow::DeviceNameUtils::ParsedName& HostCPUParsedName()
      const override;
  const std::string& HostCPUName() const override;

  void SetAllowSoftPlacement(bool enable) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_2(mht_2_v, 345, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "SetAllowSoftPlacement");

    // TODO(tfrt-devs): Move this flag to a common place that can be shared
    // by current TF and TFRT.
    GetEagerContext()->SetAllowSoftPlacement(enable);
  }
  void SetShouldStoreGraphs(bool value) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_3(mht_3_v, 353, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "SetShouldStoreGraphs");

    GetEagerContext()->SetShouldStoreGraphs(value);
  }

  tensorflow::Status EnableCollectiveOps(
      const tensorflow::ServerDef& server_def) override;

  std::unique_ptr<tensorflow::RunMetadata> ExportRunMetadata() override;

  // Find the FunctionDef by the given name and record it in RunMetadata.
  tensorflow::Status RunMetadataRecordFunction(const std::string& func_name);

  void SetLogDevicePlacement(bool enable) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_4(mht_4_v, 368, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "SetLogDevicePlacement");

    // TODO(tfrt-devs): Move this flag to a common place that can be shared
    // by current TF and TFRT.
    GetEagerContext()->SetLogDevicePlacement(enable);
  }

  void SetRunEagerOpAsFunction(bool enable) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_5(mht_5_v, 377, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "SetRunEagerOpAsFunction");

    // TODO(tfrt-devs): Move this flag to a common place that can be shared
    // by current TF and TFRT.
    GetEagerContext()->SetRunEagerOpAsFunction(enable);
  }

  void SetJitCompileRewrite(bool enable) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_6(mht_6_v, 386, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "SetJitCompileRewrite");

    // TODO(tfrt-devs): Move this flag to a common place that can be shared
    // by current TF and TFRT.
    GetEagerContext()->SetJitCompileRewrite(enable);
  }

  tensorflow::EagerExecutor& Executor() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_7(mht_7_v, 395, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "Executor");

    return GetEagerContext()->Executor();
  }
  void SetExecutorForThread(tensorflow::EagerExecutor* executor) override;

  void SetThreadLocalDevicePlacementPolicy(
      tensorflow::ContextDevicePlacementPolicy policy) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_8(mht_8_v, 404, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "SetThreadLocalDevicePlacementPolicy");

    // TODO(tfrt-devs): Move this flag to a common place that can be shared
    // by current TF and TFRT.
    GetEagerContext()->SetThreadLocalDevicePlacementPolicy(policy);
  }
  tensorflow::ContextDevicePlacementPolicy GetDevicePlacementPolicy()
      const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_9(mht_9_v, 413, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "GetDevicePlacementPolicy");

    // TODO(tfrt-devs): Move this flag to a common place that can be shared
    // by current TF and TFRT.
    return GetEagerContext()->GetDevicePlacementPolicy();
  }

  CoreRuntime* GetCoreRuntime();
  tensorflow::Status BuildFunctionRequestContext(
      tensorflow::tfrt_stub::OpKernelRunnerTable* runner_table,
      RCReference<tfrt::RequestContext>* request_context);
  tensorflow::Status BuildOpRequestContext(
      RCReference<tfrt::RequestContext>* request_context);
  tensorflow::EagerContext* GetEagerContext();
  const tensorflow::EagerContext* GetEagerContext() const;
  TfrtContext* GetTfrtContext();

  // Selects the op handler to execute the op based on the arguments. This
  // op handler selection is cheap. But it can be nullptr even it return OK
  // status.
  tensorflow::Status SelectOpHandlerFromArguments(
      const tensorflow::ImmediateExecutionOperation& op,
      OpHandler** op_handler);

  // Selects the op handler to execute the op based on NodeDef. This op handler
  // selection is expensive. It will never return nullptr unless there is an
  // error. Please only invoke this method when the cheap version fails.
  tensorflow::Status SelectOpHandlerFromNodeDef(
      const tensorflow::ImmediateExecutionOperation& op,
      const tensorflow::NodeDef* node_def, OpHandler** op_handler);

  // Returns the chain for current thread.
  AsyncValueRef<Chain>* GetChain();

  // Indicates sync or async execution.
  bool IsAsync() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_10(mht_10_v, 450, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "IsAsync");
 return context_.IsAsync(); }

  // For LLVM style RTTI.
  static bool classof(const AbstractContext* op) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_11(mht_11_v, 456, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "classof");

    return op->getKind() == kTfrt;
  }

  FunctionCache& GetFunctionCache() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_12(mht_12_v, 463, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "GetFunctionCache");
 return function_cache_; }

  OpCache& GetOpCache() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_13(mht_13_v, 468, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "GetOpCache");
 return op_cache_; }

  OpHandler* GetFallbackOpHandler();

  std::vector<std::string> GetLoggedOpsTestonly() override;

  bool UseTfrtDistributedRuntime() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_14(mht_14_v, 477, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "UseTfrtDistributedRuntime");
 return use_tfrt_distributed_runtime_; }

#if !defined(IS_MOBILE_PLATFORM)
  void SetDistributedManager(
      std::unique_ptr<tensorflow::ImmediateExecutionDistributedManager>
          distributed) override {
    distributed_manager_ = std::move(distributed);
  }

  tensorflow::ImmediateExecutionDistributedManager* GetDistributedManager()
      override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_15(mht_15_v, 490, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "GetDistributedManager");

    if (use_tfrt_distributed_runtime_) {
      return distributed_manager_.get();
    } else {
      return context_.GetEagerContext()->GetDistributedManager();
    }
  }
#endif  // !IS_MOBILE_PLATFORM

 private:
  HostContext* GetHostContext();
  ResourceContext* GetResourceContext();

  Expected<OpHandler*> GetOpHandler(const char* name);

  TfrtContext context_;

  mutable tensorflow::mutex chain_map_mu_;
  // TODO(chuanhao): Hook it up with C API to allow user to manage it.
  // Each caller thread will have its own chain to dispatch ops.
  std::unordered_map<std::thread::id, AsyncValueRef<Chain>> thread_local_chain_
      TF_GUARDED_BY(chain_map_mu_);

  std::unique_ptr<EagerOpHandlerSelector> op_handler_selector_;

  // The cache that stores functions (composite CoreRuntimeOps).
  FunctionCache function_cache_;

  // The cache that stores CoreRuntimeOps. It's separate from function cache
  // since a primitive CoreRuntimeOp is essentially a stateless function
  // pointer, and so it doesn't need ref-count to manage its lifetime.
  OpCache op_cache_;

  mutex run_metadata_mu_;
  std::unique_ptr<tensorflow::RunMetadata> run_metadata_
      TFRT_GUARDED_BY(run_metadata_mu_);

  // Use TFRT's implementation of distributed manager.
  bool use_tfrt_distributed_runtime_ = false;

  // A distributed manager that helps setup, update, and check liveness of
  // member tasks in the cluster.
  std::unique_ptr<tensorflow::ImmediateExecutionDistributedManager>
      distributed_manager_;
};

class TensorInterface : public tensorflow::AbstractTensorInterface {
 public:
  explicit TensorInterface(AsyncValueRef<Tensor> t) : tensor_(std::move(t)) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_16(mht_16_v, 541, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "TensorInterface");
}
  explicit TensorInterface(tensorflow::Tensor t) : tf_tensor_(std::move(t)) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_17(mht_17_v, 545, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "TensorInterface");
}
  ~TensorInterface() override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_18(mht_18_v, 549, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "~TensorInterface");
}

  void Release() override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_19(mht_19_v, 554, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "Release");
 delete this; }

  tensorflow::DataType Type() const override;
  int NumDims() const override;
  int64_t Dim(int dim_index) const override;
  int64_t NumElements() const override;
  size_t ByteSize() const override;
  void* Data() const override;
  bool IsAligned() const override;
  bool CanMove() const override;
  bool IsTfTensor() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_20(mht_20_v, 567, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "IsTfTensor");
 return !tensor_; }
  std::string SummarizeValue() const override;

  AsyncValueRef<tfrt::Tensor> TensorRef() const;
  tensorflow::Tensor& TfTensor() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_21(mht_21_v, 574, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "TfTensor");
 return tf_tensor_; }

 private:
  AsyncValueRef<tfrt::Tensor> tensor_;
  // NOTE(b/167608876): tensorflow::Tensor for handling non-scalar string
  // tensors, for backward compatibility. This is a temporary workaround until
  // we find a proper way to unify tensorflow::tstring and
  // tfrt::StringHostTensor.
  tensorflow::Tensor tf_tensor_;
};

class TensorHandleInterface
    : public tensorflow::ImmediateExecutionTensorHandle {
 public:
  explicit TensorHandleInterface(Value&& v, TfrtContext* context);

  explicit TensorHandleInterface(tensorflow::DataType dtype, Value&& v,
                                 TfrtContext* context);

  void Release() override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_22(mht_22_v, 596, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "Release");
 Unref(); }

  tensorflow::DataType DataType() const override;
  tensorflow::Status TensorHandleStatus() const override;
  tensorflow::Status Shape(
      tensorflow::PartialTensorShape* shape) const override;
  tensorflow::Status NumDims(int* num_dims) const override;
  tensorflow::Status NumElements(int64_t* num_elements) const override;
  tensorflow::Status Dim(int dim_index, int64_t* dim) const override;

  // DeviceName represents the device that creates the tensor handle.
  // Currently the same with BackingDeviceName.
  // TODO(b/169341326): unify device behavior between current TF and TFRT.
  const char* DeviceName(tensorflow::Status* status) const override;

  // BackingDeviceName represents the device where the tensor is physically
  // placed. DeviceName and BackingDeviceName are the same for TFRT.
  const char* BackingDeviceName(tensorflow::Status* status) const override;

  const char* DeviceType(tensorflow::Status* status) const override;

  int DeviceId(tensorflow::Status* status) const override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_23(mht_23_v, 620, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "DeviceId");

    // TODO(tfrt-devs): implement for tfrt tensor handle.
    llvm_unreachable("unimplemented method.");
  }

  tensorflow::AbstractTensorInterface* Resolve(
      tensorflow::Status* status) override;

  // TODO(b/161897666): Figure out if we can get rid of returning a new
  // pointer here and just use Ref().
  tensorflow::ImmediateExecutionTensorHandle* Copy() override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_24(mht_24_v, 633, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "Copy");

    Ref();
    return this;
  }

  TensorHandle Handle() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_25(mht_25_v, 641, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "Handle");
 return value_.get<TensorHandle>().CopyRef(); }

  Value* value() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_26(mht_26_v, 646, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "value");
 return &value_; }

  // For LLVM style RTTI.
  static bool classof(const tensorflow::AbstractTensorHandle* ptr) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_27(mht_27_v, 652, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "classof");

    return ptr->getKind() == kTfrt;
  }

 private:
  llvm::Optional<const TensorMetadata*> Metadata() const;

  tensorflow::StatusOr<tensorflow::DataType> ObtainDataTypeFromMetaData(
      const TensorMetadata*) const;

  // If the tensor handle is generated as the result of a function, the datatype
  // is known from the function output signature.
  // Therefore, we can obtain the datatype earlier, before the function
  // execution completes.
  llvm::Optional<tensorflow::DataType> dtype_;

  TfrtContext& context_;

  // Value of tfrt::TensorHandle.
  Value value_;
};

template <typename T>
inline TensorHandleInterface* TensorHandleFromInterface(T* handle) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_28(mht_28_v, 678, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "TensorHandleFromInterface");

  return tensorflow::down_cast<TensorHandleInterface*>(handle);
}

// TFRT location handler class that simply prints the error and abort the
// program on encountering any error. It's primarily for easy debugging
// TODO(kkb): Handle errors probably by raising a Python exception.
class AbortLocationHandler final : public tfrt::LocationHandler {
 public:
  tfrt::Location GetCurrentLocation();

 private:
  tfrt::DecodedLocation DecodeLocation(tfrt::Location loc) const override {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_29(mht_29_v, 693, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "DecodeLocation");

    // Return a dummy decoded location.
    return {};
  }
};

class OpAttrsInterface : public tensorflow::AbstractOpAttrs {
 public:
  explicit OpAttrsInterface(const OpAttrs* attrs,
                            tensorflow::AttrBuilder* fallback_attrs)
      : AbstractOpAttrs(
            tensorflow::AbstractOpAttrs::AbstractOpAttrsKind::kTfrt),
        attrs_(attrs),
        fallback_attrs_(fallback_attrs) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_30(mht_30_v, 709, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "OpAttrsInterface");
}
  ~OpAttrsInterface() override {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_31(mht_31_v, 713, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "~OpAttrsInterface");
}

  void GetNameAttrList(tensorflow::NameAttrList* name_and_attrs) const override;
  tensorflow::Status GetTypeList(
      absl::string_view attr_name,
      absl::InlinedVector<tensorflow::DataType, 4>* type_list) const override;

  bool GetInt(absl::string_view attr_name, int64_t* result) const override;
  bool GetFloat(absl::string_view attr_name, float* result) const override;
  bool GetBool(absl::string_view attr_name, bool* result) const override;
  bool GetType(absl::string_view attr_name,
               tensorflow::DataType* result) const override;

  const OpAttrs* GetAttrs() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_32(mht_32_v, 729, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "GetAttrs");
 return attrs_; }

  const tensorflow::AttrBuilder* GetFallbackAttrs() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_33(mht_33_v, 734, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "GetFallbackAttrs");

    return fallback_attrs_;
  }

 private:
  // TODO(fishx): Move ownership to here.
  const OpAttrs* attrs_;

  // TODO(tfrt-devs): Remove this field and generate NameAttrList from attrs_.
  // Today it is fine since we will set both attrs and fallback_attrs.
  const tensorflow::AttrBuilder* fallback_attrs_;
};

class OperationInterface : public tensorflow::ImmediateExecutionOperation {
 public:
  // All arguments come from ContextInterface.
  explicit OperationInterface(ContextInterface* context);
  ~OperationInterface() override {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_34(mht_34_v, 754, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "~OperationInterface");
}

  void Release() override {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_35(mht_35_v, 759, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "Release");
 delete this; }

  void Clear() override {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_36(mht_36_v, 764, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "Clear");
 args_.clear(); }

  tensorflow::Status Reset(const char* op,
                           const char* raw_device_name) override;
  const std::string& Name() const override {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_37(mht_37_v, 771, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "Name");
 return op_name_; }
  const std::string& DeviceName() const override {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_38(mht_38_v, 775, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "DeviceName");
 return device_name_; }
  tensorflow::Status SetDeviceName(const char* name) override;

  tensorflow::ImmediateExecutionContext* GetContext() const override {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_39(mht_39_v, 781, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "GetContext");

    return context_;
  }
  bool HasCustomDeviceInput() const override {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_40(mht_40_v, 787, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "HasCustomDeviceInput");

    return custom_device_tensor_handle_count_ > 0;
  }

  tensorflow::Status AddInput(tensorflow::AbstractTensorHandle* input) override;
  tensorflow::Status AddInputList(
      absl::Span<tensorflow::AbstractTensorHandle* const> inputs) override;
  tensorflow::Status SetInput(
      size_t index, tensorflow::ImmediateExecutionTensorHandle* input) override;
  absl::Span<tensorflow::ImmediateExecutionTensorHandle* const> GetInputs()
      const override;
  tensorflow::Status Execute(
      absl::Span<tensorflow::AbstractTensorHandle*> retvals,
      int* num_retvals) override;
  const tensorflow::OpDef* OpDef() const override {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_41(mht_41_v, 804, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "OpDef");
 return op_def_; }
  const tensorflow::NodeDef NodeDef() {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_42(mht_42_v, 808, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "NodeDef");
 return fallback_attrs_.BuildNodeDef(); }

  tensorflow::Status SetAttrString(const char* attr_name, const char* data,
                                   size_t length) override;
  tensorflow::Status SetAttrInt(const char* attr_name, int64_t value) override;
  tensorflow::Status SetAttrFloat(const char* attr_name, float value) override;
  tensorflow::Status SetAttrBool(const char* attr_name, bool value) override;
  tensorflow::Status SetAttrType(const char* attr_name,
                                 tensorflow::DataType value) override;
  tensorflow::Status SetAttrShape(const char* attr_name, const int64_t* dims,
                                  const int num_dims) override;
  tensorflow::Status SetAttrFunction(const char* attr_name,
                                     const AbstractOperation* value) override;
  tensorflow::Status SetAttrFunctionName(const char* attr_name,
                                         const char* data,
                                         size_t length) override;
  tensorflow::Status SetAttrTensor(
      const char* attr_name,
      tensorflow::AbstractTensorInterface* tensor) override;
  tensorflow::Status SetAttrStringList(const char* attr_name,
                                       const void* const* values,
                                       const size_t* lengths,
                                       int num_values) override;
  tensorflow::Status SetAttrFloatList(const char* attr_name,
                                      const float* values,
                                      int num_values) override;
  tensorflow::Status SetAttrIntList(const char* attr_name,
                                    const int64_t* values,
                                    int num_values) override;
  tensorflow::Status SetAttrTypeList(const char* attr_name,
                                     const tensorflow::DataType* values,
                                     int num_values) override;
  tensorflow::Status SetAttrBoolList(const char* attr_name,
                                     const unsigned char* values,
                                     int num_values) override;
  tensorflow::Status SetAttrShapeList(const char* attr_name,
                                      const int64_t** dims, const int* num_dims,
                                      int num_values) override;
  tensorflow::Status SetAttrFunctionList(
      const char* attr_name,
      absl::Span<const AbstractOperation*> values) override;

  tensorflow::Status InputLength(const char* input_name, int* length) override;
  tensorflow::Status OutputLength(const char* output_name,
                                  int* length) override;

  const tensorflow::AbstractOpAttrs* GetOpAttrs() const override;
  void AddAttrs(const tensorflow::AbstractOpAttrs* op_attrs) override;

  void SetStackTrace(tensorflow::ManagedStackTrace stack_trace) override {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_43(mht_43_v, 860, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "SetStackTrace");

    stack_trace_ = stack_trace;
  }

  void SetCancellationManager(
      tensorflow::CancellationManager* cancellation_manager) override {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_44(mht_44_v, 868, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "SetCancellationManager");

    // TODO(b/181368626): Support cancellation.
  }

  absl::optional<tensorflow::ManagedStackTrace> GetStackTrace() override {
    return stack_trace_;
  }

  // Currently not supported.
  void SetStepId(int64_t step_id) override {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_45(mht_45_v, 880, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "SetStepId");
}

  // For LLVM style RTTI.
  static bool classof(const AbstractOperation* ptr) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSc_api_tfrtDTh mht_46(mht_46_v, 886, "", "./tensorflow/core/tfrt/eager/c_api_tfrt.h", "classof");

    return ptr->getKind() == kTfrt;
  }

  friend class OpCache;

 private:
  // Initialize op_ field. It can be either a trivial op or a composite op.
  tensorflow::Status Initialize();

  // Note(fishx): This method is copied from current TF. We use it to infer
  // attribute like "T" in order to run device placement logic from current TF.
  void MaybeInferInputAttrs();

  // This field holds a primitive op. If the op represents a function, it
  // will be held by function_state_ below, and this field will be empty.
  CoreRuntimeOp* op_;
  RCReference<FunctionState> function_state_;
  std::string op_name_;
  // The device user requested to place the op on.
  std::string device_name_;
  bool is_function_;
  tfrt::BefAttrEncoder bef_attr_encoder_;
  // TODO(b/165412867): Remove AttrBuilder.
  tensorflow::AttrBuilder fallback_attrs_;
  const tensorflow::OpDef* op_def_;  // op definition from protobuf
  OpAttrs attrs_;
  OpAttrsInterface op_attrs_;
  llvm::SmallVector<
      tensorflow::core::RefCountPtr<tensorflow::ImmediateExecutionTensorHandle>,
      8>
      args_;
  AbortLocationHandler abort_location_handler_;
  ContextInterface* const context_;
  // TODO(kkb): Use tfrt::Location and implement TFRT async stack tracing.
  absl::optional<tensorflow::ManagedStackTrace> stack_trace_;

  int custom_device_tensor_handle_count_ = 0;
};

}  // namespace tf
}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_EAGER_C_API_TFRT_H_
