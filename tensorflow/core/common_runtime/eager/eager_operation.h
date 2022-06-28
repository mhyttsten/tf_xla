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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_OPERATION_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_OPERATION_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh() {
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


#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/managed_stack_trace.h"

namespace tensorflow {

class EagerOperation : public ImmediateExecutionOperation {
 public:
  explicit EagerOperation(tensorflow::EagerContext* ctx)
      : ImmediateExecutionOperation(kEager), ctx_(*ctx), is_function_(false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_0(mht_0_v, 209, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "EagerOperation");
}
  ~EagerOperation() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_1(mht_1_v, 213, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "~EagerOperation");

    for (ImmediateExecutionTensorHandle* h : inputs_) {
      h->Unref();
    }
  }

  void Release() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_2(mht_2_v, 222, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "Release");
 delete this; }

  void Clear() override;
  Status Reset(const char* op, const char* raw_device_name) override {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   mht_3_v.push_back("raw_device_name: \"" + (raw_device_name == nullptr ? std::string("nullptr") : std::string((char*)raw_device_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_3(mht_3_v, 230, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "Reset");

    return Reset(op, raw_device_name, false, nullptr);
  }

  const string& Name() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_4(mht_4_v, 237, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "Name");
 return attrs_.op_name(); }

  const string& DeviceName() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_5(mht_5_v, 242, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "DeviceName");
 return device_name_; }

  ImmediateExecutionContext* GetContext() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_6(mht_6_v, 247, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "GetContext");
 return &ctx_; }

  const DeviceNameUtils::ParsedName& GetDeviceParsedName() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_7(mht_7_v, 252, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "GetDeviceParsedName");

    return device_parsed_name_;
  }

  // Replaces the previous device name with the given one (see
  // AbstractOperation::SetDeviceName for more details).
  //
  // This also resets the internal device pointer, unless the given name refers
  // to a known custom device, in which case the internal device pointer is
  // updated to that device.
  Status SetDeviceName(const char* name) override;

  void SetDevice(VariantDevice device) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_8(mht_8_v, 267, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "SetDevice");

    device_ = device;
    device_name_ = absl::visit(
        [](auto* device) { return device == nullptr ? "" : device->name(); },
        device);
    DeviceNameUtils::ParseFullName(device_name_, &device_parsed_name_);
    // TODO(b/154133594): Due to intricacies of external logic, we can not
    // set this do device_name_ as it would be natural, because we need the
    // next call to SetDeviceName to reset the device pointer.
    last_set_device_name_ = "\177";  // DEL (an invalid value)
  }

  Status SetAttrValue(const char* attr_name, const AttrValue& value);

  Status AddInput(AbstractTensorHandle* input) override;
  Status AddInputList(absl::Span<AbstractTensorHandle* const> inputs) override;
  Status SetInput(size_t index, ImmediateExecutionTensorHandle* input) override;
  absl::Span<ImmediateExecutionTensorHandle* const> GetInputs() const override;
  bool HasCustomDeviceInput() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_9(mht_9_v, 288, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "HasCustomDeviceInput");

    return custom_device_tensor_handles_count_ > 0;
  }
  Status Execute(absl::Span<AbstractTensorHandle*> retvals,
                 int* num_retvals) override;
  const tensorflow::OpDef* OpDef() const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_10(mht_10_v, 296, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "OpDef");
 return op_def_; };

  Status SetAttrString(const char* attr_name, const char* data,
                       size_t length) override;
  Status SetAttrInt(const char* attr_name, int64_t value) override;
  Status SetAttrFloat(const char* attr_name, float value) override;
  Status SetAttrBool(const char* attr_name, bool value) override;
  Status SetAttrType(const char* attr_name, DataType value) override;
  Status SetAttrShape(const char* attr_name, const int64_t* dims,
                      const int num_dims) override;
  Status SetAttrFunction(const char* attr_name,
                         const AbstractOperation* value) override;
  Status SetAttrFunctionName(const char* attr_name, const char* data,
                             size_t length) override;
  Status SetAttrTensor(const char* attr_name,
                       AbstractTensorInterface* tensor) override;
  Status SetAttrStringList(const char* attr_name, const void* const* values,
                           const size_t* lengths, int num_values) override;
  Status SetAttrFloatList(const char* attr_name, const float* values,
                          int num_values) override;
  Status SetAttrIntList(const char* attr_name, const int64_t* values,
                        int num_values) override;
  Status SetAttrTypeList(const char* attr_name, const DataType* values,
                         int num_values) override;
  Status SetAttrBoolList(const char* attr_name, const unsigned char* values,
                         int num_values) override;
  Status SetAttrShapeList(const char* attr_name, const int64_t** dims,
                          const int* num_dims, int num_values) override;
  Status SetAttrFunctionList(
      const char* attr_name,
      absl::Span<const AbstractOperation*> values) override;

  Status InputLength(const char* input_name, int* length) override;
  Status OutputLength(const char* output_name, int* length) override;

  const AbstractOpAttrs* GetOpAttrs() const override;
  void AddAttrs(const AbstractOpAttrs* op_attrs) override;

  void SetStackTrace(ManagedStackTrace stack_trace) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_11(mht_11_v, 337, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "SetStackTrace");

    stack_trace_ = stack_trace;
  }

  absl::optional<ManagedStackTrace> GetStackTrace() override {
    return stack_trace_;
  }

  Status Reset(const char* op, const char* device_name, bool remote,
               EagerExecutor* executor,
               const absl::optional<EagerFunctionParams> remote_func_params =
                   absl::nullopt);

  bool is_function() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_12(mht_12_v, 353, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "is_function");
 return is_function_; }
  bool colocation_exempt() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_13(mht_13_v, 357, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "colocation_exempt");
 return colocation_exempt_; }

  tensorflow::EagerContext& EagerContext() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_14(mht_14_v, 362, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "EagerContext");
 return ctx_; }

  AttrBuilder* MutableAttrs() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_15(mht_15_v, 367, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "MutableAttrs");
 return &attrs_; }
  const AttrBuilder& Attrs() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_16(mht_16_v, 371, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "Attrs");
 return attrs_; }

  // TensorHandleInputs and MutableTensorHandleInputs first check that all
  // inputs are TensorHandles, i.e. that there are no custom device inputs. They
  // return a bad status otherwise.
  Status TensorHandleInputs(
      const absl::InlinedVector<TensorHandle*, 4>** inputs) const;
  Status MutableTensorHandleInputs(
      absl::InlinedVector<TensorHandle*, 4>** inputs);

  const absl::InlinedVector<ImmediateExecutionTensorHandle*, 4>& Inputs()
      const {
    return inputs_;
  }

  void UpdateInput(int i, TensorHandle* h);

  // Like TensorHandles, EagerOperations may be placed either on a virtual
  // CustomDevice or on a physical Device.
  VariantDevice Device() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_17(mht_17_v, 393, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "Device");
 return device_; }

  // Indicates whether the op is assigned to a device that is local to the
  // current host.
  bool IsLocal() const;

  CancellationManager* GetCancellationManager() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_18(mht_18_v, 402, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "GetCancellationManager");

    return cancellation_manager_;
  }
  void SetCancellationManager(
      CancellationManager* cancellation_manager) override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_19(mht_19_v, 409, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "SetCancellationManager");

    cancellation_manager_ = cancellation_manager;
  }

  // Assign step_id value only if op has valid step id.
  // When eager_func_params.has_value() returns true, we can directly overwrite
  // its step id according to Op's step id (if not default value). However, when
  // eager_func_params.has_value() returns false, we need to first create a new
  // EagerFuncParams object for it before assigning step_id; otherwise,
  // directly assigning step_id in this case leaves eager_func_params to be
  // in a weird state where:
  // (1) eager_func_params.has_value() returns false, but
  // (2) eager_func_params->step_id.has_value() returns true.
  void SetStepId(int64_t step_id) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_20(mht_20_v, 425, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "SetStepId");

    assert(is_function());
    if (step_id != EagerContext::kGlobalRendezvousId) {
      if (eager_func_params_.has_value()) {
        eager_func_params_->step_id = step_id;
      } else {
        eager_func_params_ = EagerFunctionParams{kInvalidOpId, step_id};
      }
    } else {
      LOG(WARNING) << "SetStepId() should not receive a gloabl rendezvous id.";
    }
  }

  EagerExecutor& Executor() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_21(mht_21_v, 441, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "Executor");
 return *executor_; }

  string DebugString() const;

  const absl::optional<EagerFunctionParams>& eager_func_params() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_22(mht_22_v, 448, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "eager_func_params");

    return eager_func_params_;
  }

  // Op name recorded for memory debugging purpose.
  const char* op_name() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_23(mht_23_v, 456, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "op_name");
 return op_name_; }

  // For LLVM style RTTI.
  static bool classof(const AbstractOperation* ptr) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_24(mht_24_v, 462, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "classof");

    return ptr->getKind() == kEager;
  }

 private:
  void AddTensorHandle(ImmediateExecutionTensorHandle* h);

  const tensorflow::OpDef* GetOpDef(Status* status);

  void ClearInferenceState() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_25(mht_25_v, 474, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "ClearInferenceState");

    op_def_ = nullptr;
    inference_arg_idx_ = 0;
    inference_attrs_.clear_no_resize();
  }

  Status MaybeInferSingleInputAttrs(ImmediateExecutionTensorHandle* handle);
  Status InferInputListAttrs(int num_inputs);

  void InferSingleTypeInputListAttrs(const OpDef::ArgDef& input_def,
                                     const DataType dtype, int num_inputs);
  void InferMixedTypeInputListAttrs(const OpDef::ArgDef& input_def,
                                    const std::vector<DataType>& dtypes);

  tensorflow::EagerContext& ctx_;
  const char* op_name_ = nullptr;
  AttrBuilder attrs_;
  const AttrTypeMap* attr_types_;

  // The number of custom device TensorHandle inputs. These inputs need to be
  // processed by CustomDeviceOpHandler first.
  int custom_device_tensor_handles_count_ = 0;
  absl::InlinedVector<ImmediateExecutionTensorHandle*, 4> inputs_;

  // The last device name given to SetDeviceName.
  // This is used to avoid having to re-process the same device in repeated
  // calls to SetDeviceName.
  string last_set_device_name_;

  // The operation's device name.
  // This contains the named passed to SetDeviceName until device_ is set,
  // at which point it contains the device_ name.
  string device_name_;

  // The parsed device name.
  // This will always contain the result of
  // DeviceNameUtils::ParseFullName(device_name_).
  DeviceNameUtils::ParsedName device_parsed_name_;

  // The operation's device.
  // This is set by the execution device placement logic, and should conform
  // with the contents of device_name_. Once it is set, the device_name_ is
  // updated accordingly.
  VariantDevice device_;

  absl::optional<ManagedStackTrace> stack_trace_;
  bool is_function_;  // Conceptually const, but can't be because of Reset
  bool colocation_exempt_;
  CancellationManager* cancellation_manager_ = nullptr;  // Not owned.
  EagerExecutor* executor_;                              // Not owned.

  absl::optional<EagerFunctionParams> eager_func_params_;

  // Inference information
  const tensorflow::OpDef* op_def_;  // op definition from protobuf
  int inference_arg_idx_;  // arg definition index for the next input to be
                           // added
  gtl::FlatSet<std::string> inference_attrs_;  // attributes inferred so far
};

inline void EagerOperation::UpdateInput(int i, TensorHandle* h) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_26(mht_26_v, 537, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "EagerOperation::UpdateInput");

  ImmediateExecutionTensorHandle** slot = &inputs_[i];
  ImmediateExecutionTensorHandle* existing = *slot;
  if (existing != h) {
    h->Ref();
    existing->Unref();
    *slot = h;  // Update inputs_[i] to h
  }
}

inline EagerOperation* OperationFromInterface(
    ImmediateExecutionOperation* operation) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_27(mht_27_v, 551, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "OperationFromInterface");

  return down_cast<EagerOperation*>(operation);
}

inline const EagerOperation* OperationFromInterface(
    const ImmediateExecutionOperation* operation) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTh mht_28(mht_28_v, 559, "", "./tensorflow/core/common_runtime/eager/eager_operation.h", "OperationFromInterface");

  return down_cast<const EagerOperation*>(operation);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_OPERATION_H_
