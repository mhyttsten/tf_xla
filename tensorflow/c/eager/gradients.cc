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
class MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc {
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
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc() {
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
#include "tensorflow/c/eager/gradients.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/gradients_internal.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace gradients {
namespace {

// TODO(b/172558015): Using the pointer address as the identifier for the tensor
// may lead to collisions. Introduce another way to get a unique id for this
// tensor.
int64_t ToId(const AbstractTensorHandle* t) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_0(mht_0_v, 201, "", "./tensorflow/c/eager/gradients.cc", "ToId");

  return static_cast<int64_t>(reinterpret_cast<uintptr_t>(t));
}

Status ZerosLike(AbstractContext* ctx, AbstractTensorHandle* t,
                 AbstractTensorHandle** result) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_1(mht_1_v, 209, "", "./tensorflow/c/eager/gradients.cc", "ZerosLike");

  AbstractOperationPtr op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op->Reset("ZerosLike", /*raw_device_name=*/nullptr));
  if (isa<tracing::TracingOperation>(op.get())) {
    TF_RETURN_IF_ERROR(dyn_cast<tracing::TracingOperation>(op.get())->SetOpName(
        absl::StrCat("ZerosLike", ToId(t)).c_str()));
  }
  TF_RETURN_IF_ERROR(op->AddInput(t));
  int num_outputs = 1;
  std::vector<AbstractTensorHandle*> outputs(num_outputs);
  TF_RETURN_IF_ERROR(
      op->Execute(absl::Span<AbstractTensorHandle*>(outputs), &num_outputs));
  *result = outputs[0];
  return Status::OK();
}
}  // namespace

Status GradientRegistry::Register(
    const string& op_name, GradientFunctionFactory gradient_function_factory) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_2(mht_2_v, 231, "", "./tensorflow/c/eager/gradients.cc", "GradientRegistry::Register");

  auto iter = registry_.find(op_name);
  if (iter != registry_.end()) {
    const string error_msg = "Gradient already exists for op: " + op_name + ".";
    return errors::AlreadyExists(error_msg);
  }
  registry_.insert({op_name, gradient_function_factory});
  return Status::OK();
}
Status GradientRegistry::Lookup(
    const ForwardOperation& op,
    std::unique_ptr<GradientFunction>* gradient_function) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_3(mht_3_v, 245, "", "./tensorflow/c/eager/gradients.cc", "GradientRegistry::Lookup");

  auto iter = registry_.find(op.op_name);
  if (iter == registry_.end()) {
    const string error_msg = "No gradient defined for op: " + op.op_name + ".";
    return errors::NotFound(error_msg);
  }
  gradient_function->reset(iter->second(op));
  return Status::OK();
}

TapeTensor::TapeTensor(AbstractTensorHandle* handle) : handle_(handle) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_4(mht_4_v, 258, "", "./tensorflow/c/eager/gradients.cc", "TapeTensor::TapeTensor");

  handle_->Ref();
}
TapeTensor::TapeTensor(const TapeTensor& other) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_5(mht_5_v, 264, "", "./tensorflow/c/eager/gradients.cc", "TapeTensor::TapeTensor");

  handle_ = other.handle_;
  handle_->Ref();
}
TapeTensor::~TapeTensor() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_6(mht_6_v, 271, "", "./tensorflow/c/eager/gradients.cc", "TapeTensor::~TapeTensor");
 handle_->Unref(); }

int64_t TapeTensor::GetID() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_7(mht_7_v, 276, "", "./tensorflow/c/eager/gradients.cc", "TapeTensor::GetID");
 return ToId(handle_); }

tensorflow::DataType TapeTensor::GetDType() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_8(mht_8_v, 281, "", "./tensorflow/c/eager/gradients.cc", "TapeTensor::GetDType");

  return handle_->DataType();
}
AbstractTensorHandle* TapeTensor::GetHandle() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_9(mht_9_v, 287, "", "./tensorflow/c/eager/gradients.cc", "TapeTensor::GetHandle");
 return handle_; }

AbstractTensorHandle* TapeTensor::ZerosLike() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_10(mht_10_v, 292, "", "./tensorflow/c/eager/gradients.cc", "TapeTensor::ZerosLike");
 return nullptr; }

class TapeVSpace
    : public eager::VSpace<AbstractTensorHandle, GradientFunction, TapeTensor> {
 public:
  explicit TapeVSpace(AbstractContext* ctx) : ctx_(ctx) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_11(mht_11_v, 300, "", "./tensorflow/c/eager/gradients.cc", "TapeVSpace");
}
  ~TapeVSpace() override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_12(mht_12_v, 304, "", "./tensorflow/c/eager/gradients.cc", "~TapeVSpace");
}

  // Returns the number of elements in the gradient tensor.
  int64_t NumElements(AbstractTensorHandle* tensor) const override;

  // Consumes references to the tensors in the gradient_tensors list and returns
  // a tensor with the result.
  AbstractTensorHandle* AggregateGradients(
      gtl::ArraySlice<AbstractTensorHandle*> gradient_tensors) const override;

  // Calls the passed-in backward function.
  // op_type is the op's name provided in RecordOperation.
  Status CallBackwardFunction(
      const string& op_type, GradientFunction* gradient_function,
      const std::vector<int64_t>& unneeded_gradients,
      gtl::ArraySlice<AbstractTensorHandle*> output_gradients,
      absl::Span<AbstractTensorHandle*> result) const override;

  // Builds a tensor filled with ones with the same shape and dtype as `t`.
  Status BuildOnesLike(const TapeTensor& t,
                       AbstractTensorHandle** result) const override;

  // Looks up the ID of a Gradient.
  int64_t TensorId(AbstractTensorHandle* tensor) const override;

  // Converts a Gradient to a TapeTensor.
  TapeTensor TapeTensorFromGradient(AbstractTensorHandle* g) const override;

  void MarkAsResult(AbstractTensorHandle* gradient) const override;

  void DeleteGradient(AbstractTensorHandle* gradient) const override;

 private:
  // The context where the aggregation op `Add` is to be created.
  AbstractContext* ctx_;
};

// Returns the number of elements in the gradient tensor.
int64_t TapeVSpace::NumElements(AbstractTensorHandle* tensor) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_13(mht_13_v, 345, "", "./tensorflow/c/eager/gradients.cc", "TapeVSpace::NumElements");

  // TODO(srbs): It seems like this is used only for performance optimization
  // and not for correctness. The only downside of keeping this 1 seems to be
  // that the gradient accumulation is unbounded and we will never
  // aggressively aggregate accumulated gradients to recover memory.
  // Revisit and fix.
  return 1;
}

// Consumes references to the tensors in the gradient_tensors list and returns
// a tensor with the result.
AbstractTensorHandle* TapeVSpace::AggregateGradients(
    gtl::ArraySlice<AbstractTensorHandle*> gradient_tensors) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_14(mht_14_v, 360, "", "./tensorflow/c/eager/gradients.cc", "TapeVSpace::AggregateGradients");

  if (gradient_tensors.size() == 1) {
    return gradient_tensors[0];
  }

  AbstractOperationPtr op(ctx_->CreateOperation());
  Status s = op->Reset("AddN", /*raw_device_name=*/nullptr);
  if (!s.ok()) {
    return nullptr;
  }
  s = op->AddInputList(gradient_tensors);
  if (!s.ok()) {
    return nullptr;
  }

  int num_outputs = 1;
  std::vector<AbstractTensorHandle*> outputs(num_outputs);
  s = op->Execute(absl::Span<AbstractTensorHandle*>(outputs), &num_outputs);
  if (!s.ok()) {
    return nullptr;
  }
  return outputs[0];
}

// Calls the passed-in backward function.
// op_type is the op's name provided in RecordOperation.
Status TapeVSpace::CallBackwardFunction(
    const string& op_type, GradientFunction* gradient_function,
    const std::vector<int64_t>& unneeded_gradients,
    gtl::ArraySlice<AbstractTensorHandle*> output_gradients,
    absl::Span<AbstractTensorHandle*> result) const {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("op_type: \"" + op_type + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_15(mht_15_v, 394, "", "./tensorflow/c/eager/gradients.cc", "TapeVSpace::CallBackwardFunction");

  if (gradient_function == nullptr) {
    return errors::InvalidArgument(
        "Provided null gradient_function for '", op_type, "'.\n",
        "If the intent is to treat this op as non-differentiable consider "
        "using RegisterNotDifferentiable or "
        "NotDifferentiableGradientFunction.");
  }
  return gradient_function->Compute(ctx_, output_gradients, result);
}

Status TapeVSpace::BuildOnesLike(const TapeTensor& t,
                                 AbstractTensorHandle** result) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_16(mht_16_v, 409, "", "./tensorflow/c/eager/gradients.cc", "TapeVSpace::BuildOnesLike");

  AbstractOperationPtr op(ctx_->CreateOperation());
  TF_RETURN_IF_ERROR(op->Reset("OnesLike", /*raw_device_name=*/nullptr));
  if (isa<tracing::TracingOperation>(op.get())) {
    TF_RETURN_IF_ERROR(dyn_cast<tracing::TracingOperation>(op.get())->SetOpName(
        absl::StrCat("OnesLike", ToId(t.GetHandle())).c_str()));
  }
  TF_RETURN_IF_ERROR(op->AddInput(t.GetHandle()));
  int num_outputs = 1;
  std::vector<AbstractTensorHandle*> outputs(num_outputs);
  TF_RETURN_IF_ERROR(
      op->Execute(absl::Span<AbstractTensorHandle*>(outputs), &num_outputs));
  *result = outputs[0];
  return Status::OK();
}

// Looks up the ID of a Gradient.
int64_t TapeVSpace::TensorId(AbstractTensorHandle* tensor) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_17(mht_17_v, 429, "", "./tensorflow/c/eager/gradients.cc", "TapeVSpace::TensorId");

  return ToId(tensor);
}

// Converts a Gradient to a TapeTensor.
TapeTensor TapeVSpace::TapeTensorFromGradient(AbstractTensorHandle* g) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_18(mht_18_v, 437, "", "./tensorflow/c/eager/gradients.cc", "TapeVSpace::TapeTensorFromGradient");

  return TapeTensor(g);
}

void TapeVSpace::MarkAsResult(AbstractTensorHandle* gradient) const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_19(mht_19_v, 444, "", "./tensorflow/c/eager/gradients.cc", "TapeVSpace::MarkAsResult");
}

void TapeVSpace::DeleteGradient(AbstractTensorHandle* gradient) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_20(mht_20_v, 449, "", "./tensorflow/c/eager/gradients.cc", "TapeVSpace::DeleteGradient");

  gradient->Unref();
}

void Tape::Watch(const AbstractTensorHandle* t) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_21(mht_21_v, 456, "", "./tensorflow/c/eager/gradients.cc", "Tape::Watch");

  GradientTape::Watch(ToId(t));
}
void Tape::RecordOperation(absl::Span<AbstractTensorHandle* const> inputs,
                           absl::Span<AbstractTensorHandle* const> outputs,
                           GradientFunction* gradient_function,
                           const string& op_name) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_22(mht_22_v, 466, "", "./tensorflow/c/eager/gradients.cc", "Tape::RecordOperation");

  std::vector<int64_t> input_ids(inputs.size());
  std::vector<tensorflow::DataType> input_dtypes(inputs.size());
  for (int i = 0; i < inputs.size(); i++) {
    input_ids[i] = ToId(inputs[i]);
    input_dtypes[i] = inputs[i]->DataType();
  }
  std::vector<TapeTensor> tape_tensors;
  tape_tensors.reserve(outputs.size());
  for (auto t : outputs) {
    tape_tensors.push_back(TapeTensor(t));
  }
  GradientTape::RecordOperation(
      op_name, tape_tensors, input_ids, input_dtypes,
      [gradient_function]() -> GradientFunction* { return gradient_function; },
      [](GradientFunction* ptr) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_23(mht_23_v, 484, "", "./tensorflow/c/eager/gradients.cc", "lambda");

        if (ptr) {
          delete ptr;
        }
      });
}
bool Tape::ShouldRecord(
    absl::Span<const AbstractTensorHandle* const> tensors) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_24(mht_24_v, 494, "", "./tensorflow/c/eager/gradients.cc", "Tape::ShouldRecord");

  std::vector<int64_t> tensor_ids(tensors.size());
  std::vector<tensorflow::DataType> tensor_dtypes(tensors.size());
  for (int i = 0; i < tensors.size(); i++) {
    tensor_ids[i] = ToId(tensors[i]);
    tensor_dtypes[i] = tensors[i]->DataType();
  }
  return GradientTape::ShouldRecord(tensor_ids, tensor_dtypes);
}
void Tape::DeleteTrace(const AbstractTensorHandle* t) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_25(mht_25_v, 506, "", "./tensorflow/c/eager/gradients.cc", "Tape::DeleteTrace");

  GradientTape::DeleteTrace(ToId(t));
}

std::vector<int64_t> MakeTensorIDList(
    absl::Span<AbstractTensorHandle* const> tensors) {
  std::vector<int64_t> ids(tensors.size());
  for (int i = 0; i < tensors.size(); i++) {
    ids[i] = ToId(tensors[i]);
  }
  return ids;
}

Status Tape::ComputeGradient(
    AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> targets,
    absl::Span<AbstractTensorHandle* const> sources,
    absl::Span<AbstractTensorHandle* const> output_gradients,
    absl::Span<AbstractTensorHandle*> result) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_26(mht_26_v, 526, "", "./tensorflow/c/eager/gradients.cc", "Tape::ComputeGradient");

  TapeVSpace vspace(ctx);
  std::vector<int64_t> target_tensor_ids = MakeTensorIDList(targets);
  std::vector<int64_t> source_tensor_ids = MakeTensorIDList(sources);
  tensorflow::gtl::FlatSet<int64_t> sources_set(source_tensor_ids.begin(),
                                                source_tensor_ids.end());
  std::unordered_map<int64_t, TapeTensor> sources_that_are_targets;
  for (int i = 0; i < target_tensor_ids.size(); ++i) {
    int64_t target_id = target_tensor_ids[i];
    if (sources_set.find(target_id) != sources_set.end()) {
      auto tensor = targets[i];
      sources_that_are_targets.insert(
          std::make_pair(target_id, TapeTensor(tensor)));
    }
  }

  TF_RETURN_IF_ERROR(GradientTape::ComputeGradient(
      vspace, target_tensor_ids, source_tensor_ids, sources_that_are_targets,
      output_gradients, result, /*build_default_zeros_grads*/ false));
  return Status::OK();
}

// Helper functions which delegate to `AbstractOperation`, update
// the state of the ForwardOperation and call the tape as appropriate.
// These APIs are mainly to facilitate testing and are subject to change.
namespace internal {
Status Reset(AbstractOperation* op_, const char* op,
             const char* raw_device_name, ForwardOperation* forward_op_) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   mht_27_v.push_back("raw_device_name: \"" + (raw_device_name == nullptr ? std::string("nullptr") : std::string((char*)raw_device_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_27(mht_27_v, 558, "", "./tensorflow/c/eager/gradients.cc", "Reset");

  forward_op_->op_name = op;
  forward_op_->attrs.Reset(op);
  return op_->Reset(op, raw_device_name);
}
Status AddInput(AbstractOperation* op_, AbstractTensorHandle* input,
                ForwardOperation* forward_op_) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_28(mht_28_v, 567, "", "./tensorflow/c/eager/gradients.cc", "AddInput");

  TF_RETURN_IF_ERROR(op_->AddInput(input));
  forward_op_->inputs.push_back(input);
  return Status::OK();
}
Status AddInputList(AbstractOperation* op_,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    ForwardOperation* forward_op_) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_29(mht_29_v, 577, "", "./tensorflow/c/eager/gradients.cc", "AddInputList");

  TF_RETURN_IF_ERROR(op_->AddInputList(inputs));
  for (auto input : inputs) {
    forward_op_->inputs.push_back(input);
  }
  return Status::OK();
}

Status SetAttrString(AbstractOperation* op_, const char* attr_name,
                     const char* data, size_t length,
                     ForwardOperation* forward_op_) {
   std::vector<std::string> mht_30_v;
   mht_30_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_30_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_30(mht_30_v, 592, "", "./tensorflow/c/eager/gradients.cc", "SetAttrString");

  forward_op_->attrs.Set(attr_name, StringPiece(data, length));
  return op_->SetAttrString(attr_name, data, length);
}
Status SetAttrInt(AbstractOperation* op_, const char* attr_name, int64_t value,
                  ForwardOperation* forward_op_) {
   std::vector<std::string> mht_31_v;
   mht_31_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_31(mht_31_v, 601, "", "./tensorflow/c/eager/gradients.cc", "SetAttrInt");

  forward_op_->attrs.Set(attr_name, static_cast<int64_t>(value));
  return op_->SetAttrInt(attr_name, value);
}
Status SetAttrFloat(AbstractOperation* op_, const char* attr_name, float value,
                    ForwardOperation* forward_op_) {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_32(mht_32_v, 610, "", "./tensorflow/c/eager/gradients.cc", "SetAttrFloat");

  forward_op_->attrs.Set(attr_name, value);
  return op_->SetAttrFloat(attr_name, value);
}
Status SetAttrBool(AbstractOperation* op_, const char* attr_name, bool value,
                   ForwardOperation* forward_op_) {
   std::vector<std::string> mht_33_v;
   mht_33_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_33(mht_33_v, 619, "", "./tensorflow/c/eager/gradients.cc", "SetAttrBool");

  forward_op_->attrs.Set(attr_name, value);
  return op_->SetAttrBool(attr_name, value);
}
Status SetAttrType(AbstractOperation* op_, const char* attr_name,
                   DataType value, ForwardOperation* forward_op_) {
   std::vector<std::string> mht_34_v;
   mht_34_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_34(mht_34_v, 628, "", "./tensorflow/c/eager/gradients.cc", "SetAttrType");

  forward_op_->attrs.Set(attr_name, value);
  return op_->SetAttrType(attr_name, value);
}
Status SetAttrShape(AbstractOperation* op_, const char* attr_name,
                    const int64_t* dims, const int num_dims,
                    ForwardOperation* forward_op_) {
   std::vector<std::string> mht_35_v;
   mht_35_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_35(mht_35_v, 638, "", "./tensorflow/c/eager/gradients.cc", "SetAttrShape");

  if (num_dims > TensorShape::MaxDimensions()) {
    return errors::InvalidArgument("Value specified for `", attr_name, "` has ",
                                   num_dims,
                                   " dimensions which is over the limit of ",
                                   TensorShape::MaxDimensions(), ".");
  }
  TensorShapeProto proto;
  if (num_dims < 0) {
    proto.set_unknown_rank(true);
  } else {
    for (int d = 0; d < num_dims; ++d) {
      proto.add_dim()->set_size(dims[d]);
    }
  }

  forward_op_->attrs.Set(attr_name, proto);
  return op_->SetAttrShape(attr_name, dims, num_dims);
}
Status SetAttrFunction(AbstractOperation* op_, const char* attr_name,
                       const AbstractOperation* value,
                       ForwardOperation* forward_op_) {
   std::vector<std::string> mht_36_v;
   mht_36_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_36(mht_36_v, 663, "", "./tensorflow/c/eager/gradients.cc", "SetAttrFunction");

  return tensorflow::errors::Unimplemented(
      "SetAttrFunction has not been implemented yet.");
}
Status SetAttrFunctionName(AbstractOperation* op_, const char* attr_name,
                           const char* value, size_t length,
                           ForwardOperation* forward_op_) {
   std::vector<std::string> mht_37_v;
   mht_37_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_37_v.push_back("value: \"" + (value == nullptr ? std::string("nullptr") : std::string((char*)value)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_37(mht_37_v, 674, "", "./tensorflow/c/eager/gradients.cc", "SetAttrFunctionName");

  return tensorflow::errors::Unimplemented(
      "SetAttrFunctionName has not been implemented "
      "yet.");
}
Status SetAttrTensor(AbstractOperation* op_, const char* attr_name,
                     AbstractTensorInterface* tensor,
                     ForwardOperation* forward_op_) {
   std::vector<std::string> mht_38_v;
   mht_38_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_38(mht_38_v, 685, "", "./tensorflow/c/eager/gradients.cc", "SetAttrTensor");

  return tensorflow::errors::Unimplemented(
      "SetAttrTensor has not been implemented yet.");
}
Status SetAttrStringList(AbstractOperation* op_, const char* attr_name,
                         const void* const* values, const size_t* lengths,
                         int num_values, ForwardOperation* forward_op_) {
   std::vector<std::string> mht_39_v;
   mht_39_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_39(mht_39_v, 695, "", "./tensorflow/c/eager/gradients.cc", "SetAttrStringList");

  std::vector<StringPiece> v(num_values);
  for (int i = 0; i < num_values; ++i) {
    v[i] = StringPiece(static_cast<const char*>(values[i]), lengths[i]);
  }
  forward_op_->attrs.Set(attr_name, v);
  return op_->SetAttrStringList(attr_name, values, lengths, num_values);
}
Status SetAttrFloatList(AbstractOperation* op_, const char* attr_name,
                        const float* values, int num_values,
                        ForwardOperation* forward_op_) {
   std::vector<std::string> mht_40_v;
   mht_40_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_40(mht_40_v, 709, "", "./tensorflow/c/eager/gradients.cc", "SetAttrFloatList");

  forward_op_->attrs.Set(attr_name,
                         gtl::ArraySlice<const float>(values, num_values));
  return op_->SetAttrFloatList(attr_name, values, num_values);
}
Status SetAttrIntList(AbstractOperation* op_, const char* attr_name,
                      const int64_t* values, int num_values,
                      ForwardOperation* forward_op_) {
   std::vector<std::string> mht_41_v;
   mht_41_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_41(mht_41_v, 720, "", "./tensorflow/c/eager/gradients.cc", "SetAttrIntList");

  forward_op_->attrs.Set(
      attr_name, gtl::ArraySlice<const int64_t>(
                     reinterpret_cast<const int64_t*>(values), num_values));
  return op_->SetAttrIntList(attr_name, values, num_values);
}
Status SetAttrTypeList(AbstractOperation* op_, const char* attr_name,
                       const DataType* values, int num_values,
                       ForwardOperation* forward_op_) {
   std::vector<std::string> mht_42_v;
   mht_42_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_42(mht_42_v, 732, "", "./tensorflow/c/eager/gradients.cc", "SetAttrTypeList");

  forward_op_->attrs.Set(attr_name,
                         gtl::ArraySlice<const DataType>(values, num_values));
  return op_->SetAttrTypeList(attr_name, values, num_values);
}
Status SetAttrBoolList(AbstractOperation* op_, const char* attr_name,
                       const unsigned char* values, int num_values,
                       ForwardOperation* forward_op_) {
   std::vector<std::string> mht_43_v;
   mht_43_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_43_v.push_back("values: \"" + (values == nullptr ? std::string("nullptr") : std::string((char*)values)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_43(mht_43_v, 744, "", "./tensorflow/c/eager/gradients.cc", "SetAttrBoolList");

  std::unique_ptr<bool[]> b(new bool[num_values]);
  for (int i = 0; i < num_values; ++i) {
    b[i] = values[i];
  }
  forward_op_->attrs.Set(attr_name,
                         gtl::ArraySlice<const bool>(b.get(), num_values));
  return op_->SetAttrBoolList(attr_name, values, num_values);
}
Status SetAttrShapeList(AbstractOperation* op_, const char* attr_name,
                        const int64_t** dims, const int* num_dims,
                        int num_values, ForwardOperation* forward_op_) {
   std::vector<std::string> mht_44_v;
   mht_44_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_44(mht_44_v, 759, "", "./tensorflow/c/eager/gradients.cc", "SetAttrShapeList");

  std::unique_ptr<TensorShapeProto[]> proto(new TensorShapeProto[num_values]);
  for (int i = 0; i < num_values; ++i) {
    const auto num_dims_i = num_dims[i];

    if (num_dims_i > TensorShape::MaxDimensions()) {
      return errors::InvalidArgument(
          strings::StrCat("Value specified for `", attr_name, "` has ",
                          num_dims_i, " dimensions which is over the limit of ",
                          TensorShape::MaxDimensions(), "."));
    }
    if (num_dims_i < 0) {
      proto[i].set_unknown_rank(true);
    } else {
      const int64_t* dims_i = dims[i];
      auto proto_i = &proto[i];
      for (int d = 0; d < num_dims_i; ++d) {
        proto_i->add_dim()->set_size(dims_i[d]);
      }
    }
  }
  forward_op_->attrs.Set(
      attr_name, gtl::ArraySlice<TensorShapeProto>(proto.get(), num_values));
  return op_->SetAttrShapeList(attr_name, dims, num_dims, num_values);
}
Status SetAttrFunctionList(AbstractOperation* op_, const char* attr_name,
                           absl::Span<const AbstractOperation*> values,
                           ForwardOperation* forward_op_) {
   std::vector<std::string> mht_45_v;
   mht_45_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_45(mht_45_v, 790, "", "./tensorflow/c/eager/gradients.cc", "SetAttrFunctionList");

  return tensorflow::errors::Unimplemented(
      "SetAttrFunctionList has not been "
      "implemented yet.");
}
Status Execute(AbstractOperation* op_, AbstractContext* ctx,
               absl::Span<AbstractTensorHandle*> retvals, int* num_retvals,
               ForwardOperation* forward_op_, Tape* tape,
               const GradientRegistry& registry) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTcc mht_46(mht_46_v, 801, "", "./tensorflow/c/eager/gradients.cc", "Execute");

  TF_RETURN_IF_ERROR(op_->Execute(retvals, num_retvals));
  for (int i = 0; i < *num_retvals; i++) {
    // TODO(srbs): Manage refcount of ForwardOperation's inputs/outputs.
    forward_op_->outputs.push_back(retvals[i]);
  }
  // TODO(b/166669239): This is needed to support AttrBuilder::Get for string
  // attributes. Number type attrs and DataType attrs work fine without this.
  // Consider getting rid of this and making the behavior between number types
  // and string consistent.
  forward_op_->attrs.BuildNodeDef();
  std::unique_ptr<GradientFunction> gradient_fn;
  TF_RETURN_IF_ERROR(registry.Lookup(*forward_op_, &gradient_fn));
  tape->RecordOperation(forward_op_->inputs, retvals, gradient_fn.release(),
                        op_->Name());
  return Status::OK();
}
}  // namespace internal

}  // namespace gradients
}  // namespace tensorflow
