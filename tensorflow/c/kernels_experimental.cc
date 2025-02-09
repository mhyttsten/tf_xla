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
class MHTracer_DTPStensorflowPScPSkernels_experimentalDTcc {
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
   MHTracer_DTPStensorflowPScPSkernels_experimentalDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSkernels_experimentalDTcc() {
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

#include "tensorflow/c/kernels_experimental.h"

#include <algorithm>
#include <utility>

#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/refcount.h"

using tensorflow::AllocatorAttributes;
using tensorflow::mutex_lock;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TF_TensorFromTensor;
using tensorflow::Var;
using tensorflow::Variant;
using tensorflow::errors::InvalidArgument;

struct TF_VariableInputLockHolder {
  TF_VariableInputLockHolder(
      std::vector<tensorflow::Var*> vars,
      std::unique_ptr<std::vector<tensorflow::mutex_lock>> locks,
      std::unique_ptr<std::vector<tensorflow::tf_shared_lock>> shared_locks)
      : vars(std::move(vars)),
        locks(std::move(locks)),
        shared_locks(std::move(shared_locks)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSkernels_experimentalDTcc mht_0(mht_0_v, 216, "", "./tensorflow/c/kernels_experimental.cc", "TF_VariableInputLockHolder");
}

  std::vector<tensorflow::Var*> vars;
  std::unique_ptr<std::vector<tensorflow::mutex_lock>> locks;
  std::unique_ptr<std::vector<tensorflow::tf_shared_lock>> shared_locks;
};

tensorflow::Status EnsureSparseVariableAccess(
    TF_OpKernelContext* ctx, bool variantType,
    void (*copyFunc)(TF_OpKernelContext* ctx, TF_Tensor* source,
                     TF_Tensor* dest),
    tensorflow::Var* var) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSkernels_experimentalDTcc mht_1(mht_1_v, 230, "", "./tensorflow/c/kernels_experimental.cc", "EnsureSparseVariableAccess");

  auto* context = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  if (var->copy_on_read_mode.load()) {
    return Status::OK();
  }
  mutex_lock ml(*var->mu());
  // Once copy-on-read mode is True the refcount is guaranteed to be 1. This can
  // also happen if there are no concurrent reads of the variable and
  // copy-on-read mode is false.
  if (var->tensor()->RefCountIsOne()) {
    var->copy_on_read_mode.store(true);
    return Status::OK();
  }
  Tensor tmp;
  if (variantType) {
    AllocatorAttributes attr;
    attr.set_on_host(true);
    TF_RETURN_IF_ERROR(context->allocate_temp(
        var->tensor()->dtype(), var->tensor()->shape(), &tmp, attr));

    const auto elements_in = var->tensor()->flat<Variant>();
    auto elements_out = tmp.flat<Variant>();
    for (int64_t i = 0; i < elements_in.size(); ++i) {
      elements_out(i) = elements_in(i);
    }
  } else {
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);
    TF_RETURN_IF_ERROR(context->allocate_temp(
        var->tensor()->dtype(), var->tensor()->shape(), &tmp, attr));
    tensorflow::Status s;
    TF_Tensor* tf_tmp = TF_TensorFromTensor(tmp, &s);
    TF_Tensor* tf_tensor = TF_TensorFromTensor(*var->tensor(), &s);
    copyFunc(ctx, tf_tensor, tf_tmp);
  }
  *var->tensor() = tmp;
  var->copy_on_read_mode.store(true);
  return Status::OK();
}

tensorflow::Status PrepareToUpdateVariable(
    TF_OpKernelContext* ctx, tensorflow::Tensor* tensor, bool copy_on_read_mode,
    bool variantType,
    void (*copyFunc)(TF_OpKernelContext* ctx, TF_Tensor* source,
                     TF_Tensor* dest)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSkernels_experimentalDTcc mht_2(mht_2_v, 278, "", "./tensorflow/c/kernels_experimental.cc", "PrepareToUpdateVariable");

  auto* context = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  if (copy_on_read_mode || !tensor->RefCountIsOne()) {
    // Tensor's buffer is in use by some read, so we need to copy before
    // updating.
    Tensor tmp;
    if (variantType) {
      AllocatorAttributes attr;
      attr.set_on_host(true);
      TF_RETURN_IF_ERROR(
          context->allocate_temp(tensor->dtype(), tensor->shape(), &tmp, attr));

      const auto elements_in = tensor->flat<Variant>();
      auto elements_out = tmp.flat<Variant>();
      for (int64_t i = 0; i < elements_in.size(); ++i) {
        elements_out(i) = elements_in(i);
      }
    } else {
      AllocatorAttributes attr;
      attr.set_gpu_compatible(true);
      attr.set_nic_compatible(true);
      TF_RETURN_IF_ERROR(
          context->allocate_temp(tensor->dtype(), tensor->shape(), &tmp, attr));
      tensorflow::Status s;
      TF_Tensor* tf_tmp = TF_TensorFromTensor(tmp, &s);
      TF_Tensor* tf_tensor = TF_TensorFromTensor(*tensor, &s);
      copyFunc(ctx, tf_tensor, tf_tmp);
    }
    *tensor = tmp;
  }
  return Status::OK();
}

tensorflow::mutex* GetTrainingVariableMutex(
    TF_OpKernelContext* ctx, int32_t input, bool sparse,
    void (*copyFunc)(TF_OpKernelContext* ctx, TF_Tensor* source,
                     TF_Tensor* dest),
    tensorflow::Var** maybe_resource) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSkernels_experimentalDTcc mht_3(mht_3_v, 318, "", "./tensorflow/c/kernels_experimental.cc", "GetTrainingVariableMutex");

  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  *maybe_resource = nullptr;
  if (cc_ctx->input_dtype(input) == tensorflow::DT_RESOURCE) {
    if (LookupResource(cc_ctx, HandleFromInput(cc_ctx, input), maybe_resource)
            .ok()) {
      if (sparse) {
        TF_CHECK_OK(
            EnsureSparseVariableAccess(ctx, false, copyFunc, *maybe_resource));
      }
      return (*maybe_resource)->mu();
    } else {
      cc_ctx->CtxFailureWithWarning(
          tensorflow::errors::Internal("Invalid variable reference."));
      return nullptr;
    }
  }
  return cc_ctx->input_ref_mutex(input);
}

void TF_AssignVariable(TF_OpKernelContext* ctx, int input_index,
                       int value_index,
                       void (*copyFunc)(TF_OpKernelContext* ctx,
                                        TF_Tensor* source, TF_Tensor* dest),
                       TF_Status* status) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSkernels_experimentalDTcc mht_4(mht_4_v, 345, "", "./tensorflow/c/kernels_experimental.cc", "TF_AssignVariable");

  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  tensorflow::core::RefCountPtr<tensorflow::Var> variable;
  const tensorflow::Tensor& value = cc_ctx->input(value_index);
  OP_REQUIRES_OK(cc_ctx, tensorflow::LookupOrCreateResource<tensorflow::Var>(
                             cc_ctx, HandleFromInput(cc_ctx, input_index),
                             &variable, [&value](tensorflow::Var** ptr) {
                               *ptr = new tensorflow::Var(value.dtype());
                               *(*ptr)->tensor() = value;
                               (*ptr)->is_initialized = true;
                               return tensorflow::Status::OK();
                             }));
  tensorflow::mutex_lock ml(*variable->mu());

  if (variable->copy_on_read_mode.load()) {
    tensorflow::Tensor tmp;
    tensorflow::AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);
    OP_REQUIRES_OK(cc_ctx, cc_ctx->allocate_temp(value.dtype(), value.shape(),
                                                 &tmp, attr));
    tensorflow::Status s;
    TF_Tensor* tf_tmp = TF_TensorFromTensor(tmp, &s);
    TF_Tensor* tf_value = TF_TensorFromTensor(value, &s);
    copyFunc(ctx, tf_value, tf_tmp);
    *variable->tensor() = tmp;
  } else {
    *variable->tensor() = value;
  }
  variable->is_initialized = true;
  TF_SetStatus(status, TF_OK, "");
}

void TF_AssignUpdateVariable(TF_OpKernelContext* ctx, int input_index,
                             int value_index, int Op, int isVariantType,
                             void (*copyFunc)(TF_OpKernelContext* ctx,
                                              TF_Tensor* source,
                                              TF_Tensor* dest),
                             void (*updateFunc)(TF_OpKernelContext* ctx,
                                                TF_Tensor* tensor,
                                                TF_Tensor* value, int Op),
                             TF_Status* tf_status) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSkernels_experimentalDTcc mht_5(mht_5_v, 389, "", "./tensorflow/c/kernels_experimental.cc", "TF_AssignUpdateVariable");

  auto* context = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  tensorflow::core::RefCountPtr<Var> variable;
  Status status =
      LookupResource(context, HandleFromInput(context, input_index), &variable);
  if (!status.ok()) {
    printf("Failed with error: %s\n", status.error_message().c_str());
    abort();
  }
  const Tensor& value = context->input(value_index);
  mutex_lock ml(*variable->mu());
  Tensor* var_tensor = variable->tensor();
  OP_REQUIRES(
      context, var_tensor->shape().IsSameSize(value.shape()),
      InvalidArgument("Cannot update variable with shape ",
                      var_tensor->shape().DebugString(),
                      " using a Tensor with shape ",
                      value.shape().DebugString(), ", shapes must be equal."));
  OP_REQUIRES_OK(context,
                 PrepareToUpdateVariable(ctx, var_tensor,
                                         variable->copy_on_read_mode.load(),
                                         isVariantType, copyFunc));
  tensorflow::Status s;
  TF_Tensor* tf_var_tensor = TF_TensorFromTensor(*var_tensor, &s);
  TF_Tensor* tf_value = TF_TensorFromTensor(value, &s);
  updateFunc(ctx, tf_var_tensor, tf_value, Op);
  TF_SetStatus(tf_status, TF_OK, "");
}

void TF_MaybeLockVariableInputMutexesInOrder(
    TF_OpKernelContext* ctx, bool do_lock, bool sparse, const int* const inputs,
    size_t len,
    void (*copyFunc)(TF_OpKernelContext* ctx, TF_Tensor* source,
                     TF_Tensor* dest),
    TF_VariableInputLockHolder** lockHolder, TF_Status* status) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSkernels_experimentalDTcc mht_6(mht_6_v, 426, "", "./tensorflow/c/kernels_experimental.cc", "TF_MaybeLockVariableInputMutexesInOrder");

  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  bool any_resource = false;
  std::vector<int> input_ids(inputs, inputs + len);
  for (auto i : input_ids) {
    if (cc_ctx->input_dtype(i) == tensorflow::DT_RESOURCE) {
      any_resource = true;
      break;
    }
  }
  if (!do_lock && !any_resource) {
    *lockHolder = new TF_VariableInputLockHolder({}, {}, {});
    TF_SetStatus(status, TF_OK, "");
    return;
  }
  std::vector<tensorflow::Var*> vars;
  std::vector<tensorflow::mutex*> mutexes;
  std::vector<int32_t> acquire_order;
  for (auto input : input_ids) {
    tensorflow::Var* var;
    tensorflow::mutex* mutex =
        GetTrainingVariableMutex(ctx, input, sparse, copyFunc, &var);
    if (var) vars.push_back(var);
    // Only lock each mutex once if duplicates exist (n^2 but n is 2 or 3).
    if (std::find(mutexes.begin(), mutexes.end(), mutex) == mutexes.end()) {
      acquire_order.push_back(mutexes.size());
      mutexes.push_back(mutex);
    }
  }
  std::sort(acquire_order.begin(), acquire_order.end(),
            [&mutexes](int a, int b) { return mutexes[a] < mutexes[b]; });

  auto locks = absl::make_unique<std::vector<tensorflow::mutex_lock>>();
  auto shared_locks =
      absl::make_unique<std::vector<tensorflow::tf_shared_lock>>();
  locks->reserve(acquire_order.size());

  for (auto input : acquire_order) {
    tensorflow::Var* var;
    tensorflow::mutex* mu =
        GetTrainingVariableMutex(ctx, input, sparse, copyFunc, &var);
    tensorflow::core::ScopedUnref scoped_unref(var);
    if (mu != nullptr) {
      if (do_lock) {
        locks->emplace_back(*mu);
      } else {
        shared_locks->emplace_back(*mu);
      }
    }
  }
  *lockHolder = new TF_VariableInputLockHolder(
      std::move(vars), std::move(locks), std::move(shared_locks));
  TF_SetStatus(status, TF_OK, "");
}

void TF_GetInputTensorFromVariable(TF_OpKernelContext* ctx, int input,
                                   bool lock_held, bool isVariantType,
                                   bool sparse,
                                   void (*copyFunc)(TF_OpKernelContext* ctx,
                                                    TF_Tensor* source,
                                                    TF_Tensor* dest),
                                   TF_Tensor** out, TF_Status* status) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSkernels_experimentalDTcc mht_7(mht_7_v, 490, "", "./tensorflow/c/kernels_experimental.cc", "TF_GetInputTensorFromVariable");

  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  tensorflow::Status s;
  if (cc_ctx->input_dtype(input) == tensorflow::DT_RESOURCE) {
    tensorflow::core::RefCountPtr<tensorflow::Var> var;
    OP_REQUIRES_OK(
        cc_ctx, LookupResource(cc_ctx, HandleFromInput(cc_ctx, input), &var));
    if (sparse) {
      OP_REQUIRES_OK(cc_ctx, EnsureSparseVariableAccess(ctx, isVariantType,
                                                        copyFunc, var.get()));
      *out = ::tensorflow::TF_TensorFromTensor(*var->tensor(), &s);
      TF_SetStatus(status, TF_OK, "");
      return;
    }
    OP_REQUIRES_OK(cc_ctx, PrepareToUpdateVariable(
                               ctx, var->tensor(),
                               var->copy_on_read_mode.load(), false, copyFunc));
    *out = ::tensorflow::TF_TensorFromTensor(*var->tensor(), &s);
    TF_SetStatus(status, TF_OK, "");
    return;
  }
  *out = ::tensorflow::TF_TensorFromTensor(
      cc_ctx->mutable_input(input, lock_held), &s);
  TF_SetStatus(status, TF_OK, "");
}

void TF_OpKernelContext_ForwardRefInputToRefOutput(TF_OpKernelContext* ctx,
                                                   int32_t input_index,
                                                   int32_t output_index) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSkernels_experimentalDTcc mht_8(mht_8_v, 521, "", "./tensorflow/c/kernels_experimental.cc", "TF_OpKernelContext_ForwardRefInputToRefOutput");

  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  if (cc_ctx->input_dtype(input_index) != tensorflow::DT_RESOURCE) {
    cc_ctx->forward_ref_input_to_ref_output(input_index, output_index);
  }
}

void TF_ReleaseVariableInputLockHolder(TF_VariableInputLockHolder* lockHolder) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSkernels_experimentalDTcc mht_9(mht_9_v, 531, "", "./tensorflow/c/kernels_experimental.cc", "TF_ReleaseVariableInputLockHolder");

  if (lockHolder != nullptr) {
    lockHolder->locks.reset();
    for (tensorflow::Var* var : lockHolder->vars) {
      var->Unref();
    }
    delete lockHolder;
  }
}

void TF_GetInputByName(TF_OpKernelContext* ctx, const char* inputName,
                       TF_Tensor** tensor, TF_Status* status) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("inputName: \"" + (inputName == nullptr ? std::string("nullptr") : std::string((char*)inputName)) + "\"");
   MHTracer_DTPStensorflowPScPSkernels_experimentalDTcc mht_10(mht_10_v, 546, "", "./tensorflow/c/kernels_experimental.cc", "TF_GetInputByName");

  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  const ::tensorflow::Tensor* cc_tensor = nullptr;
  tensorflow::Status s = cc_ctx->input(inputName, &cc_tensor);

  if (!s.ok()) {
    ::tensorflow::Set_TF_Status_from_Status(status, s);
    return;
  }
  TF_Tensor* result =
      ::tensorflow::TF_TensorFromTensor(*cc_tensor, &status->status);
  if (TF_GetCode(status) == TF_OK) {
    *tensor = result;
  }
}

void TF_OpKernelConstruction_GetAttrTensorShape(TF_OpKernelConstruction* ctx,
                                                const char* attr_name,
                                                int64_t* dims, size_t num_dims,
                                                TF_Status* status) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSkernels_experimentalDTcc mht_11(mht_11_v, 569, "", "./tensorflow/c/kernels_experimental.cc", "TF_OpKernelConstruction_GetAttrTensorShape");

  ::tensorflow::TensorShape shape;
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx);
  ::tensorflow::Status s = cc_ctx->GetAttr(attr_name, &shape);
  ::tensorflow::Set_TF_Status_from_Status(status, s);
  size_t rank = static_cast<size_t>(shape.dims());

  if (!status->status.ok()) return;

  if (num_dims != rank) {
    status->status = InvalidArgument("Expected rank is ", num_dims,
                                     " but actual rank is ", rank);
    return;
  }

  for (int i = 0; i < rank; ++i) {
    dims[i] = static_cast<int64_t>(shape.dim_size(i));
  }
}

bool TF_IsRefInput(TF_OpKernelContext* ctx, int i, TF_Status* status) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScPSkernels_experimentalDTcc mht_12(mht_12_v, 592, "", "./tensorflow/c/kernels_experimental.cc", "TF_IsRefInput");

  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  if (i < 0 || i >= cc_ctx->num_inputs()) {
    TF_SetStatus(status, TF_OUT_OF_RANGE, "input index out of range");
    return false;
  }
  TF_SetStatus(status, TF_OK, "");
  return cc_ctx->input_is_ref(i);
}
