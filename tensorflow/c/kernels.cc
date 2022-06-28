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
class MHTracer_DTPStensorflowPScPSkernelsDTcc {
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
   MHTracer_DTPStensorflowPScPSkernelsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSkernelsDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/kernels.h"

#include <memory>

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
// Required for IS_MOBILE_PLATFORM definition
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"
#if !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)
#include "tensorflow/c/experimental/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/stream.h"
#endif  // !defined(IS_MOBILE_PLATFORM) && !defined(IS_SLIM_BUILD)

using tensorflow::errors::InvalidArgument;
// This file forms the basis of a stable ABI for third-party kernel
// implementations. It is crucial that changes to this file are made cautiously
// and with a focus on maintaining both source and binary compatibility.

struct TF_KernelBuilder {
  ::tensorflow::KernelDefBuilder* cc_builder;

  void* (*create_function)(TF_OpKernelConstruction*);
  void (*compute_function)(void*, TF_OpKernelContext*);
  void (*delete_function)(void*);
};

TF_KernelBuilder* TF_NewKernelBuilder(
    const char* op_name, const char* device_name,
    void* (*create_func)(TF_OpKernelConstruction*),
    void (*compute_func)(void*, TF_OpKernelContext*),
    void (*delete_func)(void*)) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   mht_0_v.push_back("device_name: \"" + (device_name == nullptr ? std::string("nullptr") : std::string((char*)device_name)) + "\"");
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_0(mht_0_v, 223, "", "./tensorflow/c/kernels.cc", "TF_NewKernelBuilder");

  TF_KernelBuilder* result = new TF_KernelBuilder;
  result->cc_builder = new ::tensorflow::KernelDefBuilder(op_name);
  result->cc_builder->Device(device_name);
  result->create_function = create_func;
  result->compute_function = compute_func;
  result->delete_function = delete_func;
  return result;
}

void TF_DeleteKernelBuilder(TF_KernelBuilder* builder) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_1(mht_1_v, 236, "", "./tensorflow/c/kernels.cc", "TF_DeleteKernelBuilder");

  if (builder != nullptr) {
    delete builder->cc_builder;
    delete builder;
  }
}

namespace tensorflow {
namespace {

#define CASE(type)                                               \
  case DataTypeToEnum<type>::value: {                            \
    kernel_builder->cc_builder->TypeConstraint<type>(attr_name); \
    break;                                                       \
  }

void AddTypeConstraint(TF_KernelBuilder* kernel_builder, const char* attr_name,
                       const DataType dtype, TF_Status* status) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_2(mht_2_v, 257, "", "./tensorflow/c/kernels.cc", "AddTypeConstraint");

  // This needs to be under tensorflow:: namespace so that
  // TF_CALL_ALL_TYPES macro can find tensorflow::string as string.
  switch (dtype) {
    TF_CALL_ALL_TYPES(CASE);
    TF_CALL_QUANTIZED_TYPES(CASE);
    TF_CALL_quint16(CASE);
    TF_CALL_qint16(CASE);
    default:
      status->status = errors::Unimplemented("Unexpected type ", dtype);
      return;
  }
  TF_SetStatus(status, TF_OK, "");
}
#undef CASE

}  // namespace
}  // namespace tensorflow

namespace {
const tensorflow::AttrValue* GetAttrValue(TF_OpKernelConstruction* ctx,
                                          const char* attr_name,
                                          TF_Status* status) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_3(mht_3_v, 283, "", "./tensorflow/c/kernels.cc", "GetAttrValue");

  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx);
  const tensorflow::AttrValue* attr =
      ::tensorflow::AttrSlice(cc_ctx->def()).Find(attr_name);
  if (attr == nullptr) {
    status->status = InvalidArgument("Operation '", cc_ctx->def().name(),
                                     "' has no attr named '", attr_name, "'.");
  }
  return attr;
}
}  // namespace

void TF_KernelBuilder_TypeConstraint(TF_KernelBuilder* kernel_builder,
                                     const char* attr_name,
                                     const TF_DataType type,
                                     TF_Status* status) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_4(mht_4_v, 302, "", "./tensorflow/c/kernels.cc", "TF_KernelBuilder_TypeConstraint");

  tensorflow::DataType dtype = static_cast<tensorflow::DataType>(type);
  tensorflow::AddTypeConstraint(kernel_builder, attr_name, dtype, status);
}

void TF_KernelBuilder_HostMemory(TF_KernelBuilder* kernel_builder,
                                 const char* arg_name) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("arg_name: \"" + (arg_name == nullptr ? std::string("nullptr") : std::string((char*)arg_name)) + "\"");
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_5(mht_5_v, 312, "", "./tensorflow/c/kernels.cc", "TF_KernelBuilder_HostMemory");

  kernel_builder->cc_builder->HostMemory(arg_name);
}

void TF_KernelBuilder_Priority(TF_KernelBuilder* kernel_builder,
                               int32_t priority_number) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_6(mht_6_v, 320, "", "./tensorflow/c/kernels.cc", "TF_KernelBuilder_Priority");

  kernel_builder->cc_builder->Priority(priority_number);
}

namespace tensorflow {
namespace {

// An OpKernel whose methods delegate to C function pointers.
class COpKernel : public OpKernel {
 public:
  explicit COpKernel(OpKernelConstruction* ctx,
                     void* (*create_func)(TF_OpKernelConstruction*),
                     void (*compute_func)(void*, TF_OpKernelContext*),
                     void (*delete_func)(void*))
      : OpKernel(ctx), compute_func_(compute_func), delete_func_(delete_func) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_7(mht_7_v, 337, "", "./tensorflow/c/kernels.cc", "COpKernel");

    if (create_func != nullptr) {
      c_kernel_ =
          (*create_func)(reinterpret_cast<TF_OpKernelConstruction*>(ctx));
    } else {
      c_kernel_ = nullptr;
    }
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_8(mht_8_v, 349, "", "./tensorflow/c/kernels.cc", "Compute");

    (*compute_func_)(c_kernel_, reinterpret_cast<TF_OpKernelContext*>(ctx));
  }

  ~COpKernel() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_9(mht_9_v, 356, "", "./tensorflow/c/kernels.cc", "~COpKernel");

    if (delete_func_ != nullptr) {
      (*delete_func_)(c_kernel_);
    }
  }

 private:
  void (*compute_func_)(void*, TF_OpKernelContext* context);
  void (*delete_func_)(void*);
  void* c_kernel_;
};

// A KernelFactory that returns COpKernel instances.
class KernelBuilderFactory
    : public ::tensorflow::kernel_factory::OpKernelFactory {
 public:
  explicit KernelBuilderFactory(TF_KernelBuilder* builder)
      : builder_(builder) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_10(mht_10_v, 376, "", "./tensorflow/c/kernels.cc", "KernelBuilderFactory");
}
  ::tensorflow::OpKernel* Create(
      ::tensorflow::OpKernelConstruction* context) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_11(mht_11_v, 381, "", "./tensorflow/c/kernels.cc", "Create");

    return new ::tensorflow::COpKernel(context, builder_->create_function,
                                       builder_->compute_function,
                                       builder_->delete_function);
  }
  ~KernelBuilderFactory() override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_12(mht_12_v, 389, "", "./tensorflow/c/kernels.cc", "~KernelBuilderFactory");
 TF_DeleteKernelBuilder(builder_); }

 private:
  TF_KernelBuilder* builder_;
};
}  // namespace
}  // namespace tensorflow

void TF_RegisterKernelBuilder(const char* name, TF_KernelBuilder* builder,
                              TF_Status* status) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_13(mht_13_v, 402, "", "./tensorflow/c/kernels.cc", "TF_RegisterKernelBuilder");

  using tensorflow::register_kernel::Name;

  tensorflow::kernel_factory::OpKernelRegistrar(
      builder->cc_builder->Build(), name,
      absl::make_unique<tensorflow::KernelBuilderFactory>(builder));

  TF_SetStatus(status, TF_OK, "");
}

// This function is only for pluggable device.
// It will return nullptr in all other cases.
// This function is experimental and subject to change.
SP_Stream TF_GetStream(TF_OpKernelContext* ctx, TF_Status* status) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_14(mht_14_v, 418, "", "./tensorflow/c/kernels.cc", "TF_GetStream");

#if defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
  status->status = tensorflow::errors::Unimplemented(
      "Accessing device stream is not supported on mobile. File a bug at "
      "https://github.com/tensorflow/tensorflow/issues if this feature is "
      "important to you");
  return nullptr;
#else
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  if (cc_ctx->op_device_context() == nullptr) {  // CPU Device
    status->status = tensorflow::errors::FailedPrecondition(
        "Accessing device stream is not supported for a CPU device.");
    return nullptr;
  } else if (!cc_ctx->op_device_context()->IsPluggableDevice()) {
    status->status = tensorflow::errors::FailedPrecondition(
        "Accessing device stream is only supported for pluggable devices.");
    return nullptr;
  } else {  // Is a PluggableDevice
    TF_SetStatus(status, TF_OK, "");
    auto c_stream = static_cast<stream_executor::CStream*>(
        cc_ctx->op_device_context()->stream()->implementation());
    return c_stream->Handle();
  }
#endif  // defined(IS_MOBILE_PLATFORM) || defined(IS_SLIM_BUILD)
}

int TF_NumInputs(TF_OpKernelContext* ctx) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_15(mht_15_v, 447, "", "./tensorflow/c/kernels.cc", "TF_NumInputs");

  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  return cc_ctx->num_inputs();
}

int TF_NumOutputs(TF_OpKernelContext* ctx) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_16(mht_16_v, 455, "", "./tensorflow/c/kernels.cc", "TF_NumOutputs");

  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  return cc_ctx->num_outputs();
}

void TF_GetInput(TF_OpKernelContext* ctx, int i, TF_Tensor** tensor,
                 TF_Status* status) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_17(mht_17_v, 464, "", "./tensorflow/c/kernels.cc", "TF_GetInput");

  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  if (i < 0 || i >= cc_ctx->num_inputs()) {
    TF_SetStatus(status, TF_OUT_OF_RANGE, "input index out of range");
    return;
  }
  const ::tensorflow::Tensor& cc_tensor(cc_ctx->input(i));
  TF_Tensor* result =
      ::tensorflow::TF_TensorFromTensor(cc_tensor, &status->status);
  if (TF_GetCode(status) == TF_OK) {
    *tensor = result;
  }
}

void TF_SetOutput(TF_OpKernelContext* ctx, int i, const TF_Tensor* tensor,
                  TF_Status* status) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_18(mht_18_v, 482, "", "./tensorflow/c/kernels.cc", "TF_SetOutput");

  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  if (i < 0 || i >= cc_ctx->num_outputs()) {
    TF_SetStatus(status, TF_OUT_OF_RANGE, "output index out of range");
    return;
  }
  ::tensorflow::Tensor cc_tensor;
  ::tensorflow::Status s = ::tensorflow::TF_TensorToTensor(tensor, &cc_tensor);
  TF_SetStatus(status, TF_OK, "");
  ::tensorflow::Set_TF_Status_from_Status(status, s);
  if (s.ok()) {
    cc_ctx->set_output(i, cc_tensor);
  }
}

void TF_OpKernelConstruction_Failure(TF_OpKernelConstruction* ctx,
                                     TF_Status* status) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_19(mht_19_v, 501, "", "./tensorflow/c/kernels.cc", "TF_OpKernelConstruction_Failure");

  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx);
  ::tensorflow::Status s(::tensorflow::StatusFromTF_Status(status));
  cc_ctx->CtxFailure(s);
}

void TF_OpKernelContext_Failure(TF_OpKernelContext* ctx, TF_Status* status) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_20(mht_20_v, 510, "", "./tensorflow/c/kernels.cc", "TF_OpKernelContext_Failure");

  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  ::tensorflow::Status s(::tensorflow::StatusFromTF_Status(status));
  cc_ctx->CtxFailure(s);
}

void TF_OpKernelConstruction_GetAttrSize(TF_OpKernelConstruction* ctx,
                                         const char* attr_name,
                                         int32_t* list_size,
                                         int32_t* total_size,
                                         TF_Status* status) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_21(mht_21_v, 524, "", "./tensorflow/c/kernels.cc", "TF_OpKernelConstruction_GetAttrSize");

  const tensorflow::AttrValue* attr = GetAttrValue(ctx, attr_name, status);
  if (!status->status.ok()) {
    *list_size = -1;
    *total_size = -1;
    return;
  }
  switch (attr->value_case()) {
#define SINGLE_CASE(kK, attr_type, size_expr) \
  case tensorflow::AttrValue::kK:             \
    *list_size = -1;                          \
    *total_size = size_expr;                  \
    break;

    SINGLE_CASE(kS, TF_ATTR_STRING, attr->s().length());
    SINGLE_CASE(kI, TF_ATTR_INT, -1);
    SINGLE_CASE(kF, TF_ATTR_FLOAT, -1);
    SINGLE_CASE(kB, TF_ATTR_BOOL, -1);
    SINGLE_CASE(kType, TF_ATTR_TYPE, -1);
    SINGLE_CASE(kShape, TF_ATTR_SHAPE,
                attr->shape().unknown_rank() ? -1 : attr->shape().dim_size());
    SINGLE_CASE(kTensor, TF_ATTR_TENSOR, -1);
#undef SINGLE_CASE

    case tensorflow::AttrValue::kList:
      *list_size = 0;
      *total_size = -1;
#define LIST_CASE(field, attr_type, ...)      \
  if (attr->list().field##_size() > 0) {      \
    *list_size = attr->list().field##_size(); \
    __VA_ARGS__;                              \
    break;                                    \
  }

      LIST_CASE(
          s, TF_ATTR_STRING, *total_size = 0;
          for (int i = 0; i < attr->list().s_size();
               ++i) { *total_size += attr->list().s(i).size(); });
      LIST_CASE(i, TF_ATTR_INT);
      LIST_CASE(f, TF_ATTR_FLOAT);
      LIST_CASE(b, TF_ATTR_BOOL);
      LIST_CASE(type, TF_ATTR_TYPE);
      LIST_CASE(
          shape, TF_ATTR_SHAPE, *total_size = 0;
          for (int i = 0; i < attr->list().shape_size(); ++i) {
            const auto& s = attr->list().shape(i);
            *total_size += s.unknown_rank() ? 0 : s.dim_size();
          });
      LIST_CASE(tensor, TF_ATTR_TENSOR);
      LIST_CASE(tensor, TF_ATTR_FUNC);
#undef LIST_CASE
      break;

    case tensorflow::AttrValue::kPlaceholder:
      *list_size = -1;
      *total_size = -1;
      break;

    case tensorflow::AttrValue::kFunc:
      *list_size = -1;
      *total_size = -1;
      break;

    case tensorflow::AttrValue::VALUE_NOT_SET:
      status->status =
          InvalidArgument("Attribute '", attr_name, "' has no value set");
      break;
  }
}

#define DEFINE_TF_GETATTR(func, c_type, cc_type, attr_type, list_field)        \
  void TF_OpKernelConstruction_GetAttr##func(TF_OpKernelConstruction* ctx,     \
                                             const char* attr_name,            \
                                             c_type* val, TF_Status* status) { \
    TF_SetStatus(status, TF_OK, "");                                           \
    cc_type v;                                                                 \
    auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx); \
    ::tensorflow::Status s = cc_ctx->GetAttr(attr_name, &v);                   \
    ::tensorflow::Set_TF_Status_from_Status(status, s);                        \
    if (s.ok()) {                                                              \
      *val = static_cast<c_type>(v);                                           \
    }                                                                          \
  }                                                                            \
  void TF_OpKernelConstruction_GetAttr##func##List(                            \
      TF_OpKernelConstruction* ctx, const char* attr_name, c_type* vals,       \
      int max_vals, TF_Status* status) {                                       \
    TF_SetStatus(status, TF_OK, "");                                           \
    const tensorflow::AttrValue* attr = GetAttrValue(ctx, attr_name, status);  \
    if (!status->status.ok()) return;                                          \
    if (attr->value_case() != tensorflow::AttrValue::kList) {                  \
      status->status =                                                         \
          InvalidArgument("Value for '", attr_name, "' is not a list.");       \
      return;                                                                  \
    }                                                                          \
    status->status =                                                           \
        tensorflow::AttrValueHasType(*attr, "list(" attr_type ")");            \
    if (!status->status.ok()) return;                                          \
    const auto len = std::min(max_vals, attr->list().list_field##_size());     \
    for (int i = 0; i < len; ++i) {                                            \
      vals[i] = static_cast<c_type>(attr->list().list_field(i));               \
    }                                                                          \
  }

DEFINE_TF_GETATTR(Type, TF_DataType, tensorflow::DataType, "type", type)
DEFINE_TF_GETATTR(Int32, int32_t, int32_t, "int", i)
DEFINE_TF_GETATTR(Int64, int64_t, int64_t, "int", i)
DEFINE_TF_GETATTR(Float, float, float, "float", f)
DEFINE_TF_GETATTR(Bool, TF_Bool, bool, "bool", b)

void TF_OpKernelConstruction_GetAttrString(TF_OpKernelConstruction* ctx,
                                           const char* attr_name, char* value,
                                           size_t max_length,
                                           TF_Status* status) {
  std::string v;
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx);
  ::tensorflow::Status s = cc_ctx->GetAttr(attr_name, &v);
  ::tensorflow::Set_TF_Status_from_Status(status, s);

  if (!status->status.ok()) return;

  if (max_length <= 0) {
    return;
  }
  std::memcpy(value, v.data(), std::min<size_t>(v.length(), max_length));
}

void TF_OpKernelConstruction_GetAttrStringList(TF_OpKernelConstruction* ctx,
                                               const char* attr_name,
                                               char** values, size_t* lengths,
                                               int max_values, void* storage,
                                               size_t storage_size,
                                               TF_Status* status) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_22(mht_22_v, 659, "", "./tensorflow/c/kernels.cc", "TF_OpKernelConstruction_GetAttrStringList");

  std::vector<std::string> v;
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx);
  ::tensorflow::Status s = cc_ctx->GetAttr(attr_name, &v);
  ::tensorflow::Set_TF_Status_from_Status(status, s);

  if (!status->status.ok()) return;

  const auto len = std::min(max_values, static_cast<int>(v.size()));
  char* p = static_cast<char*>(storage);
  for (int i = 0; i < len; ++i) {
    const std::string& s = v[i];
    values[i] = p;
    lengths[i] = s.size();
    if ((p + s.size()) > (static_cast<char*>(storage) + storage_size)) {
      status->status = InvalidArgument(
          "Not enough storage to hold the requested list of strings");
      return;
    }
    memcpy(values[i], s.data(), s.size());
    p += s.size();
  }
}

void TF_OpKernelConstruction_GetAttrTensor(TF_OpKernelConstruction* ctx,
                                           const char* attr_name,
                                           TF_Tensor** val, TF_Status* status) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_23(mht_23_v, 689, "", "./tensorflow/c/kernels.cc", "TF_OpKernelConstruction_GetAttrTensor");

  *val = nullptr;
  ::tensorflow::Tensor t;
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx);
  ::tensorflow::Status s = cc_ctx->GetAttr(attr_name, &t);
  ::tensorflow::Set_TF_Status_from_Status(status, s);

  if (!status->status.ok()) return;

  *val = TF_TensorFromTensor(t, &status->status);
}

void TF_OpKernelConstruction_GetAttrTensorList(TF_OpKernelConstruction* ctx,
                                               const char* attr_name,
                                               TF_Tensor** vals, int max_values,
                                               TF_Status* status) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_24(mht_24_v, 708, "", "./tensorflow/c/kernels.cc", "TF_OpKernelConstruction_GetAttrTensorList");

  std::vector<::tensorflow::Tensor> v;
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx);
  ::tensorflow::Status s = cc_ctx->GetAttr(attr_name, &v);
  ::tensorflow::Set_TF_Status_from_Status(status, s);

  if (!status->status.ok()) return;

  const auto len = std::min(max_values, static_cast<int>(v.size()));
  for (int i = 0; i < len; ++i) {
    vals[i] = TF_TensorFromTensor(v[i], &status->status);
    if (!status->status.ok()) return;
  }
}

bool TF_OpKernelConstruction_HasAttr(TF_OpKernelConstruction* ctx,
                                     const char* attr_name, TF_Status* status) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_25(mht_25_v, 728, "", "./tensorflow/c/kernels.cc", "TF_OpKernelConstruction_HasAttr");

  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelConstruction*>(ctx);
  return cc_ctx->HasAttr(attr_name);
}

TF_StringView TF_OpKernelConstruction_GetName(TF_OpKernelConstruction* ctx) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_26(mht_26_v, 736, "", "./tensorflow/c/kernels.cc", "TF_OpKernelConstruction_GetName");

  auto* cc_ctx = reinterpret_cast<tensorflow::OpKernelConstruction*>(ctx);
  TF_StringView string_view_of_name;
  string_view_of_name.data = cc_ctx->def().name().data();
  string_view_of_name.len = cc_ctx->def().name().length();
  return string_view_of_name;
}

TF_DataType TF_ExpectedOutputDataType(TF_OpKernelContext* ctx, int i) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_27(mht_27_v, 747, "", "./tensorflow/c/kernels.cc", "TF_ExpectedOutputDataType");

  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(ctx);
  CHECK_GE(i, 0);
  CHECK_LT(i, cc_ctx->num_outputs());
  return static_cast<TF_DataType>(cc_ctx->expected_output_dtype(i));
}

int64_t TF_StepId(TF_OpKernelContext* ctx) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_28(mht_28_v, 757, "", "./tensorflow/c/kernels.cc", "TF_StepId");

  return reinterpret_cast<::tensorflow::OpKernelContext*>(ctx)->step_id();
}

TF_Tensor* TF_AllocateOutput(TF_OpKernelContext* context, int index,
                             TF_DataType dtype, const int64_t* dims,
                             int num_dims, size_t len, TF_Status* status) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_29(mht_29_v, 766, "", "./tensorflow/c/kernels.cc", "TF_AllocateOutput");

  TF_SetStatus(status, TF_OK, "");
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(context);
  static_assert(sizeof(int64_t) == sizeof(int64_t),
                "64-bit int types should match in size");
  tensorflow::gtl::ArraySlice<const int64_t> dimarray(
      reinterpret_cast<const int64_t*>(dims), num_dims);
  tensorflow::Tensor* tensor;
  tensorflow::Status s = cc_ctx->allocate_output(
      index, tensorflow::TensorShape(dimarray), &tensor);
  if (!s.ok()) {
    ::tensorflow::Set_TF_Status_from_Status(status, s);
    return nullptr;
  }
  TF_Tensor* tf_tensor = TF_TensorFromTensor(*tensor, &s);
  if (!s.ok()) {
    ::tensorflow::Set_TF_Status_from_Status(status, s);
    return nullptr;
  }
  return tf_tensor;
}

TF_Tensor* TF_ForwardInputOrAllocateOutput(
    TF_OpKernelContext* context, const int* candidate_input_indices,
    int num_candidate_input_indices, int output_index,
    const int64_t* output_dims, int output_num_dims, int* forwarded_input,
    TF_Status* status) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_30(mht_30_v, 795, "", "./tensorflow/c/kernels.cc", "TF_ForwardInputOrAllocateOutput");

  TF_SetStatus(status, TF_OK, "");
  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(context);

  static_assert(sizeof(int64_t) == sizeof(int64_t),
                "64-bit int types should match in size");
  tensorflow::gtl::ArraySlice<int> input_indices_array(
      candidate_input_indices, num_candidate_input_indices);
  tensorflow::gtl::ArraySlice<const int64_t> output_dimarray(
      reinterpret_cast<const int64_t*>(output_dims), output_num_dims);
  tensorflow::Tensor* output_tensor_pointer;
  tensorflow::Status s = cc_ctx->forward_input_or_allocate_output(
      input_indices_array, output_index,
      tensorflow::TensorShape(output_dimarray), &output_tensor_pointer,
      forwarded_input);
  if (!s.ok()) {
    ::tensorflow::Set_TF_Status_from_Status(status, s);
    return nullptr;
  }
  TF_Tensor* tf_tensor_output = TF_TensorFromTensor(*output_tensor_pointer, &s);
  if (!s.ok()) {
    ::tensorflow::Set_TF_Status_from_Status(status, s);
    return nullptr;
  }
  return tf_tensor_output;
}

TF_Tensor* TF_AllocateTemp(TF_OpKernelContext* context, TF_DataType dtype,
                           const int64_t* dims, int num_dims,
                           TF_AllocatorAttributes* attributes,
                           TF_Status* status) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScPSkernelsDTcc mht_31(mht_31_v, 828, "", "./tensorflow/c/kernels.cc", "TF_AllocateTemp");

  auto* cc_ctx = reinterpret_cast<::tensorflow::OpKernelContext*>(context);
  TF_SetStatus(status, TF_OK, "");
  static_assert(sizeof(int64_t) == sizeof(int64_t),
                "64-bit int types should match in size");
  tensorflow::gtl::ArraySlice<const int64_t> dimarray(
      reinterpret_cast<const int64_t*>(dims), num_dims);
  if (attributes && !attributes->struct_size) {
    TF_SetStatus(
        status, TF_INVALID_ARGUMENT,
        "TF_AllocatorAttributes struct "
        "size member must be set to TF_ALLOCATOR_ATTRIBUTES_STRUCT_SIZE");
    return nullptr;
  }
  tensorflow::AllocatorAttributes allocator_attr;
  if (attributes && attributes->on_host) {
    allocator_attr.set_on_host(true);
  }
  tensorflow::Status s;
  tensorflow::Tensor tensor;
  s = cc_ctx->allocate_temp(static_cast<tensorflow::DataType>(dtype),
                            tensorflow::TensorShape(dimarray), &tensor,
                            allocator_attr);
  if (!s.ok()) {
    ::tensorflow::Set_TF_Status_from_Status(status, s);
    return nullptr;
  }
  TF_Tensor* tf_tensor;
  tf_tensor = TF_TensorFromTensor(tensor, &s);
  if (!s.ok()) {
    ::tensorflow::Set_TF_Status_from_Status(status, s);
    return nullptr;
  }
  return tf_tensor;
}
