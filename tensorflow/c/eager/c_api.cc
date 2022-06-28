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
class MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc {
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
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc() {
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

#include "tensorflow/c/eager/c_api.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_op_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/custom_device.h"
#include "tensorflow/core/common_runtime/eager/custom_device_op_handler.h"
#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/common_runtime/eager/placement_utils.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/version.h"

// "tensorflow/core/platform/platform.h" must be included first before using
// PLATFORM_GOOGLE, IS_MOBILE_PLATFORM, etc.
#if defined(PLATFORM_GOOGLE) && !defined(LIBTPU_ON_GCE)
#include "tensorflow/core/tfrt/eager/c_api_tfrt.h"
#include "tensorflow/core/tfrt/eager/c_api_tfrt_distributed_impl.h"
#endif  // PLATFORM_GOOGLE && !LIBTPU_ON_GCE

#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/common_runtime/eager/context_distributed_manager.h"
#endif  // !IS_MOBILE_PLATFORM

using tensorflow::string;

namespace {

string DeviceName(const tensorflow::Device* d) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_0(mht_0_v, 248, "", "./tensorflow/c/eager/c_api.cc", "DeviceName");

  return (d == nullptr) ? "cpu:0" : d->name();
}

// Annotate eager runtime construction context to the given `function_def` as
// an attribute.
void AnnotateEagerRuntimeConstructionContext(
    tensorflow::FunctionDef& function_def) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_1(mht_1_v, 258, "", "./tensorflow/c/eager/c_api.cc", "AnnotateEagerRuntimeConstructionContext");

  tensorflow::AttrValue value;
  SetAttrValue("kEagerRuntime", &value);
  (*function_def.mutable_attr())["_construction_context"] = value;
}

}  // namespace

extern "C" {

TFE_ContextOptions* TFE_NewContextOptions() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_2(mht_2_v, 271, "", "./tensorflow/c/eager/c_api.cc", "TFE_NewContextOptions");
 return new TFE_ContextOptions; }

void TFE_ContextOptionsSetConfig(TFE_ContextOptions* options, const void* proto,
                                 size_t proto_len, TF_Status* status) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_3(mht_3_v, 277, "", "./tensorflow/c/eager/c_api.cc", "TFE_ContextOptionsSetConfig");

  TF_SetConfig(&options->session_options, proto, proto_len, status);
}

void TFE_ContextOptionsSetAsync(TFE_ContextOptions* options,
                                unsigned char enable) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("enable: '" + std::string(1, enable) + "'");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_4(mht_4_v, 286, "", "./tensorflow/c/eager/c_api.cc", "TFE_ContextOptionsSetAsync");

  options->async = enable;
}

void TFE_ContextOptionsSetDevicePlacementPolicy(
    TFE_ContextOptions* options, TFE_ContextDevicePlacementPolicy policy) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_5(mht_5_v, 294, "", "./tensorflow/c/eager/c_api.cc", "TFE_ContextOptionsSetDevicePlacementPolicy");

  options->device_placement_policy = policy;
}

void TFE_DeleteContextOptions(TFE_ContextOptions* options) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_6(mht_6_v, 301, "", "./tensorflow/c/eager/c_api.cc", "TFE_DeleteContextOptions");
 delete options; }

TFE_Context* TFE_NewContext(const TFE_ContextOptions* opts, TF_Status* status) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_7(mht_7_v, 306, "", "./tensorflow/c/eager/c_api.cc", "TFE_NewContext");

  if (opts->use_tfrt) {
#if defined(PLATFORM_GOOGLE) && !defined(LIBTPU_ON_GCE)
    tfrt::tf::ContextInterface* tfrt_context = new tfrt::tf::ContextInterface(
        opts->session_options.options,
        static_cast<tensorflow::ContextDevicePlacementPolicy>(
            opts->device_placement_policy),
        opts->async, opts->use_tfrt_distributed_runtime);
#if !defined(IS_MOBILE_PLATFORM)
    tfrt_context->SetDistributedManager(
        tfrt::tf::CreateDistributedManagerContext(
            tfrt_context->GetCoreRuntime()->GetHostContext()));
#endif  // !IS_MOBILE_PLATFORM
    return tensorflow::wrap(tfrt_context);
#else
    status->status = tensorflow::errors::Unimplemented("TFRT is not supported");
    return nullptr;
#endif  // PLATFORM_GOOGLE && !LIBTPU_ON_GCE
  }
  std::vector<std::unique_ptr<tensorflow::Device>> devices;
  status->status = tensorflow::DeviceFactory::AddDevices(
      opts->session_options.options, "/job:localhost/replica:0/task:0",
      &devices);
  if (!status->status.ok()) return nullptr;
  std::unique_ptr<tensorflow::DeviceMgr> device_mgr(
      new tensorflow::DynamicDeviceMgr(std::move(devices)));

  tensorflow::Rendezvous* r =
      new tensorflow::IntraProcessRendezvous(device_mgr.get());
  tensorflow::EagerContext* eager_context = new tensorflow::EagerContext(
      opts->session_options.options,
      static_cast<tensorflow::ContextDevicePlacementPolicy>(
          opts->device_placement_policy),
      opts->async, device_mgr.release(),
      /*device_mgr_owned*/ true, r,
      /*cluster_flr=*/nullptr,
      /*collective_executor_mgr=*/nullptr,
      /*run_eager_op_as_function=*/opts->run_eager_op_as_function,
      /*jit_compile_rewrite=*/opts->jit_compile_rewrite);
#if !defined(IS_MOBILE_PLATFORM)
  eager_context->SetDistributedManager(
      std::make_unique<tensorflow::EagerContextDistributedManager>(
          eager_context));
#endif  // !IS_MOBILE_PLATFORM
  return tensorflow::wrap(eager_context);
}

void TFE_DeleteContext(TFE_Context* ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_8(mht_8_v, 356, "", "./tensorflow/c/eager/c_api.cc", "TFE_DeleteContext");

  if (ctx == nullptr) {
    return;
  }

  // ctx->RefCountIsOne() should be true here.
  tensorflow::unwrap(ctx)->Release();
}

TF_DeviceList* TFE_ContextListDevices(TFE_Context* ctx, TF_Status* status) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_9(mht_9_v, 368, "", "./tensorflow/c/eager/c_api.cc", "TFE_ContextListDevices");

  TF_DeviceList* l = new TF_DeviceList;
  tensorflow::unwrap(ctx)->ListDevices(&l->response);
  return l;
}

void TFE_ContextClearCaches(TFE_Context* ctx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_10(mht_10_v, 377, "", "./tensorflow/c/eager/c_api.cc", "TFE_ContextClearCaches");

  tensorflow::unwrap(ctx)->ClearCachesAndThreadExecutors();
}

// Set server_def on the context, possibly updating it.
TF_CAPI_EXPORT extern void TFE_ContextSetServerDef(TFE_Context* ctx,
                                                   int keep_alive_secs,
                                                   const void* proto,
                                                   size_t proto_len,
                                                   TF_Status* status) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_11(mht_11_v, 389, "", "./tensorflow/c/eager/c_api.cc", "TFE_ContextSetServerDef");

#if defined(IS_MOBILE_PLATFORM)
  status->status = tensorflow::errors::Unimplemented(
      "TFE_ContextSetServerDef not supported on mobile");
#else   // !defined(IS_MOBILE_PLATFORM)
  tensorflow::ServerDef server_def;
  if (!server_def.ParseFromArray(proto, proto_len)) {
    status->status = tensorflow::errors::InvalidArgument(
        "Invalid tensorflow.ServerDef protocol buffer");
    return;
  }
  status->status =
      tensorflow::unwrap(ctx)->GetDistributedManager()->SetOrUpdateServerDef(
          server_def, /*reset_context=*/true, keep_alive_secs);
#endif  // !IS_MOBILE_PLATFORM
}

TF_CAPI_EXPORT extern void TFE_ContextUpdateServerDef(TFE_Context* ctx,
                                                      int keep_alive_secs,
                                                      const void* proto,
                                                      size_t proto_len,
                                                      TF_Status* status) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_12(mht_12_v, 413, "", "./tensorflow/c/eager/c_api.cc", "TFE_ContextUpdateServerDef");

#if defined(IS_MOBILE_PLATFORM)
  status->status = tensorflow::errors::Unimplemented(
      "TFE_ContextSetServerDef not supported on mobile");
#else   // !defined(IS_MOBILE_PLATFORM)
  tensorflow::ServerDef server_def;
  tensorflow::EagerContext* context =
      tensorflow::ContextFromInterface(tensorflow::unwrap(ctx));
  if (!server_def.ParseFromArray(proto, proto_len)) {
    status->status = tensorflow::errors::InvalidArgument(
        "Invalid tensorflow.ServerDef protocol buffer");
    return;
  } else if (context->GetContextId() ==
             tensorflow::EagerContext::kInvalidContextId) {
    status->status = tensorflow::errors::InvalidArgument(
        "Trying to update a context with invalid context id.");
  }
  status->status =
      tensorflow::unwrap(ctx)->GetDistributedManager()->SetOrUpdateServerDef(
          server_def, /*reset_context=*/false, keep_alive_secs);
#endif  // !IS_MOBILE_PLATFORM
}

TF_CAPI_EXPORT extern bool TFE_ContextCheckAlive(TFE_Context* ctx,
                                                 const char* worker_name,
                                                 TF_Status* status) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("worker_name: \"" + (worker_name == nullptr ? std::string("nullptr") : std::string((char*)worker_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_13(mht_13_v, 442, "", "./tensorflow/c/eager/c_api.cc", "TFE_ContextCheckAlive");

#if defined(IS_MOBILE_PLATFORM)
  status->status = tensorflow::errors::Unimplemented(
      "TFE_ContextSetServerDef not supported on mobile");
  return false;
#else   // !defined(IS_MOBILE_PLATFORM)
  bool is_alive;
  status->status =
      tensorflow::unwrap(ctx)->GetDistributedManager()->CheckRemoteAlive(
          worker_name, &is_alive);
  return is_alive;
#endif  // !IS_MOBILE_PLATFORM
}

TF_CAPI_EXPORT extern void TFE_ContextAsyncWait(TFE_Context* ctx,
                                                TF_Status* status) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_14(mht_14_v, 460, "", "./tensorflow/c/eager/c_api.cc", "TFE_ContextAsyncWait");

#if defined(IS_MOBILE_PLATFORM)
  status->status = tensorflow::Status::OK();
#else   // !defined(IS_MOBILE_PLATFORM)
  status->status = tensorflow::unwrap(ctx)->AsyncWait();
#endif  // !IS_MOBILE_PLATFORM
}

void TFE_ContextSetThreadLocalDevicePlacementPolicy(
    TFE_Context* ctx, TFE_ContextDevicePlacementPolicy policy) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_15(mht_15_v, 472, "", "./tensorflow/c/eager/c_api.cc", "TFE_ContextSetThreadLocalDevicePlacementPolicy");

  tensorflow::unwrap(ctx)->SetThreadLocalDevicePlacementPolicy(
      static_cast<tensorflow::ContextDevicePlacementPolicy>(policy));
}

// Note: this function looks up a thread local policy. So it should be called in
// the appropriate client thread. In particular, in async mode, it may not be
// safe to call this function from the async EagerExecutor threads.
extern TFE_ContextDevicePlacementPolicy TFE_ContextGetDevicePlacementPolicy(
    TFE_Context* ctx) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_16(mht_16_v, 484, "", "./tensorflow/c/eager/c_api.cc", "TFE_ContextGetDevicePlacementPolicy");

  return static_cast<TFE_ContextDevicePlacementPolicy>(
      tensorflow::unwrap(ctx)->GetDevicePlacementPolicy());
}

TFE_TensorHandle* TFE_NewTensorHandle(const TF_Tensor* t, TF_Status* status) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_17(mht_17_v, 492, "", "./tensorflow/c/eager/c_api.cc", "TFE_NewTensorHandle");

  tensorflow::Tensor tensor;
  status->status = tensorflow::TF_TensorToTensor(t, &tensor);
  if (!status->status.ok()) return nullptr;

  return tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(tensor));
}

void TFE_DeleteTensorHandle(TFE_TensorHandle* h) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_18(mht_18_v, 503, "", "./tensorflow/c/eager/c_api.cc", "TFE_DeleteTensorHandle");

  if (h == nullptr) return;

  tensorflow::profiler::TraceMe activity(
      "TFE_DeleteTensorHandle", tensorflow::profiler::TraceMeLevel::kInfo);
  if (h) {
    tensorflow::unwrap(h)->Release();
  }
}

TF_DataType TFE_TensorHandleDataType(TFE_TensorHandle* h) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_19(mht_19_v, 516, "", "./tensorflow/c/eager/c_api.cc", "TFE_TensorHandleDataType");

  return static_cast<TF_DataType>(tensorflow::unwrap(h)->DataType());
}

int TFE_TensorHandleNumDims(TFE_TensorHandle* h, TF_Status* status) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_20(mht_20_v, 523, "", "./tensorflow/c/eager/c_api.cc", "TFE_TensorHandleNumDims");

  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return -1;
  }

  int num_dims = -1;
  status->status = tensorflow::unwrap(h)->NumDims(&num_dims);
  return num_dims;
}

int64_t TFE_TensorHandleNumElements(TFE_TensorHandle* h, TF_Status* status) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_21(mht_21_v, 537, "", "./tensorflow/c/eager/c_api.cc", "TFE_TensorHandleNumElements");

  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return -1;
  }

  int64_t num_elements = -1;
  status->status = tensorflow::unwrap(h)->NumElements(&num_elements);
  return num_elements;
}

int64_t TFE_TensorHandleDim(TFE_TensorHandle* h, int dim_index,
                            TF_Status* status) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_22(mht_22_v, 552, "", "./tensorflow/c/eager/c_api.cc", "TFE_TensorHandleDim");

  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return -1;
  }

  int64_t dim = -1;
  status->status = tensorflow::unwrap(h)->Dim(dim_index, &dim);
  return dim;
}

const char* TFE_TensorHandleDeviceName(TFE_TensorHandle* h, TF_Status* status) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_23(mht_23_v, 566, "", "./tensorflow/c/eager/c_api.cc", "TFE_TensorHandleDeviceName");

  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }
  return tensorflow::unwrap(h)->DeviceName(&status->status);
}

const char* TFE_TensorHandleBackingDeviceName(TFE_TensorHandle* h,
                                              TF_Status* status) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_24(mht_24_v, 578, "", "./tensorflow/c/eager/c_api.cc", "TFE_TensorHandleBackingDeviceName");

  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }
  return tensorflow::unwrap(h)->BackingDeviceName(&status->status);
}

TF_CAPI_EXPORT extern TFE_TensorHandle* TFE_TensorHandleCopySharingTensor(
    TFE_TensorHandle* h, TF_Status* status) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_25(mht_25_v, 590, "", "./tensorflow/c/eager/c_api.cc", "TFE_TensorHandleCopySharingTensor");

  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }

  return tensorflow::wrap(tensorflow::unwrap(h)->Copy());
}

TF_Tensor* TFE_TensorHandleResolve(TFE_TensorHandle* h, TF_Status* status) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_26(mht_26_v, 602, "", "./tensorflow/c/eager/c_api.cc", "TFE_TensorHandleResolve");

  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }

  tensorflow::AbstractTensorInterface* t =
      tensorflow::unwrap(h)->Resolve(&status->status);
  if (t == nullptr) {
    return nullptr;
  }

  return new TF_Tensor{t};
}

void* TFE_TensorHandleDevicePointer(TFE_TensorHandle* h, TF_Status* status) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_27(mht_27_v, 620, "", "./tensorflow/c/eager/c_api.cc", "TFE_TensorHandleDevicePointer");

  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }
  tensorflow::ImmediateExecutionTensorHandle* unwrapped_handle =
      tensorflow::unwrap(h);
  // TODO(b/175427838): It would be nice to be able to use tensorflow::isa here.
  if (tensorflow::CustomDeviceTensorHandle::classof(unwrapped_handle)) {
    return tensorflow::down_cast<tensorflow::CustomDeviceTensorHandle*>(
               unwrapped_handle)
        ->DevicePointer();
  }
  // TODO(b/175427838): It would be nice to be able to use tensorflow::isa here.
  if (!tensorflow::TensorHandle::classof(unwrapped_handle)) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }
  tensorflow::TensorHandle* handle =
      tensorflow::TensorHandleFromInterface(unwrapped_handle);

  if (handle->Type() != tensorflow::TensorHandle::LOCAL) {
    status->status = tensorflow::errors::InvalidArgument(
        "TFE_TensorHandleDevicePointer may not be called on a ",
        handle->TypeString(), " tensor handle.");
    return nullptr;
  }
  tensorflow::Device* device(handle->device());
  if (device != nullptr) {
    status->status = device->Sync();
    if (!status->status.ok()) {
      return nullptr;
    }
  }
  const tensorflow::Tensor* tensor;
  status->status = handle->Tensor(&tensor);
  if (!status->status.ok()) {
    return nullptr;
  }
  return const_cast<void*>(
      static_cast<const void*>(tensor->tensor_data().data()));
}

namespace tensorflow {
namespace {
class CustomDeviceAPI : public tensorflow::CustomDevice {
 public:
  CustomDeviceAPI(TFE_Context* context, TFE_CustomDevice device, void* info,
                  string name)
      : context_(context), device_(device), info_(info), name_(name) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_28(mht_28_v, 673, "", "./tensorflow/c/eager/c_api.cc", "CustomDeviceAPI");
}

  ~CustomDeviceAPI() override {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_29(mht_29_v, 678, "", "./tensorflow/c/eager/c_api.cc", "~CustomDeviceAPI");
 device_.delete_device(info_); }

  const string& name() override {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_30(mht_30_v, 683, "", "./tensorflow/c/eager/c_api.cc", "name");
 return name_; }

  tensorflow::Status CopyTensorToDevice(
      ImmediateExecutionTensorHandle* handle,
      ImmediateExecutionTensorHandle** result) override {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_31(mht_31_v, 690, "", "./tensorflow/c/eager/c_api.cc", "CopyTensorToDevice");

    handle->Ref();
    TF_Status status;
    TFE_TensorHandle* result_handle = device_.copy_tensor_to_device(
        context_, tensorflow::wrap(handle), &status, info_);
    handle->Release();
    if (!status.status.ok()) return status.status;
    *result = tensorflow::unwrap(result_handle);
    (*result)->Ref();
    TFE_DeleteTensorHandle(result_handle);
    return status.status;
  }

  tensorflow::Status CopyTensorFromDevice(
      ImmediateExecutionTensorHandle* handle,
      const tensorflow::string& target_device_name,
      ImmediateExecutionTensorHandle** result) override {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_32(mht_32_v, 709, "", "./tensorflow/c/eager/c_api.cc", "CopyTensorFromDevice");

    TF_Status status;
    handle->Ref();
    TFE_TensorHandle* result_handle = device_.copy_tensor_from_device(
        context_, tensorflow::wrap(handle), target_device_name.c_str(), &status,
        info_);
    handle->Release();
    if (!status.status.ok()) return status.status;
    *result = tensorflow::unwrap(result_handle);
    (*result)->Ref();
    TFE_DeleteTensorHandle(result_handle);
    return status.status;
  }

  tensorflow::Status Execute(const ImmediateExecutionOperation* op,
                             ImmediateExecutionTensorHandle** retvals,
                             int* num_retvals) override {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_33(mht_33_v, 728, "", "./tensorflow/c/eager/c_api.cc", "Execute");

    std::vector<TFE_TensorHandle*> outputs(*num_retvals);
    TF_Status status;
    device_.execute(tensorflow::wrap(op), num_retvals, outputs.data(), &status,
                    info_);
    if (status.status.ok()) {
      for (int i = 0; i < *num_retvals; ++i) {
        retvals[i] = tensorflow::unwrap(outputs[i]);
        retvals[i]->Ref();
        TFE_DeleteTensorHandle(outputs[i]);
      }
    }
    return status.status;
  }

  tensorflow::Status Pack(absl::Span<ImmediateExecutionTensorHandle*> handles,
                          ImmediateExecutionTensorHandle** result) override {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_34(mht_34_v, 747, "", "./tensorflow/c/eager/c_api.cc", "Pack");

    TF_Status status;
    *result = tensorflow::unwrap(device_.pack(context_,
                                              tensorflow::wrap(handles.data()),
                                              handles.size(), &status, info_));
    return status.status;
  }

 private:
  TFE_Context* context_;
  TFE_CustomDevice device_;
  void* info_;
  string name_;
};

// An adapter which wraps the shape/data produced by C custom devices and uses
// it to implement custom device methods.
class CAPICustomDeviceTensorHandle
    : public tensorflow::CustomDeviceTensorHandle {
 public:
  CAPICustomDeviceTensorHandle(tensorflow::ImmediateExecutionContext* context,
                               tensorflow::CustomDevice* device,
                               tensorflow::DataType dtype, void* data,
                               TFE_CustomDeviceTensorHandleMethods methods)
      : tensorflow::CustomDeviceTensorHandle(context, device, dtype),
        data_(data),
        methods_(methods) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_35(mht_35_v, 776, "", "./tensorflow/c/eager/c_api.cc", "CAPICustomDeviceTensorHandle");
}

  ~CAPICustomDeviceTensorHandle() override {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_36(mht_36_v, 781, "", "./tensorflow/c/eager/c_api.cc", "~CAPICustomDeviceTensorHandle");
 methods_.deallocator(data_); }
  void* DevicePointer() const override {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_37(mht_37_v, 785, "", "./tensorflow/c/eager/c_api.cc", "DevicePointer");
 return data_; }
  Status NumDims(int* num_dims) const override {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_38(mht_38_v, 789, "", "./tensorflow/c/eager/c_api.cc", "NumDims");

    TF_Status s;
    *num_dims = methods_.num_dims(data_, &s);
    return s.status;
  }
  Status Dim(int dim_index, int64_t* dim) const override {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_39(mht_39_v, 797, "", "./tensorflow/c/eager/c_api.cc", "Dim");

    TF_Status s;
    *dim = methods_.dim(data_, dim_index, &s);
    return s.status;
  }

  bool PreferCustomSummarizer() const override {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_40(mht_40_v, 806, "", "./tensorflow/c/eager/c_api.cc", "PreferCustomSummarizer");

    return methods_.summarize != nullptr;
  }

  Status SummarizeValue(std::string& summary) const override {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_41(mht_41_v, 813, "", "./tensorflow/c/eager/c_api.cc", "SummarizeValue");

    if (methods_.summarize == nullptr) {
      return tensorflow::CustomDeviceTensorHandle::SummarizeValue(summary);
    }
    TF_Status c_status;
    std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> summary_buffer(
        methods_.summarize(data_, &c_status), TF_DeleteBuffer);
    if (!c_status.status.ok()) {
      return c_status.status;
    }
    summary = std::string(reinterpret_cast<const char*>(summary_buffer->data),
                          summary_buffer->length);
    return Status::OK();
  }

 private:
  void* const data_;
  const TFE_CustomDeviceTensorHandleMethods methods_;
};

}  // namespace
}  // namespace tensorflow

TFE_TensorHandle* TFE_NewCustomDeviceTensorHandle(
    TFE_Context* ctx, const char* device_name, TF_DataType dtype, void* data,
    TFE_CustomDeviceTensorHandleMethods methods, TF_Status* status) {
   std::vector<std::string> mht_42_v;
   mht_42_v.push_back("device_name: \"" + (device_name == nullptr ? std::string("nullptr") : std::string((char*)device_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_42(mht_42_v, 842, "", "./tensorflow/c/eager/c_api.cc", "TFE_NewCustomDeviceTensorHandle");

  tensorflow::ImmediateExecutionContext* context = tensorflow::unwrap(ctx);
  tensorflow::CustomDevice* device = nullptr;
  if (!context->GetCustomDeviceOpHandler().FindCustomDeviceFromName(device_name,
                                                                    &device)) {
    methods.deallocator(data);
    status->status =
        tensorflow::errors::InvalidArgument(device_name, " unknown device.");
    return nullptr;
  }
  return tensorflow::wrap(new tensorflow::CAPICustomDeviceTensorHandle(
      context, device, *reinterpret_cast<tensorflow::DataType*>(&dtype), data,
      methods));
}

TFE_TensorHandle* TFE_NewTensorHandleFromDeviceMemory(
    TFE_Context* ctx, const char* device_name, TF_DataType dtype,
    const int64_t* dims, int num_dims, void* data, size_t len,
    void (*deallocator)(void* data, size_t len, void* arg),
    void* deallocator_arg, TF_Status* status) {
   std::vector<std::string> mht_43_v;
   mht_43_v.push_back("device_name: \"" + (device_name == nullptr ? std::string("nullptr") : std::string((char*)device_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_43(mht_43_v, 865, "", "./tensorflow/c/eager/c_api.cc", "TFE_NewTensorHandleFromDeviceMemory");

  tensorflow::Device* device = nullptr;
  tensorflow::EagerContext* context =
      tensorflow::ContextFromInterface(tensorflow::unwrap(ctx));
  status->status = context->FindDeviceFromName(device_name, &device);
  if (!status->status.ok()) {
    deallocator(data, len, deallocator_arg);
    status->status =
        tensorflow::errors::InvalidArgument(device_name, " unknown device.");
    return nullptr;
  }
  std::vector<int64_t> dimvec(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    dimvec[i] = static_cast<int64_t>(dims[i]);
  }

  // TODO(apassos) do we need to wrap the deallocator here to make sure to sync
  // the device?
  TF_ManagedBuffer* buf =
      new TF_ManagedBuffer(data, len, deallocator, deallocator_arg,
                           /*owns_memory=*/false);

  tensorflow::Tensor t(static_cast<tensorflow::DataType>(dtype),
                       tensorflow::TensorShape(dimvec), buf);
  buf->Unref();
  return tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(
      std::move(t), device, device, context));
}

// This function will block till the operation that produces `h` has
// completed. This is only valid on local TFE_TensorHandles. Returns the size in
// bytes of the memory pointed to by the device pointer returned above.
size_t TFE_TensorHandleDeviceMemorySize(TFE_TensorHandle* h,
                                        TF_Status* status) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_44(mht_44_v, 901, "", "./tensorflow/c/eager/c_api.cc", "TFE_TensorHandleDeviceMemorySize");

  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return 0;
  }
  tensorflow::TensorHandle* handle =
      tensorflow::TensorHandleFromInterface(tensorflow::unwrap(h));
  if (handle->Type() != tensorflow::TensorHandle::LOCAL) {
    status->status = tensorflow::errors::InvalidArgument(
        "TFE_TensorHandleDeviceMemorySize may not be called on a ",
        handle->TypeString(), " tensor handle.");
    return 0;
  }
  const tensorflow::Tensor* tensor;
  status->status = handle->Tensor(&tensor);
  if (!status->status.ok()) {
    return 0;
  }
  return tensor->TotalBytes();
}

TFE_Op* TFE_NewOp(TFE_Context* ctx, const char* op_or_function_name,
                  TF_Status* status) {
   std::vector<std::string> mht_45_v;
   mht_45_v.push_back("op_or_function_name: \"" + (op_or_function_name == nullptr ? std::string("nullptr") : std::string((char*)op_or_function_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_45(mht_45_v, 927, "", "./tensorflow/c/eager/c_api.cc", "TFE_NewOp");

  tensorflow::ImmediateExecutionOperation* new_op =
      tensorflow::unwrap(ctx)->CreateOperation();
  status->status = new_op->Reset(op_or_function_name, nullptr);
  if (!status->status.ok()) {
    new_op->Release();
    new_op = nullptr;
  }
  return tensorflow::wrap(new_op);
}

void TFE_DeleteOp(TFE_Op* op) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_46(mht_46_v, 941, "", "./tensorflow/c/eager/c_api.cc", "TFE_DeleteOp");

  if (op == nullptr) {
    return;
  }

  tensorflow::unwrap(op)->Release();
}

const char* TFE_OpGetName(const TFE_Op* op, TF_Status* status) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_47(mht_47_v, 952, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpGetName");

  return tensorflow::unwrap(op)->Name().c_str();
}

TFE_Context* TFE_OpGetContext(const TFE_Op* op, TF_Status* status) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_48(mht_48_v, 959, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpGetContext");

  return tensorflow::wrap(tensorflow::unwrap(op)->GetContext());
}

void TFE_OpSetDevice(TFE_Op* op, const char* device_name, TF_Status* status) {
   std::vector<std::string> mht_49_v;
   mht_49_v.push_back("device_name: \"" + (device_name == nullptr ? std::string("nullptr") : std::string((char*)device_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_49(mht_49_v, 967, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpSetDevice");

  status->status = tensorflow::unwrap(op)->SetDeviceName(device_name);
}

const char* TFE_OpGetDevice(const TFE_Op* op, TF_Status* status) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_50(mht_50_v, 974, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpGetDevice");

  return tensorflow::unwrap(op)->DeviceName().c_str();
}

void TFE_OpAddInput(TFE_Op* op, TFE_TensorHandle* input, TF_Status* status) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_51(mht_51_v, 981, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpAddInput");

  status->status = tensorflow::unwrap(op)->AddInput(tensorflow::unwrap(input));
}

void TFE_OpAddInputList(TFE_Op* op, TFE_TensorHandle** inputs, int num_inputs,
                        TF_Status* status) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_52(mht_52_v, 989, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpAddInputList");

  status->status = tensorflow::unwrap(op)->AddInputList(
      {reinterpret_cast<tensorflow::AbstractTensorHandle**>(
           tensorflow::unwrap(inputs)),
       static_cast<size_t>(num_inputs)});
}

extern int TFE_OpGetFlatInputCount(const TFE_Op* op, TF_Status* status) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_53(mht_53_v, 999, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpGetFlatInputCount");

  return tensorflow::unwrap(op)->GetInputs().size();
}

extern TFE_TensorHandle* TFE_OpGetFlatInput(const TFE_Op* op, int index,
                                            TF_Status* status) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_54(mht_54_v, 1007, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpGetFlatInput");

  return tensorflow::wrap(tensorflow::unwrap(op)->GetInputs()[index]);
}

TF_AttrType TFE_OpGetAttrType(TFE_Op* op, const char* attr_name,
                              unsigned char* is_list, TF_Status* status) {
   std::vector<std::string> mht_55_v;
   mht_55_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_55_v.push_back("is_list: \"" + (is_list == nullptr ? std::string("nullptr") : std::string((char*)is_list)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_55(mht_55_v, 1017, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpGetAttrType");

  TF_AttrType ret = TF_ATTR_INT;
  const tensorflow::AttrTypeMap* attr_types_;
  bool is_function;
  status->status = tensorflow::AttrTypeMapForOp(
      tensorflow::unwrap(op)->Name().c_str(), &attr_types_, &is_function);
  if (!status->status.ok()) {
    return ret;
  }
  status->status =
      tensorflow::AttrTypeByName(*attr_types_, attr_name, &ret, is_list);
  return ret;
}

TF_AttrType TFE_OpNameGetAttrType(TFE_Context* ctx,
                                  const char* op_or_function_name,
                                  const char* attr_name, unsigned char* is_list,
                                  TF_Status* status) {
   std::vector<std::string> mht_56_v;
   mht_56_v.push_back("op_or_function_name: \"" + (op_or_function_name == nullptr ? std::string("nullptr") : std::string((char*)op_or_function_name)) + "\"");
   mht_56_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_56_v.push_back("is_list: \"" + (is_list == nullptr ? std::string("nullptr") : std::string((char*)is_list)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_56(mht_56_v, 1040, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpNameGetAttrType");

  TF_AttrType ret;
  TFE_Op* op = TFE_NewOp(ctx, op_or_function_name, status);
  if (status->status.ok()) {
    ret = TFE_OpGetAttrType(op, attr_name, is_list, status);
  } else {
    ret = TF_ATTR_INT;  // Same dummy return as TFE_OpGetAttrType.
  }
  TFE_DeleteOp(op);
  return ret;
}

void TFE_OpSetAttrString(TFE_Op* op, const char* attr_name, const void* value,
                         size_t length) {
   std::vector<std::string> mht_57_v;
   mht_57_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_57(mht_57_v, 1057, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpSetAttrString");

  auto s = tensorflow::unwrap(op)->SetAttrString(
      attr_name, static_cast<const char*>(value), length);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrInt(TFE_Op* op, const char* attr_name, int64_t value) {
   std::vector<std::string> mht_58_v;
   mht_58_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_58(mht_58_v, 1069, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpSetAttrInt");

  auto s = tensorflow::unwrap(op)->SetAttrInt(attr_name, value);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrFloat(TFE_Op* op, const char* attr_name, float value) {
   std::vector<std::string> mht_59_v;
   mht_59_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_59(mht_59_v, 1080, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpSetAttrFloat");

  auto s = tensorflow::unwrap(op)->SetAttrFloat(attr_name, value);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrBool(TFE_Op* op, const char* attr_name, unsigned char value) {
   std::vector<std::string> mht_60_v;
   mht_60_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_60_v.push_back("value: '" + std::string(1, value) + "'");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_60(mht_60_v, 1092, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpSetAttrBool");

  auto s = tensorflow::unwrap(op)->SetAttrBool(attr_name,
                                               (value == 0) ? false : true);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrType(TFE_Op* op, const char* attr_name, TF_DataType value) {
   std::vector<std::string> mht_61_v;
   mht_61_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_61(mht_61_v, 1104, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpSetAttrType");

  auto s = tensorflow::unwrap(op)->SetAttrType(
      attr_name, static_cast<tensorflow::DataType>(value));
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrShape(TFE_Op* op, const char* attr_name, const int64_t* dims,
                        const int num_dims, TF_Status* out_status) {
   std::vector<std::string> mht_62_v;
   mht_62_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_62(mht_62_v, 1117, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpSetAttrShape");

  out_status->status =
      tensorflow::unwrap(op)->SetAttrShape(attr_name, dims, num_dims);
}

void TFE_OpSetAttrFunction(TFE_Op* op, const char* attr_name,
                           const TFE_Op* value) {
   std::vector<std::string> mht_63_v;
   mht_63_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_63(mht_63_v, 1127, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpSetAttrFunction");

  auto s = tensorflow::unwrap(op)->SetAttrFunction(
      attr_name, tensorflow::unwrap(const_cast<TFE_Op*>(value)));
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrFunctionName(TFE_Op* op, const char* attr_name,
                               const char* data, size_t length) {
   std::vector<std::string> mht_64_v;
   mht_64_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_64_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_64(mht_64_v, 1141, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpSetAttrFunctionName");

  auto s = tensorflow::unwrap(op)->SetAttrFunctionName(attr_name, data, length);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrTensor(TFE_Op* op, const char* attr_name, TF_Tensor* tensor,
                         TF_Status* status) {
   std::vector<std::string> mht_65_v;
   mht_65_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_65(mht_65_v, 1153, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpSetAttrTensor");

  tensorflow::Tensor t;
  status->status = TF_TensorToTensor(tensor, &t);
  tensorflow::TensorInterface interface(t);
  status->status = tensorflow::unwrap(op)->SetAttrTensor(attr_name, &interface);
}

void TFE_OpSetAttrStringList(TFE_Op* op, const char* attr_name,
                             const void* const* values, const size_t* lengths,
                             int num_values) {
   std::vector<std::string> mht_66_v;
   mht_66_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_66(mht_66_v, 1166, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpSetAttrStringList");

  auto s = tensorflow::unwrap(op)->SetAttrStringList(attr_name, values, lengths,
                                                     num_values);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrFloatList(TFE_Op* op, const char* attr_name,
                            const float* values, int num_values) {
   std::vector<std::string> mht_67_v;
   mht_67_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_67(mht_67_v, 1179, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpSetAttrFloatList");

  auto s =
      tensorflow::unwrap(op)->SetAttrFloatList(attr_name, values, num_values);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrIntList(TFE_Op* op, const char* attr_name,
                          const int64_t* values, int num_values) {
   std::vector<std::string> mht_68_v;
   mht_68_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_68(mht_68_v, 1192, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpSetAttrIntList");

  auto s =
      tensorflow::unwrap(op)->SetAttrIntList(attr_name, values, num_values);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrTypeList(TFE_Op* op, const char* attr_name,
                           const TF_DataType* values, int num_values) {
   std::vector<std::string> mht_69_v;
   mht_69_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_69(mht_69_v, 1205, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpSetAttrTypeList");

  auto s = tensorflow::unwrap(op)->SetAttrTypeList(
      attr_name, reinterpret_cast<const tensorflow::DataType*>(values),
      num_values);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrBoolList(TFE_Op* op, const char* attr_name,
                           const unsigned char* values, int num_values) {
   std::vector<std::string> mht_70_v;
   mht_70_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_70_v.push_back("values: \"" + (values == nullptr ? std::string("nullptr") : std::string((char*)values)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_70(mht_70_v, 1220, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpSetAttrBoolList");

  auto s =
      tensorflow::unwrap(op)->SetAttrBoolList(attr_name, values, num_values);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrShapeList(TFE_Op* op, const char* attr_name,
                            const int64_t** dims, const int* num_dims,
                            int num_values, TF_Status* out_status) {
   std::vector<std::string> mht_71_v;
   mht_71_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_71(mht_71_v, 1234, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpSetAttrShapeList");

  out_status->status = tensorflow::unwrap(op)->SetAttrShapeList(
      attr_name, dims, num_dims, num_values);
}

void TFE_OpSetAttrFunctionList(TFE_Op* op, const char* attr_name,
                               const TFE_Op** value, int num_values) {
   std::vector<std::string> mht_72_v;
   mht_72_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_72(mht_72_v, 1244, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpSetAttrFunctionList");

  auto s = tensorflow::unwrap(op)->SetAttrFunctionList(
      attr_name, {reinterpret_cast<const tensorflow::AbstractOperation**>(
                      tensorflow::unwrap(value)),
                  static_cast<size_t>(num_values)});
  if (!s.ok()) {
    LOG(WARNING) << "Unable to set attribute: " << attr_name;
  }
}

void TFE_OpSetAttrValueProto(const TFE_Op* op, const char* attr_name,
                             const void* proto, size_t proto_len,
                             TF_Status* status) {
   std::vector<std::string> mht_73_v;
   mht_73_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_73(mht_73_v, 1260, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpSetAttrValueProto");

  tensorflow::AttrValue attr_value;
  if (!attr_value.ParseFromArray(proto, proto_len)) {
    status->status =
        tensorflow::errors::InvalidArgument("Unparseable AttrValue proto");
    return;
  }
  if (op == nullptr) {
    status->status = tensorflow::errors::InvalidArgument(
        "Got a null or uninitialized `op` argument");
    return;
  }
  tensorflow::EagerOperation* operation =
      OperationFromInterface(tensorflow::unwrap(const_cast<TFE_Op*>(op)));
  operation->MutableAttrs()->Set(attr_name, attr_value);
}

TF_CAPI_EXPORT extern int TFE_OpGetInputLength(TFE_Op* op,
                                               const char* input_name,
                                               TF_Status* status) {
   std::vector<std::string> mht_74_v;
   mht_74_v.push_back("input_name: \"" + (input_name == nullptr ? std::string("nullptr") : std::string((char*)input_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_74(mht_74_v, 1283, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpGetInputLength");

  int ret = -1;
  status->status = tensorflow::unwrap(op)->InputLength(input_name, &ret);
  return ret;
}

TF_CAPI_EXPORT extern int TFE_OpGetOutputLength(TFE_Op* op,
                                                const char* output_name,
                                                TF_Status* status) {
   std::vector<std::string> mht_75_v;
   mht_75_v.push_back("output_name: \"" + (output_name == nullptr ? std::string("nullptr") : std::string((char*)output_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_75(mht_75_v, 1295, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpGetOutputLength");

  int ret = -1;
  status->status = tensorflow::unwrap(op)->OutputLength(output_name, &ret);
  return ret;
}

void TFE_Execute(TFE_Op* op, TFE_TensorHandle** retvals, int* num_retvals,
                 TF_Status* status) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_76(mht_76_v, 1305, "", "./tensorflow/c/eager/c_api.cc", "TFE_Execute");

  tensorflow::ImmediateExecutionOperation* unwrapped_op =
      tensorflow::unwrap(op);

  status->status =
      unwrapped_op->GetContext()->GetCustomDeviceOpHandler().Execute(
          unwrapped_op,
          reinterpret_cast<tensorflow::ImmediateExecutionTensorHandle**>(
              retvals),
          num_retvals);
}

TFE_TensorHandle* TFE_TensorHandleCopyToDevice(TFE_TensorHandle* h,
                                               TFE_Context* ctx,
                                               const char* device_name,
                                               TF_Status* status) {
   std::vector<std::string> mht_77_v;
   mht_77_v.push_back("device_name: \"" + (device_name == nullptr ? std::string("nullptr") : std::string((char*)device_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_77(mht_77_v, 1324, "", "./tensorflow/c/eager/c_api.cc", "TFE_TensorHandleCopyToDevice");

  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }

  tensorflow::ImmediateExecutionContext* unwrapped_ctx =
      tensorflow::unwrap(ctx);

  auto* result =
      unwrapped_ctx->GetCustomDeviceOpHandler().CopyTensorHandleToDevice(
          unwrapped_ctx, tensorflow::unwrap(h), device_name, &status->status);

  if (status->status.ok()) {
    return tensorflow::wrap(result);
  }
  return nullptr;
}

void TFE_ContextAddFunctionDef(TFE_Context* ctx,
                               const char* serialized_function_def, size_t size,
                               TF_Status* status) {
   std::vector<std::string> mht_78_v;
   mht_78_v.push_back("serialized_function_def: \"" + (serialized_function_def == nullptr ? std::string("nullptr") : std::string((char*)serialized_function_def)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_78(mht_78_v, 1349, "", "./tensorflow/c/eager/c_api.cc", "TFE_ContextAddFunctionDef");

  tensorflow::FunctionDef function_def;
  if (!function_def.ParseFromArray(serialized_function_def, size)) {
    status->status =
        tensorflow::errors::InvalidArgument("Invalid FunctionDef proto");
    return;
  }

  AnnotateEagerRuntimeConstructionContext(function_def);
  status->status = tensorflow::unwrap(ctx)->AddFunctionDef(function_def);
}

void TFE_ContextAddFunction(TFE_Context* ctx, TF_Function* function,
                            TF_Status* status) {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_79(mht_79_v, 1365, "", "./tensorflow/c/eager/c_api.cc", "TFE_ContextAddFunction");

  AnnotateEagerRuntimeConstructionContext(function->fdef);
  status->status = tensorflow::unwrap(ctx)->AddFunctionDefWithStackTraces(
      function->fdef, function->stack_traces);
}

void TFE_ContextRemoveFunction(TFE_Context* ctx, const char* name,
                               TF_Status* status) {
   std::vector<std::string> mht_80_v;
   mht_80_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_80(mht_80_v, 1376, "", "./tensorflow/c/eager/c_api.cc", "TFE_ContextRemoveFunction");

  status->status = tensorflow::unwrap(ctx)->RemoveFunction(name);
}

unsigned char TFE_ContextHasFunction(TFE_Context* ctx, const char* name) {
   std::vector<std::string> mht_81_v;
   mht_81_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_81(mht_81_v, 1384, "", "./tensorflow/c/eager/c_api.cc", "TFE_ContextHasFunction");

  return tensorflow::unwrap(ctx)->FindFunctionDef(name) != nullptr;
}

void TFE_ContextEnableRunMetadata(TFE_Context* ctx) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_82(mht_82_v, 1391, "", "./tensorflow/c/eager/c_api.cc", "TFE_ContextEnableRunMetadata");

  tensorflow::unwrap(ctx)->SetShouldStoreGraphs(true);
}

void TFE_ContextDisableRunMetadata(TFE_Context* ctx) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_83(mht_83_v, 1398, "", "./tensorflow/c/eager/c_api.cc", "TFE_ContextDisableRunMetadata");

  tensorflow::unwrap(ctx)->SetShouldStoreGraphs(false);
}

}  // extern "C"

TFE_TensorHandle* TFE_NewTensorHandle(const tensorflow::Tensor& t,
                                      TF_Status* status) {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_84(mht_84_v, 1408, "", "./tensorflow/c/eager/c_api.cc", "TFE_NewTensorHandle");

  return tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(t));
}

void TFE_ContextExportRunMetadata(TFE_Context* ctx, TF_Buffer* buf,
                                  TF_Status* status) {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_85(mht_85_v, 1416, "", "./tensorflow/c/eager/c_api.cc", "TFE_ContextExportRunMetadata");

  auto* context = tensorflow::unwrap(ctx);
  status->status = context->AsyncWait();
  if (!status->status.ok()) return;
  auto run_metadata = context->ExportRunMetadata();
  status->status = MessageToBuffer(*run_metadata, buf);
}

namespace {
TFE_Op* GetFunc(TFE_Context* ctx, const tensorflow::NameAttrList& func,
                TF_Status* status) {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_86(mht_86_v, 1429, "", "./tensorflow/c/eager/c_api.cc", "GetFunc");

  TFE_Op* func_op = TFE_NewOp(ctx, func.name().data(), status);
  for (const auto& attr : func.attr()) {
    if (!status->status.ok()) return nullptr;
    SetOpAttrValueScalar(ctx, func_op, attr.second, attr.first.data(), status);
    if (!status->status.ok()) return nullptr;
  }
  return func_op;
}
}  // namespace

void TFE_ContextStartStep(TFE_Context* ctx) {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_87(mht_87_v, 1443, "", "./tensorflow/c/eager/c_api.cc", "TFE_ContextStartStep");

  tensorflow::unwrap(ctx)->StartStep();
}

void TFE_ContextEndStep(TFE_Context* ctx) {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_88(mht_88_v, 1450, "", "./tensorflow/c/eager/c_api.cc", "TFE_ContextEndStep");

  tensorflow::unwrap(ctx)->EndStep();
}

const TFE_OpAttrs* TFE_OpGetAttrs(const TFE_Op* op) {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_89(mht_89_v, 1457, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpGetAttrs");

  return tensorflow::wrap(tensorflow::unwrap(op)->GetOpAttrs());
}

void TFE_OpAddAttrs(TFE_Op* op, const TFE_OpAttrs* attrs) {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_90(mht_90_v, 1464, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpAddAttrs");

  tensorflow::unwrap(op)->AddAttrs(tensorflow::unwrap(attrs));
}

void TFE_OpAttrsSerialize(const TFE_OpAttrs* attrs, TF_Buffer* buf,
                          TF_Status* status) {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_91(mht_91_v, 1472, "", "./tensorflow/c/eager/c_api.cc", "TFE_OpAttrsSerialize");

  tensorflow::NameAttrList name_and_attrs;
  tensorflow::unwrap(attrs)->GetNameAttrList(&name_and_attrs);
  status->status = MessageToBuffer(name_and_attrs, buf);
}

namespace tensorflow {
void SetOpAttrValueScalar(TFE_Context* ctx, TFE_Op* op,
                          const tensorflow::AttrValue& default_value,
                          const char* attr_name, TF_Status* status) {
   std::vector<std::string> mht_92_v;
   mht_92_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_92(mht_92_v, 1485, "", "./tensorflow/c/eager/c_api.cc", "SetOpAttrValueScalar");

  switch (default_value.value_case()) {
    case tensorflow::AttrValue::kS: {
      const string& v = default_value.s();
      TFE_OpSetAttrString(op, attr_name, v.data(), v.size());
      break;
    }
    case tensorflow::AttrValue::kI:
      TFE_OpSetAttrInt(op, attr_name, static_cast<int64_t>(default_value.i()));
      break;
    case tensorflow::AttrValue::kF:
      TFE_OpSetAttrFloat(op, attr_name, default_value.f());
      break;
    case tensorflow::AttrValue::kB:
      TFE_OpSetAttrBool(op, attr_name, default_value.b());
      break;
    case tensorflow::AttrValue::kType:
      TFE_OpSetAttrType(op, attr_name,
                        static_cast<TF_DataType>(default_value.type()));
      break;
    case tensorflow::AttrValue::kShape: {
      const auto& tensor_shape = default_value.shape();
      if (tensor_shape.unknown_rank()) {
        TFE_OpSetAttrShape(op, attr_name, nullptr, -1, status);
      } else {
        const auto num_dims = tensor_shape.dim_size();
        std::unique_ptr<int64_t[]> dims(new int64_t[num_dims]);
        for (int i = 0; i < num_dims; ++i) {
          dims[i] = tensor_shape.dim(i).size();
        }
        TFE_OpSetAttrShape(op, attr_name, dims.get(), num_dims, status);
      }
    } break;
    case tensorflow::AttrValue::kFunc: {
      const auto func_op = GetFunc(ctx, default_value.func(), status);
      if (!status->status.ok()) return;
      // TODO(nareshmodi): TFE_OpSetAttrFunction and TFE_OpSetAttrFunctionList
      // require TFE_Op* and just convert it internally a NameAttrValue, so
      // consider adding an overload to the C API to make this case easier.
      TFE_OpSetAttrFunction(op, attr_name, func_op);
      TFE_DeleteOp(func_op);
    } break;
    case tensorflow::AttrValue::kList: {
      // String
      if (const int s_size = default_value.list().s_size()) {
        absl::InlinedVector<const void*, 4> values_vector;
        values_vector.reserve(s_size);
        absl::InlinedVector<size_t, 4> lengths_vector;
        lengths_vector.reserve(s_size);
        for (int i = 0; i < s_size; ++i) {
          const string& v = default_value.list().s(i);
          values_vector.push_back(v.data());
          lengths_vector.push_back(v.size());
        }
        TFE_OpSetAttrStringList(op, attr_name, values_vector.data(),
                                lengths_vector.data(), s_size);
      }

      // Int
      if (const int i_size = default_value.list().i_size()) {
        absl::InlinedVector<int64_t, 4> i_vector;
        i_vector.reserve(i_size);
        for (int i = 0; i < i_size; ++i) {
          i_vector.push_back(default_value.list().i(i));
        }
        TFE_OpSetAttrIntList(op, attr_name, i_vector.data(), i_size);
      }
      // Float
      if (const int f_size = default_value.list().f_size()) {
        absl::InlinedVector<float, 4> f_vector;
        f_vector.reserve(f_size);
        for (int i = 0; i < f_size; ++i) {
          f_vector.push_back(default_value.list().f(i));
        }
        TFE_OpSetAttrFloatList(op, attr_name, f_vector.data(), f_size);
      }
      // Bool
      if (const int b_size = default_value.list().b_size()) {
        absl::InlinedVector<unsigned char, 4> b_vector;
        b_vector.reserve(b_size);
        for (int i = 0; i < b_size; i++) {
          b_vector.push_back(default_value.list().b(i));
        }
        TFE_OpSetAttrBoolList(op, attr_name, b_vector.data(), b_size);
      }
      // Type
      if (const int type_size = default_value.list().type_size()) {
        absl::InlinedVector<unsigned int, 4> type_vector;
        type_vector.reserve(type_size);
        for (int i = 0; i < type_size; ++i) {
          type_vector.push_back(default_value.list().type(i));
        }
        TFE_OpSetAttrTypeList(
            op, attr_name,
            reinterpret_cast<const TF_DataType*>(type_vector.data()),
            type_size);
      }

      // Rest are not supported.
      if (default_value.list().shape_size() > 0 ||
          default_value.list().func_size() > 0 ||
          default_value.list().tensor_size() > 0) {
        TF_SetStatus(
            status, TF_UNIMPLEMENTED,
            tensorflow::strings::StrCat("Unable to get setfor default value: ",
                                        default_value.DebugString())
                .data());
      }
    } break;
    case tensorflow::AttrValue::kTensor:
      TF_FALLTHROUGH_INTENDED;
    case tensorflow::AttrValue::kPlaceholder:
      TF_FALLTHROUGH_INTENDED;
    case tensorflow::AttrValue::VALUE_NOT_SET:
      TF_SetStatus(
          status, TF_UNIMPLEMENTED,
          tensorflow::strings::StrCat("Unable to get setfor default value: ",
                                      default_value.DebugString())
              .data());
  }
}
}  // namespace tensorflow

namespace {
TFE_TensorHandle* DefaultCustomDevicePack(TFE_Context* context,
                                          TFE_TensorHandle** handles,
                                          int num_handles, TF_Status* status,
                                          void* device_info) {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_93(mht_93_v, 1615, "", "./tensorflow/c/eager/c_api.cc", "DefaultCustomDevicePack");

  TF_SetStatus(status, TF_UNIMPLEMENTED,
               "This custom device does not support packing tensors.");
  return nullptr;
}
}  // namespace

extern "C" {

void TFE_RegisterCustomDevice(TFE_Context* ctx, TFE_CustomDevice device,
                              const char* device_name, void* device_info,
                              TF_Status* status) {
   std::vector<std::string> mht_94_v;
   mht_94_v.push_back("device_name: \"" + (device_name == nullptr ? std::string("nullptr") : std::string((char*)device_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_apiDTcc mht_94(mht_94_v, 1630, "", "./tensorflow/c/eager/c_api.cc", "TFE_RegisterCustomDevice");

  // Fill in default values for optional functionality.
  if (device.pack == nullptr) {
    device.pack = &DefaultCustomDevicePack;
  }
  auto custom_device = std::make_unique<tensorflow::CustomDeviceAPI>(
      ctx, device, device_info, device_name);
  status->status = tensorflow::unwrap(ctx)->RegisterCustomDevice(
      device_name, std::move(custom_device));
}

}  // extern "C"
