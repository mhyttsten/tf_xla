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
class MHTracer_DTPStensorflowPScorePSkernelsPSfunction_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSfunction_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSfunction_opsDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/function_ops.h"

#include <deque>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/gradients.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

static constexpr const char* const kGradientOp =
    FunctionLibraryDefinition::kGradientOp;

ArgOp::ArgOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfunction_opsDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/function_ops.cc", "ArgOp::ArgOp");

  OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("index", &index_));
}

void ArgOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfunction_opsDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/kernels/function_ops.cc", "ArgOp::Compute");

  auto frame = ctx->call_frame();
  OP_REQUIRES(ctx, frame != nullptr, errors::Internal("no call frame"));
  const Tensor* val;

  auto validate_type = [this](const Tensor& val) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfunction_opsDTcc mht_2(mht_2_v, 226, "", "./tensorflow/core/kernels/function_ops.cc", "lambda");

    if (val.dtype() == dtype_) {
      return Status::OK();
    } else {
      return errors::InvalidArgument("Type mismatch: actual ",
                                     DataTypeString(val.dtype()),
                                     " vs. expect ", DataTypeString(dtype_));
    }
  };

  if (frame->CanConsumeArg(index_)) {
    Tensor val;
    frame->ConsumeArg(index_, &val);
    OP_REQUIRES_OK(ctx, validate_type(val));
    ctx->set_output(0, std::move(val));
  } else {
    OP_REQUIRES_OK(ctx, frame->GetArg(index_, &val));
    OP_REQUIRES_OK(ctx, validate_type(*val));
    ctx->set_output(0, *val);
  }
}

RetvalOp::RetvalOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfunction_opsDTcc mht_3(mht_3_v, 251, "", "./tensorflow/core/kernels/function_ops.cc", "RetvalOp::RetvalOp");

  OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("index", &index_));
}

void RetvalOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfunction_opsDTcc mht_4(mht_4_v, 259, "", "./tensorflow/core/kernels/function_ops.cc", "RetvalOp::Compute");

  const Tensor& val = ctx->input(0);
  OP_REQUIRES(ctx, val.dtype() == dtype_,
              errors::InvalidArgument("Type mismatch: actual ",
                                      DataTypeString(val.dtype()),
                                      " vs. expect ", DataTypeString(dtype_)));
  auto frame = ctx->call_frame();
  OP_REQUIRES(ctx, frame != nullptr, errors::Internal("no call frame"));
  OP_REQUIRES_OK(ctx, frame->SetRetval(index_, val));
}

REGISTER_SYSTEM_KERNEL_BUILDER(Name(kArgOp).Device(DEVICE_CPU), ArgOp);
REGISTER_SYSTEM_KERNEL_BUILDER(Name(kDeviceArgOp).Device(DEVICE_CPU), ArgOp);
REGISTER_SYSTEM_KERNEL_BUILDER(Name(kRetOp).Device(DEVICE_CPU), RetvalOp);
REGISTER_SYSTEM_KERNEL_BUILDER(Name(kDeviceRetOp).Device(DEVICE_CPU), RetvalOp);

// TPU ops are only registered when they are required as part of the larger
// TPU runtime, and does not need to be registered when selective registration
// is turned on.
REGISTER_KERNEL_BUILDER(Name(kRetOp).Device(DEVICE_TPU_SYSTEM), RetvalOp);

#define REGISTER(type)     \
  REGISTER_KERNEL_BUILDER( \
      Name(kArgOp).Device(DEVICE_DEFAULT).TypeConstraint<type>("T"), ArgOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER);
TF_CALL_QUANTIZED_TYPES(REGISTER);
TF_CALL_bool(REGISTER);

REGISTER_KERNEL_BUILDER(
    Name(kDeviceArgOp).Device(DEVICE_DEFAULT).TypeConstraint<int32>("T"),
    ArgOp);

REGISTER_KERNEL_BUILDER(Name(kArgOp)
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        ArgOp);
#undef REGISTER

REGISTER_KERNEL_BUILDER(Name(kArgOp)
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("output")
                            .TypeConstraint<ResourceHandle>("T"),
                        ArgOp);

REGISTER_KERNEL_BUILDER(Name(kArgOp)
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("output")
                            .TypeConstraint<tstring>("T"),
                        ArgOp);

REGISTER_KERNEL_BUILDER(
    Name(kArgOp).Device(DEVICE_DEFAULT).TypeConstraint<Variant>("T"), ArgOp);

#define REGISTER(type)                                               \
  REGISTER_KERNEL_BUILDER(                                           \
      Name(kRetOp).Device(DEVICE_DEFAULT).TypeConstraint<type>("T"), \
      RetvalOp);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER);
TF_CALL_QUANTIZED_TYPES(REGISTER);
TF_CALL_qint16(REGISTER);
TF_CALL_quint16(REGISTER);
REGISTER(Variant);
TF_CALL_bool(REGISTER);

REGISTER_KERNEL_BUILDER(Name(kRetOp)
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("input")
                            .TypeConstraint<int32>("T"),
                        RetvalOp);
REGISTER_KERNEL_BUILDER(
    Name(kDeviceRetOp).Device(DEVICE_DEFAULT).TypeConstraint<int32>("T"),
    RetvalOp);

REGISTER_KERNEL_BUILDER(Name(kRetOp)
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<ResourceHandle>("T")
                            .HostMemory("input"),
                        RetvalOp);

REGISTER_KERNEL_BUILDER(Name(kRetOp)
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<tstring>("T")
                            .HostMemory("input"),
                        RetvalOp);

#undef REGISTER

class PassOn : public OpKernel {
 public:
  explicit PassOn(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfunction_opsDTcc mht_5(mht_5_v, 353, "", "./tensorflow/core/kernels/function_ops.cc", "PassOn");

    OP_REQUIRES(ctx, ctx->num_inputs() == ctx->num_outputs(),
                errors::Internal("#inputs != #outputs : ", ctx->num_inputs(),
                                 " vs. ", ctx->num_outputs()));
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      OP_REQUIRES(
          ctx, input_type(i) == output_type(i),
          errors::Internal("Input and output types for position ", i,
                           " do not match: ", DataTypeString(input_type(i)),
                           " vs. ", DataTypeString(output_type(i))));
    }
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfunction_opsDTcc mht_6(mht_6_v, 369, "", "./tensorflow/core/kernels/function_ops.cc", "Compute");

    for (int i = 0; i < ctx->num_inputs(); ++i) {
      ctx->set_output(i, ctx->input(i));
    }
  }
};

REGISTER_SYSTEM_KERNEL_BUILDER(Name("_ListToArray").Device(DEVICE_CPU), PassOn);
REGISTER_SYSTEM_KERNEL_BUILDER(Name("_ArrayToList").Device(DEVICE_CPU), PassOn);

#define REGISTER_DEFAULT_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_ListToArray").Device(DEVICE_DEFAULT).TypeConstraint<type>("T"), \
      PassOn);                                                               \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("_ArrayToList").Device(DEVICE_DEFAULT).TypeConstraint<type>("T"), \
      PassOn);

REGISTER_DEFAULT_KERNELS(Eigen::half);
REGISTER_DEFAULT_KERNELS(float);
REGISTER_DEFAULT_KERNELS(double);

#undef REGISTER_DEFAULT_KERNELS

REGISTER_KERNEL_BUILDER(Name("_ListToArray")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        PassOn);
REGISTER_KERNEL_BUILDER(Name("_ArrayToList")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        PassOn);

class SymbolicGradientOp : public AsyncOpKernel {
 public:
  explicit SymbolicGradientOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfunction_opsDTcc mht_7(mht_7_v, 411, "", "./tensorflow/core/kernels/function_ops.cc", "SymbolicGradientOp");
}

  ~SymbolicGradientOp() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfunction_opsDTcc mht_8(mht_8_v, 416, "", "./tensorflow/core/kernels/function_ops.cc", "~SymbolicGradientOp");
}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfunction_opsDTcc mht_9(mht_9_v, 421, "", "./tensorflow/core/kernels/function_ops.cc", "ComputeAsync");

    FunctionLibraryRuntime* lib = ctx->function_library();
    OP_REQUIRES_ASYNC(ctx, lib != nullptr,
                      errors::Internal("No function library is provided."),
                      done);

    FunctionLibraryRuntime::Handle handle;
    OP_REQUIRES_OK_ASYNC(
        ctx, lib->Instantiate(kGradientOp, AttrSlice(def()), &handle), done);

    FunctionLibraryRuntime::Options opts;
    opts.rendezvous = ctx->rendezvous();
    opts.cancellation_manager = ctx->cancellation_manager();
    opts.collective_executor = ctx->collective_executor();
    opts.runner = ctx->runner();
    opts.run_all_kernels_inline = ctx->run_all_kernels_inline();
    opts.stats_collector = ctx->stats_collector();
    opts.step_container = ctx->step_container();
    std::vector<Tensor> args;
    args.reserve(ctx->num_inputs());
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      args.push_back(ctx->input(i));
    }
    std::vector<Tensor>* rets = new std::vector<Tensor>;
    profiler::TraceMe trace_me("SymbolicGradientOp");
    lib->Run(opts, handle, args, rets, [ctx, done, rets](const Status& status) {
      if (!status.ok()) {
        ctx->SetStatus(status);
      } else if (rets->size() != ctx->num_outputs()) {
        ctx->SetStatus(errors::InvalidArgument(
            "SymGrad expects to return ", ctx->num_outputs(),
            " tensor(s), but get ", rets->size(), " tensor(s) instead."));
      } else {
        for (size_t i = 0; i < rets->size(); ++i) {
          ctx->set_output(i, std::move((*rets)[i]));
        }
      }
      delete rets;
      done();
    });
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(SymbolicGradientOp);
};

REGISTER_KERNEL_BUILDER(Name(kGradientOp).Device(DEVICE_CPU),
                        SymbolicGradientOp);
REGISTER_KERNEL_BUILDER(Name(kGradientOp).Device(DEVICE_DEFAULT),
                        SymbolicGradientOp);

RemoteCallOp::RemoteCallOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfunction_opsDTcc mht_10(mht_10_v, 475, "", "./tensorflow/core/kernels/function_ops.cc", "RemoteCallOp::RemoteCallOp");

  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr(FunctionLibraryDefinition::kFuncAttr, &func_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Tin", &input_dtypes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &output_dtypes_));
}

void RemoteCallOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfunction_opsDTcc mht_11(mht_11_v, 485, "", "./tensorflow/core/kernels/function_ops.cc", "RemoteCallOp::ComputeAsync");

  FunctionLibraryRuntime* lib = ctx->function_library();
  OP_REQUIRES_ASYNC(ctx, lib != nullptr,
                    errors::Internal("No function library is provided."), done);

  const string& source_device = lib->device()->name();
  const Tensor* target;
  OP_REQUIRES_OK_ASYNC(ctx, ctx->input("target", &target), done);

  FunctionTarget function_target;
  OP_REQUIRES_OK_ASYNC(
      ctx,
      DeviceNameUtils::CanonicalizeDeviceName(
          target->scalar<tstring>()(), source_device, &function_target.first),
      done);
  function_target.second = lib;

  const string& target_device = function_target.first;
  const string& func_name = func_.name();

  FunctionLibraryRuntime::Handle handle;
  {
    mutex_lock l(mu_);
    auto cached_entry = handle_cache_.find(function_target);
    if (cached_entry != handle_cache_.end()) {
      handle = cached_entry->second;
    } else {
      VLOG(1) << "Instantiating " << func_name << " on " << target_device;
      profiler::TraceMe activity(
          [&] {
            return strings::StrCat("RemoteCall: Instantiate: ", func_name,
                                   " on ", target_device);
          },
          profiler::TraceMeLevel::kInfo);
      FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
      const auto* config = (ctx->function_library())
                               ? ctx->function_library()->config_proto()
                               : nullptr;
      if (config) {
        instantiate_opts.config_proto = *config;
      }
      instantiate_opts.target = target_device;
      OP_REQUIRES_OK_ASYNC(ctx,
                           lib->Instantiate(func_name, AttrSlice(&func_.attr()),
                                            instantiate_opts, &handle),
                           done);
      auto insert_result = handle_cache_.insert({function_target, handle});
      CHECK(insert_result.second) << "Insert unsuccessful.";
      VLOG(1) << "Instantiated " << func_name << " on " << target_device
              << ", resulting in handle: " << handle << " flr: " << lib;
    }
  }

  OpInputList arguments;
  OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("args", &arguments), done);

  FunctionLibraryRuntime::Options opts;
  opts.runner = nullptr;  // Use default runner at remote device.
  opts.run_all_kernels_inline = ctx->run_all_kernels_inline();
  opts.source_device = source_device;
  if (opts.source_device != target_device) {
    opts.remote_execution = true;
  }
  opts.create_rendezvous = true;
  CancellationManager* cancel_mgr = nullptr;
  if (ctx->cancellation_manager() != nullptr) {
    cancel_mgr = new CancellationManager(ctx->cancellation_manager());
  }
  opts.cancellation_manager = cancel_mgr;
  opts.collective_executor = ctx->collective_executor();
  std::vector<Tensor> args(arguments.begin(), arguments.end());
  opts.args_alloc_attrs.reserve(input_dtypes_.size());
  for (const auto& dtype : input_dtypes_) {
    AllocatorAttributes arg_alloc_attrs;
    arg_alloc_attrs.set_on_host(DataTypeAlwaysOnHost(dtype));
    opts.args_alloc_attrs.push_back(arg_alloc_attrs);
  }
  opts.rets_alloc_attrs.reserve(output_dtypes_.size());
  for (const auto& dtype : output_dtypes_) {
    AllocatorAttributes ret_alloc_attrs;
    ret_alloc_attrs.set_on_host(DataTypeAlwaysOnHost(dtype));
    opts.rets_alloc_attrs.push_back(ret_alloc_attrs);
  }
  auto* rets = new std::vector<Tensor>;
  VLOG(1) << "Running " << func_name << " on " << target_device
          << " with handle: " << handle;
  profiler::TraceMe trace_me(
      [&] {
        return profiler::TraceMeEncode(
            "RemoteCallOp",
            {{"func_name", func_name}, {"device", target_device}});
      },
      profiler::TraceMeLevel::kInfo);
  lib->Run(
      opts, handle, args, rets,
      [rets, done = std::move(done), func_name, ctx, cancel_mgr,
       target_device = std::move(function_target.first)](const Status& status) {
        profiler::TraceMe activity(
            [&] {
              return profiler::TraceMeEncode(
                  "RemoteCallOpDone",
                  {{"func_name", func_name}, {"device", target_device}});
            },
            profiler::TraceMeLevel::kInfo);
        if (!status.ok()) {
          ctx->SetStatus(status);
        } else {
          for (size_t i = 0; i < rets->size(); ++i) {
            ctx->set_output(i, std::move((*rets)[i]));
          }
        }
        delete cancel_mgr;
        delete rets;
        done();
      });
}

string RemoteCallOp::TraceString(const OpKernelContext& ctx,
                                 bool verbose) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfunction_opsDTcc mht_12(mht_12_v, 606, "", "./tensorflow/core/kernels/function_ops.cc", "RemoteCallOp::TraceString");

  string trace_string = profiler::TraceMeOp(
      strings::StrCat(name_view(), "__", func_.name()), type_string_view());
  if (verbose) {
    string shape = ShapeTraceString(ctx);
    if (!shape.empty()) {
      trace_string =
          profiler::TraceMeEncode(std::move(trace_string), {{"shape", shape}});
    }
  }
  return trace_string;
}

REGISTER_KERNEL_BUILDER(
    Name("RemoteCall").Device(DEVICE_CPU).HostMemory("target"), RemoteCallOp);
REGISTER_KERNEL_BUILDER(
    Name("RemoteCall").Device(DEVICE_DEFAULT).HostMemory("target"),
    RemoteCallOp);
}  // namespace tensorflow
