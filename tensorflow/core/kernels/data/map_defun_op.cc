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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_opDTcc() {
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
#include "tensorflow/core/kernels/data/map_defun_op.h"

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/batch_util.h"
#include "tensorflow/core/util/reffed_status_callback.h"

namespace tensorflow {
namespace data {

/* static */ constexpr const char* const MapDefunOp::kArguments;
/* static */ constexpr const char* const MapDefunOp::kCapturedInputs;
/* static */ constexpr const char* const MapDefunOp::kTarguments;
/* static */ constexpr const char* const MapDefunOp::kTcaptured;
/* static */ constexpr const char* const MapDefunOp::kOutputTypes;
/* static */ constexpr const char* const MapDefunOp::kOutputShapes;
/* static */ constexpr const char* const MapDefunOp::kFunc;
/* static */ constexpr const char* const MapDefunOp::kMaxIntraOpParallelism;

constexpr char kOutput[] = "output";

struct MapDefunOp::ComputeOptions {
  // These vary per MapDefunOp::ComputeAsync call, but must persist until
  // all calls to the function are complete. This struct also encapsulates
  // all the components that need to be passed to each MapFunctionCallFrame.
  OpInputList args;
  const std::vector<TensorShape> arg_shapes;
  OpInputList captured_inputs;
  const int64_t batch_size;
  std::function<void(std::function<void()>)> runner;

  // Output of a compute call
  std::vector<PartialTensorShape> output_shapes TF_GUARDED_BY(mu);
  OpOutputList output TF_GUARDED_BY(mu);
  mutex mu;

  // Create a copy of output_shapes because every `Compute` may expect a
  // different output shape.
  ComputeOptions(OpKernelContext* ctx, OpInputList args,
                 OpInputList captured_inputs,
                 std::vector<TensorShape> arg_shapes, int64_t batch_size,
                 const std::vector<PartialTensorShape>& output_shapes_attr,
                 int max_parallelism)
      : args(args),
        arg_shapes(std::move(arg_shapes)),
        captured_inputs(captured_inputs),
        batch_size(batch_size),
        output_shapes(output_shapes_attr) {
    if (max_parallelism >= 1) {
      runner = RunnerWithMaxParallelism(*ctx->runner(), max_parallelism);
    }
  }
};

class MapDefunOp::MapFunctionCallFrame : public CallFrameInterface {
 public:
  MapFunctionCallFrame(ComputeOptions* compute_opts, OpKernel* kernel,
                       size_t iter)
      : compute_opts_(compute_opts),
        kernel_(kernel),
        iter_(iter),
        sliced_args_(compute_opts_->args.size()) {}

  ~MapFunctionCallFrame() override = default;

  size_t num_args() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_opDTcc mht_0(mht_0_v, 256, "", "./tensorflow/core/kernels/data/map_defun_op.cc", "num_args");

    return compute_opts_->args.size() + compute_opts_->captured_inputs.size();
  }

  size_t num_retvals() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_opDTcc mht_1(mht_1_v, 263, "", "./tensorflow/core/kernels/data/map_defun_op.cc", "num_retvals");

    return static_cast<size_t>(kernel_->num_outputs());
  }

  Status GetArg(int index, const Tensor** val) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_opDTcc mht_2(mht_2_v, 270, "", "./tensorflow/core/kernels/data/map_defun_op.cc", "GetArg");

    if (index < 0 || index >= compute_opts_->args.size() +
                                  compute_opts_->captured_inputs.size()) {
      return errors::InvalidArgument("Mismatch in number of function inputs.");
    }

    if (index >= compute_opts_->args.size()) {
      // The function is calling for a captured input
      *val =
          &compute_opts_->captured_inputs[index - compute_opts_->args.size()];
      return Status::OK();
    }

    // NOTE: If contention on mu_ becomes problematic, we could create a vector
    // of mutexes, each guarding a different element of sliced_args_.
    mutex_lock l(mu_);
    bool result = sliced_args_[index].CopyFrom(
        compute_opts_->args[index].Slice(iter_, iter_ + 1),
        compute_opts_->arg_shapes.at(index));
    if (!result) {
      return errors::Internal("GetArg failed.");
    } else if (!sliced_args_[index].IsAligned()) {
      // Ensure alignment
      sliced_args_[index] = tensor::DeepCopy(sliced_args_[index]);
    }
    *val = &sliced_args_[index];
    return Status::OK();
  }

  Status SetRetval(int index, const Tensor& val) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_opDTcc mht_3(mht_3_v, 302, "", "./tensorflow/core/kernels/data/map_defun_op.cc", "SetRetval");

    if (index < 0 || index >= kernel_->num_outputs()) {
      return errors::InvalidArgument("Mismatch in number of function outputs.");
    }

    if (val.dtype() != kernel_->output_type(index)) {
      return errors::InvalidArgument(
          "Mismatch in function return type and expected output type for "
          "output: ",
          index);
    }
    Tensor* out;
    {  // Locking scope
      mutex_lock l(compute_opts_->mu);
      if (!compute_opts_->output_shapes.at(index).IsCompatibleWith(
              val.shape())) {
        return errors::InvalidArgument(
            "Mismatch in function retval shape, ", val.shape(),
            ", and expected output shape, ",
            compute_opts_->output_shapes.at(index).DebugString(), ".");
      }
      if (!compute_opts_->output_shapes.at(index).IsFullyDefined()) {
        // Given val, we have new information about the output shape at
        // this index. Store the shape and allocate the output accordingly.
        compute_opts_->output_shapes.at(index) = val.shape();

        TensorShape actual_shape = val.shape();
        actual_shape.InsertDim(0, compute_opts_->batch_size);
        TF_RETURN_IF_ERROR(
            compute_opts_->output.allocate(index, actual_shape, &out));
      } else {
        out = (compute_opts_->output)[index];
      }
    }
    return batch_util::CopyElementToSlice(val, out, iter_);
  }

 private:
  ComputeOptions* const compute_opts_;  // Not owned
  const OpKernel* kernel_;
  const size_t iter_;
  mutex mu_;
  std::vector<Tensor> sliced_args_ TF_GUARDED_BY(mu_);
};

MapDefunOp::MapDefunOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_opDTcc mht_4(mht_4_v, 350, "", "./tensorflow/core/kernels/data/map_defun_op.cc", "MapDefunOp::MapDefunOp");

  auto func_lib = ctx->function_library();
  OP_REQUIRES(ctx, func_lib != nullptr,
              errors::Internal("No function library."));
  const NameAttrList* func;
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kFunc, &func));
  OP_REQUIRES_OK(ctx,
                 func_lib->Instantiate(func->name(), AttrSlice(&func->attr()),
                                       &func_handle_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  OP_REQUIRES_OK(
      ctx, ctx->GetAttr(kMaxIntraOpParallelism, &max_intra_op_parallelism_));

  OP_REQUIRES(ctx, ctx->num_inputs() >= 0,
              errors::InvalidArgument("Must have at least one input."));
  OP_REQUIRES(ctx, ctx->num_outputs() >= 0,
              errors::InvalidArgument("Must have at least one output."));
  OP_REQUIRES(ctx, ctx->num_outputs() == output_shapes_.size(),
              errors::InvalidArgument(
                  "Length of output_shapes and output_types must match."));
}

void MapDefunOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_opDTcc mht_5(mht_5_v, 375, "", "./tensorflow/core/kernels/data/map_defun_op.cc", "MapDefunOp::ComputeAsync");

  ComputeOptions* compute_opts = nullptr;

  OP_REQUIRES_OK_ASYNC(ctx, SetupArgs(ctx, &compute_opts), done);

  Status s = SetupOutputs(ctx, compute_opts);
  if (!s.ok()) delete compute_opts;
  OP_REQUIRES_OK_ASYNC(ctx, s, done);

  FunctionLibraryRuntime::Options opts;
  SetRunOptions(ctx, &opts, compute_opts, /*always_collect_stats=*/false);

  // Run loop
  StatusCallback callback = std::bind(
      [](OpKernelContext* ctx, ComputeOptions* compute_opts, DoneCallback& done,
         const Status& status) {
        delete compute_opts;
        ctx->SetStatus(status);
        done();
      },
      ctx, compute_opts, std::move(done), std::placeholders::_1);

  auto* refcounted = new ReffedStatusCallback(std::move(callback));

  CancellationManager* parent_mgr = ctx->cancellation_manager();

  for (size_t i = 0; i < static_cast<size_t>(compute_opts->batch_size); ++i) {
    // We use a different cancellation manager each time the function is run
    // to avoid the race condition between a function run error and other
    // functions being cancelled as a result.
    CancellationManager* c_mgr = new CancellationManager(parent_mgr);
    opts.cancellation_manager = c_mgr;

    auto* call_frame = new MapFunctionCallFrame(compute_opts, this, i);

    refcounted->Ref();
    ctx->function_library()->Run(
        opts, func_handle_, call_frame,
        [call_frame, refcounted, c_mgr](const Status& func_status) {
          delete c_mgr;
          delete call_frame;
          refcounted->UpdateStatus(func_status);
          refcounted->Unref();
        });
  }

  // Unref 1 because refcounted is initialized with refcount = 1
  refcounted->Unref();
}

void MapDefunOp::SetRunOptions(OpKernelContext* ctx,
                               FunctionLibraryRuntime::Options* opts,
                               ComputeOptions* compute_opts,
                               bool always_collect_stats) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_opDTcc mht_6(mht_6_v, 431, "", "./tensorflow/core/kernels/data/map_defun_op.cc", "MapDefunOp::SetRunOptions");

  opts->rendezvous = ctx->rendezvous();
  if (always_collect_stats) {
    opts->stats_collector = ctx->stats_collector();
  }
  if (max_intra_op_parallelism_ >= 1) {
    opts->runner = &compute_opts->runner;
  } else {
    opts->runner = ctx->runner();
  }
  opts->run_all_kernels_inline = ctx->run_all_kernels_inline();
}

Status MapDefunOp::SetupArgs(OpKernelContext* ctx,
                             ComputeOptions** compute_opts) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_opDTcc mht_7(mht_7_v, 448, "", "./tensorflow/core/kernels/data/map_defun_op.cc", "MapDefunOp::SetupArgs");

  OpInputList arguments;
  TF_RETURN_IF_ERROR(ctx->input_list(kArguments, &arguments));
  OpInputList captured_inputs;
  TF_RETURN_IF_ERROR(ctx->input_list(kCapturedInputs, &captured_inputs));

  int64_t batch_size = arguments[0].dims() > 0 ? arguments[0].dim_size(0) : -1;

  for (size_t i = 0; i < arguments.size(); ++i) {
    if (arguments[i].dims() == 0) {
      return errors::InvalidArgument(
          "All inputs must have rank at least 1. Input ", i,
          " has a rank of 0.");
    } else if (arguments[i].dim_size(0) != batch_size) {
      return errors::InvalidArgument(
          "All inputs must have the same dimension 0. Input ", i,
          " has leading dimension ", ctx->input(i).dim_size(0),
          ", while all previous inputs have leading dimension ", batch_size);
    }
  }

  std::vector<TensorShape> arg_shapes;
  arg_shapes.reserve(arguments.size());

  for (size_t i = 0; i < arguments.size(); ++i) {
    arg_shapes.push_back(arguments[i].shape());
    arg_shapes.at(i).RemoveDim(0);
  }

  *compute_opts =
      new ComputeOptions(ctx, arguments, captured_inputs, std::move(arg_shapes),
                         batch_size, output_shapes_, max_intra_op_parallelism_);
  return Status::OK();
}

Status MapDefunOp::SetupOutputs(OpKernelContext* ctx, ComputeOptions* opts) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_defun_opDTcc mht_8(mht_8_v, 486, "", "./tensorflow/core/kernels/data/map_defun_op.cc", "MapDefunOp::SetupOutputs");

  mutex_lock l(opts->mu);
  TF_RETURN_IF_ERROR(ctx->output_list(kOutput, &opts->output));

  for (size_t i = 0; i < output_types().size(); ++i) {
    if (output_shapes_.at(i).IsFullyDefined()) {
      Tensor* out = nullptr;
      TensorShape output_shape;
      output_shapes_.at(i).AsTensorShape(&output_shape);
      output_shape.InsertDim(0, opts->batch_size);
      TF_RETURN_IF_ERROR(opts->output.allocate(i, output_shape, &out));
    }
  }
  return Status::OK();
}

namespace {
REGISTER_KERNEL_BUILDER(Name("MapDefun").Device(DEVICE_CPU), MapDefunOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
