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
class MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc() {
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
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

static string CollectiveKey(OpKernelContext* ctx, int32_t group_key,
                            int32_t instance_key) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/kernels/collective_ops.cc", "CollectiveKey");

  return strings::StrCat(group_key, ":", instance_key, ":",
                         ctx->frame_iter().frame_id, ":",
                         ctx->frame_iter().iter_id);
}

static std::unique_ptr<OpKernel> BuildOpKernel(OpKernelConstruction* c,
                                               const string& name,
                                               NodeDef* sub_node) {
  std::unique_ptr<OpKernel> k;
  if (name.empty() || name == "Id") return k;
  sub_node->set_name(name);
  sub_node->set_op(name);
  Status status;
  k = CreateOpKernel(c->device_type(), c->device(),
                     c->device()->GetAllocator(AllocatorAttributes()),
                     *sub_node, c->graph_def_version(), &status);
  if (!status.ok()) {
    c->CtxFailureWithWarning(errors::Internal(
        "Failed to build OpKernel for ", name, " : ", status.error_message()));
  }
  return k;
}

class CollectiveOpV1Kernel : public AsyncOpKernel {
 public:
  explicit CollectiveOpV1Kernel(OpKernelConstruction* c)
      : AsyncOpKernel(c), name_(name()), col_params_(new CollectiveParams()) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_1(mht_1_v, 241, "", "./tensorflow/core/kernels/collective_ops.cc", "CollectiveOpV1Kernel");
}

  ~CollectiveOpV1Kernel() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_2(mht_2_v, 246, "", "./tensorflow/core/kernels/collective_ops.cc", "~CollectiveOpV1Kernel");
 col_params_->Unref(); }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_3(mht_3_v, 251, "", "./tensorflow/core/kernels/collective_ops.cc", "ComputeAsync");

    CollectiveExecutor* col_exec = c->collective_executor();
    OP_REQUIRES_ASYNC(
        c, col_exec,
        errors::Internal(
            "Failed to get CollectiveExecutor from OpKernelContext for Op ",
            name_),
        done);
    const CancellationToken token =
        c->cancellation_manager()->get_cancellation_token();
    const bool already_cancelled =
        !c->cancellation_manager()->RegisterCallback(token, [col_exec]() {
          // We must call StartAbort() within the callback. StartAbort() relies
          // on resources that may be deallocated if all execution of a graph is
          // finished.
          col_exec->StartAbort(errors::Cancelled("op cancelled"));
        });
    OP_REQUIRES_ASYNC(c, !already_cancelled,
                      errors::Cancelled("op cancelled ", name_), done);

    auto deregister_and_done = [c, token, done = std::move(done)]() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_4(mht_4_v, 274, "", "./tensorflow/core/kernels/collective_ops.cc", "lambda");

      // Once done() is called, StartAbort() won't have any effect, so we
      // don't need to block on the deregistration. Also StartAbort() may call
      // done() and DeregisterCallback may deadlock.
      c->cancellation_manager()->TryDeregisterCallback(token);
      done();
    };
    ComputeAsyncImpl(c, col_exec, std::move(deregister_and_done));
  }

  // A string encoding instance, frame and iter to be handed off to
  // the implementation for use in generating RecvBuf keys.
  string GetCollectiveKey(OpKernelContext* c) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_5(mht_5_v, 289, "", "./tensorflow/core/kernels/collective_ops.cc", "GetCollectiveKey");

    return CollectiveKey(c, col_params_->group.group_key,
                         col_params_->instance.instance_key);
  }

  // Returns false if calling invocation of ComputeAsync should return
  // immediately.
  bool CanProceedWithCompute(OpKernelContext* c, CollectiveExecutor* col_exec,
                             const DoneCallback& done) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_6(mht_6_v, 300, "", "./tensorflow/core/kernels/collective_ops.cc", "CanProceedWithCompute");

    if (col_params_->group.group_size > col_params_->group.members.size()) {
      // This is the first invocation: Finish initializing col_params_.
      // Schedule the `CompleteParamsAsync` call on a work queue that can handle
      // blocking work because it's not guaranteed that this call cannot block.
      c->collective_executor()->RunClosure([this, c, col_exec, done]() {
        VLOG(1) << "CollectiveOpKernel CompleteParams for collective "
                << col_params_->name << " device " << c->device()->name()
                << " group " << col_params_->group.group_key << " instance "
                << col_params_->instance.instance_key;
        col_exec->CompleteParamsAsync(
            c->device()->attributes(), col_params_, c->cancellation_manager(),
            [this, c, done](const Status& s) {
              if (s.ok()) {
                col_params_->instance.impl_details.dependencies = dependencies_;
                ComputeAsync(c, done);
              } else {
                c->SetStatus(s);
                done();
              }
            });
      });
      return false;
    }
    return true;
  }

 protected:
  virtual void ComputeAsyncImpl(OpKernelContext* c,
                                CollectiveExecutor* col_exec,
                                DoneCallback done) = 0;

  string name_;
  CollectiveParams* col_params_;
  std::vector<int32> dependencies_;
};

class CollectiveGatherOpKernel : public CollectiveOpV1Kernel {
 public:
  explicit CollectiveGatherOpKernel(OpKernelConstruction* c)
      : CollectiveOpV1Kernel(c) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_7(mht_7_v, 343, "", "./tensorflow/core/kernels/collective_ops.cc", "CollectiveGatherOpKernel");

    col_params_->instance.type = GATHER_COLLECTIVE;
    OP_REQUIRES_OK(c, c->GetAttr("group_size", &col_params_->group.group_size));
    OP_REQUIRES(
        c, col_params_->group.group_size > 0,
        errors::InvalidArgument("group_size must be positive integer but got ",
                                col_params_->group.group_size));
    OP_REQUIRES_OK(c, c->GetAttr("group_key", &col_params_->group.group_key));
    OP_REQUIRES_OK(
        c, c->GetAttr("instance_key", &col_params_->instance.instance_key));
    OP_REQUIRES_OK(c, c->GetAttr("T", &col_params_->instance.data_type));
    OP_REQUIRES_OK(
        c, c->GetAttr("communication_hint",
                      &col_params_->instance.impl_details.communication_hint));
    OP_REQUIRES_OK(
        c, c->GetAttr("timeout_seconds",
                      &col_params_->instance.impl_details.timeout_seconds));
    const NodeDef& real_node = c->def();
    col_params_->name = strings::StrCat(real_node.name(), ": Gather");
    col_params_->group.device_type = c->device_type();
  }

 protected:
  void ComputeAsyncImpl(OpKernelContext* c, CollectiveExecutor* col_exec,
                        DoneCallback done) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_8(mht_8_v, 370, "", "./tensorflow/core/kernels/collective_ops.cc", "ComputeAsyncImpl");

    auto output_shape = c->input(0).shape();
    output_shape.set_dim(
        0, output_shape.dim_size(0) * col_params_->group.group_size);
    col_params_->instance.shape = output_shape;

    // Allocate output on the first pass through this function.  This must be
    // done immediately, while we're still in the executor thread.  Otherwise
    // the memory is not guaranteed to be unused by any concurrently executing
    // GPU kernel.
    if (c->mutable_output(0) == nullptr) {
      // Allocate the output tensor.
      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(
          c, c->allocate_output(0, col_params_->instance.shape, &output), done);
    }
    if (!CanProceedWithCompute(c, col_exec, done)) return;

    auto actual_done = [c, col_params = col_params_, done](const Status& s) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_9(mht_9_v, 391, "", "./tensorflow/core/kernels/collective_ops.cc", "lambda");

      VLOG(1) << "CollectiveGatherOpKernel ExecuteAsync done for collective "
              << c->op_kernel().name() << " device " << c->device()->name()
              << " group " << col_params->group.group_key << " instance "
              << col_params->instance.instance_key << " status " << s;
      col_params->Unref();
      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };
    VLOG(1) << "CollectiveGatherOpKernel ExecuteAsync start for collective "
            << col_params_->name << " device " << c->device()->name()
            << " group " << col_params_->group.group_key << " instance "
            << col_params_->instance.instance_key;
    col_params_->Ref();
    col_exec->ExecuteAsync(c, col_params_, GetCollectiveKey(c), actual_done);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CollectiveGatherOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("CollectiveGather").Device(DEVICE_CPU),
                        CollectiveGatherOpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveGather").Device(DEVICE_GPU),
                        CollectiveGatherOpKernel);

class CollectiveReduceOpKernel : public CollectiveOpV1Kernel {
 public:
  explicit CollectiveReduceOpKernel(OpKernelConstruction* c)
      : CollectiveOpV1Kernel(c) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_10(mht_10_v, 423, "", "./tensorflow/core/kernels/collective_ops.cc", "CollectiveReduceOpKernel");

    col_params_->instance.type = REDUCTION_COLLECTIVE;
    OP_REQUIRES_OK(c, c->GetAttr("group_size", &col_params_->group.group_size));
    OP_REQUIRES(
        c, col_params_->group.group_size > 0,
        errors::InvalidArgument("group_size must be positive integer but got ",
                                col_params_->group.group_size));
    OP_REQUIRES_OK(c, c->GetAttr("group_key", &col_params_->group.group_key));
    OP_REQUIRES_OK(
        c, c->GetAttr("instance_key", &col_params_->instance.instance_key));
    OP_REQUIRES_OK(
        c, c->GetAttr("subdiv_offsets",
                      &col_params_->instance.impl_details.subdiv_offsets));
    string merge_op_name;
    OP_REQUIRES_OK(c, c->GetAttr("merge_op", &merge_op_name));
    if (merge_op_name == "Max") {
      merge_op_name = "Maximum";
    } else if (merge_op_name == "Min") {
      merge_op_name = "Minimum";
    }
    string final_op_name;
    OP_REQUIRES_OK(c, c->GetAttr("final_op", &final_op_name));
    OP_REQUIRES(c, final_op_name == "Id" || final_op_name == "Div",
                errors::InvalidArgument(
                    "final_op must be one of {\"Id\", \"Div\"} but got ",
                    final_op_name));
    OP_REQUIRES_OK(c, c->GetAttr("T", &col_params_->instance.data_type));
    OP_REQUIRES_OK(c, c->GetAttr("wait_for", &dependencies_));
    OP_REQUIRES_OK(
        c, c->GetAttr("communication_hint",
                      &col_params_->instance.impl_details.communication_hint));
    OP_REQUIRES_OK(
        c, c->GetAttr("timeout_seconds",
                      &col_params_->instance.impl_details.timeout_seconds));
    VLOG(2) << "CollectiveReduce instance "
            << col_params_->instance.instance_key << " merge_op "
            << merge_op_name << " final_op " << final_op_name
            << " communication_hint "
            << col_params_->instance.impl_details.communication_hint
            << " timeout "
            << col_params_->instance.impl_details.timeout_seconds;

    const NodeDef& real_node = c->def();
    col_params_->name = strings::StrCat(real_node.name(), ": Reduce(",
                                        merge_op_name, ",", final_op_name, ")");
    col_params_->group.device_type = c->device_type();

    // Find the OpKernels by name, type and device type.
    NodeDef sub_node;
    // The merge_op takes two inputs
    sub_node.add_input(real_node.input(0));
    sub_node.add_input(real_node.input(0));
    sub_node.set_device(real_node.device());
    SetAttrValue(col_params_->instance.data_type,
                 &(*sub_node.mutable_attr())["T"]);
    merge_op_ = BuildOpKernel(c, merge_op_name, &sub_node);
    final_op_ = BuildOpKernel(c, final_op_name, &sub_node);
    col_params_->merge_op = merge_op_.get();
    col_params_->final_op = final_op_.get();
  }

 protected:
  void ComputeAsyncImpl(OpKernelContext* c, CollectiveExecutor* col_exec,
                        DoneCallback done) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_11(mht_11_v, 489, "", "./tensorflow/core/kernels/collective_ops.cc", "ComputeAsyncImpl");

    // Allocate output on the first pass through this function.  This must be
    // done immediately, while we're still in the executor thread.  Otherwise
    // the memory is not guaranteed to be unused by any concurrently executing
    // GPU kernel.
    if (c->mutable_output(0) == nullptr) {
      // Allocate the output tensor, trying to reuse the input.
      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(c,
                           c->forward_input_or_allocate_output(
                               {0}, 0, c->input(0).shape(), &output),
                           done);
      col_params_->instance.shape = c->input(0).shape();
    }
    if (!CanProceedWithCompute(c, col_exec, done)) return;

    auto actual_done = [c, col_params = col_params_, done](const Status& s) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_12(mht_12_v, 508, "", "./tensorflow/core/kernels/collective_ops.cc", "lambda");

      VLOG(1) << "CollectiveReduceOpKernel ExecuteAsync done for collective "
              << c->op_kernel().name() << " device " << c->device()->name()
              << " group " << col_params->group.group_key << " instance "
              << col_params->instance.instance_key << " status " << s;
      col_params->Unref();
      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };
    VLOG(1) << "CollectiveReduceOpKernel ExecuteAsync start for collective "
            << col_params_->name << " device " << c->device()->name()
            << " group " << col_params_->group.group_key << " instance "
            << col_params_->instance.instance_key;
    col_params_->Ref();
    col_exec->ExecuteAsync(c, col_params_, GetCollectiveKey(c), actual_done);
  }

 private:
  std::unique_ptr<OpKernel> merge_op_;
  std::unique_ptr<OpKernel> final_op_;
  TF_DISALLOW_COPY_AND_ASSIGN(CollectiveReduceOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("CollectiveReduce").Device(DEVICE_CPU),
                        CollectiveReduceOpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveReduce").Device(DEVICE_GPU),
                        CollectiveReduceOpKernel);

class CollectiveBcastSendOpKernel : public CollectiveOpV1Kernel {
 public:
  explicit CollectiveBcastSendOpKernel(OpKernelConstruction* c)
      : CollectiveOpV1Kernel(c) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_13(mht_13_v, 542, "", "./tensorflow/core/kernels/collective_ops.cc", "CollectiveBcastSendOpKernel");

    col_params_->instance.type = BROADCAST_COLLECTIVE;
    OP_REQUIRES_OK(c, c->GetAttr("group_size", &col_params_->group.group_size));
    OP_REQUIRES(
        c, col_params_->group.group_size > 0,
        errors::InvalidArgument("group_size must be positive integer but got ",
                                col_params_->group.group_size));
    OP_REQUIRES_OK(c, c->GetAttr("group_key", &col_params_->group.group_key));
    OP_REQUIRES_OK(
        c, c->GetAttr("instance_key", &col_params_->instance.instance_key));
    OP_REQUIRES_OK(c, c->GetAttr("T", &col_params_->instance.data_type));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &col_params_->instance.shape));
    OP_REQUIRES_OK(
        c, c->GetAttr("communication_hint",
                      &col_params_->instance.impl_details.communication_hint));
    OP_REQUIRES_OK(
        c, c->GetAttr("timeout_seconds",
                      &col_params_->instance.impl_details.timeout_seconds));
    col_params_->is_source = true;
    col_params_->instance.impl_details.subdiv_offsets = {0};

    col_params_->name =
        strings::StrCat(name(), ": Broadcast(", col_params_->is_source, ")");
    col_params_->group.device_type = c->device_type();
  }

 protected:
  void ComputeAsyncImpl(OpKernelContext* c, CollectiveExecutor* col_exec,
                        DoneCallback done) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_14(mht_14_v, 573, "", "./tensorflow/core/kernels/collective_ops.cc", "ComputeAsyncImpl");

    // Allocate output on the first pass through this function.  This must be
    // done immediately, while we're still in the executor thread.  Otherwise
    // the memory is not guaranteed to be unused by any concurrently executing
    // GPU kernel.
    if (c->mutable_output(0) == nullptr) {
      // Allocate the output tensor, trying to reuse the input.
      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(c,
                           c->forward_input_or_allocate_output(
                               {0}, 0, col_params_->instance.shape, &output),
                           done);
    }
    if (!CanProceedWithCompute(c, col_exec, done)) return;
    OP_REQUIRES_ASYNC(
        c, col_params_->instance.shape.IsSameSize(c->input(0).shape()),
        errors::Internal("Declared shape of op ", col_params_->name,
                         " does not match shape of input"),
        done);

    auto actual_done = [c, col_params = col_params_, done](const Status& s) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_15(mht_15_v, 596, "", "./tensorflow/core/kernels/collective_ops.cc", "lambda");

      VLOG(1) << "CollectiveBcastSendOpKernel ExecuteAsync done for collective "
              << c->op_kernel().name() << " device " << c->device()->name()
              << " group " << col_params->group.group_key << " instance "
              << col_params->instance.instance_key << " status " << s;
      col_params->Unref();
      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };
    VLOG(1) << "CollectiveBcastSendOpKernel ExecuteAsync start for collective "
            << col_params_->name << " device " << c->device()->name()
            << " group " << col_params_->group.group_key << " instance "
            << col_params_->instance.instance_key;
    col_params_->Ref();
    col_exec->ExecuteAsync(c, col_params_, GetCollectiveKey(c), actual_done);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CollectiveBcastSendOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("CollectiveBcastSend").Device(DEVICE_CPU),
                        CollectiveBcastSendOpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveBcastSend").Device(DEVICE_DEFAULT),
                        CollectiveBcastSendOpKernel);

class CollectiveBcastRecvOpKernel : public CollectiveOpV1Kernel {
 public:
  explicit CollectiveBcastRecvOpKernel(OpKernelConstruction* c)
      : CollectiveOpV1Kernel(c) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_16(mht_16_v, 628, "", "./tensorflow/core/kernels/collective_ops.cc", "CollectiveBcastRecvOpKernel");

    col_params_->instance.type = BROADCAST_COLLECTIVE;
    OP_REQUIRES_OK(c, c->GetAttr("group_size", &col_params_->group.group_size));
    OP_REQUIRES(
        c, col_params_->group.group_size > 0,
        errors::InvalidArgument("group_size must be positive integer but got ",
                                col_params_->group.group_size));
    OP_REQUIRES_OK(c, c->GetAttr("group_key", &col_params_->group.group_key));
    OP_REQUIRES_OK(
        c, c->GetAttr("instance_key", &col_params_->instance.instance_key));
    OP_REQUIRES_OK(c, c->GetAttr("T", &col_params_->instance.data_type));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &col_params_->instance.shape));
    OP_REQUIRES_OK(
        c, c->GetAttr("communication_hint",
                      &col_params_->instance.impl_details.communication_hint));
    OP_REQUIRES_OK(
        c, c->GetAttr("timeout_seconds",
                      &col_params_->instance.impl_details.timeout_seconds));
    col_params_->is_source = false;
    col_params_->instance.impl_details.subdiv_offsets = {0};

    col_params_->name =
        strings::StrCat(name(), ": Broadcast(", col_params_->is_source, ")");
    col_params_->group.device_type = c->device_type();
  }

 protected:
  void ComputeAsyncImpl(OpKernelContext* c, CollectiveExecutor* col_exec,
                        DoneCallback done) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_17(mht_17_v, 659, "", "./tensorflow/core/kernels/collective_ops.cc", "ComputeAsyncImpl");

    // Allocate output on the first pass through this function.  This must be
    // done immediately, while we're still in the executor thread.  Otherwise
    // the memory is not guaranteed to be unused by any concurrently executing
    // GPU kernel.
    if (c->mutable_output(0) == nullptr) {
      // No input, so must allocate output.
      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(
          c, c->allocate_output(0, col_params_->instance.shape, &output), done);
    }
    if (!CanProceedWithCompute(c, col_exec, done)) return;

    auto actual_done = [c, col_params = col_params_, done](const Status& s) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_18(mht_18_v, 675, "", "./tensorflow/core/kernels/collective_ops.cc", "lambda");

      VLOG(1) << "CollectiveBcastRecvOpKernel ExecuteAsync done for collective "
              << c->op_kernel().name() << " device " << c->device()->name()
              << " group " << col_params->group.group_key << " instance_key "
              << col_params->instance.instance_key << " status  " << s;
      col_params->Unref();
      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };
    VLOG(1) << "CollectiveBcastRecvOpKernel ExecuteAsync start for collective "
            << col_params_->name << " device " << c->device()->name()
            << " group " << col_params_->group.group_key << " instance "
            << col_params_->instance.instance_key;
    col_params_->Ref();
    col_exec->ExecuteAsync(c, col_params_, GetCollectiveKey(c), actual_done);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CollectiveBcastRecvOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("CollectiveBcastRecv").Device(DEVICE_CPU),
                        CollectiveBcastRecvOpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveBcastRecv").Device(DEVICE_DEFAULT),
                        CollectiveBcastRecvOpKernel);

class CollectiveAssignGroupV2OpKernel : public OpKernel {
 public:
  explicit CollectiveAssignGroupV2OpKernel(OpKernelConstruction* c)
      : OpKernel(c) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_19(mht_19_v, 707, "", "./tensorflow/core/kernels/collective_ops.cc", "CollectiveAssignGroupV2OpKernel");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_20(mht_20_v, 712, "", "./tensorflow/core/kernels/collective_ops.cc", "Compute");

    const Tensor& group_assignment = context->input(0);
    const Tensor& device_index = context->input(1);
    const Tensor& base_key = context->input(2);

    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(device_index.shape()),
        errors::InvalidArgument(
            "device_index must be a scalar, but received tensor of shape: ",
            device_index.shape().DebugString()));

    OP_REQUIRES(
        context, TensorShapeUtils::IsMatrix(group_assignment.shape()),
        errors::InvalidArgument("group_assignment must be a 2-d Tensor, but "
                                "received tensor of shape: ",
                                group_assignment.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(base_key.shape()),
                errors::InvalidArgument(
                    "base_key must be a scalar, but received tensor of shape: ",
                    base_key.shape().DebugString()));

    Tensor* group_key = nullptr;
    Tensor* group_size = nullptr;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
                                                     &group_size, attr));

    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                     &group_key, attr));

    OP_REQUIRES_OK(
        context,
        ComputeGroupKey(group_assignment, device_index.scalar<int32_t>()(),
                        base_key.scalar<int32_t>()(), group_size, group_key));
  }

 private:
  static Status ComputeGroupKey(const Tensor& group_assignment,
                                const int32_t device_index,
                                const int32_t base_key, Tensor* group_size,
                                Tensor* group_key) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_21(mht_21_v, 756, "", "./tensorflow/core/kernels/collective_ops.cc", "ComputeGroupKey");

    group_size->flat<int32_t>()(0) = group_assignment.dim_size(1);

    for (int group_id = 0; group_id < group_assignment.dim_size(0);
         group_id++) {
      int32_t key = static_cast<int32_t>(static_cast<uint32_t>(base_key) +
                                         static_cast<uint32_t>(group_id));
      if (key == 0) {
        return errors::InvalidArgument(
            "Using the reserved group_key = 0 is not allowed: group_id = ",
            group_id, ", base_key = ", base_key);
      }
      for (int color = 0; color < group_assignment.dim_size(1); color++) {
        const auto index = group_assignment.matrix<int32>()(group_id, color);
        if (index < 0 || index >= group_assignment.shape().num_elements()) {
          return errors::InvalidArgument("Not all items in group_assignment ",
                                         group_assignment.DebugString(),
                                         " is within [0, number of devices)");
        }
        if (index == device_index) {
          group_key->flat<int32_t>()(0) = key;
          VLOG(2) << " group_assignment = " << group_assignment.DebugString()
                  << " device_index = " << index
                  << " group_key = " << group_key->DebugString()
                  << " group_size = " << group_size->DebugString();
          return Status::OK();
        }
      }
    }
    return errors::InvalidArgument("device_index ", device_index,
                                   " is not found in group_assignment ",
                                   group_assignment.DebugString());
  }
};

REGISTER_KERNEL_BUILDER(Name("CollectiveAssignGroupV2").Device(DEVICE_CPU),
                        CollectiveAssignGroupV2OpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveAssignGroupV2")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("device_index")
                            .HostMemory("group_assignment")
                            .HostMemory("base_key")
                            .HostMemory("group_size")
                            .HostMemory("group_key"),
                        CollectiveAssignGroupV2OpKernel);

class CollectiveOpV2Kernel : public AsyncOpKernel {
 public:
  explicit CollectiveOpV2Kernel(OpKernelConstruction* c)
      : AsyncOpKernel(c), name_(name()), device_type_(DEVICE_DEFAULT) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_22(mht_22_v, 808, "", "./tensorflow/core/kernels/collective_ops.cc", "CollectiveOpV2Kernel");

    OP_REQUIRES_OK(c, c->GetAttr("T", &data_type_));
    OP_REQUIRES_OK(c, c->GetAttr("communication_hint", &communication_hint_));
    OP_REQUIRES_OK(c, c->GetAttr("timeout_seconds", &timeout_seconds_));
    device_type_ = c->device_type();
  }

 protected:
  // Fills common parts of CollectiveParams according to the Op, *excluding
  // output_shape*. Kernels should further work on the CollectiveParams if they
  // need to set additional fields.
  Status FillCollectiveParams(CollectiveParams* col_params,
                              CollectiveType collective_type,
                              const Tensor& group_size, const Tensor& group_key,
                              const Tensor& instance_key) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_23(mht_23_v, 825, "", "./tensorflow/core/kernels/collective_ops.cc", "FillCollectiveParams");

    if (group_size.dims() > 0) {
      return errors::InvalidArgument(
          "Unexpected dimensions on input group_size, got ",
          group_size.shape().DebugString());
    }
    if (group_key.dims() > 0) {
      return errors::InvalidArgument(
          "Unexpected dimensions on input group_key, got ",
          group_key.shape().DebugString());
    }
    if (instance_key.dims() > 0) {
      return errors::InvalidArgument(
          "Unexpected dimensions on input instance_key, got ",
          instance_key.shape().DebugString());
    }
    col_params->name = name_;
    col_params->group.device_type = device_type_;
    col_params->group.group_size = group_size.unaligned_flat<int32>()(0);
    if (col_params->group.group_size <= 0) {
      return errors::InvalidArgument(
          "group_size must be positive integer but got ",
          col_params->group.group_size);
    }
    col_params->group.group_key = group_key.unaligned_flat<int32>()(0);
    col_params->instance.type = collective_type;
    col_params->instance.instance_key = instance_key.unaligned_flat<int32>()(0);
    col_params->instance.data_type = data_type_;
    col_params->instance.impl_details.communication_hint = communication_hint_;
    col_params->instance.impl_details.timeout_seconds = timeout_seconds_;
    return Status::OK();
  }

  // Runs a collective. The output tensor must be allocated before calling this
  // method. col_params must live until done is called.
  void Run(OpKernelContext* c, CollectiveParams* col_params,
           DoneCallback done) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_24(mht_24_v, 864, "", "./tensorflow/core/kernels/collective_ops.cc", "Run");

    CollectiveExecutor* col_exec = c->collective_executor();
    OP_REQUIRES_ASYNC(
        c, col_exec,
        errors::Internal(
            "Failed to get CollectiveExecutor from OpKernelContext for Op ",
            name_),
        done);
    // Resolve the collective params.
    // Schedule the `CompleteParamsAsync` call on a work queue that can handle
    // blocking work because it's not guaranteed that this call cannot block.
    c->collective_executor()->RunClosure([c, done = std::move(done), col_params,
                                          col_exec]() {
      VLOG(1) << "Collective CompleteParams for " << col_params->name
              << " device " << c->device()->name() << " group "
              << col_params->group.group_key << " instance "
              << col_params->instance.instance_key;
      col_exec->CompleteParamsAsync(
          c->device()->attributes(), col_params, c->cancellation_manager(),
          [c, done = std::move(done), col_params, col_exec](const Status& s) {
            if (s.ok()) {
              auto actual_done = [c, col_params,
                                  done = std::move(done)](const Status& s) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_25(mht_25_v, 889, "", "./tensorflow/core/kernels/collective_ops.cc", "lambda");

                VLOG(1) << "Collective ExecuteAsync done for "
                        << col_params->name << " device " << c->device()->name()
                        << " group " << col_params->group.group_key
                        << " instance " << col_params->instance.instance_key
                        << " status " << s;
                if (!s.ok()) {
                  c->SetStatus(s);
                }
                done();
              };
              VLOG(1) << "Collective ExecuteAsync start for "
                      << col_params->name << " device " << c->device()->name()
                      << " group " << col_params->group.group_key
                      << " instance " << col_params->instance.instance_key;
              col_exec->ExecuteAsync(
                  c, col_params,
                  CollectiveKey(c, col_params->group.group_key,
                                col_params->instance.instance_key),
                  actual_done);
            } else {
              c->SetStatus(s);
              done();
            }
          });
    });
  }

 protected:
  string name_;
  DataType data_type_ = DT_INVALID;
  string communication_hint_;
  float timeout_seconds_ = 0;
  DeviceType device_type_;
};

class CollectiveReduceV2OpKernel : public CollectiveOpV2Kernel {
 public:
  explicit CollectiveReduceV2OpKernel(OpKernelConstruction* c)
      : CollectiveOpV2Kernel(c) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_26(mht_26_v, 931, "", "./tensorflow/core/kernels/collective_ops.cc", "CollectiveReduceV2OpKernel");

    string merge_op_name;
    OP_REQUIRES_OK(c, c->GetAttr("merge_op", &merge_op_name));
    if (merge_op_name == "Max") {
      merge_op_name = "Maximum";
    } else if (merge_op_name == "Min") {
      merge_op_name = "Minimum";
    }
    string final_op_name;
    OP_REQUIRES_OK(c, c->GetAttr("final_op", &final_op_name));
    OP_REQUIRES_OK(
        c, c->GetAttr("max_subdivs_per_device", &max_subdivs_per_device_));
    // Prepare OpKernels for reduction and final operations.
    // The merge_op takes two inputs
    NodeDef sub_node;
    sub_node.add_input(c->def().input(0));
    sub_node.add_input(c->def().input(0));
    sub_node.set_device(c->def().device());
    SetAttrValue(data_type_, &(*sub_node.mutable_attr())["T"]);
    merge_op_ = BuildOpKernel(c, merge_op_name, &sub_node);
    final_op_ = BuildOpKernel(c, final_op_name, &sub_node);
    name_ = strings::StrCat(c->def().name(), ": ReduceV2(", merge_op_name, ",",
                            final_op_name, ")");
    VLOG(2) << "CollectiveReduceV2 " << this << " name " << name_
            << " communication_hint " << communication_hint_;
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_27(mht_27_v, 961, "", "./tensorflow/core/kernels/collective_ops.cc", "ComputeAsync");

    auto col_params = new CollectiveParams();
    auto done_with_cleanup = [col_params, done = std::move(done)]() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_28(mht_28_v, 966, "", "./tensorflow/core/kernels/collective_ops.cc", "lambda");

      done();
      col_params->Unref();
    };
    OP_REQUIRES_OK_ASYNC(c,
                         FillCollectiveParams(col_params, REDUCTION_COLLECTIVE,
                                              /*group_size*/ c->input(1),
                                              /*group_key*/ c->input(2),
                                              /*instance_key*/ c->input(3)),
                         done_with_cleanup);
    col_params->instance.shape = c->input(0).shape();
    col_params->merge_op = merge_op_.get();
    col_params->final_op = final_op_.get();
    VLOG(1) << "CollectiveReduceV2 group_size " << col_params->group.group_size
            << " group_key " << col_params->group.group_key << " instance_key "
            << col_params->instance.instance_key;
    // Allocate the output tensor, trying to reuse the input.
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(c,
                         c->forward_input_or_allocate_output(
                             {0}, 0, col_params->instance.shape, &output),
                         done_with_cleanup);
    Run(c, col_params, std::move(done_with_cleanup));
  }

 private:
  int max_subdivs_per_device_;
  std::unique_ptr<OpKernel> merge_op_;
  std::unique_ptr<OpKernel> final_op_;
};

REGISTER_KERNEL_BUILDER(Name("CollectiveReduceV2").Device(DEVICE_CPU),
                        CollectiveReduceV2OpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveReduceV2")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("group_size")
                            .HostMemory("group_key")
                            .HostMemory("instance_key"),
                        CollectiveReduceV2OpKernel);

class CollectiveGatherV2OpKernel : public CollectiveOpV2Kernel {
 public:
  explicit CollectiveGatherV2OpKernel(OpKernelConstruction* c)
      : CollectiveOpV2Kernel(c) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_29(mht_29_v, 1012, "", "./tensorflow/core/kernels/collective_ops.cc", "CollectiveGatherV2OpKernel");

    name_ = strings::StrCat(c->def().name(), ": GatherV2");
    VLOG(2) << "CollectiveGatherV2 " << this << " name " << name_
            << " communication_hint " << communication_hint_;
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_30(mht_30_v, 1021, "", "./tensorflow/core/kernels/collective_ops.cc", "ComputeAsync");

    auto col_params = new CollectiveParams();
    auto done_with_cleanup = [col_params, done = std::move(done)]() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_31(mht_31_v, 1026, "", "./tensorflow/core/kernels/collective_ops.cc", "lambda");

      done();
      col_params->Unref();
    };
    OP_REQUIRES_OK_ASYNC(c,
                         FillCollectiveParams(col_params, GATHER_COLLECTIVE,
                                              /*group_size*/ c->input(1),
                                              /*group_key*/ c->input(2),
                                              /*instance_key*/
                                              c->input(3)),
                         done_with_cleanup);
    auto output_shape = c->input(0).shape();
    output_shape.set_dim(
        0, output_shape.dim_size(0) * col_params->group.group_size);
    col_params->instance.shape = output_shape;
    VLOG(1) << "CollectiveGatherV2 group_size " << col_params->group.group_size
            << " group_key " << col_params->group.group_key << " instance_key "
            << col_params->instance.instance_key;
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(
        c, c->allocate_output(0, col_params->instance.shape, &output),
        done_with_cleanup);
    Run(c, col_params, std::move(done_with_cleanup));
  }
};

REGISTER_KERNEL_BUILDER(Name("CollectiveGatherV2").Device(DEVICE_CPU),
                        CollectiveGatherV2OpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveGatherV2")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("group_size")
                            .HostMemory("group_key")
                            .HostMemory("instance_key"),
                        CollectiveGatherV2OpKernel);

class CollectiveBcastSendV2OpKernel : public CollectiveOpV2Kernel {
 public:
  explicit CollectiveBcastSendV2OpKernel(OpKernelConstruction* c)
      : CollectiveOpV2Kernel(c) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_32(mht_32_v, 1067, "", "./tensorflow/core/kernels/collective_ops.cc", "CollectiveBcastSendV2OpKernel");

    const bool is_source = true;
    name_ = strings::StrCat(name(), ": Broadcast(", is_source, ")");
  }

 protected:
  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_33(mht_33_v, 1076, "", "./tensorflow/core/kernels/collective_ops.cc", "ComputeAsync");

    auto col_params = new CollectiveParams();
    auto done_with_cleanup = [col_params, done = std::move(done)]() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_34(mht_34_v, 1081, "", "./tensorflow/core/kernels/collective_ops.cc", "lambda");

      done();
      col_params->Unref();
    };
    OP_REQUIRES_OK_ASYNC(c,
                         FillCollectiveParams(col_params, BROADCAST_COLLECTIVE,
                                              /*group_size*/ c->input(1),
                                              /*group_key*/ c->input(2),
                                              /*instance_key*/ c->input(3)),
                         done_with_cleanup);
    col_params->is_source = true;
    col_params->instance.shape = c->input(0).shape();
    // Add a default value for subdiv offsets, which is the same as the default
    // value in the V1 op's attribute.
    col_params->instance.impl_details.subdiv_offsets.push_back(0);
    VLOG(1) << "CollectiveBcastSendV2 group_size "
            << col_params->group.group_size << " group_key "
            << col_params->group.group_key << " instance_key "
            << col_params->instance.instance_key;
    // Allocate the output tensor, trying to reuse the input.
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(c,
                         c->forward_input_or_allocate_output(
                             {0}, 0, col_params->instance.shape, &output),
                         done_with_cleanup);
    Run(c, col_params, std::move(done_with_cleanup));
  }
};

REGISTER_KERNEL_BUILDER(Name("CollectiveBcastSendV2").Device(DEVICE_CPU),
                        CollectiveBcastSendV2OpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveBcastSendV2")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("group_size")
                            .HostMemory("group_key")
                            .HostMemory("instance_key"),
                        CollectiveBcastSendV2OpKernel);

class CollectiveBcastRecvV2OpKernel : public CollectiveOpV2Kernel {
 public:
  explicit CollectiveBcastRecvV2OpKernel(OpKernelConstruction* c)
      : CollectiveOpV2Kernel(c) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_35(mht_35_v, 1125, "", "./tensorflow/core/kernels/collective_ops.cc", "CollectiveBcastRecvV2OpKernel");

    const bool is_source = false;
    name_ = strings::StrCat(name(), ": Broadcast(", is_source, ")");
  }

 protected:
  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_36(mht_36_v, 1134, "", "./tensorflow/core/kernels/collective_ops.cc", "ComputeAsync");

    auto col_params = new CollectiveParams();
    auto done_with_cleanup = [col_params, done = std::move(done)]() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_37(mht_37_v, 1139, "", "./tensorflow/core/kernels/collective_ops.cc", "lambda");

      done();
      col_params->Unref();
    };
    OP_REQUIRES_OK_ASYNC(c,
                         FillCollectiveParams(col_params, BROADCAST_COLLECTIVE,
                                              /*group_size*/ c->input(0),
                                              /*group_key*/ c->input(1),
                                              /*instance_key*/ c->input(2)),
                         done_with_cleanup);
    col_params->is_source = false;
    TensorShape output_shape;
    OP_REQUIRES_OK_ASYNC(c, tensor::MakeShape(c->input(3), &output_shape),
                         done_with_cleanup);
    col_params->instance.shape = output_shape;
    // Add a default value for subdiv offsets, which is the same as the default
    // value in the V1 op's attribute.
    col_params->instance.impl_details.subdiv_offsets.push_back(0);
    VLOG(1) << "CollectiveBcastRecvV2 group_size "
            << col_params->group.group_size << " group_key "
            << col_params->group.group_key << " instance_key "
            << col_params->instance.instance_key;
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(
        c, c->allocate_output(0, col_params->instance.shape, &output),
        done_with_cleanup);
    Run(c, col_params, std::move(done_with_cleanup));
  }
};

REGISTER_KERNEL_BUILDER(Name("CollectiveBcastRecvV2").Device(DEVICE_CPU),
                        CollectiveBcastRecvV2OpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveBcastRecvV2")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("group_size")
                            .HostMemory("group_key")
                            .HostMemory("instance_key")
                            .HostMemory("shape"),
                        CollectiveBcastRecvV2OpKernel);

/*
 * Resource for holding group for CollectiveOps.
 * This resource is returned from CollectiveInitializeCommunicatorOpKernel
 * It generates next instance key for the group for each collective operation.
 */
class CollectiveGroupResource : public ResourceBase {
 public:
  CollectiveGroupResource(int32 group_key, int32 rank, int32 group_size,
                          string communication_hint, float timeout_seconds)
      : group_key_(group_key),
        rank_(rank),
        group_size_(group_size),
        communication_hint_(communication_hint),
        timeout_seconds_(timeout_seconds) {
   std::vector<std::string> mht_38_v;
   mht_38_v.push_back("communication_hint: \"" + communication_hint + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_38(mht_38_v, 1196, "", "./tensorflow/core/kernels/collective_ops.cc", "CollectiveGroupResource");
}

  std::string DebugString() const override {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_39(mht_39_v, 1201, "", "./tensorflow/core/kernels/collective_ops.cc", "DebugString");

    return absl::StrFormat(
        "Collective Group with group_key = %d, group_size = %d, rank = %d",
        group_key_, group_size_, rank_);
  }

  int get_next_instance_key() {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_40(mht_40_v, 1210, "", "./tensorflow/core/kernels/collective_ops.cc", "get_next_instance_key");

    return instance_key_.fetch_add(1, std::memory_order_relaxed);
  }

  int32 group_key() const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_41(mht_41_v, 1217, "", "./tensorflow/core/kernels/collective_ops.cc", "group_key");
 return group_key_; }

  int32 rank() const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_42(mht_42_v, 1222, "", "./tensorflow/core/kernels/collective_ops.cc", "rank");
 return rank_; }

  int32 group_size() const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_43(mht_43_v, 1227, "", "./tensorflow/core/kernels/collective_ops.cc", "group_size");
 return group_size_; }

  string communication_hint() const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_44(mht_44_v, 1232, "", "./tensorflow/core/kernels/collective_ops.cc", "communication_hint");
 return communication_hint_; }

  float timeout_seconds() const {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_45(mht_45_v, 1237, "", "./tensorflow/core/kernels/collective_ops.cc", "timeout_seconds");
 return timeout_seconds_; }

 private:
  int32 group_key_, rank_, group_size_;
  string communication_hint_;
  std::atomic<int> instance_key_{0};
  float timeout_seconds_ = 0;
};

class CollectiveInitializeCommunicatorOpKernel : public AsyncOpKernel {
 public:
  explicit CollectiveInitializeCommunicatorOpKernel(OpKernelConstruction* c)
      : AsyncOpKernel(c), device_type_(DEVICE_DEFAULT) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_46(mht_46_v, 1252, "", "./tensorflow/core/kernels/collective_ops.cc", "CollectiveInitializeCommunicatorOpKernel");

    OP_REQUIRES_OK(c, c->GetAttr("communication_hint", &communication_hint_));
    OP_REQUIRES_OK(c, c->GetAttr("timeout_seconds", &timeout_seconds_));
    device_type_ = c->device_type();
  }

  Status CheckInputs(Tensor group_size_t, Tensor group_key_t) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_47(mht_47_v, 1261, "", "./tensorflow/core/kernels/collective_ops.cc", "CheckInputs");

    if (group_size_t.dims() > 0) {
      return errors::InvalidArgument(
          "Unexpected dimensions on input group_size. "
          "It shoulbe a scalar, got tensor with shape ",
          group_size_t.shape().DebugString());
    }
    if (group_key_t.dims() > 0) {
      return errors::InvalidArgument(
          "Unexpected dimensions on input group_key, got ",
          group_key_t.shape().DebugString());
    }

    auto group_size = group_size_t.unaligned_flat<int32>()(0);
    if (group_size <= 0) {
      return errors::InvalidArgument(
          "group_size must be positive integer but got ", group_size);
    }
    return Status::OK();
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_48(mht_48_v, 1285, "", "./tensorflow/core/kernels/collective_ops.cc", "ComputeAsync");

    auto group_key_t = c->input(0);
    auto rank_t = c->input(1);
    auto group_size_t = c->input(2);

    OP_REQUIRES_OK_ASYNC(c, CheckInputs(group_size_t, group_key_t), done);

    auto group_size = group_size_t.unaligned_flat<int32>()(0);
    auto group_key = group_key_t.unaligned_flat<int32>()(0);
    auto rank = rank_t.unaligned_flat<int32>()(0);

    ResourceHandle resource_handle =
        MakeResourceHandle<CollectiveGroupResource>(
            c, "collective_op_group",
            absl::StrFormat("%d:r%04d", group_key, rank));

    Tensor* output_handle = nullptr;
    OP_REQUIRES_OK_ASYNC(
        c, c->allocate_output(0, TensorShape({}), &output_handle), done);
    output_handle->scalar<ResourceHandle>()() = resource_handle;

    CollectiveGroupResource* resource = new CollectiveGroupResource(
        group_key, rank, group_size, this->communication_hint_,
        this->timeout_seconds_);
    OP_REQUIRES_OK_ASYNC(
        c,
        CreateResource<CollectiveGroupResource>(c, resource_handle, resource),
        done);
    auto group_params = new CollGroupParams();
    group_params->device_type = device_type_;
    group_params->group_size = resource->group_size();
    group_params->group_key = resource->group_key();
    group_params->user_specified_rank = resource->rank();

    auto* col_exec = c->collective_executor();

    c->collective_executor()->RunClosure([c, done = std::move(done),
                                          group_params, col_exec]() {
      VLOG(1) << "Collective Group initialization for "
              << " device " << c->device()->name() << " group "
              << group_params->group_key;
      col_exec->CompleteGroupAsync(
          c->device()->attributes(), group_params, c->cancellation_manager(),
          [c, done = std::move(done), group_params](const Status& s) {
            if (s.ok()) {
              VLOG(1) << "Collective Group initialization done for device "
                      << c->device()->name() << " group "
                      << group_params->group_key << " status " << s;
            } else {
              c->SetStatus(s);
            }
            delete group_params;
            done();
          });
    });
  }

 private:
  string communication_hint_;
  DeviceType device_type_;
  float timeout_seconds_ = 0;
};

REGISTER_KERNEL_BUILDER(
    Name("CollectiveInitializeCommunicator").Device(DEVICE_CPU),
    CollectiveInitializeCommunicatorOpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveInitializeCommunicator")
                            .Device(DEVICE_GPU)
                            .HostMemory("group_size")
                            .HostMemory("group_key")
                            .HostMemory("rank"),
                        CollectiveInitializeCommunicatorOpKernel);

class CollectiveOpV3Kernel : public AsyncOpKernel {
 public:
  explicit CollectiveOpV3Kernel(OpKernelConstruction* c)
      : AsyncOpKernel(c), name_(name()), device_type_(DEVICE_DEFAULT) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_49(mht_49_v, 1364, "", "./tensorflow/core/kernels/collective_ops.cc", "CollectiveOpV3Kernel");

    OP_REQUIRES_OK(c, c->GetAttr("T", &data_type_));
    if (c->HasAttr("timeout_seconds")) {
      OP_REQUIRES_OK(c, c->GetAttr("timeout_seconds", &timeout_seconds_));
    } else {
      timeout_seconds_ = -1;
    }
    device_type_ = c->device_type();
  }

 protected:
  // Fills common parts of CollectiveParams according to the Op, *excluding
  // output_shape*. Kernels should further work on the CollectiveParams if they
  // need to set additional fields.
  Status FillCollectiveParams(CollectiveParams* col_params,
                              const Tensor& group_assignment,
                              CollectiveType collective_type,
                              CollectiveGroupResource* resource) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_50(mht_50_v, 1384, "", "./tensorflow/core/kernels/collective_ops.cc", "FillCollectiveParams");

    int64 group_id;
    int64 group_size;
    if (group_assignment.NumElements() == 0) {
      // No group assignments, perform collective as a single group.
      group_id = 0;
      group_size = resource->group_size();
    } else {
      return errors::Unimplemented("Group assignments are not supported yet.");
    }

    // Construct instance key with format:
    // <11 bits for group><21 bits for atomic incremented instance key>
    int32 instance_key = group_id << 21 | resource->get_next_instance_key();
    col_params->name = name_;
    col_params->group.device_type = device_type_;
    col_params->group.group_size = group_size;
    col_params->group.group_key = resource->group_key();
    col_params->group.user_specified_rank = resource->rank();
    col_params->instance.type = collective_type;
    col_params->instance.instance_key = instance_key;
    col_params->instance.data_type = data_type_;
    col_params->instance.impl_details.communication_hint =
        resource->communication_hint();
    col_params->instance.impl_details.timeout_seconds =
        timeout_seconds_ > 0 ? resource->timeout_seconds() : timeout_seconds_;
    col_params->run_group_initialization = false;
    return Status::OK();
  }

  // Runs a collective. The output tensor must be allocated before calling this
  // method. col_params must live until done is called.
  void Run(OpKernelContext* c, CollectiveParams* col_params,
           DoneCallback done) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_51(mht_51_v, 1420, "", "./tensorflow/core/kernels/collective_ops.cc", "Run");

    CollectiveExecutor* col_exec = c->collective_executor();
    OP_REQUIRES_ASYNC(
        c, col_exec,
        errors::Internal(
            "Failed to get CollectiveExecutor from OpKernelContext for Op ",
            name_),
        done);
    // Resolve the collective params.
    // Schedule the `CompleteParamsAsync` call on a work queue that can handle
    // blocking work because it's not guaranteed that this call cannot block.
    col_exec->RunClosure([c, done = std::move(done), col_params, col_exec]() {
      VLOG(1) << "Collective CompleteParams for " << col_params->name
              << " device " << c->device()->name() << " group "
              << col_params->group.group_key << " instance "
              << col_params->instance.instance_key;
      col_exec->CompleteParamsAsync(
          c->device()->attributes(), col_params, c->cancellation_manager(),
          [c, done = std::move(done), col_params, col_exec](const Status& s) {
            if (s.ok()) {
              auto actual_done = [c, col_params,
                                  done = std::move(done)](const Status& s) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_52(mht_52_v, 1444, "", "./tensorflow/core/kernels/collective_ops.cc", "lambda");

                VLOG(1) << "Collective ExecuteAsync done for "
                        << col_params->name << " device " << c->device()->name()
                        << " group " << col_params->group.group_key
                        << " instance " << col_params->instance.instance_key
                        << " status " << s;
                if (!s.ok()) {
                  c->SetStatus(s);
                }
                done();
              };
              VLOG(1) << "Collective ExecuteAsync start for "
                      << col_params->name << " device " << c->device()->name()
                      << " group " << col_params->group.group_key
                      << " instance " << col_params->instance.instance_key;
              col_exec->ExecuteAsync(
                  c, col_params,
                  CollectiveKey(c, col_params->group.group_key,
                                col_params->instance.instance_key),
                  actual_done);
            } else {
              c->SetStatus(s);
              done();
            }
          });
    });
  }

 protected:
  string name_;
  DataType data_type_ = DT_INVALID;
  DeviceType device_type_;
  float timeout_seconds_ = 0;
};

class CollectiveReduceV3OpKernel : public CollectiveOpV3Kernel {
 public:
  explicit CollectiveReduceV3OpKernel(OpKernelConstruction* c)
      : CollectiveOpV3Kernel(c) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_53(mht_53_v, 1485, "", "./tensorflow/core/kernels/collective_ops.cc", "CollectiveReduceV3OpKernel");

    string reduction;
    OP_REQUIRES_OK(c, c->GetAttr("reduction", &reduction));
    if (reduction == "Max") {
      reduction = "Maximum";
    } else if (reduction == "Min") {
      reduction = "Minimum";
    }
    // Prepare OpKernels for reduction and final operations.
    // The merge_op takes two inputs
    NodeDef sub_node;
    sub_node.add_input(c->def().input(0));
    sub_node.add_input(c->def().input(0));
    sub_node.set_device(c->def().device());
    SetAttrValue(data_type_, &(*sub_node.mutable_attr())["T"]);
    merge_op_ = BuildOpKernel(c, reduction, &sub_node);
    final_op_ = BuildOpKernel(c, "Id", &sub_node);
    name_ = strings::StrCat(c->def().name(), ": ReduceV3(", reduction, ")");
    VLOG(2) << "CollectiveReduceV3 " << this << " name " << name_;
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_54(mht_54_v, 1509, "", "./tensorflow/core/kernels/collective_ops.cc", "ComputeAsync");

    auto col_params = new CollectiveParams();
    auto done_with_cleanup = [col_params, done = std::move(done)]() {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_55(mht_55_v, 1514, "", "./tensorflow/core/kernels/collective_ops.cc", "lambda");

      done();
      col_params->Unref();
    };
    core::RefCountPtr<CollectiveGroupResource> resource;
    OP_REQUIRES_OK_ASYNC(c, LookupResource(c, HandleFromInput(c, 1), &resource),
                         done_with_cleanup);

    Tensor group_assignment = c->input(2);

    OP_REQUIRES_OK_ASYNC(
        c,
        FillCollectiveParams(col_params, group_assignment, REDUCTION_COLLECTIVE,
                             resource.get()),
        done_with_cleanup);
    col_params->instance.shape = c->input(0).shape();
    col_params->merge_op = merge_op_.get();
    col_params->final_op = final_op_.get();
    VLOG(1) << "CollectiveReduceV3 group_size " << col_params->group.group_size
            << " group_key " << col_params->group.group_key << " instance_key "
            << col_params->instance.instance_key;
    // Allocate the output tensor, trying to reuse the input.
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(c,
                         c->forward_input_or_allocate_output(
                             {0}, 0, col_params->instance.shape, &output),
                         done_with_cleanup);
    Run(c, col_params, std::move(done_with_cleanup));
  }

 private:
  std::unique_ptr<OpKernel> merge_op_;
  std::unique_ptr<OpKernel> final_op_;
};

REGISTER_KERNEL_BUILDER(Name("CollectiveReduceV3").Device(DEVICE_CPU),
                        CollectiveReduceV3OpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveReduceV3").Device(DEVICE_GPU),
                        CollectiveReduceV3OpKernel);

class CollectiveAllToAllV3OpKernel : public CollectiveOpV3Kernel {
 public:
  explicit CollectiveAllToAllV3OpKernel(OpKernelConstruction* c)
      : CollectiveOpV3Kernel(c) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_56(mht_56_v, 1560, "", "./tensorflow/core/kernels/collective_ops.cc", "CollectiveAllToAllV3OpKernel");

    name_ = strings::StrCat(c->def().name(), ": AllToAllV3");
    VLOG(2) << "CollectiveAllToAllV3 " << this << " name " << name_;
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_57(mht_57_v, 1568, "", "./tensorflow/core/kernels/collective_ops.cc", "ComputeAsync");

    auto col_params = new CollectiveParams();
    auto done_with_cleanup = [col_params, done = std::move(done)]() {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_opsDTcc mht_58(mht_58_v, 1573, "", "./tensorflow/core/kernels/collective_ops.cc", "lambda");

      done();
      col_params->Unref();
    };
    core::RefCountPtr<CollectiveGroupResource> resource;
    OP_REQUIRES_OK_ASYNC(c, LookupResource(c, HandleFromInput(c, 1), &resource),
                         done_with_cleanup);

    Tensor group_assignment = c->input(2);

    OP_REQUIRES_OK_ASYNC(
        c,
        FillCollectiveParams(col_params, group_assignment,
                             ALL_TO_ALL_COLLECTIVE, resource.get()),
        done_with_cleanup);
    col_params->instance.shape = c->input(0).shape();
    VLOG(1) << "CollectiveAllToAll group_size " << col_params->group.group_size
            << " group_key " << col_params->group.group_key << " instance_key "
            << col_params->instance.instance_key;
    // Allocate the output tensor, trying to reuse the input.
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(c,
                         c->forward_input_or_allocate_output(
                             {0}, 0, col_params->instance.shape, &output),
                         done_with_cleanup);
    Run(c, col_params, std::move(done_with_cleanup));
  }
};

REGISTER_KERNEL_BUILDER(Name("CollectiveAllToAllV3").Device(DEVICE_CPU),
                        CollectiveAllToAllV3OpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveAllToAllV3").Device(DEVICE_GPU),
                        CollectiveAllToAllV3OpKernel);
}  // namespace
}  // namespace tensorflow
