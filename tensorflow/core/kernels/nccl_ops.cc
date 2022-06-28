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
class MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc() {
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <vector>

#if GOOGLE_CUDA
#include "third_party/nccl/nccl.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/rccl/rccl.h"
#endif
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/nccl/nccl_manager.h"

namespace tensorflow {
namespace {

// Base class for all communicator ops that use nccl.
//
// About memory management and stream syncing:
// 1. The nccl communicator has a stream for each rank.
// 2. For input tensors to the communicator, the compute stream is passed to the
//    NcclManager which will do a needed
//    communicator_stream.ThenWaitFor(input_tensor_stream).
// 3. The done_callback of the async kernel is not called by the
//    NcclManager until after the communicator kernel is complete. This
//    is enough to a) keep the input tensor data valid for the lifetime of the
//    collective; and b) ensure the data in the output tensor is available
//    when the async op kernel's done callback is called.
class NcclAsyncOpBase : public AsyncOpKernel {
 public:
  explicit NcclAsyncOpBase(OpKernelConstruction* c) : AsyncOpKernel(c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/kernels/nccl_ops.cc", "NcclAsyncOpBase");

    OP_REQUIRES_OK(c, c->GetAttr("num_devices", &num_devices_));
    OP_REQUIRES_OK(c, c->GetAttr("shared_name", &collective_prefix_));
  }

  string GetCollectiveKey(OpKernelContext* c) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_1(mht_1_v, 222, "", "./tensorflow/core/kernels/nccl_ops.cc", "GetCollectiveKey");

    return strings::StrCat(collective_prefix_, ";", c->step_id(), ";",
                           c->frame_iter().frame_id, ":",
                           c->frame_iter().iter_id);
  }

  int num_devices() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/kernels/nccl_ops.cc", "num_devices");
 return num_devices_; }

 private:
  int num_devices_;
  string collective_prefix_;

  TF_DISALLOW_COPY_AND_ASSIGN(NcclAsyncOpBase);
};

class NcclReduceOpBase : public NcclAsyncOpBase {
 public:
  explicit NcclReduceOpBase(OpKernelConstruction* c) : NcclAsyncOpBase(c) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_3(mht_3_v, 245, "", "./tensorflow/core/kernels/nccl_ops.cc", "NcclReduceOpBase");

    string reduction;
    OP_REQUIRES_OK(c, c->GetAttr("reduction", &reduction));
    if (reduction == "min") {
      reduction_op_ = ncclMin;
    } else if (reduction == "max") {
      reduction_op_ = ncclMax;
    } else if (reduction == "sum") {
      reduction_op_ = ncclSum;
    } else if (reduction == "prod") {
      reduction_op_ = ncclProd;
    } else {
      OP_REQUIRES_OK(c,
                     errors::InvalidArgument("Invalid reduction: ", reduction));
    }
  }

  ncclRedOp_t reduction_op() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_4(mht_4_v, 265, "", "./tensorflow/core/kernels/nccl_ops.cc", "reduction_op");
 return reduction_op_; }

 private:
  ncclRedOp_t reduction_op_;
};

// To execute a single all-reduce, this kernel is called once for each of the
// <k> devices in the communicator.
class NcclAllReduceOpKernel : public NcclReduceOpBase {
 public:
  explicit NcclAllReduceOpKernel(OpKernelConstruction* c)
      : NcclReduceOpBase(c) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_5(mht_5_v, 279, "", "./tensorflow/core/kernels/nccl_ops.cc", "NcclAllReduceOpKernel");
}

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_6(mht_6_v, 284, "", "./tensorflow/core/kernels/nccl_ops.cc", "ComputeAsync");

    const Tensor* input = &c->input(0);
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(
        c, c->forward_input_or_allocate_output({0}, 0, input->shape(), &output),
        done);
    auto actual_done = [c, done](Status s) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_7(mht_7_v, 293, "", "./tensorflow/core/kernels/nccl_ops.cc", "lambda");

      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };

    auto* compute_stream = c->op_device_context()->stream();
    auto* gpu_info = c->device()->tensorflow_accelerator_device_info();
    auto participant = absl::make_unique<NcclManager::Participant>(
        compute_stream->parent(), compute_stream, gpu_info, input, output,
        /*global_rank=*/-1, std::move(actual_done));
    NcclManager::instance()->AddToAllReduce(
        std::move(participant),
        {GetCollectiveKey(c),
         /*num_local_devices=*/num_devices(),
         /*num_global_devices=*/num_devices(),
         /*communicator_key=*/"", /*source_rank=*/-1},
        reduction_op());
  }
};
REGISTER_KERNEL_BUILDER(Name("NcclAllReduce").Device(DEVICE_GPU),
                        NcclAllReduceOpKernel);

// To execute a single reduce, this kernel is called once for all but one of the
// <k> devices in the communicator, and NcclReduceRecvKernel is called once for
// the remaining device.
class NcclReduceSendKernel : public NcclReduceOpBase {
 public:
  explicit NcclReduceSendKernel(OpKernelConstruction* c)
      : NcclReduceOpBase(c) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_8(mht_8_v, 324, "", "./tensorflow/core/kernels/nccl_ops.cc", "NcclReduceSendKernel");
}

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_9(mht_9_v, 329, "", "./tensorflow/core/kernels/nccl_ops.cc", "ComputeAsync");

    auto actual_done = [c, done](Status s) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_10(mht_10_v, 333, "", "./tensorflow/core/kernels/nccl_ops.cc", "lambda");

      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };

    auto* compute_stream = c->op_device_context()->stream();
    auto* gpu_info = c->device()->tensorflow_accelerator_device_info();
    auto participant = absl::make_unique<NcclManager::Participant>(
        compute_stream->parent(), compute_stream, gpu_info, &c->input(0),
        /*output=*/nullptr, /*global_rank=*/-1, std::move(actual_done));
    NcclManager::instance()->AddReduceSend(
        std::move(participant),
        {GetCollectiveKey(c),
         /*num_local_devices=*/num_devices(),
         /*num_global_devices=*/num_devices(),
         /*communicator_key=*/"", /*source_rank=*/-1},
        reduction_op());
  }
};
REGISTER_KERNEL_BUILDER(Name("_NcclReduceSend").Device(DEVICE_GPU),
                        NcclReduceSendKernel);

// To execute a single reduce, this kernel is called once for one devices, and
// NcclReduceSendKernel is called for all other <k-1> devices in the
// communicator.
class NcclReduceRecvKernel : public NcclReduceOpBase {
 public:
  explicit NcclReduceRecvKernel(OpKernelConstruction* c)
      : NcclReduceOpBase(c) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_11(mht_11_v, 364, "", "./tensorflow/core/kernels/nccl_ops.cc", "NcclReduceRecvKernel");
}

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_12(mht_12_v, 369, "", "./tensorflow/core/kernels/nccl_ops.cc", "ComputeAsync");

    const Tensor* input = &c->input(0);
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(c, c->allocate_output(0, input->shape(), &output),
                         done);

    auto actual_done = [c, done](Status s) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_13(mht_13_v, 378, "", "./tensorflow/core/kernels/nccl_ops.cc", "lambda");

      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };

    auto* compute_stream = c->op_device_context()->stream();
    auto* gpu_info = c->device()->tensorflow_accelerator_device_info();
    auto participant = absl::make_unique<NcclManager::Participant>(
        compute_stream->parent(), compute_stream, gpu_info, input, output,
        /*global_rank=*/-1, std::move(actual_done));
    NcclManager::instance()->AddReduceRecv(
        std::move(participant),
        {GetCollectiveKey(c),
         /*num_local_devices=*/num_devices(),
         /*num_global_devices=*/num_devices(),
         /*communicator_key=*/"", /*source_rank=*/-1},
        reduction_op());
  }

 private:
  ncclRedOp_t reduction_op_;
};
REGISTER_KERNEL_BUILDER(Name("_NcclReduceRecv").Device(DEVICE_GPU),
                        NcclReduceRecvKernel);

// To execute a single broadcast, this kernel is called once for one device, and
// NcclBroadcastRecvKernel is called for all other <k-1> devices in the
// communicator.
class NcclBroadcastSendKernel : public NcclAsyncOpBase {
 public:
  explicit NcclBroadcastSendKernel(OpKernelConstruction* c)
      : NcclAsyncOpBase(c) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_14(mht_14_v, 412, "", "./tensorflow/core/kernels/nccl_ops.cc", "NcclBroadcastSendKernel");
}

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_15(mht_15_v, 417, "", "./tensorflow/core/kernels/nccl_ops.cc", "ComputeAsync");

    auto actual_done = [c, done](Status s) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_16(mht_16_v, 421, "", "./tensorflow/core/kernels/nccl_ops.cc", "lambda");

      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };

    auto* compute_stream = c->op_device_context()->stream();
    auto* gpu_info = c->device()->tensorflow_accelerator_device_info();
    auto participant = absl::make_unique<NcclManager::Participant>(
        compute_stream->parent(), compute_stream, gpu_info, &c->input(0),
        /*output=*/nullptr, /*global_rank=*/-1, std::move(actual_done));
    NcclManager::instance()->AddBroadcastSend(
        std::move(participant), {GetCollectiveKey(c),
                                 /*num_local_devices=*/num_devices(),
                                 /*num_global_devices=*/num_devices(),
                                 /*communicator_key=*/"", /*source_rank=*/-1});
  }
};
REGISTER_KERNEL_BUILDER(Name("_NcclBroadcastSend").Device(DEVICE_GPU),
                        NcclBroadcastSendKernel);

// To execute a single broadcast, this kernel is called once for all but one of
// the <k> devices in the communicator, and NcclBroadcastSendKernel is called
// once for the remaining device.
class NcclBroadcastRecvKernel : public NcclAsyncOpBase {
 public:
  explicit NcclBroadcastRecvKernel(OpKernelConstruction* c)
      : NcclAsyncOpBase(c) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_17(mht_17_v, 450, "", "./tensorflow/core/kernels/nccl_ops.cc", "NcclBroadcastRecvKernel");
}

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_18(mht_18_v, 455, "", "./tensorflow/core/kernels/nccl_ops.cc", "ComputeAsync");

    const Tensor& shape_t = c->input(0);
    TensorShape shape;
    OP_REQUIRES_OK_ASYNC(
        c, TensorShapeUtils::MakeShape(shape_t.vec<int32>(), &shape), done);
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(c, c->allocate_output(0, shape, &output), done);

    auto actual_done = [c, done](Status s) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_19(mht_19_v, 466, "", "./tensorflow/core/kernels/nccl_ops.cc", "lambda");

      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };

    auto* compute_stream = c->op_device_context()->stream();
    auto* gpu_info = c->device()->tensorflow_accelerator_device_info();
    auto participant = absl::make_unique<NcclManager::Participant>(
        compute_stream->parent(), compute_stream, gpu_info,
        /*input=*/nullptr, output, /*global_rank=*/-1, std::move(actual_done));
    NcclManager::instance()->AddBroadcastRecv(
        std::move(participant), {GetCollectiveKey(c),
                                 /*num_local_devices=*/num_devices(),
                                 /*num_global_devices=*/num_devices(),
                                 /*communicator_key=*/"", /*source_rank=*/-1});
  }
};
REGISTER_KERNEL_BUILDER(
    Name("_NcclBroadcastRecv").Device(DEVICE_GPU).HostMemory("shape"),
    NcclBroadcastRecvKernel);

// Define stub kernels for the ops that get replaced post placement.
class NcclStubKernel : public AsyncOpKernel {
 public:
  explicit NcclStubKernel(OpKernelConstruction* c) : AsyncOpKernel(c) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_20(mht_20_v, 493, "", "./tensorflow/core/kernels/nccl_ops.cc", "NcclStubKernel");
}
  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSnccl_opsDTcc mht_21(mht_21_v, 497, "", "./tensorflow/core/kernels/nccl_ops.cc", "ComputeAsync");

    c->SetStatus(errors::Unimplemented(
        "This op should be replaced during graph optimization."));
    done();
  }
};
REGISTER_KERNEL_BUILDER(Name("NcclBroadcast").Device(DEVICE_GPU),
                        NcclStubKernel);
REGISTER_KERNEL_BUILDER(Name("NcclReduce").Device(DEVICE_GPU), NcclStubKernel);

}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
