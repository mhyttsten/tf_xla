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
class MHTracer_DTPStensorflowPScorePSkernelsPSsendrecv_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsendrecv_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsendrecv_opsDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/sendrecv_ops.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {

static string GetRendezvousKeyPrefix(const string& send_device,
                                     const string& recv_device,
                                     const uint64 send_device_incarnation,
                                     const string& tensor_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("send_device: \"" + send_device + "\"");
   mht_0_v.push_back("recv_device: \"" + recv_device + "\"");
   mht_0_v.push_back("tensor_name: \"" + tensor_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSsendrecv_opsDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/kernels/sendrecv_ops.cc", "GetRendezvousKeyPrefix");

  return strings::StrCat(send_device, ";",
                         strings::FpToString(send_device_incarnation), ";",
                         recv_device, ";", tensor_name);
}

static void GetRendezvousKey(const string& key_prefix,
                             const FrameAndIter& frame_iter, string* key) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("key_prefix: \"" + key_prefix + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSsendrecv_opsDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/kernels/sendrecv_ops.cc", "GetRendezvousKey");

  key->clear();
  strings::StrAppend(key, key_prefix, ";", frame_iter.frame_id, ":",
                     frame_iter.iter_id);
}

static FrameAndIter GetFrameAndIter(OpKernelContext* ctx,
                                    bool hostmem_sendrecv) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsendrecv_opsDTcc mht_2(mht_2_v, 225, "", "./tensorflow/core/kernels/sendrecv_ops.cc", "GetFrameAndIter");

  if (hostmem_sendrecv && ctx->call_frame() != nullptr) {
    // Host memory send/recv pairs are added by
    // common_runtime/memory_types.cc.  When the pair of nodes are
    // added inside a function, we need to use the function call frame
    // to formulate the unique rendezvous key.
    return FrameAndIter(reinterpret_cast<uint64>(ctx->call_frame()), 0);
  } else {
    return ctx->frame_iter();
  }
}

SendOp::SendOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsendrecv_opsDTcc mht_3(mht_3_v, 240, "", "./tensorflow/core/kernels/sendrecv_ops.cc", "SendOp::SendOp");

  string send_device;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device", &send_device));
  string recv_device;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_device", &recv_device));
  uint64 send_device_incarnation;
  OP_REQUIRES_OK(
      ctx, ctx->GetAttr("send_device_incarnation",
                        reinterpret_cast<int64_t*>(&send_device_incarnation)));
  string tensor_name;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name));
  key_prefix_ = GetRendezvousKeyPrefix(send_device, recv_device,
                                       send_device_incarnation, tensor_name);
  // The vast majority of Send nodes are outside any loop context, so
  // proactively cache the rendezvous key for the top-level.
  GetRendezvousKey(key_prefix_, {0, 0}, &parsed_key_.buf_);
  OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(parsed_key_.buf_, &parsed_key_));
  if (!ctx->GetAttr("_hostmem_sendrecv", &hostmem_sendrecv_).ok()) {
    hostmem_sendrecv_ = false;
  }
}

void SendOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsendrecv_opsDTcc mht_4(mht_4_v, 265, "", "./tensorflow/core/kernels/sendrecv_ops.cc", "SendOp::Compute");

  OP_REQUIRES(
      ctx, ctx->rendezvous() != nullptr,
      errors::Internal("Op kernel context needs to provide a rendezvous."));

  // The device context may be passed between the Send/Recv
  // boundary, so that the device context used to produce the Tensor
  // is used when performing the copy on the recv side (which may be
  // a different device).
  Rendezvous::Args args;
  args.device_context = ctx->op_device_context();
  args.alloc_attrs = ctx->input_alloc_attr(0);

  FrameAndIter frame_iter = GetFrameAndIter(ctx, hostmem_sendrecv_);
  if (frame_iter == FrameAndIter(0, 0)) {
    // Use the cached rendezvous key.
    VLOG(2) << "Send " << parsed_key_.buf_ << " using "
            << reinterpret_cast<uintptr_t>(ctx->rendezvous());
    ctx->SetStatus(ctx->rendezvous()->Send(parsed_key_, args, ctx->input(0),
                                           ctx->is_input_dead()));
    return;
  } else {
    Rendezvous::ParsedKey in_loop_parsed;
    GetRendezvousKey(key_prefix_, frame_iter, &in_loop_parsed.buf_);
    VLOG(2) << "Send " << in_loop_parsed.buf_ << " using "
            << reinterpret_cast<uintptr_t>(ctx->rendezvous());
    OP_REQUIRES_OK(ctx,
                   Rendezvous::ParseKey(in_loop_parsed.buf_, &in_loop_parsed));

    ctx->SetStatus(ctx->rendezvous()->Send(in_loop_parsed, args, ctx->input(0),
                                           ctx->is_input_dead()));
    return;
  }
}

string SendOp::TraceString(const OpKernelContext& ctx, bool verbose) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsendrecv_opsDTcc mht_5(mht_5_v, 303, "", "./tensorflow/core/kernels/sendrecv_ops.cc", "SendOp::TraceString");

  const auto& attr = def().attr();
  auto src_it = attr.find("_src");
  auto dst_it = attr.find("_dst");
  const string& src = src_it != attr.end() ? src_it->second.s() : "";
  const string& dst = dst_it != attr.end() ? dst_it->second.s() : "";
  string op = profiler::TraceMeOp(name_view(), type_string_view());
  return profiler::TraceMeEncode(std::move(op), {{"from", src}, {"to", dst}});
}

REGISTER_KERNEL_BUILDER(Name("_Send").Device(DEVICE_CPU), SendOp);
REGISTER_KERNEL_BUILDER(Name("_Send").Device(DEVICE_DEFAULT), SendOp);
REGISTER_KERNEL_BUILDER(Name("_Send").Device(DEVICE_TPU_SYSTEM), SendOp);
REGISTER_KERNEL_BUILDER(Name("_HostSend").Device(DEVICE_TPU_SYSTEM), SendOp);

// Public alias. Added for use in Lingvo.
REGISTER_KERNEL_BUILDER(Name("Send").Device(DEVICE_CPU), SendOp);
REGISTER_KERNEL_BUILDER(Name("Send").Device(DEVICE_DEFAULT), SendOp);

REGISTER_KERNEL_BUILDER(
    Name("_HostSend").Device(DEVICE_DEFAULT).HostMemory("tensor"), SendOp);

RecvOp::RecvOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsendrecv_opsDTcc mht_6(mht_6_v, 328, "", "./tensorflow/core/kernels/sendrecv_ops.cc", "RecvOp::RecvOp");

  string send_device;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device", &send_device));
  string recv_device;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_device", &recv_device));
  uint64 send_device_incarnation;
  OP_REQUIRES_OK(
      ctx, ctx->GetAttr("send_device_incarnation",
                        reinterpret_cast<int64_t*>(&send_device_incarnation)));
  string tensor_name;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name));
  key_prefix_ = GetRendezvousKeyPrefix(send_device, recv_device,
                                       send_device_incarnation, tensor_name);
  // The vast majority of Recv nodes are outside any loop context, so
  // proactively cache the rendezvous key for the top-level.
  GetRendezvousKey(key_prefix_, {0, 0}, &parsed_key_.buf_);
  OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(parsed_key_.buf_, &parsed_key_));
  if (!ctx->GetAttr("_hostmem_sendrecv", &hostmem_sendrecv_).ok()) {
    hostmem_sendrecv_ = false;
  }
}

string RecvOp::TraceString(const OpKernelContext& ctx, bool verbose) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsendrecv_opsDTcc mht_7(mht_7_v, 353, "", "./tensorflow/core/kernels/sendrecv_ops.cc", "RecvOp::TraceString");

  const auto& attr = def().attr();
  auto src_it = attr.find("_src");
  auto dst_it = attr.find("_dst");
  const string& src = src_it != attr.end() ? src_it->second.s() : "";
  const string& dst = dst_it != attr.end() ? dst_it->second.s() : "";
  string op = profiler::TraceMeOp(name_view(), type_string_view());
  return profiler::TraceMeEncode(std::move(op), {{"from", src}, {"to", dst}});
}

namespace {
Rendezvous::DoneCallback make_recv_callback(OpKernelContext* ctx,
                                            AsyncOpKernel::DoneCallback done) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsendrecv_opsDTcc mht_8(mht_8_v, 368, "", "./tensorflow/core/kernels/sendrecv_ops.cc", "make_recv_callback");

  return [ctx, done = std::move(done)](const Status& s,
                                       const Rendezvous::Args& send_args,
                                       const Rendezvous::Args& recv_args,
                                       const Tensor& val, bool is_dead) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsendrecv_opsDTcc mht_9(mht_9_v, 375, "", "./tensorflow/core/kernels/sendrecv_ops.cc", "lambda");

    ctx->SetStatus(s);
    if (s.ok()) {
      // 'ctx' allocates the output tensor of the expected type.
      // The runtime checks whether the tensor received here is
      // the same type.
      if (!is_dead) {
        ctx->set_output(0, val);
      }
    }
    done();
  };
}
}  // namespace

void RecvOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsendrecv_opsDTcc mht_10(mht_10_v, 393, "", "./tensorflow/core/kernels/sendrecv_ops.cc", "RecvOp::ComputeAsync");

  OP_REQUIRES_ASYNC(
      ctx, ctx->rendezvous() != nullptr,
      errors::Internal("Op kernel context needs to provide a rendezvous."),
      done);

  Rendezvous::Args args;
  args.device_context = ctx->op_device_context();
  args.alloc_attrs = ctx->output_alloc_attr(0);
  args.cancellation_manager = ctx->cancellation_manager();

  FrameAndIter frame_iter = GetFrameAndIter(ctx, hostmem_sendrecv_);
  if (frame_iter == FrameAndIter(0, 0)) {
    VLOG(2) << "Recv " << parsed_key_.buf_ << " using "
            << reinterpret_cast<uintptr_t>(ctx->rendezvous());
    ctx->rendezvous()->RecvAsync(parsed_key_, args,
                                 make_recv_callback(ctx, std::move(done)));
  } else {
    Rendezvous::ParsedKey in_loop_parsed;
    GetRendezvousKey(key_prefix_, frame_iter, &in_loop_parsed.buf_);
    VLOG(2) << "Recv " << in_loop_parsed.buf_ << " using "
            << reinterpret_cast<uintptr_t>(ctx->rendezvous());
    OP_REQUIRES_OK_ASYNC(
        ctx, Rendezvous::ParseKey(in_loop_parsed.buf_, &in_loop_parsed), done);
    ctx->rendezvous()->RecvAsync(in_loop_parsed, args,
                                 make_recv_callback(ctx, std::move(done)));
  }
}

REGISTER_KERNEL_BUILDER(Name("_Recv").Device(DEVICE_CPU), RecvOp);
REGISTER_KERNEL_BUILDER(Name("_Recv").Device(DEVICE_DEFAULT), RecvOp);
REGISTER_KERNEL_BUILDER(Name("_Recv").Device(DEVICE_TPU_SYSTEM), RecvOp);
REGISTER_KERNEL_BUILDER(Name("_HostRecv").Device(DEVICE_TPU_SYSTEM), RecvOp);

// Public alias. Added for use in Lingvo.
REGISTER_KERNEL_BUILDER(Name("Recv").Device(DEVICE_CPU), RecvOp);
REGISTER_KERNEL_BUILDER(Name("Recv").Device(DEVICE_DEFAULT), RecvOp);

REGISTER_KERNEL_BUILDER(
    Name("_HostRecv").Device(DEVICE_DEFAULT).HostMemory("tensor"), RecvOp);

// Environment variable `DISABLE_HOST_SEND_RECV_REGISTRATION` is used to disable
// hostSend and hostRecv registration on CPU device in the mock environment.
static bool InitModule() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsendrecv_opsDTcc mht_11(mht_11_v, 439, "", "./tensorflow/core/kernels/sendrecv_ops.cc", "InitModule");

  if (!std::getenv("DISABLE_HOST_SEND_RECV_REGISTRATION")) {
    REGISTER_KERNEL_BUILDER(Name("_HostRecv").Device(DEVICE_CPU), RecvOp);
    REGISTER_KERNEL_BUILDER(Name("_HostSend").Device(DEVICE_CPU), SendOp);
  }
  return true;
}

static bool module_initialized = InitModule();

}  // end namespace tensorflow
