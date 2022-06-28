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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPShost_compute_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPShost_compute_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPShost_compute_opsDTcc() {
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

#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {
namespace {

// The RecvAtHost op is used to deliver data from the device at the start of a
// host compute block. Setting `device_ordinal_is_attr` to true and false
// will switch between using device ordinal as an attribute and a runtime value
// respectively. To minimize cloning of ops/functions, it may be necessary to
// have device ordinal be a runtime value.
template <bool device_ordinal_is_attr>
class RecvAtHostOp : public AsyncOpKernel {
 public:
  explicit RecvAtHostOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPShost_compute_opsDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/tpu/kernels/host_compute_ops.cc", "RecvAtHostOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("key", &key_));
    int device_ordinal = 0;
    if (device_ordinal_is_attr) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ordinal", &device_ordinal));
      OP_REQUIRES(
          ctx, device_ordinal >= 0,
          errors::Internal("RecvAtHost device_ordinal must be non negative"));
      OP_REQUIRES(ctx, ctx->num_inputs() == 1,
                  errors::Internal("RecvAtHost must have exactly one input"));
      OP_REQUIRES(ctx, ctx->input_type(0) == DT_STRING,
                  errors::Internal("RecvAtHost input must have string type"));
    } else {
      OP_REQUIRES(ctx, ctx->num_inputs() == 2,
                  errors::Internal("RecvAtHost must have exactly two inputs"));
      OP_REQUIRES(ctx, ctx->input_type(0) == DT_STRING,
                  errors::Internal("RecvAtHost input 0 must have string type"));
      OP_REQUIRES(ctx, ctx->input_type(1) == DT_INT64,
                  errors::Internal("RecvAtHost input 1 must have int64 type"));
    }

    DeviceNameUtils::ParsedName parsed_name;
    OP_REQUIRES(
        ctx,
        DeviceNameUtils::ParseFullName(ctx->device()->name(), &parsed_name),
        errors::Internal("Could not parse device name."));
    parsed_name.type = "CPU";
    parsed_name.id = 0;
    cpu_device_ = DeviceNameUtils::ParsedNameToString(parsed_name);
    if (device_ordinal_is_attr) {
      parsed_name.type = "TPU";
      parsed_name.id = device_ordinal;
      tpu_device_ = DeviceNameUtils::ParsedNameToString(parsed_name);
      VLOG(2) << "  tpu_device_ = " << tpu_device_;
      VLOG(2) << "  cpu_device_ = " << cpu_device_;
    }
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPShost_compute_opsDTcc mht_1(mht_1_v, 247, "", "./tensorflow/core/tpu/kernels/host_compute_ops.cc", "ComputeAsync");

    string tpu_device;
    if (!device_ordinal_is_attr) {
      const Tensor& device_ordinal_tensor = ctx->input(1);
      OP_REQUIRES_ASYNC(
          ctx, TensorShapeUtils::IsScalar(device_ordinal_tensor.shape()),
          errors::InvalidArgument("device_ordinal must be a scalar, not ",
                                  device_ordinal_tensor.shape().DebugString()),
          done);
      const int device_ordinal = device_ordinal_tensor.flat<int64_t>()(0);
      OP_REQUIRES_ASYNC(
          ctx, device_ordinal >= 0,
          errors::Internal("RecvAtHost device_ordinal must be non negative"),
          done);
      DeviceNameUtils::ParsedName parsed_name;
      OP_REQUIRES_ASYNC(
          ctx, DeviceNameUtils::ParseFullName(cpu_device_, &parsed_name),
          errors::Internal("Could not parse device name."), done);
      parsed_name.type = "TPU";
      parsed_name.id = device_ordinal;
      tpu_device = DeviceNameUtils::ParsedNameToString(parsed_name);
      VLOG(2) << "  tpu_device_ = " << tpu_device;
      VLOG(2) << "  cpu_device_ = " << cpu_device_;
    }

    const Tensor& input = ctx->input(0);
    VLOG(2) << input.DebugString();
    OP_REQUIRES_ASYNC(
        ctx,
        TensorShapeUtils::IsVector(input.shape()) &&
            input.shape().dim_size(0) == 3,
        errors::InvalidArgument("Input shape ", input.shape().DebugString(),
                                " is not a vector of length 3."),
        done);
    const string rendezvous_key_base = input.vec<tstring>()(1);
    OP_REQUIRES_ASYNC(
        ctx, ctx->rendezvous() != nullptr,
        errors::Internal("Op kernel context needs to provide a rendezvous."),
        done);

    // Early return if there is no output to be received. Call `done()` to
    // unblock following execution.
    if (ctx->num_outputs() == 0) {
      done();
      return;
    }

    // Make all the parsed keys before starting any rendezvous->Recv calls to
    // avoid having to deal with an error case after some Recv have been
    // started.
    std::vector<string> rendezvous_key(ctx->num_outputs());
    std::vector<Rendezvous::ParsedKey> parsed_key(ctx->num_outputs());
    for (int i = 0; i < ctx->num_outputs(); ++i) {
      rendezvous_key[i] = Rendezvous::CreateKey(
          device_ordinal_is_attr ? tpu_device_ : tpu_device,
          /*src_incarnation=*/1, cpu_device_,
          strings::StrCat(rendezvous_key_base, key_, "_dtoh_", i),
          FrameAndIter(0, 0));

      OP_REQUIRES_OK_ASYNC(
          ctx, Rendezvous::ParseKey(rendezvous_key[i], &parsed_key[i]), done);
    }

    std::atomic_int_fast32_t* counter =
        new std::atomic_int_fast32_t(ctx->num_outputs());

    int num_outputs = ctx->num_outputs();
    for (int i = 0; i < num_outputs; ++i) {
      Rendezvous::Args args;
      args.device_context = ctx->op_device_context();
      args.alloc_attrs = ctx->output_alloc_attr(i);

      const string& key = rendezvous_key[i];
      VLOG(2) << "Recv " << key;
      ctx->rendezvous()->RecvAsync(
          parsed_key[i], args,
          [ctx, i, counter, key, done](const Status& s,
                                       const Rendezvous::Args& send_args,
                                       const Rendezvous::Args& recv_args,
                                       const Tensor& val, bool is_dead) {
            ctx->SetStatus(s);
            if (s.ok()) {
              ctx->set_output(i, val);
            }
            int previously_finished = counter->fetch_sub(1);
            VLOG(2) << "Processing Recv " << key << " " << s
                    << " previously finished " << previously_finished;
            if (previously_finished == 1) {
              delete counter;
              done();
            }
          });
    }
  }

 private:
  string key_;
  string tpu_device_;
  string cpu_device_;

  // RecvAtHostOp is neither copyable nor movable.
  RecvAtHostOp(const RecvAtHostOp&) = delete;
  RecvAtHostOp& operator=(const RecvAtHostOp&) = delete;
};

// The SendFromHost op is used to deliver data to the device at the end of a
// host compute block. Setting `device_ordinal_is_attr` to true and false will
// switch between using device ordinal as an attribute and a runtime value
// respectively. To minimize cloning of ops/functions, it may be necessary to
// have device ordinal be a runtime value.
template <bool device_ordinal_is_attr>
class SendFromHostOp : public OpKernel {
 public:
  explicit SendFromHostOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPShost_compute_opsDTcc mht_2(mht_2_v, 363, "", "./tensorflow/core/tpu/kernels/host_compute_ops.cc", "SendFromHostOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("key", &key_));
    int device_ordinal = 0;
    if (device_ordinal_is_attr) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ordinal", &device_ordinal));
      OP_REQUIRES(
          ctx, device_ordinal >= 0,
          errors::Internal("SendFromHost device_ordinal must be non negative"));
      OP_REQUIRES(
          ctx, ctx->num_inputs() > 0,
          errors::Internal("SendFromHost must have at least one input"));
      OP_REQUIRES(
          ctx, ctx->input_type(ctx->num_inputs() - 1) == DT_STRING,
          errors::Internal("SendFromHost last input must have string type"));
    } else {
      OP_REQUIRES(
          ctx, ctx->num_inputs() > 1,
          errors::Internal("SendFromHost must have at least two inputs"));
      OP_REQUIRES(
          ctx, ctx->input_type(ctx->num_inputs() - 2) == DT_STRING,
          errors::Internal(
              "SendFromHost second to last input must have string type"));
      OP_REQUIRES(
          ctx, ctx->input_type(ctx->num_inputs() - 1) == DT_INT64,
          errors::Internal("SendFromHost last input must have int64 type"));
    }

    DeviceNameUtils::ParsedName parsed_name;
    OP_REQUIRES(
        ctx,
        DeviceNameUtils::ParseFullName(ctx->device()->name(), &parsed_name),
        errors::Internal("Could not parse device name."));
    parsed_name.type = "CPU";
    parsed_name.id = 0;
    cpu_device_ = DeviceNameUtils::ParsedNameToString(parsed_name);
    if (device_ordinal_is_attr) {
      parsed_name.type = "TPU";
      parsed_name.id = device_ordinal;
      tpu_device_ = DeviceNameUtils::ParsedNameToString(parsed_name);
      VLOG(2) << "  tpu_device_ = " << tpu_device_;
      VLOG(2) << "  cpu_device_ = " << cpu_device_;
    }
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPShost_compute_opsDTcc mht_3(mht_3_v, 410, "", "./tensorflow/core/tpu/kernels/host_compute_ops.cc", "Compute");

    std::string tpu_device;
    if (!device_ordinal_is_attr) {
      const Tensor& device_ordinal_tensor = ctx->input(ctx->num_inputs() - 1);
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsScalar(device_ordinal_tensor.shape()),
          errors::InvalidArgument("device_ordinal must be a scalar, not ",
                                  device_ordinal_tensor.shape().DebugString()));
      const int device_ordinal = device_ordinal_tensor.flat<int64_t>()(0);
      OP_REQUIRES(
          ctx, device_ordinal >= 0,
          errors::Internal("SendFromHost device_ordinal must be non negative"));
      DeviceNameUtils::ParsedName parsed_name;
      OP_REQUIRES(ctx,
                  DeviceNameUtils::ParseFullName(cpu_device_, &parsed_name),
                  errors::Internal("Could not parse device name."));
      parsed_name.type = "TPU";
      parsed_name.id = device_ordinal;
      tpu_device = DeviceNameUtils::ParsedNameToString(parsed_name);
      VLOG(2) << "  tpu_device_ = " << tpu_device;
      VLOG(2) << "  cpu_device_ = " << cpu_device_;
    }

    const int num_send_inputs =
        ctx->num_inputs() - (device_ordinal_is_attr ? 1 : 2);
    const Tensor& key_input = ctx->input(num_send_inputs);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsVector(key_input.shape()) &&
                    key_input.shape().dim_size(0) == 3,
                errors::InvalidArgument("Key input shape ",
                                        key_input.shape().DebugString(),
                                        " is not a vector of length 3."));
    const string rendezvous_key_base = key_input.vec<tstring>()(1);
    OP_REQUIRES(
        ctx, ctx->rendezvous() != nullptr,
        errors::Internal("Op kernel context needs to provide a rendezvous."));

    for (int i = 0; i < num_send_inputs; ++i) {
      Rendezvous::Args args;
      args.device_context = ctx->op_device_context();
      args.alloc_attrs = ctx->input_alloc_attr(i);

      // TODO(misard) Fix this once we have replication.
      const string& rendezvous_key = Rendezvous::CreateKey(
          cpu_device_, /*src_incarnation=*/1,
          device_ordinal_is_attr ? tpu_device_ : tpu_device,
          strings::StrCat(rendezvous_key_base, key_, "_htod_", i),
          FrameAndIter(0, 0));

      Rendezvous::ParsedKey parsed_key;
      OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(rendezvous_key, &parsed_key));
      VLOG(2) << "Send " << rendezvous_key;
      OP_REQUIRES_OK(
          ctx, ctx->rendezvous()->Send(parsed_key, args, ctx->input(i), false));
    }
  }

 private:
  string key_;
  string cpu_device_;
  string tpu_device_;

  // SendFromHostOp is neither copyable nor movable.
  SendFromHostOp(const SendFromHostOp&) = delete;
  SendFromHostOp& operator=(const SendFromHostOp&) = delete;
};

}  // anonymous namespace

// These ops execute on the CPU device and must specify a non-negative value for
// device_ordinal to indicate which TPU to send infeed to.
REGISTER_KERNEL_BUILDER(Name("_XlaRecvAtHost").Device(DEVICE_CPU),
                        RecvAtHostOp<true>);

REGISTER_KERNEL_BUILDER(Name("_XlaRecvAtHostV2").Device(DEVICE_CPU),
                        RecvAtHostOp<false>);

REGISTER_KERNEL_BUILDER(Name("_XlaSendFromHost").Device(DEVICE_CPU),
                        SendFromHostOp<true>);

REGISTER_KERNEL_BUILDER(Name("_XlaSendFromHostV2").Device(DEVICE_CPU),
                        SendFromHostOp<false>);

}  // namespace tensorflow
