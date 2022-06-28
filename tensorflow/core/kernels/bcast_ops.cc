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
class MHTracer_DTPStensorflowPScorePSkernelsPSbcast_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSbcast_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSbcast_opsDTcc() {
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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

// Given shapes of two tensors, computes the broadcast shape.
template <typename T>
class BCastArgsOp : public OpKernel {
 public:
  explicit BCastArgsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbcast_opsDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/kernels/bcast_ops.cc", "BCastArgsOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbcast_opsDTcc mht_1(mht_1_v, 202, "", "./tensorflow/core/kernels/bcast_ops.cc", "Compute");

    OP_REQUIRES(
        ctx, ctx->num_inputs() == 2,
        errors::Unimplemented("Broadcast for n-ary operations (n > 2)"));
    gtl::InlinedVector<BCast::Vec, 4> shapes;
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      const Tensor& in = ctx->input(i);
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(in.shape()),
                  errors::InvalidArgument("In[", i, "] must be a vector.",
                                          in.shape().DebugString()));
      BCast::Vec vec;
      for (int64_t i = 0; i < in.NumElements(); ++i) {
        vec.push_back(in.vec<T>()(i));
      }
      shapes.push_back(vec);
    }
    BCast bcast(shapes[0], shapes[1]);
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: [", absl::StrJoin(shapes[0], ","),
                    "] vs. [", absl::StrJoin(shapes[1], ","), "]"));
    Output(ctx, 0, bcast.output_shape());
  }

  bool IsExpensive() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbcast_opsDTcc mht_2(mht_2_v, 229, "", "./tensorflow/core/kernels/bcast_ops.cc", "IsExpensive");
 return false; }

 private:
  void Output(OpKernelContext* ctx, int idx, const BCast::Vec& v) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbcast_opsDTcc mht_3(mht_3_v, 235, "", "./tensorflow/core/kernels/bcast_ops.cc", "Output");

    const int64_t len = v.size();
    Tensor* o = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(idx, TensorShape({len}), &o));
    for (int64_t i = 0; i < len; ++i) {
      o->flat<T>()(i) = static_cast<T>(v[i]);
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(BCastArgsOp);
};

// Given shapes of two tensors, computes the reduction indices for the
// gradient computation.
//
// TODO(zhifengc):
//   1. Adds support for n-ary (n >= 2).
template <typename T>
class BCastGradArgsOp : public OpKernel {
 public:
  explicit BCastGradArgsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbcast_opsDTcc mht_4(mht_4_v, 258, "", "./tensorflow/core/kernels/bcast_ops.cc", "BCastGradArgsOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbcast_opsDTcc mht_5(mht_5_v, 263, "", "./tensorflow/core/kernels/bcast_ops.cc", "Compute");

    OP_REQUIRES(
        ctx, ctx->num_inputs() == 2,
        errors::Unimplemented("Broadcast for n-ary operations (n > 2)"));
    gtl::InlinedVector<BCast::Vec, 4> shapes;
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      const Tensor& in = ctx->input(i);
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(in.shape()),
                  errors::InvalidArgument("In[", i, "] must be a vector.",
                                          in.shape().DebugString()));
      BCast::Vec vec;
      for (int64_t i = 0; i < in.NumElements(); ++i) {
        vec.push_back(in.vec<T>()(i));
      }
      shapes.push_back(vec);
    }
    BCast bcast(shapes[0], shapes[1]);
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: [", absl::StrJoin(shapes[0], ","),
                    "] vs. [", absl::StrJoin(shapes[1], ","), "]"));
    Output(ctx, 0, bcast.grad_x_reduce_idx());
    Output(ctx, 1, bcast.grad_y_reduce_idx());
  }

  bool IsExpensive() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbcast_opsDTcc mht_6(mht_6_v, 291, "", "./tensorflow/core/kernels/bcast_ops.cc", "IsExpensive");
 return false; }

 private:
  void Output(OpKernelContext* ctx, int idx, const BCast::Vec& v) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbcast_opsDTcc mht_7(mht_7_v, 297, "", "./tensorflow/core/kernels/bcast_ops.cc", "Output");

    const int64_t len = v.size();
    Tensor* o = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(idx, TensorShape({len}), &o));
    for (int64_t i = 0; i < len; ++i) {
      o->flat<T>()(i) = static_cast<T>(v[i]);
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(BCastGradArgsOp);
};

REGISTER_KERNEL_BUILDER(Name("BroadcastArgs")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("s0")
                            .HostMemory("s1")
                            .HostMemory("r0"),
                        BCastArgsOp<int32>);
REGISTER_KERNEL_BUILDER(Name("BroadcastArgs")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64_t>("T")
                            .HostMemory("s0")
                            .HostMemory("s1")
                            .HostMemory("r0"),
                        BCastArgsOp<int64_t>);
REGISTER_KERNEL_BUILDER(Name("BroadcastArgs")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int32>("T")
                            .HostMemory("s0")
                            .HostMemory("s1")
                            .HostMemory("r0"),
                        BCastArgsOp<int32>);
REGISTER_KERNEL_BUILDER(Name("BroadcastArgs")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int64_t>("T")
                            .HostMemory("s0")
                            .HostMemory("s1")
                            .HostMemory("r0"),
                        BCastArgsOp<int64_t>);

REGISTER_KERNEL_BUILDER(Name("BroadcastGradientArgs")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("s0")
                            .HostMemory("s1")
                            .HostMemory("r0")
                            .HostMemory("r1"),
                        BCastGradArgsOp<int32>);
REGISTER_KERNEL_BUILDER(Name("BroadcastGradientArgs")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64_t>("T")
                            .HostMemory("s0")
                            .HostMemory("s1")
                            .HostMemory("r0")
                            .HostMemory("r1"),
                        BCastGradArgsOp<int64_t>);
REGISTER_KERNEL_BUILDER(Name("BroadcastGradientArgs")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int32>("T")
                            .HostMemory("s0")
                            .HostMemory("s1")
                            .HostMemory("r0")
                            .HostMemory("r1"),
                        BCastGradArgsOp<int32>);
REGISTER_KERNEL_BUILDER(Name("BroadcastGradientArgs")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int64_t>("T")
                            .HostMemory("s0")
                            .HostMemory("s1")
                            .HostMemory("r0")
                            .HostMemory("r1"),
                        BCastGradArgsOp<int64_t>);

}  // end namespace tensorflow
