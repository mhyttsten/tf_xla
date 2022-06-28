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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPSimage_resize_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSimage_resize_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPSimage_resize_opsDTcc() {
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

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {

class TpuCustomResizeOp : public XlaOpKernel {
 public:
  explicit TpuCustomResizeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSimage_resize_opsDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/tpu/kernels/image_resize_ops.cc", "TpuCustomResizeOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("align_corners", &align_corners_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("half_pixel_centers", &half_pixel_centers_));
  }

  xla::Shape GetOutputShape(XlaOpKernelContext* ctx) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSimage_resize_opsDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/tpu/kernels/image_resize_ops.cc", "GetOutputShape");

    std::vector<int64_t> out_size;
    auto status = ctx->ConstantInputAsIntVector(1, &out_size);
    CHECK_EQ(out_size.size(), 2) << status.ToString();
    xla::Shape output_shape =
        TensorShapeToXLAShape(ctx->output_xla_type(0), ctx->InputShape(0));
    output_shape.mutable_dimensions()[1] = out_size[0];
    output_shape.mutable_dimensions()[2] = out_size[1];
    return output_shape;
  }

  string OpaqueField() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSimage_resize_opsDTcc mht_2(mht_2_v, 225, "", "./tensorflow/core/tpu/kernels/image_resize_ops.cc", "OpaqueField");

    return absl::StrCat("\"", align_corners_, half_pixel_centers_, "\"");
  }

  void CompileGrad(XlaOpKernelContext* ctx, const char* target,
                   const xla::Shape& output_shape) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("target: \"" + (target == nullptr ? std::string("nullptr") : std::string((char*)target)) + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSimage_resize_opsDTcc mht_3(mht_3_v, 234, "", "./tensorflow/core/tpu/kernels/image_resize_ops.cc", "CompileGrad");

    auto input_shape =
        TensorShapeToXLAShape(ctx->output_xla_type(0), ctx->InputShape(0));
    if (ctx->InputShape(1).dim_sizes() == ctx->InputShape(0).dim_sizes()) {
      ctx->SetOutput(
          0, xla::ConvertElementType(ctx->Input(0), ctx->output_xla_type(0)));
      return;
    }
    // The gradient should be done in two phases for large resizes.
    auto input = ctx->Input(0);
    if (input_shape.dimensions(1) / output_shape.dimensions(1) > 3 &&
        input_shape.dimensions(2) / output_shape.dimensions(2) > 3) {
      auto intermediate_shape = output_shape;
      intermediate_shape.mutable_dimensions()[1] = input_shape.dimensions(1);
      input = xla::CustomCall(ctx->builder(), target, {ctx->Input(0)},
                              intermediate_shape, OpaqueField());
    }
    ctx->SetOutput(0, xla::CustomCall(ctx->builder(), target, {input},
                                      output_shape, OpaqueField()));
  }

  void CompileForward(XlaOpKernelContext* ctx, const char* target) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("target: \"" + (target == nullptr ? std::string("nullptr") : std::string((char*)target)) + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSimage_resize_opsDTcc mht_4(mht_4_v, 259, "", "./tensorflow/core/tpu/kernels/image_resize_ops.cc", "CompileForward");

    auto output_shape = GetOutputShape(ctx);
    if (ctx->InputShape(0).dim_size(1) == output_shape.dimensions(1) &&
        ctx->InputShape(0).dim_size(2) == output_shape.dimensions(2)) {
      ctx->SetOutput(
          0, xla::ConvertElementType(ctx->Input(0), ctx->output_xla_type(0)));
      return;
    }
    if (ctx->InputShape(0).dim_size(1) == 1 &&
        ctx->InputShape(0).dim_size(2) == 1) {
      ctx->SetOutput(0,
                     ctx->Input(0) + xla::Zeros(ctx->builder(), output_shape));
      return;
    }
    ctx->SetOutput(0, xla::CustomCall(ctx->builder(), target, {ctx->Input(0)},
                                      output_shape, OpaqueField()));
  }

 private:
  bool align_corners_;
  bool half_pixel_centers_;
};

class TpuResizeNearestNeighborOp : public TpuCustomResizeOp {
 public:
  explicit TpuResizeNearestNeighborOp(OpKernelConstruction* ctx)
      : TpuCustomResizeOp(ctx) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSimage_resize_opsDTcc mht_5(mht_5_v, 288, "", "./tensorflow/core/tpu/kernels/image_resize_ops.cc", "TpuResizeNearestNeighborOp");
}
  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSimage_resize_opsDTcc mht_6(mht_6_v, 292, "", "./tensorflow/core/tpu/kernels/image_resize_ops.cc", "Compile");

    CompileForward(ctx, "ResizeNearest");
  }
};

class TpuResizeBilinearOp : public TpuCustomResizeOp {
 public:
  explicit TpuResizeBilinearOp(OpKernelConstruction* ctx)
      : TpuCustomResizeOp(ctx) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSimage_resize_opsDTcc mht_7(mht_7_v, 303, "", "./tensorflow/core/tpu/kernels/image_resize_ops.cc", "TpuResizeBilinearOp");
}
  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSimage_resize_opsDTcc mht_8(mht_8_v, 307, "", "./tensorflow/core/tpu/kernels/image_resize_ops.cc", "Compile");

    CompileForward(ctx, "ResizeBilinear");
  }
};

class TpuResizeNearestNeighborGradOp : public TpuCustomResizeOp {
 public:
  explicit TpuResizeNearestNeighborGradOp(OpKernelConstruction* ctx)
      : TpuCustomResizeOp(ctx) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSimage_resize_opsDTcc mht_9(mht_9_v, 318, "", "./tensorflow/core/tpu/kernels/image_resize_ops.cc", "TpuResizeNearestNeighborGradOp");
}
  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSimage_resize_opsDTcc mht_10(mht_10_v, 322, "", "./tensorflow/core/tpu/kernels/image_resize_ops.cc", "Compile");

    CompileGrad(ctx, "ResizeNearestGrad", GetOutputShape(ctx));
  }
};

class TpuResizeBilinearGradOp : public TpuCustomResizeOp {
 public:
  explicit TpuResizeBilinearGradOp(OpKernelConstruction* ctx)
      : TpuCustomResizeOp(ctx) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSimage_resize_opsDTcc mht_11(mht_11_v, 333, "", "./tensorflow/core/tpu/kernels/image_resize_ops.cc", "TpuResizeBilinearGradOp");
}
  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSimage_resize_opsDTcc mht_12(mht_12_v, 337, "", "./tensorflow/core/tpu/kernels/image_resize_ops.cc", "Compile");

    auto output_shape =
        TensorShapeToXLAShape(ctx->output_xla_type(0), ctx->InputShape(1));
    CompileGrad(ctx, "ResizeBilinearGrad", output_shape);
  }
};

REGISTER_XLA_OP(Name("ResizeNearestNeighbor")
                    .CompileTimeConstantInput("size")
                    .Device(DEVICE_TPU_XLA_JIT),
                TpuResizeNearestNeighborOp);

REGISTER_XLA_OP(Name("ResizeNearestNeighborGrad")
                    .CompileTimeConstantInput("size")
                    .Device(DEVICE_TPU_XLA_JIT),
                TpuResizeNearestNeighborGradOp);

REGISTER_XLA_OP(Name("ResizeBilinear")
                    .CompileTimeConstantInput("size")
                    .Device(DEVICE_TPU_XLA_JIT),
                TpuResizeBilinearOp);

REGISTER_XLA_OP(Name("ResizeBilinearGrad").Device(DEVICE_TPU_XLA_JIT),
                TpuResizeBilinearGradOp);

}  // namespace tensorflow
