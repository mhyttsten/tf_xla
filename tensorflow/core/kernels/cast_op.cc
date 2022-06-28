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
class MHTracer_DTPStensorflowPScorePSkernelsPScast_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScast_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScast_opDTcc() {
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

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/cast_op.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

#include "tensorflow/core/kernels/cast_op_impl.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#define CURRY_TYPES2(FN, arg0)   \
  FN(arg0, bool);                \
  FN(arg0, uint8);               \
  FN(arg0, uint16);              \
  FN(arg0, uint32);              \
  FN(arg0, uint64);              \
  FN(arg0, int8);                \
  FN(arg0, int16);               \
  FN(arg0, int32);               \
  FN(arg0, int64_t);             \
  FN(arg0, Eigen::half);         \
  FN(arg0, float);               \
  FN(arg0, double);              \
  FN(arg0, std::complex<float>); \
  FN(arg0, std::complex<double>)

CastOpBase::CastOpBase(OpKernelConstruction* ctx) : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("SrcT", &external_src_dtype_));

  OP_REQUIRES_OK(ctx, ctx->GetAttr("DstT", &external_dst_dtype_));

  OP_REQUIRES_OK(ctx, ctx->GetAttr("Truncate", &use_truncation_));

  // Quantized data types use the same underlying format as their non quantized
  // version so we use the non quantized implementation for casting.
  if (external_dst_dtype_ == DT_QUINT8) {
    dst_dtype_ = DT_UINT8;
  } else if (external_dst_dtype_ == DT_QINT8) {
    dst_dtype_ = DT_INT8;
  } else if (external_dst_dtype_ == DT_QINT32) {
    dst_dtype_ = DT_INT32;
  } else if (external_dst_dtype_ == DT_QINT16) {
    dst_dtype_ = DT_INT16;
  } else if (external_dst_dtype_ == DT_QUINT16) {
    dst_dtype_ = DT_UINT16;
  } else {
    dst_dtype_ = external_dst_dtype_;
  }

  if (external_src_dtype_ == DT_QUINT8) {
    src_dtype_ = DT_UINT8;
  } else if (external_src_dtype_ == DT_QINT8) {
    src_dtype_ = DT_INT8;
  } else if (external_src_dtype_ == DT_QINT32) {
    src_dtype_ = DT_INT32;
  } else if (external_src_dtype_ == DT_QINT16) {
    src_dtype_ = DT_INT16;
  } else if (external_src_dtype_ == DT_QUINT16) {
    src_dtype_ = DT_UINT16;
  } else {
    src_dtype_ = external_src_dtype_;
  }
}

void CastOpBase::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScast_opDTcc mht_0(mht_0_v, 261, "", "./tensorflow/core/kernels/cast_op.cc", "CastOpBase::Compute");

  const Tensor& inp = ctx->input(0);
  if (work_ == nullptr) {
    ctx->set_output(0, inp);
  } else if (external_src_dtype_ != src_dtype_ ||
             external_dst_dtype_ != dst_dtype_) {
    Tensor in;
    // If the type is a quantized type we need to do a bitcast since the
    // src_dtype_ is different from external_src_type_.
    OP_REQUIRES_OK(ctx, in.BitcastFrom(inp, src_dtype_, inp.shape()));
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in.shape(), &out));
    out->set_dtype(dst_dtype_);
    work_(ctx, in, out, use_truncation_);
    out->set_dtype(external_dst_dtype_);
  } else {
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, inp.shape(), &out));
    work_(ctx, inp, out, use_truncation_);
  }
}

Status CastOpBase::Unimplemented() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScast_opDTcc mht_1(mht_1_v, 286, "", "./tensorflow/core/kernels/cast_op.cc", "CastOpBase::Unimplemented");

  return errors::Unimplemented("Cast ", DataTypeString(external_src_dtype_),
                               " to ", DataTypeString(external_dst_dtype_),
                               " is not supported");
}

CpuCastOp::CpuCastOp(OpKernelConstruction* ctx) : CastOpBase(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScast_opDTcc mht_2(mht_2_v, 295, "", "./tensorflow/core/kernels/cast_op.cc", "CpuCastOp::CpuCastOp");

  OP_REQUIRES_OK(ctx, Prepare());
}

Status CpuCastOp::Prepare() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScast_opDTcc mht_3(mht_3_v, 302, "", "./tensorflow/core/kernels/cast_op.cc", "CpuCastOp::Prepare");

  if (external_src_dtype_ == external_dst_dtype_) {
    work_ = nullptr;  // Identity
    return Status::OK();
  }
  if (src_dtype_ == DT_BOOL) {
    work_ = GetCpuCastFromBool(dst_dtype_);
  } else if (src_dtype_ == DT_UINT8) {
    work_ = GetCpuCastFromUint8(dst_dtype_);
  } else if (src_dtype_ == DT_UINT16) {
    work_ = GetCpuCastFromUint16(dst_dtype_);
  } else if (src_dtype_ == DT_UINT32) {
    work_ = GetCpuCastFromUint32(dst_dtype_);
  } else if (src_dtype_ == DT_UINT64) {
    work_ = GetCpuCastFromUint64(dst_dtype_);
  } else if (src_dtype_ == DT_INT8) {
    work_ = GetCpuCastFromInt8(dst_dtype_);
  } else if (src_dtype_ == DT_INT16) {
    work_ = GetCpuCastFromInt16(dst_dtype_);
  } else if (src_dtype_ == DT_INT32) {
    work_ = GetCpuCastFromInt32(dst_dtype_);
  } else if (src_dtype_ == DT_INT64) {
    work_ = GetCpuCastFromInt64(dst_dtype_);
  } else if (src_dtype_ == DT_HALF) {
    work_ = GetCpuCastFromHalf(dst_dtype_);
  } else if (src_dtype_ == DT_FLOAT) {
    work_ = GetCpuCastFromFloat(dst_dtype_);
  } else if (src_dtype_ == DT_DOUBLE) {
    work_ = GetCpuCastFromDouble(dst_dtype_);
  } else if (src_dtype_ == DT_COMPLEX64) {
    work_ = GetCpuCastFromComplex64(dst_dtype_);
  } else if (src_dtype_ == DT_COMPLEX128) {
    work_ = GetCpuCastFromComplex128(dst_dtype_);
  } else if (src_dtype_ == DT_BFLOAT16) {
    work_ = GetCpuCastFromBfloat(dst_dtype_);
  }

  // TODO(sesse): If CPU casting to or from Eigen::half ever becomes a
  // bottleneck, we could probably implement specialized support for
  // vectorized versions (not the least based on F16C for Haswell
  // or newer).

  return work_ == nullptr ? Unimplemented() : Status::OK();
}

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
class GpuCastOp : public CastOpBase {
 public:
  explicit GpuCastOp(OpKernelConstruction* ctx) : CastOpBase(ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScast_opDTcc mht_4(mht_4_v, 354, "", "./tensorflow/core/kernels/cast_op.cc", "GpuCastOp");

    OP_REQUIRES_OK(ctx, Prepare());
  }

 private:
  Status Prepare() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScast_opDTcc mht_5(mht_5_v, 362, "", "./tensorflow/core/kernels/cast_op.cc", "Prepare");

    if (external_src_dtype_ == external_dst_dtype_) {
      work_ = nullptr;  // Identity
      return Status::OK();
    }
    if (src_dtype_ == DT_BOOL) {
      work_ = GetGpuCastFromBool(dst_dtype_);
    } else if (src_dtype_ == DT_UINT8) {
      work_ = GetGpuCastFromUint8(dst_dtype_);
    } else if (src_dtype_ == DT_UINT16) {
      work_ = GetGpuCastFromUint16(dst_dtype_);
    } else if (src_dtype_ == DT_UINT32) {
      work_ = GetGpuCastFromUint32(dst_dtype_);
    } else if (src_dtype_ == DT_UINT64) {
      work_ = GetGpuCastFromUint64(dst_dtype_);
    } else if (src_dtype_ == DT_INT8) {
      work_ = GetGpuCastFromInt8(dst_dtype_);
    } else if (src_dtype_ == DT_INT16) {
      work_ = GetGpuCastFromInt16(dst_dtype_);
    } else if (src_dtype_ == DT_INT32) {
      work_ = GetGpuCastFromInt32(dst_dtype_);
    } else if (src_dtype_ == DT_INT64) {
      work_ = GetGpuCastFromInt64(dst_dtype_);
    } else if (src_dtype_ == DT_HALF) {
      work_ = GetGpuCastFromHalf(dst_dtype_);
    } else if (src_dtype_ == DT_FLOAT) {
      work_ = GetGpuCastFromFloat(dst_dtype_);
    } else if (src_dtype_ == DT_DOUBLE) {
      work_ = GetGpuCastFromDouble(dst_dtype_);
    } else if (src_dtype_ == DT_COMPLEX64) {
      work_ = GetGpuCastFromComplex64(dst_dtype_);
    } else if (src_dtype_ == DT_COMPLEX128) {
      work_ = GetGpuCastFromComplex128(dst_dtype_);
    } else if (src_dtype_ == DT_BFLOAT16) {
      work_ = GetGpuCastFromBfloat(dst_dtype_);
    }

    return work_ == nullptr ? Unimplemented() : Status::OK();
  }
};
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef CAST_CASE

REGISTER_KERNEL_BUILDER(Name("Cast").Device(DEVICE_CPU), CpuCastOp);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#define REGISTER_CAST_GPU(srctype, dsttype)                    \
  REGISTER_KERNEL_BUILDER(Name("Cast")                         \
                              .TypeConstraint<srctype>("SrcT") \
                              .TypeConstraint<dsttype>("DstT") \
                              .Device(DEVICE_GPU),             \
                          GpuCastOp)

#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
CURRY_TYPES2(REGISTER_CAST_GPU, bool);
CURRY_TYPES2(REGISTER_CAST_GPU, int8);
CURRY_TYPES2(REGISTER_CAST_GPU, int16);
CURRY_TYPES2(REGISTER_CAST_GPU, int32);
CURRY_TYPES2(REGISTER_CAST_GPU, int64);
CURRY_TYPES2(REGISTER_CAST_GPU, uint8);
CURRY_TYPES2(REGISTER_CAST_GPU, uint16);
CURRY_TYPES2(REGISTER_CAST_GPU, uint32);
CURRY_TYPES2(REGISTER_CAST_GPU, uint64);
CURRY_TYPES2(REGISTER_CAST_GPU, Eigen::half);
CURRY_TYPES2(REGISTER_CAST_GPU, float);
CURRY_TYPES2(REGISTER_CAST_GPU, double);
#else

#define CURRY_SUBSET_OF_TYPES(FN, arg0) \
  FN(arg0, std::complex<float>);        \
  FN(arg0, std::complex<double>)

CURRY_SUBSET_OF_TYPES(REGISTER_CAST_GPU, bool);
CURRY_SUBSET_OF_TYPES(REGISTER_CAST_GPU, int8);
CURRY_SUBSET_OF_TYPES(REGISTER_CAST_GPU, int16);
CURRY_SUBSET_OF_TYPES(REGISTER_CAST_GPU, int32);
CURRY_SUBSET_OF_TYPES(REGISTER_CAST_GPU, int64_t);
CURRY_SUBSET_OF_TYPES(REGISTER_CAST_GPU, uint8);
CURRY_SUBSET_OF_TYPES(REGISTER_CAST_GPU, uint16);
CURRY_SUBSET_OF_TYPES(REGISTER_CAST_GPU, uint32);
CURRY_SUBSET_OF_TYPES(REGISTER_CAST_GPU, uint64);
CURRY_SUBSET_OF_TYPES(REGISTER_CAST_GPU, Eigen::half);
CURRY_SUBSET_OF_TYPES(REGISTER_CAST_GPU, float);
CURRY_SUBSET_OF_TYPES(REGISTER_CAST_GPU, double);

#undef CURRY_SUBSET_OF_TYPES

#endif

CURRY_TYPES2(REGISTER_CAST_GPU, std::complex<float>);
CURRY_TYPES2(REGISTER_CAST_GPU, std::complex<double>);
REGISTER_CAST_GPU(float, bfloat16);
REGISTER_CAST_GPU(bfloat16, float);

#undef REGISTER_CAST_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM


#undef CURRY_TYPES2

// HostCast differs from Cast in that its input and output are in host memory.
REGISTER_KERNEL_BUILDER(Name("_HostCast").Device(DEVICE_CPU), CpuCastOp);
REGISTER_KERNEL_BUILDER(
    Name("_HostCast").Device(DEVICE_DEFAULT).HostMemory("x").HostMemory("y"),
    CpuCastOp);
}  // end namespace tensorflow
