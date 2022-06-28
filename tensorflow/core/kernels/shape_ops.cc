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
class MHTracer_DTPStensorflowPScorePSkernelsPSshape_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSshape_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSshape_opsDTcc() {
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

// See docs in ../ops/array_ops.cc.

#include "tensorflow/core/kernels/shape_ops.h"

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

// Shape ----------------------------------------
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int32>("out_type"),
                        ShapeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int64_t>("out_type"),
                        ShapeOp<int64_t>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("Shape")                            \
                              .Device(DEVICE_GPU)                  \
                              .HostMemory("output")                \
                              .TypeConstraint<int32>("out_type")   \
                              .TypeConstraint<type>("T"),          \
                          ShapeOp<int32>);                         \
  REGISTER_KERNEL_BUILDER(Name("Shape")                            \
                              .Device(DEVICE_GPU)                  \
                              .HostMemory("output")                \
                              .TypeConstraint<int64_t>("out_type") \
                              .TypeConstraint<type>("T"),          \
                          ShapeOp<int64_t>);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_bool(REGISTER_GPU_KERNEL);
TF_CALL_variant(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type"),
                        ShapeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64_t>("out_type"),
                        ShapeOp<int64_t>);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_DEFAULT_KERNEL(type)                              \
  REGISTER_KERNEL_BUILDER(Name("Shape")                            \
                              .Device(DEVICE_DEFAULT)              \
                              .HostMemory("output")                \
                              .TypeConstraint<int32>("out_type")   \
                              .TypeConstraint<type>("T"),          \
                          ShapeOp<int32>);                         \
  REGISTER_KERNEL_BUILDER(Name("Shape")                            \
                              .Device(DEVICE_DEFAULT)              \
                              .HostMemory("output")                \
                              .TypeConstraint<int64_t>("out_type") \
                              .TypeConstraint<type>("T"),          \
                          ShapeOp<int64_t>);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_DEFAULT_KERNEL);
TF_CALL_bool(REGISTER_DEFAULT_KERNEL);
TF_CALL_variant(REGISTER_DEFAULT_KERNEL);
#undef REGISTER_DEFAULT_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type"),
                        ShapeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64_t>("out_type"),
                        ShapeOp<int64_t>);

// ShapeN ---------------------------------------
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int32>("out_type"),
                        ShapeNOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int64_t>("out_type"),
                        ShapeNOp<int64_t>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("ShapeN")                           \
                              .Device(DEVICE_GPU)                  \
                              .HostMemory("output")                \
                              .TypeConstraint<int32>("out_type")   \
                              .TypeConstraint<type>("T"),          \
                          ShapeNOp<int32>);                        \
  REGISTER_KERNEL_BUILDER(Name("ShapeN")                           \
                              .Device(DEVICE_GPU)                  \
                              .HostMemory("output")                \
                              .TypeConstraint<int64_t>("out_type") \
                              .TypeConstraint<type>("T"),          \
                          ShapeNOp<int64_t>)

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_bool(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type"),
                        ShapeNOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64_t>("out_type"),
                        ShapeNOp<int64_t>);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_DEFAULT_KERNEL(type)                              \
  REGISTER_KERNEL_BUILDER(Name("ShapeN")                           \
                              .Device(DEVICE_DEFAULT)              \
                              .HostMemory("output")                \
                              .TypeConstraint<int32>("out_type")   \
                              .TypeConstraint<type>("T"),          \
                          ShapeNOp<int32>);                        \
  REGISTER_KERNEL_BUILDER(Name("ShapeN")                           \
                              .Device(DEVICE_DEFAULT)              \
                              .HostMemory("output")                \
                              .TypeConstraint<int64_t>("out_type") \
                              .TypeConstraint<type>("T"),          \
                          ShapeNOp<int64_t>)

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_DEFAULT_KERNEL);
TF_CALL_bool(REGISTER_DEFAULT_KERNEL);
#undef REGISTER_DEFAULT_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type"),
                        ShapeNOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64_t>("out_type"),
                        ShapeNOp<int64_t>);

// Rank ------------------------------------------
REGISTER_KERNEL_BUILDER(Name("Rank").Device(DEVICE_CPU).HostMemory("output"),
                        RankOp);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL(type)                        \
  REGISTER_KERNEL_BUILDER(Name("Rank")                   \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("output"),     \
                          RankOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_variant(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32 and bool.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Rank")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        RankOp);

REGISTER_KERNEL_BUILDER(Name("Rank")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<bool>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        RankOp);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_DEFAULT_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("Rank")                   \
                              .Device(DEVICE_DEFAULT)    \
                              .TypeConstraint<type>("T") \
                              .HostMemory("output"),     \
                          RankOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_DEFAULT_KERNEL);
TF_CALL_variant(REGISTER_DEFAULT_KERNEL);
#undef REGISTER_DEFAULT_KERNEL

// A special GPU kernel for int32 and bool.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Rank")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        RankOp);

REGISTER_KERNEL_BUILDER(Name("Rank")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<bool>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        RankOp);

// Size ------------------------------------------
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int32>("out_type"),
                        SizeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int64_t>("out_type"),
                        SizeOp<int64_t>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("Size")                             \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int32>("out_type")   \
                              .HostMemory("output"),               \
                          SizeOp<int32>);                          \
  REGISTER_KERNEL_BUILDER(Name("Size")                             \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int64_t>("out_type") \
                              .HostMemory("output"),               \
                          SizeOp<int64_t>);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_bool(REGISTER_GPU_KERNEL);
TF_CALL_variant(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SizeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64_t>("out_type")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SizeOp<int64_t>);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_DEFAULT_KERNEL(type)                              \
  REGISTER_KERNEL_BUILDER(Name("Size")                             \
                              .Device(DEVICE_DEFAULT)              \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int32>("out_type")   \
                              .HostMemory("output"),               \
                          SizeOp<int32>);                          \
  REGISTER_KERNEL_BUILDER(Name("Size")                             \
                              .Device(DEVICE_DEFAULT)              \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int64_t>("out_type") \
                              .HostMemory("output"),               \
                          SizeOp<int64_t>);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_DEFAULT_KERNEL);
TF_CALL_bool(REGISTER_DEFAULT_KERNEL);
TF_CALL_variant(REGISTER_DEFAULT_KERNEL);
#undef REGISTER_DEFAULT_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SizeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64_t>("out_type")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SizeOp<int64_t>);

// ExpandDims ------------------------------------
REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_CPU)
                            .HostMemory("dim")
                            .TypeConstraint<int32>("Tdim"),
                        ExpandDimsOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_CPU)
                            .HostMemory("dim")
                            .TypeConstraint<int64_t>("Tdim"),
                        ExpandDimsOp<int64_t>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL(type)                              \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")                   \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<type>("T")       \
                              .TypeConstraint<int32>("Tdim")   \
                              .HostMemory("dim"),              \
                          ExpandDimsOp<int32>);                \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")                   \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<type>("T")       \
                              .TypeConstraint<int64_t>("Tdim") \
                              .HostMemory("dim"),              \
                          ExpandDimsOp<int64_t>);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_bool(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tdim")
                            .HostMemory("input")
                            .HostMemory("dim")
                            .HostMemory("output"),
                        ExpandDimsOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64_t>("Tdim")
                            .HostMemory("input")
                            .HostMemory("dim")
                            .HostMemory("output"),
                        ExpandDimsOp<int64_t>);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_DEFAULT_KERNEL(type)                          \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")                   \
                              .Device(DEVICE_DEFAULT)          \
                              .TypeConstraint<type>("T")       \
                              .TypeConstraint<int32>("Tdim")   \
                              .HostMemory("dim"),              \
                          ExpandDimsOp<int32>);                \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")                   \
                              .Device(DEVICE_DEFAULT)          \
                              .TypeConstraint<type>("T")       \
                              .TypeConstraint<int64_t>("Tdim") \
                              .HostMemory("dim"),              \
                          ExpandDimsOp<int64_t>);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_DEFAULT_KERNEL);
TF_CALL_bool(REGISTER_DEFAULT_KERNEL);
#undef REGISTER_DEFAULT_KERNEL

REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tdim")
                            .HostMemory("input")
                            .HostMemory("dim")
                            .HostMemory("output"),
                        ExpandDimsOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64_t>("Tdim")
                            .HostMemory("input")
                            .HostMemory("dim")
                            .HostMemory("output"),
                        ExpandDimsOp<int64_t>);

// Squeeze ---------------------------------------
REGISTER_KERNEL_BUILDER(Name("Squeeze").Device(DEVICE_CPU), SqueezeOp);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL(type)                                   \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Squeeze").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SqueezeOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_bool(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Squeeze")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SqueezeOp);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_DEFAULT_KERNEL(type)                                   \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("Squeeze").Device(DEVICE_DEFAULT).TypeConstraint<type>("T"), \
      SqueezeOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_DEFAULT_KERNEL);
TF_CALL_bool(REGISTER_DEFAULT_KERNEL);
#undef REGISTER_DEFAULT_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Squeeze")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SqueezeOp);

class EnsureShapeOp : public OpKernel {
 public:
  explicit EnsureShapeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSshape_opsDTcc mht_0(mht_0_v, 642, "", "./tensorflow/core/kernels/shape_ops.cc", "EnsureShapeOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &expected_shape_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSshape_opsDTcc mht_1(mht_1_v, 649, "", "./tensorflow/core/kernels/shape_ops.cc", "Compute");

    TensorShape shape;
    OP_REQUIRES_OK(ctx, shape_op_helpers::GetShape(ctx, 0, &shape));

    if (!expected_shape_.IsCompatibleWith(shape)) {
      ctx->SetStatus(errors::InvalidArgument(
          "Shape of tensor ", this->def().input(0), " ", shape.DebugString(),
          " is not compatible with expected shape ",
          expected_shape_.DebugString(), "."));
    }

    // If shape matches, outputs the tensor.
    if (IsRefType(ctx->input_dtype(0))) {
      ctx->forward_ref_input_to_ref_output(0, 0);
    } else {
      ctx->set_output(0, ctx->input(0));
    }
  }

  bool IsExpensive() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSshape_opsDTcc mht_2(mht_2_v, 671, "", "./tensorflow/core/kernels/shape_ops.cc", "IsExpensive");
 return false; }

 private:
  PartialTensorShape expected_shape_;
};

// NOTE(rachelim): The kernel registrations for EnsureShapeOp are identical to
// those of the identity op, since the ops have the same device type
// constraints.
REGISTER_KERNEL_BUILDER(Name("EnsureShape").Device(DEVICE_CPU), EnsureShapeOp);

#define REGISTER_DEVICE_KERNEL(type)                                        \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("EnsureShape").Device(DEVICE_DEFAULT).TypeConstraint<type>("T"), \
      EnsureShapeOp)

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_DEVICE_KERNEL);
REGISTER_DEVICE_KERNEL(Variant);

#undef REGISTER_DEVICE_KERNEL

// A special DEVICE_DEFAULT kernel for int32 and bool.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
#define REGISTER_DEVICE_HOST_KERNEL(type)                 \
  REGISTER_KERNEL_BUILDER(Name("EnsureShape")             \
                              .Device(DEVICE_DEFAULT)     \
                              .HostMemory("input")        \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          EnsureShapeOp)

REGISTER_DEVICE_HOST_KERNEL(int32);
REGISTER_DEVICE_HOST_KERNEL(bool);
REGISTER_DEVICE_HOST_KERNEL(tstring);
REGISTER_DEVICE_HOST_KERNEL(ResourceHandle);

#undef REGISTER_DEVICE_HOST_KERNEL

}  // namespace tensorflow
