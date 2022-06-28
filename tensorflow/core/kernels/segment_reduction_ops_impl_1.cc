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
class MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_impl_1DTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_impl_1DTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_impl_1DTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/segment_reduction_ops_impl.h"

namespace tensorflow {
namespace internal {
// Static routines not in the templated class to reduce code size
Status ValidateSegmentReduction(OpKernelContext* context, const Tensor& input,
                                const Tensor& segment_ids) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_impl_1DTcc mht_0(mht_0_v, 192, "", "./tensorflow/core/kernels/segment_reduction_ops_impl_1.cc", "ValidateSegmentReduction");

  if (!TensorShapeUtils::IsVectorOrHigher(input.shape())) {
    return errors::InvalidArgument("input must be at least rank 1");
  }
  if (!TensorShapeUtils::IsVector(segment_ids.shape())) {
    return errors::InvalidArgument("segment_ids should be a vector.");
  }
  const int64_t num_indices = segment_ids.NumElements();
  if (num_indices != input.dim_size(0)) {
    return errors::InvalidArgument(
        "segment_ids should be the same size as dimension 0 of"
        " input.");
  }

  return Status::OK();
}

// check routines not in the templated class to reduce code size
Status ValidateUnsortedSegmentReduction(OpKernel* op_kernel,
                                        OpKernelContext* context,
                                        const Tensor& data,
                                        const Tensor& segment_ids,
                                        const Tensor& num_segments) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_impl_1DTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/kernels/segment_reduction_ops_impl_1.cc", "ValidateUnsortedSegmentReduction");

  if (!TensorShapeUtils::IsScalar(num_segments.shape())) {
    return errors::InvalidArgument(
        "num_segments should be a scalar, not shape ",
        num_segments.shape().DebugString());
  }

  if (!TensorShapeUtils::StartsWith(data.shape(), segment_ids.shape())) {
    return errors::InvalidArgument("data.shape = ", data.shape().DebugString(),
                                   " does not start with segment_ids.shape = ",
                                   segment_ids.shape().DebugString());
  }

  return Status::OK();
}

Status ValidateSparseSegmentReduction(OpKernelContext* context,
                                      const Tensor& input,
                                      const Tensor& indices,
                                      const Tensor& segment_ids,
                                      bool has_num_segments) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_impl_1DTcc mht_2(mht_2_v, 240, "", "./tensorflow/core/kernels/segment_reduction_ops_impl_1.cc", "ValidateSparseSegmentReduction");

  if (has_num_segments) {
    const Tensor& num_segments_t = context->input(3);
    if (!TensorShapeUtils::IsScalar(num_segments_t.shape())) {
      return errors::InvalidArgument(
          "num_segments should be a scalar, not shape ",
          num_segments_t.shape().DebugString());
    }
    int64_t output_rows =
        internal::SubtleMustCopy(num_segments_t.dtype() == DT_INT32
                                     ? num_segments_t.scalar<int32>()()
                                     : num_segments_t.scalar<int64_t>()());
    if (output_rows < 0) {
      return errors::InvalidArgument("segment ids must be >= 0");
    }
  }

  if (!TensorShapeUtils::IsVector(indices.shape())) {
    return errors::InvalidArgument("indices should be a vector.");
  }

  if (!TensorShapeUtils::IsVector(segment_ids.shape())) {
    return errors::InvalidArgument("segment_ids should be a vector.");
  }

  const int64_t num_indices = indices.NumElements();
  if (num_indices != segment_ids.NumElements()) {
    return errors::InvalidArgument(
        "segment_ids and indices should have same size.");
  }

  if (input.dims() < 1) {
    return errors::InvalidArgument("Shape must be at least rank 1");
  }

  return Status::OK();
}

}  // namespace internal

#define REGISTER_CPU_KERNEL_SEGMENT(name, functor, type, index_type, \
                                    default_value)                   \
  REGISTER_KERNEL_BUILDER(                                           \
      Name(name)                                                     \
          .Device(DEVICE_CPU)                                        \
          .TypeConstraint<type>("T")                                 \
          .TypeConstraint<index_type>("Tindices"),                   \
      SegmentReductionOp<CPUDevice, type, index_type, functor, default_value>)

#define REGISTER_REAL_CPU_KERNELS(type, index_type)                            \
  REGISTER_CPU_KERNEL_SEGMENT("SegmentSum", Eigen::internal::SumReducer<type>, \
                              type, index_type, 0);                            \
  REGISTER_CPU_KERNEL_SEGMENT(                                                 \
      "SegmentMean", Eigen::internal::MeanReducer<type>, type, index_type, 0); \
  REGISTER_CPU_KERNEL_SEGMENT(                                                 \
      "SegmentProd", Eigen::internal::ProdReducer<type>, type, index_type, 1); \
  REGISTER_CPU_KERNEL_SEGMENT("SegmentMin", Eigen::internal::MinReducer<type>, \
                              type, index_type, 0);                            \
  REGISTER_CPU_KERNEL_SEGMENT("SegmentMax", Eigen::internal::MaxReducer<type>, \
                              type, index_type, 0)

#define REGISTER_COMPLEX_CPU_KERNELS(type, index_type)                         \
  REGISTER_CPU_KERNEL_SEGMENT("SegmentSum", Eigen::internal::SumReducer<type>, \
                              type, index_type, 0);                            \
  REGISTER_CPU_KERNEL_SEGMENT(                                                 \
      "SegmentMean", Eigen::internal::MeanReducer<type>, type, index_type, 0); \
  REGISTER_CPU_KERNEL_SEGMENT(                                                 \
      "SegmentProd", Eigen::internal::ProdReducer<type>, type, index_type, 1);

#define REGISTER_REAL_CPU_KERNELS_ALL(type) \
  REGISTER_REAL_CPU_KERNELS(type, int32)

#define REGISTER_COMPLEX_CPU_KERNELS_ALL(type) \
  REGISTER_COMPLEX_CPU_KERNELS(type, int32)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_REAL_CPU_KERNELS_ALL);
REGISTER_COMPLEX_CPU_KERNELS_ALL(complex64);
REGISTER_COMPLEX_CPU_KERNELS_ALL(complex128);
#undef REGISTER_CPU_KERNEL_SEGMENT
#undef REGISTER_REAL_CPU_KERNELS
#undef REGISTER_COMPLEX_CPU_KERNELS
#undef REGISTER_REAL_CPU_KERNELS_ALL
#undef REGISTER_COMPLEX_CPU_KERNELS_ALL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL_SORTEDSEGMENT(                            \
    name, type, index_type, initial_value_functor,                    \
    empty_segment_value_functor, reduction_kernel_functor, is_mean)   \
  REGISTER_KERNEL_BUILDER(                                            \
      Name(name)                                                      \
          .Device(DEVICE_GPU)                                         \
          .TypeConstraint<type>("T")                                  \
          .TypeConstraint<index_type>("Tindices"),                    \
      SegmentReductionGPUOp<                                          \
          type, index_type,                                           \
          functor::SegmentReductionFunctor<                           \
              type, index_type, initial_value_functor,                \
              empty_segment_value_functor, reduction_kernel_functor>, \
          is_mean>)

#define REGISTER_GPU_SORTED_KERNELS(type, index_type)                         \
  REGISTER_GPU_KERNEL_SORTEDSEGMENT("SegmentSum", type, index_type,           \
                                    functor::Zero<type>, functor::Zero<type>, \
                                    functor::Sum, /*is_mean=*/false);         \
  REGISTER_GPU_KERNEL_SORTEDSEGMENT("SegmentMean", type, index_type,          \
                                    functor::Zero<type>, functor::Zero<type>, \
                                    functor::Sum, /*is_mean=*/true);          \
  REGISTER_GPU_KERNEL_SORTEDSEGMENT("SegmentProd", type, index_type,          \
                                    functor::One<type>, functor::One<type>,   \
                                    functor::Prod, /*is_mean=*/false);        \
  REGISTER_GPU_KERNEL_SORTEDSEGMENT(                                          \
      "SegmentMin", type, index_type, functor::Highest<type>,                 \
      functor::Zero<type>, functor::Min, /*is_mean=*/false);                  \
  REGISTER_GPU_KERNEL_SORTEDSEGMENT(                                          \
      "SegmentMax", type, index_type, functor::Lowest<type>,                  \
      functor::Zero<type>, functor::Max, /*is_mean=*/false);

#define REGISTER_GPU_SORTED_KERNELS_ALL(type) \
  REGISTER_GPU_SORTED_KERNELS(type, int32)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_SORTED_KERNELS_ALL);
#undef REGISTER_GPU_KERNEL_SORTEDSEGMENT
#undef REGISTER_GPU_SORTED_KERNELS
#undef REGISTER_GPU_SORTED_KERNELS_ALL
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
