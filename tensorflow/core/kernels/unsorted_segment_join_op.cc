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
class MHTracer_DTPStensorflowPScorePSkernelsPSunsorted_segment_join_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSunsorted_segment_join_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSunsorted_segment_join_opDTcc() {
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

// See docs in ../ops/string_ops.cc.

#include <string>
#include <utility>

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

namespace {

template <typename INDICES_TYPE>
gtl::InlinedVector<INDICES_TYPE, 8> GetFlattenedRelativeOffsets(
    INDICES_TYPE small_stride, INDICES_TYPE big_stride) {
  gtl::InlinedVector<INDICES_TYPE, 8> flattened_offsets(small_stride);
  for (auto i = 0; i < small_stride; i++) {
    flattened_offsets[i] = i * big_stride;
  }
  return flattened_offsets;
}

template <typename INDICES_TYPE>
std::pair<INDICES_TYPE, INDICES_TYPE> GetStrides(
    const TensorShape& input_shape, const TensorShape& segment_id_shape) {
  int64_t small_stride = 1;
  int64_t big_stride = 1;
  for (auto i = 0; i < input_shape.dims(); i++) {
    if (i < segment_id_shape.dims()) {
      small_stride *= segment_id_shape.dim_size(i);
    } else {
      big_stride *= input_shape.dim_size(i);
    }
  }
  return std::make_pair(big_stride, small_stride);
}

TensorShape GetOutputShape(const TensorShape& input_shape,
                           const TensorShape& segment_id_shape,
                           const int64_t num_segments) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSunsorted_segment_join_opDTcc mht_0(mht_0_v, 231, "", "./tensorflow/core/kernels/unsorted_segment_join_op.cc", "GetOutputShape");

  TensorShape output_shape;
  output_shape.AddDim(num_segments);
  for (size_t index = segment_id_shape.dims(); index < input_shape.dims();
       ++index) {
    output_shape.AddDim(input_shape.dim_size(index));
  }
  return output_shape;
}

}  // namespace

template <typename INDICES_TYPE, typename NUM_SEGMENTS_TYPE>
class UnsortedSegmentJoinOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  explicit UnsortedSegmentJoinOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSunsorted_segment_join_opDTcc mht_1(mht_1_v, 251, "", "./tensorflow/core/kernels/unsorted_segment_join_op.cc", "UnsortedSegmentJoinOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("separator", &separator_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSunsorted_segment_join_opDTcc mht_2(mht_2_v, 258, "", "./tensorflow/core/kernels/unsorted_segment_join_op.cc", "Compute");

    const Tensor& input = context->input(0);
    const TensorShape& input_shape = input.shape();
    const int32_t input_dims = input_shape.dims();

    const Tensor& segment_id = context->input(1);
    const TensorShape& segment_id_shape = segment_id.shape();
    const int32_t segment_dims = segment_id_shape.dims();

    const Tensor& num_segments_tensor = context->input(2);
    OP_REQUIRES(context, num_segments_tensor.NumElements() != 0,
                errors::InvalidArgument("Number of segments cannot be empty."));
    OP_REQUIRES(context,
                TensorShapeUtils::IsScalar(num_segments_tensor.shape()),
                errors::InvalidArgument("Number of segments must be a scalar"));

    auto num_segments = num_segments_tensor.scalar<NUM_SEGMENTS_TYPE>()();
    OP_REQUIRES(
        context, num_segments >= 0,
        errors::InvalidArgument(
            "Number of segments must be non-negative but got ", num_segments));

    OP_REQUIRES(context, segment_dims != 0,
                errors::InvalidArgument("Segment_id cannot have rank 0"));

    OP_REQUIRES(
        context, segment_dims <= input_dims,
        errors::OutOfRange("Invalid segment_id rank ", segment_dims,
                           " for input with ", input_dims, " dimension(s)"));
    for (auto i = 0; i < segment_dims; i++) {
      OP_REQUIRES(
          context, segment_id_shape.dim_size(i) == input_shape.dim_size(i),
          errors::InvalidArgument(
              "Segment dimension is ", segment_id_shape.dim_size(i),
              " while input dimension is ", input_dims, " in rank ", i));
    }

    // Making output tensor.
    Tensor* output_tensor = nullptr;
    TensorShape output_shape =
        GetOutputShape(input_shape, segment_id_shape, num_segments);
    OP_REQUIRES_OK(context, context->allocate_output("output", output_shape,
                                                     &output_tensor));

    // Preparating flat tensors.
    auto output_flat = output_tensor->flat<tstring>();
    auto flat_segment_id = segment_id.flat<INDICES_TYPE>();
    auto flat_input = input.flat<tstring>();

    for (int i = 0; i < flat_segment_id.size(); i++) {
      OP_REQUIRES(
          context,
          ((flat_segment_id(i) < num_segments) && (flat_segment_id(i) >= 0)),
          errors::InvalidArgument(
              "segment_ids are not allowed to exceed num_segments or"
              " to have negative values."));
    }

    int64_t big_stride;
    int64_t small_stride;
    std::tie(big_stride, small_stride) =
        GetStrides<INDICES_TYPE>(input_shape, segment_id_shape);
    auto relative_offset_set =
        GetFlattenedRelativeOffsets<INDICES_TYPE>(small_stride, big_stride);
    for (auto start_offset = 0; start_offset < big_stride; start_offset++) {
      for (auto i = 0; i < relative_offset_set.size(); i++) {
        auto output_index = start_offset + flat_segment_id(i) * big_stride;
        auto offset = start_offset + relative_offset_set[i];
        if (output_flat(output_index).length() != 0)
          output_flat(output_index).append(separator_.c_str());
        output_flat(output_index).append(flat_input(offset));
      }
    }
  }

 private:
  string separator_;
};

#define REGISTER_CPU_KERNEL(indices_type, num_segments_type)  \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("UnsortedSegmentJoin")                             \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint<indices_type>("Tindices")           \
          .TypeConstraint<num_segments_type>("Tnumsegments"), \
      UnsortedSegmentJoinOp<indices_type, num_segments_type>);

REGISTER_CPU_KERNEL(int32, int32);
REGISTER_CPU_KERNEL(int32, int64_t);
REGISTER_CPU_KERNEL(int64_t, int32);
REGISTER_CPU_KERNEL(int64_t, int64_t);
#undef REGISTER_CPU_KERNEL

}  // namespace tensorflow
