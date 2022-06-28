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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSsortingDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSsortingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSsortingDTcc() {
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

#include "tensorflow/compiler/xla/client/lib/sorting.h"

#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/loops.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

XlaOp TopK(XlaOp input, int64_t k) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSsortingDTcc mht_0(mht_0_v, 197, "", "./tensorflow/compiler/xla/client/lib/sorting.cc", "TopK");

  XlaBuilder* const builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
    int last_dim = input_shape.dimensions_size() - 1;
    int64_t last_dim_size = input_shape.dimensions(last_dim);
    // TODO(b/148796364): tune these constants for better performance.
    const int64_t kPerPartitionSize = 8192;        // 2^13
    const int64_t kLastDimSizeThreshold = 524288;  // 2^19
    const int64_t kMinNumPartitions = 8;
    const int64_t kMinimalK = 1000;
    if ((k >= kMinimalK) && (k < kPerPartitionSize) &&
        (kPerPartitionSize / k > 2) && last_dim_size >= kLastDimSizeThreshold) {
      int64_t num_partitions =
          CeilOfRatio(last_dim_size - k, kPerPartitionSize - k);
      if (num_partitions >= kMinNumPartitions) {
        return TopKWithPartitions(input, k, num_partitions);
      }
    }

    Shape iota_shape = ShapeUtil::MakeShape(S32, input_shape.dimensions());
    XlaOp iota_s32 = Iota(builder, iota_shape, last_dim);
    for (int64_t i = 0; i < input_shape.rank(); ++i) {
      if (input_shape.is_dynamic_dimension(i)) {
        // Propagate dynamic dimension from inputs to iota.
        iota_s32 = SetDimensionSize(iota_s32, GetDimensionSize(input, i), i);
      }
    }
    auto input_dims = input_shape.dimensions();

    // We can pack BF16 values to be sorted along with their index values into a
    // single 32-bit value in some cases.
    constexpr int32_t kLow16BitsLimit = int32_t{1} << 16;
    constexpr int32_t kLow16BitsMask = kLow16BitsLimit - 1;
    constexpr int32_t kHigh16BitsMask = ~kLow16BitsMask;

    // Whether to use the packed sorting algorithm for BF16 data. This change is
    // good in general, and enables a separate TPU optimization for common cases
    // as well (top-k for small k).
    const bool use_packed_bf16_sort =
        (input_shape.element_type() == BF16 && last_dim_size < kLow16BitsLimit);

    std::vector<int64_t> start_indices(input_shape.dimensions_size(), 0);
    std::vector<int64_t> limit_indices(input_dims.begin(), input_dims.end());
    limit_indices[last_dim] = k;
    std::vector<int64_t> strides(input_shape.dimensions_size(), 1);

    XlaOp values;
    XlaOp indices;
    if (use_packed_bf16_sort) {
      // Converts a 32-bit value from sign-magnitude (used for floats) to one's
      // complement (easy to compare using integer operations) or vice versa.
      auto sign_magnitude_to_from_ones_complement = [builder](const XlaOp in) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSsortingDTcc mht_1(mht_1_v, 252, "", "./tensorflow/compiler/xla/client/lib/sorting.cc", "lambda");

        constexpr int32_t kAllNonSignBits = 0x7fffffff;
        XlaOp in_s32 = BitcastConvertType(in, S32);
        return Xor(
            And(in_s32, ConstantR0<int32_t>(builder, kAllNonSignBits)),
            ShiftRightArithmetic(in_s32, ConstantR0<int32_t>(builder, 31)));
      };

      // Move input values to the high 16 bits of each 32-bit element, convert
      // them to allow integer comparisons, set the low 16 bits to one (in order
      // to reverse the sort order of the element indices), then XOR in the iota
      // result. This leads to the ones' complement version of the BF16 input in
      // the high 16 bits and the ones' complement of the indices in the low 16
      // bits.
      XlaOp input_f32_trimmed =
          Or(sign_magnitude_to_from_ones_complement(
                 BitcastConvertType(ConvertElementType(input, F32), S32)),
             ConstantR0<int32_t>(builder, kLow16BitsMask));
      XlaOp input_and_iota = Xor(input_f32_trimmed, iota_s32);

      // Sort in reverse order so the largest elements are at the beginning.
      // Breaking ties here is why the index bits need to be inverted.
      XlaOp sort_result_raw = Sort(
          {input_and_iota}, CreateScalarGtComputation({S32}, builder), last_dim,
          /*is_stable=*/false);

      // Slice off the first k values.
      sort_result_raw =
          Slice(sort_result_raw, start_indices, limit_indices, strides);
      // The k in TopK is static so we shouldn't generate a dynamic dimension
      // even if input is dynamic.
      sort_result_raw = RemoveDynamicDimension(sort_result_raw, last_dim);

      // Get the high 16 bits of each value from the sorted result and convert
      // them back to BF16.
      values = ConvertElementType(
          BitcastConvertType(
              And(sign_magnitude_to_from_ones_complement(sort_result_raw),
                  ConstantR0<int32_t>(builder, kHigh16BitsMask)),
              F32),
          BF16);

      // Get the index values from the low 16 bits of each value and invert them
      // again.
      indices = And(
          Xor(sort_result_raw, ConstantR0<int32_t>(builder, kLow16BitsMask)),
          ConstantR0<int32_t>(builder, kLow16BitsMask));
    } else {
      XlaOp sort_result =
          Sort({input, iota_s32},
               CreateScalarGtComputation({input_shape.element_type(), S32},
                                         iota_s32.builder()),
               last_dim, /*is_stable=*/true);
      values = Slice(GetTupleElement(sort_result, 0), start_indices,
                     limit_indices, strides);
      // The k in TopK is static so we shouldn't generate a dynamic dimension
      // even if input is dynamic.
      values = RemoveDynamicDimension(values, last_dim);
      indices = Slice(GetTupleElement(sort_result, 1), start_indices,
                      limit_indices, strides);
      indices = RemoveDynamicDimension(indices, last_dim);
    }

    return Tuple(builder, {values, indices});
  });
}

XlaOp TopKWithPartitions(XlaOp input, int64_t k, int64_t num_partitions) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSsortingDTcc mht_2(mht_2_v, 322, "", "./tensorflow/compiler/xla/client/lib/sorting.cc", "TopKWithPartitions");

  XlaBuilder* const builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
    int last_dim = input_shape.dimensions_size() - 1;
    // Calculate per partition size.
    auto input_dims = input_shape.dimensions();
    int64_t last_dim_size = input_shape.dimensions(last_dim);
    const int64_t per_partition_size =
        CeilOfRatio(last_dim_size, num_partitions);
    // Do normal TopK when per partition size is smaller than or equal to k.
    if (k >= per_partition_size) {
      return TopK(input, k);
    }

    Shape iota_shape = ShapeUtil::MakeShape(S32, input_shape.dimensions());
    XlaOp iota_s32 = Iota(builder, iota_shape, last_dim);
    for (int64_t i = 0; i < input_shape.rank(); ++i) {
      if (input_shape.is_dynamic_dimension(i)) {
        // Propagate dynamic dimension from inputs to iota.
        iota_s32 = SetDimensionSize(iota_s32, GetDimensionSize(input, i), i);
      }
    }

    auto topk_body_fn =
        [&](XlaOp partition, absl::Span<const XlaOp> values_and_indices,
            XlaBuilder* builder) -> StatusOr<std::vector<XlaOp>> {
      auto values = values_and_indices[0];
      auto indices = values_and_indices[1];
      auto input = values_and_indices[2];
      auto iota_s32 = values_and_indices[3];

      // Slice value and indices for this partition.
      XlaOp start = Mul(Add(partition, ConstantR0<int32_t>(builder, 1)),
                        ConstantR0<int32_t>(builder, per_partition_size));
      XlaOp sliced_input =
          DynamicSliceInMinorDims(input, {start}, {per_partition_size});
      XlaOp sliced_indices =
          DynamicSliceInMinorDims(iota_s32, {start}, {per_partition_size});
      // Concat with previous results.
      sliced_input = ConcatInDim(builder, {values, sliced_input}, last_dim);
      sliced_indices =
          ConcatInDim(builder, {indices, sliced_indices}, last_dim);
      // Sort this slice
      XlaOp sort_result =
          Sort({sliced_input, sliced_indices},
               CreateScalarGtComputation({input_shape.element_type(), S32},
                                         sliced_indices.builder()),
               last_dim, true);

      std::vector<int64_t> start_indices(input_shape.dimensions_size(), 0);
      std::vector<int64_t> limit_indices(input_dims.begin(), input_dims.end());
      std::vector<int64_t> strides(input_shape.dimensions_size(), 1);
      // Slice topk.
      start_indices[last_dim] = 0;
      limit_indices[last_dim] = k;
      values = Slice(GetTupleElement(sort_result, 0), start_indices,
                     limit_indices, strides);
      indices = Slice(GetTupleElement(sort_result, 1), start_indices,
                      limit_indices, strides);
      return std::vector<XlaOp>{values, indices, input, iota_s32};
    };

    // Get the values and indices for the first topk so that they can
    // be passed to the while loop.
    std::vector<int64_t> start_indices(input_shape.dimensions_size(), 0);
    std::vector<int64_t> limit_indices(input_dims.begin(), input_dims.end());
    std::vector<int64_t> strides(input_shape.dimensions_size(), 1);
    start_indices[last_dim] = 0;
    limit_indices[last_dim] = per_partition_size;
    // Slice value and indices for the first partition.
    XlaOp sliced_input = Slice(input, start_indices, limit_indices, strides);
    XlaOp sliced_indices =
        Slice(iota_s32, start_indices, limit_indices, strides);
    // Sort this slice
    XlaOp sort_result =
        Sort({sliced_input, sliced_indices},
             CreateScalarGtComputation({input_shape.element_type(), S32},
                                       sliced_indices.builder()),
             last_dim, /*is_stable=*/true);

    // Slice topk.
    start_indices[last_dim] = 0;
    limit_indices[last_dim] = k;
    XlaOp values = Slice(GetTupleElement(sort_result, 0), start_indices,
                         limit_indices, strides);
    XlaOp indices = Slice(GetTupleElement(sort_result, 1), start_indices,
                          limit_indices, strides);

    // Pass the result of the first TopK to the while loop and do
    // num_partition - 1 iterations.
    TF_ASSIGN_OR_RETURN(auto values_and_indices,
                        ForEachIndex(num_partitions - 1, S32, topk_body_fn,
                                     {values, indices, input, iota_s32},
                                     "topk_with_partition", builder));
    return Tuple(builder, {values_and_indices[0], values_and_indices[1]});
  });
}

}  // namespace xla
