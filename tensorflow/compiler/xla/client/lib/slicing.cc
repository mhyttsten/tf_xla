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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSslicingDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSslicingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSslicingDTcc() {
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

#include "tensorflow/compiler/xla/client/lib/slicing.h"

#include <algorithm>
#include <limits>
#include <vector>

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

XlaOp DynamicStridedSlice(XlaOp input, absl::Span<const XlaOp> base_indices,
                          absl::Span<const int64_t> window_sizes,
                          absl::Span<const int64_t> strides) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSslicingDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/xla/client/lib/slicing.cc", "DynamicStridedSlice");

  XlaOp sliced_input = DynamicSlice(input, base_indices, window_sizes);
  if (std::any_of(strides.begin(), strides.end(),
                  [](int64_t stride) { return stride != 1; })) {
    sliced_input =
        Slice(sliced_input, std::vector<int64_t>(window_sizes.size()),
              window_sizes, strides);
  }
  return sliced_input;
}

XlaOp SliceInMinorDims(XlaOp x, absl::Span<const int64_t> start,
                       absl::Span<const int64_t> end) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSslicingDTcc mht_1(mht_1_v, 215, "", "./tensorflow/compiler/xla/client/lib/slicing.cc", "SliceInMinorDims");

  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RET_CHECK(start.size() == end.size());
    int64_t n_minor_dims = start.size();

    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));

    const int64_t n_dims = shape.rank();
    TF_RET_CHECK(n_minor_dims <= n_dims);
    auto major_dims = shape.dimensions().subspan(
        /*pos=*/0,
        /*len=*/n_dims - n_minor_dims);

    // Prepends 0s in the major dim
    std::vector<int64_t> padded_start(n_dims, 0);
    std::copy(start.begin(), start.end(),
              padded_start.begin() + major_dims.size());

    // Prepends the shape of the major dims.
    std::vector<int64_t> padded_end(n_dims);
    std::copy(major_dims.begin(), major_dims.end(), padded_end.begin());
    std::copy(end.begin(), end.end(), padded_end.begin() + major_dims.size());

    std::vector<int64_t> strides(n_dims, 1);
    return Slice(x, padded_start, padded_end, strides);
  });
}

XlaOp UpdateSlice(XlaOp x, XlaOp update, absl::Span<const int64_t> start) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSslicingDTcc mht_2(mht_2_v, 247, "", "./tensorflow/compiler/xla/client/lib/slicing.cc", "UpdateSlice");

  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64_t n_dims = shape.rank();
    const int64_t start_size = start.size();
    TF_RET_CHECK(start_size == n_dims);

    // TODO(phawkins): make int64_t work on all backends, remove the int32_t
    // cast.
    std::vector<int32_t> start_as_int32(start.begin(), start.end());
    std::vector<XlaOp> start_ops(start.size());
    for (int i = 0, end = start.size(); i < end; ++i) {
      start_ops[i] = ConstantR0(builder, start_as_int32[i]);
    }
    return DynamicUpdateSlice(x, update, start_ops);
  });
}

XlaOp UpdateSliceInMinorDims(XlaOp x, XlaOp update,
                             absl::Span<const int64_t> start) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSslicingDTcc mht_3(mht_3_v, 270, "", "./tensorflow/compiler/xla/client/lib/slicing.cc", "UpdateSliceInMinorDims");

  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64_t n_dims = shape.rank();
    const int64_t n_minor_dims = start.size();
    TF_RET_CHECK(n_minor_dims <= n_dims);
    std::vector<int64_t> padded_start(n_dims, 0);
    std::copy(start.begin(), start.end(),
              padded_start.begin() + (n_dims - n_minor_dims));
    return UpdateSlice(x, update, padded_start);
  });
}

namespace {

std::vector<int64_t> ConcatVectors(absl::Span<const int64_t> xs,
                                   absl::Span<const int64_t> ys) {
  std::vector<int64_t> output(xs.size() + ys.size());
  std::copy(xs.begin(), xs.end(), output.begin());
  std::copy(ys.begin(), ys.end(), output.begin() + xs.size());
  return output;
}

StatusOr<std::vector<XlaOp>> PrependZerosInMajorDims(
    XlaOp x, absl::Span<const XlaOp> starts) {
  XlaBuilder* builder = x.builder();
  TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
  const int64_t n_dims = shape.rank();
  auto zero = ConstantR0<int32_t>(builder, 0);
  std::vector<XlaOp> padded_starts(n_dims, zero);
  for (int i = 0; i < starts.size(); ++i) {
    padded_starts[n_dims - starts.size() + i] = starts[i];
  }
  return padded_starts;
}

}  // namespace

XlaOp DynamicSliceInMinorDims(XlaOp x, absl::Span<const XlaOp> starts,
                              absl::Span<const int64_t> sizes) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSslicingDTcc mht_4(mht_4_v, 313, "", "./tensorflow/compiler/xla/client/lib/slicing.cc", "DynamicSliceInMinorDims");

  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64_t n_dims = shape.rank();
    int64_t n_minor_dims = starts.size();
    TF_RET_CHECK(n_minor_dims == sizes.size());
    TF_RET_CHECK(n_minor_dims <= n_dims);
    auto major_dims = shape.dimensions().subspan(
        /*pos=*/0,
        /*len=*/n_dims - sizes.size());
    TF_ASSIGN_OR_RETURN(auto padded_starts, PrependZerosInMajorDims(x, starts));
    auto padded_sizes = ConcatVectors(major_dims, sizes);
    return DynamicSlice(x, padded_starts, padded_sizes);
  });
}

XlaOp DynamicUpdateSliceInMinorDims(XlaOp x, XlaOp update,
                                    absl::Span<const XlaOp> starts) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSslicingDTcc mht_5(mht_5_v, 334, "", "./tensorflow/compiler/xla/client/lib/slicing.cc", "DynamicUpdateSliceInMinorDims");

  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto padded_starts, PrependZerosInMajorDims(x, starts));
    return DynamicUpdateSlice(x, update, padded_starts);
  });
}

XlaOp TorchGather(XlaOp input, XlaOp index, int64_t dim, bool sparse) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSslicingDTcc mht_6(mht_6_v, 345, "", "./tensorflow/compiler/xla/client/lib/slicing.cc", "TorchGather");

  XlaBuilder* builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape index_shape, builder->GetShape(index));
    TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
    if (ShapeUtil::ElementHasBitWidth(index_shape, 64) &&
        input_shape.dimensions(dim) < std::numeric_limits<uint32_t>::max()) {
      index = ConvertElementType(index, U32);
      index_shape.set_element_type(U32);
    }
    if (index_shape.rank() == 1) {
      return TorchIndexSelect(input, index, 0);
    }
    if (!sparse) {
      std::vector<int64_t> index_broadcast_dims;
      std::vector<int64_t> input_broadcast_dims;
      std::vector<int64_t> sizes;
      sizes.reserve(index_shape.rank());
      for (int64_t i = 0; i < index_shape.rank(); ++i) {
        if (i < dim) {
          input_broadcast_dims.push_back(i);
          index_broadcast_dims.push_back(i);
        } else if (i == dim) {
          sizes.push_back(input_shape.dimensions(i));
          input_broadcast_dims.push_back(i);
          index_broadcast_dims.push_back(i + 1);
        } else {
          input_broadcast_dims.push_back(i + 1);
          index_broadcast_dims.push_back(i + 1);
        }
        sizes.push_back(index_shape.dimensions(i));
      }
      auto mask = Eq(
          BroadcastInDim(index, sizes, index_broadcast_dims),
          Iota(builder, ShapeUtil::MakeShape(index_shape.element_type(), sizes),
               dim));
      auto masked_input = Select(
          mask, BroadcastInDim(input, sizes, input_broadcast_dims),
          Zeros(builder,
                ShapeUtil::MakeShape(input_shape.element_type(), sizes)));
      return Reduce(masked_input, Zero(builder, input_shape.element_type()),
                    CreateScalarIdentityWithZeroComputation(
                        input_shape.element_type(), builder),
                    {dim});
    }

    ShapeUtil::AppendMajorDimension(1, &index_shape);
    std::vector<XlaOp> to_concat;

    to_concat.reserve(input_shape.rank());
    for (int64_t i = 0; i < input_shape.rank(); ++i) {
      if (i == dim) {
        to_concat.push_back(Reshape(index, index_shape.dimensions()));
      } else {
        to_concat.push_back(Iota(builder, index_shape, i));
      }
    }
    XlaOp gather_indices = ConcatInDim(builder, to_concat, input_shape.rank());
    std::vector<int64_t> slice_sizes(input_shape.rank(), 1);
    GatherDimensionNumbers gather_dnums;
    gather_dnums.set_index_vector_dim(input_shape.rank());
    for (int64_t i = 0; i < input_shape.rank(); ++i) {
      gather_dnums.add_collapsed_slice_dims(i);
      gather_dnums.add_start_index_map(i);
    }
    return Gather(input, gather_indices, gather_dnums, slice_sizes);
  });
}

XlaOp TorchScatterDense(XlaOp input, XlaOp index, XlaOp src, int64_t dim,
                        const std::function<XlaOp(XlaOp, XlaOp)>& combiner) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSslicingDTcc mht_7(mht_7_v, 418, "", "./tensorflow/compiler/xla/client/lib/slicing.cc", "TorchScatterDense");

  XlaBuilder* builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape index_shape, builder->GetShape(index));
    TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
    std::vector<int64_t> index_broadcast_dims;
    std::vector<int64_t> sizes;
    const auto rank = index_shape.rank();
    sizes.reserve(rank + 1);
    for (int64_t i = 0; i < index_shape.rank(); ++i) {
      if (i < dim) {
        index_broadcast_dims.push_back(i);
      } else {
        if (i == dim) {
          sizes.push_back(input_shape.dimensions(i));
        }
        index_broadcast_dims.push_back(i + 1);
      }
      sizes.push_back(index_shape.dimensions(i));
    }
    auto mask =
        Eq(BroadcastInDim(index, sizes, index_broadcast_dims),
           Iota(builder,
                ShapeUtil::MakeShape(index_shape.element_type(), sizes), dim));
    auto masked_src =
        Select(mask, BroadcastInDim(src, sizes, index_broadcast_dims),
               Zeros(builder,
                     ShapeUtil::MakeShape(input_shape.element_type(), sizes)));

    return combiner(
        input,
        Reduce(masked_src, Zero(builder, input_shape.element_type()),
               CreateScalarComputation("reducer", input_shape.element_type(),
                                       builder, combiner),
               {dim + 1}));
  });
}

XlaOp TorchIndexSelect(XlaOp input, XlaOp index, int64_t dim,
                       int64_t batch_dims) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSslicingDTcc mht_8(mht_8_v, 460, "", "./tensorflow/compiler/xla/client/lib/slicing.cc", "TorchIndexSelect");

  XlaBuilder* builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
    TF_ASSIGN_OR_RETURN(Shape index_shape, builder->GetShape(index));
    if (dim < batch_dims) {
      return InvalidArgument(
          "Gather dim must be greater than or equal to the number of batch "
          "dims");
    }
    if (ShapeUtil::ElementHasBitWidth(index_shape, 64) &&
        input_shape.dimensions(dim) < std::numeric_limits<uint32_t>::max()) {
      index = ConvertElementType(index, U32);
      index_shape.set_element_type(U32);
    }
    std::vector<int64_t> slice_sizes = SpanToVector(input_shape.dimensions());
    GatherDimensionNumbers gather_dnums;
    gather_dnums.set_index_vector_dim(index_shape.rank());
    if (batch_dims > 0) {
      ShapeUtil::AppendMajorDimension(1, &index_shape);
      std::vector<XlaOp> to_concat;
      to_concat.reserve(batch_dims + 1);
      for (int64_t batch_dim = 0; batch_dim < batch_dims; ++batch_dim) {
        to_concat.push_back(Iota(builder, index_shape, batch_dim));
      }
      to_concat.push_back(Reshape(index, index_shape.dimensions()));
      index = ConcatInDim(builder, to_concat, gather_dnums.index_vector_dim());
    }
    for (int64_t i = 0; i < input_shape.rank(); ++i) {
      if (i < batch_dims || i == dim) {
        slice_sizes[i] = std::min<int64_t>(slice_sizes[i], 1);
        gather_dnums.add_collapsed_slice_dims(i);
        gather_dnums.add_start_index_map(i);
      } else {
        if (i < dim) {
          gather_dnums.add_offset_dims(i);
        } else {
          gather_dnums.add_offset_dims(i + gather_dnums.index_vector_dim() -
                                       (1 + batch_dims));
        }
      }
    }
    return Gather(input, index, gather_dnums, slice_sizes);
  });
}

}  // namespace xla
