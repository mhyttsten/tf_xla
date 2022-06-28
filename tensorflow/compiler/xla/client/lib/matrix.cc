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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc() {
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

#include "tensorflow/compiler/xla/client/lib/matrix.h"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

XlaOp IdentityMatrix(XlaBuilder* builder, PrimitiveType type, int64_t m,
                     int64_t n) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_0(mht_0_v, 220, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "IdentityMatrix");

  auto a = Iota(builder, U32, m);
  auto b = Iota(builder, U32, n);
  auto indicator = Eq(a, Broadcast(b, {m}), /*broadcast_dimensions=*/{0});
  return ConvertElementType(indicator, type);
}

XlaOp GetDiagonalMask(XlaOp x, int diagonal) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_1(mht_1_v, 230, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "GetDiagonalMask");

  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    auto n_dims = static_cast<int32_t>(shape.rank());
    TF_RET_CHECK(n_dims >= 2);
    auto m = shape.dimensions(n_dims - 2);
    auto n = shape.dimensions(n_dims - 1);
    absl::Span<const int64_t> major_dims =
        shape.dimensions().subspan(/*pos=*/0, /*len=*/n_dims - 2);
    auto a = Iota(builder, S32, n);
    auto b = Iota(builder, S32, m) + ConstantR0WithType(builder, S32, diagonal);
    auto indicator = Eq(b, Broadcast(a, {m}), /*broadcast_dimensions=*/{0});
    auto mask = Broadcast(indicator, major_dims);
    return mask;
  });
}

XlaOp GetMatrixDiagonal(XlaOp x, int k) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_2(mht_2_v, 251, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "GetMatrixDiagonal");

  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    auto n_dims = static_cast<int32_t>(shape.rank());
    TF_RET_CHECK(n_dims >= 2);
    const int64_t m = shape.dimensions(n_dims - 2);
    const int64_t n = shape.dimensions(n_dims - 1);

    if (k <= -m || k >= n) {
      auto zero_size_shape = shape;
      zero_size_shape.DeleteDimension(n_dims - 1);
      zero_size_shape.set_dimensions(n_dims - 2, 0);
      return ConstantLiteral(builder, Literal{zero_size_shape});
    }
    auto mask = GetDiagonalMask(x, k);

    int64_t reduce_dim = n_dims - 1;
    if ((k == 0 && m >= n) || k < 0) {
      reduce_dim = n_dims - 2;
    }
    auto result = Reduce(
        Select(mask, x, Zeros(builder, shape)), ScalarLike(x, 0),
        CreateScalarIdentityWithZeroComputation(shape.element_type(), builder),
        {reduce_dim});
    // k == 0, we can save one slice op.
    if (k == 0) {
      return result;
    }
    return SliceInMinorDims(result, {0},
                            {k > 0 ? std::min(m, n - k) : std::min(n, m + k)});
  });
}

XlaOp GetMatrixDiagonalViaGather(XlaOp x, int k) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_3(mht_3_v, 288, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "GetMatrixDiagonalViaGather");

  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    auto n_dims = static_cast<int32_t>(shape.rank());
    TF_RET_CHECK(n_dims >= 2);
    const int64_t m = shape.dimensions(n_dims - 2);
    const int64_t n = shape.dimensions(n_dims - 1);

    // The start_indices has a shape of {diag_len, 2}, and each pair of value in
    // its dimension 1 represents the (row, col) of the diagonal. We set
    // index_vector_dim to 1 and make start_index_map and collapsed_slice_dims
    // contain the same two dimension indices. This makes sure that the (row,
    // col) pairs in start_indices are propagated to the indices for the two
    // collapsed dimensions in the operand indices through start_index_map.
    const int64_t num_index_dims = 2;
    const int64_t axis = n_dims - num_index_dims;

    // Calculate the indices of diagonal part with offset k.
    const int64_t diag_len =
        std::max(std::min(m + std::min(k, 0), n - std::max(k, 0)), int64_t{0});
    XlaOp diag_base_indices = BroadcastInDim(Iota(builder, S32, diag_len),
                                             {diag_len, num_index_dims}, {0});
    XlaOp diag_offset =
        Broadcast(ConstantR1<int>(builder, {std::max(-k, 0), std::max(k, 0)}),
                  {diag_len});
    XlaOp start_indices = Add(diag_base_indices, diag_offset);

    // Example of a 3D diag-part extracting diagonal part with offset=1 out of a
    // tensor of shape [2,5,4].
    //
    //  operand = s32[2,5,4] parameter(0)
    //  indices = s32[3,2] parameter(1)
    //  gather = s32[2,3] gather(operand, indices),
    //       offset_dims={0},
    //       collapsed_slice_dims={1,2},
    //       start_index_map={1,2},
    //       index_vector_dim=1,
    //       slice_sizes={2, 1, 1}

    xla::GatherDimensionNumbers dim_numbers;
    std::vector<int64_t> slice_sizes;
    slice_sizes.reserve(n_dims);
    for (int64_t i = 0; i < n_dims; i++) {
      int64_t window_bound;
      if (axis <= i) {
        dim_numbers.add_collapsed_slice_dims(i);
        dim_numbers.add_start_index_map(i);
        window_bound = (shape.dimensions(i) != 0) ? 1 : 0;
      } else {
        dim_numbers.add_offset_dims(i);
        window_bound = shape.dimensions(i);
      }
      slice_sizes.push_back(window_bound);
    }

    dim_numbers.set_index_vector_dim(1);

    return Gather(x, start_indices, dim_numbers, slice_sizes,
                  /*indices_are_sorted=*/true);
  });
}

XlaOp SetMatrixDiagonal(XlaOp matrix, XlaOp diag, int k) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_4(mht_4_v, 354, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "SetMatrixDiagonal");

  XlaBuilder* builder = matrix.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(matrix));
    TF_ASSIGN_OR_RETURN(Shape diag_shape, builder->GetShape(diag));
    auto n_dims = static_cast<int32_t>(shape.rank());
    TF_RET_CHECK(n_dims >= 2);
    const int64_t m = shape.dimensions(n_dims - 2);
    const int64_t n = shape.dimensions(n_dims - 1);
    const int64_t d = diag_shape.dimensions(n_dims - 2);
    std::vector<int64_t> broadcast_dims(n_dims - 1);
    absl::c_iota(broadcast_dims, 0);
    int64_t pad_high = m - d;
    if (k < 0) {
      ++(broadcast_dims.back());
      pad_high = n - d;
    }

    if (pad_high != 0) {
      PaddingConfig padding_config;
      for (int64_t i = 0; i < diag_shape.rank() - 1; ++i) {
        auto* dims = padding_config.add_dimensions();
        dims->set_edge_padding_low(0);
        dims->set_interior_padding(0);
        dims->set_edge_padding_high(0);
      }
      auto* dims = padding_config.add_dimensions();
      dims->set_edge_padding_low(0);
      dims->set_interior_padding(0);
      dims->set_edge_padding_high(pad_high);
      diag = Pad(diag, ScalarLike(diag, 0), padding_config);
    }

    return Select(GetDiagonalMask(matrix, k),
                  BroadcastInDim(diag, shape.dimensions(), broadcast_dims),
                  matrix);
  });
}

XlaOp TriangleMask(XlaOp x, int diagonal) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_5(mht_5_v, 396, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "TriangleMask");

  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64_t n_dims = shape.rank();
    TF_RET_CHECK(n_dims >= 2);
    const int64_t m = shape.dimensions(n_dims - 2);
    const int64_t n = shape.dimensions(n_dims - 1);
    absl::Span<const int64_t> major_dims =
        shape.dimensions().subspan(/*pos=*/0, /*len=*/n_dims - 2);
    auto a = Iota(builder, S32, n);
    auto b = Iota(builder, S32, m) + ConstantR0<int32_t>(builder, diagonal);
    XlaOp indicator;
    indicator = Ge(b, Broadcast(a, {m}), /*broadcast_dimensions=*/{0});
    return Broadcast(indicator, major_dims);
  });
}

XlaOp Triangle(XlaOp x, bool lower) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_6(mht_6_v, 417, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "Triangle");

  return lower ? Select(TriangleMask(x, 0), x, ZerosLike(x))
               : Select(TriangleMask(x, -1), ZerosLike(x), x);
}

XlaOp UpperTriangle(XlaOp x) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_7(mht_7_v, 425, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "UpperTriangle");
 return Triangle(x, false); }

XlaOp LowerTriangle(XlaOp x) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_8(mht_8_v, 430, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "LowerTriangle");
 return Triangle(x, true); }

XlaOp Symmetrize(XlaOp x, bool lower) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_9(mht_9_v, 435, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "Symmetrize");

  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    if (shape.rank() < 2) {
      return InvalidArgument(
          "Argument to symmetrize must have >= 2 dimensions, got %s",
          shape.ToString());
    }
    const int64_t m = ShapeUtil::GetDimension(shape, -2);
    const int64_t n = ShapeUtil::GetDimension(shape, -1);
    if (m != n) {
      return InvalidArgument(
          "The two most minor dimensions of the argument to symmetrize must be "
          "equal size, got %s",
          shape.ToString());
    }
    auto mask = lower ? TriangleMask(x, 0) : Not(TriangleMask(x, -1));
    if (primitive_util::IsComplexType(shape.element_type())) {
      auto re = Select(mask, Real(x), TransposeInMinorDims(Real(x)));
      auto im_mask = lower ? TriangleMask(x, -1) : Not(TriangleMask(x, 0));
      auto im = Select(im_mask, Imag(x), ZerosLike(Imag(x)));
      im = Select(mask, im, -TransposeInMinorDims(im));
      return Complex(re, im);
    } else {
      return Select(mask, x, TransposeInMinorDims(x));
    }
  });
}

namespace {
absl::optional<std::array<std::vector<int64_t>, 3>> EinsumDiagonalLabels(
    absl::Span<const int64_t> config) {
  std::vector<int64_t> unique_labels;
  std::vector<int64_t> reduce_dims;
  std::vector<int64_t> broadcast_dims;
  for (auto label = config.begin(); label != config.end(); ++label) {
    auto first_label = absl::c_find(config, *label);
    auto dim = label - config.begin();
    if (first_label == label) {
      unique_labels.push_back(*label);
      broadcast_dims.push_back(dim);
    } else {
      reduce_dims.push_back(dim);
    }
  }
  if (unique_labels.size() == config.size()) {
    return absl::nullopt;
  }
  return {{unique_labels, reduce_dims, broadcast_dims}};
}

// Masks a tensor such that only the diagonal of repeated indices are non-zero.
// The result of this can be used to create a diagonal matrix with an identity
// reduction.
xla::XlaOp EinsumDiagonalMask(XlaOp x, absl::Span<const int64_t> config) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_10(mht_10_v, 493, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "EinsumDiagonalMask");

  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape x_shape, builder->GetShape(x));
    Shape iota_shape = x_shape;
    iota_shape.set_element_type(S32);
    XlaOp mask = ConstantR0(builder, true);

    for (auto label = config.begin(); label != config.end(); ++label) {
      const int64_t dim = label - config.begin();
      auto first_label = absl::c_find(config, *label);
      if (first_label != label) {
        const int64_t first_dim = first_label - config.begin();
        mask = And(mask, Eq(Iota(builder, iota_shape, first_dim),
                            Iota(builder, iota_shape, dim)));
      }
    }
    return Select(mask, x, ZerosLike(x));
  });
}

xla::XlaOp EinsumDiagonal(XlaOp x, absl::Span<const int64_t> config) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_11(mht_11_v, 517, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "EinsumDiagonal");

  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    auto labels = EinsumDiagonalLabels(config);
    if (!labels) {
      return x;
    }
    auto zero = ScalarLike(x, 0);
    TF_ASSIGN_OR_RETURN(Shape x_shape, builder->GetShape(x));
    return Reduce(EinsumDiagonalMask(x, config), zero,
                  CreateScalarIdentityWithZeroComputation(
                      x_shape.element_type(), builder),
                  labels->at(1));
  });
}

xla::XlaOp EinsumInverseDiagonal(XlaOp x, absl::Span<const int64_t> config) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_12(mht_12_v, 536, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "EinsumInverseDiagonal");

  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    auto labels = EinsumDiagonalLabels(config);
    if (!labels) {
      return x;
    }
    TF_ASSIGN_OR_RETURN(Shape x_shape, builder->GetShape(x));
    std::vector<int64_t> broadcast_sizes;
    int64_t x_dim = 0;
    for (auto label = config.begin(); label != config.end(); ++label) {
      auto first_label = absl::c_find(config, *label);
      if (first_label == label) {
        broadcast_sizes.push_back(x_shape.dimensions(x_dim));
        ++x_dim;
      } else {
        broadcast_sizes.push_back(
            broadcast_sizes[first_label - config.begin()]);
      }
    }
    x = BroadcastInDim(x, broadcast_sizes, labels->at(2));
    return EinsumDiagonalMask(x, config);
  });
}
}  // namespace

namespace {
// Helper method to remove dimensions from a shape and dot dimension numbers
// used to implement implicit broadcasting.
template <typename C>
void DeleteDimsFromContainer(absl::Span<const int64_t> to_delete, Shape* shape,
                             C* batch_dims, C* contracting_dims) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_13(mht_13_v, 570, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "DeleteDimsFromContainer");

  if (to_delete.empty()) {
    return;
  }
  for (int64_t i = to_delete.size() - 1; i >= 0; --i) {
    int64_t dim = to_delete[i];
    shape->DeleteDimension(dim);
    for (auto& b : *batch_dims) {
      if (b > dim) {
        --b;
      }
    }
    for (auto& c : *contracting_dims) {
      if (c > dim) {
        --c;
      }
    }
  }
}
}  // namespace

xla::XlaOp Einsum(xla::XlaOp x, absl::Span<const int64_t> x_config,
                  xla::XlaOp y, absl::Span<const int64_t> y_config,
                  absl::Span<const int64_t> output_config,
                  xla::PrecisionConfig::Precision precision,
                  absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_14(mht_14_v, 598, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "Einsum");

  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    auto x_diagonal_labels = EinsumDiagonalLabels(x_config);
    if (x_diagonal_labels) {
      return Einsum(EinsumDiagonal(x, x_config), x_diagonal_labels->at(0), y,
                    y_config, output_config, precision, preferred_element_type);
    }
    auto y_diagonal_labels = EinsumDiagonalLabels(y_config);
    if (y_diagonal_labels) {
      return Einsum(x, x_config, EinsumDiagonal(y, y_config),
                    y_diagonal_labels->at(0), output_config, precision,
                    preferred_element_type);
    }
    auto output_diagonal_labels = EinsumDiagonalLabels(output_config);
    if (output_diagonal_labels) {
      return EinsumInverseDiagonal(
          Einsum(x, x_config, y, y_config, output_diagonal_labels->at(0),
                 precision, preferred_element_type),
          output_config);
    }

    TF_ASSIGN_OR_RETURN(Shape x_shape, builder->GetShape(x));
    TF_ASSIGN_OR_RETURN(Shape y_shape, builder->GetShape(y));
    const int64_t x_rank = x_config.size();
    const int64_t y_rank = y_config.size();
    const int64_t output_rank = output_config.size();
    absl::flat_hash_set<int64_t> x_map;
    absl::flat_hash_set<int64_t> y_map;
    absl::flat_hash_set<int64_t> output_map;

    for (auto d : x_config) {
      x_map.insert(d);
    }

    for (auto d : y_config) {
      y_map.insert(d);
    }

    for (auto d : output_config) {
      output_map.insert(d);
    }

    DotDimensionNumbers dnums;
    auto is_batch_dim = [&](int64_t d) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_15(mht_15_v, 645, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "lambda");

      return x_map.contains(d) && y_map.contains(d) && output_map.contains(d);
    };
    auto is_contracting = [&](int64_t d) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_16(mht_16_v, 651, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "lambda");

      return x_map.contains(d) && y_map.contains(d);
    };

    auto rhs_dimension_number = [&](int64_t d) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_17(mht_17_v, 658, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "lambda");

      return absl::c_find(y_config, d) - y_config.begin();
    };

    absl::InlinedVector<int64_t, 8> rhs_outer_dims;
    absl::InlinedVector<int64_t, 8> lhs_outer_dims;
    absl::InlinedVector<int64_t, 8> rhs_delete_dims;
    absl::InlinedVector<int64_t, 8> lhs_delete_dims;
    for (int64_t i = 0; i < x_rank; ++i) {
      auto dim_name = x_config[i];
      const int64_t rhs_dim = rhs_dimension_number(dim_name);

      if (is_batch_dim(dim_name)) {
        if (x_shape.dimensions(i) == y_shape.dimensions(rhs_dim)) {
          dnums.add_lhs_batch_dimensions(i);
          dnums.add_rhs_batch_dimensions(rhs_dim);
        } else if (x_shape.dimensions(i) == 1) {
          rhs_outer_dims.push_back(rhs_dim);
          lhs_delete_dims.push_back(i);
        } else {
          lhs_outer_dims.push_back(i);
          rhs_delete_dims.push_back(rhs_dim);
        }
      } else if (is_contracting(dim_name)) {
        if (x_shape.dimensions(i) == y_shape.dimensions(rhs_dim)) {
          dnums.add_lhs_contracting_dimensions(i);
          dnums.add_rhs_contracting_dimensions(rhs_dim);
        } else if (x_shape.dimensions(i) == 1) {
          rhs_outer_dims.push_back(rhs_dim);
          lhs_delete_dims.push_back(i);
        } else {
          lhs_outer_dims.push_back(i);
          rhs_delete_dims.push_back(rhs_dim);
        }
      } else {
        lhs_outer_dims.push_back(i);
      }
    }

    for (int64_t i = 0; i < y_rank; ++i) {
      auto dim_name = y_config[i];
      if (!is_batch_dim(dim_name) && !is_contracting(dim_name)) {
        rhs_outer_dims.push_back(i);
      }
    }

    absl::c_sort(rhs_outer_dims);
    absl::InlinedVector<int64_t, 8> output_transpose_dims;

    auto output_dimension_number = [&](int64_t d) -> absl::optional<int64_t> {
      auto pos = absl::c_find(output_config, d);
      if (pos == output_config.end()) {
        return absl::nullopt;
      }
      return pos - output_config.begin();
    };

    for (auto d : dnums.lhs_batch_dimensions()) {
      output_transpose_dims.push_back(*output_dimension_number(x_config[d]));
    }

    for (auto d : lhs_outer_dims) {
      if (auto output_dim = output_dimension_number(x_config[d])) {
        output_transpose_dims.push_back(*output_dim);
        continue;
      }
      lhs_delete_dims.push_back(d);
    }

    for (auto d : rhs_outer_dims) {
      if (auto output_dim = output_dimension_number(y_config[d])) {
        output_transpose_dims.push_back(*output_dim);
        continue;
      }
      rhs_delete_dims.push_back(d);
    }

    const int64_t transpose_rank = output_transpose_dims.size();
    std::vector<int64_t> transpose_dims(output_rank);
    for (int64_t i = 0; i < transpose_rank; ++i) {
      transpose_dims[output_transpose_dims[i]] = i;
    }

    // Remove ones that where broadcasted from the x and the y shape and adjust
    // the dimension numbers that are more minor than those dimensions.
    absl::c_sort(lhs_delete_dims);
    DeleteDimsFromContainer(lhs_delete_dims, &x_shape,
                            dnums.mutable_lhs_batch_dimensions(),
                            dnums.mutable_lhs_contracting_dimensions());

    absl::c_sort(rhs_delete_dims);
    DeleteDimsFromContainer(rhs_delete_dims, &y_shape,
                            dnums.mutable_rhs_batch_dimensions(),
                            dnums.mutable_rhs_contracting_dimensions());
    if (!lhs_delete_dims.empty()) {
      x = Reduce(x, ScalarLike(x, 0),
                 CreateScalarAddComputation(x_shape.element_type(), builder),
                 lhs_delete_dims);
    }

    if (!rhs_delete_dims.empty()) {
      y = Reduce(y, ScalarLike(y, 0),
                 CreateScalarAddComputation(y_shape.element_type(), builder),
                 rhs_delete_dims);
    }

    PrecisionConfig precision_proto;
    precision_proto.add_operand_precision(precision);
    precision_proto.add_operand_precision(precision);
    auto dot =
        DotGeneral(x, y, dnums, &precision_proto, preferred_element_type);
    dot = Transpose(dot, transpose_dims);
    if (transpose_rank == output_rank) {
      return dot;
    }

    auto is_output_only = [&](int64_t d) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_18(mht_18_v, 777, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "lambda");

      return output_map.contains(d) && !x_map.contains(d) && !y_map.contains(d);
    };

    int64_t dot_dim = 0;
    std::vector<int64_t> new_dims;
    new_dims.reserve(output_rank);
    TF_ASSIGN_OR_RETURN(Shape dot_shape, builder->GetShape(dot));
    for (auto d : output_config) {
      if (is_output_only(d)) {
        new_dims.push_back(1);
      } else {
        new_dims.push_back(dot_shape.dimensions(dot_dim));
      }
    }
    return Reshape(dot, new_dims);
  });
}

XlaOp BatchDot(XlaOp x, XlaOp y, PrecisionConfig::Precision precision,
               absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_19(mht_19_v, 800, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "BatchDot");

  return BatchDot(x, false, y, false, precision, preferred_element_type);
}

XlaOp BatchDot(XlaOp x, bool transpose_x, XlaOp y, bool transpose_y,
               PrecisionConfig::Precision precision,
               absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_20(mht_20_v, 809, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "BatchDot");

  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    std::string string("...mk,...kn->...mn");
    if (transpose_x) {
      std::swap(string[3], string[4]);
    }
    if (transpose_y) {
      std::swap(string[6 + 3], string[6 + 4]);
    }
    return Einsum(x, y, string, precision, preferred_element_type);
  });
}

StatusOr<std::array<std::vector<int64_t>, 3>> ParseEinsumString(
    absl::string_view einsum_config, int64_t x_rank, int64_t y_rank) {
  std::array<std::vector<int64_t>, 3> einsum_config_numeric;
  std::vector<absl::string_view> main_split =
      absl::StrSplit(einsum_config, ',');
  if (main_split.size() != 2) {
    return InvalidArgument("Expected one \",\" in einsum_config.");
  }

  auto maybe_invalid_character = [](char d) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("d: '" + std::string(1, d) + "'");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_21(mht_21_v, 836, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "lambda");

    if (absl::ascii_isalpha(d)) {
      return Status::OK();
    }
    if (d == '.') {
      return InvalidArgument("Unsupported \".\" in einsum config.");
    }
    return InvalidArgument("Unexpected character in einsum config.");
  };

  auto string_config_to_numeric =
      [&](absl::string_view config, bool is_input_config, int64_t input_rank,
          int64_t ellipsis_rank,
          std::vector<int64_t>* numeric_config) -> StatusOr<int64_t> {
    std::vector<absl::string_view> splits = absl::StrSplit(config, "...");
    if (splits.empty()) {
      return ellipsis_rank;
    }
    if (splits.size() > 2) {
      return InvalidArgument("Too many ellipses (\"...\") in einsum config.");
    }
    // There is one split if we don't have an ellipsis, and two splits if we do.
    const bool has_ellipsis = splits.size() > 1;
    // We only compute ellipsis_rank for input configs.
    if (is_input_config && has_ellipsis) {
      // ellipsis_rank is input rank minus the number of named labels.
      ellipsis_rank = input_rank -
                      static_cast<int64_t>(splits[0].size() + splits[1].size());
      if (ellipsis_rank < 0) {
        return InvalidArgument(
            "Too few dimensions in the input for the given einsum config.");
      }
    }
    for (char d : splits[0]) {
      TF_RETURN_IF_ERROR(maybe_invalid_character(d));
      numeric_config->push_back(static_cast<int64_t>(d));
    }
    if (has_ellipsis) {
      // For input configs, we use the value of ellipsis_rank we just computed.
      // For output config, we use the existing value of ellipsis_rank.
      for (int64_t i = ellipsis_rank; i > 0; --i) {
        numeric_config->push_back(-i);
      }
      for (char d : splits[1]) {
        TF_RETURN_IF_ERROR(maybe_invalid_character(d));
        numeric_config->push_back(static_cast<int64_t>(d));
      }
    }
    return ellipsis_rank;
  };

  TF_ASSIGN_OR_RETURN(
      const int64_t x_ellipsis_rank,
      string_config_to_numeric(main_split[0],
                               /*is_input_config=*/true, x_rank,
                               /*ellipsis_rank=*/0, &einsum_config_numeric[0]));

  std::vector<absl::string_view> y_output_split =
      absl::StrSplit(main_split[1], "->");
  if (y_output_split.size() != 2) {
    return InvalidArgument("Expected one \"->\" in einsum_config.");
  }

  TF_ASSIGN_OR_RETURN(
      const int64_t y_ellipsis_rank,
      string_config_to_numeric(y_output_split[0],
                               /*is_input_config=*/true, y_rank,
                               /*ellipsis_rank=*/0, &einsum_config_numeric[1]));

  // Replace ellipsis in output_config with numeric labels with the same
  // ellipsis rank as in the inputs.
  // Note: This implementation doesn't support different-rank broadcasting.
  TF_ASSIGN_OR_RETURN(
      std::ignore,
      string_config_to_numeric(
          y_output_split[1], /*is_input_config=*/false,
          /*input_rank=*/0,
          /*ellipsis_rank=*/std::max(x_ellipsis_rank, y_ellipsis_rank),
          &einsum_config_numeric[2]));
  return einsum_config_numeric;
}

std::string NormalizeEinsumString(absl::string_view einsum_config) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("einsum_config: \"" + std::string(einsum_config.data(), einsum_config.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_22(mht_22_v, 922, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "NormalizeEinsumString");

  if (einsum_config.find("->") != einsum_config.npos) {
    return "";
  }
  bool has_ellipsis = einsum_config.find("...") != einsum_config.npos;
  std::map<char, int64_t> chars;
  for (char c : einsum_config) {
    if (absl::ascii_isalpha(c)) {
      ++chars[c];
    }
  }
  std::string new_config(einsum_config.begin(), einsum_config.end());
  new_config.append("->");
  if (has_ellipsis) {
    new_config.append("...");
  }
  for (auto p : chars) {
    if (p.second == 1) {
      new_config.push_back(p.first);
    }
  }
  return new_config;
}

XlaOp Einsum(XlaOp x, XlaOp y, absl::string_view einsum_config,
             PrecisionConfig::Precision precision,
             absl::optional<PrimitiveType> preferred_element_type) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("einsum_config: \"" + std::string(einsum_config.data(), einsum_config.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_23(mht_23_v, 952, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "Einsum");

  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    auto new_config = NormalizeEinsumString(einsum_config);
    if (!new_config.empty()) {
      return Einsum(x, y, new_config, precision, preferred_element_type);
    }
    TF_ASSIGN_OR_RETURN(Shape x_shape, builder->GetShape(x));
    TF_ASSIGN_OR_RETURN(Shape y_shape, builder->GetShape(y));
    TF_ASSIGN_OR_RETURN(
        auto einsum_config_numeric,
        ParseEinsumString(einsum_config, x_shape.rank(), y_shape.rank()));
    return Einsum(x, einsum_config_numeric[0], y, einsum_config_numeric[1],
                  einsum_config_numeric[2], precision, preferred_element_type);
  });
}

XlaOp Einsum(XlaOp x, absl::string_view einsum_config,
             PrecisionConfig::Precision precision) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("einsum_config: \"" + std::string(einsum_config.data(), einsum_config.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_24(mht_24_v, 974, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "Einsum");

  return Einsum(ScalarLike(x, 1), x, absl::StrCat(",", einsum_config),
                precision);
}

XlaOp TransposeInMinorDims(XlaOp x) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_25(mht_25_v, 982, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "TransposeInMinorDims");

  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64_t n_dims = shape.rank();
    TF_RET_CHECK(n_dims >= 2);
    std::vector<int64_t> permutation(n_dims);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::swap(permutation[n_dims - 1], permutation[n_dims - 2]);
    return Transpose(x, permutation);
  });
}

XlaOp MaybeTransposeInMinorDims(XlaOp x, bool transpose) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSmatrixDTcc mht_26(mht_26_v, 998, "", "./tensorflow/compiler/xla/client/lib/matrix.cc", "MaybeTransposeInMinorDims");

  return transpose ? TransposeInMinorDims(x) : x;
}

}  // namespace xla
