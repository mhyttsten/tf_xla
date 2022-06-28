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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"

#include <tuple>
#include <vector>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace llvm_ir {

IrArray::Index::Index(absl::Span<llvm::Value* const> multidim,
                      llvm::Value* linear, const Shape& shape,
                      llvm::Type* index_type)
    : Index(multidim, shape, index_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_0(mht_0_v, 209, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::Index::Index");

  CHECK_NE(linear, nullptr);
  linear_ = linear;
}

void IrArray::Index::Delinearize(std::vector<llvm::Value*>* multidim,
                                 llvm::Value* linear, const Shape& shape,
                                 llvm::IRBuilder<>* b) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::Index::Delinearize");

  int64_t divisor = 1;
  const Layout& layout = shape.layout();
  for (int64_t i = 0; i < layout.minor_to_major_size(); ++i) {
    int64_t dimension = layout.minor_to_major(i);
    int64_t size_of_current_dimension = shape.dimensions(dimension);

    // If i is not the last dimension, compute
    //   (linear_index / divisor) % current_dimension.
    // If i is the last dimension, we can skip the mod, because we assume that
    // linear is in bounds.
    //
    // TODO(jlebar): We could add bounds checks here and elsewhere in this file,
    // guarded under some sort of xla-memcheck flag.  This might be particularly
    // useful because cuda-memcheck can't help us much in XLA: Most of our
    // memory lives in one big allocation, so cuda-memcheck can't detect
    // out-of-bounds accesses.
    auto* quot = b->CreateUDiv(linear, GetConstantWithIndexType(divisor));
    if (i < layout.minor_to_major_size() - 1) {
      (*multidim)[dimension] = b->CreateURem(
          quot, GetConstantWithIndexType(size_of_current_dimension));
    } else {
      (*multidim)[dimension] = quot;
    }
    divisor *= size_of_current_dimension;
  }
}

void IrArray::Index::Delinearize(std::vector<llvm::Value*>* multidim,
                                 llvm::Value* linear, const Shape& shape,
                                 absl::Span<llvm::Value*> dynamic_dims,
                                 llvm::IRBuilder<>* b) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_2(mht_2_v, 253, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::Index::Delinearize");

  CHECK_EQ(shape.dimensions_size(), dynamic_dims.size());
  CHECK_EQ(multidim_.size(), shape.rank());
  llvm::Value* divisor = GetConstantWithIndexType(1);
  const Layout& layout = shape.layout();
  for (int64_t i = 0; i < layout.minor_to_major_size(); ++i) {
    int64_t dimension = layout.minor_to_major(i);

    // If i is not the last dimension, compute
    //   (linear_index / divisor) % current_dimension.
    // If i is the last dimension, we can skip the mod, because we assume that
    // linear is in bounds.
    auto* quot = b->CreateUDiv(linear, divisor, "quot");
    if (i < layout.minor_to_major_size() - 1) {
      (*multidim)[dimension] =
          b->CreateURem(quot, dynamic_dims[dimension], "dim_value");
      divisor = b->CreateMul(divisor, dynamic_dims[dimension], "divisor");
    } else {
      (*multidim)[dimension] = quot;
    }
  }
}

IrArray::Index::Index(llvm::Value* linear, const Shape& shape,
                      llvm::IRBuilder<>* b)
    : multidim_(shape.rank()),
      linear_(linear),
      layout_(shape.layout()),
      dims_(shape.dimensions().begin(), shape.dimensions().end()) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_3(mht_3_v, 284, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::Index::Index");

  CHECK_NE(linear, nullptr);
  index_type_ = linear->getType();
  CHECK(LayoutUtil::HasLayout(shape))
      << "Shape " << ShapeUtil::HumanStringWithLayout(shape)
      << " should have a layout.";
  Delinearize(&multidim_, linear, shape, b);
}

IrArray::Index::Index(llvm::Value* linear,
                      absl::Span<llvm::Value* const> multidim,
                      const Shape& shape, llvm::IRBuilder<>* b)
    : multidim_(shape.rank()),
      linear_(linear),
      layout_(shape.layout()),
      dims_(shape.dimensions().begin(), shape.dimensions().end()) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_4(mht_4_v, 302, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::Index::Index");

  CHECK_NE(linear, nullptr);
  index_type_ = linear->getType();
  CHECK_EQ(multidim.size(), shape.rank());
  for (auto dim : multidim) {
    if (dim) {
      CHECK_EQ(dim->getType(), index_type_);
    }
  }
  CHECK(LayoutUtil::HasLayout(shape))
      << "Shape " << ShapeUtil::HumanStringWithLayout(shape)
      << " should have a layout.";
  Delinearize(&multidim_, linear, shape, b);
  for (int i = 0; i < multidim.size(); ++i) {
    if (multidim[i] != nullptr) {
      multidim_[i] = multidim[i];
    }
  }
}

IrArray::Index::Index(llvm::Value* linear, const Shape& shape,
                      absl::Span<llvm::Value*> dynamic_dims,
                      llvm::IRBuilder<>* b)
    : multidim_(shape.rank()),
      linear_(linear),
      layout_(shape.layout()),
      dims_(shape.dimensions().begin(), shape.dimensions().end()) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_5(mht_5_v, 331, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::Index::Index");

  CHECK_NE(linear, nullptr);
  index_type_ = linear->getType();
  CHECK(LayoutUtil::HasLayout(shape))
      << "Shape " << ShapeUtil::HumanStringWithLayout(shape)
      << " should have a layout.";
  Delinearize(&multidim_, linear, shape, dynamic_dims, b);
}

IrArray::Index::Index(absl::Span<llvm::Value* const> multidim,
                      absl::Span<int64_t const> dimensions,
                      llvm::Type* index_type)
    : Index(multidim, ShapeUtil::MakeShape(/*arbitrary*/ PRED, dimensions),
            index_type) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_6(mht_6_v, 347, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::Index::Index");
}

IrArray::Index::Index(absl::Span<llvm::Value* const> multidim,
                      const Shape& shape, llvm::Type* index_type)
    : multidim_(multidim.begin(), multidim.end()),
      linear_(nullptr),
      layout_(shape.layout()),
      dims_(shape.dimensions().begin(), shape.dimensions().end()),
      index_type_(index_type) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_7(mht_7_v, 358, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::Index::Index");

  CHECK_NE(index_type_, nullptr);
  CHECK_EQ(shape.dimensions_size(), multidim.size());
  for (const auto* dim : multidim) {
    CHECK_NE(dim, nullptr);
  }
  CHECK(LayoutUtil::HasLayout(shape))
      << "Shape " << ShapeUtil::HumanStringWithLayout(shape)
      << " should have a layout.";
}

IrArray::IrArray(llvm::Value* base_ptr, Shape shape)
    : base_ptr_(base_ptr), shape_(std::move(shape)) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_8(mht_8_v, 373, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::IrArray");

  TF_CHECK_OK(ShapeUtil::ValidateShape(shape));
  CHECK(base_ptr_->getType()->isPointerTy());
  int depth = 0;
  element_type_ = base_ptr_->getType()->getPointerElementType();
  while (llvm::ArrayType* array_type =
             llvm::dyn_cast<llvm::ArrayType>(element_type_)) {
    element_type_ = array_type->getElementType();
    ++depth;
  }

  if (!shape_.IsArray() || ShapeUtil::IsScalar(shape_)) {
    DCHECK(depth == 1 || depth == 0) << depth;
  } else {
    DCHECK_EQ(depth, shape_.rank()) << shape.ShortDebugString();
  }
}

// Returns whether the given linear index is valid on the given shape.
bool IrArray::Index::LinearValidOnShape(const Shape& a) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_9(mht_9_v, 395, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::Index::LinearValidOnShape");

  auto b = ShapeUtil::MakeShape(a.element_type(), dims_);
  *b.mutable_layout() = layout_;
  return linear_ != nullptr &&
         ShapeUtil::ElementsIn(a) == ShapeUtil::ElementsIn(b) &&
         ShapeUtil::ReshapeIsBitcast(a, b);
}

IrArray::Index IrArray::Index::SourceIndexOfReshape(
    const Shape& output_shape, const Shape& input_shape,
    llvm::IRBuilder<>* builder) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_10(mht_10_v, 408, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::Index::SourceIndexOfReshape");

  CHECK_EQ(multidim_.size(), output_shape.rank());
  std::vector<llvm::Value*> source_multidim_index(
      input_shape.rank(), llvm::UndefValue::get(index_type_));
  auto trivial_reshape =
      ShapeUtil::InsertedOrDeleted1SizedDimensions(input_shape, output_shape);
  if (std::get<0>(trivial_reshape)) {
    // The 1-sized dimensions which only appear in 'input_shape'.
    auto deleted_dims_indices = std::get<1>(trivial_reshape);
    // The 1-sized dimensions which only appear in 'output_shape'.
    auto inserted_dims_indices = std::get<2>(trivial_reshape);

    // This is a two-way merge of 'deleted_dims_indices' with indexing into
    // 'source_multidim_index', and a two-way merge of 'inserted_dims_indices'
    // with indexing into 'multidim_'. When we find a dimension in
    // 'source_multidim_index' which does not belong to 'deleted_dims_indices',
    // we retrieve the corresponding value from 'multidim_' (skipping any
    // indices that appear in 'inserted_dims_indices').
    for (int64_t i = 0, j = 0, k = 0, l = 0; i < source_multidim_index.size();
         ++i) {
      if (j == deleted_dims_indices.size() || deleted_dims_indices[j] > i) {
        // This is a dimension that was preserved. Take the matching value from
        // multidim_.
        while (l < inserted_dims_indices.size() &&
               inserted_dims_indices[l] == k) {
          // Skip 1-sized dimensions.
          ++k;
          ++l;
        }
        source_multidim_index[i] = multidim_[k];
        ++k;
      } else {
        // This is a 1-sized dimension that only appears in the operand.
        source_multidim_index[i] = GetConstantWithIndexType(0);
        ++j;
      }
    }
  } else {
    const auto common_factors =
        CommonFactors(input_shape.dimensions(), output_shape.dimensions());
    // We compute the source indices in each common factor from only the target
    // indices in the same common factor.
    for (ssize_t k = common_factors.size() - 2; k >= 0; --k) {
      absl::Span<int64_t const> dimensions = output_shape.dimensions().subspan(
          common_factors[k].second,
          common_factors[k + 1].second - common_factors[k].second);
      llvm::Value* logical_linear_index =
          Index(absl::Span<llvm::Value* const>(multidim_).subspan(
                    common_factors[k].second,
                    common_factors[k + 1].second - common_factors[k].second),
                dimensions, index_type_)
              .Linearize(dimensions, builder);
      // Delinearizes logical_linear_index for the source array in row-major
      // collapsed order. The first rank-1 indices are the remainder of the
      // linear index by each dimension size.
      for (int64_t i = common_factors[k + 1].first - 1;
           i >= common_factors[k].first; --i) {
        llvm::Value* divisor =
            GetConstantWithIndexType(input_shape.dimensions(i));
        if (input_shape.dimensions(i) == 1) {
          source_multidim_index[i] = GetConstantWithIndexType(0);
        } else if (i == common_factors[k].first) {
          source_multidim_index[i] = logical_linear_index;
        } else {
          source_multidim_index[i] =
              builder->CreateURem(logical_linear_index, divisor);
        }
        logical_linear_index =
            builder->CreateUDiv(logical_linear_index, divisor);
      }
    }
  }

  if (linear() != nullptr && LayoutUtil::HasLayout(input_shape) &&
      LayoutUtil::HasLayout(output_shape) &&
      ShapeUtil::ReshapeIsBitcast(input_shape, output_shape)) {
    return Index(source_multidim_index, linear(), input_shape, index_type_);
  }
  return Index(source_multidim_index, input_shape, index_type_);
}

IrArray::Index IrArray::Index::SourceIndexOfSlice(
    const Shape& operand_shape, absl::Span<const int64_t> starts,
    absl::Span<const int64_t> strides, llvm::IRBuilder<>* builder) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_11(mht_11_v, 494, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::Index::SourceIndexOfSlice");

  std::vector<llvm::Value*> source_multi_index(multidim_.size());
  for (int i = 0; i < multidim_.size(); ++i) {
    int64_t stride = strides[i];
    if (stride != 1) {
      source_multi_index[i] = builder->CreateAdd(
          builder->CreateMul(multidim_[i], GetConstantWithIndexType(stride)),
          GetConstantWithIndexType(starts[i]));
    } else {
      source_multi_index[i] =
          builder->CreateAdd(multidim_[i], GetConstantWithIndexType(starts[i]));
    }
  }
  return Index(source_multi_index, operand_shape, index_type_);
}

IrArray::Index IrArray::Index::SourceIndexOfTranspose(
    const Shape& shape, const Shape& operand_shape,
    absl::Span<const int64_t> dimension_mapping) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_12(mht_12_v, 515, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::Index::SourceIndexOfTranspose");

  std::vector<llvm::Value*> operand_multidim_index =
      PermuteInverse(multidim(), dimension_mapping);

  if (linear() != nullptr && LayoutUtil::HasLayout(operand_shape) &&
      LayoutUtil::HasLayout(shape) &&
      ShapeUtil::TransposeIsBitcast(operand_shape, shape, dimension_mapping)) {
    return Index(operand_multidim_index, linear(), operand_shape, index_type_);
  }

  return Index(operand_multidim_index, operand_shape, index_type_);
}

IrArray::Index IrArray::Index::SourceIndexOfBitcast(
    const Shape& shape, const Shape& operand_shape,
    llvm::IRBuilder<>* builder) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_13(mht_13_v, 533, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::Index::SourceIndexOfBitcast");

  CHECK(LayoutUtil::HasLayout(shape) && LayoutUtil::HasLayout(operand_shape));

  // In case the bitcast is just a reshape, we can use SourceIndexOfReshape()
  // instead. This will reuse linear() if possible, so we don't have to build a
  // new 'linear_index'.
  if (ShapeUtil::ReshapeIsBitcast(operand_shape, shape)) {
    return SourceIndexOfReshape(shape, operand_shape, builder);
  }

  // If we have a linear index, we can definitely use it because we know the
  // operation is a bitcast. This will recompute the multi-dimensional index for
  // the operand based on the linear index.
  if (linear() != nullptr) {
    return Index(linear(), operand_shape, builder);
  }

  // First linearize the index coming from the output of the bitcast. We want
  // the physical index of the element in the buffer. This is like Linearize,
  // but takes the layout into account.
  int64_t scale = 1;
  llvm::Value* linear_index = GetConstantWithIndexType(0);
  for (auto dimension : LayoutUtil::MinorToMajor(shape)) {
    linear_index = builder->CreateAdd(
        linear_index,
        builder->CreateMul(multidim_[dimension],
                           GetConstantWithIndexType(scale), "",
                           /*HasNUW=*/true, /*HasNSW=*/true),
        "", /*HasNUW=*/true, /*HasNSW=*/true);
    scale *= shape.dimensions(dimension);
  }

  return Index(linear_index, operand_shape, builder);
}

IrArray::Index IrArray::Index::SourceIndexOfBroadcast(
    const Shape& shape, const Shape& operand_shape,
    absl::Span<const int64_t> dimension_mapping,
    llvm::IRBuilder<>* builder) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_14(mht_14_v, 574, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::Index::SourceIndexOfBroadcast");

  int64_t rank = operand_shape.rank();
  std::vector<llvm::Value*> source_index(rank);
  for (int64_t i = 0; i < rank; ++i) {
    source_index[i] = multidim_[dimension_mapping[i]];
  }
  if (linear_ == nullptr || !LayoutUtil::HasLayout(operand_shape) ||
      !LayoutUtil::HasLayout(shape) || rank == 1) {
    return Index(source_index, operand_shape, index_type_);
  }
  // High-level idea: we can reuse the linear index if the broadcasted
  // dimensions are contiguous, and this part of the operation is a bitcast.
  // The other dimensions can be masked out with a div and a mod operation.
  std::vector<int64_t> logical_to_physical =
      LayoutUtil::MakeLogicalToPhysical(shape.layout());
  int64_t output_rank = shape.rank();
  // The minimum physical dimension that is broadcasted.
  int64_t min_broadcasted_dimension = output_rank;
  // The maximum physical dimension that is broadcasted.
  int64_t max_broadcasted_dimension = -1;
  for (int64_t i = 0; i < rank; ++i) {
    int64_t physical_dim = logical_to_physical[dimension_mapping[i]];
    min_broadcasted_dimension =
        std::min(min_broadcasted_dimension, physical_dim);
    max_broadcasted_dimension =
        std::max(max_broadcasted_dimension, physical_dim);
  }
  bool contiguous_broadcast_dimensions =
      max_broadcasted_dimension - min_broadcasted_dimension == rank - 1;
  if (!contiguous_broadcast_dimensions) {
    return Index(source_index, operand_shape, index_type_);
  }
  // Check if the mapped dimensions are a bitcast.
  std::vector<int64_t> operand_logical_to_physical =
      LayoutUtil::MakeLogicalToPhysical(operand_shape.layout());
  for (int64_t i = 0; i < rank; ++i) {
    if (operand_logical_to_physical[i] !=
        logical_to_physical[dimension_mapping[i]] - min_broadcasted_dimension) {
      return Index(source_index, operand_shape, index_type_);
    }
  }
  llvm::Value* linear = linear_;
  int64_t divisor = 1;
  for (int64_t i = max_broadcasted_dimension + 1; i < output_rank; ++i) {
    divisor *= shape.dimensions(LayoutUtil::Major(shape.layout(), i));
  }
  if (divisor > 1) {
    linear = builder->CreateUDiv(linear, GetConstantWithIndexType(divisor));
  }
  if (min_broadcasted_dimension > 0) {
    int64_t mod = 1;
    for (int64_t i = min_broadcasted_dimension; i <= max_broadcasted_dimension;
         ++i) {
      mod *= shape.dimensions(LayoutUtil::Major(shape.layout(), i));
    }
    linear = builder->CreateURem(linear, GetConstantWithIndexType(mod));
  }
  return Index(source_index, linear, operand_shape, index_type_);
}

llvm::Value* IrArray::Index::Linearize(absl::Span<const int64_t> dimensions,
                                       llvm::IRBuilder<>* builder) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_15(mht_15_v, 638, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::Index::Linearize");

  // Each dimension is multiplied by the product of the sizes of all
  // earlier dimensions and added to the accumulator logical_linear_index.
  CHECK_EQ(size(), dimensions.size());
  llvm::Value* logical_linear_index = GetConstantWithIndexType(0);
  int64_t multiplier = 1;
  for (ssize_t i = size() - 1; i >= 0; --i) {
    llvm::Value* addend =
        builder->CreateMul((*this)[i], GetConstantWithIndexType(multiplier), "",
                           /*HasNUW=*/true, /*HasNSW=*/true);
    addend = builder->CreateZExtOrTrunc(addend, index_type_);
    logical_linear_index = builder->CreateAdd(logical_linear_index, addend, "",
                                              /*HasNUW=*/true, /*HasNSW=*/true);
    multiplier *= dimensions[i];
  }
  return logical_linear_index;
}

llvm::Value* IrArray::Index::Linearize(
    const std::vector<llvm::Value*>& dynamic_dims,
    llvm::IRBuilder<>* builder) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_16(mht_16_v, 661, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::Index::Linearize");

  // Each dimension is multiplied by the product of the sizes of all
  // earlier dimensions and added to the accumulator logical_linear_index.
  CHECK_EQ(size(), dynamic_dims.size());
  llvm::Value* logical_linear_index = GetConstantWithIndexType(0);
  llvm::Value* multiplier = GetConstantWithIndexType(1);
  for (ssize_t i = size() - 1; i >= 0; --i) {
    llvm::Value* addend = builder->CreateMul((*this)[i], multiplier, "",
                                             /*HasNUW=*/true, /*HasNSW=*/true);
    addend = builder->CreateZExtOrTrunc(addend, index_type_);
    logical_linear_index = builder->CreateAdd(logical_linear_index, addend, "",
                                              /*HasNUW=*/true, /*HasNSW=*/true);
    if (i) {
      multiplier = builder->CreateMul(multiplier, dynamic_dims[i],
                                      /*Name=*/"multiplier");
    }
  }
  return logical_linear_index;
}

llvm::Value* IrArray::EmitArrayElementAddress(const IrArray::Index& index,
                                              llvm::IRBuilder<>* b,
                                              absl::string_view name,
                                              bool use_linear_index) const {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_17(mht_17_v, 688, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::EmitArrayElementAddress");

  if (ShapeUtil::IsScalar(shape_)) {
    // Special handling of scalars: a scalar pretends to have the same value for
    // every index, thus effectively implementing broadcasting of its value
    // over higher-rank arrays.
    return base_ptr_;
  }
  CHECK_EQ(index.size(), shape_.rank());
  CHECK(index.ShapeIsCompatible(shape_))
      << "Shape " << index.AsShapeWithType(shape_.element_type()).ToString(true)
      << " is not compatible with " << shape_.ToString(true);

  if (use_linear_index && index.LinearValidOnShape(shape_)) {
    llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();
    llvm::Type* type = PrimitiveTypeToIrType(shape_.element_type(), module);
    return b->CreateInBoundsGEP(
        type, b->CreateBitCast(base_ptr_, type->getPointerTo()), index.linear(),
        llvm_ir::AsStringRef(name));
  }

  std::vector<llvm::Value*> actual_index;
  for (int64_t i = 0; i < index.size(); ++i) {
    // When dimension i is of size 1, LLVM optimization is able to replace
    // index[i] with 0. However, setting index[i] to 0 here still allows LLVM to
    // produce better code in some cases.
    auto dim = shape_.dimensions(i);
    actual_index.push_back(
        dim == 1 ? llvm::ConstantInt::get(index[i]->getType(), 0) : index[i]);
  }

  // "base_ptr_" has the type of "<ir_type_for_its_shape>*"
  // (e.g. [3 x [2 x float]]*). Therefore, the address of the indexed element
  // should be computed by
  //
  //   getelementptr base_ptr_, 0, most major index, ..., most minor index
  CHECK_GT(index.size(), 0);
  std::vector<llvm::Value*> gep_indices(
      1, llvm::ConstantInt::get(index[0]->getType(), 0));
  for (int64_t i = 0; i < LayoutUtil::MinorToMajor(shape_).size(); ++i) {
    int64_t dimension = LayoutUtil::Major(shape_.layout(), i);
    gep_indices.push_back(actual_index[dimension]);
  }
  return b->CreateInBoundsGEP(base_ptr_->getType()->getPointerElementType(),
                              base_ptr_, gep_indices,
                              llvm_ir::AsStringRef(name));
}

void IrArray::AnnotateLoadStoreInstructionWithMetadata(
    llvm::Instruction* instruction) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_18(mht_18_v, 739, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::AnnotateLoadStoreInstructionWithMetadata");

  CHECK(llvm::isa<llvm::LoadInst>(instruction) ||
        llvm::isa<llvm::StoreInst>(instruction));
  CHECK(!llvm::isa<llvm::StoreInst>(instruction) || !is_invariant_)
      << "Trying to create a store to an invariant IRArray.";

  for (const auto& kind_md_pair : metadata_) {
    instruction->setMetadata(kind_md_pair.first, kind_md_pair.second);
  }
}

llvm::Value* IrArray::EmitReadArrayElement(const Index& index,
                                           llvm::IRBuilder<>* b,
                                           absl::string_view name,
                                           bool use_linear_index) const {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_19(mht_19_v, 757, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::EmitReadArrayElement");

  llvm::Value* element_address =
      EmitArrayElementAddress(index, b, name, use_linear_index);
  llvm::LoadInst* load =
      b->CreateLoad(element_address->getType()->getPointerElementType(),
                    element_address, llvm_ir::AsStringRef(name));
  AnnotateLoadStoreInstructionWithMetadata(load);
  return load;
}

void IrArray::EmitWriteArrayElement(const Index& index, llvm::Value* value,
                                    llvm::IRBuilder<>* b,
                                    bool use_linear_index) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_20(mht_20_v, 772, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::EmitWriteArrayElement");

  llvm::Value* element_address =
      EmitArrayElementAddress(index, b, "", use_linear_index);
  llvm::StoreInst* store = b->CreateStore(value, element_address);
  AnnotateLoadStoreInstructionWithMetadata(store);
}

IrArray IrArray::CastToShape(const Shape& new_shape,
                             llvm::IRBuilder<>* b) const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_21(mht_21_v, 783, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::CastToShape");

  llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();
  llvm::Type* new_ir_type = llvm_ir::ShapeToIrType(new_shape, module);
  IrArray new_irarray(
      b->CreatePointerCast(base_ptr_, new_ir_type->getPointerTo()), new_shape);
  new_irarray.metadata_ = metadata_;
  return new_irarray;
}

bool IrArray::Index::ShapeIsCompatible(const Shape& a, const Shape& b) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_22(mht_22_v, 795, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "IrArray::Index::ShapeIsCompatible");

  // Compute strides for two sides of the comparison. Sometimes different shapes
  // give the same strides:
  //   [10, 20, 30, 1]{3,2,1,0} vs [10, 20, 1, 30]{3,2,1,0}
  // which should be considered compatible.
  const auto get_strides = [](const Shape& shape) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSir_arrayDTcc mht_23(mht_23_v, 803, "", "./tensorflow/compiler/xla/service/llvm_ir/ir_array.cc", "lambda");

    int rank = shape.dimensions().size();
    int64_t stride = 1;
    std::vector<int64_t> strides;
    for (int i = 0; i < rank; i++) {
      auto dim = shape.dimensions(shape.layout().minor_to_major(i));
      if (dim != 1) {
        stride *= dim;
        strides.push_back(stride);
      }
    }
    return strides;
  };

  return get_strides(a) == get_strides(b);
}

}  // namespace llvm_ir
}  // namespace xla
