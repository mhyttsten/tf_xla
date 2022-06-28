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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSdynamic_shaped_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSdynamic_shaped_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSdynamic_shaped_opsDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/lib/dynamic_shaped_ops.h"

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
namespace xla {
namespace {

// Given a list of shapes, create a shape whose dimensions are largest among all
// inputs.
//
// e.g.,
// shape_a = f32[10, 50]
// shape_b = f32[100, 10]
//
// result = f32[max(shape_a[0], shape_b[0]), max(shape_a[1], shape_b[1])]
//        = f32[100, 50]
Shape FindMaxShape(absl::Span<const Shape*> shapes) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSdynamic_shaped_opsDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/xla/client/lib/dynamic_shaped_ops.cc", "FindMaxShape");

  CHECK(!shapes.empty());
  if (shapes[0]->IsTuple()) {
    // Recurse into sub-element.
    std::vector<Shape> results;
    results.reserve(shapes[0]->tuple_shapes_size());
    for (int i = 0; i < shapes[0]->tuple_shapes_size(); ++i) {
      std::vector<const Shape*> subshapes;
      subshapes.reserve(shapes.size());
      for (int64_t j = 0; j < shapes.size(); ++j) {
        subshapes.push_back(&shapes[j]->tuple_shapes(i));
      }
      results.push_back(FindMaxShape(absl::MakeSpan(subshapes)));
    }
    return ShapeUtil::MakeTupleShape(results);
  }
  Shape result = *shapes[0];

  for (const Shape* shape : shapes) {
    CHECK(result.rank() == shape->rank());
    for (int64_t dim = 0; dim < result.rank(); ++dim) {
      if (shape->dimensions(dim) > result.dimensions(dim)) {
        result.set_dimensions(dim, shape->dimensions(dim));
      }
    }
  }
  return result;
}

XlaOp ReconsileBranchDifference(const Shape& left_branch_shape,
                                const Shape& right_branch_shape,
                                XlaOp left_root) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSdynamic_shaped_opsDTcc mht_1(mht_1_v, 238, "", "./tensorflow/compiler/xla/client/lib/dynamic_shaped_ops.cc", "ReconsileBranchDifference");

  if (left_branch_shape.IsTuple()) {
    // Invariant sanity check -- Left branch and right branch need to have
    // compatible shapes.
    CHECK(right_branch_shape.IsTuple() &&
          left_branch_shape.tuple_shapes_size() ==
              right_branch_shape.tuple_shapes_size());
    // Recurse into sub-element.
    std::vector<XlaOp> results;
    results.reserve(left_branch_shape.tuple_shapes_size());
    for (int i = 0; i < left_branch_shape.tuple_shapes_size(); ++i) {
      XlaOp sub_tuple = GetTupleElement(left_root, i);
      XlaOp elem = ReconsileBranchDifference(left_branch_shape.tuple_shapes(i),
                                             right_branch_shape.tuple_shapes(i),
                                             sub_tuple);
      results.push_back(elem);
    }
    return Tuple(left_root.builder(), results);
  }
  XlaOp result = left_root;
  // Invariant sanity check -- Left branch and right branch need to have
  // compatible shapes.
  CHECK(!right_branch_shape.IsTuple());
  CHECK(left_branch_shape.rank() == right_branch_shape.rank());
  for (int64_t dim = 0; dim < left_branch_shape.rank(); ++dim) {
    XlaOp original_dim = GetDimensionSize(result, dim);
    if (left_branch_shape.dimensions(dim) <
        right_branch_shape.dimensions(dim)) {
      int64_t diff = right_branch_shape.dimensions(dim) -
                     left_branch_shape.dimensions(dim);

      result = PadInDim(
          result, Zero(result.builder(), left_branch_shape.element_type()), dim,
          0, diff);
    }
    if (left_branch_shape.dimensions(dim) !=
        right_branch_shape.dimensions(dim)) {
      result = SetDimensionSize(result, original_dim, dim);
    }
  }
  return result;
}
}  // namespace
XlaOp DynamicConditional(XlaBuilder* builder, XlaOp predicate,
                         XlaOp true_operand,
                         const XlaComputation& true_computation,
                         XlaOp false_operand,
                         const XlaComputation& false_computation) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSdynamic_shaped_opsDTcc mht_2(mht_2_v, 288, "", "./tensorflow/compiler/xla/client/lib/dynamic_shaped_ops.cc", "DynamicConditional");

  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    auto true_shape =
        true_computation.GetProgramShape().ConsumeValueOrDie().result();

    auto false_shape =
        false_computation.GetProgramShape().ConsumeValueOrDie().result();

    if (ShapeUtil::Compatible(true_shape, false_shape)) {
      return xla::Conditional(predicate, true_operand, true_computation,
                              false_operand, false_computation);
    }

    auto reconsile_branch = [](const Shape& root_shape,
                               const Shape& operand_shape,
                               const Shape& reference_root_shape,
                               const XlaComputation& computation) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSdynamic_shaped_opsDTcc mht_3(mht_3_v, 307, "", "./tensorflow/compiler/xla/client/lib/dynamic_shaped_ops.cc", "lambda");

      xla::XlaBuilder builder("dynamic_builder");
      auto param = xla::Parameter(&builder, 0, operand_shape, "param");
      auto call = Call(&builder, computation, {param});

      ReconsileBranchDifference(root_shape, reference_root_shape, call);
      return builder.Build();
    };
    TF_ASSIGN_OR_RETURN(
        auto true_computation_rewritten,
        reconsile_branch(true_shape,
                         builder->GetShape(true_operand).ValueOrDie(),
                         false_shape, true_computation));

    TF_ASSIGN_OR_RETURN(
        auto false_computation_rewritten,
        reconsile_branch(false_shape,
                         builder->GetShape(false_operand).ValueOrDie(),
                         true_shape, false_computation));
    return xla::Conditional(predicate, true_operand, true_computation_rewritten,
                            false_operand, false_computation_rewritten);
  });
}

XlaOp DynamicConditional(
    XlaBuilder* builder, XlaOp branch_index,
    absl::Span<const XlaComputation* const> branch_computations,
    absl::Span<const XlaOp> branch_operands) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSdynamic_shaped_opsDTcc mht_4(mht_4_v, 337, "", "./tensorflow/compiler/xla/client/lib/dynamic_shaped_ops.cc", "DynamicConditional");

  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    std::vector<Shape> root_shapes;
    root_shapes.reserve(branch_computations.size());
    for (int64_t i = 0; i < branch_computations.size(); ++i) {
      TF_ASSIGN_OR_RETURN(auto program_shape,
                          branch_computations[i]->GetProgramShape());
      root_shapes.push_back(program_shape.result());
    }
    TF_RET_CHECK(!root_shapes.empty());
    bool all_shapes_compatible =
        absl::c_all_of(root_shapes, [&](const Shape& shape) {
          return ShapeUtil::Compatible(root_shapes[0], shape);
        });
    if (all_shapes_compatible) {
      // All shapes are compatible, fall back to static case.
      return xla::Conditional(branch_index, branch_computations,
                              branch_operands);
    }

    std::vector<const Shape*> root_shapes_ptrs;
    root_shapes_ptrs.reserve(root_shapes.size());
    for (int64_t i = 0; i < root_shapes.size(); ++i) {
      root_shapes_ptrs.push_back(&root_shapes[i]);
    }

    Shape max_shape = FindMaxShape(absl::MakeSpan(root_shapes_ptrs));

    auto reconsile_branch = [](const Shape& root_shape,
                               const Shape& operand_shape,
                               const Shape& reference_root_shape,
                               const XlaComputation& computation) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPSdynamic_shaped_opsDTcc mht_5(mht_5_v, 371, "", "./tensorflow/compiler/xla/client/lib/dynamic_shaped_ops.cc", "lambda");

      xla::XlaBuilder builder("dynamic_builder");
      auto param = xla::Parameter(&builder, 0, operand_shape, "param");
      auto call = Call(&builder, computation, {param});

      ReconsileBranchDifference(root_shape, reference_root_shape, call);
      return builder.Build();
    };
    std::vector<XlaComputation> rewritten_computations;
    rewritten_computations.reserve(branch_computations.size());

    for (int64_t i = 0; i < branch_computations.size(); ++i) {
      TF_ASSIGN_OR_RETURN(Shape branch_operand_shape,
                          builder->GetShape(branch_operands[i]));

      TF_ASSIGN_OR_RETURN(auto rewritten,
                          reconsile_branch(root_shapes[i], branch_operand_shape,
                                           max_shape, *branch_computations[i]));
      rewritten_computations.push_back(std::move(rewritten));
    }
    std::vector<const XlaComputation*> rewritten_computation_ptrs;
    rewritten_computation_ptrs.reserve(branch_computations.size());
    for (int64_t i = 0; i < branch_computations.size(); ++i) {
      rewritten_computation_ptrs.push_back(&rewritten_computations[i]);
    }
    return xla::Conditional(branch_index, rewritten_computation_ptrs,
                            branch_operands);
  });
}

StatusOr<XlaOp> SetDimensionSizeWithRebound(ValueInference* value_inference,
                                            XlaOp operand, XlaOp dimension_size,
                                            int64_t dimension) {
  auto inferred_bound_status_or = value_inference->AnalyzeConstant(
      dimension_size, xla::ValueInferenceMode::kUpperBound);

  auto dynamism_status_or = value_inference->AnalyzeIsDynamic(dimension_size);
  TF_RETURN_IF_ERROR(inferred_bound_status_or.status());
  TF_RETURN_IF_ERROR(dynamism_status_or.status());
  if (inferred_bound_status_or->AllValid()) {
    int64_t inferred_bound = inferred_bound_status_or->Get<int32_t>({}).value();
    TF_ASSIGN_OR_RETURN(auto* shape_ptr,
                        operand.builder()->GetShapePtr(operand));
    // Found a tighter bound, do a slice.
    if (shape_ptr->dimensions(dimension) > inferred_bound) {
      operand = xla::SliceInDim(operand, 0, inferred_bound, 1, dimension);
    }
  }
  if (dynamism_status_or->Get<bool>({})) {
    // dimension size is dynamic, make output dynamic by calling set dimension
    // size.
    operand = xla::SetDimensionSize(operand, dimension_size, dimension);
  }
  return operand;
}

StatusOr<XlaOp> SetAllDimensionSizes(ValueInference* value_inference,
                                     XlaOp operand, XlaOp size_vector) {
  auto builder = value_inference->builder();
  TF_RETURN_IF_ERROR(builder->GetCurrentStatus());
  TF_ASSIGN_OR_RETURN(auto shape_ptr, builder->GetShapePtr(operand));

  for (int64_t i = 0; i < shape_ptr->rank(); ++i) {
    // If a dimension is dynamic, call set-dimension-size on the output.
    auto dim_size = xla::Slice(size_vector, {i}, {i + 1}, {1});
    dim_size = xla::Reshape(dim_size, {});
    dim_size = xla::ConvertElementType(dim_size, xla::S32);
    TF_ASSIGN_OR_RETURN(auto dynamism,
                        value_inference->AnalyzeIsDynamic(dim_size));
    if (dynamism.Get<bool>({})) {
      operand = xla::SetDimensionSize(operand, dim_size, i);
    }
  }
  return operand;
}
}  // namespace xla
