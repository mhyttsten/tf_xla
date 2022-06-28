/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_FRAMEWORK_COMMON_SHAPE_FNS_H_
#define TENSORFLOW_CORE_FRAMEWORK_COMMON_SHAPE_FNS_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fnsDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fnsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fnsDTh() {
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


#include <array>

#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

namespace shape_inference {

// Like GetWindowedOutputSize, but deals with DimensionHandles. Does not support
// EXPLICIT padding.
Status GetWindowedOutputSizeFromDims(InferenceContext* c,
                                     DimensionHandle input_size,
                                     DimensionOrConstant filter_size,
                                     int64_t stride, Padding padding_type,
                                     DimensionHandle* output_size);

// The V2 version computes the same outputs with arbitrary dilation_rate, and
// supports EXPLICIT padding. For detailed equations, refer to the comments
// for GetWindowedOutputSizeV2(). The 'padding_before' and 'padding_after'
// parameters are only used if padding_type == EXPLICIT.
Status GetWindowedOutputSizeFromDimsV2(
    InferenceContext* c, DimensionHandle input_size,
    DimensionOrConstant filter_size, int64_t dilation_rate, int64_t stride,
    Padding padding_type, int64_t padding_before, int64_t padding_after,
    DimensionHandle* output_size);

// Transfers shape of input(0) to output(0).
Status UnchangedShape(shape_inference::InferenceContext* c);

// Transfers shape of input(0) to output(0), after asserting its rank is <rank>.
inline Status UnchangedShapeWithRank(shape_inference::InferenceContext* c,
                                     int32_t rank) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fnsDTh mht_0(mht_0_v, 220, "", "./tensorflow/core/framework/common_shape_fns.h", "UnchangedShapeWithRank");

  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), rank, &out));
  c->set_output(0, out);
  return Status::OK();
}

// Transfers shape of input(0) to output(0), after asserting its rank >= <rank>.
inline Status UnchangedShapeWithRankAtLeast(
    shape_inference::InferenceContext* c, int32_t rank) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fnsDTh mht_1(mht_1_v, 232, "", "./tensorflow/core/framework/common_shape_fns.h", "UnchangedShapeWithRankAtLeast");

  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), rank, &out));
  c->set_output(0, out);
  return Status::OK();
}

// Transfers shape of input(0) to output(0), after asserting its rank <= <rank>.
inline Status UnchangedShapeWithRankAtMost(shape_inference::InferenceContext* c,
                                           int32_t rank) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fnsDTh mht_2(mht_2_v, 244, "", "./tensorflow/core/framework/common_shape_fns.h", "UnchangedShapeWithRankAtMost");

  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), rank, &out));
  c->set_output(0, out);
  return Status::OK();
}

// Shape function for use with ops no outputs.
inline Status NoOutputs(shape_inference::InferenceContext* c) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fnsDTh mht_3(mht_3_v, 255, "", "./tensorflow/core/framework/common_shape_fns.h", "NoOutputs");

  return Status::OK();
}

// Shape function for ops that output a single scalar value.
inline Status ScalarShape(shape_inference::InferenceContext* c) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fnsDTh mht_4(mht_4_v, 263, "", "./tensorflow/core/framework/common_shape_fns.h", "ScalarShape");

  c->set_output(0, c->Scalar());
  return Status::OK();
}

// Shape function for binary ops where both inputs and the output match.
inline Status MergeBothInputsShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fnsDTh mht_5(mht_5_v, 272, "", "./tensorflow/core/framework/common_shape_fns.h", "MergeBothInputsShapeFn");

  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(1), &out));
  c->set_output(0, out);
  return Status::OK();
}

// Shape function for dataset iterators.
Status DatasetIteratorShape(shape_inference::InferenceContext* c);

// Returns a new shape with the specified dims arranged in the specified
// format. The returned value is owned by this context.
// Note: if format = "FORMAT_NCHW_VECT_C" then C represents the outer_depth.
Status MakeShapeFromFormat(TensorFormat format, DimensionOrConstant N,
                           const std::vector<DimensionOrConstant>& spatial,
                           DimensionOrConstant C, ShapeHandle* out,
                           shape_inference::InferenceContext* context);

// Shape function for MatMul-like operations.
Status MatMulShape(shape_inference::InferenceContext* c);

// Shape function for Batched MatMul-like operations with broadcasting across
// batch dimensions.
Status BatchMatMulV2Shape(shape_inference::InferenceContext* c);

// Shape function for BatchMatMul-like operations
Status BatchMatMulShape(shape_inference::InferenceContext* c);

// Shape function for Einsum.
Status EinsumShape(shape_inference::InferenceContext* c);

// Shape function for BiasAdd-like operations.
Status BiasAddShape(shape_inference::InferenceContext* c);

// Shape function for BiasAddGrad-like operations.
Status BiasAddGradShape(shape_inference::InferenceContext* c);

// Shape function for Conv2D-like operations that support explicit padding.
Status Conv2DShapeWithExplicitPadding(shape_inference::InferenceContext* c);

// Shape function for Conv2D-like operations that do not support explicit
// padding.
Status Conv2DShape(shape_inference::InferenceContext* c);

// Shape function for Conv3D-like operations.
Status Conv3DShape(shape_inference::InferenceContext* c);

// Shape function for DepthwiseConv2D-like operations that support explicit
// padding.
Status DepthwiseConv2DNativeShapeWithExplicitPadding(
    shape_inference::InferenceContext* c);

// Shape function for DepthwiseConv2D-like operations that do not support
// explicit padding.
Status DepthwiseConv2DNativeShape(shape_inference::InferenceContext* c);

// Shape function for Conv2DBackpropInput.
Status Conv2DBackpropInputShape(shape_inference::InferenceContext* c);

// Shape function for Conv2DBackpropFilterWithBias.
Status Conv2DBackpropFilterWithBiasShape(shape_inference::InferenceContext* c);

// Shape function for AvgPool-like operations.
Status AvgPoolShape(shape_inference::InferenceContext* c);

// Shape function for AvgPoolGrad-like operations.
Status AvgPoolGradShape(shape_inference::InferenceContext* c);

// Shape function for FusedBatchNorm and FusedBatchNormV2 operations.
Status FusedBatchNormShape(shape_inference::InferenceContext* c);

// Shape function for FusedBatchNormV3 operations.
Status FusedBatchNormV3Shape(shape_inference::InferenceContext* c);

// Shape function for _FusedBatchNormEx operations.
Status FusedBatchNormExShape(shape_inference::InferenceContext* c);

// Shape function for FusedBatchNormGrad and FusedBatchNormGradV2 operations.
Status FusedBatchNormGradShape(shape_inference::InferenceContext* c);

// Shape function for _FusedBatchNormGradEx operations.
Status FusedBatchNormGradExShape(shape_inference::InferenceContext* c);

// Shape function for MatrixDiagPartV2 and MatrixDiagPartV3 operations.
Status MatrixDiagPartV2Shape(shape_inference::InferenceContext* c);

// Shape function for MatrixDiagV2 and MatrixDiagV3 operations.
Status MatrixDiagV2Shape(shape_inference::InferenceContext* c);

// Shape function for MatrixSetDiagV2 and MatrixSetDiagV3 operations.
Status MatrixSetDiagV2Shape(shape_inference::InferenceContext* c);

// Shape function for MaxPool-like operations that support explicit padding.
Status MaxPoolShapeWithExplicitPadding(shape_inference::InferenceContext* c);

// Shape function for MaxPool-like operations that do not support explicit
// padding.
Status MaxPoolShape(shape_inference::InferenceContext* c);

// Shape function for MaxPoolV2-like operations.
Status MaxPoolV2Shape(shape_inference::InferenceContext* c, int num_inputs);

// Shape function for MaxPoolGrad-like operations.
Status MaxPoolGradShape(shape_inference::InferenceContext* c);

// Shape function for 3D Pooling operations.
Status Pool3DShape(shape_inference::InferenceContext* c);

// Shape function for MaxPool3DGrad-like operations.
Status MaxPool3DGradShape(shape_inference::InferenceContext* c);

// Shape function for AvgPool3DGrad-like operations.
Status AvgPool3DGradShape(shape_inference::InferenceContext* c);

// Shape function for use with ops whose output shapes are unknown.
Status UnknownShape(shape_inference::InferenceContext* c);

// Shape function for reduction operations.
Status ReductionShape(shape_inference::InferenceContext* c);

// Shape function for unsorted segment operations.
Status UnsortedSegmentReductionShapeFn(InferenceContext* c);

// Shape function for concat operations.
// <num_inputs_to_concat> is the number of inputs to concatenate and are taken
// from inputs
// [1,num_inputs_to_concat] of the op.  Input 0 is the concat_dim input.
Status ConcatShape(shape_inference::InferenceContext* c,
                   int num_inputs_to_concat);

// Shape function for concat operations.
Status ConcatV2Shape(shape_inference::InferenceContext* c);

Status QuantizedConcatV2Shape(InferenceContext* c, int num_inputs_to_concat);

// Shape function for binary operators that broadcast their inputs
// and with output to output_index.
// Note: out cannot be NULL.
Status BroadcastBinaryOpOutputShapeFnHelper(InferenceContext* c,
                                            ShapeHandle shape_x,
                                            ShapeHandle shape_y,
                                            bool incompatible_shape_error,
                                            ShapeHandle* out);

// Shape function for binary operators that broadcast their inputs
// and with output to output_index.
inline Status BroadcastBinaryOpOutputShapeFn(InferenceContext* c,
                                             int output_index) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fnsDTh mht_6(mht_6_v, 422, "", "./tensorflow/core/framework/common_shape_fns.h", "BroadcastBinaryOpOutputShapeFn");

  ShapeHandle out;
  TF_RETURN_IF_ERROR(BroadcastBinaryOpOutputShapeFnHelper(
      c, c->input(0), c->input(1), true, &out));
  c->set_output(output_index, out);
  return Status::OK();
}

// Shape function for binary operators that broadcast their inputs.
// Tested by ops/math_ops_test.cc.
inline Status BroadcastBinaryOpShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fnsDTh mht_7(mht_7_v, 435, "", "./tensorflow/core/framework/common_shape_fns.h", "BroadcastBinaryOpShapeFn");

  return BroadcastBinaryOpOutputShapeFn(c, 0);
}

// Shape function for random operations.
Status RandomShape(shape_inference::InferenceContext* c);

// Shape function for Slice operations.
Status SliceShape(shape_inference::InferenceContext* c);

// Validates the 3 component tensors of a sparse tensor have the proper
// shapes. This mimics SparseTensor.__init__ in python/framework/ops.py.
Status ValidateSparseTensor(InferenceContext* c, ShapeHandle indices_shape,
                            ShapeHandle values_shape, ShapeHandle shape_shape);

Status ValidateVariableResourceHandle(
    InferenceContext* c, std::vector<ShapeAndType>* shape_and_type);

// Shape function for GatherNd operations.
Status GatherNdShape(InferenceContext* c);

// Helper shape function for ScatterNd.../TensorScatter... operations.
Status ScatterNdShapeHelper(InferenceContext* c, ShapeHandle indices_shape,
                            ShapeHandle updates_shape, ShapeHandle input_shape);

// Shape function for ops with an explicit "shape" attribute.
Status ExplicitShape(InferenceContext* c);

// Shape function for multiple-output ops with an explicit "shapes" attribute.
Status ExplicitShapes(InferenceContext* c);

// Shape function for SparseReduceMax and SparseReduceSum.
Status SparseReduceShapeFn(InferenceContext* c);

// Shape function for QuantizedConv2D op.
Status QuantizedConv2DShape(InferenceContext* c);

// Shape function for QuantizedAvgPool op
Status QuantizedAvgPoolShape(InferenceContext* c);

// Shape function for QuantizeV2 op
Status QuantizeV2Shape(InferenceContext* c);

// Shape function for ReduceScatter ops
Status ReduceScatterShape(shape_inference::InferenceContext* c);

}  // namespace shape_inference

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_COMMON_SHAPE_FNS_H_
