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
class MHTracer_DTPStensorflowPScorePSopsPSdata_flow_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSdata_flow_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSdata_flow_opsDTcc() {
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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

Status DequeueManyV2Shape(InferenceContext* c, ShapeHandle n_shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSdata_flow_opsDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/ops/data_flow_ops.cc", "DequeueManyV2Shape");

  auto* t = c->input_handle_shapes_and_types(0);
  if (t != nullptr && t->size() == c->num_outputs()) {
    for (int i = 0; i < c->num_outputs(); ++i) {
      ShapeHandle combined_shape;
      TF_RETURN_IF_ERROR(
          c->Concatenate(n_shape, (*t)[i].shape, &combined_shape));
      c->set_output(i, combined_shape);
    }
    return Status::OK();
  } else {
    return shape_inference::UnknownShape(c);
  }
}

}  // namespace

// --------------------------------------------------------------------------

REGISTER_OP("DynamicPartition")
    .Input("data: T")
    .Input("partitions: int32")
    .Output("outputs: num_partitions * T")
    .Attr("num_partitions: int")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      int64_t num_partitions;
      TF_RETURN_IF_ERROR(c->GetAttr("num_partitions", &num_partitions));

      ShapeHandle data_shape = c->input(0);
      ShapeHandle partitions_shape = c->input(1);

      if (!c->RankKnown(partitions_shape)) {
        return shape_inference::UnknownShape(c);
      }

      const int64_t rank = c->Rank(partitions_shape);

      // data shape must start with partitions_shape
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(
          c->MergePrefix(data_shape, partitions_shape, &unused, &unused));

      // The partition shape is dynamic in the 0th dimension, and matches
      // data_shape in the remaining dimensions.
      ShapeHandle unknown_dim0 = c->MakeShape({c->UnknownDim()});

      ShapeHandle data_suffix_shape;
      TF_RETURN_IF_ERROR(c->Subshape(data_shape, rank, &data_suffix_shape));
      ShapeHandle result_shape;
      TF_RETURN_IF_ERROR(
          c->Concatenate(unknown_dim0, data_suffix_shape, &result_shape));

      for (int i = 0; i < c->num_outputs(); ++i) {
        c->set_output(i, result_shape);
      }

      return Status::OK();
    });

namespace {

Status DynamicStitchShapeFunction(InferenceContext* c) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPSdata_flow_opsDTcc mht_1(mht_1_v, 263, "", "./tensorflow/core/ops/data_flow_ops.cc", "DynamicStitchShapeFunction");

  int32_t num_partitions;
  TF_RETURN_IF_ERROR(c->GetAttr("N", &num_partitions));

  bool all_indices_constant = true;
  int32_t max_index = -1;
  ShapeHandle extra_shape = c->UnknownShape();
  for (int i = 0; i < num_partitions; ++i) {
    const Tensor* indices_t = c->input_tensor(i);
    if (indices_t == nullptr) {
      all_indices_constant = false;
    }

    ShapeHandle indices_shape = c->input(i);
    ShapeHandle data_shape = c->input(i + num_partitions);
    if (!c->RankKnown(indices_shape)) {
      continue;
    }
    const int64_t indices_rank = c->Rank(indices_shape);

    // Assert that data_shape starts with indices_shape.
    ShapeHandle unused;
    TF_RETURN_IF_ERROR(
        c->MergePrefix(data_shape, indices_shape, &unused, &unused));

    // The rest belongs to output.
    ShapeHandle rest;
    TF_RETURN_IF_ERROR(c->Subshape(data_shape, indices_rank, &rest));
    TF_RETURN_IF_ERROR(c->Merge(extra_shape, rest, &extra_shape));

    if (indices_t != nullptr) {
      // The length is based on the highest index from flattened indices.
      const int32* indices = indices_t->flat<int32>().data();
      int64_t count = indices_t->NumElements();
      for (int64_t i = 0; i < count; ++i) {
        if (indices[i] > max_index) {
          max_index = indices[i];
        }
      }
    }
  }

  ShapeHandle output_shape = c->Vector(
      all_indices_constant ? c->MakeDim(max_index + 1) : c->UnknownDim());
  TF_RETURN_IF_ERROR(c->Concatenate(output_shape, extra_shape, &output_shape));
  c->set_output(0, output_shape);
  return Status::OK();
}

}  // namespace

REGISTER_OP("DynamicStitch")
    .Input("indices: N * int32")
    .Input("data: N * T")
    .Output("merged: T")
    .Attr("N : int >= 1")
    .Attr("T : type")
    .SetShapeFn(DynamicStitchShapeFunction);

REGISTER_OP("ParallelDynamicStitch")
    .Input("indices: N * int32")
    .Input("data: N * T")
    .Output("merged: T")
    .Attr("N : int >= 1")
    .Attr("T : type")
    .SetShapeFn(DynamicStitchShapeFunction);

// --------------------------------------------------------------------------

namespace {
Status TwoElementVectorInputsAndScalarOutputs(InferenceContext* c) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSopsPSdata_flow_opsDTcc mht_2(mht_2_v, 336, "", "./tensorflow/core/ops/data_flow_ops.cc", "TwoElementVectorInputsAndScalarOutputs");

  ShapeHandle handle;
  DimensionHandle unused_handle;
  for (int i = 0; i < c->num_inputs(); ++i) {
    TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &handle));
    TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_handle));
  }
  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->Scalar());
  }
  return Status::OK();
}

Status TwoElementOutput(InferenceContext* c) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSopsPSdata_flow_opsDTcc mht_3(mht_3_v, 352, "", "./tensorflow/core/ops/data_flow_ops.cc", "TwoElementOutput");

  c->set_output(0, c->Vector(2));
  return Status::OK();
}
}  // namespace

REGISTER_OP("RandomShuffleQueue")
    .Output("handle: Ref(string)")
    .Attr("component_types: list(type) >= 1")
    .Attr("shapes: list(shape) >= 0 = []")
    .Attr("capacity: int = -1")
    .Attr("min_after_dequeue: int = 0")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("RandomShuffleQueueV2")
    .Output("handle: resource")
    .Attr("component_types: list(type) >= 1")
    .Attr("shapes: list(shape) >= 0 = []")
    .Attr("capacity: int = -1")
    .Attr("min_after_dequeue: int = 0")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("FIFOQueue")
    .Output("handle: Ref(string)")
    .Attr("component_types: list(type) >= 1")
    .Attr("shapes: list(shape) >= 0 = []")
    .Attr("capacity: int = -1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("FIFOQueueV2")
    .Output("handle: resource")
    .Attr("component_types: list(type) >= 1")
    .Attr("shapes: list(shape) >= 0 = []")
    .Attr("capacity: int = -1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("PaddingFIFOQueue")
    .Output("handle: Ref(string)")
    .Attr("component_types: list(type) >= 1")
    .Attr("shapes: list(shape) >= 0 = []")
    .Attr("capacity: int = -1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("PaddingFIFOQueueV2")
    .Output("handle: resource")
    .Attr("component_types: list(type) >= 1")
    .Attr("shapes: list(shape) >= 0 = []")
    .Attr("capacity: int = -1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("PriorityQueue")
    .Output("handle: Ref(string)")
    .Attr("component_types: list(type) >= 0 = []")
    .Attr("shapes: list(shape) >= 0")
    .Attr("capacity: int = -1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("PriorityQueueV2")
    .Output("handle: resource")
    .Attr("component_types: list(type) >= 0 = []")
    .Attr("shapes: list(shape) >= 0")
    .Attr("capacity: int = -1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("FakeQueue")
    .Input("resource: resource")
    .Output("handle: Ref(string)")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("QueueEnqueue")
    .Input("handle: Ref(string)")
    .Input("components: Tcomponents")
    .Attr("Tcomponents: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("QueueEnqueueV2")
    .Input("handle: resource")
    .Input("components: Tcomponents")
    .Attr("Tcomponents: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("QueueEnqueueMany")
    .Input("handle: Ref(string)")
    .Input("components: Tcomponents")
    .Attr("Tcomponents: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("QueueEnqueueManyV2")
    .Input("handle: resource")
    .Input("components: Tcomponents")
    .Attr("Tcomponents: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("QueueDequeue")
    .Input("handle: Ref(string)")
    .Output("components: component_types")
    .Attr("component_types: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("QueueDequeueV2")
    .Input("handle: resource")
    .Output("components: component_types")
    .Attr("component_types: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn([](InferenceContext* c) {
      auto* t = c->input_handle_shapes_and_types(0);
      if (t != nullptr && t->size() == c->num_outputs()) {
        for (int i = 0; i < c->num_outputs(); ++i) {
          c->set_output(i, (*t)[i].shape);
        }
        return Status::OK();
      } else {
        return shape_inference::UnknownShape(c);
      }
    });

REGISTER_OP("QueueDequeueMany")
    .Input("handle: Ref(string)")
    .Input("n: int32")
    .Output("components: component_types")
    .Attr("component_types: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("QueueDequeueManyV2")
    .Input("handle: resource")
    .Input("n: int32")
    .Output("components: component_types")
    .Attr("component_types: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle n_shape;
      if (c->input_tensor(1) == nullptr) {
        n_shape = c->Vector(InferenceContext::kUnknownDim);
      } else {
        const int32_t n = c->input_tensor(1)->scalar<int32>()();
        if (n < 0) {
          return errors::InvalidArgument("Input 'n' must be >= 0, but is ", n);
        }
        n_shape = c->Vector(n);
      }
      return DequeueManyV2Shape(c, n_shape);
    });

REGISTER_OP("QueueDequeueUpTo")
    .Input("handle: Ref(string)")
    .Input("n: int32")
    .Output("components: component_types")
    .Attr("component_types: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("QueueDequeueUpToV2")
    .Input("handle: resource")
    .Input("n: int32")
    .Output("components: component_types")
    .Attr("component_types: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn([](InferenceContext* c) {
      return DequeueManyV2Shape(c, c->Vector(InferenceContext::kUnknownDim));
    });

REGISTER_OP("QueueClose")
    .Input("handle: Ref(string)")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs)
    .Attr("cancel_pending_enqueues: bool = false");

REGISTER_OP("QueueCloseV2")
    .Input("handle: resource")
    .SetShapeFn(shape_inference::NoOutputs)
    .Attr("cancel_pending_enqueues: bool = false");

REGISTER_OP("QueueIsClosed")
    .Input("handle: Ref(string)")
    .Output("is_closed: bool")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("QueueIsClosedV2")
    .Input("handle: resource")
    .Output("is_closed: bool")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("QueueSize")
    .Input("handle: Ref(string)")
    .Output("size: int32")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs);

REGISTER_OP("QueueSizeV2")
    .Input("handle: resource")
    .Output("size: int32")
    .SetShapeFn(shape_inference::UnchangedShape);

// --------------------------------------------------------------------------

REGISTER_OP("AccumulatorNumAccumulated")
    .Input("handle: Ref(string)")
    .Output("num_accumulated: int32")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("AccumulatorSetGlobalStep")
    .Input("handle: Ref(string)")
    .Input("new_global_step: int64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return Status::OK();
    });

REGISTER_OP("ConditionalAccumulator")
    .Output("handle: Ref(string)")
    .Attr("dtype: numbertype")
    .Attr("shape: shape")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("reduction_type: { 'MEAN', 'SUM' } = 'MEAN' ")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(2));
      return Status::OK();
    });

REGISTER_OP("AccumulatorApplyGradient")
    .Input("handle: Ref(string)")
    .Input("local_step: int64")
    .Input("gradient: dtype")
    .Attr("dtype: numbertype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return Status::OK();
    });

REGISTER_OP("AccumulatorTakeGradient")
    .Input("handle: Ref(string)")
    .Input("num_required: int32")
    .Output("average: dtype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      // Shape of output is the shape of the accumulator referenced
      // by 'handle', but which is not available here, so we lose
      // shape information.
      return shape_inference::UnknownShape(c);
    })
    .Attr("dtype: numbertype");

// -----------------V2 accumulators that use resource -------------------------

REGISTER_OP("ResourceAccumulatorNumAccumulated")
    .Input("handle: resource")
    .Output("num_accumulated: int32")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ResourceAccumulatorSetGlobalStep")
    .Input("handle: resource")
    .Input("new_global_step: int64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return Status::OK();
    });

REGISTER_OP("ResourceConditionalAccumulator")
    .Output("handle: resource")
    .Attr("dtype: numbertype")
    .Attr("shape: shape")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("reduction_type: { 'MEAN', 'SUM' } = 'MEAN' ")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(2));
      return Status::OK();
    });

REGISTER_OP("ResourceAccumulatorApplyGradient")
    .Input("handle: resource")
    .Input("local_step: int64")
    .Input("gradient: dtype")
    .Attr("dtype: numbertype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return Status::OK();
    });

REGISTER_OP("ResourceAccumulatorTakeGradient")
    .Input("handle: resource")
    .Input("num_required: int32")
    .Output("average: dtype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      // Shape of output is the shape of the accumulator referenced
      // by 'handle', but which is not available here, so we lose
      // shape information.
      return shape_inference::UnknownShape(c);
    })
    .Attr("dtype: numbertype");

// TODO(nponomareva): change these all to use resources.
REGISTER_OP("SparseConditionalAccumulator")
    .Output("handle: Ref(string)")
    .Attr("dtype: numbertype")
    .Attr("shape: shape")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("reduction_type: { 'MEAN', 'SUM' } = 'MEAN' ")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(2));
      return Status::OK();
    });

REGISTER_OP("SparseAccumulatorApplyGradient")
    .Input("handle: Ref(string)")
    .Input("local_step: int64")
    .Input("gradient_indices: int64")
    .Input("gradient_values: dtype")
    .Input("gradient_shape: int64")
    .Attr("dtype: numbertype")
    .Attr("has_known_shape: bool")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return Status::OK();
    });

REGISTER_OP("SparseAccumulatorTakeGradient")
    .Input("handle: Ref(string)")
    .Input("num_required: int32")
    .Output("indices: int64")
    .Output("values: dtype")
    .Output("shape: int64")
    .Attr("dtype: numbertype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      // Shape of output is the shape of the accumulator referenced
      // by 'handle', but which is not available here, so we lose
      // shape information.
      return shape_inference::UnknownShape(c);
    });

// --------------------------------------------------------------------------

REGISTER_OP("StackV2")
    .Input("max_size: int32")
    .Output("handle: resource")
    .Attr("elem_type: type")
    .Attr("stack_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("StackPushV2")
    .Input("handle: resource")
    .Input("elem: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("swap_memory: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });

REGISTER_OP("StackPopV2")
    .Input("handle: resource")
    .Output("elem: elem_type")
    .Attr("elem_type: type")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("StackCloseV2")
    .Input("handle: resource")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs);

// Deprecated ref-typed variants of stack.

REGISTER_OP("Stack")
    .Output("handle: Ref(string)")
    .Attr("elem_type: type")
    .Attr("stack_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("StackPush")
    .Input("handle: Ref(string)")
    .Input("elem: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("swap_memory: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });

REGISTER_OP("StackPop")
    .Input("handle: Ref(string)")
    .Output("elem: elem_type")
    .Attr("elem_type: type")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("StackClose")
    .Input("handle: Ref(string)")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs);

// --------------------------------------------------------------------------

REGISTER_OP("TensorArrayV3")
    .Input("size: int32")
    .Attr("dtype: type")
    .Attr("element_shape: shape = { unknown_rank: true }")
    .Attr("dynamic_size: bool = false")
    .Attr("clear_after_read: bool = true")
    .Attr("identical_element_shapes: bool = false")
    .Attr("tensor_array_name: string = ''")
    .Output("handle: resource")
    .Output("flow: float")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      c->set_output(0, c->Vector(2));
      c->set_output(1, c->Scalar());
      bool identical_shapes;
      TF_RETURN_IF_ERROR(
          c->GetAttr("identical_element_shapes", &identical_shapes));
      DataType t;
      TF_RETURN_IF_ERROR(c->GetAttr("dtype", &t));
      PartialTensorShape p;
      TF_RETURN_IF_ERROR(c->GetAttr("element_shape", &p));
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(p, &s));
      if (c->FullyDefined(s) || identical_shapes) {
        c->set_output_handle_shapes_and_types(
            0, std::vector<shape_inference::ShapeAndType>{{s, t}});
      }
      return Status::OK();
    });

REGISTER_OP("TensorArrayGradV3")
    .Input("handle: resource")
    .Input("flow_in: float")
    .Output("grad_handle: resource")
    .Output("flow_out: float")
    .Attr("source: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      c->set_output(0, c->Vector(2));
      c->set_output(1, c->Scalar());
      if (c->input_handle_shapes_and_types(0)) {
        c->set_output_handle_shapes_and_types(
            0, *c->input_handle_shapes_and_types(0));
      }
      return Status::OK();
    });

REGISTER_OP("TensorArrayGradWithShape")
    .Input("handle: resource")
    .Input("flow_in: float")
    .Input("shape_to_prepend: int32")
    .Output("grad_handle: resource")
    .Output("flow_out: float")
    .Attr("source: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      c->set_output(0, c->Vector(2));
      c->set_output(1, c->Scalar());
      auto* shape_and_type = c->input_handle_shapes_and_types(0);
      if (shape_and_type) {
        auto input_shape = (*shape_and_type)[0].shape;
        auto dtype = (*shape_and_type)[0].dtype;
        // Note that shape_to_preped is a rank 1 Tensor representing a shape.
        // The size of dimension 0 is the number of dimensions we need to add to
        // output shape.
        int64_t prepend_rank = c->Value(c->Dim(c->input(2), 0));
        if (c->RankKnown(input_shape) &&
            prepend_rank != InferenceContext::kUnknownDim) {
          int32_t input_rank = c->Rank(input_shape);
          std::vector<DimensionHandle> dims;
          dims.reserve(prepend_rank + input_rank);
          for (int i = 0; i < prepend_rank; ++i) {
            dims.push_back(c->UnknownDim());
          }
          for (int i = 0; i < input_rank; ++i) {
            dims.push_back(c->Dim(input_shape, i));
          }
          c->set_output_handle_shapes_and_types(0,
                                                {{c->MakeShape(dims), dtype}});
        } else {
          c->set_output_handle_shapes_and_types(0,
                                                {{c->UnknownShape(), dtype}});
        }
      }
      return Status::OK();
    });

REGISTER_OP("TensorArrayWriteV3")
    .Input("handle: resource")
    .Input("index: int32")
    .Input("value: T")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));

      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data != nullptr && !handle_data->empty()) {
        shape_inference::ShapeAndType shape_and_type = (*handle_data)[0];
        ShapeHandle value_shape = c->input(2);
        TF_RETURN_IF_ERROR(
            c->Merge(shape_and_type.shape, value_shape, &unused));
      }

      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("TensorArrayReadV3")
    .Input("handle: resource")
    .Input("index: int32")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      auto shapes = c->input_handle_shapes_and_types(0);
      if (shapes != nullptr && !shapes->empty()) {
        ShapeHandle tensor_shape = shapes->at(0).shape;
        c->set_output(0, tensor_shape);
        return Status::OK();
      } else {
        return shape_inference::UnknownShape(c);
      }
    });

REGISTER_OP("TensorArrayGatherV3")
    .Input("handle: resource")
    .Input("indices: int32")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Attr("dtype: type")
    .Attr("element_shape: shape = { unknown_rank: true }")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle indices;
      ShapeHandle unused;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &indices));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(0), 0), 2, &unused_dim));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      auto shapes = c->input_handle_shapes_and_types(0);
      if (shapes != nullptr && !shapes->empty()) {
        ShapeHandle tensor_shape = shapes->at(0).shape;
        ShapeHandle output_shape;
        TF_RETURN_IF_ERROR(
            c->Concatenate(indices, tensor_shape, &output_shape));
        c->set_output(0, output_shape);
        return Status::OK();
      } else {
        PartialTensorShape p;
        TF_RETURN_IF_ERROR(c->GetAttr("element_shape", &p));
        ShapeHandle s;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(p, &s));
        ShapeHandle output_shape;
        TF_RETURN_IF_ERROR(c->Concatenate(indices, s, &output_shape));
        c->set_output(0, output_shape);
        return Status::OK();
      }
    });

REGISTER_OP("TensorArrayScatterV3")
    .Input("handle: resource")
    .Input("indices: int32")
    .Input("value: T")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle indices;
      ShapeHandle unused;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &indices));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(0), 0), 2, &unused_dim));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      ShapeHandle value_shape;
      // Assert that the length of the indices tensor is equal to the first
      // dimension of the value tensor.
      TF_RETURN_IF_ERROR(
          c->MergePrefix(c->input(2), indices, &value_shape, &indices));
      auto shapes = c->input_handle_shapes_and_types(0);
      if (shapes != nullptr && !shapes->empty()) {
        ShapeHandle tensor_shape = shapes->at(0).shape;
        ShapeHandle fed_shape;
        TF_RETURN_IF_ERROR(c->Subshape(value_shape, 1, &fed_shape));
        TF_RETURN_IF_ERROR(c->Merge(tensor_shape, fed_shape, &fed_shape));
      }
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("TensorArrayConcatV3")
    .Input("handle: resource")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Output("lengths: int64")
    .Attr("dtype: type")
    .Attr("element_shape_except0: shape = { unknown_rank: true }")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      c->set_output(0, c->UnknownShape());
      c->set_output(1, c->Vector(c->UnknownDim()));
      return Status::OK();
    });

REGISTER_OP("TensorArraySplitV3")
    .Input("handle: resource")
    .Input("value: T")
    .Input("lengths: int64")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("TensorArraySizeV3")
    .Input("handle: resource")
    .Input("flow_in: float")
    .Output("size: int32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("TensorArrayCloseV3")
    .Input("handle: resource")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      return Status::OK();
    });

// --------------------------------------------------------------------------

// Deprecated TensorArray methods

REGISTER_OP("TensorArray")
    .Input("size: int32")
    .Attr("dtype: type")
    .Attr("dynamic_size: bool = false")
    .Attr("clear_after_read: bool = true")
    .Attr("tensor_array_name: string = ''")
    .Attr("element_shape: shape = { unknown_rank: true }")
    .Output("handle: Ref(string)")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(16, "Use TensorArrayV3");
REGISTER_OP("TensorArrayV2")
    .Input("size: int32")
    .Attr("dtype: type")
    .Attr("element_shape: shape = { unknown_rank: true }")
    .Attr("dynamic_size: bool = false")
    .Attr("clear_after_read: bool = true")
    .Attr("tensor_array_name: string = ''")
    .Output("handle: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      c->set_output(0, c->Vector(2));
      return Status::OK();
    })
    .Deprecated(26, "Use TensorArrayV3");
REGISTER_OP("TensorArrayGrad")
    .Input("handle: string")
    .Input("flow_in: float")
    .Output("grad_handle: Ref(string)")
    .Attr("source: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(16, "Use TensorArrayGradV3");
REGISTER_OP("TensorArrayGradV2")
    .Input("handle: string")
    .Input("flow_in: float")
    .Output("grad_handle: string")
    .Attr("source: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      c->set_output(0, c->Vector(2));
      return Status::OK();
    })
    .Deprecated(26, "Use TensorArrayGradV3");
REGISTER_OP("TensorArrayWrite")
    .Input("handle: Ref(string)")
    .Input("index: int32")
    .Input("value: T")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(16, "Use TensorArrayWriteV3");
REGISTER_OP("TensorArrayWriteV2")
    .Input("handle: string")
    .Input("index: int32")
    .Input("value: T")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      return shape_inference::ScalarShape(c);
    })
    .Deprecated(26, "Use TensorArrayWriteV3");
REGISTER_OP("TensorArrayRead")
    .Input("handle: Ref(string)")
    .Input("index: int32")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(16, "Use TensorArrayReadV3");
REGISTER_OP("TensorArrayReadV2")
    .Input("handle: string")
    .Input("index: int32")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      return shape_inference::UnknownShape(c);
    })
    .Deprecated(26, "Use TensorArrayReadV3");
REGISTER_OP("TensorArrayPack")
    .Input("handle: Ref(string)")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Attr("dtype: type")
    .Attr("element_shape: shape = { unknown_rank: true }")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(16, "Use TensorArrayGatherV3 with RangeOp");
REGISTER_OP("TensorArrayUnpack")
    .Input("handle: Ref(string)")
    .Input("value: T")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(20, "Use TensorArrayScatterV3 with RangeOp");
REGISTER_OP("TensorArrayGather")
    .Input("handle: Ref(string)")
    .Input("indices: int32")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Attr("dtype: type")
    .Attr("element_shape: shape = { unknown_rank: true }")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(16, "Use TensorArrayGatherV3");
REGISTER_OP("TensorArrayGatherV2")
    .Input("handle: string")
    .Input("indices: int32")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Attr("dtype: type")
    .Attr("element_shape: shape = { unknown_rank: true }")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(0), 0), 2, &unused_dim));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      return shape_inference::UnknownShape(c);
    })
    .Deprecated(26, "Use TensorArrayGatherV3");
REGISTER_OP("TensorArrayScatter")
    .Input("handle: Ref(string)")
    .Input("indices: int32")
    .Input("value: T")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(19, "Use TensorArrayGradV3");
REGISTER_OP("TensorArrayScatterV2")
    .Input("handle: string")
    .Input("indices: int32")
    .Input("value: T")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(0), 0), 2, &unused_dim));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      return shape_inference::ScalarShape(c);
    })
    .Deprecated(26, "Use TensorArrayScatterV3");
REGISTER_OP("TensorArrayConcat")
    .Input("handle: Ref(string)")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Output("lengths: int64")
    .Attr("dtype: type")
    .Attr("element_shape_except0: shape = { unknown_rank: true }")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(16, "Use TensorArrayGradV3");
REGISTER_OP("TensorArrayConcatV2")
    .Input("handle: string")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Output("lengths: int64")
    .Attr("dtype: type")
    .Attr("element_shape_except0: shape = { unknown_rank: true }")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      c->set_output(0, c->UnknownShape());
      c->set_output(1, c->Vector(c->UnknownDim()));
      return Status::OK();
    });
REGISTER_OP("TensorArraySplit")
    .Input("handle: Ref(string)")
    .Input("value: T")
    .Input("lengths: int64")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(16, "Use TensorArraySplitV3");
REGISTER_OP("TensorArraySplitV2")
    .Input("handle: string")
    .Input("value: T")
    .Input("lengths: int64")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      return shape_inference::ScalarShape(c);
    })
    .Deprecated(26, "Use TensorArraySplitV3");
REGISTER_OP("TensorArraySize")
    .Input("handle: Ref(string)")
    .Input("flow_in: float")
    .Output("size: int32")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(16, "Use TensorArraySizeV3");
REGISTER_OP("TensorArraySizeV2")
    .Input("handle: string")
    .Input("flow_in: float")
    .Output("size: int32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      return shape_inference::ScalarShape(c);
    })
    .Deprecated(26, "Use TensorArraySizeV3");
REGISTER_OP("TensorArrayClose")
    .Input("handle: Ref(string)")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); })
    .Deprecated(16, "Use TensorArrayCloseV3");
REGISTER_OP("TensorArrayCloseV2")
    .Input("handle: string")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      return Status::OK();
    })
    .Deprecated(26, "Use TensorArrayCloseV3");

// --------------------------------------------------------------------------

REGISTER_OP("Barrier")
    .SetIsStateful()
    .Output("handle: Ref(string)")
    .Attr("component_types: list(type) >= 1")
    .Attr("shapes: list(shape) >= 0 = []")
    .Attr("capacity: int = -1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("BarrierInsertMany")
    .Input("handle: Ref(string)")
    .Input("keys: string")
    .Input("values: T")
    .Attr("T: type")
    .Attr("component_index: int")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle keys = c->input(1);
      ShapeHandle values = c->input(2);
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      TF_RETURN_IF_ERROR(c->WithRank(keys, 1, &keys));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(values, 1, &values));
      TF_RETURN_IF_ERROR(c->Merge(keys, c->Vector(c->Dim(values, 0)), &handle));
      return Status::OK();
    });

REGISTER_OP("BarrierTakeMany")
    .Input("handle: Ref(string)")
    .Input("num_elements: int32")
    .Output("indices: int64")
    .Output("keys: string")
    .Output("values: component_types")
    .Attr("component_types: list(type) >= 1")
    .Attr("allow_small_batch: bool = false")
    .Attr("wait_for_incomplete: bool = false")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("BarrierClose")
    .Input("handle: Ref(string)")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs)
    .Attr("cancel_pending_enqueues: bool = false");

REGISTER_OP("BarrierReadySize")
    .Input("handle: Ref(string)")
    .Output("size: int32")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs);

REGISTER_OP("BarrierIncompleteSize")
    .Input("handle: Ref(string)")
    .Output("size: int32")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs);

// --------------------------------------------------------------------------

REGISTER_OP("GetSessionHandle")
    .Input("value: T")
    .Output("handle: string")
    .Attr("T: type")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("GetSessionHandleV2")
    .Input("value: T")
    .Output("handle: resource")
    .Attr("T: type")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("GetSessionTensor")
    .Input("handle: string")
    .Output("value: dtype")
    .Attr("dtype: type")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      return shape_inference::UnknownShape(c);
    });

REGISTER_OP("DeleteSessionTensor")
    .Input("handle: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      return Status::OK();
    });

REGISTER_OP("Stage")
    .Input("values: dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("Unstage")
    .Output("values: dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("StagePeek")
    .Input("index: int32")
    .Output("values: dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("StageSize")
    .Output("size: int32")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(shape_inference::ScalarShape)
    .SetIsStateful();

REGISTER_OP("StageClear")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

// UnorderedMap
REGISTER_OP("MapStage")
    .Input("key: int64")
    .Input("indices: int32")
    .Input("values: fake_dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("fake_dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .SetIsStateful();

REGISTER_OP("MapPeek")
    .Input("key: int64")
    .Input("indices: int32")
    .Output("values: dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("MapUnstage")
    .Input("key: int64")
    .Input("indices: int32")
    .Output("values: dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("MapUnstageNoKey")
    .Input("indices: int32")
    .Output("key: int64")
    .Output("values: dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("MapSize")
    .Output("size: int32")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .SetIsStateful();

REGISTER_OP("MapIncompleteSize")
    .Output("size: int32")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .SetIsStateful();

REGISTER_OP("MapClear")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .SetIsStateful();

// OrderedMap
REGISTER_OP("OrderedMapStage")
    .Input("key: int64")
    .Input("indices: int32")
    .Input("values: fake_dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("fake_dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .SetIsStateful();

REGISTER_OP("OrderedMapPeek")
    .Input("key: int64")
    .Input("indices: int32")
    .Output("values: dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("OrderedMapUnstage")
    .Input("key: int64")
    .Input("indices: int32")
    .Output("values: dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("OrderedMapUnstageNoKey")
    .Input("indices: int32")
    .Output("key: int64")
    .Output("values: dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("OrderedMapSize")
    .Output("size: int32")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .SetIsStateful();

REGISTER_OP("OrderedMapIncompleteSize")
    .Output("size: int32")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .SetIsStateful();

REGISTER_OP("OrderedMapClear")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .SetIsStateful();

REGISTER_OP("RecordInput")
    .Output("records: string")
    .Attr("file_pattern: string")
    .Attr("file_random_seed: int = 301")
    .Attr("file_shuffle_shift_ratio: float = 0")
    .Attr("file_buffer_size: int = 10000")
    .Attr("file_parallelism: int = 16")
    .Attr("batch_size: int = 32")
    .Attr("compression_type: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

}  // namespace tensorflow
