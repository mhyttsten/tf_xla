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
class MHTracer_DTPStensorflowPScorePSopsPStraining_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPStraining_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPStraining_opsDTcc() {
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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

template <bool is_resource>
ShapeHandle ShapeOrHandleShape(InferenceContext* c, int input) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPStraining_opsDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/ops/training_ops.cc", "ShapeOrHandleShape");

  auto* handle_data = c->input_handle_shapes_and_types(input);
  if (handle_data != nullptr && !handle_data->empty() &&
      (*handle_data)[0].dtype != DT_INVALID) {
    return (*handle_data)[0].shape;
  }
  return c->input(input);
}

template <>
ShapeHandle ShapeOrHandleShape<true>(InferenceContext* c, int input) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPStraining_opsDTcc mht_1(mht_1_v, 208, "", "./tensorflow/core/ops/training_ops.cc", "ShapeOrHandleShape<true>");

  auto* handle_data = c->input_handle_shapes_and_types(input);
  if (handle_data != nullptr && !handle_data->empty() &&
      (*handle_data)[0].dtype != DT_INVALID) {
    return (*handle_data)[0].shape;
  }
  // If a resource input is missing shape information, we should return
  // UnknownShape rather than the shape of the input, which is a scalar
  // resource handle.
  return c->UnknownShape();
}

// Handle the gradient and, if <is_sparse>, indices inputs.
// <s> is an input+output parameter, containing the current known input shape to
// the gradient.
template <bool is_sparse, bool is_resource>
static Status HandleGradAndIndicesInputs(InferenceContext* c, int grad_idx,
                                         ShapeHandle* s) {
  ShapeHandle grad = ShapeOrHandleShape<is_resource>(c, grad_idx);
  if (!is_sparse) {
    TF_RETURN_IF_ERROR(c->Merge(*s, grad, s));
    return Status::OK();
  }
  // Indices is a vector where indices.dim[0].rank == grad[0].rank.
  ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(grad_idx + 1), 1, &indices));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(indices, 0), c->Dim(grad, 0), &unused));
  // Trailing part of grad matches trailing part of *s.
  ShapeHandle grad_unknown_first;
  TF_RETURN_IF_ERROR(
      c->ReplaceDim(grad, 0, c->UnknownDim(), &grad_unknown_first));
  TF_RETURN_IF_ERROR(c->Merge(*s, grad_unknown_first, s));

  return Status::OK();
}

template <bool is_resource>
static Status ApplyGradientDescentShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSopsPStraining_opsDTcc mht_2(mht_2_v, 249, "", "./tensorflow/core/ops/training_ops.cc", "ApplyGradientDescentShapeFn");

  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape<is_resource>(c, 0);     // var
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));  // alpha
  TF_RETURN_IF_ERROR(c->Merge(s, c->input(2), &s));          // delta
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyGradientDescent")
    .Input("var: Ref(T)")
    .Input("alpha: T")
    .Input("delta: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyGradientDescentShapeFn<false>);

REGISTER_OP("ResourceApplyGradientDescent")
    .Input("var: resource")
    .Input("alpha: T")
    .Input("delta: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyGradientDescentShapeFn<true>);

template <bool is_sparse, bool is_resource>
Status ApplyProximalGradientDescentShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape<is_resource>(c, 0);     // var
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));  // alpha
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));  // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));  // l2
  TF_RETURN_IF_ERROR(HandleGradAndIndicesInputs<is_sparse, is_resource>(
      c, 4 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyProximalGradientDescent")
    .Input("var: Ref(T)")
    .Input("alpha: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("delta: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyProximalGradientDescentShapeFn</*is_sparse=*/false,
                                                    /*is_resource=*/false>);

REGISTER_OP("SparseApplyProximalGradientDescent")
    .Input("var: Ref(T)")
    .Input("alpha: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyProximalGradientDescentShapeFn</*is_sparse=*/true,
                                                    /*is_resource=*/false>);

REGISTER_OP("ResourceApplyProximalGradientDescent")
    .Input("var: resource")
    .Input("alpha: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("delta: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyProximalGradientDescentShapeFn</*is_sparse=*/false,
                                                    /*is_resource=*/true>);

REGISTER_OP("ResourceSparseApplyProximalGradientDescent")
    .Input("var: resource")
    .Input("alpha: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyProximalGradientDescentShapeFn</*is_sparse=*/true,
                                                    /*is_resource=*/true>);

template <bool is_sparse, bool is_resource>
static Status ApplyAdadeltaShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape<is_resource>(c, 0);  // var
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 2), &s));  // accum update
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));     // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));     // rho
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));     // epsilon
  TF_RETURN_IF_ERROR(HandleGradAndIndicesInputs<is_sparse, is_resource>(
      c, 6 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyAdadelta")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("accum_update: Ref(T)")
    .Input("lr: T")
    .Input("rho: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
        ApplyAdadeltaShapeFn</*is_sparse=*/false, /*is_resource=*/false>);

REGISTER_OP("SparseApplyAdadelta")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("accum_update: Ref(T)")
    .Input("lr: T")
    .Input("rho: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
        ApplyAdadeltaShapeFn</*is_sparse=*/true, /*is_resource=*/false>);

REGISTER_OP("ResourceApplyAdadelta")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("accum_update: resource")
    .Input("lr: T")
    .Input("rho: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
        ApplyAdadeltaShapeFn</*is_sparse=*/false, /*is_resource=*/true>);

REGISTER_OP("ResourceSparseApplyAdadelta")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("accum_update: resource")
    .Input("lr: T")
    .Input("rho: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyAdadeltaShapeFn</*is_sparse=*/true, /*is_resource=*/true>);

template <bool is_sparse, bool is_resource>
static Status ApplyAdagradShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape<is_resource>(c, 0);  // var
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));     // lr
  TF_RETURN_IF_ERROR(HandleGradAndIndicesInputs<is_sparse, is_resource>(
      c, 3 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyAdagrad")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("lr: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("update_slots: bool = true")
    .SetShapeFn(
        ApplyAdagradShapeFn</*is_sparse=*/false, /*is_resource=*/false>);

REGISTER_OP("ResourceApplyAdagrad")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("lr: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("update_slots: bool = true")
    .SetShapeFn(ApplyAdagradShapeFn</*is_sparse=*/false, /*is_resource=*/true>);

REGISTER_OP("SparseApplyAdagrad")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("lr: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .Attr("update_slots: bool = true")
    .SetShapeFn(ApplyAdagradShapeFn</*is_sparse=*/true, /*is_resource=*/false>);

REGISTER_OP("ResourceSparseApplyAdagrad")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("lr: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .Attr("update_slots: bool = true")
    .SetShapeFn(ApplyAdagradShapeFn</*is_sparse=*/true, /*is_resource=*/true>);

template <bool is_sparse, bool is_resource>
static Status ApplyAdagradV2ShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape<is_resource>(c, 0);  // var
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));     // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));     // epsilon
  TF_RETURN_IF_ERROR(HandleGradAndIndicesInputs<is_sparse, is_resource>(
      c, 4 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyAdagradV2")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("lr: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("update_slots: bool = true")
    .SetShapeFn(
        ApplyAdagradV2ShapeFn</*is_sparse=*/false, /*is_resource=*/false>);

REGISTER_OP("ResourceApplyAdagradV2")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("lr: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("update_slots: bool = true")
    .SetShapeFn(
        ApplyAdagradV2ShapeFn</*is_sparse=*/false, /*is_resource=*/true>);

REGISTER_OP("SparseApplyAdagradV2")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("lr: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .Attr("update_slots: bool = true")
    .SetShapeFn(
        ApplyAdagradV2ShapeFn</*is_sparse=*/true, /*is_resource=*/false>);

REGISTER_OP("ResourceSparseApplyAdagradV2")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("lr: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .Attr("update_slots: bool = true")
    .SetShapeFn(
        ApplyAdagradV2ShapeFn</*is_sparse=*/true, /*is_resource=*/true>);

template <bool is_sparse, bool is_resource>
static Status ApplyProximalAdagradShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape<is_resource>(c, 0);  // var
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));     // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));     // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));     // l2
  TF_RETURN_IF_ERROR(HandleGradAndIndicesInputs<is_sparse, is_resource>(
      c, 5 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyProximalAdagrad")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyProximalAdagradShapeFn</*is_sparse=*/false,
                                            /*is_resource=*/false>);

REGISTER_OP("ResourceApplyProximalAdagrad")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyProximalAdagradShapeFn</*is_sparse=*/false,
                                            /*is_resource=*/false>);

REGISTER_OP("SparseApplyProximalAdagrad")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
        ApplyProximalAdagradShapeFn</*is_sparse=*/true, /*is_resource=*/false>);

REGISTER_OP("ResourceSparseApplyProximalAdagrad")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
        ApplyProximalAdagradShapeFn</*is_sparse=*/true, /*is_resource=*/true>);

template <bool is_sparse, bool is_resource>
static Status ApplyAdagradDAShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape<is_resource>(c, 0);  // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape<is_resource>(c, 1),
                              &s));  // grad_accumulator
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape<is_resource>(c, 2),
                              &s));  // gradient_squared_accumulator
  TF_RETURN_IF_ERROR(HandleGradAndIndicesInputs<is_sparse, is_resource>(
      c, 3 /* grad_idx */, &s));
  int idx = is_sparse ? 5 : 4;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // global step
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyAdagradDA")
    .Input("var: Ref(T)")
    .Input("gradient_accumulator: Ref(T)")
    .Input("gradient_squared_accumulator: Ref(T)")
    .Input("grad: T")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("global_step: int64")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
        ApplyAdagradDAShapeFn</*is_sparse=*/false, /*is_resource=*/false>);

REGISTER_OP("SparseApplyAdagradDA")
    .Input("var: Ref(T)")
    .Input("gradient_accumulator: Ref(T)")
    .Input("gradient_squared_accumulator: Ref(T)")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("global_step: int64")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
        ApplyAdagradDAShapeFn</*is_sparse=*/true, /*is_resource=*/false>);

REGISTER_OP("ResourceApplyAdagradDA")
    .Input("var: resource")
    .Input("gradient_accumulator: resource")
    .Input("gradient_squared_accumulator: resource")
    .Input("grad: T")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("global_step: int64")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
        ApplyAdagradDAShapeFn</*is_sparse=*/false, /*is_resource=*/true>);

REGISTER_OP("ResourceSparseApplyAdagradDA")
    .Input("var: resource")
    .Input("gradient_accumulator: resource")
    .Input("gradient_squared_accumulator: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("global_step: int64")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
        ApplyAdagradDAShapeFn</*is_sparse=*/true, /*is_resource=*/true>);

template <bool is_sparse, bool is_resource>
static Status ApplyFtrlShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape<is_resource>(c, 0);  // var
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 2), &s));  // linear
  TF_RETURN_IF_ERROR(HandleGradAndIndicesInputs<is_sparse, is_resource>(
      c, 3 /* grad_idx */, &s));
  int idx = is_sparse ? 5 : 4;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr_power
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyFtrl")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("linear: Ref(T)")
    .Input("grad: T")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("lr_power: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("multiply_linear_by_lr: bool = false")
    .SetShapeFn(ApplyFtrlShapeFn</*is_sparse=*/false, /*is_resource=*/false>);

REGISTER_OP("SparseApplyFtrl")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("linear: Ref(T)")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("lr_power: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .Attr("multiply_linear_by_lr: bool = false")
    .SetShapeFn(ApplyFtrlShapeFn</*is_sparse=*/true, /*is_resource=*/false>);

REGISTER_OP("ResourceApplyFtrl")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("linear: resource")
    .Input("grad: T")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("lr_power: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("multiply_linear_by_lr: bool = false")
    .SetShapeFn(ApplyFtrlShapeFn</*is_sparse=*/false, /*is_resource=*/true>);

REGISTER_OP("ResourceSparseApplyFtrl")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("linear: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("lr_power: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .Attr("multiply_linear_by_lr: bool = false")
    .SetShapeFn(ApplyFtrlShapeFn</*is_sparse=*/true, /*is_resource=*/true>);

REGISTER_OP("ApplyFtrlV2")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("linear: Ref(T)")
    .Input("grad: T")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l2_shrinkage: T")
    .Input("lr_power: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("multiply_linear_by_lr: bool = false")
    .SetShapeFn(ApplyFtrlShapeFn</*is_sparse=*/false, /*is_resource=*/false>);

REGISTER_OP("SparseApplyFtrlV2")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("linear: Ref(T)")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l2_shrinkage: T")
    .Input("lr_power: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .Attr("multiply_linear_by_lr: bool = false")
    .SetShapeFn(ApplyFtrlShapeFn</*is_sparse=*/true, /*is_resource=*/false>);

REGISTER_OP("ResourceApplyFtrlV2")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("linear: resource")
    .Input("grad: T")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l2_shrinkage: T")
    .Input("lr_power: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("multiply_linear_by_lr: bool = false")
    .SetShapeFn(ApplyFtrlShapeFn</*is_sparse=*/false, /*is_resource=*/true>);

REGISTER_OP("ResourceSparseApplyFtrlV2")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("linear: resource")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l2_shrinkage: T")
    .Input("lr_power: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .Attr("multiply_linear_by_lr: bool = false")
    .SetShapeFn(ApplyFtrlShapeFn</*is_sparse=*/true, /*is_resource=*/true>);

template <bool is_sparse, bool is_resource>
static Status ApplyMomentumShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape<is_resource>(c, 0);  // var
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));     // lr
  TF_RETURN_IF_ERROR(HandleGradAndIndicesInputs<is_sparse, is_resource>(
      c, 3 /* grad_idx */, &s));
  int idx = is_sparse ? 5 : 4;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // momentum
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyMomentum")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("lr: T")
    .Input("grad: T")
    .Input("momentum: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("use_nesterov: bool = false")
    .SetShapeFn(
        ApplyMomentumShapeFn</*is_sparse=*/false, /*is_resource=*/false>);

REGISTER_OP("SparseApplyMomentum")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("lr: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("momentum: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .Attr("use_nesterov: bool = false")
    .SetShapeFn(
        ApplyMomentumShapeFn</*is_sparse=*/true, /*is_resource=*/false>);

REGISTER_OP("ResourceApplyMomentum")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("lr: T")
    .Input("grad: T")
    .Input("momentum: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("use_nesterov: bool = false")
    .SetShapeFn(
        ApplyMomentumShapeFn</*is_sparse=*/false, /*is_resource=*/true>);

REGISTER_OP("ResourceSparseApplyMomentum")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("lr: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("momentum: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .Attr("use_nesterov: bool = false")
    .SetShapeFn(ApplyMomentumShapeFn</*is_sparse=*/true, /*is_resource=*/true>);

REGISTER_OP("ResourceApplyKerasMomentum")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("lr: T")
    .Input("grad: T")
    .Input("momentum: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("use_nesterov: bool = false")
    .SetShapeFn(
        ApplyMomentumShapeFn</*is_sparse=*/false, /*is_resource=*/true>);

REGISTER_OP("ResourceSparseApplyKerasMomentum")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("lr: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("momentum: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .Attr("use_nesterov: bool = false")
    .SetShapeFn(ApplyMomentumShapeFn</*is_sparse=*/true, /*is_resource=*/true>);

template <bool is_resource>
static Status ApplyAdamShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSopsPStraining_opsDTcc mht_3(mht_3_v, 952, "", "./tensorflow/core/ops/training_ops.cc", "ApplyAdamShapeFn");

  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape<is_resource>(c, 0);  // var
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));     // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));     // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));     // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));     // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));     // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));     // epsilon
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs</*is_sparse=*/false, is_resource>(
          c, 9 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyAdam")
    .Input("var: Ref(T)")
    .Input("m: Ref(T)")
    .Input("v: Ref(T)")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("lr: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("use_nesterov: bool = false")
    .SetShapeFn(ApplyAdamShapeFn</*is_resource=*/false>);

REGISTER_OP("ResourceApplyAdam")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("lr: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("use_nesterov: bool = false")
    .SetShapeFn(ApplyAdamShapeFn</*is_resource=*/true>);

template <bool is_resource>
static Status ApplyAdamWithAmsgradShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSopsPStraining_opsDTcc mht_4(mht_4_v, 1011, "", "./tensorflow/core/ops/training_ops.cc", "ApplyAdamWithAmsgradShapeFn");

  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape<is_resource>(c, 0);  // var
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 3), &s));  // vhat
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));     // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));     // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));     // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));     // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));     // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(9), 0, &unused));     // epsilon
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs</*is_sparse=*/false, is_resource>(
          c, 10 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ResourceApplyAdamWithAmsgrad")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("vhat: resource")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("lr: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyAdamWithAmsgradShapeFn</*is_resource=*/true>);

template <bool is_resource>
static Status ApplyAdaMaxShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSopsPStraining_opsDTcc mht_5(mht_5_v, 1055, "", "./tensorflow/core/ops/training_ops.cc", "ApplyAdaMaxShapeFn");

  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape<is_resource>(c, 0);  // var
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));     // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));     // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));     // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));     // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));     // epsilon
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs</*is_sparse=*/false, is_resource>(
          c, 8 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyAdaMax")
    .Input("var: Ref(T)")
    .Input("m: Ref(T)")
    .Input("v: Ref(T)")
    .Input("beta1_power: T")
    .Input("lr: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyAdaMaxShapeFn</*is_resource=*/false>);

REGISTER_OP("ResourceApplyAdaMax")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("beta1_power: T")
    .Input("lr: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyAdaMaxShapeFn</*is_resource=*/true>);

template <bool is_sparse, bool is_resource>
static Status ApplyRMSPropShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape<is_resource>(c, 0);  // var
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 1), &s));  // ms
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 2), &s));  // mom
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));     // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));     // rho
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));     // momentum
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));     // epsilon
  TF_RETURN_IF_ERROR(HandleGradAndIndicesInputs<is_sparse, is_resource>(
      c, 7 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyRMSProp")
    .Input("var: Ref(T)")
    .Input("ms: Ref(T)")
    .Input("mom: Ref(T)")
    .Input("lr: T")
    .Input("rho: T")
    .Input("momentum: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
        ApplyRMSPropShapeFn</*is_sparse=*/false, /*is_resource=*/false>);

REGISTER_OP("SparseApplyRMSProp")
    .Input("var: Ref(T)")
    .Input("ms: Ref(T)")
    .Input("mom: Ref(T)")
    .Input("lr: T")
    .Input("rho: T")
    .Input("momentum: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyRMSPropShapeFn</*is_sparse=*/true, /*is_resource=*/false>);

REGISTER_OP("ResourceApplyRMSProp")
    .Input("var: resource")
    .Input("ms: resource")
    .Input("mom: resource")
    .Input("lr: T")
    .Input("rho: T")
    .Input("momentum: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyRMSPropShapeFn</*is_sparse=*/false, /*is_resource=*/true>);

REGISTER_OP("ResourceSparseApplyRMSProp")
    .Input("var: resource")
    .Input("ms: resource")
    .Input("mom: resource")
    .Input("lr: T")
    .Input("rho: T")
    .Input("momentum: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyRMSPropShapeFn</*is_sparse=*/true, /*is_resource=*/true>);

template <bool is_sparse, bool is_resource>
static Status ApplyCenteredRMSPropShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape<is_resource>(c, 0);  // var
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 1), &s));  // ms
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 2), &s));  // mg
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 3), &s));  // mom
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));     // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));     // rho
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));     // momentum
  TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));     // epsilon
  TF_RETURN_IF_ERROR(HandleGradAndIndicesInputs<is_sparse, is_resource>(
      c, 8 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyCenteredRMSProp")
    .Input("var: Ref(T)")
    .Input("mg: Ref(T)")
    .Input("ms: Ref(T)")
    .Input("mom: Ref(T)")
    .Input("lr: T")
    .Input("rho: T")
    .Input("momentum: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyCenteredRMSPropShapeFn</*is_sparse=*/false,
                                            /*is_resource=*/false>);

REGISTER_OP("SparseApplyCenteredRMSProp")
    .Input("var: Ref(T)")
    .Input("mg: Ref(T)")
    .Input("ms: Ref(T)")
    .Input("mom: Ref(T)")
    .Input("lr: T")
    .Input("rho: T")
    .Input("momentum: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
        ApplyCenteredRMSPropShapeFn</*is_sparse=*/true, /*is_resource=*/false>);

REGISTER_OP("ResourceApplyCenteredRMSProp")
    .Input("var: resource")
    .Input("mg: resource")
    .Input("ms: resource")
    .Input("mom: resource")
    .Input("lr: T")
    .Input("rho: T")
    .Input("momentum: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
        ApplyCenteredRMSPropShapeFn</*is_sparse=*/false, /*is_resource=*/true>);

REGISTER_OP("ResourceSparseApplyCenteredRMSProp")
    .Input("var: resource")
    .Input("mg: resource")
    .Input("ms: resource")
    .Input("mom: resource")
    .Input("lr: T")
    .Input("rho: T")
    .Input("momentum: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
        ApplyCenteredRMSPropShapeFn</*is_sparse=*/true, /*is_resource=*/true>);

template <bool is_resource>
static Status ApplyAddSignShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSopsPStraining_opsDTcc mht_6(mht_6_v, 1276, "", "./tensorflow/core/ops/training_ops.cc", "ApplyAddSignShapeFn");

  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape<is_resource>(c, 0);  // var
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));     // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));     // alpha
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));     // sign_decay
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));     // beta
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs</*is_sparse=*/false, is_resource>(
          c, 6 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyAddSign")
    .Input("var: Ref(T)")
    .Input("m: Ref(T)")
    .Input("lr: T")
    .Input("alpha: T")
    .Input("sign_decay: T")
    .Input("beta: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyAddSignShapeFn</*is_resource=*/false>);

REGISTER_OP("ResourceApplyAddSign")
    .Input("var: resource")
    .Input("m: resource")
    .Input("lr: T")
    .Input("alpha: T")
    .Input("sign_decay: T")
    .Input("beta: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyAddSignShapeFn</*is_resource=*/true>);

template <bool is_resource>
static Status ApplyPowerSignShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSopsPStraining_opsDTcc mht_7(mht_7_v, 1323, "", "./tensorflow/core/ops/training_ops.cc", "ApplyPowerSignShapeFn");

  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape<is_resource>(c, 0);  // var
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));     // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));     // logbase
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));     // sign_delay
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));     // beta
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs</*is_sparse=*/false, is_resource>(
          c, 6 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyPowerSign")
    .Input("var: Ref(T)")
    .Input("m: Ref(T)")
    .Input("lr: T")
    .Input("logbase: T")
    .Input("sign_decay: T")
    .Input("beta: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyPowerSignShapeFn</*is_resource=*/false>);

REGISTER_OP("ResourceApplyPowerSign")
    .Input("var: resource")
    .Input("m: resource")
    .Input("lr: T")
    .Input("logbase: T")
    .Input("sign_decay: T")
    .Input("beta: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyPowerSignShapeFn</*is_resource=*/true>);

}  // namespace tensorflow
