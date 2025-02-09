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
class MHTracer_DTPStensorflowPScorePSopsPSstate_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSstate_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSstate_opsDTcc() {
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

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("VariableV2")
    .Output("ref: Ref(dtype)")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape);

REGISTER_OP("Variable")
    .Output("ref: Ref(dtype)")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      PartialTensorShape shape;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &shape));

      // Variable has legacy behavior where we cannot tell the difference
      // between a scalar shape attribute and 'unknown shape'.  So if the shape
      // is a scalar, we return an unknown shape.
      if (shape.dims() <= 0) {
        return shape_inference::UnknownShape(c);
      }

      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(shape, &out));
      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("IsVariableInitialized")
    .Input("ref: Ref(dtype)")
    .Output("is_initialized: bool")
    .Attr("dtype: type")
    .SetAllowsUninitializedInput()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("TemporaryVariable")
    .Output("ref: Ref(dtype)")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("var_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape);

REGISTER_OP("DestroyTemporaryVariable")
    .Input("ref: Ref(T)")
    .Output("value: T")
    .Attr("T: type")
    .Attr("var_name: string")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Assign")
    .Input("ref: Ref(T)")
    .Input("value: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: type")
    .Attr("validate_shape: bool = true")
    .Attr("use_locking: bool = true")
    .SetAllowsUninitializedInput()
    .SetShapeFn([](InferenceContext* c) {
      bool validate_shape;
      TF_RETURN_IF_ERROR(c->GetAttr("validate_shape", &validate_shape));
      if (validate_shape) {
        return shape_inference::MergeBothInputsShapeFn(c);
      }

      c->set_output(0, c->input(1));
      return Status::OK();
    });

REGISTER_OP("AssignAdd")
    .Input("ref: Ref(T)")
    .Input("value: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn);

REGISTER_OP("AssignSub")
    .Input("ref: Ref(T)")
    .Input("value: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn);

namespace {

Status ScatterUpdateShape(InferenceContext* c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSstate_opsDTcc mht_0(mht_0_v, 285, "", "./tensorflow/core/ops/state_ops.cc", "ScatterUpdateShape");

  ShapeHandle var_shape = c->input(0);
  ShapeHandle indices_shape = c->input(1);

  ShapeHandle unused_updates_shape;
  ShapeHandle concat;
  ShapeHandle var_subshape;
  TF_RETURN_IF_ERROR(c->Subshape(var_shape, 1, &var_subshape));
  TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, var_subshape, &concat));
  TF_RETURN_IF_ERROR(
      InferenceContext::Rank(c->input(2)) == 0
          ? Status::OK()
          : c->Merge(c->input(2), concat, &unused_updates_shape));

  c->set_output(0, var_shape);
  return Status::OK();
}

Status ScatterNdUpdateShape(InferenceContext* c) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPSstate_opsDTcc mht_1(mht_1_v, 306, "", "./tensorflow/core/ops/state_ops.cc", "ScatterNdUpdateShape");

  ShapeHandle input_shape = c->input(0);
  if (c->input_handle_shapes_and_types(0) != nullptr) {
    const auto& shape_and_type = *(c->input_handle_shapes_and_types(0));
    if (!shape_and_type.empty()) {
      input_shape = shape_and_type[0].shape;
    }
  }
  ShapeHandle indices_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &indices_shape));
  ShapeHandle updates_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 1, &updates_shape));
  return shape_inference::ScatterNdShapeHelper(c, indices_shape, updates_shape,
                                               input_shape);
}

}  // namespace

REGISTER_OP("ScatterUpdate")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = true")
    .SetShapeFn(ScatterUpdateShape);

REGISTER_OP("ScatterAdd")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterUpdateShape);

REGISTER_OP("ScatterSub")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterUpdateShape);

REGISTER_OP("ScatterMul")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterUpdateShape);

REGISTER_OP("ScatterDiv")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterUpdateShape);

REGISTER_OP("ScatterMin")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: {half, bfloat16, float, double, int32, int64}")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterUpdateShape);

REGISTER_OP("ScatterMax")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: {half, bfloat16, float, double, int32, int64}")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterUpdateShape);

REGISTER_OP("ScatterNdUpdate")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = true")
    .SetShapeFn(ScatterNdUpdateShape);

REGISTER_OP("ResourceScatterNdUpdate")
    .Input("ref: resource")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = true")
    .SetShapeFn(ScatterNdUpdateShape);

REGISTER_OP("ResourceScatterNdAdd")
    .Input("ref: resource")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = true")
    .SetShapeFn(ScatterNdUpdateShape);

REGISTER_OP("ResourceScatterNdSub")
    .Input("ref: resource")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = true")
    .SetShapeFn(ScatterNdUpdateShape);

REGISTER_OP("ResourceScatterNdMin")
    .Input("ref: resource")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = true")
    .SetShapeFn(ScatterNdUpdateShape);

REGISTER_OP("ResourceScatterNdMax")
    .Input("ref: resource")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = true")
    .SetShapeFn(ScatterNdUpdateShape);

REGISTER_OP("ScatterNdAdd")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterNdUpdateShape);

REGISTER_OP("ScatterNdSub")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterNdUpdateShape);

REGISTER_OP("ScatterNdMax")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterNdUpdateShape);

REGISTER_OP("ScatterNdMin")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterNdUpdateShape);

REGISTER_OP("CountUpTo")
    .Input("ref: Ref(T)")
    .Output("output: T")
    .Attr("limit: int")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle output;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &output));
      c->set_output(0, output);
      return Status::OK();
    });

REGISTER_OP("ResourceCountUpTo")
    .Input("resource: resource")
    .Output("output: T")
    .Attr("limit: int")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data == nullptr || handle_data->empty()) {
        return errors::InvalidArgument("Handle has no shape/type information.");
      }
      shape_inference::ShapeAndType shape_and_type = (*handle_data)[0];
      DataType value_dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("T", &value_dtype));
      if (value_dtype != shape_and_type.dtype) {
        return errors::InvalidArgument(
            "Data types do not match: ", DataTypeString(value_dtype), " and ",
            DataTypeString(shape_and_type.dtype));
      }
      ShapeHandle output;
      TF_RETURN_IF_ERROR(c->WithRank(shape_and_type.shape, 0, &output));
      c->set_output(0, output);
      return Status::OK();
    });

}  // namespace tensorflow
