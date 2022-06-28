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
class MHTracer_DTPStensorflowPScorePSopsPSresource_variable_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSresource_variable_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSresource_variable_opsDTcc() {
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

// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

#include <algorithm>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeAndType;
using ::tensorflow::shape_inference::ShapeHandle;

namespace tensorflow {

namespace {

Status ReadVariableShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSresource_variable_opsDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/ops/resource_variable_ops.cc", "ReadVariableShapeFn");

  // The user can add a "_shape" atribute to ReadVariableOp nodes. It is
  // useful for inferring shapes in a function, when no shape information
  // is passed about input resources. The user can annotate the graph using
  // the variable capture list of the function.
  // If the "_shape" attribute is found, it is used to set the output shape.
  PartialTensorShape p;
  Status annotation_found_status = c->GetAttr("_shape", &p);
  if (annotation_found_status.ok()) {
    ShapeHandle s;
    TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(p, &s));
    c->set_output(0, s);
  } else {
    std::vector<ShapeAndType> shape_and_type;
    TF_RETURN_IF_ERROR(
        shape_inference::ValidateVariableResourceHandle(c, &shape_and_type));
    c->set_output(0, shape_and_type[0].shape);
    if (shape_and_type[0].dtype == DT_VARIANT && shape_and_type.size() > 1) {
      std::vector<ShapeAndType> variant_shape_and_type;
      std::copy(shape_and_type.begin() + 1, shape_and_type.end(),
                std::back_inserter(variant_shape_and_type));
      c->set_output_handle_shapes_and_types(0, variant_shape_and_type);
    }
  }
  return Status::OK();
}

Status ReadVariablesShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPSresource_variable_opsDTcc mht_1(mht_1_v, 233, "", "./tensorflow/core/ops/resource_variable_ops.cc", "ReadVariablesShapeFn");

  int n;
  TF_RETURN_IF_ERROR(c->GetAttr("N", &n));
  DataTypeVector value_dtypes;
  TF_RETURN_IF_ERROR(c->GetAttr("dtypes", &value_dtypes));
  if (n != value_dtypes.size()) {
    return errors::InvalidArgument(
        "Mismatched number of arguments to ReadVariablesOp");
  }
  for (int i = 0; i < n; ++i) {
    ShapeAndType shape_and_type;
    auto* handle_data = c->input_handle_shapes_and_types(i);
    if (handle_data == nullptr || handle_data->empty()) {
      shape_and_type.shape = c->UnknownShape();
      shape_and_type.dtype = DT_INVALID;
    } else {
      shape_and_type = (*handle_data)[0];
      if (shape_and_type.dtype != value_dtypes[i]) {
        return errors::InvalidArgument(
            "Trying to read variable with wrong dtype. "
            "Expected ",
            DataTypeString(shape_and_type.dtype), " got ",
            DataTypeString(value_dtypes[i]));
      }
    }
    c->set_output(i, shape_and_type.shape);
  }
  return Status::OK();
}

}  // namespace

REGISTER_OP("VarHandleOp")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .Attr("allowed_devices: list(string) = []")
    .Output("resource: resource")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Scalar());
      DataType t;
      TF_RETURN_IF_ERROR(c->GetAttr("dtype", &t));
      PartialTensorShape p;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &p));
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(p, &s));
      c->set_output_handle_shapes_and_types(0,
                                            std::vector<ShapeAndType>{{s, t}});

      return Status::OK();
    });

REGISTER_OP("_VarHandlesOp")
    .Attr("containers: list(string)")
    .Attr("shared_names: list(string)")
    .Attr("N: int >= 0")
    .Attr("dtypes: list(type)")
    .Attr("shapes: list(shape)")
    .Output("resources: N * resource")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      int n;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &n));
      DataTypeVector dtypes;
      TF_RETURN_IF_ERROR(c->GetAttr("dtypes", &dtypes));
      std::vector<PartialTensorShape> shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("shapes", &shapes));
      if (dtypes.size() != n) {
        return errors::InvalidArgument("Mismatched number of dtypes (n=", n,
                                       ", num dtypes=", dtypes.size(), ")");
      }
      if (shapes.size() != n) {
        return errors::InvalidArgument("Mismatched number of shapes (n=", n,
                                       ", num shapes=", shapes.size(), ")");
      }
      for (int i = 0; i < n; ++i) {
        c->set_output(i, c->Scalar());
        ShapeHandle s;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(shapes[i], &s));
        c->set_output_handle_shapes_and_types(
            i, std::vector<ShapeAndType>{{s, dtypes[i]}});
      }

      return Status::OK();
    });

REGISTER_OP("ReadVariableOp")
    .Input("resource: resource")
    .Output("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn(ReadVariableShapeFn);

REGISTER_OP("_ReadVariablesOp")
    .Attr("N: int >= 0")
    .Input("resources: N * resource")
    .Output("values: dtypes")
    .Attr("dtypes: list(type)")
    .SetShapeFn(ReadVariablesShapeFn);

Status ReadGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSopsPSresource_variable_opsDTcc mht_2(mht_2_v, 337, "", "./tensorflow/core/ops/resource_variable_ops.cc", "ReadGrad");

  // clang-format off
  *g = FunctionDefHelper::Define(
      // Arg defs
      {"x: resource", "dy: float"},
      // Ret val defs
      {"dy: float"},
      // Attr defs
      {},
      // Nodes
      {});
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("ReadVariableOp", ReadGrad);

REGISTER_OP("DestroyResourceOp")
    .Input("resource: resource")
    .Attr("ignore_lookup_error: bool = true")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

Status CreateAssignShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSopsPSresource_variable_opsDTcc mht_3(mht_3_v, 362, "", "./tensorflow/core/ops/resource_variable_ops.cc", "CreateAssignShapeFn");

  std::vector<ShapeAndType> handle_shape_and_type;
  TF_RETURN_IF_ERROR(shape_inference::ValidateVariableResourceHandle(
      c, &handle_shape_and_type));

  ShapeHandle value_shape = c->input(1);
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(
      c->Merge(handle_shape_and_type[0].shape, value_shape, &unused));

  if (handle_shape_and_type[0].dtype == DT_VARIANT &&
      handle_shape_and_type.size() > 1 &&
      c->input_handle_shapes_and_types(1) != nullptr) {
    auto* value_handle_shape_and_type = c->input_handle_shapes_and_types(1);
    if (value_handle_shape_and_type->size() !=
        handle_shape_and_type.size() - 1) {
      return errors::InvalidArgument(
          "Incompatible handle variant shape_and_type size and input "
          "shape_and_type size: ",
          handle_shape_and_type.size() - 1, " vs. ",
          value_handle_shape_and_type->size());
    }
  }
  return Status::OK();
}

REGISTER_OP("AssignVariableOp")
    .Input("resource: resource")
    .Input("value: dtype")
    .Attr("dtype: type")
    .Attr("validate_shape: bool = false")
    .SetShapeFn(CreateAssignShapeFn);

REGISTER_OP("AssignAddVariableOp")
    .Input("resource: resource")
    .Input("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn(CreateAssignShapeFn);

REGISTER_OP("AssignSubVariableOp")
    .Input("resource: resource")
    .Input("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn(CreateAssignShapeFn);

REGISTER_OP("VarIsInitializedOp")
    .Input("resource: resource")
    .Output("is_initialized: bool")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);

Status VariableShapeShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSopsPSresource_variable_opsDTcc mht_4(mht_4_v, 415, "", "./tensorflow/core/ops/resource_variable_ops.cc", "VariableShapeShapeFn");

  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data == nullptr || handle_data->empty()) {
    c->set_output(0, c->Vector(c->UnknownDim()));
    return Status::OK();
  }
  ShapeHandle var_shape = (*handle_data)[0].shape;
  int64_t rank = c->RankKnown(var_shape) ? c->Rank(var_shape)
                                         : InferenceContext::kUnknownDim;
  c->set_output(0, c->Vector(rank));
  return Status::OK();
}

REGISTER_OP("VariableShape")
    .Input("input: resource")
    .Output("output: out_type")
    .Attr("out_type: {int32, int64} = DT_INT32")
    .SetShapeFn(VariableShapeShapeFn);

REGISTER_OP("ResourceGather")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Attr("batch_dims: int = 0")
    .Attr("validate_indices: bool = true")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn([](InferenceContext* c) {
      std::vector<ShapeAndType> handle_shape_and_type;
      TF_RETURN_IF_ERROR(shape_inference::ValidateVariableResourceHandle(
          c, &handle_shape_and_type));

      ShapeHandle indices_shape = c->input(1);

      ShapeHandle unused;
      int32_t batch_dims;
      TF_RETURN_IF_ERROR(c->GetAttr("batch_dims", &batch_dims));
      if (batch_dims < 0)
        return errors::InvalidArgument("batch_dims is negative (", batch_dims,
                                       ")");

      TF_RETURN_IF_ERROR(c->WithRankAtLeast(handle_shape_and_type[0].shape,
                                            batch_dims + 1, &unused));

      TF_RETURN_IF_ERROR(
          c->WithRankAtLeast(indices_shape, batch_dims, &unused));

      ShapeHandle params_subshape1;
      TF_RETURN_IF_ERROR(c->Subshape(handle_shape_and_type[0].shape, 0,
                                     batch_dims, &params_subshape1));

      ShapeHandle params_subshape2;
      TF_RETURN_IF_ERROR(c->Subshape(handle_shape_and_type[0].shape,
                                     batch_dims + 1, &params_subshape2));

      ShapeHandle indices_subshape;
      TF_RETURN_IF_ERROR(
          c->Subshape(indices_shape, batch_dims, &indices_subshape));

      // The out shape is params_shape[:batch_dims] +
      // indices_shape[batch_dims:] + params_shape[batch_dims+1:].
      ShapeHandle out;
      TF_RETURN_IF_ERROR(
          c->Concatenate(params_subshape1, indices_subshape, &out));
      TF_RETURN_IF_ERROR(c->Concatenate(out, params_subshape2, &out));

      c->set_output(0, out);
      if (handle_shape_and_type[0].dtype == DT_VARIANT &&
          !handle_shape_and_type.empty()) {
        std::vector<ShapeAndType> variant_shape_and_type;
        std::copy(handle_shape_and_type.begin() + 1,
                  handle_shape_and_type.end(),
                  std::back_inserter(variant_shape_and_type));
        c->set_output_handle_shapes_and_types(0, variant_shape_and_type);
      }
      return Status::OK();
    });

REGISTER_OP("ResourceGatherNd")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn(shape_inference::GatherNdShape);

namespace {

Status ResourceScatterUpdateShape(InferenceContext* c) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSopsPSresource_variable_opsDTcc mht_5(mht_5_v, 506, "", "./tensorflow/core/ops/resource_variable_ops.cc", "ResourceScatterUpdateShape");

  std::vector<ShapeAndType> handle_shape_and_type;
  TF_RETURN_IF_ERROR(shape_inference::ValidateVariableResourceHandle(
      c, &handle_shape_and_type));
  ShapeHandle var_shape = handle_shape_and_type[0].shape;
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
  if (handle_shape_and_type[0].dtype == DT_VARIANT &&
      handle_shape_and_type.size() > 1 &&
      c->input_handle_shapes_and_types(2) != nullptr) {
    auto* value_handle_shape_and_type = c->input_handle_shapes_and_types(2);
    if (value_handle_shape_and_type->size() !=
        handle_shape_and_type.size() - 1) {
      return errors::InvalidArgument(
          "Incompatible handle variant shape_and_type size and input "
          "shape_and_type size: ",
          handle_shape_and_type.size() - 1, " vs. ",
          value_handle_shape_and_type->size());
    }
  }
  return Status::OK();
}

}  // namespace

REGISTER_OP("ResourceScatterAdd")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: numbertype")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn(ResourceScatterUpdateShape);

REGISTER_OP("ResourceScatterSub")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: numbertype")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn(ResourceScatterUpdateShape);

REGISTER_OP("ResourceScatterMul")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: numbertype")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn(ResourceScatterUpdateShape);

REGISTER_OP("ResourceScatterDiv")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: numbertype")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn(ResourceScatterUpdateShape);

REGISTER_OP("ResourceScatterMin")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: numbertype")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn(ResourceScatterUpdateShape);

REGISTER_OP("ResourceScatterMax")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: numbertype")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn(ResourceScatterUpdateShape);

REGISTER_OP("ResourceScatterUpdate")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: type")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn(ResourceScatterUpdateShape);

REGISTER_OP("MutexV2")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("resource: resource")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("MutexLock")
    .Input("mutex: resource")
    .Output("mutex_lock: variant")
    .SetIsStateful()
    .SetTypeConstructor(full_type::Nullary(TFT_MUTEX_LOCK))
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("ConsumeMutexLock")
    .Input("mutex_lock: variant")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); });

}  // namespace tensorflow
