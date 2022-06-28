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
class MHTracer_DTPStensorflowPScorePSopsPSfunctional_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSfunctional_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSfunctional_opsDTcc() {
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
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::InferenceContext;

REGISTER_OP("SymbolicGradient")
    .Input("input: Tin")
    .Output("output: Tout")
    .Attr("Tin: list(type)")
    .Attr("Tout: list(type)")
    .Attr("f: func")
    .SetShapeFn([](InferenceContext* c) {
      if (c->num_inputs() < c->num_outputs()) {
        return errors::InvalidArgument("len(inputs) < len(outputs)");
      }
      std::vector<DataType> types;
      TF_RETURN_IF_ERROR(c->GetAttr("Tin", &types));
      // Say, (u, v) = f(x, y, z), _symbolic_gradient(f) is a function of
      // (x, y, z, du, dv) -> (dx, dy, dz). Therefore, shapes of its
      // outputs (dx, dy, dz) are the same as (x, y, z).
      for (int i = 0; i < c->num_outputs(); ++i) {
        if (types[i] == DT_RESOURCE) {
          const std::vector<shape_inference::ShapeAndType>* handle_type =
              c->input_handle_shapes_and_types(i);
          if (handle_type != nullptr) {
            c->set_output(i, handle_type->at(0).shape);
          } else {
            c->set_output(i, c->UnknownShape());
          }
        } else {
          c->set_output(i, c->input(i));
        }
      }
      return Status::OK();
    });

REGISTER_OP("RemoteCall")
    .Input("target: string")
    .Input("args: Tin")
    .Output("output: Tout")
    .Attr("Tin: list(type)")
    .Attr("Tout: list(type)")
    .Attr("f: func")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(drpng): remove this.
REGISTER_OP("_If")
    .Input("cond: Tcond")
    .Input("input: Tin")
    .Output("output: Tout")
    .Attr("Tcond: type")
    .Attr("Tin: list(type)")
    .Attr("Tout: list(type)")
    .Attr("then_branch: func")
    .Attr("else_branch: func")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
output = cond ? then_branch(input) : else_branch(input)

cond: A Tensor. If the tensor is a scalar of non-boolean type, the
    scalar is converted to a boolean according to the
    following rule: if the scalar is a numerical value, non-zero means
    True and zero means False; if the scalar is a string, non-empty
    means True and empty means False. If the tensor is not a scalar,
    being empty means False and being non-empty means True.
input: A list of input tensors.
then_branch: A function that takes 'inputs' and returns a list of
    tensors, whose types are the same as what else_branch returns.
else_branch: A function that takes 'inputs' and returns a list of
    tensors.  whose types are the same as what then_branch returns.
)doc");

Status IfShapeInferenceFn(shape_inference::InferenceContext* c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSfunctional_opsDTcc mht_0(mht_0_v, 262, "", "./tensorflow/core/ops/functional_ops.cc", "IfShapeInferenceFn");

  std::vector<PartialTensorShape> output_shapes;
  TF_RETURN_IF_ERROR(c->GetAttr("output_shapes", &output_shapes));
  // If `output_shapes` attr is set use that as the shapes of the outputs
  // else return unknown shapes.
  if (output_shapes.empty()) return shape_inference::UnknownShape(c);
  if (output_shapes.size() != c->num_outputs()) {
    return errors::InvalidArgument(
        "`output_shapes` must be the same length as num outputs (",
        output_shapes.size(), " vs. ", c->num_outputs());
  }
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    shape_inference::ShapeHandle output_shape_handle;
    TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(
        output_shapes[i], &output_shape_handle));
    c->set_output(static_cast<int>(i), output_shape_handle);
  }
  return Status::OK();
}

REGISTER_OP("StatelessIf")
    .Input("cond: Tcond")
    .Input("input: Tin")
    .Output("output: Tout")
    .Attr("Tcond: type")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .Attr("then_branch: func")
    .Attr("else_branch: func")
    .Attr("output_shapes: list(shape) = []")
    .SetShapeFn(IfShapeInferenceFn);

REGISTER_OP("If")
    .Input("cond: Tcond")
    .Input("input: Tin")
    .Output("output: Tout")
    .Attr("Tcond: type")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .Attr("then_branch: func")
    .Attr("else_branch: func")
    .Attr("output_shapes: list(shape) = []")
    .SetIsStateful()
    .SetShapeFn(IfShapeInferenceFn);

Status CaseShapeInferenceFn(shape_inference::InferenceContext* c) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPSfunctional_opsDTcc mht_1(mht_1_v, 310, "", "./tensorflow/core/ops/functional_ops.cc", "CaseShapeInferenceFn");

  std::vector<PartialTensorShape> output_shapes;
  TF_RETURN_IF_ERROR(c->GetAttr("output_shapes", &output_shapes));
  // If `output_shapes` attr is set use that as the shapes of the outputs
  // else return unknown shapes.
  if (output_shapes.empty()) return shape_inference::UnknownShape(c);
  if (output_shapes.size() != c->num_outputs()) {
    return errors::InvalidArgument(
        "`output_shapes` must be the same length as num outputs (",
        output_shapes.size(), " vs. ", c->num_outputs());
  }
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    shape_inference::ShapeHandle output_shape_handle;
    TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(
        output_shapes[i], &output_shape_handle));
    c->set_output(static_cast<int>(i), output_shape_handle);
  }
  return Status::OK();
}

REGISTER_OP("StatelessCase")
    .Input("branch_index: int32")
    .Input("input: Tin")
    .Output("output: Tout")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .Attr("branches: list(func) >= 1")
    .Attr("output_shapes: list(shape) = []")
    .SetShapeFn(CaseShapeInferenceFn);

REGISTER_OP("Case")
    .Input("branch_index: int32")
    .Input("input: Tin")
    .Output("output: Tout")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .Attr("branches: list(func) >= 1")
    .Attr("output_shapes: list(shape) = []")
    .SetIsStateful()
    .SetShapeFn(CaseShapeInferenceFn);

// TODO(drpng): remove this.
REGISTER_OP("_While")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: list(type) >= 0")
    .Attr("cond: func")
    .Attr("body: func")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      for (int i = 0; i < c->num_outputs(); ++i) {
        c->set_output(i, c->input(i));
      }
      return Status::OK();
    })
    .Doc(R"doc(
output = input; While (Cond(output)) { output = Body(output) }

input: A list of input tensors whose types are T.
output: A list of output tensors whose types are T.
cond: A function takes 'input' and returns a tensor.  If the tensor is
    a scalar of non-boolean, the scalar is converted to a boolean
    according to the following rule: if the scalar is a numerical
    value, non-zero means True and zero means False; if the scalar is
    a string, non-empty means True and empty means False. If the
    tensor is not a scalar, non-emptiness means True and False
    otherwise.
body: A function that takes a list of tensors and returns another
      list of tensors. Both lists have the same types as specified
      by T.
)doc");

Status WhileShapeInferenceFn(shape_inference::InferenceContext* c) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSopsPSfunctional_opsDTcc mht_2(mht_2_v, 385, "", "./tensorflow/core/ops/functional_ops.cc", "WhileShapeInferenceFn");

  std::vector<PartialTensorShape> output_shapes;
  TF_RETURN_IF_ERROR(c->GetAttr("output_shapes", &output_shapes));
  // If `output_shapes` attr is set use that as the shapes of the outputs
  // else use the input shapes.
  if (!output_shapes.empty()) {
    if (output_shapes.size() != c->num_outputs()) {
      return errors::InvalidArgument(
          "`output_shapes` must be the same length as num outputs (",
          output_shapes.size(), " vs. ", c->num_outputs());
    }
    for (size_t i = 0; i < output_shapes.size(); ++i) {
      shape_inference::ShapeHandle output_shape_handle;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(
          output_shapes[i], &output_shape_handle));
      c->set_output(static_cast<int>(i), output_shape_handle);
    }
  } else {
    for (int i = 0; i < c->num_outputs(); ++i) {
      c->set_output(i, c->input(i));
    }
  }
  return Status::OK();
}

REGISTER_OP("While")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: list(type) >= 0")
    .Attr("cond: func")
    .Attr("body: func")
    .Attr("output_shapes: list(shape) = []")
    .Attr("parallel_iterations: int = 10")
    .SetIsStateful()
    .SetShapeFn(WhileShapeInferenceFn);

REGISTER_OP("StatelessWhile")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: list(type) >= 0")
    .Attr("cond: func")
    .Attr("body: func")
    .Attr("output_shapes: list(shape) = []")
    .Attr("parallel_iterations: int = 10")
    .SetShapeFn(WhileShapeInferenceFn);

REGISTER_OP("ToBool")
    .Input("input: T")
    .Output("output: bool")
    .Attr("T: type")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("For")
    .Input("start: int32")
    .Input("limit: int32")
    .Input("delta: int32")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: list(type) >= 0")
    .Attr("body: func")
    .SetShapeFn(shape_inference::UnknownShape);

// While no useful shape function is registered for function call ops directly,
// ShapeRefiner is run by default to perform shape inference.
REGISTER_OP("PartitionedCall")
    .Input("args: Tin")
    .Output("output: Tout")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .Attr("f: func")
    .Attr("config: string = ''")
    .Attr("config_proto: string = ''")
    .Attr("executor_type: string = ''")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("StatefulPartitionedCall")
    .Input("args: Tin")
    .Output("output: Tout")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .Attr("f: func")
    .Attr("config: string = ''")  // Deprecated in favor of config_proto
    .Attr("config_proto: string = ''")
    .Attr("executor_type: string = ''")
    .SetIsStateful()
    .SetIsDistributedCommunication()
    .SetShapeFn(shape_inference::UnknownShape);

// This op is used as a placeholder in If branch functions. It doesn't provide a
// valid output when run, so must either be removed (e.g. replaced with a
// function input) or guaranteed not to be used (e.g. if mirroring an
// intermediate output needed for the gradient computation of the other branch).
REGISTER_OP("FakeParam")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .SetShapeFn([](InferenceContext* c) {
      PartialTensorShape shape;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &shape));
      shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(shape, &out));
      c->set_output(0, out);
      return Status::OK();
    });

// Returns the device index.
REGISTER_OP("DeviceIndex")
    .Output("index: int32")
    .Attr("device_names: list(string)")
    .SetShapeFn(shape_inference::ScalarShape)
    .SetDoNotOptimize();

}  // end namespace tensorflow
