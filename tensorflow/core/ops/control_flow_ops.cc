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
class MHTracer_DTPStensorflowPScorePSopsPScontrol_flow_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPScontrol_flow_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPScontrol_flow_opsDTcc() {
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
using shape_inference::ShapeHandle;

// --------------------------------------------------------------------------
namespace {

Status SwitchShape(InferenceContext* c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPScontrol_flow_opsDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/ops/control_flow_ops.cc", "SwitchShape");

  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
  ShapeHandle out = c->input(0);
  c->set_output(0, out);
  c->set_output(1, out);

  // Handle resource shape / dtype.
  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data != nullptr) {
    c->set_output_handle_shapes_and_types(0, *handle_data);
    c->set_output_handle_shapes_and_types(1, *handle_data);
  }
  return Status::OK();
}

Status SwitchNShape(InferenceContext* c) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPScontrol_flow_opsDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/ops/control_flow_ops.cc", "SwitchNShape");

  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
  ShapeHandle out = c->input(0);
  int num_outs;
  TF_RETURN_IF_ERROR(c->GetAttr("num_outs", &num_outs));
  for (int i = 0; i < num_outs; i++) {
    c->set_output(i, out);
  }

  // Handle resource shape / dtype.
  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data != nullptr) {
    for (int i = 0; i < num_outs; i++) {
      c->set_output_handle_shapes_and_types(i, *handle_data);
    }
  }
  return Status::OK();
}

}  // namespace

REGISTER_OP("Switch")
    .Input("data: T")
    .Input("pred: bool")
    .Output("output_false: T")
    .Output("output_true: T")
    .Attr("T: type")
    .SetForwardTypeFn(full_type::ReplicateInput(0, 2))
    .SetShapeFn(SwitchShape);

REGISTER_OP("RefSwitch")
    .Input("data: Ref(T)")
    .Input("pred: bool")
    .Output("output_false: Ref(T)")
    .Output("output_true: Ref(T)")
    .Attr("T: type")
    .SetAllowsUninitializedInput()
    .SetShapeFn(SwitchShape);

REGISTER_OP("_SwitchN")
    .Input("data: T")
    .Input("output_index: int32")
    .Output("outputs: num_outs * T")
    .Attr("num_outs: int >= 1")
    .Attr("T: type")
    .SetShapeFn(SwitchNShape);

// --------------------------------------------------------------------------
REGISTER_OP("RefSelect")
    .Input("index: int32")
    .Input("inputs: Ref(N * T)")
    .Output("output: Ref(T)")
    .Attr("T: type")
    .Attr("N: int >= 1")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      ShapeHandle first_input = c->input(1);
      if (!c->FullyDefined(first_input)) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }
      // If any inputs aren't fully defined or don't match, we return unknown.
      for (int i = 2; i < c->num_inputs(); ++i) {
        ShapeHandle input = c->input(i);
        if (!c->FullyDefined(input) ||
            !c->Merge(first_input, input, &unused).ok()) {
          c->set_output(0, c->UnknownShape());
          return Status::OK();
        }
      }
      c->set_output(0, first_input);
      return Status::OK();
    });

// --------------------------------------------------------------------------
namespace {
Status MergeShape(InferenceContext* c) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSopsPScontrol_flow_opsDTcc mht_2(mht_2_v, 297, "", "./tensorflow/core/ops/control_flow_ops.cc", "MergeShape");

  ShapeHandle out = c->input(0);
  if (!c->RankKnown(out)) {
    out = c->UnknownShape();
  } else {
    int32_t rank = c->Rank(out);
    for (int i = 1; i < c->num_inputs(); ++i) {
      ShapeHandle input = c->input(i);
      if (!c->RankKnown(input) || c->Rank(input) != rank) {
        out = c->UnknownShape();
        break;
      }

      for (int d = 0; d < rank; ++d) {
        if (c->Value(c->Dim(input, d)) != c->Value(c->Dim(out, d))) {
          TF_RETURN_IF_ERROR(c->ReplaceDim(out, d, c->UnknownDim(), &out));
        }
      }
    }
  }
  c->set_output(0, out);
  c->set_output(1, c->Scalar());
  return Status::OK();
}
}  // namespace

REGISTER_OP("Merge")
    .Input("inputs: N * T")
    .Output("output: T")
    .Output("value_index: int32")
    .Attr("T: type")
    .Attr("N: int >= 1")
    .SetForwardTypeFn(full_type::Merge())
    .SetShapeFn(MergeShape);

REGISTER_OP("RefMerge")
    .Input("inputs: Ref(N * T)")
    .Output("output: Ref(T)")
    .Output("value_index: int32")
    .Attr("T: type")
    .Attr("N: int >= 1")
    .SetShapeFn(MergeShape);

// --------------------------------------------------------------------------
REGISTER_OP("Enter")
    .Input("data: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("frame_name: string")
    .Attr("is_constant: bool = false")
    .Attr("parallel_iterations: int = 10")
    .SetForwardTypeFn(full_type::ReplicateInput())
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->UnknownShape());

      // Handle resource shape / dtype, if present.
      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data != nullptr) {
        c->set_output_handle_shapes_and_types(0, *handle_data);
      }
      // Propagate shape if output is a constant.
      bool is_constant;
      TF_RETURN_IF_ERROR(c->GetAttr("is_constant", &is_constant));
      if (is_constant) {
        c->set_output(0, c->input(0));
      }

      return Status::OK();
    });

// --------------------------------------------------------------------------
REGISTER_OP("RefEnter")
    .Input("data: Ref(T)")
    .Output("output: Ref(T)")
    .Attr("T: type")
    .Attr("frame_name: string")
    .Attr("is_constant: bool = false")
    .Attr("parallel_iterations: int = 10")
    .SetShapeFn(shape_inference::UnchangedShape);

// --------------------------------------------------------------------------
REGISTER_OP("Exit")
    .Input("data: T")
    .Output("output: T")
    .Attr("T: type")
    .SetForwardTypeFn(full_type::ReplicateInput())
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("RefExit")
    .Input("data: Ref(T)")
    .Output("output: Ref(T)")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnchangedShape);

// --------------------------------------------------------------------------
REGISTER_OP("NextIteration")
    .Input("data: T")
    .Output("output: T")
    .Attr("T: type")
    .SetForwardTypeFn(full_type::ReplicateInput())
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("RefNextIteration")
    .Input("data: Ref(T)")
    .Output("output: Ref(T)")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnchangedShape);

// --------------------------------------------------------------------------
REGISTER_OP("LoopCond")
    .Input("input: bool")
    .Output("output: bool")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRank(c, 0);
    });

// --------------------------------------------------------------------------
REGISTER_OP("ControlTrigger").SetShapeFn(shape_inference::NoOutputs);

// --------------------------------------------------------------------------
REGISTER_OP("Abort")
    .Attr("error_msg: string = ''")
    .Attr("exit_without_error: bool = false")
    .SetShapeFn(shape_inference::NoOutputs);

}  // namespace tensorflow
