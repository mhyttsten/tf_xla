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
class MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScppPSrenderersPSop_implementation_rendererDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScppPSrenderersPSop_implementation_rendererDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScppPSrenderersPSop_implementation_rendererDTcc() {
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
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/op_implementation_renderer.h"

#include "tensorflow/c/experimental/ops/gen/common/view_util.h"
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/renderer.h"
#include "tensorflow/c/experimental/ops/gen/cpp/views/arg_view.h"
#include "tensorflow/c/experimental/ops/gen/cpp/views/attr_view.h"
#include "tensorflow/c/experimental/ops/gen/cpp/views/op_view.h"

namespace tensorflow {
namespace generator {
namespace cpp {

OpImplementationRenderer::OpImplementationRenderer(RendererContext context,
                                                   OpView op)
    : Renderer(context), op_(op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScppPSrenderersPSop_implementation_rendererDTcc mht_0(mht_0_v, 198, "", "./tensorflow/c/experimental/ops/gen/cpp/renderers/op_implementation_renderer.cc", "OpImplementationRenderer::OpImplementationRenderer");
}

void OpImplementationRenderer::Render() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScppPSrenderersPSop_implementation_rendererDTcc mht_1(mht_1_v, 203, "", "./tensorflow/c/experimental/ops/gen/cpp/renderers/op_implementation_renderer.cc", "OpImplementationRenderer::Render");

  RenderInitialization();

  if (op_.IsListOp()) {
    RenderExecutionListOp();
  } else if (op_.NumOutputs() == 0) {
    RenderExecutionZeroOutputs();
  } else if (op_.NumOutputs() == 1) {
    RenderExecutionSingleOutput();
  } else {
    RenderExecutionMultipleOutputs();
  }
}

void OpImplementationRenderer::RenderInitialization() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScppPSrenderersPSop_implementation_rendererDTcc mht_2(mht_2_v, 220, "", "./tensorflow/c/experimental/ops/gen/cpp/renderers/op_implementation_renderer.cc", "OpImplementationRenderer::RenderInitialization");

  // Create Op variable and initialize it
  Statement("AbstractOperationPtr $0(ctx->CreateOperation())",
            op_.VariableName());
  TFStatement(Call(op_.VariableName(), "Reset",
                   {op_.OpNameString(), "/*raw_device_name=*/nullptr"}));
  TFStatement(Call("MaybeSetOpName", {op_.VariableName() + ".get()", "name"}));
  // Set each input
  for (const ArgView& ar : op_.Inputs()) {
    TFStatement(Call(op_.VariableName(), ar.SetterMethod(), ar.SetterArgs()));
  }
  // Set each attribute
  for (const AttrView& ar : op_.Attributes()) {
    TFStatement(Call(op_.VariableName(), ar.SetterMethod(), ar.SetterArgs()));
  }
}

void OpImplementationRenderer::RenderExecutionListOp() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScppPSrenderersPSop_implementation_rendererDTcc mht_3(mht_3_v, 240, "", "./tensorflow/c/experimental/ops/gen/cpp/renderers/op_implementation_renderer.cc", "OpImplementationRenderer::RenderExecutionListOp");

  ArgView input_arg = op_.OnlyInput();
  ArgView output_arg = op_.OnlyOutput();
  Statement("int num_retvals = $0.size()", output_arg.VariableName());
  Statement("return " + Call(op_.VariableName(), "Execute",
                             {output_arg.VariableName(), "&num_retvals"}));
}

void OpImplementationRenderer::RenderExecutionSingleOutput() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScppPSrenderersPSop_implementation_rendererDTcc mht_4(mht_4_v, 251, "", "./tensorflow/c/experimental/ops/gen/cpp/renderers/op_implementation_renderer.cc", "OpImplementationRenderer::RenderExecutionSingleOutput");

  ArgView output_arg = op_.OnlyOutput();
  Statement("int num_retvals = 1");
  Statement("return $0->Execute(absl::MakeSpan($1, 1), &num_retvals)",
            op_.VariableName(), output_arg.VariableName());
}

void OpImplementationRenderer::RenderExecutionMultipleOutputs() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScppPSrenderersPSop_implementation_rendererDTcc mht_5(mht_5_v, 261, "", "./tensorflow/c/experimental/ops/gen/cpp/renderers/op_implementation_renderer.cc", "OpImplementationRenderer::RenderExecutionMultipleOutputs");

  Statement("int num_retvals = $0", op_.NumOutputs());
  Statement("AbstractTensorHandle* temp_outputs[$0]", op_.NumOutputs());
  Statement("Status status = $0->Execute(temp_outputs, &num_retvals)",
            op_.VariableName());

  for (const ArgView& arg : op_.Outputs()) {
    Statement("*$0 = temp_outputs[$1]", arg.VariableName(), arg.Position());
  }

  Statement("return status");
}

void OpImplementationRenderer::RenderExecutionZeroOutputs() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScppPSrenderersPSop_implementation_rendererDTcc mht_6(mht_6_v, 277, "", "./tensorflow/c/experimental/ops/gen/cpp/renderers/op_implementation_renderer.cc", "OpImplementationRenderer::RenderExecutionZeroOutputs");

  Statement("int num_retvals = 0");
  Statement("std::vector<AbstractTensorHandle*> dummy_outputs");
  Statement("return $0->Execute(absl::MakeSpan(dummy_outputs), &num_retvals)",
            op_.VariableName());
}

}  // namespace cpp
}  // namespace generator
}  // namespace tensorflow
