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
class MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradientsDTcc {
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
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradientsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradientsDTcc() {
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

#include "tensorflow/cc/framework/while_gradients.h"

#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/while_loop.h"

namespace tensorflow {
namespace {

using ops::BodyGraphBuilderFn;
using ops::BuildWhileLoop;
using ops::CondGraphBuilderFn;

Output ToOutput(OutputTensor output_tensor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradientsDTcc mht_0(mht_0_v, 200, "", "./tensorflow/cc/framework/while_gradients.cc", "ToOutput");

  return Output(const_cast<Node*>(output_tensor.node), output_tensor.index);
}

std::vector<Output> ToOutputVector(
    const std::vector<OutputTensor>& output_tensors) {
  const int n = output_tensors.size();
  std::vector<Output> result;
  result.reserve(n);
  for (int i = 0; i < n; ++i) result.push_back(ToOutput(output_tensors[i]));
  return result;
}

// The backprop loop counter and main backprop loop run in their own execution
// frame (conceptually, the main forward loop and forward loop counter run
// together in a frame, then the backprop loop counter and backprop loop run
// together in a different frame). This returns the frame name to use for the
// backprop while loops.
// TODO(skyewm): make sure this is unique among existing frame names
string BackPropFrameName(const string& forward_frame_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("forward_frame_name: \"" + forward_frame_name + "\"");
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradientsDTcc mht_1(mht_1_v, 223, "", "./tensorflow/cc/framework/while_gradients.cc", "BackPropFrameName");

  return strings::StrCat(forward_frame_name, "_backprop");
}

// Creates a loop that counts the number of iterations performed by the
// while loop associated with `while_ctx`. The returned output yields the
// iteration count.
Status AddForwardLoopCounter(WhileContext* while_ctx, const Scope& scope,
                             Output* count) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradientsDTcc mht_2(mht_2_v, 234, "", "./tensorflow/cc/framework/while_gradients.cc", "AddForwardLoopCounter");

  // Create while loop:
  //   i = 0
  //   while forward loop predicate is true:
  //     ++i

  Output zero = ops::Const(scope, 0, {});

  // Condition function that returns condition output from original while loop.
  CondGraphBuilderFn cond_fn = [while_ctx](const Scope& scope,
                                           const std::vector<Output>& inputs,
                                           Output* output) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradientsDTcc mht_3(mht_3_v, 248, "", "./tensorflow/cc/framework/while_gradients.cc", "lambda");

    *output = ToOutput(while_ctx->cond_output());
    return Status::OK();
  };

  // Body function that adds one to input.
  BodyGraphBuilderFn body_fn = [](const Scope& scope,
                                  const std::vector<Output>& inputs,
                                  std::vector<Output>* outputs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradientsDTcc mht_4(mht_4_v, 259, "", "./tensorflow/cc/framework/while_gradients.cc", "lambda");

    DCHECK_EQ(inputs.size(), 1);
    outputs->emplace_back(ops::Add(scope, inputs[0], 1));
    return scope.status();
  };

  // Note that this loop runs in the same execution frame as the forward loop.
  std::vector<Output> outputs;
  TF_RETURN_IF_ERROR(BuildWhileLoop(scope, {zero}, cond_fn, body_fn,
                                    while_ctx->frame_name(), &outputs,
                                    /* create_while_ctx */ false));
  *count = outputs[0];
  return Status::OK();
}

// Creates a loop that executes `loop_count` times. The returned output is the
// boolean predicate indicating if the loop is still executing. This is used to
// drive the gradient computation for the while loop associated with
// `while_ctx`.
Status AddBackPropLoopCounter(WhileContext* while_ctx, const Output& loop_count,
                              const Scope& scope,
                              Output* backprop_execution_pred) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradientsDTcc mht_5(mht_5_v, 283, "", "./tensorflow/cc/framework/while_gradients.cc", "AddBackPropLoopCounter");

  // Create while loop:
  //   n = loop_count
  //   while n > 0:
  //     --n

  // Condition function that returns input > 0.
  CondGraphBuilderFn cond_fn = [](const Scope& scope,
                                  const std::vector<Output>& inputs,
                                  Output* output) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradientsDTcc mht_6(mht_6_v, 295, "", "./tensorflow/cc/framework/while_gradients.cc", "lambda");

    DCHECK_EQ(inputs.size(), 1);
    *output = ops::Greater(scope, inputs[0], 0);
    return scope.status();
  };

  // Body function that subtracts one from input.
  BodyGraphBuilderFn body_fn = [](const Scope& scope,
                                  const std::vector<Output>& inputs,
                                  std::vector<Output>* outputs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradientsDTcc mht_7(mht_7_v, 307, "", "./tensorflow/cc/framework/while_gradients.cc", "lambda");

    DCHECK_EQ(inputs.size(), 1);
    outputs->emplace_back(ops::Subtract(scope, inputs[0], 1));
    return scope.status();
  };

  string frame_name = BackPropFrameName(while_ctx->frame_name());
  std::vector<Output> outputs;
  TF_RETURN_IF_ERROR(BuildWhileLoop(
      scope, {loop_count}, cond_fn, body_fn, frame_name, &outputs,
      /* create_while_ctx */ false, backprop_execution_pred));
  return Status::OK();
}

// Creates the main backprop loop that computes the gradient of the loop
// associated with `while_ctx`. `grad_inputs` are the partial derivatives
// w.r.t. the loop outputs, i.e. the exit nodes. `backprop_execution_pred` is
// the predicate to use for the backprop loop (see AddBackPropLoopCounter()).
// The partial derivatives w.r.t. the loop inputs, i.e. the input loop vars, are
// returned in `grad_outputs`.
Status AddWhileGradientLoop(WhileContext* while_ctx,
                            const std::vector<Output>& grad_inputs,
                            const Output& backprop_execution_pred,
                            const Scope& parent_scope,
                            std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradientsDTcc mht_8(mht_8_v, 334, "", "./tensorflow/cc/framework/while_gradients.cc", "AddWhileGradientLoop");

  DCHECK_EQ(grad_inputs.size(), while_ctx->body_outputs().size());
  DCHECK_EQ(while_ctx->body_inputs().size(), while_ctx->body_outputs().size());

  Scope scope = parent_scope.NewSubScope("while");

  // Create while loop:
  //   while backprop_execution_pred:
  //     forward loop body gradient

  // Condition function that returns 'backprop_execution_pred'.
  CondGraphBuilderFn cond_fn = [backprop_execution_pred](
                                   const Scope& scope,
                                   const std::vector<Output>& inputs,
                                   Output* output) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradientsDTcc mht_9(mht_9_v, 351, "", "./tensorflow/cc/framework/while_gradients.cc", "lambda");

    *output = backprop_execution_pred;
    return Status::OK();
  };

  // Body function that builds while body gradient subgraph.
  BodyGraphBuilderFn body_fn = [while_ctx](const Scope& scope,
                                           const std::vector<Output>& inputs,
                                           std::vector<Output>* outputs) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradientsDTcc mht_10(mht_10_v, 362, "", "./tensorflow/cc/framework/while_gradients.cc", "lambda");

    std::vector<Output> body_outputs =
        ToOutputVector(while_ctx->body_outputs());
    std::vector<Output> body_inputs = ToOutputVector(while_ctx->body_inputs());
    return AddSymbolicGradients(scope, body_outputs, body_inputs, inputs,
                                outputs);
  };

  string frame_name = BackPropFrameName(while_ctx->frame_name());
  TF_RETURN_IF_ERROR(BuildWhileLoop(scope, grad_inputs, cond_fn, body_fn,
                                    frame_name, grad_outputs,
                                    /* create_while_ctx */ false));
  return Status::OK();
}

}  // namespace

Status AddWhileLoopGradient(WhileContext* while_ctx, const Scope& scope,
                            const std::vector<Output>& grad_inputs,
                            std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSwhile_gradientsDTcc mht_11(mht_11_v, 384, "", "./tensorflow/cc/framework/while_gradients.cc", "AddWhileLoopGradient");

  Output forward_loop_count;
  TF_RETURN_IF_ERROR(AddForwardLoopCounter(
      while_ctx, scope.NewSubScope("ForwardLoopCounter"), &forward_loop_count));

  // TODO(skyewm): can we combine the backprop loop counter and main gradient
  // loop into a single loop? The original Python code doesn't combine the
  // loops, but I'm not sure why.
  Output backprop_counter_cond;
  TF_RETURN_IF_ERROR(AddBackPropLoopCounter(
      while_ctx, forward_loop_count, scope.NewSubScope("BackPropLoopCounter"),
      &backprop_counter_cond));

  return AddWhileGradientLoop(while_ctx, grad_inputs, backprop_counter_cond,
                              scope, grad_outputs);
}

}  // namespace tensorflow
