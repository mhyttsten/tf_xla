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
class MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSmultiplex_4PSmultiplex_4_opDTcc {
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
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSmultiplex_4PSmultiplex_4_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSmultiplex_4PSmultiplex_4_opDTcc() {
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

#include <cstdint>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

// Use a namespace when registering by prepending the
// package's name to the op’s name and separate with a '>'.
// This is the recommendation for out-of-tree ops to avoid name collisions in
// "Best practices for custom operations in TensorFlow"
// https://github.com/tensorflow/community/blob/master/rfcs/20190726-custom-ops.md

// Please use the appropriate namespace for your project
namespace tensorflow {
namespace custom_op_examples {

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

Status MultiplexShapeFunction(InferenceContext* c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSmultiplex_4PSmultiplex_4_opDTcc mht_0(mht_0_v, 203, "", "./tensorflow/examples/custom_ops_doc/multiplex_4/multiplex_4_op.cc", "MultiplexShapeFunction");

  int64_t num_cond_a;
  TF_RETURN_IF_ERROR(c->GetAttr("N", &num_cond_a));
  tensorflow::shape_inference::ShapeHandle unused;
  // There are N `cond` inputs, N `a` inputs and 1 `b` input in this order.
  // Check that all `cond` inputs and all `a` inputs have the same shape
  // as the `b` input.
  int64_t last = 2 * num_cond_a;
  for (int64_t i = 0; i < last; i++) {
    TF_RETURN_IF_ERROR(c->Merge(c->input(i), c->input(last), &unused));
  }
  c->set_output(0, c->input(last));
  return tensorflow::Status::OK();
}

REGISTER_OP("Examples>MultiplexDense")
    .Input("cond: N * bool")
    .Input("a_values: N * T")
    .Input("b_values: T")
    .Output("output_values: T")
    .Attr("T: type")
    .Attr("N: int = 1")
    .SetShapeFn(MultiplexShapeFunction)
    .Doc(R"doc(
Return elements chosen from `a_values` or `b_values` depending on `cond`.

When `a_values` and `cond` are tenors (i.e. N=1), this is similar to `np.where`
and `tf.where`. When `a_values` and `cond` are lists of tensors (i.e. N>1),
this is similar to `np.select`. In either case these are simplified to only
handle dense tensors, no optional parameters, no broadcasting, etc..

cond: tf.Tensor or list of tf.Tensor of type bool. If it is a list, `a_values`
      must be a list of the same length. Where True, yield the corresponding
      element from `a_values` (with priority to the first one encountered in
      lists), otherwise yield `b_values`.
a_values: tf.Tensor or list of tf.Tensor. Each tensor has the same type and
          shape as `b_values`. If it is a list, `cond` must be a list of the
          same length.
b_values: tf.Tensor with the same type and shape as the `a_values` if it is a
          tensor or as every element of `a_values` if `a_values` is a list.
output_values: A tf.Tensor with elements from `a_values` where `cond` is True,
               and elements from `b` elsewhere.
)doc");

}  // namespace custom_op_examples
}  // namespace tensorflow
