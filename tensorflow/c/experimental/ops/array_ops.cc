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
class MHTracer_DTPStensorflowPScPSexperimentalPSopsPSarray_opsDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSarray_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSopsPSarray_opsDTcc() {
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

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/c/experimental/ops/array_ops.h"

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/tracing_utils.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"

using tensorflow::tracing::MaybeSetOpName;

namespace tensorflow {
namespace ops {

// Op: Identity()
// Summary: Return a tensor with the same shape and contents as the input tensor
// or value.
//
// Description:
Status Identity(AbstractContext* ctx, AbstractTensorHandle* const input,
                AbstractTensorHandle** output, const char* name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSarray_opsDTcc mht_0(mht_0_v, 207, "", "./tensorflow/c/experimental/ops/array_ops.cc", "Identity");

  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Identity", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(input));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(output, 1), &num_retvals);
}

// Op: IdentityN()
// Summary: Returns a list of tensors with the same shapes and contents as the
// input
//
// Description:
//   tensors.
//
//   This op can be used to override the gradient for complicated functions. For
//   example, suppose y = f(x) and we wish to apply a custom function g for
//   backprop such that dx = g(dy). In Python,
//
//   ```python
//   with tf.get_default_graph().gradient_override_map(
//       {'IdentityN': 'OverrideGradientWithG'}):
//     y, _ = identity_n([f(x), x])
//
//   @tf.RegisterGradient('OverrideGradientWithG')
//   def ApplyG(op, dy, _):
//     return [None, g(dy)]  # Do not backprop to f(x).
//   ```
Status IdentityN(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> input,
                 absl::Span<AbstractTensorHandle*> output, const char* name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSarray_opsDTcc mht_1(mht_1_v, 242, "", "./tensorflow/c/experimental/ops/array_ops.cc", "IdentityN");

  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("IdentityN", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInputList(input));
  int num_retvals = output.size();
  return op_ptr->Execute(output, &num_retvals);
}

// Op: ZerosLike()
// Summary: Returns a tensor of zeros with the same shape and type as x.
//
// Description:
Status ZerosLike(AbstractContext* ctx, AbstractTensorHandle* const x,
                 AbstractTensorHandle** y, const char* name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSarray_opsDTcc mht_2(mht_2_v, 260, "", "./tensorflow/c/experimental/ops/array_ops.cc", "ZerosLike");

  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("ZerosLike", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(y, 1), &num_retvals);
}

// Op: Shape()
// Summary: Returns the shape of a tensor.
//
// Description:
//   This operation returns a 1-D integer tensor representing the shape of
//   `input`.
//
//   For example:
//
//   ```
//   # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
//   shape(t) ==> [2, 2, 3]
//   ```
Status Shape(AbstractContext* ctx, AbstractTensorHandle* const input,
             AbstractTensorHandle** output, DataType out_type,
             const char* name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSarray_opsDTcc mht_3(mht_3_v, 288, "", "./tensorflow/c/experimental/ops/array_ops.cc", "Shape");

  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Shape", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(input));
  TF_RETURN_IF_ERROR(op_ptr->SetAttrType("out_type", out_type));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(output, 1), &num_retvals);
}

// Op: ExpandDims()
// Summary: Inserts a dimension of 1 into a tensor's shape.
//
// Description:
//   Given a tensor `input`, this operation inserts a dimension of 1 at the
//   dimension index `axis` of `input`'s shape. The dimension index `axis`
//   starts at zero; if you specify a negative number for `axis` it is counted
//   backward from the end.
//
//   This operation is useful if you want to add a batch dimension to a single
//   element. For example, if you have a single image of shape `[height, width,
//   channels]`, you can make it a batch of 1 image with `expand_dims(image,
//   0)`, which will make the shape `[1, height, width, channels]`.
//
//   Other examples:
//
//   ```
//   # 't' is a tensor of shape [2]
//   shape(expand_dims(t, 0)) ==> [1, 2]
//   shape(expand_dims(t, 1)) ==> [2, 1]
//   shape(expand_dims(t, -1)) ==> [2, 1]
//
//   # 't2' is a tensor of shape [2, 3, 5]
//   shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
//   shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
//   shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
//   ```
//
//   This operation requires that:
//
//   `-1-input.dims() <= dim <= input.dims()`
//
//   This operation is related to `squeeze()`, which removes dimensions of
//   size 1.
Status ExpandDims(AbstractContext* ctx, AbstractTensorHandle* const input,
                  AbstractTensorHandle* const dim,
                  AbstractTensorHandle** output, const char* name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSarray_opsDTcc mht_4(mht_4_v, 338, "", "./tensorflow/c/experimental/ops/array_ops.cc", "ExpandDims");

  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("ExpandDims", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(input));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(dim));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(output, 1), &num_retvals);
}

// Op: OnesLike()
// Summary: Returns a tensor of ones with the same shape and type as x.
//
// Description:
Status OnesLike(AbstractContext* ctx, AbstractTensorHandle* const x,
                AbstractTensorHandle** y, const char* name) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSarray_opsDTcc mht_5(mht_5_v, 357, "", "./tensorflow/c/experimental/ops/array_ops.cc", "OnesLike");

  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("OnesLike", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(y, 1), &num_retvals);
}

}  // namespace ops
}  // namespace tensorflow
