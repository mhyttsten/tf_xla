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
class MHTracer_DTPStensorflowPScPSexperimentalPSopsPSnn_opsDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSnn_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSopsPSnn_opsDTcc() {
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

#include "tensorflow/c/experimental/ops/nn_ops.h"

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/tracing_utils.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"

using tensorflow::tracing::MaybeSetOpName;

namespace tensorflow {
namespace ops {

// Op: SparseSoftmaxCrossEntropyWithLogits()
// Summary: Computes softmax cross entropy cost and gradients to backpropagate.
//
// Description:
//   Unlike `SoftmaxCrossEntropyWithLogits`, this operation does not accept
//   a matrix of label probabilities, but rather a single label per row
//   of features.  This label is considered to have probability 1.0 for the
//   given row.
//
//   Inputs are the logits, not probabilities.
Status SparseSoftmaxCrossEntropyWithLogits(AbstractContext* ctx,
                                           AbstractTensorHandle* const features,
                                           AbstractTensorHandle* const labels,
                                           AbstractTensorHandle** loss,
                                           AbstractTensorHandle** backprop,
                                           const char* name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSnn_opsDTcc mht_0(mht_0_v, 216, "", "./tensorflow/c/experimental/ops/nn_ops.cc", "SparseSoftmaxCrossEntropyWithLogits");

  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("SparseSoftmaxCrossEntropyWithLogits",
                                   /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(features));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(labels));
  int num_retvals = 2;
  AbstractTensorHandle* temp_outputs[2];
  Status status = op_ptr->Execute(temp_outputs, &num_retvals);
  *loss = temp_outputs[0];
  *backprop = temp_outputs[1];
  return status;
}

// Op: ReluGrad()
// Summary: Computes rectified linear gradients for a Relu operation.
//
// Description:
Status ReluGrad(AbstractContext* ctx, AbstractTensorHandle* const gradients,
                AbstractTensorHandle* const features,
                AbstractTensorHandle** backprops, const char* name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSnn_opsDTcc mht_1(mht_1_v, 241, "", "./tensorflow/c/experimental/ops/nn_ops.cc", "ReluGrad");

  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("ReluGrad", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(gradients));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(features));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(backprops, 1), &num_retvals);
}

// Op: Relu()
// Summary: Computes rectified linear: `max(features, 0)`.
//
// Description:
//   See: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
//   Example usage:
//   >>> tf.nn.relu([-2., 0., 3.]).numpy()
//   array([0., 0., 3.], dtype=float32)
Status Relu(AbstractContext* ctx, AbstractTensorHandle* const features,
            AbstractTensorHandle** activations, const char* name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSnn_opsDTcc mht_2(mht_2_v, 264, "", "./tensorflow/c/experimental/ops/nn_ops.cc", "Relu");

  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Relu", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(features));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(activations, 1), &num_retvals);
}

// Op: BiasAdd()
// Summary: Adds `bias` to `value`.
//
// Description:
//   This is a special case of `tf.add` where `bias` is restricted to be 1-D.
//   Broadcasting is supported, so `value` may have any number of dimensions.
Status BiasAdd(AbstractContext* ctx, AbstractTensorHandle* const value,
               AbstractTensorHandle* const bias, AbstractTensorHandle** output,
               const char* data_format, const char* name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("data_format: \"" + (data_format == nullptr ? std::string("nullptr") : std::string((char*)data_format)) + "\"");
   mht_3_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSnn_opsDTcc mht_3(mht_3_v, 286, "", "./tensorflow/c/experimental/ops/nn_ops.cc", "BiasAdd");

  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("BiasAdd", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(value));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(bias));
  TF_RETURN_IF_ERROR(
      op_ptr->SetAttrString("data_format", data_format, strlen(data_format)));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(output, 1), &num_retvals);
}

// Op: BiasAddGrad()
// Summary: The backward operation for "BiasAdd" on the "bias" tensor.
//
// Description:
//   It accumulates all the values from out_backprop into the feature dimension.
//   For NHWC data format, the feature dimension is the last. For NCHW data
//   format, the feature dimension is the third-to-last.
Status BiasAddGrad(AbstractContext* ctx,
                   AbstractTensorHandle* const out_backprop,
                   AbstractTensorHandle** output, const char* data_format,
                   const char* name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("data_format: \"" + (data_format == nullptr ? std::string("nullptr") : std::string((char*)data_format)) + "\"");
   mht_4_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSnn_opsDTcc mht_4(mht_4_v, 313, "", "./tensorflow/c/experimental/ops/nn_ops.cc", "BiasAddGrad");

  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("BiasAddGrad", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(out_backprop));
  TF_RETURN_IF_ERROR(
      op_ptr->SetAttrString("data_format", data_format, strlen(data_format)));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(output, 1), &num_retvals);
}

}  // namespace ops
}  // namespace tensorflow
