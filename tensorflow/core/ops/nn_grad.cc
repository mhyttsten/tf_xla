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
class MHTracer_DTPStensorflowPScorePSopsPSnn_gradDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSnn_gradDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSnn_gradDTcc() {
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

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

typedef FunctionDefHelper FDH;

Status SoftmaxGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSnn_gradDTcc mht_0(mht_0_v, 194, "", "./tensorflow/core/ops/nn_grad.cc", "SoftmaxGrad");

  // clang-format off
  *g = FDH::Define(
      "SoftmaxGrad",
      // Arg defs
      {"x: T", "grad_softmax: T"},
      // Ret val defs
      {"grad_x: T"},
      // Attr defs
      {{"T: {float, double, bfloat16}"}},
      // Nodes
      // Based on _SoftmaxGrad in nn_grad.py.
      {
        {{"softmax"}, "Softmax", {"x"}, {{"T", "$T"}}},
        {{"n0"}, "Mul", {"grad_softmax", "softmax"}, {{"T", "$T"}}},
        FDH::Const<int32>("indices", {-1}),
        {{"n1"}, "Sum", {"n0", "indices"}, {{"keep_dims", true}, {"T", "$T"}}},
        {{"n2"}, "Sub", {"grad_softmax", "n1"}, {{"T", "$T"}}},
        {{"grad_x"}, "Mul", {"n2", "softmax"}, {{"T", "$T"}}}
      });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("Softmax", SoftmaxGrad);

Status LogSoftmaxGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPSnn_gradDTcc mht_1(mht_1_v, 222, "", "./tensorflow/core/ops/nn_grad.cc", "LogSoftmaxGrad");

  // clang-format off
  *g = FDH::Define(
      "LogSoftmaxGrad",
      // Arg defs
      {"x: T", "grad_logsoftmax: T"},
      // Ret val defs
      {"grad_x: T"},
      // Attr defs
      {{"T: {float, double}"}},
      // Nodes
      // Based on _LogSoftmaxGrad in nn_grad.py.
      {
        {{"softmax"}, "Softmax", {"x"}, {{"T", "$T"}}},
        FDH::Const<int32>("indices", {-1}),
        {{"n0"}, "Sum", {"grad_logsoftmax", "indices"},
         {{"keep_dims", true}, {"T", "$T"}}},
        {{"n1"}, "Mul", {"n0", "softmax"}, {{"T", "$T"}}},
        {{"grad_x"}, "Sub", {"grad_logsoftmax", "n1"}, {{"T", "$T"}}}
      });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("LogSoftmax", LogSoftmaxGrad);

Status ReluGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSopsPSnn_gradDTcc mht_2(mht_2_v, 250, "", "./tensorflow/core/ops/nn_grad.cc", "ReluGrad");

  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: T", "dy: T"},
      // Ret val defs
      {"dx: T"},
      // Attr defs
      {{"T: {float, double}"}},
      // Nodes
      {
        {{"dx"}, "ReluGrad", {"dy", "x"}, {{"T", "$T"}}}
      });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("Relu", ReluGrad);

Status Relu6Grad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSopsPSnn_gradDTcc mht_3(mht_3_v, 271, "", "./tensorflow/core/ops/nn_grad.cc", "Relu6Grad");

  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: T", "dy: T"},
      // Ret val defs
      {"dx: T"},
      // Attr defs
      {{"T: {float, double}"}},
      // Nodes
      {
        {{"dx"}, "Relu6Grad", {"dy", "x"}, {{"T", "$T"}}}
      });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("Relu6", Relu6Grad);

Status CrossEntropyGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSopsPSnn_gradDTcc mht_4(mht_4_v, 292, "", "./tensorflow/core/ops/nn_grad.cc", "CrossEntropyGrad");

  // clang-format off
  *g = FDH::Define(
    // Arg defs
    {"features: T", "labels: T", "dcost_dloss: T", "donotcare: T"},
    // Ret val defs
    {"dcost_dfeatures: T", "dcost_dlabels: T"},
    // Attr defs
    {{"T: {float, double}"}},
    // Nodes
    {
      // _, dloss_dfeatures = CrossEntropy(features, labels)
      {{"donotcare_loss", "dloss_dfeatures"}, "CrossEntropy",
       {"features", "labels"}, {{"T", "$T"}}},
      // dcost_dloss is of shape [batch_size].
      // dcost_dloss_mat is of shape [batch_size, 1].
      FDH::Const("neg1", -1),
      {{"dcost_dloss_mat"}, "ExpandDims", {"dcost_dloss", "neg1"},
       {{"T", "$T"}}},
      // chain rule: dcost/dfeatures = dcost/dloss * dloss/dfeatures
      {{"dcost_dfeatures"}, "Mul", {"dcost_dloss_mat", "dloss_dfeatures"},
       {{"T", "$T"}}},
      {{"dcost_dlabels"}, "ZerosLike", {"labels"}, {{"T", "$T"}}},
    });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("CrossEntropy", CrossEntropyGrad);

Status Conv2DGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSopsPSnn_gradDTcc mht_5(mht_5_v, 324, "", "./tensorflow/core/ops/nn_grad.cc", "Conv2DGrad");

  // clang-format off
  *g = FDH::Define(
    // Arg defs
    {"input: T", "filter: T", "grad: T"},
    // Ret val defs
    {"input_grad: T", "filter_grad: T"},
    // Attr defs
    {"T: {float, double}",
     "strides: list(int)",
     "use_cudnn_on_gpu: bool = true",
     GetPaddingAttrString(),
     GetConvnetDataFormatAttrString()},
    // Nodes
    {
      {{"i_shape"}, "Shape", {"input"}, {{"T", "$T"}}},
      {{"input_grad"}, "Conv2DBackpropInput", {"i_shape", "filter", "grad"},
       /*Attrs=*/{{"T", "$T"},
                  {"strides", "$strides"},
                  {"padding", "$padding"},
                  {"data_format", "$data_format"},
                  {"use_cudnn_on_gpu", "$use_cudnn_on_gpu"}}},

      {{"f_shape"}, "Shape", {"filter"}, {{"T", "$T"}}},
      {{"filter_grad"}, "Conv2DBackpropFilter", {"input", "f_shape", "grad"},
       /*Attrs=*/{{"T", "$T"},
                  {"strides", "$strides"},
                  {"padding", "$padding"},
                  {"data_format", "$data_format"},
                  {"use_cudnn_on_gpu", "$use_cudnn_on_gpu"}}},
    });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("Conv2D", Conv2DGrad);

Status MaxPoolGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSopsPSnn_gradDTcc mht_6(mht_6_v, 363, "", "./tensorflow/core/ops/nn_grad.cc", "MaxPoolGrad");

  // clang-format off
  *g = FDH::Define(
    // Arg defs
    {"input: T", "grad: T"},
    // Ret val defs
    {"output: T"},
    // Attr defs
    {"T: {float, half} = DT_FLOAT",
     "ksize: list(int) >= 4",
     "strides: list(int) >= 4",
     GetPaddingAttrString()},
    // Nodes
    {
      // Invoke MaxPool again to recompute the outputs (removed by CSE?).
      {{"maxpool"}, "MaxPool", {"input"},
       /*Attrs=*/{{"T", "$T"},
                  {"ksize", "$ksize"},
                  {"strides", "$strides"},
                  {"padding", "$padding"}}},
      {{"output"}, "MaxPoolGrad", {"input", "maxpool", "grad"},
       /*Attrs=*/{{"T", "$T"},
                  {"ksize", "$ksize"},
                  {"strides", "$strides"},
                  {"padding", "$padding"}}}
    });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("MaxPool", MaxPoolGrad);

Status AvgPoolGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSopsPSnn_gradDTcc mht_7(mht_7_v, 397, "", "./tensorflow/core/ops/nn_grad.cc", "AvgPoolGrad");

  // clang-format off
  *g = FDH::Define(
    // Arg defs
    {"input: T", "grad: T"},
    // Ret val defs
    {"output: T"},
    // Attr defs
    {"T: {float, half} = DT_FLOAT",
     "ksize: list(int) >= 4",
     "strides: list(int) >= 4",
     GetPaddingAttrString()},
    // Nodes
    {
      {{"i_shape"}, "Shape", {"input"}, {{"T", "$T"}}},
      {{"output"}, "AvgPoolGrad", {"i_shape", "grad"},
       /*Attrs=*/{{"T", "$T"},
                  {"ksize", "$ksize"},
                  {"strides", "$strides"},
                  {"padding", "$padding"}}}
    });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("AvgPool", AvgPoolGrad);

Status MaxPoolGradGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSopsPSnn_gradDTcc mht_8(mht_8_v, 426, "", "./tensorflow/core/ops/nn_grad.cc", "MaxPoolGradGrad");

  // clang-format off
  *g = FDH::Define(
    // Arg defs
    {"input: T", "grad: T"},
    // Ret val defs
    {"output: T"},
    // Attr defs
    {"T: {float, half} = DT_FLOAT",
     "ksize: list(int) >= 4",
     "strides: list(int) >= 4",
     GetPaddingAttrString()},
    // Nodes
    {
      // Invoke MaxPool again to recompute the outputs (removed by CSE?).
      {{"maxpool"}, "MaxPool", {"input"},
       /*Attrs=*/{{"T", "$T"},
                  {"ksize", "$ksize"},
                  {"strides", "$strides"},
                  {"padding", "$padding"}}},
      {{"output"}, "MaxPoolGradGrad", {"input", "maxpool", "grad"},
       /*Attrs=*/{{"T", "$T"},
                  {"ksize", "$ksize"},
                  {"strides", "$strides"},
                  {"padding", "$padding"}}}
    });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("MaxPoolGrad", MaxPoolGradGrad);

Status BiasAddGrad(const AttrSlice& attrs, FunctionDef* g) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSopsPSnn_gradDTcc mht_9(mht_9_v, 460, "", "./tensorflow/core/ops/nn_grad.cc", "BiasAddGrad");

  // clang-format off
  *g = FDH::Define(
    // Arg defs
    {"input: T", "bias: T", "grad: T"},
    // Ret val defs
    {"grad: T", "bias_grad: T"},
    // Attr defs
    {{"T: {float, double}"},
     GetConvnetDataFormatAttrString()},
    // Nodes
    {
      {{"bias_grad"}, "BiasAddGrad", {"grad"},
           /*Attrs=*/{{"T", "$T"},
                      {"data_format", "$data_format"}}}
    });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("BiasAdd", BiasAddGrad);

}  // end namespace tensorflow
