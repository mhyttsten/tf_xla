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
class MHTracer_DTPStensorflowPSccPSgradientsPSimage_gradDTcc {
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
   MHTracer_DTPStensorflowPSccPSgradientsPSimage_gradDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSgradientsPSimage_gradDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/image_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"

namespace tensorflow {
namespace ops {
namespace {

REGISTER_NO_GRADIENT_OP("NonMaxSuppression");
REGISTER_NO_GRADIENT_OP("NonMaxSuppressionV2");
REGISTER_NO_GRADIENT_OP("NonMaxSuppressionV3");
REGISTER_NO_GRADIENT_OP("NonMaxSuppressionV4");
REGISTER_NO_GRADIENT_OP("NonMaxSuppressionV5");

Status ResizeNearestNeighborGradHelper(const Scope& scope, const Operation& op,
                                       const std::vector<Output>& grad_inputs,
                                       std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSimage_gradDTcc mht_0(mht_0_v, 203, "", "./tensorflow/cc/gradients/image_grad.cc", "ResizeNearestNeighborGradHelper");

  bool align_corners;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "align_corners", &align_corners));
  bool half_pixel_centers;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "half_pixel_centers",
                                 &half_pixel_centers));
  // The internal gradient implementation needs the shape of the input image.
  // x_shape = shape(x)[1:3]
  //         = slice(shape(x), {1}, {3 - 1})
  auto x_shape = Slice(scope, Shape(scope, op.input(0)), {1}, {2});
  grad_outputs->push_back(internal::ResizeNearestNeighborGrad(
      scope, grad_inputs[0], x_shape,
      internal::ResizeNearestNeighborGrad::AlignCorners(align_corners)
          .HalfPixelCenters(half_pixel_centers)));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("ResizeNearestNeighbor", ResizeNearestNeighborGradHelper);

Status ResizeBilinearGradHelper(const Scope& scope, const Operation& op,
                                const std::vector<Output>& grad_inputs,
                                std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSimage_gradDTcc mht_1(mht_1_v, 228, "", "./tensorflow/cc/gradients/image_grad.cc", "ResizeBilinearGradHelper");

  bool align_corners;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "align_corners", &align_corners));
  bool half_pixel_centers;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "half_pixel_centers",
                                 &half_pixel_centers));
  grad_outputs->push_back(internal::ResizeBilinearGrad(
      scope, grad_inputs[0], op.input(0),
      internal::ResizeBilinearGrad::AlignCorners(align_corners)
          .HalfPixelCenters(half_pixel_centers)));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("ResizeBilinear", ResizeBilinearGradHelper);

Status ResizeBicubicGradHelper(const Scope& scope, const Operation& op,
                               const std::vector<Output>& grad_inputs,
                               std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSimage_gradDTcc mht_2(mht_2_v, 249, "", "./tensorflow/cc/gradients/image_grad.cc", "ResizeBicubicGradHelper");

  bool align_corners;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "align_corners", &align_corners));
  bool half_pixel_centers;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "half_pixel_centers",
                                 &half_pixel_centers));

  grad_outputs->push_back(internal::ResizeBicubicGrad(
      scope, grad_inputs[0], op.input(0),
      internal::ResizeBicubicGrad::AlignCorners(align_corners)
          .HalfPixelCenters(half_pixel_centers)));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("ResizeBicubic", ResizeBicubicGradHelper);

Status ScaleAndTranslateGradHelper(const Scope& scope, const Operation& op,
                                   const std::vector<Output>& grad_inputs,
                                   std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSimage_gradDTcc mht_3(mht_3_v, 271, "", "./tensorflow/cc/gradients/image_grad.cc", "ScaleAndTranslateGradHelper");

  string kernel_type;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "kernel_type", &kernel_type));
  bool antialias;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "antialias", &antialias));
  grad_outputs->push_back(internal::ScaleAndTranslateGrad(
      scope, grad_inputs[0], op.input(0), op.input(2), op.input(3),
      internal::ScaleAndTranslateGrad::KernelType(kernel_type)
          .Antialias(antialias)));

  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}

REGISTER_GRADIENT_OP("ScaleAndTranslate", ScaleAndTranslateGradHelper);

Status CropAndResizeGradHelper(const Scope& scope, const Operation& op,
                               const std::vector<Output>& grad_inputs,
                               std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSimage_gradDTcc mht_4(mht_4_v, 295, "", "./tensorflow/cc/gradients/image_grad.cc", "CropAndResizeGradHelper");

  DataType input_type;
  string method;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "method", &method));
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "T", &input_type));
  auto image_shape = Shape(scope, op.input(0));
  grad_outputs->push_back(CropAndResizeGradImage(
      scope, grad_inputs[0], op.input(1), op.input(2), image_shape, input_type,
      CropAndResizeGradImage::Method(method)));
  grad_outputs->push_back(CropAndResizeGradBoxes(
      scope, grad_inputs[0], op.input(0), op.input(1), op.input(2)));
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}

REGISTER_GRADIENT_OP("CropAndResize", CropAndResizeGradHelper);
}  // anonymous namespace
}  // namespace ops
}  // namespace tensorflow
