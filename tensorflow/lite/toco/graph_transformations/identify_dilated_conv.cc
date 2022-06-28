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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_dilated_convDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_dilated_convDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_dilated_convDTcc() {
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
#include <string>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

// A dilated convolution can be emulated with a regular convolution by chaining
// SpaceToBatch and BatchToSpace ops before and after it:
//
//     SpaceToBatchND -> Conv2D -> BatchToSpaceND
//
// This method was common before Conv2D fully supported dilated convolution in
// TensorFlow. This transformation detects this "emulation", and replaces it
// with a true dilated convolution, eliminating the SpaceToBatch and
// BatchtoSpace ops.
//
// Detecting this alone would be relatively easy. However, in practice some
// extra ops are used, so we detect the following patterns:
//
//
//   SpaceToBatchND -> Expand -> Conv2D -> Squeeze -> BatchToSpaceND -> BiasAdd
//
//   Pad -> SpaceToBatchND -> Expand -> Conv2D -> Squeeze -> BatchToSpaceND ->
//   BiasAdd
//
//   SpaceToBatchND -> Expand -> Conv2D -> Squeeze -> Pad -> BatchToSpaceND ->
//   BiasAdd
//
//   SpaceToBatchND -> Expand -> Conv2D -> Squeeze -> BiasAdd -> BatchToSpaceND
//
//   SpaceToBatchND -> Conv2D -> Pad -> BatchToSpaceND -> BiasAdd
//
//   SpaceToBatchND -> Conv2D -> BatchToSpaceND -> BiasAdd
//
//
// The Expand/Squeeze combination is used to adapt a 3D array (such as in
// WaveNet) to the 4D arrays that Conv2D requires. Padding and BiasAdd are
// thrown in just for the extra headache. Padding adapts non-conforming input
// sizes, and can be discarded. The bias is necessary, so is kept.

template <typename T>
bool ResolveDilatedConv(Model* model, Operator* conv_base_op, Operator* stb_op,
                        Operator* post_stb_op, bool has_expand_op,
                        int dilation_factor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_dilated_convDTcc mht_0(mht_0_v, 231, "", "./tensorflow/lite/toco/graph_transformations/identify_dilated_conv.cc", "ResolveDilatedConv");

  auto* conv_op = static_cast<T*>(conv_base_op);
  if (conv_op->inputs.size() != 2) {
    // The conv op must only have weights, no bias.
    return false;
  }
  CHECK_EQ(conv_op->outputs.size(), 1);

  // Squeeze Op
  auto* post_conv_op = GetOpWithInput(*model, conv_op->outputs[0]);
  if (!post_conv_op) {
    return false;
  }
  if (has_expand_op) {
    if (post_conv_op->type != OperatorType::kSqueeze) {
      // If an expand op was used, the post-conv op must be a squeeze op
      return false;
    }
    CHECK_EQ(post_conv_op->inputs.size(), 1);
    CHECK_EQ(post_conv_op->outputs.size(), 1);
  }

  // Pad Op
  const auto* pad_op = has_expand_op
                           ? GetOpWithInput(*model, post_conv_op->outputs[0])
                           : GetOpWithInput(*model, conv_op->outputs[0]);
  bool has_pad_op = false;
  if (pad_op && pad_op->type == OperatorType::kPad) {
    has_pad_op = true;
    CHECK_EQ(pad_op->inputs.size(), 2);
    CHECK_EQ(pad_op->outputs.size(), 1);
  }
  // TODO(mjmatthews): Perform validity checking on padding dimensions.

  // Pre-BatchToSpace Bias Op
  auto* next_op = has_pad_op
                      ? GetOpWithInput(*model, pad_op->outputs[0])
                      : has_expand_op
                            ? GetOpWithInput(*model, post_conv_op->outputs[0])
                            : GetOpWithInput(*model, conv_op->outputs[0]);
  bool has_bias_before_bts = false;
  if (next_op->type == OperatorType::kAdd) {
    has_bias_before_bts = true;
  }
  auto final_op = GetOpWithInput(*model, next_op->outputs[0]);

  // BatchToSpace Op
  const auto* bts_op = has_bias_before_bts ? final_op : next_op;
  if (bts_op->type != OperatorType::kBatchToSpaceND) {
    return false;
  }
  CHECK_EQ(bts_op->inputs.size(), 3);
  CHECK_EQ(bts_op->outputs.size(), 1);

  // Post-BatchToSpace Bias Op
  Operator* bias_add_op = !has_bias_before_bts ? final_op : next_op;
  if (bias_add_op->type != OperatorType::kAdd) {
    // Bias op is required before or after BatchToSpace
    return false;
  }
  CHECK_EQ(bias_add_op->inputs.size(), 2);
  CHECK_EQ(bias_add_op->outputs.size(), 1);

  //   If still Pad Op is not present, there might be possiblity it is added
  //   before STB Op like below Pad -> SpaceToBatchND -> Expand -> Conv2D ->
  //   Squeeze -> BatchToSpaceND -> BiasAdd So eliminate this Pad Op as well
  if (!has_pad_op) {
    auto* pre_stb_pad_op = GetOpWithOutput(*model, stb_op->inputs[0]);
    // If it is a Pad Op then just rewire the Input of Pad Op with Input of STB
    if (pre_stb_pad_op && pre_stb_pad_op->type == OperatorType::kPad) {
      stb_op->inputs[0] = pre_stb_pad_op->inputs[0];
      has_pad_op = true;
      pad_op = pre_stb_pad_op;
    }
  }

  // 2. RE-WIRE OPERATORS
  // ***************************************************************************
  // Re-use the existing Conv2D op.
  conv_op->dilation_width_factor = dilation_factor;
  conv_op->dilation_height_factor = dilation_factor;
  conv_op->padding.type = PaddingType::kSame;

  // Rewire the ops to bypass SpaceToBatch, BatchToSpace, and Pad.
  bias_add_op->outputs[0] = final_op->outputs[0];
  if (has_expand_op) {
    bias_add_op->inputs[0] = post_conv_op->outputs[0];
    post_conv_op->inputs[0] = conv_op->outputs[0];
    conv_op->inputs[0] = post_stb_op->outputs[0];
    post_stb_op->inputs[0] = stb_op->inputs[0];
  } else {
    bias_add_op->inputs[0] = conv_op->outputs[0];
    conv_op->inputs[0] = stb_op->inputs[0];
  }
  // TODO(mjmatthews): Connect bias directly into the Conv2D?

  // 3. DELETE LEFTOVER OPERATORS
  // ***************************************************************************
  DeleteOpAndArrays(model, bts_op);
  DeleteOpAndArrays(model, stb_op);
  if (has_pad_op) {
    DeleteOpAndArrays(model, pad_op);
  }

  return true;
}

::tensorflow::Status IdentifyDilatedConv::Run(Model* model,
                                              std::size_t op_index,
                                              bool* modified) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_dilated_convDTcc mht_1(mht_1_v, 343, "", "./tensorflow/lite/toco/graph_transformations/identify_dilated_conv.cc", "IdentifyDilatedConv::Run");

  *modified = false;
  const auto it = model->operators.begin() + op_index;
  auto* stb_op = it->get();

  // 1. IDENTIFY OPERATORS
  // ***************************************************************************
  // SpaceToBatch Op.
  if (stb_op->type != OperatorType::kSpaceToBatchND) {
    return ::tensorflow::Status::OK();
  }
  if (stb_op->inputs.size() != 3) {
    return ::tensorflow::Status::OK();
  }
  CHECK_EQ(stb_op->outputs.size(), 1);
  // Extract the dilation factor from Input[1] of SpaceToBatch
  // TODO(mjmatthews): Support 2D dilation factors.
  const auto& block_shape_array = model->GetArray(stb_op->inputs[1]);
  if (!block_shape_array.buffer) {
    return ::tensorflow::Status::OK();
  }
  CHECK_EQ(block_shape_array.shape().dimensions_count(), 1);
  int dilation_factor =
      block_shape_array.Array::GetBuffer<ArrayDataType::kInt32>().data[0];

  // Expand Op
  auto* post_stb_op = GetOpWithInput(*model, stb_op->outputs[0]);
  if (!post_stb_op) {
    return ::tensorflow::Status::OK();
  }
  bool has_expand_op = false;
  if (post_stb_op->type == OperatorType::kExpandDims) {
    has_expand_op = true;
    CHECK_EQ(post_stb_op->inputs.size(), 2);
    CHECK_EQ(post_stb_op->outputs.size(), 1);
  }

  // Conv Op
  const std::string& input_of_conv_op =
      has_expand_op ? post_stb_op->outputs[0] : stb_op->outputs[0];
  auto* conv_base_op = GetOpWithInput(*model, input_of_conv_op);
  bool changed = false;
  if (conv_base_op->type == OperatorType::kConv) {
    changed = ResolveDilatedConv<ConvOperator>(model, conv_base_op, stb_op,
                                               post_stb_op, has_expand_op,
                                               dilation_factor);
    if (changed) {
      LOG(INFO) << "Replaced sub-network with Dilated Conv2D op outputting \""
                << conv_base_op->outputs[0] << "\".";
    }
  } else if (identify_depthwise_conv_ &&
             conv_base_op->type == OperatorType::kDepthwiseConv) {
    changed = ResolveDilatedConv<DepthwiseConvOperator>(
        model, conv_base_op, stb_op, post_stb_op, has_expand_op,
        dilation_factor);
    if (changed) {
      LOG(INFO)
          << "Replaced sub-network with Dilated DepthwiseConv2D op outputting "
          << "\"" << conv_base_op->outputs[0] << "\".";
    }
  }

  *modified = changed;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
