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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposer_factoryDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposer_factoryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposer_factoryDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer_factory.h"

#include "tensorflow/core/grappler/op_types.h"

namespace tensorflow {
namespace grappler {

std::shared_ptr<Transposer> TransposerFactory::GetTransposer(
    const NodeDef& node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposer_factoryDTcc mht_0(mht_0_v, 193, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer_factory.cc", "TransposerFactory::GetTransposer");

  // Check layout sensitive ops.
  if (IsDefaultLayoutSensitiveOp(node)) {
    return GetOrCreateIfNotFound<DefaultLayoutSensitiveOpTransposer>(
        "DefaultLayoutSensitiveOp");
  }
  if (IsAvgPoolGrad(node)) {
    return GetOrCreateIfNotFound<AvgPoolGradTransposer>("AvgPoolGrad");
  }
  if (IsBiasAddV2(node)) {
    return GetOrCreateIfNotFound<BiasAddTransposer>("BiasAdd");
  }
  if (IsBiasAddGrad(node)) {
    return GetOrCreateIfNotFound<BiasAddGradTransposer>("BiasAddGrad");
  }
  if (IsConv2DBackpropFilter(node) ||
      IsDepthwiseConv2dNativeBackpropFilter(node)) {
    return GetOrCreateIfNotFound<Conv2DBackpropFilterTransposer>(
        "Conv2DBackpropFilter");
  }
  if (IsConv2DBackpropInput(node) ||
      IsDepthwiseConv2dNativeBackpropInput(node)) {
    return GetOrCreateIfNotFound<Conv2DBackpropInputTransposer>(
        "Conv2DBackpropInput");
  }
  if (IsConv3D(node)) {
    return GetOrCreateIfNotFound<Conv3DTransposer>("Conv3D");
  }
  if (IsConv3DBackpropInputV2(node)) {
    return GetOrCreateIfNotFound<Conv3DBackpropInputTransposer>(
        "Conv3DBackpropInput");
  }
  if (IsConv3DBackpropFilterV2(node)) {
    return GetOrCreateIfNotFound<Conv3DBackpropFilterTransposer>(
        "Conv3DBackpropFilter");
  }
  if (IsFusedBatchNormEx(node)) {
    return GetOrCreateIfNotFound<FusedBatchNormExTransposer>(
        "FusedBatchNormEx");
  }
  if (IsFusedBatchNormGrad(node)) {
    return GetOrCreateIfNotFound<FusedBatchNormGradTransposer>(
        "FusedBatchNormGrad");
  }
  if (IsMaxPoolV2(node)) {
    return GetOrCreateIfNotFound<MaxPoolV2Transposer>("MaxPoolV2");
  }
  if (IsMaxPoolGrad(node) || IsMaxPoolGradGradV1(node)) {
    return GetOrCreateIfNotFound<MaxPoolGradTransposer>("MaxPoolGrad");
  }
  if (IsMaxPoolGradV2(node) || IsMaxPoolGradGradV2(node)) {
    return GetOrCreateIfNotFound<MaxPoolGradV2Transposer>("MaxPoolGradV2");
  }
  // Check layout agnostic ops.
  if (IsDefaultLayoutAgnosticOp(node)) {
    return GetOrCreateIfNotFound<DefaultLayoutAgnosticOpTransposer>(
        "DefaultLayoutAgnosticOp");
  }
  if (IsAddN(node)) {
    return GetOrCreateIfNotFound<AddNTransposer>("AddN");
  }
  if (IsBinaryOp(node)) {
    return GetOrCreateIfNotFound<BinaryOpTransposer>("BinaryOp");
  }
  if (IsConcat(node)) {
    return GetOrCreateIfNotFound<ConcatOpTransposer>("Concat");
  }
  if (IsFill(node)) {
    return GetOrCreateIfNotFound<FillOpTransposer>("Fill");
  }
  if (IsIdentityN(node)) {
    return GetOrCreateIfNotFound<IdentityNTransposer>("IdentityN");
  }
  if (IsMerge(node)) {
    return GetOrCreateIfNotFound<MergeTransposer>("Merge");
  }
  if (IsMirrorPad(node) || IsMirrorPadGrad(node) || IsPad(node)) {
    return GetOrCreateIfNotFound<PadTransposer>("Pad");
  }
  if (IsReduceOp(node)) {
    return GetOrCreateIfNotFound<ReduceTransposer>("ReduceOp");
  }
  if (IsReverseV2(node)) {
    return GetOrCreateIfNotFound<ReverseV2Transposer>("ReverseV2");
  }
  if (IsSelect(node)) {
    return GetOrCreateIfNotFound<SelectTransposer>("Select");
  }
  if (IsShape(node)) {
    return GetOrCreateIfNotFound<ShapeTransposer>("Shape");
  }
  if (IsShapeN(node)) {
    return GetOrCreateIfNotFound<ShapeNTransposer>("ShapeN");
  }
  if (IsSlice(node)) {
    return GetOrCreateIfNotFound<SliceTransposer>("Slice");
  }
  if (IsSplit(node)) {
    return GetOrCreateIfNotFound<SplitTransposer>("Split");
  }
  if (IsSplitV(node)) {
    return GetOrCreateIfNotFound<SplitVTransposer>("SplitV");
  }
  if (IsSqueeze(node)) {
    return GetOrCreateIfNotFound<SqueezeTransposer>("Squeeze");
  }
  if (IsStridedSlice(node)) {
    return GetOrCreateIfNotFound<StridedSliceTransposer>("StridedSlice");
  }
  if (IsSwitch(node)) {
    return GetOrCreateIfNotFound<SwitchTransposer>("Switch");
  }
  if (IsTernaryOp(node)) {
    return GetOrCreateIfNotFound<TernaryOpTransposer>("TernaryOp");
  }
  if (IsTile(node)) {
    return GetOrCreateIfNotFound<TileTransposer>("Tile");
  }
  if (IsUnaryGrad(node)) {
    return GetOrCreateIfNotFound<UnaryGradTransposer>("UnaryGrad");
  }
  return nullptr;
}

}  // namespace grappler
}  // namespace tensorflow
