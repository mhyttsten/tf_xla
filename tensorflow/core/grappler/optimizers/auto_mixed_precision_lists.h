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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_AUTO_MIXED_PRECISION_LISTS_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_AUTO_MIXED_PRECISION_LISTS_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSauto_mixed_precision_listsDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSauto_mixed_precision_listsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSauto_mixed_precision_listsDTh() {
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


#include <string>

#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace grappler {

// Represents the four lists of ops: the allow list, infer list, deny list, and
// clear list. These lists determine which ops are converted to fp16/bf16
// (referred to as 'f16' for short) and which ops stay as fp32.
class AutoMixedPrecisionLists {
 public:
  virtual ~AutoMixedPrecisionLists() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSauto_mixed_precision_listsDTh mht_0(mht_0_v, 202, "", "./tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h", "~AutoMixedPrecisionLists");
}

  // Returns the set of ops that are considered numerically-safe (for execution
  // in f16), performance-critical, and can run in f16. These ops are always
  // converted to f16.
  virtual gtl::FlatSet<string> AllowList() = 0;
  // Returns the set of ops that can run in f16 and are considered numerically-
  // safe (for execution in f16), but which may be made unsafe by an upstream
  // denylist op.
  virtual gtl::FlatSet<string> InferList() = 0;
  // Returns the set of ops that are considered numerically-dangerous (i.e.,
  // unsafe for execution in f16) and whose effects may also be observed in
  // downstream nodes (e.g. for f16, in Exp -> Add, the Add is unsafe due to
  // the Exp).
  virtual gtl::FlatSet<string> DenyList() = 0;
  // Returns the set of ops that do not have numerically-significant effects
  // (i.e., they are always considered safe for execution in f16 precision), and
  // can run in f16.
  virtual gtl::FlatSet<string> ClearList() = 0;

 protected:
  // Adds or removes ops from list if certain environmental variables are set.
  static void UpdateList(const string& list_name, gtl::FlatSet<string>* list) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("list_name: \"" + list_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSauto_mixed_precision_listsDTh mht_1(mht_1_v, 228, "", "./tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h", "UpdateList");

    CHECK(list_name == "ALLOWLIST" || list_name == "INFERLIST" ||  // Crash OK.
          list_name == "DENYLIST" || list_name == "CLEARLIST" ||
          // TODO(reedwm): for bkwds compat; remove when no longer necessary:
          list_name == "WHITELIST" || list_name == "GRAYLIST" ||
          list_name == "BLACKLIST");
    string add_env_var =
        "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_" + list_name + "_ADD";
    string remove_env_var =
        "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_" + list_name + "_REMOVE";
    string to_add, to_remove;
    TF_CHECK_OK(ReadStringFromEnvVar(add_env_var, "", &to_add));
    TF_CHECK_OK(ReadStringFromEnvVar(remove_env_var, "", &to_remove));
    for (const auto& x : str_util::Split(to_add, ",")) {
      list->insert(x);
    }
    for (const auto& x : str_util::Split(to_remove, ",")) {
      list->erase(x);
    }
  }

  // Subclasses should include these on the ClearList.
  static void AddTensorListOps(gtl::FlatSet<string>* list) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSauto_mixed_precision_listsDTh mht_2(mht_2_v, 253, "", "./tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h", "AddTensorListOps");

    // Note: if a data structure op (such as TensorListPopBack) is added here,
    // IsTensorListReaderOp or IsTensorListWriterOp may need to be modified
    // LINT.IfChange
    constexpr const char* tensor_list_ops[] = {
        "TensorListConcat",     "TensorListConcatLists",
        "TensorListConcatV2",   "TensorListGather",
        "TensorListGetItem",    "TensorListPopBack",
        "TensorListPushBack",   "TensorListPushBackBatch",
        "TensorListFromTensor", "TensorListScatter",
        "TensorListScatterV2",  "TensorListScatterIntoExistingList",
        "TensorListSetItem",    "TensorListSplit",
        "TensorListStack"};
    // LINT.ThenChange(//tensorflow/core/grappler/optimizers/auto_mixed_precision.cc)
    for (auto op : tensor_list_ops) {
      list->insert(op);
    }
  }
};

class AutoMixedPrecisionListsCuda : public AutoMixedPrecisionLists {
 private:
  static bool IsPseudoFastMath() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSauto_mixed_precision_listsDTh mht_3(mht_3_v, 278, "", "./tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h", "IsPseudoFastMath");

    string optimization_level;
    TF_CHECK_OK(
        ReadStringFromEnvVar("TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_LEVEL", "",
                             &optimization_level));
    optimization_level = str_util::Uppercase(optimization_level);
    return optimization_level == "TENSOR_CORES_ONLY";
  }

 public:
  AutoMixedPrecisionListsCuda(int cuda_version, int cudnn_version)
      : cuda_version_(cuda_version), cudnn_version_(cudnn_version) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSauto_mixed_precision_listsDTh mht_4(mht_4_v, 292, "", "./tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h", "AutoMixedPrecisionListsCuda");
}

  gtl::FlatSet<string> AllowList() override {
    auto list = gtl::FlatSet<string>{
        "BlockLSTM",
        "BlockLSTMV2",
        "BlockLSTMGrad",
        "BlockLSTMGradV2",
        "Conv2D",
        "Conv2DBackpropFilter",
        "Conv2DBackpropInput",
        "CudnnRNN",
        "CudnnRNNBackprop",
        "CudnnRNNBackpropV2",
        "CudnnRNNBackpropV3",
        "CudnnRNNV2",
        "CudnnRNNV3",
        "Einsum",
        "FusedConv2DBiasActivation",
        "FusedSparseConvGpuV2",
        "GRUBlockCell",
        "GRUBlockCellGrad",
        "LSTMBlockCell",
        "LSTMBlockCellGrad",
        "MatMul",
    };
#if TENSORFLOW_USE_ROCM
    if (true) {
#else
    if (cuda_version_ >= 9010) {
      // Fp16 BatchMatMul is slow before CUDA 9.1.
#endif
      list.insert("BatchMatMul");
      list.insert("BatchMatMulV2");
    }
    if (cudnn_version_ >= 7602) {
      // Fp16 3D conv is slow before CUDNN 7.6.2.
      list.insert("Conv3D");
      list.insert("Conv3DBackpropFilter");
      list.insert("Conv3DBackpropFilterV2");
      list.insert("Conv3DBackpropInput");
      list.insert("Conv3DBackpropInputV2");
    }
    if (cudnn_version_ >= 8000) {
      list.insert("DepthwiseConv2dNative");
      list.insert("DepthwiseConv2dNativeBackpropFilter");
      list.insert("DepthwiseConv2dNativeBackpropInput");
    }
    UpdateList("ALLOWLIST", &list);
    // For backwards compatibility, keeping the original env variable here.
    // TODO(reedwm): This should be removed if we don't have active users.
    UpdateList("WHITELIST", &list);

    return list;
  }

  gtl::FlatSet<string> InferList() override {
    if (IsPseudoFastMath()) {
      return gtl::FlatSet<string>{};
    }

    auto list = gtl::FlatSet<string>{
        "Add",
        "AddN",
        "AddV2",
        "AvgPool",
        "AvgPool3D",
        "AvgPool3DGrad",
        "AvgPoolGrad",
        "BiasAdd",
        "BiasAddGrad",
        "BiasAddV1",
        "Elu",
        "EluGrad",
        "Erf",
        "Erfc",
        "FloorDiv",
        "FusedBatchNormV2",
        "FusedBatchNormGradV2",
        "FusedBatchNormV3",
        "FusedBatchNormGradV3",
        "_FusedBatchNormEx",
        "Inv",
        "LeakyRelu",
        "LeakyReluGrad",
        "Log",
        "Log1p",
        "LogSoftmax",
        "Mul",
        "Prod",
        "RealDiv",
        "Reciprocal",
        "Selu",
        "SeluGrad",
        "Sigmoid",
        "SigmoidGrad",
        "Softmax",
        "Softplus",
        "SoftplusGrad",
        "Softsign",
        "SoftsignGrad",
        "Sqrt",
        "Sub",
        "Tanh",
        "TanhGrad",
    };
    UpdateList("INFERLIST", &list);
    // For backwards compatibility, keeping the original env variable here.
    // TODO(reedwm): This should be removed if we don't have active users.
    UpdateList("GRAYLIST", &list);
    return list;
  }

  gtl::FlatSet<string> DenyList() override {
    if (IsPseudoFastMath()) {
      return gtl::FlatSet<string>{};
    }

    auto list = gtl::FlatSet<string>{
        "Exp",
        "Expm1",
        "L2Loss",
        "Mean",
        "Pow",
        "SaveV2",
        "SoftmaxCrossEntropyWithLogits",
        "SparseSoftmaxCrossEntropyWithLogits",
        "Sum",
    };
    UpdateList("DENYLIST", &list);
    // For backwards compatibility, keeping the original env variable here.
    // TODO(reedwm): This should be removed if we don't have active users.
    UpdateList("BLACKLIST", &list);
    return list;
  }

  gtl::FlatSet<string> ClearList() override {
    if (IsPseudoFastMath()) {
      return gtl::FlatSet<string>{};
    }

    auto list = gtl::FlatSet<string>{
        "Abs",
        "ArgMax",
        "ArgMin",
        "BatchToSpace",
        "BatchToSpaceND",
        "BroadcastTo",
        "Ceil",
        "CheckNumerics",
        "ClipByValue",
        "Concat",
        "ConcatV2",
        "DepthToSpace",
        "DynamicPartition",
        "DynamicStitch",
        "Enter",
        "EnsureShape",
        "Equal",
        "Exit",
        "ExpandDims",
        "Fill",
        "Floor",
        "Gather",
        "GatherNd",
        "GatherV2",
        "Greater",
        "GreaterEqual",
        "Identity",
        "IdentityN",
        "IsFinite",
        "IsInf",
        "IsNan",
        "Less",
        "LessEqual",
        "Max",
        "MaxPool",
        "MaxPool3D",
        "MaxPool3DGrad",
        "MaxPool3DGradGrad",
        "MaxPoolGrad",
        "MaxPoolGradGrad",
        "MaxPoolGradGradV2",
        "MaxPoolGradV2",
        "MaxPoolV2",
        "Maximum",
        "Merge",
        "Min",
        "Minimum",
        "MirrorPad",
        "MirrorPadGrad",
        "Neg",
        "NextIteration",
        "NotEqual",
        "OneHot",
        "OnesLike",
        "Pack",
        "Pad",
        "PadV2",
        "PreventGradient",
        "Rank",
        "Relu",
        "Relu6",
        "Relu6Grad",
        "ReluGrad",
        "Reshape",
        "ResizeNearestNeighbor",
        "ResizeNearestNeighborGrad",
        "Reverse",
        "ReverseSequence",
        "ReverseV2",
        "Round",
        "Select",
        "SelectV2",
        "Shape",
        "ShapeN",
        "Sign",
        "Size",
        "Slice",
        "Snapshot",
        "SpaceToBatch",
        "SpaceToBatchND",
        "SpaceToDepth",
        "Split",
        "SplitV",
        "Squeeze",
        "StopGradient",
        "StridedSlice",
        "StridedSliceGrad",
        "Switch",
        "Tile",
        "TopK",
        "TopKV2",
        "Transpose",
        "Unpack",
        "Where",
        "ZerosLike",
    };
    AddTensorListOps(&list);
    UpdateList("CLEARLIST", &list);
    return list;
  }

 private:
  int cuda_version_;
  int cudnn_version_;
};

class AutoMixedPrecisionListsMkl : public AutoMixedPrecisionLists {
 public:
  AutoMixedPrecisionListsMkl() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSauto_mixed_precision_listsDTh mht_5(mht_5_v, 545, "", "./tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h", "AutoMixedPrecisionListsMkl");
}

  // Only ops which are supported by MKL in bfloat16 should be added to the
  // allow list, infer list, or clear list.
  gtl::FlatSet<string> AllowList() override {
    auto list = gtl::FlatSet<string>{"Conv2D",
                                     "Conv2DBackpropFilter",
                                     "Conv2DBackpropInput",
                                     "Conv3D",
                                     "Conv3DBackpropFilterV2",
                                     "Conv3DBackpropInputV2",
                                     "DepthwiseConv2dNative",
                                     "DepthwiseConv2dNativeBackpropFilter",
                                     "DepthwiseConv2dNativeBackpropInput",
                                     "MatMul",
                                     "BatchMatMul",
                                     "BatchMatMulV2"};

    UpdateList("ALLOWLIST", &list);
    // For backwards compatibility, keeping the original env variable here.
    // TODO(reedwm): This should be removed if we don't have active users.
    UpdateList("WHITELIST", &list);
    return list;
  }

  gtl::FlatSet<string> InferList() override {
    auto list = gtl::FlatSet<string>{"Add",
                                     "AddN",
                                     "AddV2",
                                     "AvgPool",
                                     "AvgPool3D",
                                     "AvgPool3DGrad",
                                     "AvgPoolGrad",
                                     "BiasAdd",
                                     "BiasAddGrad",
                                     "BiasAddV1",
                                     "FusedBatchNormV2",
                                     "FusedBatchNormGradV2",
                                     "FusedBatchNormV3",
                                     "FusedBatchNormGradV3",
                                     "LeakyRelu",
                                     "LeakyReluGrad",
                                     "Mul",
                                     "Sub",
                                     "Elu",
                                     "EluGrad",
                                     "FloorDiv",
                                     "_FusedBatchNormEx",
                                     "Log",
                                     "Log1p",
                                     "LogSoftmax",
                                     "Prod",
                                     "RealDiv",
                                     "Reciprocal",
                                     "Selu",
                                     "SeluGrad",
                                     "Sigmoid",
                                     "SigmoidGrad",
                                     "Softmax",
                                     "Softplus",
                                     "SoftplusGrad",
                                     "Softsign",
                                     "SoftsignGrad",
                                     "Sqrt",
                                     "Tanh",
                                     "TanhGrad"};
    UpdateList("INFERLIST", &list);
    // For backwards compatibility, keeping the original env variable here.
    // TODO(reedwm): This should be removed if we don't have active users.
    UpdateList("GRAYLIST", &list);
    return list;
  }

  gtl::FlatSet<string> DenyList() override {
    auto list = gtl::FlatSet<string>{
        "Exp",
        "Expm1",
        "L2Loss",
        "Mean",
        "Pow",
        "SaveV2",
        "SoftmaxCrossEntropyWithLogits",
        "SparseSoftmaxCrossEntropyWithLogits",
        "Sum",
    };
    UpdateList("DENYLIST", &list);
    // For backwards compatibility, keeping the original env variable here.
    // TODO(reedwm): This should be removed if we don't have active users.
    UpdateList("BLACKLIST", &list);
    return list;
  }

  gtl::FlatSet<string> ClearList() override {
    auto list = gtl::FlatSet<string>{
        "Abs",
        "ArgMax",
        "ArgMin",
        "BatchToSpace",
        "BatchToSpaceND",
        "BroadcastTo",
        "Ceil",
        "CheckNumerics",
        "ClipByValue",
        "Concat",
        "ConcatV2",
        "DepthToSpace",
        "DynamicPartition",
        "DynamicStitch",
        "EnsureShape",
        "Enter",
        "Equal",
        "Exit",
        "ExpandDims",
        "Fill",
        "Floor",
        "Gather",
        "GatherNd",
        "GatherV2",
        "Greater",
        "GreaterEqual",
        "Identity",
        "IsFinite",
        "IsInf",
        "IsNan",
        "Less",
        "LessEqual",
        "Max",
        "Maximum",
        "MaxPool",
        "MaxPool3D",
        "MaxPool3DGrad",
        "MaxPoolGrad",
        "MaxPoolGradGrad",
        "MaxPoolGradGradV2",
        "MaxPoolGradV2",
        "MaxPoolV2",
        "Merge",
        "Min",
        "Minimum",
        "MirrorPad",
        "MirrorPadGrad",
        "Neg",
        "NextIteration",
        "NotEqual",
        "OnesLike",
        "Pack",
        "Pad",
        "PadV2",
        "PreventGradient",
        "Rank",
        "Relu",
        "Relu6",
        "Relu6Grad",
        "ReluGrad",
        "Reshape",
        "ResizeNearestNeighbor",
        "ResizeNearestNeighborGrad",
        "Reverse",
        "ReverseSequence",
        "ReverseV2",
        "Round",
        "Select",
        "SelectV2",
        "Shape",
        "ShapeN",
        "Sign",
        "Slice",
        "Snapshot",
        "SpaceToBatch",
        "SpaceToBatchND",
        "SpaceToDepth",
        "Split",
        "SplitV",
        "Squeeze",
        "StopGradient",
        "StridedSlice",
        "StridedSliceGrad",
        "Switch",
        "Tile",
        "TopK",
        "TopKV2",
        "Transpose",
        "Where",
        "Unpack",
        "ZerosLike",
    };
    AddTensorListOps(&list);
    UpdateList("CLEARLIST", &list);
    return list;
  }
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_AUTO_MIXED_PRECISION_LISTS_H_
