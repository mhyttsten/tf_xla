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
#ifndef TENSORFLOW_LITE_TOCO_GRAPH_TRANSFORMATIONS_GRAPH_TRANSFORMATIONS_H_
#define TENSORFLOW_LITE_TOCO_GRAPH_TRANSFORMATIONS_GRAPH_TRANSFORMATIONS_H_
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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh() {
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


#include <cstddef>
#include <initializer_list>
#include <unordered_set>
#include <vector>

#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/toco_port.h"

namespace toco {

class GraphTransformation {
 public:
  virtual ::tensorflow::Status Run(Model* model, std::size_t op_index,
                                   bool* modified) = 0;
  virtual const char* Name() const = 0;
  virtual ~GraphTransformation() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_0(mht_0_v, 202, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "~GraphTransformation");
}
  // Returns the list of messages that this graph transformation
  // generated since ClearMessages() was called.
  const std::vector<std::string>& Messages() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_1(mht_1_v, 208, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "Messages");
 return messages_; }
  // Clears the list of messages; should be called after every
  // run of this graph transformation.
  void ClearMessages() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_2(mht_2_v, 214, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "ClearMessages");
 return messages_.clear(); }
  // Adds a message; normally only called by the graph transformation
  // itself during its run (this function could be protected).
  template <typename... Args>
  void AddMessageF(const char* format, const Args&... args) {
    return messages_.push_back(toco::port::StringF(format, args...));
  }

 protected:
  GraphTransformation() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_3(mht_3_v, 226, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "GraphTransformation");
}

  // List of messages generated by this graph transformation.
  std::vector<std::string> messages_;

 private:
  GraphTransformation(const GraphTransformation& other) = delete;
  GraphTransformation(const GraphTransformation&& other) = delete;
};

class GraphTransformationsSet {
 public:
  // The choice of a container with fully-specified iteration order
  // ensures that graph transformations are always run in the same order,
  // which avoids having toco randomly fail or produce different results
  // depending on the toolchain. Ideally success/results should be independent
  // of the order in which graph transformations are run, but that's
  // unfortunately not currently guaranteed to be the case.
  using TransformationsContainer =
      std::vector<std::unique_ptr<GraphTransformation>>;

  GraphTransformationsSet() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_4(mht_4_v, 250, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "GraphTransformationsSet");
}
  GraphTransformationsSet(
      const std::initializer_list<GraphTransformation*> transformations) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_5(mht_5_v, 255, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "GraphTransformationsSet");

    for (GraphTransformation* t : transformations) {
      Add(t);
    }
  }
  void Add(GraphTransformation* transformation) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_6(mht_6_v, 263, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "Add");

    const std::string& name = transformation->Name();
    CHECK(!names_.count(name));
    names_.insert(name);
    transformations_.emplace_back(transformation);
  }
  TransformationsContainer::const_iterator begin() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_7(mht_7_v, 272, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "begin");

    return transformations_.begin();
  }
  TransformationsContainer::const_iterator end() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_8(mht_8_v, 278, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "end");

    return transformations_.end();
  }
  bool empty() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_9(mht_9_v, 284, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "empty");
 return transformations_.empty(); }

 private:
  GraphTransformationsSet(const GraphTransformationsSet& other) = delete;
  GraphTransformationsSet(const GraphTransformationsSet&& other) = delete;
  std::vector<std::unique_ptr<GraphTransformation>> transformations_;
  // Names of transformations in the set. Only used to guard against dupes.
  std::unordered_set<std::string> names_;
};

// Run the given list of graph transformations on the model.
// The message is only for logging purposes.
// The transformations is a rvalue reference, indicating that
// nothing else will use these pointers. The user is supposed to
// construct GraphTransformation objects by using 'new', pass us
// the resulting raw pointers, and this RunGraphTransformations
// takes care of delete'ing these pointers.
tensorflow::Status RunGraphTransformationsWithStatus(
    Model* model, const std::string& msg,
    const GraphTransformationsSet& transformations);

inline void RunGraphTransformations(
    Model* model, const std::string& msg,
    const GraphTransformationsSet& transformations) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("msg: \"" + msg + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_10(mht_10_v, 311, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "RunGraphTransformations");

  auto s = RunGraphTransformationsWithStatus(model, msg, transformations);
  CHECK(s.ok()) << s.error_message();
}

#define DECLARE_GRAPH_TRANSFORMATION(GTName)                     \
  class GTName : public GraphTransformation {                    \
   public:                                                       \
    ::tensorflow::Status Run(Model* model, std::size_t op_index, \
                             bool* modified) override;           \
    const char* Name() const override { return #GTName; }        \
  };

// List of all graph transformations
DECLARE_GRAPH_TRANSFORMATION(ConvertExpandDimsToReshape)
DECLARE_GRAPH_TRANSFORMATION(ConvertMatrixSetDiagV2OrV3ToV1)
DECLARE_GRAPH_TRANSFORMATION(ConvertMatrixDiagV2OrV3ToV1)
DECLARE_GRAPH_TRANSFORMATION(ConvertPureConvToDepthwise)
DECLARE_GRAPH_TRANSFORMATION(ConvertReorderAxes)
DECLARE_GRAPH_TRANSFORMATION(ConvertSqueezeToReshape)
DECLARE_GRAPH_TRANSFORMATION(ConvertTrivialAddNToAdd)
DECLARE_GRAPH_TRANSFORMATION(ConvertTrivialPackToReshape)
DECLARE_GRAPH_TRANSFORMATION(ConvertTrivialTileToConcat)
DECLARE_GRAPH_TRANSFORMATION(ConvertTrivialTransposeToReshape)
DECLARE_GRAPH_TRANSFORMATION(EnsureBiasVectors)
DECLARE_GRAPH_TRANSFORMATION(FuseActivationFunctions)
DECLARE_GRAPH_TRANSFORMATION(FuseBinaryIntoFollowingAffine)
DECLARE_GRAPH_TRANSFORMATION(FuseBinaryIntoPrecedingAffine)
DECLARE_GRAPH_TRANSFORMATION(FuseBroadcastIntoFollowingBinary)
DECLARE_GRAPH_TRANSFORMATION(GroupBidirectionalSequenceLstm)
DECLARE_GRAPH_TRANSFORMATION(GroupBidirectionalSequenceRnn)
DECLARE_GRAPH_TRANSFORMATION(GroupDynamicBidirectionalSequenceLstm)
DECLARE_GRAPH_TRANSFORMATION(GroupDynamicBidirectionalSequenceRnn)
DECLARE_GRAPH_TRANSFORMATION(IdentifyL2Normalization)
DECLARE_GRAPH_TRANSFORMATION(IdentifyL2Pool)
DECLARE_GRAPH_TRANSFORMATION(IdentifyLstmCell)
DECLARE_GRAPH_TRANSFORMATION(IdentifyHardSwish)
DECLARE_GRAPH_TRANSFORMATION(SplitLstmCellInputs)
DECLARE_GRAPH_TRANSFORMATION(MergeLstmCellInputs)
DECLARE_GRAPH_TRANSFORMATION(MergeReshapeIntoPrecedingTranspose)
DECLARE_GRAPH_TRANSFORMATION(IdentifyRelu1)
DECLARE_GRAPH_TRANSFORMATION(IdentifyPRelu)
DECLARE_GRAPH_TRANSFORMATION(MakeInitialDequantizeOperator)
DECLARE_GRAPH_TRANSFORMATION(MoveBinaryOperatorBeforeReshape)
DECLARE_GRAPH_TRANSFORMATION(PropagateActivationFunctionIntoConstants)
DECLARE_GRAPH_TRANSFORMATION(PropagateArrayDataTypes)
DECLARE_GRAPH_TRANSFORMATION(PropagateFakeQuantNumBits)
DECLARE_GRAPH_TRANSFORMATION(PropagateFixedSizes)
DECLARE_GRAPH_TRANSFORMATION(HardcodeMinMax)
DECLARE_GRAPH_TRANSFORMATION(Quantize)
DECLARE_GRAPH_TRANSFORMATION(RemoveFinalDequantizeOp)
DECLARE_GRAPH_TRANSFORMATION(RemoveSuccessiveTranspose)
DECLARE_GRAPH_TRANSFORMATION(RemoveTensorFlowAssert)
DECLARE_GRAPH_TRANSFORMATION(RemoveTensorFlowIdentity)
DECLARE_GRAPH_TRANSFORMATION(RemoveTrivialBinaryOperator)
DECLARE_GRAPH_TRANSFORMATION(RemoveTrivialConcatenation)
DECLARE_GRAPH_TRANSFORMATION(RemoveTrivialConcatenationInput)
DECLARE_GRAPH_TRANSFORMATION(RemoveTrivialFakeQuant)
DECLARE_GRAPH_TRANSFORMATION(RemoveTrivialSlice)
DECLARE_GRAPH_TRANSFORMATION(RemoveTrivialQuantizedActivationFunc)
DECLARE_GRAPH_TRANSFORMATION(RemoveTrivialQuantizedMinMax)
DECLARE_GRAPH_TRANSFORMATION(RemoveUnusedOp)
DECLARE_GRAPH_TRANSFORMATION(ResolveBatchNormalization)
DECLARE_GRAPH_TRANSFORMATION(ResolveConstantBinaryOperator)
DECLARE_GRAPH_TRANSFORMATION(ResolveConstantUnaryOperator)
DECLARE_GRAPH_TRANSFORMATION(CreateIm2colArrays)
DECLARE_GRAPH_TRANSFORMATION(DropIm2colArrays)
DECLARE_GRAPH_TRANSFORMATION(ReadArrayMinmaxAndNarrowRangeFromFakeQuant)
DECLARE_GRAPH_TRANSFORMATION(ReorderElementwiseUnary)
DECLARE_GRAPH_TRANSFORMATION(ReorderReshapeTranspose)
DECLARE_GRAPH_TRANSFORMATION(ResolveReorderAxes)
DECLARE_GRAPH_TRANSFORMATION(ResolveTensorFlowConcat)
DECLARE_GRAPH_TRANSFORMATION(ResolveTensorFlowMatMul)
DECLARE_GRAPH_TRANSFORMATION(ResolveTensorFlowMerge)
DECLARE_GRAPH_TRANSFORMATION(ResolveSqueezeAttributes)
DECLARE_GRAPH_TRANSFORMATION(ResolveTensorFlowSwitch)
DECLARE_GRAPH_TRANSFORMATION(ResolveConstantConcatenation)
DECLARE_GRAPH_TRANSFORMATION(ResolveConstantReshape)
DECLARE_GRAPH_TRANSFORMATION(ResolveConstantTranspose)
DECLARE_GRAPH_TRANSFORMATION(DropFakeQuant)
DECLARE_GRAPH_TRANSFORMATION(UnfuseActivationFunctions)
DECLARE_GRAPH_TRANSFORMATION(UnrollBatchMatMul)
DECLARE_GRAPH_TRANSFORMATION(ResolveSpaceToBatchNDAttributes)
DECLARE_GRAPH_TRANSFORMATION(ResolveBatchToSpaceNDAttributes)
DECLARE_GRAPH_TRANSFORMATION(ResolvePadAttributes)
DECLARE_GRAPH_TRANSFORMATION(ResolvePadV2Attributes)
DECLARE_GRAPH_TRANSFORMATION(ResolveReduceAttributes)
DECLARE_GRAPH_TRANSFORMATION(ResolveReshapeAttributes)
DECLARE_GRAPH_TRANSFORMATION(ResolveSliceAttributes)
DECLARE_GRAPH_TRANSFORMATION(ResolveStridedSliceAttributes)
DECLARE_GRAPH_TRANSFORMATION(ResolveTransposeAttributes)
DECLARE_GRAPH_TRANSFORMATION(ResolveConstantPack)
DECLARE_GRAPH_TRANSFORMATION(ResolveConstantRandomUniform)
DECLARE_GRAPH_TRANSFORMATION(ResolveConstantRange)
DECLARE_GRAPH_TRANSFORMATION(ResolveConstantShapeOrRank)
DECLARE_GRAPH_TRANSFORMATION(ResolveConstantSlice)
DECLARE_GRAPH_TRANSFORMATION(ResolveConstantStridedSlice)
DECLARE_GRAPH_TRANSFORMATION(ResolveConstantFill)
DECLARE_GRAPH_TRANSFORMATION(ResolveConstantGather)
DECLARE_GRAPH_TRANSFORMATION(ResolveConstantSelect)
DECLARE_GRAPH_TRANSFORMATION(ResolveConstantTile)
DECLARE_GRAPH_TRANSFORMATION(ResolveMultiplyByZero)
DECLARE_GRAPH_TRANSFORMATION(Dequantize)
DECLARE_GRAPH_TRANSFORMATION(UnpartitionEmbeddingLookup)
DECLARE_GRAPH_TRANSFORMATION(ShuffleFCWeights)
DECLARE_GRAPH_TRANSFORMATION(ResolveFakeQuantArgsFromVars)
DECLARE_GRAPH_TRANSFORMATION(ResolveGatherAttributes)
DECLARE_GRAPH_TRANSFORMATION(IdentifyNearestUpsample)

class PropagateDefaultMinMax : public GraphTransformation {
 public:
  ::tensorflow::Status Run(Model* model, std::size_t op_index,
                           bool* modified) override;
  const char* Name() const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_11(mht_11_v, 427, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "Name");
 return "PropagateDefaultMinMax"; }

  bool has_any_ranges_defined() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_12(mht_12_v, 432, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "has_any_ranges_defined");
 return !type_ranges_.empty(); }
  void DefineTypeRange(ArrayDataType data_type, double min, double max) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_13(mht_13_v, 436, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "DefineTypeRange");

    MinMax minmax;
    minmax.min = min;
    minmax.max = max;
    type_ranges_.emplace_back(data_type, minmax);
  }

 private:
  bool SetArrayMinMax(const std::string& array_name, Array* array);
  std::vector<std::pair<ArrayDataType, MinMax>> type_ranges_;
};

class RemoveTrivialReshape : public GraphTransformation {
 public:
  ::tensorflow::Status Run(Model* model, std::size_t op_index,
                           bool* modified) override;
  const char* Name() const override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_14(mht_14_v, 455, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "Name");
 return "RemoveTrivialReshape"; }
  bool treat_expand_dims_as_trivial() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_15(mht_15_v, 459, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "treat_expand_dims_as_trivial");

    return treat_expand_dims_as_trivial_;
  }
  void set_treat_expand_dims_as_trivial(bool val) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_16(mht_16_v, 465, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "set_treat_expand_dims_as_trivial");

    treat_expand_dims_as_trivial_ = val;
  }

 private:
  bool treat_expand_dims_as_trivial_ = false;
};

class ResolveConstantFakeQuant : public GraphTransformation {
 public:
  ::tensorflow::Status Run(Model* model, std::size_t op_index,
                           bool* modified) override;
  const char* Name() const override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_17(mht_17_v, 480, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "Name");
 return "ResolveConstantFakeQuant"; }

  // True if the num_bits should adjust the final data type.
  bool propagate_fake_quant_num_bits() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_18(mht_18_v, 486, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "propagate_fake_quant_num_bits");

    return propagate_fake_quant_num_bits_;
  }
  void set_propagate_fake_quant_num_bits(bool val) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_19(mht_19_v, 492, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "set_propagate_fake_quant_num_bits");

    propagate_fake_quant_num_bits_ = val;
  }

 private:
  bool propagate_fake_quant_num_bits_ = false;
};

class EnsureUint8WeightsSafeForFastInt8Kernels : public GraphTransformation {
 public:
  ::tensorflow::Status Run(Model* model, std::size_t op_index,
                           bool* modified) override;
  const char* Name() const override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_20(mht_20_v, 507, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "Name");

    return "EnsureUint8WeightsSafeForFastInt8Kernels";
  }
  bool allow_nudging_weights() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_21(mht_21_v, 513, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "allow_nudging_weights");
 return allow_nudging_weights_; }
  void set_allow_nudging_weights(bool val) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_22(mht_22_v, 517, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "set_allow_nudging_weights");
 allow_nudging_weights_ = val; }

  bool has_default_ranges_flag() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_23(mht_23_v, 522, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "has_default_ranges_flag");
 return has_default_ranges_flag_; }
  void set_has_default_ranges_flag(bool val) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_24(mht_24_v, 526, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "set_has_default_ranges_flag");
 has_default_ranges_flag_ = val; }

 private:
  bool allow_nudging_weights_ = false;
  bool has_default_ranges_flag_ = false;
};

class IdentifyDilatedConv : public GraphTransformation {
 public:
  ::tensorflow::Status Run(Model* model, std::size_t op_index,
                           bool* modified) override;
  const char* Name() const override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_25(mht_25_v, 540, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "Name");
 return "IdentifyDilatedConv"; }
  bool identify_depthwise_conv() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_26(mht_26_v, 544, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "identify_depthwise_conv");
 return identify_depthwise_conv_; }
  void set_identify_depthwise_conv(bool val) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTh mht_27(mht_27_v, 548, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.h", "set_identify_depthwise_conv");
 identify_depthwise_conv_ = val; }

 private:
  bool identify_depthwise_conv_ = true;
};

#undef DECLARE_GRAPH_TRANSFORMATION

}  // end namespace toco

#endif  // TENSORFLOW_LITE_TOCO_GRAPH_TRANSFORMATIONS_GRAPH_TRANSFORMATIONS_H_
