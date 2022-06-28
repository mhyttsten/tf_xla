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

#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTh() {
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


// Utility functions and classes for implementing delegates.

#include <functional>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace delegates {

// Creates a new Read/Write tensor having the same shape as the original, but
// with a different type. Note that this might void existing references to
// tensors.
TfLiteStatus CreateNewTensorWithDifferentType(TfLiteContext* context,
                                              const int original_tensor_index,
                                              TfLiteType new_type,
                                              TfLiteTensor** new_tensor,
                                              int* new_tensor_index);

using IsNodeSupportedFn =
    std::function<bool(TfLiteContext*, TfLiteNode*, TfLiteRegistration*,
                       std::string* unsupported_details)>;

// A utility class to help model graph parition.
// Note the class *needs* to be used in TfLiteDelegate::Prepare.
class GraphPartitionHelper {
 public:
  GraphPartitionHelper(TfLiteContext* context,
                       IsNodeSupportedFn is_node_supported_fn)
      : context_(context), is_node_supported_fn_(is_node_supported_fn) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTh mht_0(mht_0_v, 223, "", "./tensorflow/lite/delegates/utils.h", "GraphPartitionHelper");
}

  GraphPartitionHelper(TfLiteContext* context,
                       const std::vector<int>& supported_node_indices)
      : context_(context),
        num_total_nodes_(supported_node_indices.size()),
        supported_nodes_(
            ConvertVectorToTfLiteIntArray(supported_node_indices)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTh mht_1(mht_1_v, 233, "", "./tensorflow/lite/delegates/utils.h", "GraphPartitionHelper");
}

  virtual ~GraphPartitionHelper() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTh mht_2(mht_2_v, 238, "", "./tensorflow/lite/delegates/utils.h", "~GraphPartitionHelper");

    TfLiteIntArrayFree(supported_nodes_);
    TfLiteIntArrayFree(original_execution_plan_);
  }

  // Partition the graph into node subsets such that each subset could be
  // replaced with one delegate kernel (i.e. a kTfLiteBuiltinDelegate op).
  // If 'unsupported_nodes_info' is provided, it will be populated with
  // information about all different unsupported nodes.
  virtual TfLiteStatus Partition(std::set<std::string>* unsupported_nodes_info);

  // Returns the first n largest partitions or all if #partitions is less than
  // 'n' and each parition has at least (>=) 'min_nodes_per_partition' nodes.
  // Note that partitions are ranked according to the number of nodes that
  // a partition has, and the returned TfLiteDelegateParams objects are *owned*
  // by the TfLite runtime.
  // TODO(b/156707497): remove this and use GetNodesOfFirstNLargestPartitions
  std::vector<TfLiteDelegateParams*> GetFirstNLargestPartitions(
      int n = std::numeric_limits<int>::max(),
      int min_nodes_per_partition = 0) const;

  // Returns a list of node indices of all nodes from the first n largest
  // partitions. If there are fewer paritions than n, all nodes will be
  // returned. The partition is ranked according to the number of nodes.
  std::vector<int> GetNodesOfFirstNLargestPartitions(
      int n = std::numeric_limits<int>::max(),
      int min_nodes_per_partition = 0) {
    // Separated implementation that can be overrided, to preserve default value
    return GetNodesOfFirstNLargestPartitionsImpl(n, min_nodes_per_partition);
  }

  int num_total_nodes() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTh mht_3(mht_3_v, 272, "", "./tensorflow/lite/delegates/utils.h", "num_total_nodes");
 return num_total_nodes_; }
  int num_supported_nodes() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTh mht_4(mht_4_v, 276, "", "./tensorflow/lite/delegates/utils.h", "num_supported_nodes");
 return num_supported_nodes_; }
  int num_partitions() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTh mht_5(mht_5_v, 280, "", "./tensorflow/lite/delegates/utils.h", "num_partitions");
 return partitions_.size(); }

 protected:
  virtual bool IsNodeSupported(TfLiteContext* context, TfLiteNode* node,
                               TfLiteRegistration* registration, int node_id,
                               std::string* unsupported_details) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTh mht_6(mht_6_v, 288, "", "./tensorflow/lite/delegates/utils.h", "IsNodeSupported");

    return is_node_supported_fn_(context, node, registration,
                                 unsupported_details);
  }
  virtual std::vector<int> GetNodesOfFirstNLargestPartitionsImpl(
      int n, int min_nodes_per_partition);

  TfLiteContext* const context_ = nullptr;

  // Doesn't own the memory of each TfLiteDelegateParams object as it's
  // managed by the TfLite runtime itself. See
  // TfLiteContext::PreviewDelegatePartitioning for details.
  std::vector<TfLiteDelegateParams*> partitions_;

  // Copy of (pre-delegation) execution plan obtained from TfLiteContext in
  // PrepareSupportedNodes
  TfLiteIntArray* original_execution_plan_ = nullptr;

 private:
  // Generate a list of supported nodes (i.e. populating 'supported_nodes_') by
  // iterating over all nodes (i,e. those listed in the execution_plan
  // associated w/ 'context_').
  // If 'unsupported_nodes_info' is provided, it will be populated with
  // information about all different unsupported nodes.
  TfLiteStatus PrepareSupportedNodes(
      std::set<std::string>* unsupported_nodes_info = nullptr);

  // The number of total nodes passed in for partitioning (i.e. the
  // execution_plan size associated w/ 'context_')
  int num_total_nodes_ = 0;

  int num_supported_nodes_ = 0;

  // Tells if a node is supported as it could be delegated.
  const IsNodeSupportedFn is_node_supported_fn_ = nullptr;

  // Contains an array of supported node indices.
  TfLiteIntArray* supported_nodes_ = nullptr;  // owns the memory
};

// Specialized partitioner for graphs that possibly contain fp16 tensors.
//
// From nodes that accept fp16 inputs, this delegates the following:
// 1. All nodes (except DEQUANTIZE) that are supported with constant fp16 inputs
// by the delegate (in the TFLite graph, these nodes take in dequantized FP32
// outputs).
// 2. All fp16 DEQUANTIZE nodes that have *all* their consumers in the *first*
// delegated partition. This is because TFLite's partitioning algorithm
// greedily puts all such nodes in the first partition.
class FP16GraphPartitionHelper : public GraphPartitionHelper {
 public:
  FP16GraphPartitionHelper(TfLiteContext* context,
                           IsNodeSupportedFn is_node_supported_fn)
      : GraphPartitionHelper(context, std::move(is_node_supported_fn)) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTh mht_7(mht_7_v, 344, "", "./tensorflow/lite/delegates/utils.h", "FP16GraphPartitionHelper");
}

 protected:
  // Specialized function to handle fp16 nodes.
  bool IsNodeSupported(TfLiteContext* context, TfLiteNode* node,
                       TfLiteRegistration* registration, int node_id,
                       std::string* unsupported_details) override;

  // This will remap input tensors by removing FP16 to FP32 dequantized tensors.
  std::vector<int> GetNodesOfFirstNLargestPartitionsImpl(
      int n, int min_nodes_per_partition) override;

 private:
  // This remaps fp32 inputs of the given node to their corresponding fp16
  // version, if applicable. Can be summarized as:
  // fp16 -> DEQUANTIZE -> fp32 -> OP -> output
  // becomes
  // fp16 -> OP -> output
  void RemapFp16InputTensors(TfLiteNode* node,
                             std::vector<int>* orig_inputs) const;

  // Performs the above remapping for all nodes in the given list, without
  // tracking the original inputs.
  void RemapFp16InputTensors(const std::vector<int>& nodes) const;

  // ('dequantize' here refers to fp16 DEQUANTIZE)
  // Mapping of dequantize nodes' output tensor-id to its node id.
  // TODO(b/156707497): Use absl hash_maps here.
  std::unordered_map<int, int> constant_dequant_nodes_;
  // Mapping of DEQUANTIZE node's output (fp32) to its input (fp16).
  std::unordered_map<int, int> constant_dequant_map_;
};

}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_H_
