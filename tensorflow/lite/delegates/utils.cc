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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTcc() {
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

#include "tensorflow/lite/delegates/utils.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {

TfLiteStatus CreateNewTensorWithDifferentType(TfLiteContext* context,
                                              const int original_tensor_index,
                                              TfLiteType new_type,
                                              TfLiteTensor** new_tensor,
                                              int* new_tensor_index) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/delegates/utils.cc", "CreateNewTensorWithDifferentType");

  TF_LITE_ENSURE_STATUS(context->AddTensors(context, 1, new_tensor_index));
  const TfLiteTensor& original_tensor = context->tensors[original_tensor_index];
  *new_tensor = &context->tensors[*new_tensor_index];
  (*new_tensor)->type = new_type;
  (*new_tensor)->allocation_type = kTfLiteArenaRw;
  const auto* original_dims = original_tensor.dims;
  TfLiteIntArray* dims = TfLiteIntArrayCreate(original_dims->size);
  for (int i = 0; i < original_dims->size; ++i) {
    dims->data[i] = original_dims->data[i];
  }
  if (context->ResizeTensor(context, *new_tensor, dims) != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context, "Could not resize new delegate tensor");
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus GraphPartitionHelper::Partition(
    std::set<std::string>* unsupported_nodes_info) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTcc mht_1(mht_1_v, 225, "", "./tensorflow/lite/delegates/utils.cc", "GraphPartitionHelper::Partition");

  const auto prepare_status = PrepareSupportedNodes(unsupported_nodes_info);
  if (prepare_status != kTfLiteOk) return prepare_status;

  TfLiteDelegateParams* partition_params_array_ = nullptr;
  int num_partitions_ = 0;
  if (context_->PreviewDelegatePartitioning(context_, supported_nodes_,
                                            &partition_params_array_,
                                            &num_partitions_) != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context_, "Unable to preview delegate partition.\n");
    return kTfLiteError;
  }

  for (int i = 0; i < num_partitions_; ++i) {
    partitions_.push_back(partition_params_array_ + i);
  }

  return kTfLiteOk;
}

std::vector<TfLiteDelegateParams*>
GraphPartitionHelper::GetFirstNLargestPartitions(
    int n, int min_nodes_per_partition) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTcc mht_2(mht_2_v, 250, "", "./tensorflow/lite/delegates/utils.cc", "GraphPartitionHelper::GetFirstNLargestPartitions");

  // In general, the number of partitions in a delegate is never likely to be
  // high enough to cause latency issues. Also considering this is generally a
  // one-time work, we simply unconditionally sort partitions here according to
  // the size.
  std::vector<TfLiteDelegateParams*> sorted_partitions(partitions_);
  std::sort(sorted_partitions.begin(), sorted_partitions.end(),
            [](TfLiteDelegateParams* left, TfLiteDelegateParams* right) {
              // Reverse sort
              return left->nodes_to_replace->size >
                     right->nodes_to_replace->size;
            });

  std::vector<TfLiteDelegateParams*> results;
  auto p_it = sorted_partitions.begin();
  const int total = sorted_partitions.size();
  for (int i = 0; i < std::min(total, n); ++i, ++p_it) {
    auto* p = (*p_it);
    if (p->nodes_to_replace->size < min_nodes_per_partition) {
      break;
    }
    results.push_back(p);
  }
  return results;
}

std::vector<int> GraphPartitionHelper::GetNodesOfFirstNLargestPartitionsImpl(
    int n, int min_nodes_per_partition) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTcc mht_3(mht_3_v, 280, "", "./tensorflow/lite/delegates/utils.cc", "GraphPartitionHelper::GetNodesOfFirstNLargestPartitionsImpl");

  auto first_n_partitions =
      GetFirstNLargestPartitions(n, min_nodes_per_partition);
  std::vector<int> ops_to_replace;
  for (const auto p : first_n_partitions) {
    auto nodes = p->nodes_to_replace;
    ops_to_replace.insert(ops_to_replace.end(), nodes->data,
                          nodes->data + nodes->size);
  }
  return ops_to_replace;
}

TfLiteStatus GraphPartitionHelper::PrepareSupportedNodes(
    std::set<std::string>* unsupported_nodes_info) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTcc mht_4(mht_4_v, 296, "", "./tensorflow/lite/delegates/utils.cc", "GraphPartitionHelper::PrepareSupportedNodes");

  if (!is_node_supported_fn_) return kTfLiteOk;

  TfLiteIntArray* execution_plan = nullptr;
  auto status = context_->GetExecutionPlan(context_, &execution_plan);
  if (status != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context_, "Unable to get graph execution plan.\n");
    return status;
  }
  // context->GetExecutionPlan invalidates memory obtained from previous calls,
  // which is dangerous if a delegate's IsNodeSupportedFn uses it anywhere.
  // So we store a copy to ensure validity.
  num_total_nodes_ = execution_plan->size;
  original_execution_plan_ = TfLiteIntArrayCreate(execution_plan->size);
  std::memcpy(original_execution_plan_->data, execution_plan->data,
              num_total_nodes_ * sizeof(int32_t));

  supported_nodes_ = TfLiteIntArrayCreate(num_total_nodes_);
  supported_nodes_->size = 0;
  for (int node_id : TfLiteIntArrayView(original_execution_plan_)) {
    TfLiteNode* node;
    TfLiteRegistration* registration;

    status = context_->GetNodeAndRegistration(context_, node_id, &node,
                                              &registration);
    if (status != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context_,
                         "Couldn't get node and registration info for op: %d\n",
                         node_id);
      supported_nodes_->size = 0;
      return status;
    }

    std::string unsupported_details;
    if (IsNodeSupported(context_, node, registration, node_id,
                        &unsupported_details)) {
      supported_nodes_->data[supported_nodes_->size++] = node_id;
    } else if (unsupported_nodes_info) {
      std::string node_info = GetOpNameByRegistration(*registration);
      node_info.append(": ");
      node_info.append(unsupported_details);
      unsupported_nodes_info->insert(node_info);
    }
  }

  num_supported_nodes_ = supported_nodes_->size;
  return kTfLiteOk;
}

std::vector<int>
FP16GraphPartitionHelper::GetNodesOfFirstNLargestPartitionsImpl(
    int n, int min_nodes_per_partition) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTcc mht_5(mht_5_v, 350, "", "./tensorflow/lite/delegates/utils.cc", "FP16GraphPartitionHelper::GetNodesOfFirstNLargestPartitionsImpl");

  std::vector<int> ops_to_replace;

  if (num_supported_nodes() + constant_dequant_nodes_.size() ==
      num_total_nodes()) {
    // Scenario 1: Full Delegation.
    // We delegate all nodes in this case to avoid unnecessary partitions due to
    // FP16 DEQUANT nodes. This is safe to do since no non-delegated op needs
    // the output of such a DEQUANT.
    for (int node_id : TfLiteIntArrayView(original_execution_plan_)) {
      ops_to_replace.push_back(node_id);
    }
  } else {
    // Scenario 2: Partial Delegation.
    // In this case, we just select the top 'n' applicable node subsets to
    // delegate, devoid of any FP16 DEQUANT ops. Handling the latter is tricky
    // in partial delegation cases & causes edge cases if non-delegated nodes
    // consume their output. So we keep all of them on CPU.
    auto first_n_partitions =
        GetFirstNLargestPartitions(n, min_nodes_per_partition);
    if (first_n_partitions.empty()) return ops_to_replace;
    for (int i = 0; i < first_n_partitions.size(); ++i) {
      auto nodes = first_n_partitions[i]->nodes_to_replace;
      ops_to_replace.insert(ops_to_replace.end(), nodes->data,
                            nodes->data + nodes->size);
    }
  }

  // Modify the inputs of relevant ops that support fp16 constants.
  RemapFp16InputTensors(ops_to_replace);
  return ops_to_replace;
}

bool FP16GraphPartitionHelper::IsNodeSupported(
    TfLiteContext* context, TfLiteNode* node, TfLiteRegistration* registration,
    int node_id, std::string* unsupported_details) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTcc mht_6(mht_6_v, 388, "", "./tensorflow/lite/delegates/utils.cc", "FP16GraphPartitionHelper::IsNodeSupported");

  if (registration->builtin_code == kTfLiteBuiltinDequantize) {
    auto& dequantize_input = context_->tensors[node->inputs->data[0]];
    if (dequantize_input.type == kTfLiteFloat16 &&
        IsConstantTensor(&dequantize_input)) {
      // Update mappings if this node is a fp16 DEQUANTIZE node that
      // works on a **constant** input tensor.
      // If the input is not a constant, the remapping that we do here will
      // cause bugs due to preceding ops such as DENSIFY.
      constant_dequant_map_[node->outputs->data[0]] = node->inputs->data[0];
      constant_dequant_nodes_[node->outputs->data[0]] = node_id;
      // We do not accept these ops right now.
      // This is done to support use-cases where a DEQUANTIZE output might be
      // consumed by a CPU op.
      return false;
    }
  }

  // To check if a (possibly) FP16 node is supported, we temporarily point the
  // node's inputs to the original fp16 tensors. This 'mutated' node is then
  // passed to the base IsNodeSupported function for checking. After the check,
  // we remap the original node inputs, so that the TFLite graph remains the
  // same.
  std::vector<int> orig_inputs;
  if (!constant_dequant_nodes_.empty()) {
    RemapFp16InputTensors(node, &orig_inputs);
  }

  const auto is_supported = GraphPartitionHelper::IsNodeSupported(
      context, node, registration, node_id, unsupported_details);

  if (!orig_inputs.empty() && node->inputs->size == orig_inputs.size()) {
    // Remapping happened. Restore original inputs.
    for (int j = 0; j < node->inputs->size; ++j) {
      node->inputs->data[j] = orig_inputs[j];
    }
  }
  return is_supported;
}

void FP16GraphPartitionHelper::RemapFp16InputTensors(
    const std::vector<int>& nodes) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTcc mht_7(mht_7_v, 432, "", "./tensorflow/lite/delegates/utils.cc", "FP16GraphPartitionHelper::RemapFp16InputTensors");

  for (int node_id : nodes) {
    TfLiteNode* node;
    TfLiteRegistration* registration;
    TfLiteStatus status = context_->GetNodeAndRegistration(
        context_, node_id, &node, &registration);
    if (status != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context_,
                         "Couldn't get node and registration info for op: %d\n",
                         node_id);
    }
    RemapFp16InputTensors(node, nullptr /* orig_inputs*/);
  }
}

void FP16GraphPartitionHelper::RemapFp16InputTensors(
    TfLiteNode* node, std::vector<int>* orig_inputs) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsDTcc mht_8(mht_8_v, 451, "", "./tensorflow/lite/delegates/utils.cc", "FP16GraphPartitionHelper::RemapFp16InputTensors");

  TfLiteIntArray* inputs = node->inputs;
  auto inputs_view = TfLiteIntArrayView(inputs);
  // Prepopulate 'orig_inputs' first and clear it if there's no input from a
  // dequant op.
  if (orig_inputs) {
    orig_inputs->clear();
    orig_inputs->reserve(inputs->size);
    for (auto tid : inputs_view) {
      orig_inputs->push_back(tid);
    }
  }
  // Fix this node's inputs (i.e. prune out the preceding dequantize node) in
  // order to test if it is supported.
  bool is_remapped = false;
  for (int j = 0; j < inputs->size; ++j) {
    const int input_tid = inputs->data[j];
    const auto it = constant_dequant_map_.find(input_tid);
    if (it != constant_dequant_map_.end()) {
      inputs->data[j] = it->second;
      is_remapped = true;
    }
  }
  if (!is_remapped && orig_inputs) orig_inputs->clear();
}

}  // namespace delegates
}  // namespace tflite
