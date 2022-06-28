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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc() {
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

#include "tensorflow/core/grappler/utils.h"

#include <iterator>
#include <memory>
#include <queue>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {
namespace {
template <typename T>
bool SafeSetDoubleScalarTensorValue(double value, Tensor* tensor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/grappler/utils.cc", "SafeSetDoubleScalarTensorValue");

  using RealType = typename Eigen::NumTraits<T>::Real;
  if (value > static_cast<double>(Eigen::NumTraits<RealType>::highest()) ||
      value < static_cast<double>(Eigen::NumTraits<RealType>::lowest())) {
    return false;
  }
  tensor->flat<T>()(0) = static_cast<T>(value);
  return true;
}

template <typename T>
bool SafeSetIntScalarTensorValue(int value, Tensor* tensor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_1(mht_1_v, 227, "", "./tensorflow/core/grappler/utils.cc", "SafeSetIntScalarTensorValue");

  using RealType = typename Eigen::NumTraits<T>::Real;
  if (value > static_cast<int>(Eigen::NumTraits<RealType>::highest()) ||
      value < static_cast<int>(Eigen::NumTraits<RealType>::lowest())) {
    return false;
  }
  tensor->flat<T>()(0) = static_cast<T>(value);
  return true;
}

// Is 'node' an operator that consumes only the shape of its input, not the
// data itself?
// TODO(ezhulenev): move to op_types.h. Requires to break circular dependency.
// TODO(ezhulenev): what about Identity passing tensor to Shape consumer?
bool IsShapeConsumer(const NodeDef& node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_2(mht_2_v, 244, "", "./tensorflow/core/grappler/utils.cc", "IsShapeConsumer");

  const string& op = node.op();
  return op == "Shape" || op == "ShapeN" || op == "Rank" || op == "Size";
}

}  // namespace

string TensorIdToString(const TensorId& tensor_id) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_3(mht_3_v, 254, "", "./tensorflow/core/grappler/utils.cc", "TensorIdToString");

  return tensor_id.index() == 0 ? string(tensor_id.node())
                                : tensor_id.ToString();
}

string SafeTensorIdToString(const SafeTensorId& tensor_id) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_4(mht_4_v, 262, "", "./tensorflow/core/grappler/utils.cc", "SafeTensorIdToString");

  return tensor_id.index() == 0 ? tensor_id.node() : tensor_id.ToString();
}

bool IsSameInput(const string& name1, const string& name2) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name1: \"" + name1 + "\"");
   mht_5_v.push_back("name2: \"" + name2 + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_5(mht_5_v, 271, "", "./tensorflow/core/grappler/utils.cc", "IsSameInput");

  if (name1 == name2) return true;
  TensorId tensor1 = ParseTensorName(name1);
  TensorId tensor2 = ParseTensorName(name2);
  return tensor1 == tensor2;
}

bool IsControlInput(absl::string_view name) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_6(mht_6_v, 282, "", "./tensorflow/core/grappler/utils.cc", "IsControlInput");

  return !name.empty() && name[0] == '^';
}

bool IsControlInput(const TensorId& tensor_id) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_7(mht_7_v, 289, "", "./tensorflow/core/grappler/utils.cc", "IsControlInput");
 return tensor_id.index() < 0; }

string AddPrefixToNodeName(const string& name, const string& prefix,
                           const string& delimiter) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("name: \"" + name + "\"");
   mht_8_v.push_back("prefix: \"" + prefix + "\"");
   mht_8_v.push_back("delimiter: \"" + delimiter + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_8(mht_8_v, 298, "", "./tensorflow/core/grappler/utils.cc", "AddPrefixToNodeName");

  if (!name.empty()) {
    if (name[0] == '^') {
      return absl::StrCat("^", prefix, delimiter, name.substr(1));
    }
  }
  return absl::StrCat(prefix, delimiter, name);
}

string AddPrefixToNodeName(const string& name, const string& prefix) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("name: \"" + name + "\"");
   mht_9_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_9(mht_9_v, 312, "", "./tensorflow/core/grappler/utils.cc", "AddPrefixToNodeName");

  return AddPrefixToNodeName(name, prefix, "/");
}

bool ExecuteWithTimeout(std::function<void()> fn, const int64_t timeout_in_ms,
                        thread::ThreadPool* const thread_pool) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_10(mht_10_v, 320, "", "./tensorflow/core/grappler/utils.cc", "ExecuteWithTimeout");

  if (timeout_in_ms <= 0) {
    fn();
    return true;
  }
  auto done = std::make_shared<Notification>();
  thread_pool->Schedule([done, fn]() {
    fn();
    done->Notify();
  });
  const bool notified =
      WaitForNotificationWithTimeout(done.get(), timeout_in_ms * 1000);
  return notified;
}

string AsControlDependency(const NodeDef& node) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_11(mht_11_v, 338, "", "./tensorflow/core/grappler/utils.cc", "AsControlDependency");

  return absl::StrCat("^", node.name());
}

string AsControlDependency(const string& node_name) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_12(mht_12_v, 346, "", "./tensorflow/core/grappler/utils.cc", "AsControlDependency");

  CHECK(!node_name.empty());
  return (!node_name.empty() && node_name[0] == '^')
             ? node_name
             : absl::StrCat("^", node_name);
}

bool NodeIsOnCpu(const NodeDef* node) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_13(mht_13_v, 356, "", "./tensorflow/core/grappler/utils.cc", "NodeIsOnCpu");

  string task, device;
  return DeviceNameUtils::SplitDeviceName(node->device(), &task, &device) &&
         absl::StartsWith(device, DEVICE_CPU);
}

bool NodeIsOnGpu(const NodeDef* node) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_14(mht_14_v, 365, "", "./tensorflow/core/grappler/utils.cc", "NodeIsOnGpu");

  string task, device;
  return DeviceNameUtils::SplitDeviceName(node->device(), &task, &device) &&
         absl::StartsWith(device, DEVICE_GPU);
}

int NumOutputs(const NodeDef& node, GraphDef* graph) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_15(mht_15_v, 374, "", "./tensorflow/core/grappler/utils.cc", "NumOutputs");

  int num_outputs = 0;
  const OpDef* op_def = nullptr;
  auto status = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def);
  if (status.ok()) {
    for (const auto& output : op_def->output_arg()) {
      if (!output.type_list_attr().empty()) {
        num_outputs +=
            node.attr().at(output.type_list_attr()).list().type_size();
      } else if (!output.number_attr().empty()) {
        num_outputs += node.attr().at(output.number_attr()).i();
      } else {
        num_outputs++;
      }
    }
  } else {
    FunctionLibraryDefinition fdef(OpRegistry::Global(), graph->library());
    auto status = fdef.LookUpOpDef(node.op(), &op_def);
    if (status.ok()) {
      num_outputs = op_def->output_arg_size();
    }
  }
  return num_outputs;
}

bool HasControlInputs(const NodeDef& node) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_16(mht_16_v, 402, "", "./tensorflow/core/grappler/utils.cc", "HasControlInputs");

  const int num_inputs = node.input_size();
  if (num_inputs > 0 && IsControlInput(node.input(num_inputs - 1))) {
    return true;
  }
  return false;
}

bool HasRegularInputs(const NodeDef& node) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_17(mht_17_v, 413, "", "./tensorflow/core/grappler/utils.cc", "HasRegularInputs");

  const int num_inputs = node.input_size();
  if (num_inputs > 0 && !IsControlInput(node.input(0))) {
    return true;
  }
  return false;
}

int NumNonControlInputs(const NodeDef& node) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_18(mht_18_v, 424, "", "./tensorflow/core/grappler/utils.cc", "NumNonControlInputs");

  int num_inputs = 0;
  for (; num_inputs < node.input_size(); ++num_inputs) {
    const string& input = node.input(num_inputs);
    if (IsControlInput(input)) {
      return num_inputs;
    }
  }
  return num_inputs;
}

int NumControlInputs(const NodeDef& node) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_19(mht_19_v, 438, "", "./tensorflow/core/grappler/utils.cc", "NumControlInputs");

  int num_inputs = 0;
  for (; num_inputs < node.input_size(); ++num_inputs) {
    const string& input = node.input(node.input_size() - num_inputs - 1);
    if (!IsControlInput(input)) {
      return num_inputs;
    }
  }
  return num_inputs;
}

bool HasRegularOutputs(const NodeDef& node, const NodeMap& node_map) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_20(mht_20_v, 452, "", "./tensorflow/core/grappler/utils.cc", "HasRegularOutputs");

  for (const NodeDef* output : node_map.GetOutputs(node.name())) {
    for (const string& node_as_input : output->input()) {
      if (IsControlInput(node_as_input)) break;

      TensorId tensor = ParseTensorName(node_as_input);
      if (tensor.node() == node.name()) {
        return true;
      }
    }
  }
  return false;
}

bool HasControlOutputs(const NodeDef& node, const NodeMap& node_map) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_21(mht_21_v, 469, "", "./tensorflow/core/grappler/utils.cc", "HasControlOutputs");

  for (const NodeDef* output : node_map.GetOutputs(node.name())) {
    for (int idx = output->input_size() - 1; idx >= 0; --idx) {
      const string& node_as_input = output->input(idx);
      if (!IsControlInput(node_as_input)) break;

      TensorId tensor = ParseTensorName(node_as_input);
      if (tensor.node() == node.name()) {
        return true;
      }
    }
  }
  return false;
}

int NumControlOutputs(const NodeDef& node, const NodeMap& node_map) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_22(mht_22_v, 487, "", "./tensorflow/core/grappler/utils.cc", "NumControlOutputs");

  int num_outputs = 0;
  for (const NodeDef* output : node_map.GetOutputs(node.name())) {
    for (int idx = output->input_size() - 1; idx >= 0; --idx) {
      const string& node_as_input = output->input(idx);
      if (!IsControlInput(node_as_input)) break;

      TensorId tensor = ParseTensorName(node_as_input);
      if (tensor.node() == node.name()) {
        ++num_outputs;
      }
    }
  }
  return num_outputs;
}

int NumNonControlOutputs(const NodeDef& node, const NodeMap& node_map) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_23(mht_23_v, 506, "", "./tensorflow/core/grappler/utils.cc", "NumNonControlOutputs");

  int num_outputs = 0;
  for (const NodeDef* output : node_map.GetOutputs(node.name())) {
    for (const string& node_as_input : output->input()) {
      if (IsControlInput(node_as_input)) {
        break;
      }
      if (node_as_input == node.name()) {
        ++num_outputs;
      } else {
        const TensorId tensor = ParseTensorName(node_as_input);
        if (tensor.node() == node.name()) {
          ++num_outputs;
        }
      }
    }
  }
  return num_outputs;
}

int NumNonControlDataOutputs(const NodeDef& node, const NodeMap& node_map) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_24(mht_24_v, 529, "", "./tensorflow/core/grappler/utils.cc", "NumNonControlDataOutputs");

  int num_data_outputs = 0;
  for (const NodeDef* output : node_map.GetOutputs(node.name())) {
    if (IsShapeConsumer(*output)) continue;

    for (int i = 0; i < output->input_size(); ++i) {
      const string& input = output->input(i);
      if (!IsControlInput(input) && NodeName(input) == node.name()) {
        ++num_data_outputs;
        break;
      }
    }
  }
  return num_data_outputs;
}

// Returns the data type in attribute `attr_name` of `node`. If that attribute
// doesn't exist, returns DT_INVALID.
DataType GetDataTypeFromAttr(const NodeDef& node, const string& type_attr) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("type_attr: \"" + type_attr + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_25(mht_25_v, 551, "", "./tensorflow/core/grappler/utils.cc", "GetDataTypeFromAttr");

  if (!node.attr().count(type_attr)) {
    return DT_INVALID;
  }
  const auto& attr = node.attr().at(type_attr);
  if (attr.value_case() != AttrValue::kType) {
    return DT_INVALID;
  }
  return attr.type();
}

NodeDef* GetTailOfChain(const NodeDef& source, const NodeMap& node_map,
                        bool follow_control_input,
                        const std::function<bool(const NodeDef&)>& pred_fn) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_26(mht_26_v, 567, "", "./tensorflow/core/grappler/utils.cc", "GetTailOfChain");

  const NodeDef* current = &source;
  const NodeDef* next = current;
  while (next == &source || (next != nullptr && pred_fn(*next))) {
    current = next;
    if (current->input_size() == 0 ||
        (!follow_control_input && IsControlInput(current->input(0)))) {
      break;
    }
    next = node_map.GetNode(current->input(0));
    if (next == nullptr) {
      LOG(ERROR) << "Node not found: " << current->input(0);
    }
  }
  return const_cast<NodeDef*>(current);
}

// Every permutation is a product of one or more cycles. Iterate over the cycles
// in the permutation, and convert each of those into a product of
// transpositions (swaps): https://en.wikipedia.org/wiki/Cyclic_permutation
void PermuteNodesInPlace(GraphDef* graph, std::vector<int>* permutation,
                         bool invert_permutation) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_27(mht_27_v, 591, "", "./tensorflow/core/grappler/utils.cc", "PermuteNodesInPlace");

  CHECK_EQ(graph->node_size(), permutation->size());
  std::vector<int> inv_perm(permutation->size(), 0);
  if (invert_permutation) {
    for (size_t n = 0; n < permutation->size(); ++n) {
      inv_perm[(*permutation)[n]] = n;
    }
    permutation->swap(inv_perm);
  }
  for (int n = 0, end = permutation->size(); n + 1 < end; ++n) {
    while (n != (*permutation)[n]) {
      std::size_t r = (*permutation)[n];
      graph->mutable_node()->SwapElements(n, r);
      std::swap((*permutation)[n], (*permutation)[r]);
    }
  }
}

void DedupControlInputs(NodeDef* node) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_28(mht_28_v, 612, "", "./tensorflow/core/grappler/utils.cc", "DedupControlInputs");

  absl::flat_hash_set<string> inputs;
  int pos = 0;
  while (pos < node->input_size()) {
    const string& input = node->input(pos);
    if (!inputs.insert(NodeName(input)).second && IsControlInput(input)) {
      node->mutable_input()->SwapElements(pos, node->input_size() - 1);
      node->mutable_input()->RemoveLast();
    } else {
      ++pos;
    }
  }
}

namespace {

template <typename UniqueContainer>
void EraseNodesFromGraphImpl(const UniqueContainer& nodes_to_delete,
                             GraphDef* graph) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_29(mht_29_v, 633, "", "./tensorflow/core/grappler/utils.cc", "EraseNodesFromGraphImpl");

  static_assert(std::is_same<typename UniqueContainer::value_type, int>::value,
                "Need to pass container of ints");

  int last = graph->node_size() - 1;
  for (auto it = nodes_to_delete.rbegin(); it != nodes_to_delete.rend(); ++it) {
    const int index = *it;
    graph->mutable_node()->SwapElements(index, last);
    last--;
  }
  graph->mutable_node()->DeleteSubrange(last + 1, nodes_to_delete.size());
}

template <typename T>
inline void STLSortAndRemoveDuplicates(T* v) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_30(mht_30_v, 650, "", "./tensorflow/core/grappler/utils.cc", "STLSortAndRemoveDuplicates");

  std::sort(v->begin(), v->end());
  v->erase(std::unique(v->begin(), v->end()), v->end());
}

}  // namespace

void EraseNodesFromGraph(const std::set<int>& nodes_to_delete,
                         GraphDef* graph) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_31(mht_31_v, 661, "", "./tensorflow/core/grappler/utils.cc", "EraseNodesFromGraph");

  EraseNodesFromGraphImpl(nodes_to_delete, graph);
}

void EraseNodesFromGraph(std::vector<int>&& nodes_to_delete, GraphDef* graph) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_32(mht_32_v, 668, "", "./tensorflow/core/grappler/utils.cc", "EraseNodesFromGraph");

  STLSortAndRemoveDuplicates(&nodes_to_delete);
  EraseNodesFromGraphImpl(nodes_to_delete, graph);
}

void EraseNodesFromGraph(const std::set<string>& nodes_to_delete,
                         GraphDef* graph) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_33(mht_33_v, 677, "", "./tensorflow/core/grappler/utils.cc", "EraseNodesFromGraph");

  std::vector<int> nodes_idx_to_delete;
  nodes_idx_to_delete.reserve(nodes_to_delete.size());
  for (int i = 0; i < graph->node_size(); ++i) {
    if (nodes_to_delete.count(graph->node(i).name()))
      nodes_idx_to_delete.push_back(i);
  }
  EraseNodesFromGraphImpl(nodes_idx_to_delete, graph);
}

#define HANDLE_DOUBLE_CASE(DTYPE)                                     \
  case DTYPE:                                                         \
    if (!SafeSetDoubleScalarTensorValue<EnumToDataType<DTYPE>::Type>( \
            static_cast<double>(value), tensor)) {                    \
      return errors::InvalidArgument("Cannot store value ", value,    \
                                     " in tensor of type " #DTYPE);   \
    }                                                                 \
    break

#define HANDLE_INT_CASE(DTYPE)                                               \
  case DTYPE:                                                                \
    if (!SafeSetIntScalarTensorValue<EnumToDataType<DTYPE>::Type>(value,     \
                                                                  tensor)) { \
      return errors::InvalidArgument("Cannot store value ", value,           \
                                     " in tensor of type " #DTYPE);          \
    }                                                                        \
    break

Status SetTensorValue(DataType dtype, int value, Tensor* tensor) {
  // TODO(rmlarsen): Support more general shapes.
  // TODO(lyandy): Change `value` to be int64 once int64 -> qint32 is supported.
  if (tensor->NumElements() != 1) {
    return errors::InvalidArgument(
        "Expected scalar tensor, got num_elements = ", tensor->NumElements());
  }
  switch (dtype) {
    HANDLE_DOUBLE_CASE(DT_HALF);
    HANDLE_DOUBLE_CASE(DT_BFLOAT16);
    HANDLE_DOUBLE_CASE(DT_BOOL);
    HANDLE_DOUBLE_CASE(DT_FLOAT);
    HANDLE_DOUBLE_CASE(DT_DOUBLE);
    HANDLE_DOUBLE_CASE(DT_UINT8);
    HANDLE_DOUBLE_CASE(DT_INT8);
    HANDLE_DOUBLE_CASE(DT_UINT16);
    HANDLE_DOUBLE_CASE(DT_INT16);
    HANDLE_DOUBLE_CASE(DT_INT32);
    HANDLE_DOUBLE_CASE(DT_INT64);
    HANDLE_DOUBLE_CASE(DT_COMPLEX64);
    HANDLE_DOUBLE_CASE(DT_COMPLEX128);
    HANDLE_INT_CASE(DT_QINT8);
    HANDLE_INT_CASE(DT_QUINT8);
    HANDLE_INT_CASE(DT_QINT16);
    HANDLE_INT_CASE(DT_QUINT16);
    HANDLE_INT_CASE(DT_QINT32);
    default:
      return errors::InvalidArgument("Unsupported type ",
                                     DataTypeString(dtype));
  }
  return Status::OK();
}

#undef HANDLE_CASE

Status CheckAttrExists(const NodeDef& node, const string& key) {
  if (!HasNodeAttr(node, key)) {
    return errors::InvalidArgument("Node '", node.name(), "' lacks '", key,
                                   "' attr: ", node.ShortDebugString());
  }
  return Status::OK();
}

Status CheckAttrsExist(const NodeDef& node, absl::Span<const string> keys) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_34(mht_34_v, 751, "", "./tensorflow/core/grappler/utils.cc", "CheckAttrsExist");

  for (const string& key : keys) {
    TF_RETURN_IF_ERROR(CheckAttrExists(node, key));
  }
  return Status::OK();
}

Status IsKernelRegisteredForNode(
    absl::string_view node_name, bool has_experimental_debug_info,
    const NodeDef_ExperimentalDebugInfo& experimental_debug_info,
    absl::string_view node_op, absl::string_view node_device,
    AttrSlice node_attrs) {
   std::vector<std::string> mht_35_v;
   mht_35_v.push_back("node_name: \"" + std::string(node_name.data(), node_name.size()) + "\"");
   mht_35_v.push_back("node_op: \"" + std::string(node_op.data(), node_op.size()) + "\"");
   mht_35_v.push_back("node_device: \"" + std::string(node_device.data(), node_device.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_35(mht_35_v, 768, "", "./tensorflow/core/grappler/utils.cc", "IsKernelRegisteredForNode");

  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(node_device, &parsed_name)) {
    return errors::InvalidArgument("Could not parse device name: ",
                                   node_device);
  }
  return FindKernelDef(DeviceType(parsed_name.type), node_name,
                       has_experimental_debug_info, experimental_debug_info,
                       node_op, node_device, node_attrs, nullptr, nullptr);
}

Status IsKernelRegisteredForNode(const NodeDef& node) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_36(mht_36_v, 782, "", "./tensorflow/core/grappler/utils.cc", "IsKernelRegisteredForNode");

  return IsKernelRegisteredForNode(node.name(),
                                   node.has_experimental_debug_info(),
                                   node.experimental_debug_info(), node.op(),
                                   node.device(), AttrSlice(&node.attr()));
}

namespace {
void RemoveAttributes(const std::vector<absl::string_view>& to_remove,
                      NodeDef* node) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_37(mht_37_v, 794, "", "./tensorflow/core/grappler/utils.cc", "RemoveAttributes");

  if (to_remove.size() == node->attr_size()) {
    node->clear_attr();
  } else {
    for (const auto& key : to_remove) {
      node->mutable_attr()->erase(string(key));
    }
  }
}
}  // namespace

int EraseRegularNodeAttributes(NodeDef* node) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_38(mht_38_v, 808, "", "./tensorflow/core/grappler/utils.cc", "EraseRegularNodeAttributes");

  std::vector<absl::string_view> to_remove;
  for (const auto& attr : node->attr()) {
    if (!attr.first.empty() && (attr.first)[0] != '_') {
      to_remove.push_back(attr.first);
    }
  }
  RemoveAttributes(to_remove, node);
  return to_remove.size();
}

int EraseNodeOutputAttributes(NodeDef* node) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsDTcc mht_39(mht_39_v, 822, "", "./tensorflow/core/grappler/utils.cc", "EraseNodeOutputAttributes");

  std::vector<absl::string_view> to_remove;
  for (const auto& attr : node->attr()) {
    const string& attr_name = attr.first;
    if (attr_name == "_xla_inferred_shapes" ||
        absl::StartsWith(attr_name, "_output_")) {
      to_remove.push_back(attr_name);
    }
  }
  RemoveAttributes(to_remove, node);
  return to_remove.size();
}

}  // end namespace grappler
}  // end namespace tensorflow
