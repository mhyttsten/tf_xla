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
class MHTracer_DTPStensorflowPSlitePStoolsPSlist_flex_opsDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSlist_flex_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSlist_flex_opsDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/list_flex_ops.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "json/json.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace flex {

std::string OpListToJSONString(const OpKernelSet& flex_ops) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSlist_flex_opsDTcc mht_0(mht_0_v, 205, "", "./tensorflow/lite/tools/list_flex_ops.cc", "OpListToJSONString");

  Json::Value result(Json::arrayValue);
  for (const OpKernel& op : flex_ops) {
    Json::Value op_kernel(Json::arrayValue);
    op_kernel.append(Json::Value(op.op_name));
    op_kernel.append(Json::Value(op.kernel_name));
    result.append(op_kernel);
  }
  return Json::FastWriter().write(result);
}

// Find the class name of the op kernel described in the node_def from the pool
// of registered ops. If no kernel class is found, return an empty string.
string FindTensorflowKernelClass(tensorflow::NodeDef* node_def) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSlist_flex_opsDTcc mht_1(mht_1_v, 221, "", "./tensorflow/lite/tools/list_flex_ops.cc", "FindTensorflowKernelClass");

  if (!node_def || node_def->op().empty()) {
    LOG(FATAL) << "Invalid NodeDef";
  }

  const tensorflow::OpRegistrationData* op_reg_data;
  auto status =
      tensorflow::OpRegistry::Global()->LookUp(node_def->op(), &op_reg_data);
  if (!status.ok()) {
    LOG(FATAL) << "Op " << node_def->op() << " not found: " << status;
  }
  AddDefaultsToNodeDef(op_reg_data->op_def, node_def);

  tensorflow::DeviceNameUtils::ParsedName parsed_name;
  if (!tensorflow::DeviceNameUtils::ParseFullName(node_def->device(),
                                                  &parsed_name)) {
    LOG(FATAL) << "Failed to parse device from node_def: "
               << node_def->ShortDebugString();
  }
  string class_name;
  if (!tensorflow::FindKernelDef(
           tensorflow::DeviceType(parsed_name.type.c_str()), *node_def,
           nullptr /* kernel_def */, &class_name)
           .ok()) {
    LOG(FATAL) << "Failed to find kernel class for op: " << node_def->op();
  }
  return class_name;
}

void AddFlexOpsFromModel(const tflite::Model* model, OpKernelSet* flex_ops) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSlist_flex_opsDTcc mht_2(mht_2_v, 253, "", "./tensorflow/lite/tools/list_flex_ops.cc", "AddFlexOpsFromModel");

  // Read flex ops.
  auto* subgraphs = model->subgraphs();
  if (!subgraphs) return;
  for (int subgraph_index = 0; subgraph_index < subgraphs->size();
       ++subgraph_index) {
    const tflite::SubGraph* subgraph = subgraphs->Get(subgraph_index);
    auto* operators = subgraph->operators();
    auto* opcodes = model->operator_codes();
    if (!operators || !opcodes) continue;
    for (int i = 0; i < operators->size(); ++i) {
      const tflite::Operator* op = operators->Get(i);
      const tflite::OperatorCode* opcode = opcodes->Get(op->opcode_index());
      if (tflite::GetBuiltinCode(opcode) != tflite::BuiltinOperator_CUSTOM ||
          !tflite::IsFlexOp(opcode->custom_code()->c_str())) {
        continue;
      }

      // Remove the "Flex" prefix from op name.
      std::string flex_op_name(opcode->custom_code()->c_str());
      std::string tf_op_name =
          flex_op_name.substr(strlen(tflite::kFlexCustomCodePrefix));

      // Read NodeDef and find the op kernel class.
      if (op->custom_options_format() !=
          tflite::CustomOptionsFormat_FLEXBUFFERS) {
        LOG(FATAL) << "Invalid CustomOptionsFormat";
      }
      const flatbuffers::Vector<uint8_t>* custom_opt_bytes =
          op->custom_options();
      if (custom_opt_bytes && custom_opt_bytes->size()) {
        // NOLINTNEXTLINE: It is common to use references with flatbuffer.
        const flexbuffers::Vector& v =
            flexbuffers::GetRoot(custom_opt_bytes->data(),
                                 custom_opt_bytes->size())
                .AsVector();
        std::string nodedef_str = v[1].AsString().str();
        tensorflow::NodeDef nodedef;
        if (nodedef_str.empty() || !nodedef.ParseFromString(nodedef_str)) {
          LOG(FATAL) << "Failed to parse data into a valid NodeDef";
        }
        // Flex delegate only supports running flex ops with CPU.
        *nodedef.mutable_device() = "/CPU:0";
        std::string kernel_class = FindTensorflowKernelClass(&nodedef);
        flex_ops->insert({tf_op_name, kernel_class});
      }
    }
  }
}
}  // namespace flex
}  // namespace tflite
