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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSfuse_broadcast_into_following_binaryDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSfuse_broadcast_into_following_binaryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSfuse_broadcast_into_following_binaryDTcc() {
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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

// Returns true if the given op is strictly a broadcasting operation.
// This is commonly seen as a Concat of the same input multiple times, and is
// often generated from Tile ops that were converted via the
// convert_trivial_tile_to_concat transformation.
bool IsBroadcastingOp(const Model& model, Operator* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSfuse_broadcast_into_following_binaryDTcc mht_0(mht_0_v, 202, "", "./tensorflow/lite/toco/graph_transformations/fuse_broadcast_into_following_binary.cc", "IsBroadcastingOp");

  // Concatenation of identical inputs is usually a broadcast.
  if (op->type == OperatorType::kConcatenation) {
    // Verify that all inputs are the same.
    for (size_t i = 1; i < op->inputs.size(); ++i) {
      if (op->inputs[i] != op->inputs[0]) {
        return false;
      }
    }
    return true;
  }

  // There are other things we could look for (Stack/etc) when needed.
  return false;
}

}  // namespace

// Finds an operation that looks like a broadcast (concat of the same sources
// along the last dimension) and drops it by relying on the ability of certain
// binary ops to perform an implicit broadcast.
::tensorflow::Status FuseBroadcastIntoFollowingBinary::Run(Model* model,
                                                           std::size_t op_index,
                                                           bool* modified) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSfuse_broadcast_into_following_binaryDTcc mht_1(mht_1_v, 228, "", "./tensorflow/lite/toco/graph_transformations/fuse_broadcast_into_following_binary.cc", "FuseBroadcastIntoFollowingBinary::Run");

  *modified = false;
  const auto binary_it = model->operators.begin() + op_index;
  auto* binary_op = binary_it->get();

  // Test for binary ops of types that we know how to resolve
  if (binary_op->inputs.size() != 2) {
    return ::tensorflow::Status::OK();
  }
  if (binary_op->type != OperatorType::kAdd &&
      binary_op->type != OperatorType::kMul &&
      binary_op->type != OperatorType::kSub &&
      binary_op->type != OperatorType::kDiv) {
    return ::tensorflow::Status::OK();
  }

  // NOTE: either of these ops may be nullptr if the input array is constant.
  Operator* const op[2] = {
      GetOpWithOutput(*model, binary_op->inputs[0]),
      GetOpWithOutput(*model, binary_op->inputs[1]),
  };

  // Check whether either input is a broadcast-like concat.
  bool is_op_0_broadcast = op[0] && IsBroadcastingOp(*model, op[0]);
  bool is_op_1_broadcast = op[1] && IsBroadcastingOp(*model, op[1]);
  if (!is_op_0_broadcast && !is_op_1_broadcast) {
    // Neither input is a broadcast-looking thing.
    AddMessageF("Neither input looks broadcasty");
    return ::tensorflow::Status::OK();
  } else if (is_op_0_broadcast && is_op_1_broadcast) {
    AddMessageF(
        "Unable to fuse broadcast into %s as both inputs (%s, %s) are "
        "broadcasts",
        LogName(*binary_op), op[0] ? LogName(*op[0]) : "(?)",
        op[1] ? LogName(*op[1]) : "(?)");
    return ::tensorflow::Status::OK();
  }
  int broadcast_index = is_op_0_broadcast ? 0 : 1;

  // Just pull out the input of the broadcast op and pass it directly to the
  // binary op.
  AddMessageF("Fusing broadcast op %s into the following binary %s",
              LogName(*op[broadcast_index]), LogName(*binary_op));
  binary_op->inputs[broadcast_index] = op[broadcast_index]->inputs[0];

  // We leave the broadcast op in; it'll get cleaned up if it's not used later.
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
