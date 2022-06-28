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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSremove_successive_transposeDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSremove_successive_transposeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSremove_successive_transposeDTcc() {
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
#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

bool TransformsToIdentity(std::vector<int> const& perm1,
                          std::vector<int> const& perm2) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSremove_successive_transposeDTcc mht_0(mht_0_v, 196, "", "./tensorflow/lite/toco/graph_transformations/remove_successive_transpose.cc", "TransformsToIdentity");

  if (perm2.size() != perm1.size() || perm1.empty()) {
    return false;
  }
  // perm1 is the order of the indices after first transpose. When perm1 is
  // reordered according to perm2, if the result is simple increasing sequence
  // i.e., range(0, perm1.size()), then the two transposes cancel each other.
  for (size_t i = 0; i < perm1.size(); ++i) {
    if (perm1[i] < 0 || perm1[i] >= static_cast<int>(perm1.size()) ||
        perm2[i] < 0 || perm2[i] >= static_cast<int>(perm1.size())) {
      return false;
    }
    if (perm1[perm2[i]] != static_cast<int>(i)) {
      return false;
    }
  }
  return true;
}

void ReplaceOpInputsWith(Model* model, const std::string& lookfor,
                         const std::string& replacewith) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("lookfor: \"" + lookfor + "\"");
   mht_1_v.push_back("replacewith: \"" + replacewith + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSremove_successive_transposeDTcc mht_1(mht_1_v, 221, "", "./tensorflow/lite/toco/graph_transformations/remove_successive_transpose.cc", "ReplaceOpInputsWith");

  for (const auto& op : model->operators) {
    for (size_t i = 0; i < op->inputs.size(); ++i) {
      if (op->inputs[i] == lookfor) {
        op->inputs[i] = replacewith;
      }
    }
  }
}

}  // namespace

::tensorflow::Status RemoveSuccessiveTranspose::Run(Model* model,
                                                    std::size_t op_index,
                                                    bool* modified) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSremove_successive_transposeDTcc mht_2(mht_2_v, 238, "", "./tensorflow/lite/toco/graph_transformations/remove_successive_transpose.cc", "RemoveSuccessiveTranspose::Run");

  *modified = false;
  auto op = model->operators.begin() + op_index;
  if (op->get()->type != OperatorType::kTranspose) {
    return ::tensorflow::Status::OK();
  }

  TransposeOperator* t_op = static_cast<TransposeOperator*>(op->get());
  if (CountOpsWithInput(*model, t_op->outputs[0]) != 1) {
    return ::tensorflow::Status::OK();
  }
  Operator* next = GetOpWithInput(*model, t_op->outputs[0]);
  if (!next || next->type != OperatorType::kTranspose) {
    return ::tensorflow::Status::OK();
  }

  TransposeOperator* t_next = static_cast<TransposeOperator*>(next);
  if (!CountOpsWithInput(*model, t_next->outputs[0])) {
    return ::tensorflow::Status::OK();
  }

  if (TransformsToIdentity(t_op->perm, t_next->perm)) {
    // Find the input tensor that uses the results of transpose t_next, then
    // make it point to the input of t_op, effectively isolating both the
    // transposes from the graph.
    ReplaceOpInputsWith(model, t_next->outputs[0], t_op->inputs[0]);
    DeleteOpAndArrays(model, t_next);
    DeleteOpAndArrays(model, t_op);
    *modified = true;
  }

  return ::tensorflow::Status::OK();
}

}  // namespace toco
