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
class MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePScond_builderDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePScond_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePScond_builderDTcc() {
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

#include "tensorflow/core/tpu/graph_rewrite/cond_builder.h"

#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/tpu/graph_rewrite/incomplete_nodedef_builder.h"

namespace tensorflow {

CondBuilder::CondBuilder(string name, string device, const NodeDebugInfo& debug,
                         Graph* graph)
    : graph_(graph), name_(std::move(name)), device_(std::move(device)) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   mht_0_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePScond_builderDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/tpu/graph_rewrite/cond_builder.cc", "CondBuilder::CondBuilder");

  auto new_name = [graph, this](string suffix) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("suffix: \"" + suffix + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePScond_builderDTcc mht_1(mht_1_v, 203, "", "./tensorflow/core/tpu/graph_rewrite/cond_builder.cc", "lambda");

    return graph->NewName(strings::StrCat(name_, "/", suffix));
  };
  TF_CHECK_OK(
      IncompleteNodeDefBuilder::Identity(new_name("pred"), DT_BOOL, debug)
          .Device(device_)
          .Build(graph_, &pred_));
  Node* switch_pred;
  TF_CHECK_OK(
      IncompleteNodeDefBuilder::Switch(new_name("switch_pred"), DT_BOOL, debug)
          .Device(device_)
          .Build(graph_, &switch_pred));
  graph_->AddEdge(pred(), 0, switch_pred, 0);
  graph_->AddEdge(pred(), 0, switch_pred, 1);
  TF_CHECK_OK(
      IncompleteNodeDefBuilder::Identity(new_name("switch_f"), DT_BOOL, debug)
          .Device(device_)
          .Build(graph_, &switch_f_));
  TF_CHECK_OK(
      IncompleteNodeDefBuilder::Identity(new_name("switch_t"), DT_BOOL, debug)
          .Device(device_)
          .Build(graph_, &switch_t_));
  graph_->AddEdge(switch_pred, kElseBranch, switch_f_, 0);
  graph_->AddEdge(switch_pred, kThenBranch, switch_t_, 0);
  Node* merge_pred;
  TF_CHECK_OK(IncompleteNodeDefBuilder::Merge(new_name("merge_pred"), DT_BOOL,
                                              debug, /*n=*/2)
                  .Device(device_)
                  .Build(graph_, &merge_pred));
  graph_->AddEdge(switch_f_, 0, merge_pred, kElseBranch);
  graph_->AddEdge(switch_t_, 0, merge_pred, kThenBranch);
  // Note: when additional return values are added then there should be a
  // control dependency between those merge nodes and control_successor_ to
  // ensure that it is control successor of conditional.
  control_successor_ = merge_pred;
}

Node* CondBuilder::pred() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePScond_builderDTcc mht_2(mht_2_v, 243, "", "./tensorflow/core/tpu/graph_rewrite/cond_builder.cc", "CondBuilder::pred");
 return pred_; }

Node* CondBuilder::switch_f() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePScond_builderDTcc mht_3(mht_3_v, 248, "", "./tensorflow/core/tpu/graph_rewrite/cond_builder.cc", "CondBuilder::switch_f");
 return switch_f_; }

Node* CondBuilder::switch_t() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePScond_builderDTcc mht_4(mht_4_v, 253, "", "./tensorflow/core/tpu/graph_rewrite/cond_builder.cc", "CondBuilder::switch_t");
 return switch_t_; }

Node* CondBuilder::control_successor() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePScond_builderDTcc mht_5(mht_5_v, 258, "", "./tensorflow/core/tpu/graph_rewrite/cond_builder.cc", "CondBuilder::control_successor");
 return control_successor_; }

Status CondBuilder::AddInput(const string& input_name, const DataType& type,
                             const string& device, const NodeDebugInfo& debug,
                             Node** input) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("input_name: \"" + input_name + "\"");
   mht_6_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePScond_builderDTcc mht_6(mht_6_v, 267, "", "./tensorflow/core/tpu/graph_rewrite/cond_builder.cc", "CondBuilder::AddInput");

  auto b = IncompleteNodeDefBuilder::Switch(
      graph_->NewName(strings::StrCat(name_, "/", input_name)), type, debug);
  TF_RETURN_IF_ERROR(b.Device(device).Build(graph_, input));
  graph_->AddEdge(pred(), 0, *input, 1);
  return Status::OK();
}

}  // namespace tensorflow
