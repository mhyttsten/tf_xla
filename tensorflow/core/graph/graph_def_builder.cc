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
class MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTcc {
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
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/graph_def_builder.h"

#include <utility>

#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

GraphDefBuilder::Options::Options(Graph* graph, Status* status)
    : graph_(graph), status_(status) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/graph/graph_def_builder.cc", "GraphDefBuilder::Options::Options");
}
GraphDefBuilder::Options::~Options() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTcc mht_1(mht_1_v, 199, "", "./tensorflow/core/graph/graph_def_builder.cc", "GraphDefBuilder::Options::~Options");
}

GraphDefBuilder::Options GraphDefBuilder::Options::WithName(
    StringPiece name) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTcc mht_2(mht_2_v, 205, "", "./tensorflow/core/graph/graph_def_builder.cc", "GraphDefBuilder::Options::WithName");

  return Options(*this).WithNameImpl(name);
}
GraphDefBuilder::Options GraphDefBuilder::Options::WithDevice(
    StringPiece device) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTcc mht_3(mht_3_v, 212, "", "./tensorflow/core/graph/graph_def_builder.cc", "GraphDefBuilder::Options::WithDevice");

  return Options(*this).WithDeviceImpl(device);
}
GraphDefBuilder::Options GraphDefBuilder::Options::WithControlInput(
    Node* control_input) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTcc mht_4(mht_4_v, 219, "", "./tensorflow/core/graph/graph_def_builder.cc", "GraphDefBuilder::Options::WithControlInput");

  return Options(*this).WithControlInputImpl(control_input);
}
GraphDefBuilder::Options GraphDefBuilder::Options::WithControlInputs(
    gtl::ArraySlice<Node*> control_inputs) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTcc mht_5(mht_5_v, 226, "", "./tensorflow/core/graph/graph_def_builder.cc", "GraphDefBuilder::Options::WithControlInputs");

  return Options(*this).WithControlInputsImpl(control_inputs);
}
GraphDefBuilder::Options GraphDefBuilder::Options::WithNameImpl(
    StringPiece name) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTcc mht_6(mht_6_v, 233, "", "./tensorflow/core/graph/graph_def_builder.cc", "GraphDefBuilder::Options::WithNameImpl");

  name_ = string(name);
  return *this;
}
GraphDefBuilder::Options GraphDefBuilder::Options::WithDeviceImpl(
    StringPiece device) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTcc mht_7(mht_7_v, 241, "", "./tensorflow/core/graph/graph_def_builder.cc", "GraphDefBuilder::Options::WithDeviceImpl");

  device_ = string(device);
  return *this;
}
GraphDefBuilder::Options GraphDefBuilder::Options::WithControlInputImpl(
    Node* control_input) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTcc mht_8(mht_8_v, 249, "", "./tensorflow/core/graph/graph_def_builder.cc", "GraphDefBuilder::Options::WithControlInputImpl");

  control_inputs_.push_back(control_input);
  return *this;
}
GraphDefBuilder::Options GraphDefBuilder::Options::WithControlInputsImpl(
    gtl::ArraySlice<Node*> control_inputs) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTcc mht_9(mht_9_v, 257, "", "./tensorflow/core/graph/graph_def_builder.cc", "GraphDefBuilder::Options::WithControlInputsImpl");

  control_inputs_.insert(control_inputs_.end(), control_inputs.begin(),
                         control_inputs.end());
  return *this;
}

Status GraphDefBuilder::ToGraphDef(GraphDef* graph_def) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTcc mht_10(mht_10_v, 266, "", "./tensorflow/core/graph/graph_def_builder.cc", "GraphDefBuilder::ToGraphDef");

  if (status_.ok()) {
    graph_.ToGraphDef(graph_def);
    *graph_def->mutable_library() = flib_def_.ToProto();
  }
  return status_;
}

string GraphDefBuilder::Options::GetNameForOp(StringPiece op) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTcc mht_11(mht_11_v, 277, "", "./tensorflow/core/graph/graph_def_builder.cc", "GraphDefBuilder::Options::GetNameForOp");

  if (name_.empty()) return graph_->NewName(op);
  return name_;
}

Node* GraphDefBuilder::Options::FinalizeBuilder(NodeBuilder* builder) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTcc mht_12(mht_12_v, 285, "", "./tensorflow/core/graph/graph_def_builder.cc", "GraphDefBuilder::Options::FinalizeBuilder");

  builder->ControlInputs(control_inputs_);
  if (!device_.empty()) builder->Device(device_);
  for (const auto& attr : attrs_) {
    builder->Attr(attr.first, attr.second);
  }

  Node* returned_node;
  UpdateStatus(builder->Finalize(graph_, &returned_node));
  return returned_node;
}

void GraphDefBuilder::Options::UpdateStatus(const Status& status) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTcc mht_13(mht_13_v, 300, "", "./tensorflow/core/graph/graph_def_builder.cc", "GraphDefBuilder::Options::UpdateStatus");

  if (status_ == nullptr) {
    TF_CHECK_OK(status);
  } else {
    status_->Update(status);
  }
}

namespace ops {

Node* SourceOp(const string& op_name, const GraphDefBuilder::Options& opts) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTcc mht_14(mht_14_v, 314, "", "./tensorflow/core/graph/graph_def_builder.cc", "SourceOp");

  if (opts.HaveError()) return nullptr;
  NodeBuilder node_builder(opts.GetNameForOp(op_name), op_name,
                           opts.op_registry());
  return opts.FinalizeBuilder(&node_builder);
}

Node* UnaryOp(const string& op_name, NodeOut input,
              const GraphDefBuilder::Options& opts) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTcc mht_15(mht_15_v, 326, "", "./tensorflow/core/graph/graph_def_builder.cc", "UnaryOp");

  if (opts.HaveError()) return nullptr;
  NodeBuilder node_builder(opts.GetNameForOp(op_name), op_name,
                           opts.op_registry());
  node_builder.Input(std::move(input));
  return opts.FinalizeBuilder(&node_builder);
}

Node* BinaryOp(const string& op_name, NodeOut a, NodeOut b,
               const GraphDefBuilder::Options& opts) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTcc mht_16(mht_16_v, 339, "", "./tensorflow/core/graph/graph_def_builder.cc", "BinaryOp");

  if (opts.HaveError()) return nullptr;
  NodeBuilder node_builder(opts.GetNameForOp(op_name), op_name,
                           opts.op_registry());
  node_builder.Input(std::move(a)).Input(std::move(b));
  return opts.FinalizeBuilder(&node_builder);
}

Node* TernaryOp(const string& op_name, NodeOut a, NodeOut b, NodeOut c,
                const GraphDefBuilder::Options& opts) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTcc mht_17(mht_17_v, 352, "", "./tensorflow/core/graph/graph_def_builder.cc", "TernaryOp");

  if (opts.HaveError()) return nullptr;
  NodeBuilder node_builder(opts.GetNameForOp(op_name), op_name,
                           opts.op_registry());
  node_builder.Input(std::move(a)).Input(std::move(b)).Input(std::move(c));
  return opts.FinalizeBuilder(&node_builder);
}

}  // end namespace ops
}  // end namespace tensorflow
