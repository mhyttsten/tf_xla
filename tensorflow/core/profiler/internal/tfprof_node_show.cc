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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_node_showDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_node_showDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_node_showDTcc() {
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

/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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
#include "tensorflow/core/profiler/internal/tfprof_node_show.h"

namespace tensorflow {
namespace tfprof {
namespace {}

ShowNode::ShowNode(const TFGraphNode* node) : node(node), account(false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_node_showDTcc mht_0(mht_0_v, 190, "", "./tensorflow/core/profiler/internal/tfprof_node_show.cc", "ShowNode::ShowNode");

  ReInit(-1);
}

void ShowNode::ReInit(int64_t step) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_node_showDTcc mht_1(mht_1_v, 197, "", "./tensorflow/core/profiler/internal/tfprof_node_show.cc", "ShowNode::ReInit");

  mutable_proto()->set_name(name());
  mutable_proto()->clear_devices();
  if (!node->canonical_device().empty()) {
    mutable_proto()->add_devices(node->canonical_device());
  }
  mutable_proto()->set_run_count(node->run_count(step));
  mutable_proto()->set_exec_micros(node->exec_micros(step));
  mutable_proto()->set_accelerator_exec_micros(
      node->accelerator_exec_micros(step));
  mutable_proto()->set_cpu_exec_micros(node->cpu_exec_micros(step));

  mutable_proto()->set_requested_bytes(node->requested_bytes(step));
  mutable_proto()->set_peak_bytes(node->peak_bytes(step));
  mutable_proto()->set_residual_bytes(node->residual_bytes(step));
  mutable_proto()->set_output_bytes(node->output_bytes(step));

  mutable_proto()->set_float_ops(node->float_ops(step));

  mutable_proto()->clear_input_shapes();
  for (const auto& inp : node->input_shapes()) {
    (*mutable_proto()->mutable_input_shapes())[inp.first].MergeFrom(
        VecToShapeProto(inp.second));
  }
  proto_.set_parameters(node->parameters());
}

GraphNodeProto* ShowNode::mutable_proto() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_node_showDTcc mht_2(mht_2_v, 227, "", "./tensorflow/core/profiler/internal/tfprof_node_show.cc", "ShowNode::mutable_proto");
 return &proto_; }

const GraphNodeProto& ShowNode::proto() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_node_showDTcc mht_3(mht_3_v, 232, "", "./tensorflow/core/profiler/internal/tfprof_node_show.cc", "ShowNode::proto");
 return proto_; }

void ShowNode::AggregateTotalStats(ShowNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_node_showDTcc mht_4(mht_4_v, 237, "", "./tensorflow/core/profiler/internal/tfprof_node_show.cc", "ShowNode::AggregateTotalStats");

  GraphNodeProto* node_pb = node->mutable_proto();
  mutable_proto()->set_total_run_count(proto().total_run_count() +
                                       node_pb->total_run_count());
  mutable_proto()->set_total_definition_count(
      proto().total_definition_count() + node_pb->total_definition_count());
  mutable_proto()->set_total_exec_micros(proto().total_exec_micros() +
                                         node_pb->total_exec_micros());
  mutable_proto()->set_total_accelerator_exec_micros(
      proto().total_accelerator_exec_micros() +
      node_pb->total_accelerator_exec_micros());
  mutable_proto()->set_total_cpu_exec_micros(proto().total_cpu_exec_micros() +
                                             node_pb->total_cpu_exec_micros());

  mutable_proto()->set_total_requested_bytes(proto().total_requested_bytes() +
                                             node_pb->total_requested_bytes());
  mutable_proto()->set_total_peak_bytes(proto().total_peak_bytes() +
                                        node_pb->total_peak_bytes());
  mutable_proto()->set_total_residual_bytes(proto().total_residual_bytes() +
                                            node_pb->total_residual_bytes());
  mutable_proto()->set_total_output_bytes(proto().total_output_bytes() +
                                          node_pb->total_output_bytes());
  mutable_proto()->set_total_parameters(proto().total_parameters() +
                                        node_pb->total_parameters());
  mutable_proto()->set_total_float_ops(proto().total_float_ops() +
                                       node_pb->total_float_ops());
}

void ShowNode::AddSelfToTotalStats() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_node_showDTcc mht_5(mht_5_v, 268, "", "./tensorflow/core/profiler/internal/tfprof_node_show.cc", "ShowNode::AddSelfToTotalStats");

  mutable_proto()->set_total_definition_count(proto().total_definition_count() +
                                              1);
  mutable_proto()->set_total_run_count(proto().total_run_count() +
                                       proto().run_count());
  mutable_proto()->set_total_exec_micros(proto().total_exec_micros() +
                                         proto().exec_micros());
  mutable_proto()->set_total_accelerator_exec_micros(
      proto().total_accelerator_exec_micros() +
      proto().accelerator_exec_micros());
  mutable_proto()->set_total_cpu_exec_micros(proto().total_cpu_exec_micros() +
                                             proto().cpu_exec_micros());

  mutable_proto()->set_total_requested_bytes(proto().total_requested_bytes() +
                                             proto().requested_bytes());
  mutable_proto()->set_total_peak_bytes(proto().total_peak_bytes() +
                                        proto().peak_bytes());
  mutable_proto()->set_total_residual_bytes(proto().total_residual_bytes() +
                                            proto().residual_bytes());
  mutable_proto()->set_total_output_bytes(proto().total_output_bytes() +
                                          proto().output_bytes());

  mutable_proto()->set_total_parameters(proto().total_parameters() +
                                        proto().parameters());
  mutable_proto()->set_total_float_ops(proto().total_float_ops() +
                                       proto().float_ops());
}

void ShowNode::ResetTotalStats() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_node_showDTcc mht_6(mht_6_v, 299, "", "./tensorflow/core/profiler/internal/tfprof_node_show.cc", "ShowNode::ResetTotalStats");

  formatted_str.clear();

  mutable_proto()->set_total_definition_count(0);
  mutable_proto()->set_total_run_count(0);
  mutable_proto()->set_total_exec_micros(0);
  mutable_proto()->set_total_accelerator_exec_micros(0);
  mutable_proto()->set_total_cpu_exec_micros(0);

  mutable_proto()->set_total_requested_bytes(0);
  mutable_proto()->set_total_peak_bytes(0);
  mutable_proto()->set_total_residual_bytes(0);
  mutable_proto()->set_total_output_bytes(0);

  mutable_proto()->set_total_parameters(0);
  mutable_proto()->set_total_float_ops(0);
  mutable_proto()->mutable_children()->Clear();
}

ShowMultiNode::ShowMultiNode(TFMultiGraphNode* node)
    : node(node), account(false), show(false) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_node_showDTcc mht_7(mht_7_v, 322, "", "./tensorflow/core/profiler/internal/tfprof_node_show.cc", "ShowMultiNode::ShowMultiNode");

  ReInit(-1, {".*"});
}

bool ShowMultiNode::ReInit(int64_t step,
                           const std::vector<string>& type_regexes) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_node_showDTcc mht_8(mht_8_v, 330, "", "./tensorflow/core/profiler/internal/tfprof_node_show.cc", "ShowMultiNode::ReInit");

  bool has_matched_type = node->SnapshotNodes(step, type_regexes);

  std::vector<ShowNode> snodes;
  mutable_proto()->mutable_graph_nodes()->Clear();
  for (const auto& it : node->graph_nodes()) {
    ShowNode snode(it.second);
    snodes.push_back(snode);
    snodes.back().ReInit(step);
    snodes.back().AddSelfToTotalStats();
    mutable_proto()->add_graph_nodes()->MergeFrom(snodes.back().proto());
  }

  mutable_proto()->set_name(name());
  mutable_proto()->set_exec_micros(node->exec_micros());
  mutable_proto()->set_accelerator_exec_micros(node->accelerator_exec_micros());
  mutable_proto()->set_cpu_exec_micros(node->cpu_exec_micros());

  mutable_proto()->set_requested_bytes(node->requested_bytes());
  mutable_proto()->set_peak_bytes(node->peak_bytes());
  mutable_proto()->set_residual_bytes(node->residual_bytes());
  mutable_proto()->set_output_bytes(node->output_bytes());

  mutable_proto()->set_float_ops(node->float_ops());

  mutable_proto()->set_parameters(node->parameters());
  return has_matched_type;
}

MultiGraphNodeProto* ShowMultiNode::mutable_proto() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_node_showDTcc mht_9(mht_9_v, 362, "", "./tensorflow/core/profiler/internal/tfprof_node_show.cc", "ShowMultiNode::mutable_proto");
 return &proto_; }

const MultiGraphNodeProto& ShowMultiNode::proto() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_node_showDTcc mht_10(mht_10_v, 367, "", "./tensorflow/core/profiler/internal/tfprof_node_show.cc", "ShowMultiNode::proto");
 return proto_; }

void ShowMultiNode::AggregateTotalStats(ShowMultiNode* node) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_node_showDTcc mht_11(mht_11_v, 372, "", "./tensorflow/core/profiler/internal/tfprof_node_show.cc", "ShowMultiNode::AggregateTotalStats");

  MultiGraphNodeProto* node_pb = node->mutable_proto();
  mutable_proto()->set_total_exec_micros(proto().total_exec_micros() +
                                         node_pb->total_exec_micros());
  mutable_proto()->set_total_accelerator_exec_micros(
      proto().total_accelerator_exec_micros() +
      node_pb->total_accelerator_exec_micros());
  mutable_proto()->set_total_cpu_exec_micros(proto().total_cpu_exec_micros() +
                                             node_pb->total_cpu_exec_micros());

  mutable_proto()->set_total_requested_bytes(proto().total_requested_bytes() +
                                             node_pb->total_requested_bytes());
  mutable_proto()->set_total_peak_bytes(proto().total_peak_bytes() +
                                        node_pb->total_peak_bytes());
  mutable_proto()->set_total_residual_bytes(proto().total_residual_bytes() +
                                            node_pb->total_residual_bytes());
  mutable_proto()->set_total_output_bytes(proto().total_output_bytes() +
                                          node_pb->total_output_bytes());

  mutable_proto()->set_total_parameters(proto().total_parameters() +
                                        node_pb->total_parameters());
  mutable_proto()->set_total_float_ops(proto().total_float_ops() +
                                       node_pb->total_float_ops());
}

void ShowMultiNode::AddSelfToTotalStats() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_node_showDTcc mht_12(mht_12_v, 400, "", "./tensorflow/core/profiler/internal/tfprof_node_show.cc", "ShowMultiNode::AddSelfToTotalStats");

  mutable_proto()->set_total_exec_micros(proto().total_exec_micros() +
                                         proto().exec_micros());
  mutable_proto()->set_total_accelerator_exec_micros(
      proto().total_accelerator_exec_micros() +
      proto().accelerator_exec_micros());
  mutable_proto()->set_total_cpu_exec_micros(proto().total_cpu_exec_micros() +
                                             proto().cpu_exec_micros());

  mutable_proto()->set_total_requested_bytes(proto().total_requested_bytes() +
                                             proto().requested_bytes());
  mutable_proto()->set_total_peak_bytes(proto().total_peak_bytes() +
                                        proto().peak_bytes());
  mutable_proto()->set_total_residual_bytes(proto().total_residual_bytes() +
                                            proto().residual_bytes());
  mutable_proto()->set_total_output_bytes(proto().total_output_bytes() +
                                          proto().output_bytes());

  mutable_proto()->set_total_parameters(proto().total_parameters() +
                                        proto().parameters());
  mutable_proto()->set_total_float_ops(proto().total_float_ops() +
                                       proto().float_ops());
}

void ShowMultiNode::ResetTotalStats() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_node_showDTcc mht_13(mht_13_v, 427, "", "./tensorflow/core/profiler/internal/tfprof_node_show.cc", "ShowMultiNode::ResetTotalStats");

  formatted_str.clear();
  mutable_proto()->set_total_exec_micros(0);
  mutable_proto()->set_total_accelerator_exec_micros(0);
  mutable_proto()->set_total_cpu_exec_micros(0);

  mutable_proto()->set_total_requested_bytes(0);
  mutable_proto()->set_total_peak_bytes(0);
  mutable_proto()->set_total_residual_bytes(0);
  mutable_proto()->set_total_output_bytes(0);

  mutable_proto()->set_total_parameters(0);
  mutable_proto()->set_total_float_ops(0);
  mutable_proto()->mutable_children()->Clear();
}

}  // namespace tfprof
}  // namespace tensorflow
