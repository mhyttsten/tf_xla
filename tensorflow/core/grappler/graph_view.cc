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
class MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTcc() {
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

#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

namespace {
int OpPortIdToArgId(const NodeDef& node,
                    const protobuf::RepeatedPtrField<OpDef::ArgDef>& args,
                    int port_id) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/grappler/graph_view.cc", "OpPortIdToArgId");

  for (int arg_id = 0; arg_id < args.size(); ++arg_id) {
    if (port_id < 0) {
      return -1;
    } else if (port_id == 0) {
      return arg_id;
    }

    // Default is 1 port per arg.
    int n = 1;

    const auto& arg = args.Get(arg_id);
    if (!arg.number_attr().empty()) {
      n = node.attr().at(arg.number_attr()).i();
    } else if (!arg.type_list_attr().empty()) {
      n = node.attr().at(arg.type_list_attr()).list().type_size();
    }

    if (n < 0) {
      // This should never happen.
      DCHECK_GE(n, 0);
      return -1;
    } else if (port_id < n) {
      return arg_id;
    }
    port_id -= n;
  }

  return -1;
}
}  // end namespace

int OpOutputPortIdToArgId(const NodeDef& node, const OpDef& op, int port_id) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTcc mht_1(mht_1_v, 230, "", "./tensorflow/core/grappler/graph_view.cc", "OpOutputPortIdToArgId");

  return OpPortIdToArgId(node, op.output_arg(), port_id);
}

int OpInputPortIdToArgId(const NodeDef& node, const OpDef& op, int port_id) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTcc mht_2(mht_2_v, 237, "", "./tensorflow/core/grappler/graph_view.cc", "OpInputPortIdToArgId");

  return OpPortIdToArgId(node, op.input_arg(), port_id);
}

bool HasSingleFanoutNode(const GraphView& graph_view, const NodeDef* node,
                         int port) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTcc mht_3(mht_3_v, 245, "", "./tensorflow/core/grappler/graph_view.cc", "HasSingleFanoutNode");

  const auto output = GraphView::OutputPort(node, port);
  return graph_view.GetFanout(output).size() <= 1;
}

bool HasFanouts(const GraphView& graph_view, const NodeDef* node, int port) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTcc mht_4(mht_4_v, 253, "", "./tensorflow/core/grappler/graph_view.cc", "HasFanouts");

  const auto output = GraphView::OutputPort(node, port);
  return !graph_view.GetFanout(output).empty();
}

bool HasControlFanin(const GraphView& graph_view, const NodeDef* node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTcc mht_5(mht_5_v, 261, "", "./tensorflow/core/grappler/graph_view.cc", "HasControlFanin");

  const auto control_port = GraphView::InputPort(node, Graph::kControlSlot);
  return !graph_view.GetFanin(control_port).empty();
}

bool HasControlFanout(const GraphView& graph_view, const NodeDef* node) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTcc mht_6(mht_6_v, 269, "", "./tensorflow/core/grappler/graph_view.cc", "HasControlFanout");

  const auto control_port = GraphView::OutputPort(node, Graph::kControlSlot);
  return !graph_view.GetFanout(control_port).empty();
}

bool HasControlFaninOrFanout(const GraphView& graph_view, const NodeDef* node) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_viewDTcc mht_7(mht_7_v, 277, "", "./tensorflow/core/grappler/graph_view.cc", "HasControlFaninOrFanout");

  return HasControlFanin(graph_view, node) ||
         HasControlFanout(graph_view, node);
}

}  // end namespace grappler
}  // end namespace tensorflow
