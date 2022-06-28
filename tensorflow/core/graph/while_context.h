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

#ifndef TENSORFLOW_CORE_GRAPH_WHILE_CONTEXT_H_
#define TENSORFLOW_CORE_GRAPH_WHILE_CONTEXT_H_
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
class MHTracer_DTPStensorflowPScorePSgraphPSwhile_contextDTh {
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
   MHTracer_DTPStensorflowPScorePSgraphPSwhile_contextDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPSwhile_contextDTh() {
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


#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

// Information about a while loop. Every user-defined while loop has an
// associated WhileContext, i.e., there is a WhileContext for every execution
// frame. Created with the while loop and used during gradient
// construction. Note that the gradient graph of while loop contains while loops
// itself, but these do not generate separate WhileContexts.
//
// TODO(skyewm): this is currently insufficient to handle nested loops and
// conditionals (and possibly other requirements). This may change a lot in the
// future to support these features.
//
// TODO(skyewm): de/serialize in MetaGraphDef so imported while loops will be
// differentiable. Figure out backwards compatibility story.
class WhileContext {
 public:
  WhileContext(StringPiece frame_name, std::vector<Node*> enter_nodes,
               std::vector<Node*> exit_nodes, OutputTensor cond_output,
               std::vector<OutputTensor> body_inputs,
               std::vector<OutputTensor> body_outputs);

  const string& frame_name() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPSwhile_contextDTh mht_0(mht_0_v, 211, "", "./tensorflow/core/graph/while_context.h", "frame_name");
 return frame_name_; }
  const std::vector<Node*>& enter_nodes() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPSwhile_contextDTh mht_1(mht_1_v, 215, "", "./tensorflow/core/graph/while_context.h", "enter_nodes");
 return enter_nodes_; }
  const std::vector<Node*>& exit_nodes() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgraphPSwhile_contextDTh mht_2(mht_2_v, 219, "", "./tensorflow/core/graph/while_context.h", "exit_nodes");
 return exit_nodes_; }
  const OutputTensor& cond_output() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgraphPSwhile_contextDTh mht_3(mht_3_v, 223, "", "./tensorflow/core/graph/while_context.h", "cond_output");
 return cond_output_; }
  const std::vector<OutputTensor>& body_inputs() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgraphPSwhile_contextDTh mht_4(mht_4_v, 227, "", "./tensorflow/core/graph/while_context.h", "body_inputs");
 return body_inputs_; }
  const std::vector<OutputTensor>& body_outputs() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgraphPSwhile_contextDTh mht_5(mht_5_v, 231, "", "./tensorflow/core/graph/while_context.h", "body_outputs");

    return body_outputs_;
  }

 private:
  // Each user-defined while loop defines a new execution frame, which is
  // uniquely identified by its frame name. Frames are used by the executor to
  // manage the iterations of a loop. See the FrameState comment in
  // core/common_runtime/executor.cc for more details.
  const string frame_name_;

  // The enter nodes defining the input loop variables to the while loop. This
  // vector defines the order of the loop variables.
  const std::vector<Node*> enter_nodes_;

  // The exit nodes defining the outputs of the while loop. These are in loop
  // variable order.
  const std::vector<Node*> exit_nodes_;

  // The boolean output of the loop predicate.
  const OutputTensor cond_output_;

  // The inputs and outputs to the loop body.
  const std::vector<OutputTensor> body_inputs_;
  const std::vector<OutputTensor> body_outputs_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_WHILE_CONTEXT_H_
