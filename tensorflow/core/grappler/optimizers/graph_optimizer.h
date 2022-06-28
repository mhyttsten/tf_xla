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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GRAPH_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GRAPH_OPTIMIZER_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizerDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizerDTh() {
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


#include <string>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace grappler {

class Cluster;
struct GrapplerItem;

// An abstract interface for an algorithm for generating a candidate
// optimization of a GrapplerItem for running on a cluster.
class GraphOptimizer {
 public:
  GraphOptimizer() : deadline_usec_(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizerDTh mht_0(mht_0_v, 206, "", "./tensorflow/core/grappler/optimizers/graph_optimizer.h", "GraphOptimizer");
}
  virtual ~GraphOptimizer() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizerDTh mht_1(mht_1_v, 210, "", "./tensorflow/core/grappler/optimizers/graph_optimizer.h", "~GraphOptimizer");
}

  virtual string name() const = 0;

  // Returns true if the optimizer requires a valid function library to perform
  // graph optimization. If false, optimized GrapplerItem will have a stub
  // instead of real function library (all function signatures and attributes
  // will be valid, but function body will be empty). Most of the optimizers
  // that do not instantiate functions should return true.
  virtual bool UsesFunctionLibrary() const = 0;

  // Routine called to allow an algorithm to propose a rewritten graph
  // for the graph, feeds and fetches in "item" to run more efficiently
  // on "cluster". If the returned status is Status::OK() then
  // *optimized_graph contains the rewritten graph.
  // Returns an error status if it failed to generate a solution.
  //
  // A return value of error::Aborted() can be used signal early termination of
  // the optimizer, e.g. if the optimization turned out to be a no-op. In this
  // case the content of *optimized_graph is undefined.
  virtual Status Optimize(Cluster* cluster, const GrapplerItem& item,
                          GraphDef* optimized_graph) = 0;

  // Subclasses may define a version of Optimize that consumes item.
  virtual Status Optimize(Cluster* cluster, GrapplerItem&& item,
                          GraphDef* optimized_graph) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizerDTh mht_2(mht_2_v, 238, "", "./tensorflow/core/grappler/optimizers/graph_optimizer.h", "Optimize");

    return Optimize(cluster, item, optimized_graph);
  }

  // Set deadline in microseconds since epoch. A value of zero means no
  // deadline.
  void set_deadline_usec(uint64 deadline_usec) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizerDTh mht_3(mht_3_v, 247, "", "./tensorflow/core/grappler/optimizers/graph_optimizer.h", "set_deadline_usec");

    deadline_usec_ = deadline_usec;
  }
  uint64 deadline_usec() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizerDTh mht_4(mht_4_v, 253, "", "./tensorflow/core/grappler/optimizers/graph_optimizer.h", "deadline_usec");
 return deadline_usec_; }
  bool DeadlineExceeded() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgraph_optimizerDTh mht_5(mht_5_v, 257, "", "./tensorflow/core/grappler/optimizers/graph_optimizer.h", "DeadlineExceeded");

    return deadline_usec_ > 0 && Env::Default()->NowMicros() > deadline_usec_;
  }

 private:
  uint64 deadline_usec_;
};

#define GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED()                              \
  do {                                                                      \
    if (this->DeadlineExceeded()) {                                         \
      return errors::DeadlineExceeded(this->name(), " exceeded deadline."); \
    }                                                                       \
  } while (0)

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GRAPH_OPTIMIZER_H_
