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
#ifndef TENSORFLOW_LITE_TOCO_TENSORFLOW_GRAPH_MATCHING_CLUSTER_H_
#define TENSORFLOW_LITE_TOCO_TENSORFLOW_GRAPH_MATCHING_CLUSTER_H_
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
class MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSclusterDTh {
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
   MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSclusterDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSclusterDTh() {
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
#include <vector>

#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tensorflow_graph_matching/cluster_utils.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"

namespace toco {

// The base class for Cluster. A cluster is group of nodes all related to each
// other because their name match a given "pattern", which shows they all belong
// to a composite op supported in TFLite. The nodes in a cluster will be
// collapsed into a single composite op node plus a series of constant nodes
// holding the input parameters to that node. The nodes in a cluster are assumed
// to be using the same device. By changing the "pattern" we can have different
// subclasses of the base Cluster class.
class Cluster {
 public:
  virtual ~Cluster() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSclusterDTh mht_0(mht_0_v, 210, "", "./tensorflow/lite/toco/tensorflow_graph_matching/cluster.h", "~Cluster");
}

  virtual void CreateNodes() = 0;

  // Save the following info from the original GraphDef this cluster is from:
  // 1- a pointer to the GraphDef
  // 2- All the nodes in GraphDef which belong to this cluster.
  void SetGraphDefInfo(const tensorflow::GraphDef* graph_def);

  const std::string& GetName() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSclusterDTh mht_1(mht_1_v, 222, "", "./tensorflow/lite/toco/tensorflow_graph_matching/cluster.h", "GetName");
 return name_; }

  const std::vector<std::unique_ptr<tensorflow::NodeDef>>& GetNewNodes() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSclusterDTh mht_2(mht_2_v, 227, "", "./tensorflow/lite/toco/tensorflow_graph_matching/cluster.h", "GetNewNodes");

    return new_nodes_;
  }

  const std::vector<const tensorflow::NodeDef*>& GetNodes() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSclusterDTh mht_3(mht_3_v, 234, "", "./tensorflow/lite/toco/tensorflow_graph_matching/cluster.h", "GetNodes");
 return nodes_; }

  void SetName(const std::string& name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSclusterDTh mht_4(mht_4_v, 240, "", "./tensorflow/lite/toco/tensorflow_graph_matching/cluster.h", "SetName");
 name_ = name; }

  void SetDevice(const std::string& device) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSclusterDTh mht_5(mht_5_v, 246, "", "./tensorflow/lite/toco/tensorflow_graph_matching/cluster.h", "SetDevice");
 device_ = device; }

  // Find the input(s) and output(s) of this Cluster.
  bool FindClusterInputsAndOutputs();

 protected:
  std::string name_;
  std::string device_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;

  // Used to hold the pointers to nodes which are in this cluster. These nodes
  // are pointing to the nodes in graph_def_.
  std::vector<const tensorflow::NodeDef*> nodes_;

  // Used to cache the newly generated nodes: like the nodes created by
  // collapsing Const nodes, or the nodes which is used to show the composite
  // op.
  std::vector<std::unique_ptr<tensorflow::NodeDef>> new_nodes_;

  const tensorflow::GraphDef* graph_def_; /*Not owned*/
};

// A factory interface for cluster class.
// It defines a virtual function interface which is responsible for creating
// a cluster. Each cluster factory is responsible to pack a cluster of nodes
// into a cluster using a name-based pattern matching approach.
class ClusterFactoryInterface {
 public:
  virtual ~ClusterFactoryInterface() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStocoPStensorflow_graph_matchingPSclusterDTh mht_6(mht_6_v, 278, "", "./tensorflow/lite/toco/tensorflow_graph_matching/cluster.h", "~ClusterFactoryInterface");
}

  // Creates a cluster of nodes using a name-based pattern matching approach. It
  // uses a node as a seed and if its name matches a certain pattern, then it
  // builds the cluster around that node.
  virtual std::unique_ptr<Cluster> CreateCluster(
      const tensorflow::NodeDef& node,
      const tensorflow::GraphDef& graph_def) const = 0;
};

}  // end namespace toco

#endif  // TENSORFLOW_LITE_TOCO_TENSORFLOW_GRAPH_MATCHING_CLUSTER_H_
