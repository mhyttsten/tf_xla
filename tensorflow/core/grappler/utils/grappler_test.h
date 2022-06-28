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

#ifndef TENSORFLOW_CORE_GRAPPLER_UTILS_GRAPPLER_TEST_H_
#define TENSORFLOW_CORE_GRAPPLER_UTILS_GRAPPLER_TEST_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTh() {
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


#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace grappler {

class GrapplerTest : public ::testing::Test {
 public:
  GrapplerTest();

 protected:
  void DisableAllOptimizers();
  void EnableAllOptimizers();

  std::vector<Tensor> EvaluateNodes(
      const GraphDef& graph, const std::vector<string>& node_names) const;

  std::vector<Tensor> EvaluateNodes(
      const GraphDef& graph, const std::vector<string>& node_names,
      const std::vector<std::pair<string, Tensor>>& inputs) const;

  std::vector<Tensor> EvaluateFetchNodes(const GrapplerItem& item) const;

  NodeDef* AddNode(const string& name, const string& op,
                   const std::vector<string>& inputs,
                   const std::vector<std::pair<string, AttrValue>>& attributes,
                   GraphDef* graph) const;

  void DisableAllOptimizers(RewriterConfig* cfg);

  // Checks if two graphs are equal. Both graphs must have the same set of nodes
  // with the same inputs and attributes. Nodes can be in different order.
  //
  // NOTE: This function uses EXPECT/ASSERT macros to check node properties
  // equality, and adds all failures to the current test.
  void CompareGraphs(GraphDef want, GraphDef got) const;

  // Checks if two nodes have the same name, op, inputs and attributes.
  //
  // NOTE: This function uses EXPECT/ASSERT macros to check node properties
  // equality, and adds all failures to the current test.
  void CompareNodes(const NodeDef& want, const NodeDef& got) const;

  // Checks if two functions are equal. Both functions must have the same set of
  // nodes with the same inputs and attributes. Nodes can be in different order.
  //
  // NOTE: This function uses EXPECT/ASSERT macros to check node properties
  // equality, and adds all failures to the current test.
  void CompareFunctions(FunctionDef want, FunctionDef got) const;

  // Checks if node 'src' is directly connected to the input($position) of
  // 'dst'.
  bool IsNodesDirectlyConnected(const NodeMap& node_map, const string& src,
                                const string& dst, int position = 0);

  // Counts nodes of the given op-type in a graph.
  int CountOpNodes(const GraphDef& graph, const string& op);

  // Get a random tensor with given shape.
  template <DataType DTYPE>
  Tensor GenerateRandomTensor(const TensorShape& shape) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTh mht_0(mht_0_v, 258, "", "./tensorflow/core/grappler/utils/grappler_test.h", "GenerateRandomTensor");

    typedef typename EnumToDataType<DTYPE>::Type T;
    Tensor tensor(DTYPE, shape);
    for (auto i = 0; i < tensor.NumElements(); i++)
      tensor.flat<T>()(i) = i + random::New64() % 10;
    return tensor;
  }

  // Creates a random tensor with given shape using `setRandom`.
  template <DataType DTYPE>
  Tensor GenerateTensorWithSetRandom(const TensorShape& shape) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTh mht_1(mht_1_v, 271, "", "./tensorflow/core/grappler/utils/grappler_test.h", "GenerateTensorWithSetRandom");

    typedef typename EnumToDataType<DTYPE>::Type T;
    Tensor tensor(DTYPE, shape);
    tensor.flat<T>().setRandom();
    return tensor;
  }

  // Get a constant tensor with given shape.
  template <DataType DTYPE>
  Tensor GenerateConstantTensor(
      const TensorShape& shape,
      typename EnumToDataType<DTYPE>::Type value) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTh mht_2(mht_2_v, 285, "", "./tensorflow/core/grappler/utils/grappler_test.h", "GenerateConstantTensor");

    typedef typename EnumToDataType<DTYPE>::Type T;
    Tensor tensor(DTYPE, shape);
    for (auto i = 0; i < tensor.NumElements(); i++) tensor.flat<T>()(i) = value;
    return tensor;
  }

  inline tensorflow::Scope CreateScopeWithDevice(absl::string_view device) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("device: \"" + std::string(device.data(), device.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTh mht_3(mht_3_v, 296, "", "./tensorflow/core/grappler/utils/grappler_test.h", "CreateScopeWithDevice");

    return tensorflow::Scope::NewRootScope().WithDevice(string(device));
  }

 private:
  SessionOptions options_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_UTILS_GRAPPLER_TEST_H_
