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

#ifndef TENSORFLOW_CORE_GRAPH_GRAPH_DEF_BUILDER_H_
#define TENSORFLOW_CORE_GRAPH_GRAPH_DEF_BUILDER_H_
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
class MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTh {
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
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTh() {
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

#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

// Given a function like:
//   namespace ops {
//   Node* Identity(NodeOut input, const GraphDefBuilder::Options& opts) {
//     if (opts.HaveError()) return nullptr;
//     static const string kOpName = "Identity";
//     NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
//                              opts.op_registry());
//     node_builder.Input(input);
//     return opts.FinalizeBuilder(&node_builder);
//   }
//   }  // namespace ops
//
//   // Or, alternatively:
//   namespace ops {
//   Node* Identity(NodeOut input, const GraphDefBuilder::Options& opts) {
//     static const string kOpName = "Identity";
//     return UnaryOp(kOpName, input, opts);
//   }
//   }  // namespace ops
//
// You call it like:
//   GraphDefBuilder b;
//   using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
//   Node* na = Const(7, b.opts());
//   // Note: WithName() returns a copy, opts is unchanged.
//   Node* nb = Const(5, b.opts().WithName("control-input"));
//   Node* nc = Identity(na, b.opts().WithControlInput(nb));
//   GraphDef graph_def;
//   Status status = b.ToGraphDef(&graph_def);
//   if (!status.ok()) { /* Handle error */ }
//
// In tests you can skip the status handling via:
//   GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
//   ...
//   b.ToGraphDef(&graph_def);

class GraphDefBuilder {
 public:
  // Options for adding a Node to a Graph.
  class Options {
   public:
    // Sets the Graph (that Nodes will be added to) and the status.  The
    // status may be set to nullptr, in which case errors cause CHECK
    // failures.  The graph and status must outlive *this.
    Options(Graph* graph, Status* status);
    ~Options();

    // Methods for setting options.  These are const methods: they
    // return a copy of *this with the option set.
    Options WithName(StringPiece name) const;
    Options WithDevice(StringPiece device) const;
    Options WithControlInput(Node* control_input) const;
    Options WithControlInputs(gtl::ArraySlice<Node*> control_inputs) const;

    // Override the default value for an optional attr.
    template <class T>
    Options WithAttr(StringPiece attr_name, T&& value) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTh mht_0(mht_0_v, 257, "", "./tensorflow/core/graph/graph_def_builder.h", "WithAttr");

      return Options(*this).WithAttrImpl(attr_name, std::forward<T>(value));
    }
    // Note: overload needed to allow {...} expressions for value.
    template <class T>
    Options WithAttr(StringPiece attr_name,
                     std::initializer_list<T> value) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTh mht_1(mht_1_v, 266, "", "./tensorflow/core/graph/graph_def_builder.h", "WithAttr");

      return WithAttr<std::initializer_list<T>>(attr_name, std::move(value));
    }

    // Methods for using options from a function that creates a Node.

    // Returns true if the status associated with *this has an error.
    // Use this to skip processing that may depend on prior results.
    bool HaveError() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTh mht_2(mht_2_v, 277, "", "./tensorflow/core/graph/graph_def_builder.h", "HaveError");
 return status_ != nullptr && !status_->ok(); }

    // Returns a string representation of the status associated with *this.
    // Returns the string `"OK"` if the status doesn't have any error.
    string StatusToString() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTh mht_3(mht_3_v, 284, "", "./tensorflow/core/graph/graph_def_builder.h", "StatusToString");

      return status_->ok() ? "OK" : status_->error_message();
    }

    // Given the Op type name, return a name for a node of that type.
    // Uses the value set in WithName() if that has been called.  Otherwise,
    // returns a name built out of the Op type name.
    string GetNameForOp(StringPiece op) const;

    // Sets the device, adds control inputs, adds attrs, and calls Finalize().
    // If Finalize returns an error, it is saved and this function returns
    // nullptr.
    Node* FinalizeBuilder(NodeBuilder* builder) const;

    // Updates the associated status, if any, or calls TF_CHECK_OK if none.
    void UpdateStatus(const Status& status) const;

    // Accessor
    const OpRegistryInterface* op_registry() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTh mht_4(mht_4_v, 305, "", "./tensorflow/core/graph/graph_def_builder.h", "op_registry");

      return graph_->op_registry();
    }

   private:
    Options WithNameImpl(StringPiece name);
    Options WithDeviceImpl(StringPiece device);
    Options WithControlInputImpl(Node* control_input);
    Options WithControlInputsImpl(gtl::ArraySlice<Node*> control_inputs);
    template <class T>
    Options WithAttrImpl(StringPiece name, T&& value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTh mht_5(mht_5_v, 318, "", "./tensorflow/core/graph/graph_def_builder.h", "WithAttrImpl");

      attrs_.emplace_back(string(name), AttrValue());
      SetAttrValue(std::forward<T>(value), &attrs_.back().second);
      return *this;
    }

    Graph* const graph_;
    Status* const status_;
    string name_;
    string device_;
    std::vector<Node*> control_inputs_;
    std::vector<std::pair<string, AttrValue>> attrs_;
  };

  // Start building a new graph.
  explicit GraphDefBuilder(
      const OpRegistryInterface* op_registry = OpRegistry::Global())
      : graph_(op_registry), flib_def_(op_registry), opts_(&graph_, &status_) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTh mht_6(mht_6_v, 338, "", "./tensorflow/core/graph/graph_def_builder.h", "GraphDefBuilder");
}

  // For use in tests, where you want to fail immediately on error instead
  // of checking the status at the end.
  enum TestFailImmediatelyType { kFailImmediately };
  explicit GraphDefBuilder(
      TestFailImmediatelyType,
      const OpRegistryInterface* op_registry = OpRegistry::Global())
      : graph_(op_registry), flib_def_(op_registry), opts_(&graph_, nullptr) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTh mht_7(mht_7_v, 349, "", "./tensorflow/core/graph/graph_def_builder.h", "GraphDefBuilder");
}

  // Gets the Options with the associated Graph and Status.
  const Options& opts() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTh mht_8(mht_8_v, 355, "", "./tensorflow/core/graph/graph_def_builder.h", "opts");
 return opts_; }

  // Once all the nodes have been added, call this to get whether it was
  // successful, and if so fill *graph_def.
  Status ToGraphDef(GraphDef* graph_def) const;

  // Adds the function and gradient definitions in `fdef_lib` to this graph's op
  // registry. Ignores duplicate functions, and returns a bad status if an
  // imported function differs from an existing function or op with the same
  // name.
  Status AddFunctionLibrary(const FunctionDefLibrary& fdef_lib) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTh mht_9(mht_9_v, 368, "", "./tensorflow/core/graph/graph_def_builder.h", "AddFunctionLibrary");

    return flib_def_.AddLibrary(fdef_lib);
  }

  // Returns whether a user-defined function with `name` already exists in the
  // graph.
  bool HasFunction(const string& name) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_def_builderDTh mht_10(mht_10_v, 378, "", "./tensorflow/core/graph/graph_def_builder.h", "HasFunction");

    return flib_def_.Find(name) != nullptr;
  }

 private:
  Graph graph_;
  FunctionLibraryDefinition flib_def_;
  Status status_;
  Options opts_;
};

namespace ops {

// A NodeOut may either be a regular input or back input.  Regular
// inputs are specified via either a Node* or a Node* and an output
// index.  Back inputs are specified by a node name, output index, and
// output type.
typedef NodeBuilder::NodeOut NodeOut;

// For adding an Op with no inputs to a GraphDefBuilder.
Node* SourceOp(const string& op_name, const GraphDefBuilder::Options& opts);

// For adding an Op with one input to a GraphDefBuilder.
Node* UnaryOp(const string& op_name, NodeOut input,
              const GraphDefBuilder::Options& opts);

// For adding an Op with two inputs to a GraphDefBuilder.
Node* BinaryOp(const string& op_name, NodeOut a, NodeOut b,
               const GraphDefBuilder::Options& opts);

// For adding an Op with three inputs to a GraphDefBuilder.
Node* TernaryOp(const string& op_name, NodeOut a, NodeOut b, NodeOut c,
                const GraphDefBuilder::Options& opts);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_GRAPH_DEF_BUILDER_H_
