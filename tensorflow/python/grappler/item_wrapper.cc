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
class MHTracer_DTPStensorflowPSpythonPSgrapplerPSitem_wrapperDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSgrapplerPSitem_wrapperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSgrapplerPSitem_wrapperDTcc() {
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

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = pybind11;

class ColocationGroups {
 public:
  void Group(const std::string& x, const std::string& y) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("x: \"" + x + "\"");
   mht_0_v.push_back("y: \"" + y + "\"");
   MHTracer_DTPStensorflowPSpythonPSgrapplerPSitem_wrapperDTcc mht_0(mht_0_v, 211, "", "./tensorflow/python/grappler/item_wrapper.cc", "Group");

    Rep* x_root = Find(x);
    Rep* y_root = Find(y);

    // x and y are already in the same set
    if (x_root == y_root) {
      return;
    }
    // x and y are not in same set, so we merge them
    // Use the occasion to strengthen what we know about the handle by merging
    // the information about the 2 subsets.
    if (x_root->rank < y_root->rank) {
      x_root->parent = y_root;
    } else if (x_root->rank > y_root->rank) {
      y_root->parent = x_root;
    } else {
      // Arbitrarily make one root the new parent
      y_root->parent = x_root;
      x_root->rank = x_root->rank + 1;
    }
  }

  void ExtractGroups(std::vector<std::vector<std::string>>* groups) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSgrapplerPSitem_wrapperDTcc mht_1(mht_1_v, 236, "", "./tensorflow/python/grappler/item_wrapper.cc", "ExtractGroups");

    groups->reserve(nodes_.size());
    std::unordered_map<const Rep*, int> group_ids;
    for (const auto& rep : nodes_) {
      Rep* r = Find(rep.first);
      auto it = group_ids.find(r);
      std::vector<std::string>* g;
      if (it == group_ids.end()) {
        int id = group_ids.size();
        group_ids[r] = id;
        groups->resize(id + 1);
        g = &groups->back();
      } else {
        int id = it->second;
        g = &((*groups)[id]);
      }
      g->push_back(rep.first);
    }
  }

 private:
  struct Rep {
    // Parent in the tree used to encode the set.
    Rep* parent;
    // Rank in the tree, used to figure out how to compress the path to the root
    // of the tree.
    int rank;
    // The node.
    std::string value;
  };

  Rep* Find(const std::string& n) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("n: \"" + n + "\"");
   MHTracer_DTPStensorflowPSpythonPSgrapplerPSitem_wrapperDTcc mht_2(mht_2_v, 271, "", "./tensorflow/python/grappler/item_wrapper.cc", "Find");

    auto it = nodes_.find(n);
    if (it == nodes_.end()) {
      // This is the first time we process this handle, create an entry for it.
      Rep* node = new Rep;
      node->parent = node;
      node->rank = 0;
      node->value = n;
      nodes_[n] = node;
      return node;
    }
    // Return the representative for the set, which is the root of the tree.
    // Apply path compression to speedup future queries.
    Rep* node = it->second;
    Rep* root = node->parent;
    while (root != root->parent) {
      root = root->parent;
    }
    while (node->parent != root) {
      Rep* next = node->parent;
      node->parent = root;
      node = next;
    }
    return root;
  }

  std::unordered_map<std::string, Rep*> nodes_;
};

PYBIND11_MAKE_OPAQUE(tensorflow::grappler::GrapplerItem);

PYBIND11_MODULE(_pywrap_tf_item, m) {
  py::class_<tensorflow::grappler::GrapplerItem> grappler_item(
      m, "tensorflow::grappler::GrapplerItem");

  m.def("TF_NewItem",
        [](const py::bytes& serialized_metagraph, bool ignore_colocation,
           bool ignore_user_placement) -> tensorflow::grappler::GrapplerItem* {
          tensorflow::MetaGraphDef metagraph;
          if (!metagraph.ParseFromString(std::string(serialized_metagraph))) {
            throw std::invalid_argument(
                "The MetaGraphDef could not be parsed as a valid protocol "
                "buffer");
          }
          if (metagraph.collection_def().count("train_op") == 0) {
            MaybeRaiseRegisteredFromStatus(tensorflow::errors::InvalidArgument(
                "train_op not specified in the metagraph"));
          }

          tensorflow::grappler::ItemConfig cfg;
          cfg.ignore_user_placement = ignore_user_placement;
          cfg.ignore_colocation = ignore_colocation;
          std::unique_ptr<tensorflow::grappler::GrapplerItem> item =
              tensorflow::grappler::GrapplerItemFromMetaGraphDef(
                  "item", metagraph, cfg);
          if (item == nullptr) {
            MaybeRaiseRegisteredFromStatus(
                tensorflow::errors::InvalidArgument("Invalid metagraph"));
          }
          return item.release();
        });

  m.def("TF_IdentifyImportantOps",
        [](tensorflow::grappler::GrapplerItem* item,
           bool sort_topologically) -> std::vector<std::string> {
          std::vector<const tensorflow::NodeDef*> main_ops =
              item->MainOpsFanin();
          std::vector<const tensorflow::NodeDef*> enqueue_ops =
              item->EnqueueOpsFanin();
          std::unordered_set<std::string> op_names;
          for (auto op : main_ops) {
            op_names.insert(op->name());
          }
          for (auto op : enqueue_ops) {
            op_names.insert(op->name());
          }

          std::vector<std::string> ops;
          if (sort_topologically) {
            tensorflow::GraphDef subgraph;
            for (const tensorflow::NodeDef& node : item->graph.node()) {
              if (op_names.find(node.name()) != op_names.end()) {
                *subgraph.add_node() = node;
              }
            }
            tensorflow::MaybeRaiseFromStatus(
                tensorflow::grappler::TopologicalSort(&subgraph));
            for (const tensorflow::NodeDef& node : subgraph.node()) {
              ops.push_back(node.name());
            }
          } else {
            for (const auto& op_name : op_names) {
              ops.push_back(op_name);
            }
          }
          return ops;
        });

  m.def("TF_GetOpProperties",
        [](tensorflow::grappler::GrapplerItem* item)
            -> std::unordered_map<std::string, std::vector<py::bytes>> {
          tensorflow::grappler::GraphProperties properties(*item);
          tensorflow::MaybeRaiseFromStatus(properties.InferStatically(false));

          std::unordered_map<std::string, std::vector<py::bytes>> props;
          for (const auto& node : item->graph.node()) {
            const std::string& node_name = node.name();
            const std::vector<tensorflow::OpInfo::TensorProperties>&
                output_props = properties.GetOutputProperties(node_name);

            std::vector<py::bytes> prop;
            prop.reserve(output_props.size());
            for (const auto& output_prop : output_props) {
              prop.push_back(output_prop.SerializeAsString());
            }
            props[node_name] = prop;
          }
          return props;
        });

  m.def("TF_GetColocationGroups",
        [](tensorflow::grappler::GrapplerItem* item)
            -> std::vector<std::vector<std::string>> {
          ColocationGroups groupings;
          tensorflow::OpRegistry* registry = tensorflow::OpRegistry::Global();
          for (const auto& node : item->graph.node()) {
            const tensorflow::OpDef* op_def;
            if (!registry->LookUpOpDef(node.op(), &op_def).ok()) {
              continue;
            }
            tensorflow::NameRangeMap inputs;
            tensorflow::NameRangeMap outputs;
            if (!tensorflow::NameRangesForNode(node, *op_def, &inputs, &outputs)
                     .ok()) {
              continue;
            }
            for (const auto& arg : op_def->input_arg()) {
              if (!arg.is_ref()) {
                continue;
              }
              const auto& range = inputs[arg.name()];
              for (int i = range.first; i < range.second; ++i) {
                groupings.Group(node.name(),
                                tensorflow::grappler::NodeName(node.input(i)));
              }
            }
          }

          std::vector<std::vector<std::string>> groups;
          groupings.ExtractGroups(&groups);
          return groups;
        });
}
