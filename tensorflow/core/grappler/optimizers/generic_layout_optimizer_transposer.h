/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GENERIC_LAYOUT_OPTIMIZER_TRANSPOSER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GENERIC_LAYOUT_OPTIMIZER_TRANSPOSER_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh() {
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


#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/costs/virtual_placer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {

constexpr char kAttrSrcFormat[] = "src_format";
constexpr char kAttrDstFormat[] = "dst_format";
constexpr char kAttrOutputShape[] = "_output_shapes";
constexpr char kGPU[] = "GPU";
constexpr char kCPU[] = "CPU";

// TransposeContext owns all data members. Must initialize GraphProperties,
// FrameView, GraphDef and MutableGraphView with the same graph. NodeDef
// pointers in FrameView, GraphDef and MutableGraphView must point to nodes in
// the same GraphDef instance.
struct TransposeContext {
  // Initializes TransposeContext with given GrapplerItem. Because initializing
  // FrameMap and GraphProperties may return error, we initialize
  // TransposeContext outside constructor.
  static Status InitializeTransposeContext(bool assume_valid_feeds,
                                           const GrapplerItem& item,
                                           const Cluster* cluster,
                                           TransposeContext* context);

  static Status InitializeTransposeContext(const GrapplerItem& item,
                                           const Cluster* cluster,
                                           TransposeContext* context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_0(mht_0_v, 230, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "InitializeTransposeContext");

    return InitializeTransposeContext(false, item, cluster, context);
  }

  // Sets data formats to convert from and to for specified device type.
  void AssignDeviceAndDataFormats(absl::string_view target_device,
                                  absl::string_view src_format,
                                  absl::string_view dst_format);

  FrameView frames;
  GraphDef graph;
  // Number of nodes in the original graph. As new nodes are appended to the end
  // of the graph, all new nodes should have a node index greater than or equal
  // to this.
  int num_nodes;
  absl::flat_hash_set<string> nodes_to_preserve;
  std::unique_ptr<GraphProperties> graph_properties;
  std::unique_ptr<utils::MutableGraphView> graph_view;
  std::unique_ptr<const VirtualPlacer> virtual_placer;

  string target_device;
  string src_format;
  string dst_format;
  absl::flat_hash_map<char, int> src_dim_indices;
  absl::flat_hash_map<char, int> dst_dim_indices;
  std::vector<int> src_to_dst;
  std::vector<int> dst_to_src;

  string enforced_layout;
};

class Transposer {
 public:
  explicit Transposer() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_1(mht_1_v, 266, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "Transposer");
}

  Transposer(const Transposer&) = delete;
  Transposer& operator=(const Transposer&) = delete;

  virtual ~Transposer() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_2(mht_2_v, 274, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "~Transposer");
}

  // Returns true iff the node should be processed by this transposer.
  // NodeProcessors may perform additional oprand specific checks before
  // processing if necessary.
  // Following common conditions are checked:
  // * node's device matches target device
  // * node's source format matches config's source format
  // * node has output
  bool ShouldProcess(const TransposeContext& context,
                     const utils::MutableNodeView& node) const;

  // Transposes given node from src format to dst format. Also perform other
  // necessary operations to guarantee the graph produce the same result.
  // Eg. Add Transpose node sets before fanin ports and after fanout ports.
  virtual Status TransposeNode(TransposeContext* context,
                               utils::MutableNodeView* node) = 0;

  // Creates a Const node for permutation. If node with node_name already exits,
  // return and reuse it.
  Status CreateConstPermNode(TransposeContext* context,
                             absl::string_view node_name,
                             absl::string_view device,
                             absl::Span<const int> permutation,
                             absl::string_view control_node_name,
                             utils::MutationNewNode* added_node);

  // Creates a TransposeNode with given properties. If node with node_name
  // already exits, return and reuse it.
  // A const perm node is also created and connected to the 2nd fanin.
  // control_node_name is ignored if it is empty.
  Status CreateTransposeNode(
      TransposeContext* context, absl::string_view name_format,
      const DataType& data_type, absl::string_view device,
      TensorShapeProto fanin_shape, absl::Span<const int> permutation,
      absl::string_view control_node_name, utils::MutationNewNode* added_node,
      string* transpose_node_name);

  // Update all edges between dst_node->fanin[dst_ports] and dst_node by
  // inserting an op node.
  Status UpdateFaninEdgesWithOp(TransposeContext* context,
                                absl::Span<const int> dst_ports,
                                utils::MutableNodeView* dst_node,
                                absl::string_view op);

  // Update all edges between src_node:src_ports and nodes take
  // src_node:src_ports as fanin. Also update attr _output_shape of src_node.
  Status UpdateFanoutEdgesWithOp(TransposeContext* context,
                                 absl::Span<const int> src_ports,
                                 utils::MutableNodeView* src_node,
                                 absl::string_view op);

  // Creates a DataFromat node with given properties.
  // DataFromat op is either DataFormatVecPermute or DataFormatDimMap.
  Status CreateDataFormatNode(TransposeContext* context,
                              absl::string_view node_name, absl::string_view op,
                              absl::string_view device,
                              const DataType& data_type, bool is_fanin_on_host,
                              bool is_src_format_to_dst_format,
                              utils::MutationNewNode* added_node);

 protected:
  int GetFanoutPortRank(const utils::MutableNodeView& node, int port) const;
  bool IsFanoutPortRankN(const utils::MutableNodeView& node, int port,
                         int n) const;
  bool IsFanoutPortsRankN(const utils::MutableNodeView& node,
                          absl::Span<const int> ports, int n) const;
  int GetFaninPortRank(const utils::MutableNodeView& node, int port) const;
  bool IsFaninPortRankN(const utils::MutableNodeView& node, int port,
                        int n) const;

  // Checks if fanin at specified port(s) has dimensions `dims` iff fanin is a
  // Const. If fanin is not a Const, no dimensions will be checked and this will
  // return true.
  bool IsFaninPortDimsNIfConst(const utils::MutableNodeView& node, int port,
                               absl::Span<const int> dims) const;
  bool IsFaninPortsDimsNIfConst(const utils::MutableNodeView& node,
                                absl::Span<const int> ports,
                                absl::Span<const int> dims) const;
  bool CanProcessNode(const TransposeContext& context,
                      const utils::MutableNodeView& node) const;
  // Update all edges between dst_node->fanin[dst_ports] and dst_node.
  // A node with op is created and inserted between all edges.
  // op is one of Transpose, DataFormatVecPermute or DataFormatDimMap.
  Status UpdateEdge(TransposeContext* context, absl::string_view name_format,
                    absl::string_view op, const AttrValue* input_shape,
                    bool is_in_frame, bool is_src_format_to_dst_format,
                    const int src_port, const int dst_port,
                    utils::MutableNodeView* src_node,
                    utils::MutableNodeView* dst_node);
  string GetFaninNameFormat(absl::string_view node_name, int port,
                            absl::string_view src_format,
                            absl::string_view dst_format);
  string GetFanoutNameFormat(absl::string_view node_name, int port, int index,
                             absl::string_view src_format,
                             absl::string_view dst_format);
  string LayoutOptimizerNode(absl::string_view node_name);
  string GetReshapeNodeNameFormat(absl::string_view node_name, int index,
                                  absl::string_view src_format,
                                  absl::string_view dst_format);
  string GetShapeConstNodeNameFormat(absl::string_view node_name, int index);
};

class LayoutSensitiveOpTransposer : public Transposer {
 public:
  explicit LayoutSensitiveOpTransposer() : Transposer() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_3(mht_3_v, 382, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "LayoutSensitiveOpTransposer");
}

  // Updates attrs data_format, ksize, strides of the given node to dst_format.
  // _output_shape is updated during UpdateOutputEdges.
  Status UpdateNode(TransposeContext* context, utils::MutableNodeView* node);
};

// Layout sensitive op transposers.

class DefaultLayoutSensitiveOpTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit DefaultLayoutSensitiveOpTransposer()
      : LayoutSensitiveOpTransposer() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_4(mht_4_v, 397, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "DefaultLayoutSensitiveOpTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class BiasAddTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit BiasAddTransposer() : LayoutSensitiveOpTransposer() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_5(mht_5_v, 408, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "BiasAddTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class AvgPoolGradTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit AvgPoolGradTransposer() : LayoutSensitiveOpTransposer() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_6(mht_6_v, 419, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "AvgPoolGradTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class BiasAddGradTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit BiasAddGradTransposer() : LayoutSensitiveOpTransposer() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_7(mht_7_v, 430, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "BiasAddGradTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class Conv2DBackpropFilterTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit Conv2DBackpropFilterTransposer() : LayoutSensitiveOpTransposer() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_8(mht_8_v, 441, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "Conv2DBackpropFilterTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class Conv2DBackpropInputTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit Conv2DBackpropInputTransposer() : LayoutSensitiveOpTransposer() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_9(mht_9_v, 452, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "Conv2DBackpropInputTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class Conv3DTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit Conv3DTransposer() : LayoutSensitiveOpTransposer() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_10(mht_10_v, 463, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "Conv3DTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class Conv3DBackpropFilterTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit Conv3DBackpropFilterTransposer() : LayoutSensitiveOpTransposer() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_11(mht_11_v, 474, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "Conv3DBackpropFilterTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class Conv3DBackpropInputTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit Conv3DBackpropInputTransposer() : LayoutSensitiveOpTransposer() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_12(mht_12_v, 485, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "Conv3DBackpropInputTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class FusedBatchNormExTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit FusedBatchNormExTransposer() : LayoutSensitiveOpTransposer() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_13(mht_13_v, 496, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "FusedBatchNormExTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class FusedBatchNormGradTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit FusedBatchNormGradTransposer() : LayoutSensitiveOpTransposer() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_14(mht_14_v, 507, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "FusedBatchNormGradTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;

 private:
  bool IsTraining(const utils::MutableNodeView& node) const;
};

class MaxPoolV2Transposer : public LayoutSensitiveOpTransposer {
 public:
  explicit MaxPoolV2Transposer() : LayoutSensitiveOpTransposer() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_15(mht_15_v, 521, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "MaxPoolV2Transposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class MaxPoolGradTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit MaxPoolGradTransposer() : LayoutSensitiveOpTransposer() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_16(mht_16_v, 532, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "MaxPoolGradTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class MaxPoolGradV2Transposer : public LayoutSensitiveOpTransposer {
 public:
  explicit MaxPoolGradV2Transposer() : LayoutSensitiveOpTransposer() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_17(mht_17_v, 543, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "MaxPoolGradV2Transposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

// Layout agnostic op transposers.

class LayoutAgnosticOpTransposer : public Transposer {
 public:
  explicit LayoutAgnosticOpTransposer() : Transposer() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_18(mht_18_v, 556, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "LayoutAgnosticOpTransposer");
}

 protected:
  bool IsAfterDstToSrcTransform(const TransposeContext& context,
                                const utils::MutableNodeView& node) const;

  std::vector<int> GetVariadicNDFaninPorts(const TransposeContext& context,
                                           const utils::MutableNodeView& node,
                                           int rank) const;
};

class DefaultLayoutAgnosticOpTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit DefaultLayoutAgnosticOpTransposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_19(mht_19_v, 572, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "DefaultLayoutAgnosticOpTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class AddNTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit AddNTransposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_20(mht_20_v, 583, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "AddNTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class BinaryOpTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit BinaryOpTransposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_21(mht_21_v, 594, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "BinaryOpTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;

 private:
  bool IsNDOperateWithMD(const utils::MutableNodeView& node, int n, int m);
  bool IsFaninShapeSupported(const utils::MutableNodeView& node, int rank);
  std::vector<int> GetNDDataFaninPorts(const utils::MutableNodeView& node,
                                       int rank);
  Status AddNodeShapeConst(utils::Mutation* mutation,
                           absl::string_view node_name,
                           absl::string_view node_device, bool node_in_frame,
                           int num_channels, absl::string_view depended_node,
                           int rank);
  Status AddNodeReshape(utils::Mutation* mutation, absl::string_view node_name,
                        absl::string_view node_device,
                        absl::string_view input_name,
                        absl::string_view shape_const_node_name,
                        const DataType& data_type);
  Status MaybeReshapeVectorFanin(TransposeContext* context,
                                 utils::MutableNodeView* node, int rank);
};

class ConcatOpTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit ConcatOpTransposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_22(mht_22_v, 623, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "ConcatOpTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class FillOpTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit FillOpTransposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_23(mht_23_v, 634, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "FillOpTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class IdentityNTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit IdentityNTransposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_24(mht_24_v, 645, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "IdentityNTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class MergeTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit MergeTransposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_25(mht_25_v, 656, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "MergeTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;

 private:
  bool IsEveryFaninAfterDstToSrcTransform(
      const TransposeContext& context,
      const utils::MutableNodeView& node) const;
};

class PadTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit PadTransposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_26(mht_26_v, 672, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "PadTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class ReduceTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit ReduceTransposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_27(mht_27_v, 683, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "ReduceTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;

 private:
  bool KeepDims(const utils::MutableNodeView& node);
  bool IsAlongAxis(const Tensor& tensor, absl::Span<const int> axis, int rank);
  bool IsReduceAxisSupported(const TransposeContext& context,
                             const utils::MutableNodeView& node, int rank);
};

class ReverseV2Transposer : public LayoutAgnosticOpTransposer {
 public:
  explicit ReverseV2Transposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_28(mht_28_v, 700, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "ReverseV2Transposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class SelectTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit SelectTransposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_29(mht_29_v, 711, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "SelectTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;

 protected:
  bool IsFaninScalarVector4D(const utils::MutableNodeView& fanin, int port);
  std::vector<int> GetFaninPorts(const utils::MutableNodeView& fanin, int port);
};

class ShapeTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit ShapeTransposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_30(mht_30_v, 726, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "ShapeTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class ShapeNTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit ShapeNTransposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_31(mht_31_v, 737, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "ShapeNTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class SliceTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit SliceTransposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_32(mht_32_v, 748, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "SliceTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class SplitTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit SplitTransposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_33(mht_33_v, 759, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "SplitTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class SplitVTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit SplitVTransposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_34(mht_34_v, 770, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "SplitVTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class SqueezeTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit SqueezeTransposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_35(mht_35_v, 781, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "SqueezeTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;

 private:
  bool IsInputConvertible(const TransposeContext& context,
                          const utils::MutableNodeView& node) const;
  bool IsAlongAxis(const AttrValue& attr, absl::Span<const int> axis,
                   int rank) const;
  bool IsDimsSupported(const TransposeContext& context,
                       const utils::MutableNodeView& node) const;
  Status UpdateSqueezeDims(TransposeContext* context,
                           utils::MutableNodeView* node);
};

class StridedSliceTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit StridedSliceTransposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_36(mht_36_v, 802, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "StridedSliceTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;

 private:
  bool IsMaskZero(const utils::MutableNodeView& node, absl::string_view mask);
  bool HasOnlyBeginEndMask(const utils::MutableNodeView& node);
  Status PermuteMask(TransposeContext* context, utils::MutableNodeView* node,
                     absl::string_view mask);
};

class SwitchTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit SwitchTransposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_37(mht_37_v, 819, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "SwitchTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class TernaryOpTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit TernaryOpTransposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_38(mht_38_v, 830, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "TernaryOpTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class TileTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit TileTransposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_39(mht_39_v, 841, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "TileTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

class UnaryGradTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit UnaryGradTransposer() : LayoutAgnosticOpTransposer() {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_40(mht_40_v, 852, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "UnaryGradTransposer");
}

  Status TransposeNode(TransposeContext* context,
                       utils::MutableNodeView* node) override;
};

// Utils.

// Permutes elements according to permutation and replaces the original values.
// Permutation and values must have same size.
template <typename T>
Status PermuteSingle(absl::string_view location,
                     absl::Span<const int> permutation, T* values) {
   std::vector<std::string> mht_41_v;
   mht_41_v.push_back("location: \"" + std::string(location.data(), location.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_41(mht_41_v, 868, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "PermuteSingle");

  DCHECK(values != nullptr);
  int permutation_size = permutation.size();
  if (values->size() != permutation_size) {
    return Status(tensorflow::error::Code::INVALID_ARGUMENT,
                  absl::StrCat("Size of values ", values->size(),
                               " does not match size of permutation ",
                               permutation_size, " @ ", location));
  }
  typedef typename T::value_type V;
  std::vector<V> elements(values->begin(), values->end());
  int index = 0;
  for (V& element : *values) {
    element = elements[permutation[index++]];
  }
  return Status::OK();
}

// Permutes two elements at a time according to permutation and replaces the
// original values. Values must be twice the size of permutation.
template <typename T>
Status PermuteDouble(absl::string_view location,
                     absl::Span<const int> permutation, T* values) {
   std::vector<std::string> mht_42_v;
   mht_42_v.push_back("location: \"" + std::string(location.data(), location.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTh mht_42(mht_42_v, 894, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h", "PermuteDouble");

  DCHECK(values != nullptr);
  int permutation_size = permutation.size();
  if (values->size() != permutation_size * 2) {
    return Status(tensorflow::error::Code::INVALID_ARGUMENT,
                  absl::StrCat("Size of values ", values->size(),
                               " does not match twice the size of permutation ",
                               permutation_size, " @ ", location));
  }
  typedef typename T::value_type V;
  std::vector<V> elements(values->begin(), values->end());
  for (int i = 0; i < values->size(); i = i + 2) {
    const int permutation_index = permutation[i / 2];
    (*values)[i] = elements[permutation_index * 2];
    (*values)[i + 1] = elements[permutation_index * 2 + 1];
  }
  return Status::OK();
}

string GetDeviceName(const VirtualPlacer* virtual_placer, const NodeDef& node);

bool IsDefaultLayoutSensitiveOp(const NodeDef& node);

bool IsLayoutSensitiveOp(const NodeDef& node);

bool IsDefaultLayoutAgnosticOp(const NodeDef& node);

bool IsLayoutAgnosticOp(const NodeDef& node);

bool IsTernaryOp(const NodeDef& node);

bool IsUnaryGrad(const NodeDef& node);

bool IsMaxPoolV2(const NodeDef& node);

bool IsMaxPoolGradV2(const NodeDef& node);

bool IsMaxPoolGradGradV1(const NodeDef& node);

bool IsMaxPoolGradGradV2(const NodeDef& node);

bool IsBinaryOp(const NodeDef& node);

bool IsReduceOp(const NodeDef& node);

std::vector<int> GetDataFaninPorts(const utils::MutableNodeView& node);

std::vector<int> GetDataFanoutPorts(const utils::MutableNodeView& node);

// Returns a value of constant input to the `node` at `index`, iff `predicate`
// evaluated to true. Returns true if `tensor` was populated with data.
bool GetValueAttrFromConstInputNode(
    const utils::MutableNodeView& node,
    const std::function<bool(const NodeDef&)>& predicate, int index,
    Tensor* tensor);

bool IsDataFormatOp(const utils::MutableNodeView& node);

absl::flat_hash_map<char, int> GetDimensionIndices(
    absl::string_view data_format);

std::vector<int> GetPermutation(
    const absl::flat_hash_map<char, int>& src_dim_indices,
    absl::string_view dst_format);

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GENERIC_LAYOUT_OPTIMIZER_TRANSPOSER_H_
