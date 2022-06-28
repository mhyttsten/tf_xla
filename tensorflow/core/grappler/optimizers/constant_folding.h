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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CONSTANT_FOLDING_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CONSTANT_FOLDING_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSconstant_foldingDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSconstant_foldingDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSconstant_foldingDTh() {
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


#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

const char kConstantFoldingConst[] = "ConstantFolding";
const char kConstantFoldingCtrl[] = "ConstantFoldingCtrl";
extern const int64_t kMaxConstantSize;

// Constant folding optimization for a graph.
class ConstantFolding : public GraphOptimizer {
 public:
  // The size limit will only be considered if the newly created node is greater
  // than original_size (optional).
  static Status CreateNodeDef(const string& name, const TensorValue& tensor,
                              NodeDef* node, size_t original_size = 0);
  static string AddControlDependency(const string& input_name, GraphDef* graph,
                                     NodeMap* node_map);

  explicit ConstantFolding(DeviceBase* cpu_device,
                           bool disable_compressed_tensor_optimization = false,
                           bool fold_quantization_emulation = true);
  ConstantFolding(RewriterConfig::Toggle opt_level, DeviceBase* cpu_device,
                  bool disable_compressed_tensor_optimization = false,
                  bool fold_quantization_emulation = true);

  ~ConstantFolding() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSconstant_foldingDTh mht_0(mht_0_v, 224, "", "./tensorflow/core/grappler/optimizers/constant_folding.h", "~ConstantFolding");
}

  string name() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSconstant_foldingDTh mht_1(mht_1_v, 229, "", "./tensorflow/core/grappler/optimizers/constant_folding.h", "name");
 return "constant_folding"; };

  bool UsesFunctionLibrary() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSconstant_foldingDTh mht_2(mht_2_v, 234, "", "./tensorflow/core/grappler/optimizers/constant_folding.h", "UsesFunctionLibrary");
 return false; }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* output) override;

 private:
  bool ForwardInputs(NodeDef* node, absl::Span<const int> inputs_to_forward);
  string OptimizedNodeName(const NodeDef& node, StringPiece suffix) const;
  bool OptimizedNodeExists(const NodeDef& node, StringPiece suffix) const;

  bool IsReallyConstant(const NodeDef& node) const;

  bool GetTensorFromConstNode(const string& node_name_or_input, Tensor* tensor);

  Status MaterializeShapes(const GraphProperties& properties);

  Status MaterializeBroadcastGradientArgs(const NodeDef& node,
                                          const GraphProperties& properties);
  Status MaterializeReductionIndices(NodeDef* node,
                                     const GraphProperties& properties);
  Status MaterializeConstantValuedNode(NodeDef* node,
                                       const GraphProperties& properties);
  Status MaterializeOutputValues(NodeDef* node,
                                 const GraphProperties& properties);
  Status MaterializeConstants(const GraphProperties& properties);

  bool IsFoldable(const NodeDef& node, const GraphProperties* properties);
  bool IsFoldableUncached(const NodeDef& node,
                          const GraphProperties* properties) const;
  bool MaybeFoldable(const NodeDef& node,
                     const GraphProperties* properties) const;

  Status EvaluateNode(const NodeDef& node,
                      const gtl::InlinedVector<TensorValue, 4>& inputs,
                      gtl::InlinedVector<TensorValue, 4>* output) const;

  Status EvaluateOneFoldable(const NodeDef& node, std::vector<NodeDef>* outputs,
                             bool* result_too_large);

  Status FoldMergeNode(NodeDef* node, GraphDef* output_graph);
  Status FoldNode(NodeDef* node, GraphDef* output_graph,
                  bool* result_too_large);

  bool IsOnes(const NodeDef& node) const;
  bool IsZeros(const NodeDef& node) const;
  bool ReplaceOperationWithBroadcastTo(int input_to_broadcast,
                                       const GraphProperties& properties,
                                       NodeDef* node, GraphDef* graph);
  void ReplaceOperationWithIdentity(int input_to_forward,
                                    const GraphProperties& properties,
                                    NodeDef* node, GraphDef* graph);
  void ReplaceOperationWithSnapshot(int input_to_forward,
                                    const GraphProperties& properties,
                                    NodeDef* node, GraphDef* graph);
  void ReplaceOperationWithNoOp(NodeDef* node, GraphProperties* properties,
                                GraphDef* graph);
  void ReplaceBinaryOperationWithBroadcastTo(int input_to_broadcast,
                                             const GraphProperties& properties,
                                             NodeDef* node, GraphDef* graph);
  void ReplaceSubtractionFromZeroByNegation(NodeDef* node, GraphDef* graph);
  Status ReplaceOperationWithConstant(double value,
                                      const GraphProperties& properties,
                                      const TensorShapeProto& shape,
                                      NodeDef* node, GraphDef* graph);

  // Notice: Destroys *value.
  Status ReplaceOperationWithConstantTensor(DataType dtype, TensorProto* value,
                                            NodeDef* node, GraphDef* graph);

  void ReplaceDivisionOfOnesByReciprocal(NodeDef* node, GraphDef* graph);
  Status FoldGraph(const GraphProperties& properties, GraphDef* output,
                   absl::flat_hash_set<string>* nodes_to_not_simplify);

  Status IsSimplifiableReshape(const NodeDef& node,
                               const GraphProperties& properties) const;
  Status SimplifyGraph(GraphDef* optimized_graph, GraphProperties* properties,
                       absl::flat_hash_set<string>* nodes_to_not_simplify);
  Status SimplifyNode(NodeDef* node, GraphDef* optimized_graph,
                      GraphProperties* properties);

  Status RunOptimizationPass(Cluster* cluster, GrapplerItem* item,
                             GraphProperties* properties,
                             GraphDef* optimized_graph);

  // Applies partial constant folding for Concat which is not commutative.
  // Returns true if the transformation applied successfully.
  bool PartialConcatConstFolding(GraphDef* optimized_graph,
                                 GraphProperties* properties, NodeDef* node);

  // Applies partial constant folding for associative operators AddN and
  // AccumulateNV2. Returns true if the transformation applied successfully.
  bool PartialAssocOpConstFolding(GraphDef* optimized_graph,
                                  GraphProperties* properties, NodeDef* node);

  // Applies partial constant propagation through IdentityN operator.
  // Returns true if the transformation applied successfully.
  bool PartialConstPropThroughIdentityN(NodeDef* node);

  struct ConstantPushDownContext {
    NodeDef* op_child;
    NodeDef* const_child;
    bool left_child_is_const;
    bool right_child_is_const;
    NodeDef* left_leaf;
    NodeDef* right_leaf;
    bool left_leaf_is_const;
    bool right_leaf_is_const;

    // Shape & type information.
    const std::vector<OpInfo::TensorProperties>* parent_input_props;
    const std::vector<OpInfo::TensorProperties>* op_child_input_props;
  };

  // Populates ctx with pointers to the nodes in expression tree for which
  // constant pushdown optimization is being considered, corresponding to one of
  // the following configurations:
  //
  //               parent                            parent
  //               /    \                            /    \
  //        op_child   const_child            const_child op_child
  //         /     \                                       /     \
  //    left_leaf  right_leaf                        left_leaf  right_leaf
  //
  // Returns true if the expression is possible amenable for optimization.
  // Returns false if must_have_properties is true and input properties for
  // parent and op_child are not known.
  bool PrepareConstantPushDown(const NodeDef& parent,
                               const GraphProperties& properties,
                               bool must_have_properties,
                               ConstantPushDownContext* ctx) const;

  // Pushes down constants on '+', '-', '*', and '/' operators if applicable.
  // Returns true if the transformation applied successfully.
  bool ConstantPushDown(GraphProperties* properties, GraphDef* optimized_graph,
                        NodeDef* node);

  // Pushes down constants on '+' and 'BiasAdd' operators if applicable.
  // Returns true if the graph was modified.
  bool ConstantPushDownBiasAdd(GraphProperties* properties,
                               GraphDef* optimized_graph, NodeDef* node);

  // Aggregate constants present around a conv operator. Returns true if the
  // transformation was applied successfully.
  bool MulConvPushDown(GraphDef* optimized_graph, NodeDef* node,
                       const GraphProperties& properties);

  // Strength reduces floating point division by a constant Div(x, const) to
  // multiplication by the reciprocal Mul(x, Reciprocal(const)).
  bool ReduceDivToReciprocalMul(GraphDef* optimized_graph, NodeDef* node);

  // Simplifies arithmetic operations with ones or zeros. Returns the status,
  // and updates the success input argument that denotes if any simplification
  // was applied.
  Status SimplifyArithmeticOperations(const GraphProperties& properties,
                                      bool use_shape_info,
                                      GraphDef* optimized_graph, NodeDef* node);

  // Simplifies a Reshape operation to an Identity operation if applicable.
  bool SimplifyReshape(const GraphProperties& properties, bool use_shape_info,
                       NodeDef* node);

  // Returns true iff the node is a reduction and its reduction indices are
  // constant. Sets *indices_is_empty to true if the set of dimensions to reduce
  // along is empty (this happens often in the gradient graphs).
  bool IsReductionWithConstantIndices(const NodeDef& node,
                                      bool* indices_is_empty) const;
  // Returns true if theres a possibility that a Reduce node could be simplified
  // to an Identity/Reshape.
  bool IsReductionCandidateForSimplification(
      const NodeDef& node, const GraphProperties& properties,
      TensorShapeProto* input_tensor_shape,
      TensorShapeProto* output_tensor_shape, bool* is_single_element_op) const;
  // Returns true iff this reduction can be reduced to an identity (i.e if the
  // input dimensions to reduce along are all of size 1 and keep_dims is true).
  bool IsReductionSimplifiableToIdentity(
      const NodeDef& node, const TensorShapeProto& input_shape, bool keep_dims,
      const gtl::InlinedVector<TensorValue, 4>& reduction_indices_vector) const;
  // Changes a reduction into an Identity op, returning true on success.
  bool ReplaceReductionWithIdentity(NodeDef* node) const;

  // Simplifies a Reduction operation to an Identity/Reshape operation if
  // applicable.
  bool SimplifyReduction(GraphDef* optimized_graph,
                         const GraphProperties& properties, NodeDef* node);

  // Switch(x, x) will always feed false to its false branch and true to
  // its true branch. By rewriting the graph a bit, we can propagate these
  // constants down the two output branches, and just use control dependencies
  // to trigger the selected one at runtime. For example,
  //
  //     +------+
  // x-->|Switch|-->a  (in practice there may be multiple consumers of each
  // x-->|      |-->b   output branch.)
  //     +------+
  //
  // Is rewritten as
  //
  //     +------+
  // x-->|Switch|-->Identity--^>Const(false)-->a
  // x-->|      |-->Identity--^>Const(true)-->b
  //     +------+
  bool SimplifySwitch(GraphDef* optimized_graph, NodeDef* node);

  // Moves constants past Enter node if applicable.
  bool MoveConstantsPastEnter(GraphDef* optimized_graph, NodeDef* node);

  // Simplifies Pack operation if applicable.
  bool SimplifyPack(GraphDef* optimized_graph, NodeDef* node);

  // Simplifies a Squeeze operation to an Identity operation if applicable.
  void SimplifySqueeze(const GraphProperties& properties, bool use_shape_info,
                       GraphDef* optimized_graph, NodeDef* node);

  // Simplifies a Pad operation to an Identity operation if applicable.
  Status SimplifyPad(const GraphProperties& properties, bool use_shape_info,
                     GraphDef* optimized_graph, NodeDef* node);

  // Simplifies a Tile operation to an Identity operation if applicable.
  Status SimplifyTile(const GraphProperties& properties, bool use_shape_info,
                      GraphDef* optimized_graph, NodeDef* node);

  // Simplifies a StridedSlice operation to an Identity operation if applicable.
  Status SimplifyStridedSlice(const GraphProperties& properties,
                              bool use_shape_info, GraphDef* optimized_graph,
                              NodeDef* node);

  // Simplifies a Slice operation to an Identity operation if applicable.
  Status SimplifySlice(const GraphProperties& properties, bool use_shape_info,
                       GraphDef* optimized_graph, NodeDef* node);

  // Simplify a Case operation where the output_idx is known.
  bool SimplifyCase(GraphDef* optimized_graph, NodeDef* node);

  // Simplify a Select operation where the predicates are all true or all false.
  bool SimplifySelect(const GraphProperties& properties,
                      GraphDef* optimized_graph, NodeDef* node);

  // Replaces variable updates that are effectively no-ops with NoOp nodes.
  void RemoveRedundantVariableUpdates(GraphProperties* properties,
                                      GraphDef* optimized_graph, NodeDef* node);

  // Removes Reverse op over dimensions with size 1.
  Status RemoveReverse(const GraphProperties& properties, bool use_shape_info,
                       GraphDef* optimized_graph, NodeDef* node);

  // Removes RandomShuffle op if it is scalar or first dimension is of size 1.
  void RemoveRandomShuffle(const GraphProperties& properties,
                           bool use_shape_info, GraphDef* optimized_graph,
                           NodeDef* node);

  // Removes Shuffle or Transpose op over dimensions of size 1.
  Status RemoveShuffleOrTranspose(const GraphProperties& properties,
                                  bool use_shape_info,
                                  GraphDef* optimized_graph, NodeDef* node);

  // Removes Split or SplitV node if possible.
  void RemoveSplitOrSplitV(const GraphProperties& properties,
                           GraphDef* optimized_graph, NodeDef* node);

  bool GetConcatAxis(const NodeDef& node, int* axis);
  bool MergeConcat(bool use_shape_info, GraphProperties* properties,
                   GraphDef* optimized_graph, NodeDef* node);

  Status AddQuantizedMatMulMinMaxOutConstNodes(NodeDef* node,
                                               GraphDef* optimized_graph);

  // Points to an externally provided device or to owned_device_;
  RewriterConfig::Toggle opt_level_;
  DeviceBase* cpu_device_;
  std::unique_ptr<DeviceBase> owned_device_;

  std::unique_ptr<ResourceMgr> resource_mgr_;
  GraphDef* graph_;
  std::unique_ptr<NodeMap> node_map_;
  std::unordered_set<string> nodes_to_preserve_;
  // TODO(rmlarsen): Could these be keyed on absl::string_view?
  absl::flat_hash_set<string> nodes_allowlist_;
  absl::flat_hash_set<string> feed_nodes_;
  absl::flat_hash_map<string, bool> maybe_foldable_nodes_;
  bool has_fetch_;
  bool graph_modified_;
  bool graph_contains_assign_or_inplace_op_;
  bool disable_compressed_tensor_optimization_;
  bool fold_quantization_emulation_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CONSTANT_FOLDING_H_
