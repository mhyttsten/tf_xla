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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc() {
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

#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/export_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/translate_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/verify_suitable_for_graph_export.h"
#include "tensorflow/compiler/mlir/utils/name_utils.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
using llvm::dyn_cast;
using llvm::isa;
using mlir::BlockArgument;
using mlir::Dialect;
using mlir::Operation;
using mlir::SymbolTable;
using mlir::Value;
using mlir::func::FuncOp;
using stream_executor::port::StatusOr;

namespace {

constexpr char kDeviceAttr[] = "tf.device";
constexpr char kResourceArgUniqueIdAttr[] = "tf._resource_arg_unique_id";
constexpr char kEntryFuncAttr[] = "tf.entry_function";
constexpr char kAliasingAttr[] = "tf.aliasing_output";

// OpOrArgLocNameMapper that legalizes the returned name.
class LegalizedOpOrValLocNameMapper : public OpOrArgLocNameMapper {
 private:
  std::string GetName(OpOrVal op_or_val) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc mht_0(mht_0_v, 260, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.cc", "GetName");

    std::string name = OpOrArgLocNameMapper::GetName(op_or_val);
    assert(!name.empty() && "expected non-empty name");
    mlir::LegalizeNodeName(name);
    return name;
  }
};

// Finds first inner op if `op` is a tf_executor.island. Otherwise `op` is
// returned.
Operation* GetIslandInnerOpOrSelf(mlir::Operation* op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc mht_1(mht_1_v, 273, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.cc", "GetIslandInnerOpOrSelf");

  auto island = llvm::dyn_cast<mlir::tf_executor::IslandOp>(op);
  if (island) return &island.GetBody().front();
  return op;
}

// Stateful helper class to export a function into a Graph.
class Exporter {
 public:
  // Converts the given Module to a Graph. The given module should only contain
  // one entry function, which is identified by name "main". This entry function
  // is converted to the base of the graph graph. The rest of the functions are
  // converted to the library functions in that graph.
  static Status Convert(mlir::ModuleOp module, const GraphExportConfig& configs,
                        std::unique_ptr<Graph>* graph,
                        FunctionLibraryDefinition* flib_def,
                        absl::flat_hash_set<Node*>* control_ret_nodes);

  // Converts a given FuncOp to a FunctionDef and adds it to the function
  // definition library
  static Status ConvertLibFunction(
      const GraphExportConfig& configs, const Dialect* tf_dialect,
      const SymbolTable& symbol_table, FuncOp function,
      FunctionDefLibrary* flib, llvm::SmallDenseSet<FuncOp>& visited_functions);

  // Converts the given FuncOp to a Graph. The arguments and returns of
  // function are added to the graph with special op names kArgOp and kRetOp.
  // Later on, this graph can be converted a function definition and added to
  // another graph.
  static StatusOr<std::unique_ptr<Graph>> Convert(
      const GraphExportConfig& configs, const Dialect* tf_dialect,
      const SymbolTable& symbol_table, FuncOp function,
      FunctionDefLibrary* flib, llvm::SmallDenseSet<FuncOp>& visited_functions,
      absl::flat_hash_set<Node*>* control_ret_nodes);

 private:
  explicit Exporter(Graph* graph, const Dialect* tf_dialect)
      : graph_(graph), tf_dialect_(tf_dialect) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc mht_2(mht_2_v, 313, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.cc", "Exporter");
}

  Status AddArgumentNode(BlockArgument arg, unsigned index,
                         llvm::StringRef name);
  Status AddFetchNode(FuncOp function, mlir::tf_executor::FetchOp fetch,
                      llvm::ArrayRef<llvm::StringRef> names);
  Status AddInstructionNode(Operation* inst);
  Status AddEdge(Operation* inst);

  StatusOr<std::unique_ptr<NodeDef>> GetArgumentNode(BlockArgument arg,
                                                     unsigned index,
                                                     llvm::StringRef name);
  StatusOr<std::unique_ptr<NodeDef>> GetReturnNode(FuncOp function,
                                                   Value operand,
                                                   unsigned index,
                                                   llvm::StringRef name);
  Status GetControlRetNodes(mlir::tf_executor::FetchOp fetch,
                            absl::flat_hash_set<Node*>* control_ret_nodes);
  // Adds one edge between src_node and dst_node. If it is not a control edge,
  // an index is used to find out the right operand of the dst_node.
  Status AddEdgeBetweenNodes(Value src, Node* dst_node, unsigned dst_index);

  Graph* graph_;
  LegalizedOpOrValLocNameMapper op_to_name_;
  absl::flat_hash_map<Operation*, Node*> nodes_;
  llvm::DenseMap<BlockArgument, Node*> args_;
  // One single return operation can return multiple results, and each of them
  // will be converted to one node in the graph.
  typedef absl::InlinedVector<Node*, 4> NodeVector;
  absl::flat_hash_map<Operation*, NodeVector> returns_;
  const mlir::Dialect* tf_dialect_;
};

StatusOr<std::unique_ptr<NodeDef>> Exporter::GetArgumentNode(
    BlockArgument arg, unsigned index, llvm::StringRef name) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc mht_3(mht_3_v, 350, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.cc", "Exporter::GetArgumentNode");

  auto func = arg.getParentRegion()->getParentOfType<FuncOp>();

  auto node_def = absl::make_unique<NodeDef>();
  if (!name.empty())
    node_def->set_name(std::string(ParseTensorName(name.str()).node()));
  else
    node_def->set_name(
        std::string(op_to_name_.GetUniqueName(func.getName().str())));

  node_def->set_op(FunctionLibraryDefinition::kArgOp);

  mlir::TensorType arg_type = arg.getType().cast<mlir::TensorType>();
  if (auto resource_type =
          arg_type.getElementType().dyn_cast<mlir::TF::ResourceType>()) {
    llvm::ArrayRef<mlir::TensorType> subtypes = resource_type.getSubtypes();
    if (!subtypes.empty()) {
      AttrValue handle_dtypes_attr;
      AttrValue handle_shapes_attr;
      for (mlir::TensorType subtype : subtypes) {
        DataType dtype;
        TF_RETURN_IF_ERROR(ConvertToDataType(subtype.getElementType(), &dtype));
        handle_dtypes_attr.mutable_list()->add_type(dtype);

        SetTensorShapeProto(subtype,
                            handle_shapes_attr.mutable_list()->add_shape());
      }

      (*node_def->mutable_attr())["_handle_dtypes"] = handle_dtypes_attr;
      (*node_def->mutable_attr())["_handle_shapes"] = handle_shapes_attr;
    }
  }

  TF_RETURN_IF_ERROR(
      SetShapeAttribute("_output_shapes", arg_type, node_def->mutable_attr()));

  DataType dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(arg_type.getElementType(), &dtype));
  AttrValue type_attr;
  type_attr.set_type(dtype);
  (*node_def->mutable_attr())["T"] = type_attr;

  AttrValue index_attr;
  index_attr.set_i(index);
  (*node_def->mutable_attr())["index"] = index_attr;

  if (auto device_attr =
          func.getArgAttrOfType<mlir::StringAttr>(index, kDeviceAttr))
    *node_def->mutable_device() = device_attr.getValue().str();

  llvm::ArrayRef<mlir::NamedAttribute> func_arg_i_attrs =
      func.getArgAttrs(index);
  absl::flat_hash_set<absl::string_view> attrs_to_ignore = {kDeviceAttr,
                                                            kAliasingAttr};
  TF_RETURN_IF_ERROR(ConvertAttributes(func_arg_i_attrs, attrs_to_ignore,
                                       /*remove_ref_type=*/false,
                                       node_def->mutable_attr()));

  return node_def;
}

StatusOr<std::unique_ptr<NodeDef>> Exporter::GetReturnNode(
    FuncOp function, Value operand, unsigned index, llvm::StringRef name) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc mht_4(mht_4_v, 415, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.cc", "Exporter::GetReturnNode");

  auto node_def = absl::make_unique<NodeDef>();
  if (!name.empty())
    node_def->set_name(std::string(ParseTensorName(name.str()).node()));
  else
    node_def->set_name(
        std::string(op_to_name_.GetUniqueName(function.getName().str())));

  node_def->set_op(FunctionLibraryDefinition::kRetOp);
  DataType dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(
      operand.getType().cast<mlir::TensorType>().getElementType(), &dtype));
  AttrValue type_attr;
  type_attr.set_type(dtype);
  (*node_def->mutable_attr())["T"] = type_attr;
  AttrValue index_attr;
  index_attr.set_i(index);
  (*node_def->mutable_attr())["index"] = index_attr;

  if (auto device_attr =
          function.getResultAttrOfType<mlir::StringAttr>(index, kDeviceAttr))
    *node_def->mutable_device() = device_attr.getValue().str();

  llvm::ArrayRef<mlir::NamedAttribute> func_res_i_attrs =
      function.getResultAttrs(index);
  absl::flat_hash_set<absl::string_view> attrs_to_ignore = {kDeviceAttr};
  TF_RETURN_IF_ERROR(ConvertAttributes(func_res_i_attrs, attrs_to_ignore,
                                       /*remove_ref_type=*/false,
                                       node_def->mutable_attr()));

  return node_def;
}

Status Exporter::AddEdgeBetweenNodes(Value src, Node* dst_node,
                                     unsigned dst_index) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc mht_5(mht_5_v, 452, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.cc", "Exporter::AddEdgeBetweenNodes");

  if (auto input_result = src.dyn_cast<mlir::OpResult>()) {
    auto* input_inst = GetIslandInnerOpOrSelf(input_result.getOwner());
    // Replaces the input node with NextIteration sink if it is a NextIteration
    // source.
    if (auto next_iter_source =
            llvm::dyn_cast<mlir::tf_executor::NextIterationSourceOp>(
                input_inst))
      input_inst = next_iter_source.GetSink();

    auto node_it = nodes_.find(input_inst);
    TF_RET_CHECK(node_it != nodes_.end())
        << "Use of OpResult encountered before def!";
    if (input_result.getType().isa<mlir::tf_executor::ControlType>()) {
      graph_->AddControlEdge(node_it->second, dst_node);
    } else {
      graph_->AddEdge(node_it->second, input_result.getResultNumber(), dst_node,
                      dst_index);
    }
    return Status::OK();
  }

  auto input_arg = src.cast<BlockArgument>();
  auto input_node_it = args_.find(input_arg);
  TF_RET_CHECK(input_node_it != args_.end())
      << "Use of BlockArgument encounted before def!";
  // For argument, there is only one result output, so the index is always 0.
  graph_->AddEdge(input_node_it->second, 0, dst_node, dst_index);
  return Status::OK();
}

Status Exporter::AddEdge(Operation* inst) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc mht_6(mht_6_v, 486, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.cc", "Exporter::AddEdge");

  // For tf_executor.fetch, add only its data edges. Control edges are captured
  // later.
  if (auto fetch = llvm::dyn_cast<mlir::tf_executor::FetchOp>(inst)) {
    for (auto operand_and_idx : llvm::enumerate(fetch.getOperands())) {
      Value operand = operand_and_idx.value();
      if (operand.getType().isa<mlir::tf_executor::ControlType>()) break;

      auto* dst_node = returns_[fetch][operand_and_idx.index()];
      TF_RETURN_IF_ERROR(AddEdgeBetweenNodes(operand, dst_node, 0));
    }

    return Status::OK();
  }

  // For tf_executor.NextIteration.Sink, skip its token operand and add data and
  // control edges with their index offset by 1.
  if (auto next_iter_sink =
          llvm::dyn_cast<mlir::tf_executor::NextIterationSinkOp>(inst)) {
    auto* dst_node = nodes_[inst];
    TF_RETURN_IF_ERROR(
        AddEdgeBetweenNodes(next_iter_sink.input(), dst_node, 0));
    for (auto control_and_idx : llvm::enumerate(next_iter_sink.controlInputs()))
      TF_RETURN_IF_ERROR(AddEdgeBetweenNodes(control_and_idx.value(), dst_node,
                                             control_and_idx.index() + 1));

    return Status::OK();
  }

  // For tf_executor.NextIteration.Source, op can be skipped as it is assumed
  // there are no operands.
  if (llvm::isa<mlir::tf_executor::NextIterationSourceOp>(inst)) {
    assert(inst->getNumOperands() == 0);
    return Status::OK();
  }

  Operation* op = GetIslandInnerOpOrSelf(inst);
  auto* dst_node = nodes_[op];
  int operand_offset = 0;
  // For tf_executor.island, add data edges from its wrapped op before control
  // edges.
  if (auto island = llvm::dyn_cast<mlir::tf_executor::IslandOp>(inst)) {
    for (auto operand_and_idx : llvm::enumerate(op->getOperands()))
      TF_RETURN_IF_ERROR(AddEdgeBetweenNodes(operand_and_idx.value(), dst_node,
                                             operand_and_idx.index()));

    operand_offset = op->getNumOperands();
  }

  // For all other ops (including tf_executor.island), add remaining edges.
  for (auto operand_and_idx : llvm::enumerate(inst->getOperands()))
    TF_RETURN_IF_ERROR(
        AddEdgeBetweenNodes(operand_and_idx.value(), dst_node,
                            operand_and_idx.index() + operand_offset));

  return Status::OK();
}

Status Exporter::AddInstructionNode(Operation* inst) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc mht_7(mht_7_v, 547, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.cc", "Exporter::AddInstructionNode");

  std::unique_ptr<NodeDef> node_def;
  auto name = op_to_name_.GetUniqueName(inst);
  // Convert registered TF ops to NodeDef. Only registered ops are handled to
  // ensure that PopulateDerivedAttrs adds the correct attributes.
  TF_ASSIGN_OR_RETURN(node_def,
                      ConvertTFDialectOpToNodeDef(
                          inst, name, /*ignore_unregistered_attrs=*/false));

  TF_ASSIGN_OR_RETURN(Node * node, graph_->AddNode(*node_def));
  DCHECK(node != nullptr);
  nodes_[inst] = node;
  return Status::OK();
}

bool IsEntryFunctionArg(BlockArgument arg) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc mht_8(mht_8_v, 565, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.cc", "IsEntryFunctionArg");

  return arg.getParentRegion()->getParentOfType<FuncOp>().getName() == "main";
}

// Creates argument nodes from Block argument. If a name is supplied, that
// name will be used instead of generating a unique name.
Status Exporter::AddArgumentNode(BlockArgument arg, unsigned index,
                                 llvm::StringRef name) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc mht_9(mht_9_v, 575, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.cc", "Exporter::AddArgumentNode");

  TF_ASSIGN_OR_RETURN(auto node_def, GetArgumentNode(arg, index, name));
  TF_ASSIGN_OR_RETURN(Node * node, graph_->AddNode(*node_def));
  args_[arg] = node;
  return Status::OK();
}

// Creates return nodes per operand of a FetchOp. If names is supplied, those
// names will be used per node in order instead of generating a unique name.
Status Exporter::AddFetchNode(FuncOp function, mlir::tf_executor::FetchOp fetch,
                              llvm::ArrayRef<llvm::StringRef> names) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc mht_10(mht_10_v, 588, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.cc", "Exporter::AddFetchNode");

  auto& return_nodes = returns_[fetch];
  for (auto operand_and_idx : llvm::enumerate(fetch.getOperands())) {
    if (operand_and_idx.value().getType().isa<mlir::tf_executor::ControlType>())
      break;

    TF_ASSIGN_OR_RETURN(
        auto node_def,
        GetReturnNode(function, operand_and_idx.value(),
                      operand_and_idx.index(),
                      names.empty() ? "" : names[operand_and_idx.index()]));
    TF_ASSIGN_OR_RETURN(Node * node, graph_->AddNode(*node_def));
    return_nodes.push_back(node);
  }
  return Status::OK();
}

// Collects control ret Nodes based on tf_executor.graph's associated
// tf_executor.fetch control inputs.
Status Exporter::GetControlRetNodes(
    mlir::tf_executor::FetchOp fetch,
    absl::flat_hash_set<Node*>* control_ret_nodes) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc mht_11(mht_11_v, 612, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.cc", "Exporter::GetControlRetNodes");

  for (Value fetch_operand : fetch.getOperands()) {
    if (fetch_operand.getType().isa<mlir::tf_executor::ControlType>()) {
      Operation* defining_op =
          GetIslandInnerOpOrSelf(fetch_operand.getDefiningOp());
      auto node_it = nodes_.find(defining_op);
      TF_RET_CHECK(node_it != nodes_.end());
      control_ret_nodes->insert(node_it->second);
    }
  }
  return Status::OK();
}

StatusOr<std::unique_ptr<Graph>> Exporter::Convert(
    const GraphExportConfig& configs, const Dialect* tf_dialect,
    const SymbolTable& symbol_table, FuncOp function, FunctionDefLibrary* flib,
    llvm::SmallDenseSet<FuncOp>& visited_functions,
    absl::flat_hash_set<Node*>* control_ret_nodes) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc mht_12(mht_12_v, 632, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.cc", "Exporter::Convert");

  mlir::Block& block = function.front();

  // Extract input & output names if set.
  llvm::SmallVector<llvm::StringRef, 2> input_names;
  llvm::SmallVector<llvm::StringRef, 2> output_names;
  llvm::SmallVector<llvm::StringRef, 2> unique_output_names;
  auto dict_attr =
      function->getAttrOfType<mlir::DictionaryAttr>(kEntryFuncAttr);
  if (dict_attr) {
    TF_RET_CHECK(dict_attr.get("inputs").isa<mlir::StringAttr>())
        << "inputs missing in entry function attribute";
    TF_RET_CHECK(dict_attr.get("outputs").isa<mlir::StringAttr>())
        << "outputs missing in entry function attribute";
    dict_attr.get("inputs").cast<mlir::StringAttr>().getValue().split(
        input_names, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    dict_attr.get("outputs").cast<mlir::StringAttr>().getValue().split(
        output_names, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  }

  auto graph = absl::make_unique<Graph>(OpRegistry::Global());

  // Extract version info.
  VersionDef versions;
  auto module = function->getParentOfType<mlir::ModuleOp>();
  if (mlir::succeeded(ExtractTfVersions(module, &versions))) {
    graph->set_versions(versions);
  }

  Exporter exporter(graph.get(), tf_dialect);

  auto graph_op = llvm::cast<mlir::tf_executor::GraphOp>(block.front());

  // Set input and output names and increment the use counter for them to help
  // generate unique names.
  if (!output_names.empty()) {
    const int num_data_results = graph_op.getNumResults();
    const int64_t output_names_size = output_names.size();
    TF_RET_CHECK(output_names_size == num_data_results)
        << "output names (" << output_names.size()
        << ") != terminator operands (" << num_data_results << ")";
    llvm::DenseMap<Operation*, llvm::StringRef> output_op_to_name;
    llvm::StringMap<Operation*> name_to_op;
    for (const auto& it : llvm::enumerate(graph_op.GetFetch().getOperands())) {
      // Skip control rets.
      const int64_t index = it.index();
      if (index >= num_data_results) break;
      // TODO(jpienaar): If there is a result index specified, ensure only one
      // and that it matches the result index of the op.
      std::string name(output_names[index]);
      auto tensor_id = ParseTensorName(name);
      std::string tensor_id_node(tensor_id.node());
      assert(!tensor_id_node.empty() && "expected non-empty name");
      mlir::LegalizeNodeName(tensor_id_node);

      // Ensure name does not get reused.
      unique_output_names.push_back(
          exporter.op_to_name_.GetUniqueName(tensor_id_node));
    }
  }

  if (!input_names.empty()) {
    TF_RET_CHECK(input_names.size() == block.getNumArguments());
    for (const auto& it : llvm::enumerate(function.getArguments())) {
      // TODO(lyandy): Update when changing feed/fetch import.
      std::string name(input_names[it.index()]);
      assert(!name.empty() && "expected non-empty name");
      mlir::LegalizeNodeName(name);
      auto tensor_id = ParseTensorName(name);
      TF_RET_CHECK(tensor_id.index() == 0)
          << "input port designation not supported";
      // Only assign user of argument the input name if the main graph did not
      // have its _Arg nodes lifted into the functions arguments.
      // Ensure name does not get reused.
      (void)exporter.op_to_name_.GetUniqueName(name);
    }
  }

  // Adds nodes for basic block (function) arguments.
  for (auto it : llvm::enumerate(block.getArguments())) {
    int index = it.index();
    auto arg = it.value();
    mlir::Type type = arg.getType();
    if (!type.isa<mlir::TensorType>()) {
      return errors::InvalidArgument(
          "FuncOps arguments must have tensor types. Found ",
          mlir::debugString(type), " in function ", function.getName().str());
    }

    TF_RETURN_IF_ERROR(exporter.AddArgumentNode(
        arg, index, !input_names.empty() ? input_names[index] : ""));
  }

  auto convert_called_function = [&](llvm::StringRef name) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc mht_13(mht_13_v, 728, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.cc", "lambda");

    auto func = symbol_table.lookup<FuncOp>(name);
    if (func != nullptr) {
      TF_RETURN_IF_ERROR(ConvertLibFunction(configs, tf_dialect, symbol_table,
                                            func, flib, visited_functions));
      // TODO(prakalps): Optimize to only add the requested function to graph
      // library rather than the all the functions exported so far.
      TF_RETURN_IF_ERROR(graph->AddFunctionLibrary(*flib));
    }
    return Status::OK();
  };

  // Adds nodes for operations.
  for (Operation& inst : graph_op.GetBody()) {
    for (auto type : inst.getResultTypes())
      if (!type.isa<mlir::TensorType, mlir::tf_executor::ControlType,
                    mlir::tf_executor::TokenType>())
        return errors::InvalidArgument(
            "Values must be of tensor type, TensorFlow control type, or "
            "TensorFlow token type. Found ",
            mlir::debugString(type));

    if (llvm::isa<mlir::tf_executor::NextIterationSourceOp>(inst)) {
      // Skip tf_executor.NextIteration.Source as associated
      // tf_executor.NextIteration.Sink will be used instead.
      continue;
    } else if (auto fetch = llvm::dyn_cast<mlir::tf_executor::FetchOp>(inst)) {
      TF_RETURN_IF_ERROR(
          exporter.AddFetchNode(function, fetch, unique_output_names));
    } else if (auto island =
                   llvm::dyn_cast<mlir::tf_executor::IslandOp>(inst)) {
      Operation& inner_op = island.GetBody().front();
      auto op_name = GetTensorFlowOpName(inner_op.getName().getStringRef());
      if (op_name.ok()) {
        // If it is TF Control dialect specific op, look up custom operation
        // in the module and first convert that, then add it to function
        // definition library
        // TODO(prakalps): If two functions have cyclic dependence, this will
        // introduce an infinite loop.
        TF_RETURN_IF_ERROR(convert_called_function(op_name.ValueOrDie().str()));
      }

      if (IsLegacyCallInstruction(&inner_op)) {
        TF_RETURN_IF_ERROR(convert_called_function(
            inner_op.getAttrOfType<mlir::SymbolRefAttr>("f")
                .getLeafReference()
                .getValue()));
      }

      TF_RETURN_IF_ERROR(exporter.AddInstructionNode(&inner_op));
    } else {
      TF_RETURN_IF_ERROR(exporter.AddInstructionNode(&inst));
    }
  }
  // Adds edges between the argument, operation and return nodes.
  for (Operation& inst : graph_op.GetBody()) {
    TF_RETURN_IF_ERROR(exporter.AddEdge(&inst));
  }
  // Fixes the edges between the inserted nodes and special "_SOURCE" and
  // "_SINK".
  FixupSourceAndSinkEdges(graph.get());

  TF_RETURN_IF_ERROR(
      exporter.GetControlRetNodes(graph_op.GetFetch(), control_ret_nodes));

  return graph;
}

Status Exporter::ConvertLibFunction(
    const GraphExportConfig& configs, const Dialect* tf_dialect,
    const SymbolTable& symbol_table, FuncOp function, FunctionDefLibrary* flib,
    llvm::SmallDenseSet<FuncOp>& visited_functions) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc mht_14(mht_14_v, 802, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.cc", "Exporter::ConvertLibFunction");

  // Return early if the function has already been exported.
  bool is_new_function = visited_functions.insert(function).second;
  if (!is_new_function) return Status::OK();

  auto function_name = function.getName().str();

  // TODO(fengliuai): use a small flib_def to reduce overhead
  absl::flat_hash_set<Node*> control_ret_nodes;
  TF_ASSIGN_OR_RETURN(
      auto sub_graph,
      Exporter::Convert(configs, tf_dialect, symbol_table, function, flib,
                        visited_functions, &control_ret_nodes));
  const auto control_ret = [&](const Node* n) -> absl::optional<string> {
    return control_ret_nodes.contains(n)
               ? absl::make_optional<string>(n->name())
               : absl::nullopt;
  };
  FunctionDef func_def;
  TF_RETURN_IF_ERROR(
      GraphToFunctionDef(*sub_graph, function_name, control_ret, &func_def));

  // The node defs in FunctionDef might contain debug info which was added
  // by the GraphToFunctionDef method. We should remove it if we don't want
  // to export them to avoid failing the roundtrip test.
  if (!configs.export_debug_info) {
    for (auto& node_def : *func_def.mutable_node_def()) {
      node_def.clear_experimental_debug_info();
    }
  }

  // Checks for gradient attribute. If present converts the gradient function
  // and populates the GradientDef.
  auto grad_string = mlir::TF::TensorFlowDialect::GetGradientAttrName();
  if (auto attr =
          function->getAttrOfType<mlir::FlatSymbolRefAttr>(grad_string)) {
    auto grad_func = symbol_table.lookup<FuncOp>(attr.getValue());
    TF_RETURN_IF_ERROR(ConvertLibFunction(configs, tf_dialect, symbol_table,
                                          grad_func, flib, visited_functions));
    GradientDef grad;
    grad.set_function_name(function_name);
    grad.set_gradient_func(grad_func.getName().str());
    *flib->add_gradient() = grad;
  }

  auto stateful_string = mlir::TF::TensorFlowDialect::GetStatefulAttrName();
  if (auto attr = function->getAttrOfType<mlir::UnitAttr>(stateful_string)) {
    func_def.mutable_signature()->set_is_stateful(true);
  }

  // Ignore the gradient and is_stateful attribute on the function as they have
  // been handled above. Ignore the entry func attribute as it is an MLIR
  // metadata attribute and is not required in the function definition.
  absl::flat_hash_set<absl::string_view> attrs_to_ignore = {
      grad_string.data(), stateful_string.data(), kEntryFuncAttr};
  llvm::SmallVector<mlir::NamedAttribute, 8> funcAttrs(
      function->getDialectAttrs());
  TF_RETURN_IF_ERROR(ConvertAttributes(funcAttrs, attrs_to_ignore,
                                       /*remove_ref_type=*/false,
                                       func_def.mutable_attr()));

  for (int i = 0, e = function.getNumArguments(); i < e; ++i) {
    if (auto resource_arg_unique_id_attr =
            function.getArgAttrOfType<mlir::IntegerAttr>(
                i, kResourceArgUniqueIdAttr)) {
      (*func_def.mutable_resource_arg_unique_id())[i] =
          resource_arg_unique_id_attr.getInt();
    }
  }

  (*flib->add_function()) = std::move(func_def);
  return Status::OK();
}

Status Exporter::Convert(mlir::ModuleOp module,
                         const GraphExportConfig& configs,
                         std::unique_ptr<Graph>* graph,
                         FunctionLibraryDefinition* flib_def,
                         absl::flat_hash_set<Node*>* control_ret_nodes) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc mht_15(mht_15_v, 883, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.cc", "Exporter::Convert");

  mlir::StringAttr entry_func_id =
      mlir::StringAttr::get(module.getContext(), "main");
  absl::optional<FuncOp> entry_func;
  FunctionDefLibrary flib;
  llvm::SmallDenseSet<FuncOp> visited_functions;
  auto tf_dialect = module.getContext()->getLoadedDialect("tf");
  // Construct SymbolTable to enable cheap function lookups. The cost
  // of constructing the table is offset by the number of queries.
  SymbolTable symbol_table(module);
  for (auto function : module.getOps<FuncOp>()) {
    if (function.isExternal())
      return errors::FailedPrecondition("External functions not supported");

    if (function.getName() == entry_func_id &&
        !configs.export_entry_func_to_flib) {
      entry_func.emplace(function);
    } else {
      TF_RETURN_IF_ERROR(ConvertLibFunction(configs, tf_dialect, symbol_table,
                                            function, &flib,
                                            visited_functions));
    }
  }

  if (!configs.export_entry_func_to_flib) {
    if (!entry_func.has_value())
      return errors::FailedPrecondition(
          "entry function `main` must be present");

    // Updates the graph and the function library definition.
    TF_ASSIGN_OR_RETURN(
        *graph,
        Exporter::Convert(configs, tf_dialect, symbol_table, entry_func.value(),
                          &flib, visited_functions, control_ret_nodes));
    // Add FunctionDefs and GradientDefs of MLIR functions to graph's function
    // library. If duplicate FunctionDefs already exist (can happen if exporter
    // had already added some FunctionDefs to the library to support legacy
    // calls), they are ignored.
    TF_RETURN_IF_ERROR(graph->get()->AddFunctionLibrary(flib));
  }

  for (auto& func_def : flib.function()) {
    TF_RETURN_IF_ERROR(flib_def->AddFunctionDef(func_def));
  }
  for (auto& grad_def : flib.gradient()) {
    TF_RETURN_IF_ERROR(flib_def->AddGradientDef(grad_def));
  }
  return Status::OK();
}
}  // namespace

Status ConvertMlirToGraph(mlir::ModuleOp module,
                          const GraphExportConfig& configs,
                          std::unique_ptr<Graph>* graph,
                          FunctionLibraryDefinition* flib_def,
                          absl::flat_hash_set<Node*>* control_ret_nodes) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc mht_16(mht_16_v, 941, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.cc", "ConvertMlirToGraph");

  mlir::StatusScopedDiagnosticHandler sh(module.getContext());
  if (failed(VerifyExportSuitable(module))) return sh.ConsumeStatus();
  return sh.Combine(
      Exporter::Convert(module, configs, graph, flib_def, control_ret_nodes));
}

Status ConvertMlirToGraph(mlir::ModuleOp module,
                          const GraphExportConfig& configs,
                          std::unique_ptr<Graph>* graph,
                          FunctionLibraryDefinition* flib_def) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc mht_17(mht_17_v, 954, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.cc", "ConvertMlirToGraph");

  absl::flat_hash_set<Node*> control_ret_nodes;
  return ConvertMlirToGraph(module, configs, graph, flib_def,
                            &control_ret_nodes);
}

StatusOr<std::unique_ptr<GraphDef>> ConvertMlirToGraphdef(
    mlir::ModuleOp module, const GraphExportConfig& configs) {
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  std::unique_ptr<Graph> graph;
  TF_RETURN_IF_ERROR(ConvertMlirToGraph(module, configs, &graph, &flib_def));

  // If the entry function is exported to flib, then no graph is constructed.
  // Construct one in that case.
  if (configs.export_entry_func_to_flib) {
    graph = std::make_unique<Graph>(OpRegistry::Global());
    // TODO(hinsu): Avoid Proto -> Memory -> Proto conversion here.
    FunctionDefLibrary flib = flib_def.ToProto();
    TF_RETURN_IF_ERROR(graph->AddFunctionLibrary(flib));
  }

  auto graphdef = absl::make_unique<GraphDef>();
  graph->ToGraphDef(graphdef.get());
  if (!configs.export_library) graphdef->clear_library();
  if (!configs.export_shapes) {
    for (auto& node_def : *graphdef->mutable_node()) {
      node_def.mutable_attr()->erase("shape");
    }
  }
  if (!configs.export_debug_info) {
    for (auto& node_def : *graphdef->mutable_node()) {
      node_def.clear_experimental_debug_info();
    }
  }
  return graphdef;
}

stream_executor::port::Status ConvertMlirFunctionToFunctionLibraryDef(
    FuncOp func, const GraphExportConfig& configs, FunctionDef* function_def) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_graphdefDTcc mht_18(mht_18_v, 996, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.cc", "ConvertMlirFunctionToFunctionLibraryDef");

  Dialect* tf_dialect = func.getContext()->getLoadedDialect("tf");
  FunctionDefLibrary flib;
  llvm::SmallDenseSet<FuncOp> visited_functions;
  // Construct SymbolTable to enable cheap function lookups. The cost
  // of constructing the table is offset by the number of queries. Even
  // though this only converts one function in theory, this function
  // may have gradient associated which would result in a lookup. This
  // could be made lazy if we find this to be broad.
  SymbolTable symbol_table(func->getParentOfType<mlir::ModuleOp>());
  TF_RETURN_IF_ERROR(Exporter::ConvertLibFunction(
      configs, tf_dialect, symbol_table, func, &flib, visited_functions));
  for (auto& func_def : flib.function()) {
    if (func_def.signature().name() == func.getName()) {
      *function_def = func_def;
      return Status::OK();
    }
  }
  return errors::InvalidArgument(
      "Function couldn't be found in the FunctionDefLibrary after converting "
      "from MLIR");
}

}  // namespace tensorflow
