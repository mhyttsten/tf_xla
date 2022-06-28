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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSresource_spmd_expanderDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSresource_spmd_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSresource_spmd_expanderDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/mlir/expansions/resource_spmd_expander.h"

#include <algorithm>
#include <string>

#include "absl/strings/str_join.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/utils/array_container_utils.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {
namespace {

template <typename AttrType>
std::vector<AttrType> CreateOrGetMutableAttributeList(
    mlir::tf_device::ClusterOp op, std::string attr_name) {
  auto array_attribute = op->getAttrOfType<mlir::ArrayAttr>(attr_name);

  std::vector<AttrType> output;
  if (array_attribute) auto attr_list = array_attribute.getValue().vec();
  return output;
}

Status ValidateAndAssignResourceInputLayout(mlir::tf_device::ClusterOp op,
                                            const std::string& layout_string,
                                            const int resource_arg_index,
                                            mlir::OpBuilder* builder) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("layout_string: \"" + layout_string + "\"");
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSresource_spmd_expanderDTcc mht_0(mht_0_v, 230, "", "./tensorflow/dtensor/mlir/expansions/resource_spmd_expander.cc", "ValidateAndAssignResourceInputLayout");

  const auto add_layout_as_attributes =
      [&](std::vector<mlir::StringRef> new_resource_layouts,
          std::vector<int> new_resource_indices, int resource_arg_index,
          std::string layout) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("layout: \"" + layout + "\"");
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSresource_spmd_expanderDTcc mht_1(mht_1_v, 238, "", "./tensorflow/dtensor/mlir/expansions/resource_spmd_expander.cc", "lambda");

        new_resource_layouts.emplace_back(layout);
        new_resource_indices.emplace_back(resource_arg_index);
        op->setAttr(kNewResourceArgLayouts,
                    builder->getStrArrayAttr(
                        llvm::ArrayRef<mlir::StringRef>(new_resource_layouts)));
        op->setAttr(kNewResourceLayoutIndices,
                    builder->getI32VectorAttr(new_resource_indices));
      };

  auto resource_input_layouts_attrs =
      CreateOrGetMutableAttributeList<mlir::StringAttr>(op,
                                                        kNewResourceArgLayouts);
  auto resource_input_indices_attrs =
      CreateOrGetMutableAttributeList<mlir::IntegerAttr>(
          op, kNewResourceLayoutIndices);
  std::vector<llvm::StringRef> mutable_input_layouts;
  std::vector<int> mutable_input_indices;
  for (auto layout_index_pair :
       llvm::zip(resource_input_indices_attrs, resource_input_layouts_attrs)) {
    mutable_input_indices.emplace_back(std::get<0>(layout_index_pair).getInt());
    mutable_input_layouts.emplace_back(
        std::get<1>(layout_index_pair).getValue());
  }

  if (!mutable_input_indices.empty()) {
    assert(mutable_input_indices.size() == mutable_input_layouts.size());

    auto it = std::find(mutable_input_indices.begin(),
                        mutable_input_indices.end(), resource_arg_index);

    if (it != mutable_input_indices.end()) {
      // Input layout for given resource was already inferred from previous
      // SPMD expansions. Check that layouts of resource are consistent.
      auto previous_layout = mutable_input_layouts[std::distance(
          mutable_input_indices.begin(), it)];

      // TODO(hongjunchoi): Implement relayout logic for resource ops.
      if (layout_string != previous_layout.str())
        return errors::InvalidArgument(
            "Trying to assign a variable to a resource with a different "
            "layout.");
    } else {
      add_layout_as_attributes(mutable_input_layouts, mutable_input_indices,
                               resource_arg_index, layout_string);
    }
  } else {
    add_layout_as_attributes(mutable_input_layouts, mutable_input_indices,
                             resource_arg_index, layout_string);
  }
  return Status::OK();
}

}  // namespace

StatusOr<mlir::Operation*> ResourceSPMDExpander::ExpandOp(mlir::Operation* op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSresource_spmd_expanderDTcc mht_2(mht_2_v, 296, "", "./tensorflow/dtensor/mlir/expansions/resource_spmd_expander.cc", "ResourceSPMDExpander::ExpandOp");

  // These ops need no special handling.
  if (llvm::isa<mlir::TF::VarHandleOp, mlir::TF::DestroyResourceOp,
                mlir::TF::VarIsInitializedOp>(op))
    return InferSPMDExpandedLocalShape(op);

  mlir::OpBuilder builder(op);

  // Output of read variable may need to be sliced, so it needs to be treated
  // specially.
  if (llvm::isa<mlir::TF::ReadVariableOp>(op)) {
    builder.setInsertionPointAfter(op);
    TF_ASSIGN_OR_RETURN(auto output_layout, ExtractSingleLayoutFromOp(op));
    TF_ASSIGN_OR_RETURN(auto input_layout,
                        ExtractLayoutFromOperand(op->getOperand(0)));
    if (!output_layout)
      TF_RETURN_WITH_CONTEXT(errors::Internal("output layout is missing"));
    if (!input_layout)
      TF_RETURN_WITH_CONTEXT(errors::Internal("input layout is missing"));
    InferSPMDExpandedLocalShape(op);
    llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
    TF_ASSIGN_OR_RETURN(
        auto final_output,
        EmitAllScatter(builder, op->getOpResult(0), input_layout.value(),
                       output_layout.value(), &newly_created_ops));
    op->getOpResult(0).replaceAllUsesExcept(final_output, newly_created_ops);
    return final_output.getDefiningOp();
  }

  if (!llvm::isa<mlir::TF::AssignVariableOp, mlir::TF::AssignAddVariableOp,
                 mlir::TF::AssignSubVariableOp>(op))
    TF_RETURN_WITH_CONTEXT(errors::Internal("unsupported resource op"));

  TF_ASSIGN_OR_RETURN(absl::optional<Layout> output_layout,
                      ExtractSingleLayoutFromOp(op));
  TF_ASSIGN_OR_RETURN(absl::optional<Layout> resource_layout,
                      ExtractLayoutFromOperand(op->getOperand(0)));
  TF_ASSIGN_OR_RETURN(absl::optional<Layout> value_layout,
                      ExtractLayoutFromOperand(op->getOperand(1)));

  // For assignment operations, the layout for the resource (first operand),
  // when not present, is, inferred from the layout of the input value (second
  // operand). We attach the inferred layout to the resource.
  // Note that in the case that input_resource_value.getDefiningOp() exists, it
  // is a DTensorLayout and this means that the corresponding block argument
  // already has a layout set.
  // If the resource is specified in the graph as an op (e.g. VarHandleOp), we
  // attach the layout directly. Otherwise, the resource is an argument to the
  // SPMD function, and we attach the layout to the appropriate argument.
  auto input_resource_value = op->getOpOperand(0).get();
  if (auto resource_producing_op = input_resource_value.getDefiningOp()) {
    if (!resource_layout)
      TF_RETURN_WITH_CONTEXT(errors::Internal("missing layout on resource"));
    if (!value_layout)
      TF_RETURN_WITH_CONTEXT(errors::Internal("missing layout on value"));
    if (resource_layout != value_layout) {
      TF_ASSIGN_OR_RETURN(auto new_value,
                          EmitRelayout(op->getOperand(1), value_layout.value(),
                                       resource_layout.value()));
      op->setOperand(1, new_value);
    }
  } else {
    if ((!resource_layout || resource_layout->IsEmpty()) && !value_layout)
      TF_RETURN_WITH_CONTEXT(errors::Internal(
          "at least one of resource or value layout must be set"));
    // This error should not happen: if resource_layout is set, then we expect
    // a DTensorLayout op between the resource tensor and this op, so we should
    // actaully be in the if case rather than the else case.
    if (resource_layout && !resource_layout->IsEmpty() && value_layout &&
        resource_layout != value_layout)
      TF_RETURN_WITH_CONTEXT(errors::Internal(
          "if both resource and value layout are set they must be equal"));

    auto block_arg = input_resource_value.dyn_cast<mlir::BlockArgument>();
    auto enclosing_device_cluster =
        op->getParentOfType<mlir::tf_device::ClusterOp>();

    if (!enclosing_device_cluster)
      TF_RETURN_WITH_CONTEXT(
          errors::InvalidArgument("op must be enclosed by a cluster"));

    auto block_arg_index = block_arg.getArgNumber();

    // If layout of resource already exists, then check that layouts are
    // consistent. Otherwise, add newly inferred layout of resource argument
    // as attributes to the enclosing cluster op to be propagated to custom
    // device.
    std::string layout_string;
    if (resource_layout && !resource_layout->IsEmpty())
      layout_string = resource_layout->ToString();
    else
      layout_string = value_layout->ToString();
    TF_RETURN_IF_ERROR(ValidateAndAssignResourceInputLayout(
        enclosing_device_cluster, layout_string, block_arg_index, &builder));
  }

  return InferSPMDExpandedLocalShape(op);
}

StatusOr<llvm::DenseMap<int, Layout>>
ResourceSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // VarHandle and VarIsInitialized have 0 rank outputs.
  if (llvm::isa<mlir::TF::VarHandleOp, mlir::TF::VarIsInitializedOp>(op))
    return llvm::DenseMap<int, Layout>({{0, Layout::Empty()}});

  // Handling of resource destruction is no-op.
  if (llvm::isa<mlir::TF::DestroyResourceOp>(op))
    return llvm::DenseMap<int, Layout>();

  // Read variable ops have one input so infer the output layout if input
  // layout exists.
  if (llvm::isa<mlir::TF::ReadVariableOp>(op)) return input_layouts;

  // These ops do not have outputs, so do not infer any layout.
  if (llvm::isa<mlir::TF::AssignVariableOp, mlir::TF::AssignAddVariableOp,
                mlir::TF::AssignSubVariableOp>(op)) {
    return llvm::DenseMap<int, Layout>();
  }
  // Return an error if not any of the ops above.
  return errors::InvalidArgument(
      llvm::formatv(
          "Found unexpected resource op {0} during layout propagation.",
          OpName(op))
          .str());
}

StatusOr<llvm::DenseMap<int, Layout>>
ResourceSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts,
    const llvm::DenseMap<int, Layout>& output_layouts) {
  // For Assign* ops, propagate the resource tensor layout to the tensor if
  // resource tensor layout exists.
  if (llvm::isa<mlir::TF::AssignVariableOp, mlir::TF::AssignAddVariableOp,
                mlir::TF::AssignSubVariableOp>(op)) {
    if (input_layouts.find(0) != input_layouts.end())
      return llvm::DenseMap<int, Layout>({{1, input_layouts.lookup(0)}});
    return llvm::DenseMap<int, Layout>();
  }
  // Handling of these ops are no-ops.
  if (llvm::isa<mlir::TF::DestroyResourceOp, mlir::TF::VarHandleOp,
                mlir::TF::VarIsInitializedOp, mlir::TF::ReadVariableOp>(op))
    return llvm::DenseMap<int, Layout>();

  // Return an error if not any of the ops above.
  return errors::InvalidArgument(
      llvm::formatv(
          "Found unexpected resource op {0} during layout propagation.",
          OpName(op))
          .str());
}

}  // namespace dtensor
}  // namespace tensorflow
