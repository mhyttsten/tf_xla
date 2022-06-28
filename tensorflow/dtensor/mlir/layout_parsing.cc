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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSlayout_parsingDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSlayout_parsingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSlayout_parsingDTcc() {
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

#include "tensorflow/dtensor/mlir/layout_parsing.h"

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
namespace dtensor {
namespace {

bool OpUsesV2LayoutAnnotation(mlir::Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSlayout_parsingDTcc mht_0(mht_0_v, 211, "", "./tensorflow/dtensor/mlir/layout_parsing.cc", "OpUsesV2LayoutAnnotation");

  return !op->getUsers().empty() &&
         llvm::all_of(op->getUsers(), [](mlir::Operation* user_op) {
           return llvm::isa<mlir::TF::DTensorLayout>(user_op);
         });
}

}  // namespace

StatusOr<absl::optional<Layout>> ExtractSingleLayoutFromOp(
    mlir::Operation* op, std::string attr_name) {
  absl::optional<Layout> out;

  // If v2 layout propagation algorithm is used, parse layout from DTensorLayout
  // op.
  if (OpUsesV2LayoutAnnotation(op)) {
    // If DTensorLayout is used, then DTensorLayout op is the only consumer for
    // the operation output value.
    auto users = op->getUsers();
    out.emplace(llvm::cast<mlir::TF::DTensorLayout>(*users.begin()).layout());
  } else {
    TF_ASSIGN_OR_RETURN(auto layouts, ExtractLayoutFromOp(op, attr_name));
    if (layouts.empty()) return out;
    if (layouts.size() != 1) {
      return errors::Internal(
          "Extracting single layout on Op that has multiple layout attached is "
          "ambiguous. op : ",
          op->getName().getStringRef().str());
    }
    out.swap(layouts[0]);
  }
  return out;
}

StatusOr<absl::optional<Layout>> ExtractSingleLayoutFromOp(
    mlir::Operation* op) {
  return ExtractSingleLayoutFromOp(op, kLayoutAttr);
}

StatusOr<Layout> ExtractRequiredSingleLayoutFromOp(mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(absl::optional<Layout> layout,
                      ExtractSingleLayoutFromOp(op));
  if (!layout) return errors::Internal("expected layout missing");

  return *layout;
}

StatusOr<std::vector<absl::optional<Layout>>> ExtractLayoutFromOp(
    mlir::Operation* op, std::string attr_name) {
  std::vector<absl::optional<Layout>> outs;
  outs.reserve(op->getNumResults());

  // If v2 layout propagation algorithm is used, parse layout from DTensorLayout
  // op.
  if (OpUsesV2LayoutAnnotation(op)) {
    for (auto op_result : op->getOpResults()) {
      outs.emplace_back(
          llvm::cast<mlir::TF::DTensorLayout>(*op_result.getUsers().begin())
              .layout());
    }
  } else {
    auto serialized_layouts = op->getAttrOfType<mlir::ArrayAttr>(attr_name);
    if (!serialized_layouts) return outs;

    for (auto const& attr : serialized_layouts) {
      auto attr_str = attr.cast<mlir::StringAttr>().getValue().str();
      if (!attr_str.empty()) {
        TF_ASSIGN_OR_RETURN(auto layout, Layout::FromString(attr_str));
        outs.emplace_back(std::move(layout));
      } else {
        outs.emplace_back(absl::nullopt);
      }
    }
  }
  return outs;
}

StatusOr<std::vector<absl::optional<Layout>>> ExtractLayoutFromOp(
    mlir::Operation* op) {
  return ExtractLayoutFromOp(op, kLayoutAttr);
}

StatusOr<std::vector<Layout>> ExtractRequiredLayoutFromOp(mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(std::vector<absl::optional<Layout>> optional_layouts,
                      ExtractLayoutFromOp(op));
  std::vector<Layout> layouts;
  for (const absl::optional<Layout>& layout : optional_layouts) {
    if (!layout) return errors::Internal("expected layout missing");
    layouts.emplace_back(*layout);
  }

  return layouts;
}

StatusOr<Mesh> ExtractDeviceMeshEnclosingCluster(mlir::Operation* op) {
  auto enclosing_cluster = op->getParentOfType<mlir::tf_device::ClusterOp>();
  if (!enclosing_cluster)
    return errors::InvalidArgument("op is not inside a device mesh cluster.");

  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshFromOp(enclosing_cluster));
  if (!mesh)
    return errors::InvalidArgument(
        "op's enclosing device cluster does not have mesh defined.");

  return *mesh;
}

StatusOr<absl::optional<Mesh>> ExtractDeviceMeshFromOp(mlir::Operation* op) {
  absl::optional<Mesh> extracted_mesh;
  if (op == nullptr) return extracted_mesh;

  auto mesh_str_attr = op->getAttrOfType<mlir::StringAttr>(kMeshAttr);
  if (!mesh_str_attr) return extracted_mesh;

  TF_ASSIGN_OR_RETURN(Mesh mesh,
                      Mesh::FromString(mesh_str_attr.getValue().str()));

  extracted_mesh.emplace(std::move(mesh));
  return extracted_mesh;
}

StatusOr<absl::optional<Layout>> ExtractLayoutFromOperand(mlir::Value operand) {
  if (auto op_result = operand.dyn_cast<mlir::OpResult>()) {
    mlir::Operation* op = op_result.getDefiningOp();
    absl::optional<Layout> out;
    if (auto layout_op = llvm::dyn_cast<mlir::TF::DTensorLayout>(op)) {
      out.emplace(layout_op.layout());
    } else {
      const int result_number = op_result.getResultNumber();
      TF_ASSIGN_OR_RETURN(auto layouts, ExtractLayoutFromOp(op, kLayoutAttr));

      if (layouts.empty()) return out;

      if (result_number >= layouts.size()) {
        return errors::Internal(
            "Expect to extract the ", result_number,
            "-th output's layout, but "
            "only see ",
            layouts.size(), " outputs: ", op->getName().getStringRef().str());
      }
      out.swap(layouts[result_number]);
    }
    return out;
  }

  auto block_arg = operand.dyn_cast<mlir::BlockArgument>();
  if (!block_arg)
    return errors::Internal(
        "Operand is not either a OpResult or a BlockArgument. This should not "
        "happen.");
  auto func_op = mlir::dyn_cast_or_null<mlir::func::FuncOp>(
      block_arg.getOwner()->getParentOp());
  if (!func_op) {
    return errors::InvalidArgument("op must be enclosed by a function");
  }

  absl::optional<Layout> extracted_layout;
  auto layout_attr = func_op.getArgAttrOfType<mlir::StringAttr>(
      block_arg.getArgNumber(), kCustomDeviceAttr);
  if (!layout_attr) return extracted_layout;

  TF_ASSIGN_OR_RETURN(auto layout,
                      Layout::FromString(layout_attr.getValue().str()));
  extracted_layout.emplace(std::move(layout));
  return extracted_layout;
}

StatusOr<Layout> ExtractRequiredLayoutFromOperand(mlir::Value operand) {
  TF_ASSIGN_OR_RETURN(absl::optional<Layout> layout,
                      ExtractLayoutFromOperand(operand));
  if (!layout) return errors::Internal("expected layout missing");

  return *layout;
}

StatusOr<std::vector<Layout>> ExtractRequiredLayoutFromOperands(
    mlir::Operation* op) {
  std::vector<Layout> layouts;
  for (const auto& operand : op->getOpOperands()) {
    TF_ASSIGN_OR_RETURN(auto operand_layout,
                        ExtractRequiredLayoutFromOperand(operand.get()));
    layouts.emplace_back(operand_layout);
  }
  return layouts;
}

void SetLayoutOnOp(mlir::Operation* op, mlir::OpBuilder builder,
                   absl::Span<const absl::optional<Layout>> layouts) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSlayout_parsingDTcc mht_1(mht_1_v, 401, "", "./tensorflow/dtensor/mlir/layout_parsing.cc", "SetLayoutOnOp");

  llvm::SmallVector<std::string, 8> serialized_layouts;
  for (auto const& layout : layouts) {
    serialized_layouts.emplace_back(layout.has_value() ? layout->ToString()
                                                       : "");
  }
  op->setAttr(kLayoutAttr,
              builder.getStrArrayAttr(llvm::SmallVector<llvm::StringRef, 8>(
                  serialized_layouts.begin(), serialized_layouts.end())));
}

void SetLayoutOnOp(mlir::Operation* op,
                   absl::Span<const absl::optional<Layout>> layouts) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSlayout_parsingDTcc mht_2(mht_2_v, 416, "", "./tensorflow/dtensor/mlir/layout_parsing.cc", "SetLayoutOnOp");

  SetLayoutOnOp(op, mlir::OpBuilder(op), layouts);
}

void SetSingleLayoutOnOp(mlir::Operation* op, const Layout& layout) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSlayout_parsingDTcc mht_3(mht_3_v, 423, "", "./tensorflow/dtensor/mlir/layout_parsing.cc", "SetSingleLayoutOnOp");

  SetLayoutOnOp(op, mlir::OpBuilder(op), {absl::optional<Layout>(layout)});
}

StatusOr<absl::optional<Layout>> ExtractLayoutFromFunctionReturnAttr(
    mlir::func::ReturnOp return_op, const int return_index) {
  absl::optional<Layout> layout;
  // If value feeds into func op return op, then check to see if layout
  // attribute is set for the return value.
  auto function = return_op->getParentOfType<mlir::func::FuncOp>();
  auto layout_attr_from_func_result =
      function.getResultAttrOfType<mlir::StringAttr>(return_index,
                                                     kCustomDefaultLayoutAttr);
  if (!layout_attr_from_func_result) return layout;

  const std::string layout_string =
      layout_attr_from_func_result.getValue().str();
  auto result_layout_or_status = Layout::FromString(layout_string);
  if (!result_layout_or_status.ok())
    return errors::InvalidArgument(
        llvm::formatv("Malformed default return layout received. {0} Received "
                      "layout : {1}",
                      result_layout_or_status.status().error_message(),
                      layout_string)
            .str());

  layout.emplace(result_layout_or_status.ValueOrDie());
  return layout;
}

}  // namespace dtensor
}  // namespace tensorflow
