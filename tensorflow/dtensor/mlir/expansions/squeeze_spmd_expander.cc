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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSsqueeze_spmd_expanderDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSsqueeze_spmd_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSsqueeze_spmd_expanderDTcc() {
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

#include "tensorflow/dtensor/mlir/expansions/squeeze_spmd_expander.h"

#include <utility>

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {
namespace {

std::set<int64_t> GetSqueezeDims(mlir::Operation* op, int64_t rank) {
  auto array_attribute = op->getAttrOfType<mlir::ArrayAttr>("squeeze_dims");
  std::set<int64_t> squeeze_dims;
  if (array_attribute) {
    auto attr_list = array_attribute.getValue().vec();
    for (const auto& attr : attr_list) {
      int64_t dim = attr.cast<mlir::IntegerAttr>().getValue().getSExtValue();
      // Offset the negative indices to positive range.
      squeeze_dims.insert((dim + rank) % rank);
    }
  }
  return squeeze_dims;
}

}  // namespace

StatusOr<llvm::DenseMap<int, Layout>> SqueezeSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // If there is no tensor layout then do not infer any output layouts.
  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  const Layout& input_layout = input_layouts.lookup(0);
  TF_ASSIGN_OR_RETURN(auto shape, ExtractGlobalInputShape(op->getOpOperand(0)));
  std::set<int64_t> squeeze_dims = GetSqueezeDims(op, /*rank=*/shape.size());

  std::vector<ShardingSpec> layout_specs;
  layout_specs.reserve(input_layout.rank());
  for (int64 i = 0; i < input_layout.rank(); ++i) {
    if (squeeze_dims.empty()) {
      if (shape[i] > 1) {
        layout_specs.push_back(input_layout.dim(i));
      }
    } else {
      if (squeeze_dims.find(i) == squeeze_dims.end()) {
        layout_specs.push_back(input_layout.dim(i));
      }
    }
  }

  TF_ASSIGN_OR_RETURN(const Layout output_layout,
                      Layout::GetLayout(layout_specs, input_layout.mesh()));
  return llvm::DenseMap<int, Layout>({{0, output_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>>
SqueezeSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  // If there is no output layout present then do not infer any operand layouts.
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();

  const Layout& output_layout = output_layouts.lookup(0);
  TF_ASSIGN_OR_RETURN(auto shape, ExtractGlobalInputShape(op->getOpOperand(0)));
  std::set<int64_t> squeeze_dims = GetSqueezeDims(op, /*rank=*/shape.size());

  ShardingSpec unsharded_spec;
  unsharded_spec.set_sharding_spec(Layout::kUnshardedDim);

  std::vector<ShardingSpec> layout_specs;
  layout_specs.reserve(output_layout.rank());
  size_t j = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (squeeze_dims.empty()) {
      if (shape[i] > 1) {
        layout_specs.push_back(output_layout.dim(j++));
      } else {
        layout_specs.push_back(unsharded_spec);
      }
    } else {
      if (squeeze_dims.find(i) == squeeze_dims.end()) {
        layout_specs.push_back(output_layout.dim(j++));
      } else {
        layout_specs.push_back(unsharded_spec);
      }
    }
  }

  TF_ASSIGN_OR_RETURN(const Layout input_layout,
                      Layout::GetLayout(layout_specs, output_layout.mesh()));
  return llvm::DenseMap<int, Layout>({{0, input_layout}});
}

StatusOr<mlir::Operation*> SqueezeSPMDExpander::ExpandOp(mlir::Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSsqueeze_spmd_expanderDTcc mht_0(mht_0_v, 284, "", "./tensorflow/dtensor/mlir/expansions/squeeze_spmd_expander.cc", "SqueezeSPMDExpander::ExpandOp");

  auto squeeze_op = mlir::cast<mlir::TF::SqueezeOp>(op);
  TF_ASSIGN_OR_RETURN(auto layout, ExtractSingleLayoutFromOp(op));

  if (!layout) {
    return errors::InvalidArgument(
        "layout of SqueezeOp must be known before SPMD expansion.");
  }

  TF_ASSIGN_OR_RETURN(auto input_shape,
                      ExtractGlobalInputShape(op->getOpOperand(0)));
  std::set<int64_t> squeeze_dims =
      GetSqueezeDims(op, /*rank=*/input_shape.size());

  if (squeeze_dims.empty()) {
    // If the squeeze dim is empty, make sure the squeeze only happens on the
    // dims that is not sharded and has global_shape is 1. Otherwise if the
    // local shape happens to have size 1 on the dim, it got squeezed
    // unexpected.
    for (int i = 0; i < input_shape.size(); ++i) {
      // Global shape 1 implies the dim cannot be sharded -- worst case it can
      // be sharded over a mesh with dim size 1, and we would squeeze it as is.
      if (input_shape[i] == 1) {
        squeeze_dims.insert(i);
      }
    }
    mlir::OpBuilder builder(squeeze_op);
    squeeze_op->setAttr("squeeze_dims",
                        builder.getI64ArrayAttr(llvm::SmallVector<int64_t>(
                            squeeze_dims.begin(), squeeze_dims.end())));
  }

  return InferSPMDExpandedLocalShape(squeeze_op);
}

}  // namespace dtensor
}  // namespace tensorflow
