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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSbias_add_spmd_expanderDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSbias_add_spmd_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSbias_add_spmd_expanderDTcc() {
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

#include "tensorflow/dtensor/mlir/expansions/bias_add_spmd_expander.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/expansions/elementwise_spmd_expander.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"

namespace tensorflow {
namespace dtensor {

namespace {

int get_c_dimension_idx(const Layout& layout, llvm::StringRef data_format) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSbias_add_spmd_expanderDTcc mht_0(mht_0_v, 209, "", "./tensorflow/dtensor/mlir/expansions/bias_add_spmd_expander.cc", "get_c_dimension_idx");

  // If format is "N...C", the bias is added to the last dimension.
  int c_dim_idx = layout.sharding_spec_strs().size() - 1;
  if (data_format.startswith("NC")) {
    // If format is "NC...", the bias is added to the 'C' dimension.
    c_dim_idx = layout.sharding_spec_strs().size() - 3;
  }
  return c_dim_idx;
}

}  // namespace

StatusOr<mlir::Operation*> BiasAddExpander::ExpandOp(mlir::Operation* op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSbias_add_spmd_expanderDTcc mht_1(mht_1_v, 224, "", "./tensorflow/dtensor/mlir/expansions/bias_add_spmd_expander.cc", "BiasAddExpander::ExpandOp");

  TF_ASSIGN_OR_RETURN(auto output_layout,
                      ExtractRequiredSingleLayoutFromOp(op));
  mlir::TF::BiasAddOp bias_add_op = llvm::cast<mlir::TF::BiasAddOp>(op);
  const llvm::StringRef data_format = bias_add_op.data_format();
  const int c_dim_idx = get_c_dimension_idx(output_layout, data_format);

  // Bias add op has 2 inputs: value and bias.
  assert(op->getOpOperands().size() == 2);
  mlir::OpOperand& input = op->getOpOperand(0);
  TF_ASSIGN_OR_RETURN(Layout input_layout,
                      ExtractRequiredLayoutFromOperand(input.get()));

  mlir::OpOperand& bias = op->getOpOperand(1);

  TF_ASSIGN_OR_RETURN(const Layout bias_layout,
                      ExtractRequiredLayoutFromOperand(bias.get()));

  // Check if output is sharded more, change input layout to match output
  // layout.
  int64_t num_input_shards =
      input_layout.num_shards_for_dim(input_layout.dim(c_dim_idx));
  int64_t num_output_shards =
      output_layout.num_shards_for_dim(output_layout.dim(c_dim_idx));

  if (num_input_shards < num_output_shards) {
    mlir::Value output;
    std::vector<std::string> input_new_specs =
        output_layout.sharding_spec_strs();
    TF_ASSIGN_OR_RETURN(
        const Layout new_input_layout,
        Layout::GetLayout(input_new_specs, input_layout.mesh()));
    TF_ASSIGN_OR_RETURN(
        output, EmitRelayout(input.get(), input_layout, new_input_layout));
    input.set(output);
    input_layout = new_input_layout;
  }

  // Map bias layout sharding to match sharding for 'c' dimension of input, if
  // not same already.
  if (bias_layout.sharding_spec(0) != input_layout.sharding_spec(c_dim_idx)) {
    mlir::Value output;

    std::vector<std::string> bias_new_specs = {
        input_layout.sharding_spec_strs()[c_dim_idx]};
    TF_ASSIGN_OR_RETURN(const Layout new_bias_layout,
                        Layout::GetLayout(bias_new_specs, bias_layout.mesh()));
    TF_ASSIGN_OR_RETURN(output,
                        EmitRelayout(bias.get(), bias_layout, new_bias_layout));
    bias.set(output);
  }

  // Perform SPMD operation locally
  mlir::Operation* new_local_op = InferSPMDExpandedLocalShape(op);

  // Convert result layout to output layout.
  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  TF_ASSIGN_OR_RETURN(mlir::Value relayout_output,
                      EmitRelayout(new_local_op->getOpResult(0), input_layout,
                                   output_layout, &newly_created_ops));
  op->getResult(0).replaceAllUsesExcept(relayout_output, newly_created_ops);
  return relayout_output.getDefiningOp();
}

StatusOr<llvm::DenseMap<int, Layout>> BiasAddExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // If we do not have an input layout then do not infer an output layout.
  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  Layout input_layout = input_layouts.lookup(0);
  mlir::TF::BiasAddOp bias_add_op = llvm::cast<mlir::TF::BiasAddOp>(op);
  llvm::StringRef data_format = bias_add_op.data_format();
  int c_dim_idx = get_c_dimension_idx(input_layout, data_format);

  std::vector<std::string> new_output_layout_specs =
      input_layout.sharding_spec_strs();
  if (Layout::IsUnshardedDimension(new_output_layout_specs[c_dim_idx]) &&
      input_layouts.find(1) != input_layouts.end()) {
    // Shard c_dim using bias sharding as long as the sharding spec is not
    // already used in input for some other dimension.
    Layout bias_layout = input_layouts.lookup(1);
    std::string bias_sharding = bias_layout.sharding_spec(0);
    if (std::find(new_output_layout_specs.begin(),
                  new_output_layout_specs.end(),
                  bias_sharding) == new_output_layout_specs.end()) {
      new_output_layout_specs[c_dim_idx] = bias_layout.sharding_spec(0);
    }
  }
  TF_ASSIGN_OR_RETURN(
      Layout new_output_layout,
      Layout::GetLayout(new_output_layout_specs, input_layout.mesh()));

  return llvm::DenseMap<int, Layout>({{0, new_output_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>> BiasAddExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();

  llvm::DenseMap<int, Layout> input_layouts;

  // If output layout is given, match input_layout and bias layout to match
  // it.
  Layout output_layout = output_layouts.lookup(0);

  // Bias layout should match 'C' dimension of input layout.
  mlir::TF::BiasAddOp bias_add_op = llvm::cast<mlir::TF::BiasAddOp>(op);
  llvm::StringRef data_format = bias_add_op.data_format();
  const int c_dim_idx = get_c_dimension_idx(output_layout, data_format);

  std::vector<std::string> bias_new_specs = {
      output_layout.sharding_spec_strs()[c_dim_idx]};
  TF_ASSIGN_OR_RETURN(Layout new_bias_layout,
                      Layout::GetLayout(bias_new_specs, output_layout.mesh()));

  return llvm::DenseMap<int, Layout>(
      {{0, output_layout}, {1, new_bias_layout}});
}
}  // namespace dtensor
}  // namespace tensorflow
