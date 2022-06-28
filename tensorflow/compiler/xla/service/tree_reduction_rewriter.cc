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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStree_reduction_rewriterDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStree_reduction_rewriterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStree_reduction_rewriterDTcc() {
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

#include "tensorflow/compiler/xla/service/tree_reduction_rewriter.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {

class ReductionRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit ReductionRewriterVisitor(int64_t reduce_window_size)
      : reduce_window_size_(reduce_window_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStree_reduction_rewriterDTcc mht_0(mht_0_v, 212, "", "./tensorflow/compiler/xla/service/tree_reduction_rewriter.cc", "ReductionRewriterVisitor");
}

  Status HandleReduce(HloInstruction *hlo) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStree_reduction_rewriterDTcc mht_1(mht_1_v, 217, "", "./tensorflow/compiler/xla/service/tree_reduction_rewriter.cc", "HandleReduce");

    HloInstruction *reduced_op = hlo->mutable_operand(0);
    HloInstruction *initial_value = hlo->mutable_operand(1);
    const Shape &input_shape = reduced_op->shape();
    const Shape &reduce_shape = hlo->shape();

    if (!reduce_shape.IsArray()) {
      // TODO(b/210786051): Implement tree reduction rewrite for variadic
      // reductions on CPU as well.
      VLOG(1) << "Skipping rewrite for variadic reduction";
      return Status::OK();
    }

    // All of the reduced dimensions is smaller than the window size,
    // do not perform the rewrite.
    if (absl::c_all_of(hlo->dimensions(), [&](int64_t reduced_dim) {
          return input_shape.dimensions(reduced_dim) <= reduce_window_size_;
        })) {
      VLOG(1) << "Skipping tree reduction rewrite: all reduced dimensions are "
                 "smaller than "
              << reduce_window_size_;
      return Status::OK();
    }

    std::vector<int64_t> window_dimensions;
    std::vector<int64_t> window_strides;
    for (int64_t dim_idx = 0; dim_idx < input_shape.rank(); dim_idx++) {
      if (!absl::c_linear_search(hlo->dimensions(), dim_idx)) {
        window_dimensions.push_back(1);
        window_strides.push_back(1);
        continue;
      }

      int64_t window_size_for_dim =
          std::min(input_shape.dimensions(dim_idx), reduce_window_size_);

      window_dimensions.push_back(window_size_for_dim);
      window_strides.push_back(window_size_for_dim);
    }

    std::vector<std::pair<int64_t, int64_t>> padding =
        MakePadding(input_shape.dimensions(), window_dimensions, window_strides,
                    Padding::kSame);

    TF_ASSIGN_OR_RETURN(
        Window window, ShapeInference::InferWindowFromDimensions(
                           window_dimensions, window_strides, padding, {}, {}));

    TF_ASSIGN_OR_RETURN(Shape intermediate_shape,
                        ShapeInference::InferReduceWindowShape(
                            input_shape, initial_value->shape(), window));

    HloInstruction *reduce_window =
        hlo->parent()->AddInstruction(HloInstruction::CreateReduceWindow(
            intermediate_shape, reduced_op, initial_value, window,
            hlo->to_apply()));

    std::unique_ptr<HloInstruction> new_output =
        HloInstruction::CreateReduce(reduce_shape, reduce_window, initial_value,
                                     hlo->dimensions(), hlo->to_apply());

    return ReplaceWithNewInstruction(hlo, std::move(new_output));
  }

 private:
  int64_t reduce_window_size_;
};

StatusOr<bool> TreeReductionRewriter::Run(HloModule *module) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStree_reduction_rewriterDTcc mht_2(mht_2_v, 288, "", "./tensorflow/compiler/xla/service/tree_reduction_rewriter.cc", "TreeReductionRewriter::Run");

  ReductionRewriterVisitor visitor(reduce_window_size_);
  bool changed = false;
  for (const auto &computation : module->MakeNonfusionComputations()) {
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
    changed |= visitor.changed();
  }

  return changed;
}

}  // end namespace xla
