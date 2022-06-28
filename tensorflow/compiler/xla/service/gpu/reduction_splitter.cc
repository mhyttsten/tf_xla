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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSreduction_splitterDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSreduction_splitterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSreduction_splitterDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/reduction_splitter.h"

#include <algorithm>

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace gpu {

class ReductionSplitterVisitor : public DfsHloRewriteVisitor {
 public:
  Status HandleReduce(HloInstruction *reduce) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSreduction_splitterDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/xla/service/gpu/reduction_splitter.cc", "HandleReduce");

    VLOG(4) << "Input: " << reduce->ToString();

    // Reductions with contiguous dimensions are lowered to efficient code. No
    // need to split such ops.
    if (IsReductionFromOrToContiguousDimensions(*reduce)) {
      return Status::OK();
    }
    if (reduce->dimensions().size() < 2) {
      return Status::OK();
    }
    if (!reduce->shape().IsArray()) {
      // TODO(cheshire): Handle variadic reduction.
      return Status::OK();
    }

    HloInstruction *operand = reduce->mutable_operand(0);
    const Shape &shape = operand->shape();
    CHECK(shape == LayoutUtil::GetWithDefaultLayout(shape))
        << "Default layout should be enforced on reduction operand";
    // Verify that contiguous dimensions have been grouped by the
    // ReductionDimensionGrouper pass.
    for (int64_t i = 0; i < reduce->dimensions().size(); ++i) {
      for (int64_t j = i + 1; j < reduce->dimensions().size(); ++j) {
        CHECK(abs(reduce->dimensions(i) - reduce->dimensions(j)) > 1)
            << "Reduction dimensions must not be consecutive";
      }
    }

    // The reduce op has non-contiguous dimensions. Look for the dimension with
    // the largest shape dimension. Reducing along this dimension first will
    // reduce the output size most effectively.
    int64_t max_shape_dim = 0;
    int64_t max_reduce_dim = 0;
    const auto &input_shape = reduce->operand(0)->shape();
    for (int64_t i = 0; i < reduce->dimensions().size(); ++i) {
      if (input_shape.dimensions(reduce->dimensions(i)) > max_shape_dim) {
        max_reduce_dim = reduce->dimensions(i);
        max_shape_dim = input_shape.dimensions(max_reduce_dim);
      }
    }
    // TODO(tjoerg): Run microbenchmarks to tune this threshold.
    if (max_shape_dim < 128) {
      return Status::OK();
    }

    // Split the reduction into a pre-reduction and a final reduction.
    VLOG(3) << "Splitting reduction " << reduce->name() << " at dimension "
            << max_reduce_dim;
    std::vector<int64_t> pre_reduce_dims;
    pre_reduce_dims.push_back(max_reduce_dim);
    std::vector<int64_t> pre_reduce_shape_dims(input_shape.dimensions().begin(),
                                               input_shape.dimensions().end());
    pre_reduce_shape_dims.erase(pre_reduce_shape_dims.begin() + max_reduce_dim);
    Shape pre_reduce_shape = ShapeUtil::MakeShape(
        reduce->shape().element_type(), pre_reduce_shape_dims);
    std::unique_ptr<HloInstruction> pre_reduce = HloInstruction::CreateReduce(
        pre_reduce_shape, reduce->mutable_operand(0),
        reduce->mutable_operand(1), pre_reduce_dims, reduce->to_apply());
    pre_reduce->set_metadata(reduce->metadata());

    std::vector<int64_t> final_reduce_dims(reduce->dimensions().begin(),
                                           reduce->dimensions().end());
    final_reduce_dims.erase(
        std::remove(final_reduce_dims.begin(), final_reduce_dims.end(),
                    max_reduce_dim),
        final_reduce_dims.end());
    for (int64_t i = 0; i < final_reduce_dims.size(); ++i) {
      if (final_reduce_dims[i] > max_reduce_dim) {
        final_reduce_dims[i]--;
      }
    }
    std::unique_ptr<HloInstruction> final_reduce = HloInstruction::CreateReduce(
        reduce->shape(),
        reduce->parent()->AddInstruction(std::move(pre_reduce)),
        reduce->mutable_operand(1), final_reduce_dims, reduce->to_apply());
    return ReplaceWithNewInstruction(reduce, std::move(final_reduce));
  }
};

StatusOr<bool> ReductionSplitter::Run(HloModule *module) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSreduction_splitterDTcc mht_1(mht_1_v, 282, "", "./tensorflow/compiler/xla/service/gpu/reduction_splitter.cc", "ReductionSplitter::Run");

  TF_ASSIGN_OR_RETURN(bool changed,
                      ReductionSplitterVisitor().RunOnModule(module));
  return changed;
}

}  // namespace gpu
}  // namespace xla
