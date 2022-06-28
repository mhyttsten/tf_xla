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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HORIZONTAL_LOOP_FUSION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HORIZONTAL_LOOP_FUSION_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPShorizontal_loop_fusionDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPShorizontal_loop_fusionDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPShorizontal_loop_fusionDTh() {
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


#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// This optimization pass horizontally fuses computations for reducing kernel
// launch overhead while increasing kernel launch dims on GPU. The initial
// motivation of this horizontal fusion is due to the observation that the
// training optimizer phase (e.g., AdamOptimizer and L2Loss, etc.) typically
// has many small kernels as a result of applying the same formula on many
// training parameters (or variables in Tensorflow). Fusing these small
// kernels, hence, provides performance gain.
//
// Theoretically speaking, we may implement a cycle detection algorithm to make
// sure no cycles are created after fusion. However, cycle detection check is
// somewhat cumbersome; also, we observe that naive horizontal fusion of
// arbitrary kernels may not be profitable due to control divergence and
// possible increase of memory bandwidth pressure due to uncoalesced memory
// accesses (note that horizontal fusion does not change the amount of memory
// read+written at all). In practice, a simple yet effective heuristic is used
// to avoid these issues while addressing the known beneficial cases. That is,
// we simply search for fusion candidates by looking for instructions whose
// outputs are all consumed by the same instruction. This catches the cases in
// the training optimizer phase, as the candidate instructions are typically
// consumed only by the ROOT tuple of the entry computation.
//
// The following illustrates the mechanism of the horizontal fusion. Before
// fusion, there are two trivial kernels in the illustrating example. One has
// only a Mul op, while the other consists of only an Add op. Since they are
// only consumed by the same (ROOT) tuple instruction, horizontal fusion is
// triggered.
//
// i0 i1   i2 i3
//  | |     | |
//  v v     v v
//  Mul     Add
//   |       |
//   v       v
//  (ROOT) tuple
//
// We horizontally fuse them into the below pattern.
//
// i0 i1   i2 i3       +++ (Slice) Input Fusion
//  | |     | |          +
//  v v     v v          +
//  Mul     Add          +
//   |       |           +
//   v       v           +
// Reshape0  Reshape1    +
//   |       |           +
//   v       v           +
//  Concatenate          +
//   |       |           +
//   v       v           +
//  Slice0  Slice1     +++
//   |       |
//   v       v
// Reshape2  Reshape3
//   |       |
//   v       v
//  (ROOT) tuple
//
// Note that this fusion style provides an important advantage that kernels of
// different shapes can be horizontally fused. The first pair of reshapes
// (i.e., Reshape0 and Reshape1) reshape the dims to 1 dimension, so that the
// outputs of the fused kernels can (always) be concatenated. The second pair
// of reshapes (Reshape2 and Reshape3) restore the original shapes to the
// output tensors.
//
// No extra copies are introduced by the horizontal fusion. Besides Reshape2
// and Reshape3, the other instructions are fused into an input fusion; the
// output dims of the concatenate will be used as the kernel launch dims.
// Instruction bitcasts can be used for Reshape2 and Reshape3 as long as the
// outputs of Mul and Add are row-major.
class GpuHorizontalLoopFusion : public HloModulePass {
 public:
  GpuHorizontalLoopFusion() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPShorizontal_loop_fusionDTh mht_0(mht_0_v, 267, "", "./tensorflow/compiler/xla/service/gpu/horizontal_loop_fusion.h", "GpuHorizontalLoopFusion");
}

  absl::string_view name() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPShorizontal_loop_fusionDTh mht_1(mht_1_v, 272, "", "./tensorflow/compiler/xla/service/gpu/horizontal_loop_fusion.h", "name");

    return "gpu_horizontal_loop_fusion";
  }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  StatusOr<bool> RunOnComputation(HloComputation*);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_HORIZONTAL_LOOP_FUSION_H_
