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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_DFS_HLO_VISITOR_WITH_DEFAULT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_DFS_HLO_VISITOR_WITH_DEFAULT_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh() {
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


#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

// DfsHloVisitor with default action based on the HloInstruction being visited.
// Users should not use this class directly, but use the type aliases
// DfsHloVisitorWithDefault/ConstDfsHloVisitorWithDefault instead.
//
// Do *not* add an override to this class if the opcode is covered by
// HandleElementwiseUnary/Binary. These opcode handlers dispatch to
// HandleElementwiseUnary/Binary in DfsHloVisitorBase. Adding such a handler
// here will break passes which rely on the HandleElementwiseUnary/Binary
// handling these opcodes.
template <typename HloInstructionPtr>
class DfsHloVisitorWithDefaultBase
    : public DfsHloVisitorBase<HloInstructionPtr> {
 public:
  DfsHloVisitorWithDefaultBase() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_0(mht_0_v, 216, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "DfsHloVisitorWithDefaultBase");
}
  ~DfsHloVisitorWithDefaultBase() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_1(mht_1_v, 220, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "~DfsHloVisitorWithDefaultBase");
}

  // Default action performed on HloInstruction.
  virtual Status DefaultAction(HloInstructionPtr hlo_instruction) = 0;

  Status HandleElementwiseUnary(HloInstructionPtr hlo) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_2(mht_2_v, 228, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleElementwiseUnary");

    return DefaultAction(hlo);
  }
  Status HandleElementwiseBinary(HloInstructionPtr hlo) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_3(mht_3_v, 234, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleElementwiseBinary");

    return DefaultAction(hlo);
  }

  Status HandleBatchNormTraining(HloInstructionPtr hlo) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_4(mht_4_v, 241, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleBatchNormTraining");

    return DefaultAction(hlo);
  }

  Status HandleBatchNormInference(HloInstructionPtr hlo) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_5(mht_5_v, 248, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleBatchNormInference");

    return DefaultAction(hlo);
  }

  Status HandleBatchNormGrad(HloInstructionPtr hlo) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_6(mht_6_v, 255, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleBatchNormGrad");

    return DefaultAction(hlo);
  }

  Status HandleClamp(HloInstructionPtr clamp) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_7(mht_7_v, 262, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleClamp");

    return DefaultAction(clamp);
  }
  Status HandleConcatenate(HloInstructionPtr concatenate) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_8(mht_8_v, 268, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleConcatenate");

    return DefaultAction(concatenate);
  }
  Status HandleSelect(HloInstructionPtr select) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_9(mht_9_v, 274, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleSelect");

    return DefaultAction(select);
  }
  Status HandleTupleSelect(HloInstructionPtr tuple_select) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_10(mht_10_v, 280, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleTupleSelect");

    return DefaultAction(tuple_select);
  }
  Status HandleDot(HloInstructionPtr dot) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_11(mht_11_v, 286, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleDot");

    return DefaultAction(dot);
  }
  Status HandleConvolution(HloInstructionPtr convolution) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_12(mht_12_v, 292, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleConvolution");

    return DefaultAction(convolution);
  }
  Status HandleFft(HloInstructionPtr fft) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_13(mht_13_v, 298, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleFft");

    return DefaultAction(fft);
  }
  Status HandleTriangularSolve(HloInstructionPtr hlo) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_14(mht_14_v, 304, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleTriangularSolve");

    return DefaultAction(hlo);
  }
  Status HandleCholesky(HloInstructionPtr hlo) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_15(mht_15_v, 310, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleCholesky");

    return DefaultAction(hlo);
  }
  Status HandleOptimizationBarrier(HloInstructionPtr hlo) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_16(mht_16_v, 316, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleOptimizationBarrier");

    return DefaultAction(hlo);
  }
  Status HandleAllGather(HloInstructionPtr crs) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_17(mht_17_v, 322, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleAllGather");

    return DefaultAction(crs);
  }
  Status HandleAllGatherStart(HloInstructionPtr crs) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_18(mht_18_v, 328, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleAllGatherStart");

    return DefaultAction(crs);
  }
  Status HandleAllGatherDone(HloInstructionPtr crs) override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_19(mht_19_v, 334, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleAllGatherDone");

    return DefaultAction(crs);
  }
  Status HandleAllReduce(HloInstructionPtr crs) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_20(mht_20_v, 340, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleAllReduce");

    return DefaultAction(crs);
  }
  Status HandleReduceScatter(HloInstructionPtr hlo) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_21(mht_21_v, 346, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleReduceScatter");

    return DefaultAction(hlo);
  }
  Status HandleAllReduceStart(HloInstructionPtr hlo) override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_22(mht_22_v, 352, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleAllReduceStart");

    return DefaultAction(hlo);
  }
  Status HandleAllReduceDone(HloInstructionPtr hlo) override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_23(mht_23_v, 358, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleAllReduceDone");

    return DefaultAction(hlo);
  }
  Status HandleAllToAll(HloInstructionPtr hlo) override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_24(mht_24_v, 364, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleAllToAll");

    return DefaultAction(hlo);
  }
  Status HandleCollectivePermute(HloInstructionPtr hlo) override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_25(mht_25_v, 370, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleCollectivePermute");

    return DefaultAction(hlo);
  }
  Status HandleCollectivePermuteStart(HloInstructionPtr hlo) override {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_26(mht_26_v, 376, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleCollectivePermuteStart");

    return DefaultAction(hlo);
  }
  Status HandleCollectivePermuteDone(HloInstructionPtr hlo) override {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_27(mht_27_v, 382, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleCollectivePermuteDone");

    return DefaultAction(hlo);
  }
  Status HandleReplicaId(HloInstructionPtr hlo) override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_28(mht_28_v, 388, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleReplicaId");

    return DefaultAction(hlo);
  }
  Status HandlePartitionId(HloInstructionPtr hlo) override {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_29(mht_29_v, 394, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandlePartitionId");

    return DefaultAction(hlo);
  }
  Status HandleRng(HloInstructionPtr random) override {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_30(mht_30_v, 400, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleRng");

    return DefaultAction(random);
  }
  Status HandleRngBitGenerator(HloInstructionPtr random) override {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_31(mht_31_v, 406, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleRngBitGenerator");

    return DefaultAction(random);
  }
  Status HandleRngGetAndUpdateState(HloInstructionPtr random) override {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_32(mht_32_v, 412, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleRngGetAndUpdateState");

    return DefaultAction(random);
  }
  Status HandleInfeed(HloInstructionPtr infeed) override {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_33(mht_33_v, 418, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleInfeed");

    return DefaultAction(infeed);
  }
  Status HandleOutfeed(HloInstructionPtr outfeed) override {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_34(mht_34_v, 424, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleOutfeed");

    return DefaultAction(outfeed);
  }
  Status HandleReverse(HloInstructionPtr reverse) override {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_35(mht_35_v, 430, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleReverse");

    return DefaultAction(reverse);
  }
  Status HandleSort(HloInstructionPtr sort) override {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_36(mht_36_v, 436, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleSort");

    return DefaultAction(sort);
  }
  Status HandleConstant(HloInstructionPtr constant) override {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_37(mht_37_v, 442, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleConstant");

    return DefaultAction(constant);
  }
  Status HandleIota(HloInstructionPtr iota) override {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_38(mht_38_v, 448, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleIota");

    return DefaultAction(iota);
  }
  Status HandleGetTupleElement(HloInstructionPtr get_tuple_element) override {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_39(mht_39_v, 454, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleGetTupleElement");

    return DefaultAction(get_tuple_element);
  }
  Status HandleParameter(HloInstructionPtr parameter) override {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_40(mht_40_v, 460, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleParameter");

    return DefaultAction(parameter);
  }
  Status HandleFusion(HloInstructionPtr fusion) override {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_41(mht_41_v, 466, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleFusion");

    return DefaultAction(fusion);
  }
  Status HandleCall(HloInstructionPtr call) override {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_42(mht_42_v, 472, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleCall");

    return DefaultAction(call);
  }
  Status HandleCustomCall(HloInstructionPtr custom_call) override {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_43(mht_43_v, 478, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleCustomCall");

    return DefaultAction(custom_call);
  }
  Status HandleSlice(HloInstructionPtr slice) override {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_44(mht_44_v, 484, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleSlice");

    return DefaultAction(slice);
  }
  Status HandleDynamicSlice(HloInstructionPtr dynamic_slice) override {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_45(mht_45_v, 490, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleDynamicSlice");

    return DefaultAction(dynamic_slice);
  }
  Status HandleDynamicUpdateSlice(
      HloInstructionPtr dynamic_update_slice) override {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_46(mht_46_v, 497, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleDynamicUpdateSlice");

    return DefaultAction(dynamic_update_slice);
  }
  Status HandleTuple(HloInstructionPtr tuple) override {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_47(mht_47_v, 503, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleTuple");

    return DefaultAction(tuple);
  }
  Status HandleMap(HloInstructionPtr map) override {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_48(mht_48_v, 509, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleMap");

    return DefaultAction(map);
  }
  Status HandleReduce(HloInstructionPtr reduce) override {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_49(mht_49_v, 515, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleReduce");

    return DefaultAction(reduce);
  }
  Status HandleReduceWindow(HloInstructionPtr reduce_window) override {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_50(mht_50_v, 521, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleReduceWindow");

    return DefaultAction(reduce_window);
  }
  Status HandleSelectAndScatter(HloInstructionPtr select_and_scatter) override {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_51(mht_51_v, 527, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleSelectAndScatter");

    return DefaultAction(select_and_scatter);
  }
  Status HandleBitcast(HloInstructionPtr bitcast) override {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_52(mht_52_v, 533, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleBitcast");

    return DefaultAction(bitcast);
  }
  Status HandleBroadcast(HloInstructionPtr broadcast) override {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_53(mht_53_v, 539, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleBroadcast");

    return DefaultAction(broadcast);
  }
  Status HandlePad(HloInstructionPtr pad) override {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_54(mht_54_v, 545, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandlePad");

    return DefaultAction(pad);
  }
  Status HandleDynamicReshape(HloInstructionPtr dynamic_reshape) override {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_55(mht_55_v, 551, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleDynamicReshape");

    return DefaultAction(dynamic_reshape);
  }
  Status HandleReshape(HloInstructionPtr reshape) override {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_56(mht_56_v, 557, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleReshape");

    return DefaultAction(reshape);
  }
  Status HandleTranspose(HloInstructionPtr transpose) override {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_57(mht_57_v, 563, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleTranspose");

    return DefaultAction(transpose);
  }
  Status HandleWhile(HloInstructionPtr xla_while) override {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_58(mht_58_v, 569, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleWhile");

    return DefaultAction(xla_while);
  }
  Status HandleConditional(HloInstructionPtr conditional) override {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_59(mht_59_v, 575, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleConditional");

    return DefaultAction(conditional);
  }
  Status HandleAsyncStart(HloInstructionPtr async_start) override {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_60(mht_60_v, 581, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleAsyncStart");

    return DefaultAction(async_start);
  }
  Status HandleAsyncUpdate(HloInstructionPtr async_update) override {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_61(mht_61_v, 587, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleAsyncUpdate");

    return DefaultAction(async_update);
  }
  Status HandleAsyncDone(HloInstructionPtr async_done) override {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_62(mht_62_v, 593, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleAsyncDone");

    return DefaultAction(async_done);
  }
  Status HandleCopyStart(HloInstructionPtr copy_start) override {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_63(mht_63_v, 599, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleCopyStart");

    return DefaultAction(copy_start);
  }
  Status HandleCopyDone(HloInstructionPtr copy_done) override {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_64(mht_64_v, 605, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleCopyDone");

    return DefaultAction(copy_done);
  }
  Status HandleRecv(HloInstructionPtr recv) override {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_65(mht_65_v, 611, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleRecv");

    return DefaultAction(recv);
  }
  Status HandleRecvDone(HloInstructionPtr recv_done) override {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_66(mht_66_v, 617, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleRecvDone");

    return DefaultAction(recv_done);
  }
  Status HandleSend(HloInstructionPtr send) override {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_67(mht_67_v, 623, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleSend");

    return DefaultAction(send);
  }
  Status HandleSendDone(HloInstructionPtr send_done) override {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_68(mht_68_v, 629, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleSendDone");

    return DefaultAction(send_done);
  }
  Status HandleGather(HloInstructionPtr gather) override {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_69(mht_69_v, 635, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleGather");

    return DefaultAction(gather);
  }
  Status HandleScatter(HloInstructionPtr scatter) override {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_70(mht_70_v, 641, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleScatter");

    return DefaultAction(scatter);
  }
  Status HandleAfterAll(HloInstructionPtr token) override {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_71(mht_71_v, 647, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleAfterAll");

    return DefaultAction(token);
  }
  Status HandleGetDimensionSize(HloInstructionPtr get_size) override {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_72(mht_72_v, 653, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleGetDimensionSize");

    return DefaultAction(get_size);
  }
  Status HandleSetDimensionSize(HloInstructionPtr get_size) override {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_73(mht_73_v, 659, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleSetDimensionSize");

    return DefaultAction(get_size);
  }
  Status HandleAddDependency(HloInstructionPtr add_dependency) override {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_74(mht_74_v, 665, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "HandleAddDependency");

    return DefaultAction(add_dependency);
  }

  // Invoked to inform the visitor that the traversal has completed, and that
  // the root was "root".
  Status FinishVisit(HloInstructionPtr /*root*/) override {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_75(mht_75_v, 674, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "FinishVisit");

    return Status::OK();
  }

 private:
  DfsHloVisitorWithDefaultBase(const DfsHloVisitorWithDefaultBase&) = delete;
  DfsHloVisitorWithDefaultBase& operator=(const DfsHloVisitorWithDefaultBase&) =
      delete;
};

// Users should use these type aliases which are only two valid instantiations.
using DfsHloVisitorWithDefault = DfsHloVisitorWithDefaultBase<HloInstruction*>;
using ConstDfsHloVisitorWithDefault =
    DfsHloVisitorWithDefaultBase<const HloInstruction*>;

// A common base class for visitors performing rewriting operation.
//
// Subclasses call ReplaceWithNewInstruction and ReplaceInstruction while
// visiting.
class DfsHloRewriteVisitor : public DfsHloVisitorWithDefault {
 public:
  // Runs a visitor on the module and returns whether the module has changed.
  StatusOr<bool> RunOnModule(HloModule* module) {
    bool is_changed = false;
    for (const auto& computation : module->computations()) {
      TF_RETURN_IF_ERROR(computation->Accept(this));
      is_changed |= changed();
    }
    return is_changed;
  }

  // Default visitor action is to do nothing and return OK.
  Status DefaultAction(HloInstruction* /*hlo_instruction*/) override {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_76(mht_76_v, 709, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "DefaultAction");

    return Status::OK();
  }

  bool changed() const {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_77(mht_77_v, 716, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "changed");
 return changed_; }

 protected:
  // Replaces the existing HLO instruction old_instruction, with
  // new_instruction, and marks the optimizer status as changed.
  // Returns the Status representing the result of the replace operation.
  Status ReplaceWithNewInstruction(
      HloInstruction* old_instruction,
      std::unique_ptr<HloInstruction> new_instruction) {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_78(mht_78_v, 727, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "ReplaceWithNewInstruction");

    VLOG(3) << "Replacing instruction:";
    VLOG(3) << "  old: " << old_instruction->ToString();
    VLOG(3) << "  new: " << new_instruction->ToString();
    TF_RETURN_IF_ERROR(old_instruction->parent()->ReplaceWithNewInstruction(
        old_instruction, std::move(new_instruction)));
    changed_ = true;
    return Status::OK();
  }

  // Replaces the existing HLO instruction old_instruction, with
  // new_instruction, and marks the optimizer status as changed.
  // Returns the Status representing the result of the replace operation.
  StatusOr<bool> ReplaceInstruction(HloInstruction* old_instruction,
                                    HloInstruction* new_instruction,
                                    bool preserve_sharding) {
    VLOG(3) << "Replacing instruction:";
    VLOG(3) << "  old: " << old_instruction->ToString();
    VLOG(3) << "  new: " << new_instruction->ToString();
    TF_ASSIGN_OR_RETURN(
        bool changed, old_instruction->parent()->ReplaceInstruction(
                          old_instruction, new_instruction, preserve_sharding));
    changed_ |= changed;
    return changed;
  }

  Status ReplaceInstruction(HloInstruction* old_instruction,
                            HloInstruction* new_instruction) {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_79(mht_79_v, 757, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "ReplaceInstruction");

    TF_ASSIGN_OR_RETURN(bool changed,
                        ReplaceInstruction(old_instruction, new_instruction,
                                           /*preserve_sharding=*/false));
    DCHECK(changed);
    return Status::OK();
  }

  bool changed_ = false;
};

// (Const)FunctionVisitor lets you transform an
// std::function<Status((const) HloInstruction*)> into a (Const)DfsHloVisitor.
//
// This is useful if you have code that needs to handle visitors in the form of
// both std::function and DfsHloVisitor.  You can wrap the function in a
// FunctionVisitor and then treat it like any other DfsHloVisitor.
template <typename HloInstructionPtr>
class FunctionVisitorBase
    : public DfsHloVisitorWithDefaultBase<HloInstructionPtr> {
 public:
  explicit FunctionVisitorBase(
      std::function<Status(HloInstructionPtr)> visitor_func)
      : visitor_func_(std::move(visitor_func)) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_80(mht_80_v, 783, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "FunctionVisitorBase");
}

  Status DefaultAction(HloInstructionPtr hlo_instruction) override {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitor_with_defaultDTh mht_81(mht_81_v, 788, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h", "DefaultAction");

    return visitor_func_(hlo_instruction);
  }

 private:
  FunctionVisitorBase(const FunctionVisitorBase&) = delete;
  FunctionVisitorBase& operator=(const FunctionVisitorBase&) = delete;

  std::function<Status(HloInstructionPtr)> visitor_func_;
};

using FunctionVisitor = FunctionVisitorBase<HloInstruction*>;
using ConstFunctionVisitor = FunctionVisitorBase<const HloInstruction*>;

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DFS_HLO_VISITOR_WITH_DEFAULT_H_
