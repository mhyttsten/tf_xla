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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_DFS_HLO_VISITOR_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_DFS_HLO_VISITOR_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh() {
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


#include <type_traits>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

class HloComputation;
class HloInstruction;

// A postorder depth-first HloInstruction visitor. When Handle* is called on an
// instruction, all its operands were already visited. User code can subclass
// this to iterate over an HloInstruction DAG. The Handle* routines have
// operands / data unpacked for ease of use in the visitor subclass.
//
// No instruction will ever be visited twice; however, the root instruction will
// be reported again when the traversal is done via a call to FinishVisit.
//
// A subclass must override at least
// (either HandleElementwiseUnary or all the Handle methods for unary ops) and
// (either HandleElementwiseBinary or all the Handle methods for binary ops)).
// The default Handle methods for (unary, binary) ops call
// (HandleElementwiseUnary, HandleElementwiseBinary).
// The default (HandleElementwiseUnary, HandleElementwiseBinary) return an
// "unimplemented" error status.
//
// Note: this may change to an iterator in the future for flexibility purposes.
//
// Users should not use this class directly, but use the type-aliases
// DfsHloVisitor/ConstDfsHloVisitor instead.
template <typename HloInstructionPtr>
class DfsHloVisitorBase {
  static_assert(
      std::is_same<HloInstruction*, HloInstructionPtr>::value ||
          std::is_same<const HloInstruction*, HloInstructionPtr>::value,
      "Template argument expected to be HloInstruction* or const "
      "HloInstruction*");

 public:
  DfsHloVisitorBase() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_0(mht_0_v, 235, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "DfsHloVisitorBase");
}
  virtual ~DfsHloVisitorBase() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_1(mht_1_v, 239, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "~DfsHloVisitorBase");
}

  // These routines are self-descriptive, see class comment for usage
  // information.

  virtual Status HandleElementwiseUnary(HloInstructionPtr hlo);
  virtual Status HandleElementwiseBinary(HloInstructionPtr hlo);

  virtual Status HandleClamp(HloInstructionPtr hlo) = 0;
  virtual Status HandleSelect(HloInstructionPtr hlo) = 0;
  virtual Status HandleTupleSelect(HloInstructionPtr hlo) = 0;
  virtual Status HandleMaximum(HloInstructionPtr hlo) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_2(mht_2_v, 253, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleMaximum");

    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleMinimum(HloInstructionPtr hlo) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_3(mht_3_v, 259, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleMinimum");

    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleConcatenate(HloInstructionPtr hlo) = 0;
  virtual Status HandleConvert(HloInstructionPtr hlo) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_4(mht_4_v, 266, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleConvert");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleBitcastConvert(HloInstructionPtr hlo) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_5(mht_5_v, 272, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleBitcastConvert");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleCopy(HloInstructionPtr hlo) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_6(mht_6_v, 278, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleCopy");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleComplex(HloInstructionPtr hlo) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_7(mht_7_v, 284, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleComplex");

    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleMultiply(HloInstructionPtr hlo) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_8(mht_8_v, 290, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleMultiply");

    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleDot(HloInstructionPtr hlo) = 0;
  virtual Status HandlePower(HloInstructionPtr hlo) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_9(mht_9_v, 297, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandlePower");

    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleSqrt(HloInstructionPtr hlo) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_10(mht_10_v, 303, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleSqrt");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleRsqrt(HloInstructionPtr hlo) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_11(mht_11_v, 309, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleRsqrt");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleCbrt(HloInstructionPtr hlo) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_12(mht_12_v, 315, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleCbrt");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleConvolution(HloInstructionPtr hlo) = 0;
  virtual Status HandleFft(HloInstructionPtr fft) = 0;
  virtual Status HandleTriangularSolve(HloInstructionPtr hlo) = 0;
  virtual Status HandleCholesky(HloInstructionPtr hlo) = 0;
  virtual Status HandleOptimizationBarrier(HloInstructionPtr hlo) = 0;
  virtual Status HandleAllGather(HloInstructionPtr hlo) = 0;
  virtual Status HandleAllGatherStart(HloInstructionPtr hlo) = 0;
  virtual Status HandleAllGatherDone(HloInstructionPtr hlo) = 0;
  virtual Status HandleAllReduce(HloInstructionPtr hlo) = 0;
  virtual Status HandleReduceScatter(HloInstructionPtr hlo) = 0;
  virtual Status HandleAllReduceStart(HloInstructionPtr hlo) = 0;
  virtual Status HandleAllReduceDone(HloInstructionPtr hlo) = 0;
  virtual Status HandleAllToAll(HloInstructionPtr hlo) = 0;
  virtual Status HandleCollectivePermute(HloInstructionPtr hlo) = 0;
  virtual Status HandleCollectivePermuteStart(HloInstructionPtr hlo) = 0;
  virtual Status HandleCollectivePermuteDone(HloInstructionPtr hlo) = 0;
  virtual Status HandleReplicaId(HloInstructionPtr hlo) = 0;
  virtual Status HandlePartitionId(HloInstructionPtr hlo) = 0;
  virtual Status HandleGetDimensionSize(HloInstructionPtr hlo) = 0;
  virtual Status HandleSetDimensionSize(HloInstructionPtr hlo) = 0;
  virtual Status HandleCompare(HloInstructionPtr hlo) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_13(mht_13_v, 341, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleCompare");

    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleAdd(HloInstructionPtr hlo) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_14(mht_14_v, 347, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleAdd");

    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleDivide(HloInstructionPtr hlo) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_15(mht_15_v, 353, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleDivide");

    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleRemainder(HloInstructionPtr hlo) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_16(mht_16_v, 359, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleRemainder");

    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleSubtract(HloInstructionPtr hlo) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_17(mht_17_v, 365, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleSubtract");

    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleAbs(HloInstructionPtr hlo) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_18(mht_18_v, 371, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleAbs");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleAtan2(HloInstructionPtr hlo) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_19(mht_19_v, 377, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleAtan2");

    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleRound(HloInstructionPtr hlo) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_20(mht_20_v, 383, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleRound");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleLogistic(HloInstructionPtr hlo) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_21(mht_21_v, 389, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleLogistic");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleSign(HloInstructionPtr hlo) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_22(mht_22_v, 395, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleSign");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleNegate(HloInstructionPtr hlo) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_23(mht_23_v, 401, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleNegate");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleExp(HloInstructionPtr hlo) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_24(mht_24_v, 407, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleExp");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleExpm1(HloInstructionPtr hlo) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_25(mht_25_v, 413, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleExpm1");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleFloor(HloInstructionPtr hlo) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_26(mht_26_v, 419, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleFloor");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleCeil(HloInstructionPtr hlo) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_27(mht_27_v, 425, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleCeil");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleLog(HloInstructionPtr hlo) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_28(mht_28_v, 431, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleLog");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleClz(HloInstructionPtr hlo) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_29(mht_29_v, 437, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleClz");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleLog1p(HloInstructionPtr hlo) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_30(mht_30_v, 443, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleLog1p");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleCos(HloInstructionPtr hlo) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_31(mht_31_v, 449, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleCos");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleSin(HloInstructionPtr hlo) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_32(mht_32_v, 455, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleSin");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleTanh(HloInstructionPtr hlo) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_33(mht_33_v, 461, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleTanh");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleReal(HloInstructionPtr hlo) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_34(mht_34_v, 467, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleReal");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleImag(HloInstructionPtr hlo) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_35(mht_35_v, 473, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleImag");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleIsFinite(HloInstructionPtr hlo) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_36(mht_36_v, 479, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleIsFinite");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleAnd(HloInstructionPtr hlo) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_37(mht_37_v, 485, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleAnd");

    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleNot(HloInstructionPtr hlo) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_38(mht_38_v, 491, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleNot");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleOr(HloInstructionPtr hlo) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_39(mht_39_v, 497, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleOr");

    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleXor(HloInstructionPtr hlo) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_40(mht_40_v, 503, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleXor");

    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandlePopulationCount(HloInstructionPtr hlo) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_41(mht_41_v, 509, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandlePopulationCount");

    return HandleElementwiseUnary(hlo);
  }
  virtual Status HandleShiftLeft(HloInstructionPtr hlo) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_42(mht_42_v, 515, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleShiftLeft");

    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleShiftRightArithmetic(HloInstructionPtr hlo) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_43(mht_43_v, 521, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleShiftRightArithmetic");

    return HandleElementwiseBinary(hlo);
  }
  virtual Status HandleShiftRightLogical(HloInstructionPtr hlo) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_44(mht_44_v, 527, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleShiftRightLogical");

    return HandleElementwiseBinary(hlo);
  }

  virtual Status HandleReducePrecision(HloInstructionPtr hlo) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_45(mht_45_v, 534, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleReducePrecision");

    return HandleElementwiseUnary(hlo);
  }

  virtual Status HandleDomain(HloInstructionPtr hlo) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_46(mht_46_v, 541, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "HandleDomain");

    return HandleElementwiseUnary(hlo);
  }

  virtual Status HandleInfeed(HloInstructionPtr hlo) = 0;
  virtual Status HandleOutfeed(HloInstructionPtr hlo) = 0;
  virtual Status HandleRng(HloInstructionPtr hlo) = 0;
  virtual Status HandleRngBitGenerator(HloInstructionPtr hlo) = 0;
  virtual Status HandleRngGetAndUpdateState(HloInstructionPtr hlo) = 0;
  virtual Status HandleReverse(HloInstructionPtr hlo) = 0;
  virtual Status HandleSort(HloInstructionPtr hlo) = 0;
  virtual Status HandleConstant(HloInstructionPtr hlo) = 0;
  virtual Status HandleIota(HloInstructionPtr hlo) = 0;
  virtual Status HandleGetTupleElement(HloInstructionPtr hlo) = 0;
  virtual Status HandleReduce(HloInstructionPtr hlo) = 0;
  virtual Status HandleBitcast(HloInstructionPtr hlo) = 0;
  virtual Status HandleBroadcast(HloInstructionPtr hlo) = 0;
  virtual Status HandleReshape(HloInstructionPtr hlo) = 0;
  virtual Status HandleDynamicReshape(HloInstructionPtr hlo) = 0;
  virtual Status HandleTranspose(HloInstructionPtr hlo) = 0;
  virtual Status HandleParameter(HloInstructionPtr hlo) = 0;
  virtual Status HandleFusion(HloInstructionPtr hlo) = 0;
  virtual Status HandleCall(HloInstructionPtr hlo) = 0;
  virtual Status HandleCustomCall(HloInstructionPtr hlo) = 0;
  virtual Status HandleSlice(HloInstructionPtr hlo) = 0;
  virtual Status HandleDynamicSlice(HloInstructionPtr hlo) = 0;
  virtual Status HandleDynamicUpdateSlice(HloInstructionPtr hlo) = 0;
  virtual Status HandleTuple(HloInstructionPtr hlo) = 0;
  virtual Status HandleMap(HloInstructionPtr hlo) = 0;
  virtual Status HandleReduceWindow(HloInstructionPtr hlo) = 0;
  virtual Status HandleSelectAndScatter(HloInstructionPtr hlo) = 0;
  virtual Status HandleWhile(HloInstructionPtr hlo) = 0;
  virtual Status HandleConditional(HloInstructionPtr hlo) = 0;
  virtual Status HandleGather(HloInstructionPtr hlo) = 0;
  virtual Status HandleScatter(HloInstructionPtr hlo) = 0;

  virtual Status HandlePad(HloInstructionPtr hlo) = 0;

  virtual Status HandleAsyncStart(HloInstructionPtr hlo) = 0;
  virtual Status HandleAsyncUpdate(HloInstructionPtr hlo) = 0;
  virtual Status HandleAsyncDone(HloInstructionPtr hlo) = 0;

  virtual Status HandleCopyStart(HloInstructionPtr copy_start) = 0;
  virtual Status HandleCopyDone(HloInstructionPtr copy_done) = 0;

  virtual Status HandleSend(HloInstructionPtr send) = 0;
  virtual Status HandleSendDone(HloInstructionPtr send_done) = 0;

  virtual Status HandleRecv(HloInstructionPtr recv) = 0;
  virtual Status HandleRecvDone(HloInstructionPtr recv_done) = 0;

  virtual Status HandleBatchNormTraining(HloInstructionPtr hlo) = 0;

  virtual Status HandleBatchNormInference(HloInstructionPtr hlo) = 0;

  virtual Status HandleBatchNormGrad(HloInstructionPtr hlo) = 0;

  virtual Status HandleAddDependency(HloInstructionPtr add_dependency) = 0;
  virtual Status HandleAfterAll(HloInstructionPtr token) = 0;

  // Invoked to inform the visitor that the traversal has completed, and that
  // the root was "root".
  virtual Status FinishVisit(HloInstructionPtr root) = 0;

  // 3 possible visitation states of HLO instructions. Each instruction's
  // state only flows one way: kNotVisited -> kVisiting -> kVisited.
  enum VisitState {
    kNotVisited = 0,
    kVisiting = 1,
    kVisited = 2,
  };

  VisitState GetVisitState(int id) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_47(mht_47_v, 616, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "GetVisitState");

    auto iter = visit_state_.find(id);
    if (iter == visit_state_.end()) {
      return VisitState::kNotVisited;
    }
    return iter->second;
  }
  VisitState GetVisitState(const HloInstruction& instruction);

  // Resize internal state if necessary to hold state for ids <= num.
  // This call is purely a performance hint and can be omitted without
  // affecting correctness.
  void ReserveVisitStates(int num) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_48(mht_48_v, 631, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "ReserveVisitStates");
 visit_state_.reserve(num); }
  size_t VisitStateCapacity() const {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_49(mht_49_v, 635, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "VisitStateCapacity");
 return visit_state_.capacity(); }

  // Useful when we want to visit the same computation more than once with the
  // same visitor.
  void ResetVisitStates() {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_50(mht_50_v, 642, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "ResetVisitStates");

    // Clear the map, but don't resize the capacity across uses -- Calculating
    // and reserving space could be expensive, and we always use the same
    // module->instruction_count() as the capacity.
    visit_state_.erase(visit_state_.begin(), visit_state_.end());
  }

  // Useful when we want to free up the memory used by the visit state without
  // destroying the actual visitor subclass.
  void DestroyVisitState() {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_51(mht_51_v, 654, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "DestroyVisitState");

    visit_state_ = absl::flat_hash_map<int, VisitState>{};
  }

  void SetVisitState(int id, VisitState state) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_52(mht_52_v, 661, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "SetVisitState");
 visit_state_[id] = state; }

  // Sets the visitation state of the given instruction as kVisiting.
  //
  // Precondition: current state must be kNotVisited.
  void SetVisiting(const HloInstruction& instruction);

  // Sets the visitation state of the given instruction as kVisited.
  //
  // Precondition: current state must be either kNotVisited or kVisiting.
  void SetVisited(const HloInstruction& instruction);

  // Returns whether the state of the given instruction is kVisiting.
  bool IsVisiting(const HloInstruction& instruction) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_53(mht_53_v, 677, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "IsVisiting");

    return GetVisitState(instruction) == kVisiting;
  }

  // Returns whether the state of the given instruction is kVisited.
  bool DidVisit(const HloInstruction& instruction) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_54(mht_54_v, 685, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "DidVisit");

    return GetVisitState(instruction) == kVisited;
  }

  // Returns whether the state of the given instruction is kNotVisited.
  bool NotVisited(const HloInstruction& instruction) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdfs_hlo_visitorDTh mht_55(mht_55_v, 693, "", "./tensorflow/compiler/xla/service/dfs_hlo_visitor.h", "NotVisited");

    return GetVisitState(instruction) == kNotVisited;
  }

  // This method should be overridden by subclasses that wish to run some
  // operation on an op before its Handle* visitor method is called.
  //
  // For any HLO op, the order of calls is:
  //
  //   Preprocess(op);
  //   Handle/OpType/(op);
  //   Postprocess(op);
  //
  // Overriding methods should call DfsHloVisitor::Preprocess before doing their
  // own preprocessing.
  virtual Status Preprocess(HloInstructionPtr hlo);

  // This method should be overridden by subclasses that wish to run some
  // operation on an op after its Handle* visitor method is called. See
  // Preprocess for more details.
  //
  // Overriding methods should call DfsHloVisitor::Postprocess after doing their
  // own postprocessing.
  virtual Status Postprocess(HloInstructionPtr hlo);

 private:
  absl::flat_hash_map<int, VisitState> visit_state_;

  DfsHloVisitorBase(const DfsHloVisitorBase&) = delete;
  DfsHloVisitorBase& operator=(const DfsHloVisitorBase&) = delete;
};

// Explicit instantiations in dfs_hlo_visitor.cc.
extern template class DfsHloVisitorBase<HloInstruction*>;
extern template class DfsHloVisitorBase<const HloInstruction*>;

// Users should use one of these two type aliases, which are the only two valid
// instantiations of DfsHloVisitorBase.
using DfsHloVisitor = DfsHloVisitorBase<HloInstruction*>;
using ConstDfsHloVisitor = DfsHloVisitorBase<const HloInstruction*>;

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DFS_HLO_VISITOR_H_
