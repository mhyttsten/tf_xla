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

// Class to create computations for testing rematerialization methods.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_REMATERIALIZATION_TEST_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_REMATERIALIZATION_TEST_UTILS_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_rematerialization_test_utilsDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_rematerialization_test_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_rematerialization_test_utilsDTh() {
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


#include <memory>
#include <string>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_rematerialization.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {

class RematerializationTestBase : public HloTestBase {
 protected:
  // Creates and returns a computation which can benefit from
  // rematerialization. The computation looks like:
  //
  //   F32[1] %param = {...}
  //   F32[] %reshape = reshape(F32[], param)
  //   F32[1024] %bcast = broadcast(%param)
  //   F32[1024] %negate = negate(%bcast)
  //   F32[2048] %concat_1 = concat({%negate, %negate})
  //   F32[1] %slice_1 = slice(%concat_1, {0:1})
  //   F32[1025] %concat_2 = concat({%bcast, %slice_1})
  //   F32[1] %slice_2 = slice(%concat_2, {0:1});
  //
  // The instruction %bcast can be rematerialized before its use at %concat_2
  // to reduce peak memory usage. This avoids %bcast and %concat_1 being
  // simultaneously live. Peak memory use is about 16KB before rematerialization
  // (during execution of %concat_1) and about 12KB after rematerializing %bcast
  // for its use in %concat_2.
  std::unique_ptr<HloComputation> MakeRematerializableComputation(
      const std::string& suffix = "") {
    auto builder = HloComputation::Builder(TestName() + suffix);
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, vec1_shape_, "param"));
    auto reshape = builder.AddInstruction(
        HloInstruction::CreateReshape(scalar_shape_, param));
    auto bcast = builder.AddInstruction(
        HloInstruction::CreateBroadcast(vec1024_shape_, reshape, {}));
    auto negate = builder.AddInstruction(
        HloInstruction::CreateUnary(vec1024_shape_, HloOpcode::kNegate, bcast));
    auto concat_1 = builder.AddInstruction(HloInstruction::CreateConcatenate(
        ShapeUtil::MakeShape(xla::F32, {2048}), {negate, negate},
        /*dimension=*/0));
    auto slice_1 = builder.AddInstruction(HloInstruction::CreateSlice(
        vec1_shape_, concat_1, /*start_indices=*/{0},
        /*limit_indices=*/{1},
        /*strides=*/{1}));
    auto concat_2 = builder.AddInstruction(HloInstruction::CreateConcatenate(
        ShapeUtil::MakeShape(xla::F32, {1025}), {bcast, slice_1},
        /*dimension=*/0));
    // Add a final slice to make the parameter shape match the output shape
    // which is necessary to use this computation in a while.
    builder.AddInstruction(HloInstruction::CreateSlice(vec1_shape_, concat_2,
                                                       /*start_indices=*/{0},
                                                       /*limit_indices=*/{1},
                                                       /*strides=*/{1}));
    return builder.Build();
  }

  // Creates and returns a computation which includes a while and can benefit
  // from rematerialization. The computation looks like:
  //
  //   F32[] %param = {...}
  //   F32[1024] %bcast = broadcast(%param)
  //   F32[1] %slice_1 = slice(%bcast, {0:1})
  //   F32[1] %while = while(%slice_1, while_body, while_cond)
  //   F32[1025] %concat = concat({%bcast, %while})
  //   F32[1] %slice_2 = slice(%concat, {0:1});
  //
  // The instruction %bcast can be rematerialized before its use at %concat to
  // reduce peak memory usage. This avoids %bcast being live during execution of
  // the while. Peak memory use is maximum of 8K and 4K plus the memory use of
  // the while subcomputations.
  std::unique_ptr<HloComputation> MakeRematerializableWhileComputation(
      HloComputation* while_cond, HloComputation* while_body,
      const std::string& suffix = "") {
    auto builder = HloComputation::Builder(TestName() + suffix);
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, vec1_shape_, "param"));
    auto reshape = builder.AddInstruction(
        HloInstruction::CreateReshape(scalar_shape_, param));
    auto bcast = builder.AddInstruction(
        HloInstruction::CreateBroadcast(vec1024_shape_, reshape, {}));
    auto slice_1 = builder.AddInstruction(
        HloInstruction::CreateSlice(vec1_shape_, bcast, /*start_indices=*/{0},
                                    /*limit_indices=*/{1},
                                    /*strides=*/{1}));
    auto while_inst = builder.AddInstruction(HloInstruction::CreateWhile(
        vec1_shape_, while_cond, while_body, slice_1));
    auto concat = builder.AddInstruction(HloInstruction::CreateConcatenate(
        ShapeUtil::MakeShape(xla::F32, {1025}), {bcast, while_inst},
        /*dimension=*/0));
    builder.AddInstruction(HloInstruction::CreateSlice(vec1_shape_, concat,
                                                       /*start_indices=*/{0},
                                                       /*limit_indices=*/{1},
                                                       /*strides=*/{1}));
    return builder.Build();
  }

  // Create and return a trivial computation appropriate for use as a while
  // condition.
  std::unique_ptr<HloComputation> MakeConditionComputation() {
    auto builder = HloComputation::Builder(TestName() + ".cond");
    builder.AddInstruction(
        HloInstruction::CreateParameter(0, vec1_shape_, "param"));
    builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
    return builder.Build();
  }

  // Return the byte size of the top-level buffer of the given shape.
  static int64_t ByteSizeOf(const Shape& shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_rematerialization_test_utilsDTh mht_0(mht_0_v, 308, "", "./tensorflow/compiler/xla/service/hlo_rematerialization_test_utils.h", "ByteSizeOf");

    return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
  }

 protected:
  // Various shapes used in the canned computations.
  const Shape scalar_shape_ = ShapeUtil::MakeShape(xla::F32, {});
  const Shape vec1_shape_ = ShapeUtil::MakeShape(xla::F32, {1});
  const Shape vec1024_shape_ = ShapeUtil::MakeShape(xla::F32, {1024});
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_REMATERIALIZATION_TEST_UTILS_H_
