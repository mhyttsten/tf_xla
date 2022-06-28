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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_DYNAMIC_PADDER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_DYNAMIC_PADDER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_padderDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_padderDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_padderDTh() {
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


#include "tensorflow/compiler/xla/service/dynamic_dimension_inference.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// With bounded shapes, only part of the shape contains effective data and the
// rest contains padded data, whose value can be anything depending on the
// source of the data. When a bounded shape is directly consumed by an
// instruction that collapses dimensions (reduce for example), the padding data
// would affect result of the instruction.
//
// DynamicPadder uses DynamicDimensionInference to detect bounded shapes in a
// hlo module, it then inserts certain instructions to reset the padding into an
// identity value so that in doesn't affect the result of subsequent
// instruction. For example, it'd reset the padding to 0 before a bounded shape
// is consumed by a reduce-sum.
//
// Dynamic_padder removes dynamic shapes from the entry computation, and inserts
// custom calls (with dynamic shapes), which are lowered by specialized
// emitters: PadToStatic and SliceToDynamic.

// Each instruction can have one of the three modes in supporting dynamic
// lowering.
enum OpDynamismSupport {
  // There is no support for dynamic lowering -- dynamic padder will make sure
  // the input to that op has static bound by rewriting the op (e.g, extra space
  // in reduce_sum will be padded with 0).
  kNoSupport = 0,
  // The op can take either dynamic input or static input.
  kOptional,
  // The op only has a dynamic lowering, dynamic padder will make sure the input
  // to this op is in dynamic form.
  kRequired,
};

struct DynamicPadderOptions {
  // Returns true if given instruction supports native dynamic lowering. If
  // so, dynamic padder will not attempt to pad it.
  using OpSupportsDynamismHandler =
      std::function<OpDynamismSupport(HloInstruction*)>;

  OpSupportsDynamismHandler op_supports_dynamism_handler = nullptr;

  // Instruct how to inference output dynamic dimensions of custom calls.
  DynamicDimensionInference::CustomCallInferenceHandler custom_call_handler =
      nullptr;

  // If `slice_dynamic_output` is true, insert 'slice_to_dynamic' ops to all
  // outputs that are inferred to be dynamic.
  bool slice_dynamic_output = true;

  // If set to true, pessimisticly assumes runtime shape checks may fail and
  // returns a compile-time error.
  DynamicDimensionInference::ShapeCheckMode shape_check_mode =
      DynamicDimensionInference::ShapeCheckMode::kIgnore;
};

class DynamicPadder : public HloModulePass {
 public:
  explicit DynamicPadder(DynamicPadderOptions options = DynamicPadderOptions())
      : options_(options) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_padderDTh mht_0(mht_0_v, 248, "", "./tensorflow/compiler/xla/service/dynamic_padder.h", "DynamicPadder");
}

  absl::string_view name() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_padderDTh mht_1(mht_1_v, 253, "", "./tensorflow/compiler/xla/service/dynamic_padder.h", "name");
 return "dynamic_padder"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  DynamicPadderOptions options_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DYNAMIC_PADDER_H_
