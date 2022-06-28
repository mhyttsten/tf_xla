/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_VALUE_INFERENCE_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_VALUE_INFERENCE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSvalue_inferenceDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSvalue_inferenceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSvalue_inferenceDTh() {
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


#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
// OptionalLiteral is an augmented literal class which returns optional
// values for each index (the value can be either valid or invalid). The
// implementation keeps two literals, a value literal, holding both the valid
// and garabage value, and a masking literal representing if a value is valid or
// garbage.
class OptionalLiteral {
 public:
  explicit OptionalLiteral(Literal value, Literal mask)
      : value_(std::move(value)), mask_(std::move(mask)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSvalue_inferenceDTh mht_0(mht_0_v, 208, "", "./tensorflow/compiler/xla/client/value_inference.h", "OptionalLiteral");
}

  template <typename NativeT>
  absl::optional<NativeT> Get(absl::Span<const int64_t> element_index,
                              ShapeIndex shape_index = {}) const {
    if (mask_.Get<bool>(element_index, shape_index)) {
      return absl::nullopt;
    } else {
      return value_.Get<NativeT>(element_index, shape_index);
    }
  }

  // Returns true if all values in this literal slice are value.
  bool AllValid() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSvalue_inferenceDTh mht_1(mht_1_v, 224, "", "./tensorflow/compiler/xla/client/value_inference.h", "AllValid");
 return mask_.IsAll(0); }

  // Get value out of this slice if all values are valid. Otherwise returns
  // nullopt.
  absl::optional<LiteralSlice> GetValue() {
    if (!AllValid()) {
      return absl::nullopt;
    }
    return LiteralSlice(value_);
  }

 private:
  Literal value_;
  Literal mask_;
};

enum ValueInferenceMode {
  // Inference the constant value itself.
  kValue = 0,
  // Inference upper-bound and lower-bound of the value. Bounds are inclusive.
  kUpperBound,
  kLowerBound,
};

class ValueInference {
 public:
  // ValueInference analyzes values in XlaOp answers following questions:
  // - What's the upper-bound of each value in a tensor.
  // - What's the lower-bound of each value in a tensor.
  // - What's the constant value of each tensor.
  // - Whether or not each value in a tensor is dynamic.
  explicit ValueInference(XlaBuilder* builder) : builder_(builder) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSvalue_inferenceDTh mht_2(mht_2_v, 258, "", "./tensorflow/compiler/xla/client/value_inference.h", "ValueInference");

    CHECK(builder_);
  }
  StatusOr<Literal> AnalyzeIsDynamic(XlaOp op);
  // Returns an OptionalLiteral. Each individual value of the literal is
  // the concrete constant value if it can be inferred, otherwise a nullopt.
  StatusOr<OptionalLiteral> AnalyzeConstant(XlaOp op, ValueInferenceMode mode);

  // Returns underlying xla builder.
  XlaBuilder* builder() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSvalue_inferenceDTh mht_3(mht_3_v, 270, "", "./tensorflow/compiler/xla/client/value_inference.h", "builder");
 return builder_; }

 private:
  // Given an op handle, returns a simplified version of the handle inside a
  // int64_t Literal. If the a -1 value for the handle means invalid
  // simplification and the result shouldn't be used.
  StatusOr<Literal> SimplifyOp(int64_t handle);

  // Perform CSE on a given handle, and return an equivalent handle if seen
  // before. Otherwise, returns nullopt.
  StatusOr<absl::optional<int64_t>> CseOpHandle(int64_t handle);
  XlaBuilder* builder_;
  HloEvaluator evaluator_;
  // A map from instruction_hash to handle that helps perform CSE.
  absl::flat_hash_map<int64_t, int64_t> cse_map_;
};
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_VALUE_INFERENCE_H_
