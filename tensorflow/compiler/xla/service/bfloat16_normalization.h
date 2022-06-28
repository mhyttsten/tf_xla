/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_BFLOAT16_NORMALIZATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_BFLOAT16_NORMALIZATION_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_normalizationDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_normalizationDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_normalizationDTh() {
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


#include "tensorflow/compiler/xla/service/bfloat16_support.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// A pass which adds F32 <-> BF16 conversions for HLO instructions that do not
// support BF16 input/output or mixed precision, according to the passed-in
// backend-specific BF16 support rules.
class BFloat16Normalization : public HloModulePass {
 public:
  explicit BFloat16Normalization(const BFloat16Support* bfloat16_support)
      : bfloat16_support_(bfloat16_support) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_normalizationDTh mht_0(mht_0_v, 200, "", "./tensorflow/compiler/xla/service/bfloat16_normalization.h", "BFloat16Normalization");
}

  ~BFloat16Normalization() override = default;
  absl::string_view name() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_normalizationDTh mht_1(mht_1_v, 206, "", "./tensorflow/compiler/xla/service/bfloat16_normalization.h", "name");
 return "bf16-normalization"; }

  // Run BF16 normalization on the given computation. Returns whether the
  // computation was changed.
  StatusOr<bool> Run(HloModule* module) override;

 private:
  const BFloat16Support* bfloat16_support_;
};

// A pass that unconditionally removes the mixed F32/BF16 uses in HLO
// instructions (excluding convert) by adding F32 <-> BF16 conversions. Unlike
// BFloat16Normalization, this pass does not use a backend-specific
// BFloat16Support, and does not change HLOs that have BF16 data if they do not
// use mixed precision; it removes mixed precision even if the backend supports
// it. This pass is used to make the HLO module valid for other HLO passes which
// do not support mixed precision.
class BFloat16MixedPrecisionRemoval : public HloModulePass {
 public:
  BFloat16MixedPrecisionRemoval() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_normalizationDTh mht_2(mht_2_v, 228, "", "./tensorflow/compiler/xla/service/bfloat16_normalization.h", "BFloat16MixedPrecisionRemoval");
}

  ~BFloat16MixedPrecisionRemoval() override = default;

  absl::string_view name() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_normalizationDTh mht_3(mht_3_v, 235, "", "./tensorflow/compiler/xla/service/bfloat16_normalization.h", "name");

    return "bf16-mixed-precision-removal";
  }

  // Run mixed precision removal on the given computation. Returns whether the
  // computation was changed.
  StatusOr<bool> Run(HloModule* module) override {
    BFloat16Normalization normalization(&no_mixed_precision_support_);
    return normalization.Run(module);
  }

 private:
  class BFloat16SupportForMixedPrecisionRemoval : public BFloat16Support {
   public:
    BFloat16SupportForMixedPrecisionRemoval() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_normalizationDTh mht_4(mht_4_v, 252, "", "./tensorflow/compiler/xla/service/bfloat16_normalization.h", "BFloat16SupportForMixedPrecisionRemoval");
}

    ~BFloat16SupportForMixedPrecisionRemoval() override = default;

    bool SupportsBF16Operand(const HloInstruction& hlo,
                             int64_t operand_index) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_normalizationDTh mht_5(mht_5_v, 260, "", "./tensorflow/compiler/xla/service/bfloat16_normalization.h", "SupportsBF16Operand");

      return true;
    }

    bool SupportsBF16Output(const HloInstruction& hlo) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_normalizationDTh mht_6(mht_6_v, 267, "", "./tensorflow/compiler/xla/service/bfloat16_normalization.h", "SupportsBF16Output");

      return true;
    }

    bool SupportsMixedPrecisions(const HloInstruction& hlo) const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbfloat16_normalizationDTh mht_7(mht_7_v, 274, "", "./tensorflow/compiler/xla/service/bfloat16_normalization.h", "SupportsMixedPrecisions");

      return false;
    }
  } no_mixed_precision_support_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_BFLOAT16_NORMALIZATION_H_
