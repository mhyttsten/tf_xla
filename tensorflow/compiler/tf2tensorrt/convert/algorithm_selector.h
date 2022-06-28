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
#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_ALGORITHM_SELECTOR_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_ALGORITHM_SELECTOR_H_
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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSalgorithm_selectorDTh {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSalgorithm_selectorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSalgorithm_selectorDTh() {
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

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include <array>
#include <memory>
#include <set>

#include "absl/types/optional.h"
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

// Implements core algorithm selection logic in a testable manner. The policy
// implemented depends on the given TRT version. We have this class because TRT
// interfaces make it difficult to directly test an IAlgorithmSelector
// implementation.
class AlgorithmSelectorImpl {
 public:
  using TRTVersion = std::array<int, 4>;
  using ImplementationID = int64_t;
  using TacticID = int64_t;

  static constexpr TRTVersion CompileTimeTRTVersion() {
    return TRTVersion{NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH,
                      NV_TENSORRT_BUILD};
  }

  explicit AlgorithmSelectorImpl(
      const TRTVersion& version = CompileTimeTRTVersion())
      : version_(version) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSalgorithm_selectorDTh mht_0(mht_0_v, 215, "", "./tensorflow/compiler/tf2tensorrt/convert/algorithm_selector.h", "AlgorithmSelectorImpl");
}

  bool IsShuffleLayer(ImplementationID id) const;

  bool IsBannedTactic(TacticID id) const;

  // Returns true if the algorithm implementing the IShuffleLayer is acceptable.
  bool AllowShuffleAlgorithm(TacticID tactic, nvinfer1::DataType input_dtype,
                             nvinfer1::TensorFormat input_format) const;

  bool IsTrtVersionGE(const TRTVersion& version) const;

  // Returns true if we know at compile time that the the algorithm selector
  // should be required. This is a conservative estimate.
  bool IsAlgorithmSelectorRequired() const;

  static std::set<TacticID> GetBannedTRT72TuringTactics();

 private:
  TRTVersion version_;
};

// Impelements the TRT IAlgorithmSelector interface. The method
// "selectAlgorithms" selects allowable algorithms for each layer, and
// "reportAlgorithms" summarizes the algorithms selected by TensorRT.
class TftrtAlgorithmSelector : public nvinfer1::IAlgorithmSelector {
 private:
  using TacticID = AlgorithmSelectorImpl::TacticID;

  // An index we should choose for all algorithms. Used for debugging.
  absl::optional<int32_t> fixed_algorithm_idx_;

  AlgorithmSelectorImpl selector_;

 public:
  TftrtAlgorithmSelector();

  // If the environment variable TF_TRT_FIXED_ALGORITHM_ID is empty, this
  // function returns nullopt. Otherwise, it returns the specified number.
  static absl::optional<int64_t> GetFixedAlgorithmID();

  // Returns true if the algorithm associated with context is acceptable.
  bool AlgorithmPolicy(const nvinfer1::IAlgorithmContext& context,
                       const nvinfer1::IAlgorithm& alg) const;

  // This function fills the array "selection" with the indices of selected
  // algorithm candidates from "algoChoices", each of which is an implementation
  // for the kernel described by the given IAlgorithmContext. It should return a
  // number in [0, nbChoices] indicating the number of selected indices. If 0 is
  // returned, TensorRT will use its default selection mechanism.
  int32_t selectAlgorithms(const nvinfer1::IAlgorithmContext& algoContext,
                           const nvinfer1::IAlgorithm* const* algoChoices,
                           int32_t nbChoices,
                           int32_t* selection) noexcept override;

  // Called by TensorRT to report choices it made.
  void reportAlgorithms(const nvinfer1::IAlgorithmContext* const* algoContexts,
                        const nvinfer1::IAlgorithm* const* algoChoices,
                        int32_t nbAlgorithms) noexcept override;

  bool IsRequired() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSalgorithm_selectorDTh mht_1(mht_1_v, 278, "", "./tensorflow/compiler/tf2tensorrt/convert/algorithm_selector.h", "IsRequired");

    return selector_.IsAlgorithmSelectorRequired() ||
           fixed_algorithm_idx_ != absl::nullopt;
  }
};

// Returns an initialized AlgorithmSelector if an algorithm selector is required
// for the current TRT version. Otherwise, returns nullptr.
std::unique_ptr<TftrtAlgorithmSelector> MaybeCreateAlgorithmSelector();

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_ALGORITHM_SELECTOR_H_
