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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSalgorithm_selectorDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSalgorithm_selectorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSalgorithm_selectorDTcc() {
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
#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "tensorflow/compiler/tf2tensorrt/convert/algorithm_selector.h"

#include <utility>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/core/util/env_var.h"
#include "third_party/tensorrt/NvInfer.h"

// getAlgorithmIOInfo is deprecated in TRT >= 8, replaced by
// getAlgorithmIOInfoByIndex.
#if IS_TRT_VERSION_GE(8, 0, 0, 0)
#define ALGORITHM_IO_INFO_BY_IDX(alg, idx) *(alg).getAlgorithmIOInfoByIndex(idx)
#else
#define ALGORITHM_IO_INFO_BY_IDX(alg, idx) (alg).getAlgorithmIOInfo(idx)
#endif

namespace nvinfer1 {

std::ostream& operator<<(std::ostream& os,
                         const nvinfer1::IAlgorithmContext& ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSalgorithm_selectorDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/tf2tensorrt/convert/algorithm_selector.cc", "operator<<");

  os << "AlgorithmContext(name=" << ctx.getName()
     << ",nbInputs=" << ctx.getNbInputs() << ",nbOutputs=" << ctx.getNbOutputs()
     << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const nvinfer1::IAlgorithm& alg) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSalgorithm_selectorDTcc mht_1(mht_1_v, 216, "", "./tensorflow/compiler/tf2tensorrt/convert/algorithm_selector.cc", "operator<<");

  const nvinfer1::IAlgorithmVariant& variant = alg.getAlgorithmVariant();
  os << "Algorithm("
     << "variant.implementation=" << variant.getImplementation()
     << ",variant.tactic=" << variant.getTactic()
     << ",timingMSec=" << alg.getTimingMSec()
     << ",workspaceSize=" << alg.getWorkspaceSize() << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const nvinfer1::IAlgorithmIOInfo& info) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSalgorithm_selectorDTcc mht_2(mht_2_v, 230, "", "./tensorflow/compiler/tf2tensorrt/convert/algorithm_selector.cc", "operator<<");

  os << "IOTensor(format=" << info.getTensorFormat()
     << ",dtype=" << info.getDataType() << ",strides=" << info.getStrides()
     << ")";
  return os;
}
}  // namespace nvinfer1

namespace tensorflow {
namespace tensorrt {
namespace convert {

bool operator>=(const AlgorithmSelectorImpl::TRTVersion& lhs,
                const AlgorithmSelectorImpl::TRTVersion& rhs) {
  if (lhs[0] > rhs[0]) return true;
  if (lhs[0] == rhs[0] && lhs[1] > rhs[1]) return true;
  if (lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] > rhs[2]) return true;
  if (lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2] &&
      lhs[3] >= rhs[3]) {
    return true;
  }
  return false;
}

bool AlgorithmSelectorImpl::IsTrtVersionGE(const TRTVersion& version) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSalgorithm_selectorDTcc mht_3(mht_3_v, 257, "", "./tensorflow/compiler/tf2tensorrt/convert/algorithm_selector.cc", "AlgorithmSelectorImpl::IsTrtVersionGE");

  return version_ >= version;
}

bool AlgorithmSelectorImpl::IsShuffleLayer(ImplementationID id) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSalgorithm_selectorDTcc mht_4(mht_4_v, 264, "", "./tensorflow/compiler/tf2tensorrt/convert/algorithm_selector.cc", "AlgorithmSelectorImpl::IsShuffleLayer");

  if (IsTrtVersionGE({8, 2, 0, 0})) {
    return id == 0x80000000 + 13;
  }
  if (IsTrtVersionGE({8, 0, 0, 0})) {
    return id == 0x80000000 + 14;
  }
  if (IsTrtVersionGE({7, 2, 0, 0})) {
    return id == 0x80000000 + 16;
  }
  return id == 18;
}

std::set<AlgorithmSelectorImpl::TacticID>
AlgorithmSelectorImpl::GetBannedTRT72TuringTactics() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSalgorithm_selectorDTcc mht_5(mht_5_v, 281, "", "./tensorflow/compiler/tf2tensorrt/convert/algorithm_selector.cc", "AlgorithmSelectorImpl::GetBannedTRT72TuringTactics");

  static const std::set<TacticID> banned_turing_72{
      // turing_fp16_s1688cudnn_fp16_128x128_ldg8_relu_f2f_exp_medium_nhwc_gelu_tn_v1
      -5927686925093575778,
      // turing_fp16_s1688cudnn_fp16_128x128_ldg8_relu_f2f_exp_interior_nhwc_gelu_tn_v1
      -3848538574386518527,
      // turing_fp16_s1688cudnn_fp16_128x128_ldg8_relu_f2f_exp_small_nhwc_gelu_tn_v1
      -959009792490796596};
  return banned_turing_72;
}

bool AlgorithmSelectorImpl::IsBannedTactic(TacticID id) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSalgorithm_selectorDTcc mht_6(mht_6_v, 295, "", "./tensorflow/compiler/tf2tensorrt/convert/algorithm_selector.cc", "AlgorithmSelectorImpl::IsBannedTactic");

  // Disable problematic FP16-Turing tactics in TensorRT 7.2.
  if (IsTrtVersionGE({7, 2, 0, 0}) && !IsTrtVersionGE({8, 0, 0, 0})) {
    auto banned_turing_72 = GetBannedTRT72TuringTactics();
    return banned_turing_72.find(id) != banned_turing_72.end();
  }
  return false;
}

bool AlgorithmSelectorImpl::AllowShuffleAlgorithm(
    TacticID tactic, nvinfer1::DataType input_dtype,
    nvinfer1::TensorFormat input_format) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSalgorithm_selectorDTcc mht_7(mht_7_v, 309, "", "./tensorflow/compiler/tf2tensorrt/convert/algorithm_selector.cc", "AlgorithmSelectorImpl::AllowShuffleAlgorithm");

  if (IsTrtVersionGE({8, 0, 0, 0}) && !IsTrtVersionGE({8, 0, 3, 0})) {
    // Reject shuffle node when input format is linear row major INT8
    // format in TensorRT 8.0 GA.
    return !(input_format == nvinfer1::TensorFormat::kLINEAR &&
             input_dtype == nvinfer1::DataType::kINT8);
  }

  if (IsTrtVersionGE({7, 2, 0, 0}) && !IsTrtVersionGE({8, 0, 0, 0})) {
    // For TRT 7.2, accept shuffle node when input format is not 32-wide
    // channel vectorized row major FP32 format
    return !(input_format == nvinfer1::TensorFormat::kCHW32 &&
             input_dtype == nvinfer1::DataType::kFLOAT);
  }
  return true;
}

bool AlgorithmSelectorImpl::IsAlgorithmSelectorRequired() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSalgorithm_selectorDTcc mht_8(mht_8_v, 329, "", "./tensorflow/compiler/tf2tensorrt/convert/algorithm_selector.cc", "AlgorithmSelectorImpl::IsAlgorithmSelectorRequired");

  // If we are in turing for TensorRT 7.2, we need the  selector for shuffle and
  // avoiding specfic Turing tactics.
  if (IsTrtVersionGE({7, 2, 0, 0}) && !IsTrtVersionGE({8, 0, 0, 0})) {
    return true;
  }

  // If we are in TensorRT 8.0 GA, we want to reject certain types of shuffles.
  if (IsTrtVersionGE({8, 0, 0, 0}) && !IsTrtVersionGE({8, 0, 3, 0})) {
    return true;
  }

  return false;
}

namespace {

string FormatAlgorithmList(const nvinfer1::IAlgorithmContext& ctx,
                           absl::Span<const nvinfer1::IAlgorithm* const> algs) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSalgorithm_selectorDTcc mht_9(mht_9_v, 350, "", "./tensorflow/compiler/tf2tensorrt/convert/algorithm_selector.cc", "FormatAlgorithmList");

  return absl::StrFormat(
      "%s:\n\t%s", absl::FormatStreamed(ctx),
      absl::StrJoin(
          algs, "\n\t",
          [&ctx](std::string* out, const nvinfer1::IAlgorithm* const alg) {
            absl::StrAppendFormat(out, "%s", absl::FormatStreamed(*alg));
            for (int i = 0; i < ctx.getNbInputs() + ctx.getNbOutputs(); i++) {
              absl::StrAppendFormat(
                  out, "\n\t\t%s",
                  absl::FormatStreamed(ALGORITHM_IO_INFO_BY_IDX(*alg, i)));
            }
          }));
}

}  // namespace

TftrtAlgorithmSelector::TftrtAlgorithmSelector()
    : fixed_algorithm_idx_(GetFixedAlgorithmID()),
      selector_(AlgorithmSelectorImpl::CompileTimeTRTVersion()) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSalgorithm_selectorDTcc mht_10(mht_10_v, 372, "", "./tensorflow/compiler/tf2tensorrt/convert/algorithm_selector.cc", "TftrtAlgorithmSelector::TftrtAlgorithmSelector");
}

absl::optional<int64_t> TftrtAlgorithmSelector::GetFixedAlgorithmID() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSalgorithm_selectorDTcc mht_11(mht_11_v, 377, "", "./tensorflow/compiler/tf2tensorrt/convert/algorithm_selector.cc", "TftrtAlgorithmSelector::GetFixedAlgorithmID");

  int64_t trt_algorithm_idx = 0;
  constexpr auto null_idx =
      std::numeric_limits<decltype(trt_algorithm_idx)>::min();
  Status status = tensorflow::ReadInt64FromEnvVar("TF_TRT_FIXED_ALGORITHM_ID",
                                                  /*default_val=*/null_idx,
                                                  &trt_algorithm_idx);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return absl::nullopt;
  }
  if (trt_algorithm_idx != null_idx) {
    return std::max(static_cast<int32_t>(trt_algorithm_idx), 0);
  }
  return absl::nullopt;
}

bool TftrtAlgorithmSelector::AlgorithmPolicy(
    const nvinfer1::IAlgorithmContext& context,
    const nvinfer1::IAlgorithm& alg) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSalgorithm_selectorDTcc mht_12(mht_12_v, 399, "", "./tensorflow/compiler/tf2tensorrt/convert/algorithm_selector.cc", "TftrtAlgorithmSelector::AlgorithmPolicy");

  const nvinfer1::IAlgorithmVariant& variant = alg.getAlgorithmVariant();

  // Check if this tactic ID is banned.
  TacticID tactic_id = variant.getTactic();
  if (selector_.IsBannedTactic(tactic_id)) {
    return false;
  }

  if (selector_.IsShuffleLayer(variant.getImplementation())) {
    return selector_.AllowShuffleAlgorithm(
        tactic_id, alg.getAlgorithmIOInfo(0).getDataType(),
        alg.getAlgorithmIOInfo(0).getTensorFormat());
  }
  return true;
}

int32_t TftrtAlgorithmSelector::selectAlgorithms(
    const nvinfer1::IAlgorithmContext& algoContext,
    const nvinfer1::IAlgorithm* const* algoChoices, int32_t nbChoices,
    int32_t* selection) noexcept {
  if (fixed_algorithm_idx_) {
    LOG(WARNING) << "Forcing TRT algorithm selection to: ID = "
                 << *fixed_algorithm_idx_;
    selection[0] = std::min(*fixed_algorithm_idx_, nbChoices - 1);
    return 1;
  }

  int num_selections = 0;

  VLOG(1) << "Algorithm selection choices: "
          << FormatAlgorithmList(algoContext,
                                 absl::MakeSpan(algoChoices, nbChoices));

  for (int i = 0; i < nbChoices; i++) {
    const nvinfer1::IAlgorithm& alg = *algoChoices[i];

    // Check layer-specific issues.
    if (!AlgorithmPolicy(algoContext, alg)) {
      LOG(WARNING) << absl::StrFormat("Rejecting Algorithm: %s ",
                                      absl::FormatStreamed(alg));
      continue;
    }
    selection[num_selections++] = i;
  }
  return num_selections;
}

// Called by TensorRT to report choices it made.
void TftrtAlgorithmSelector::reportAlgorithms(
    const nvinfer1::IAlgorithmContext* const* algoContexts,
    const nvinfer1::IAlgorithm* const* algoChoices,
    int32_t nbAlgorithms) noexcept {
  if (VLOG_IS_ON(1)) {
    string selection_msg = "Algorithms selected:\n";
    for (int i = 0; i < nbAlgorithms; i++) {
      absl::StrAppend(&selection_msg,
                      FormatAlgorithmList(*algoContexts[i],
                                          absl::MakeSpan(algoChoices + i, 1)));
    }
    VLOG(1) << selection_msg;
  }
}

std::unique_ptr<TftrtAlgorithmSelector> MaybeCreateAlgorithmSelector() {
  auto selector = std::make_unique<TftrtAlgorithmSelector>();

  if (selector->IsRequired()) {
    return selector;
  }

  return nullptr;
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
