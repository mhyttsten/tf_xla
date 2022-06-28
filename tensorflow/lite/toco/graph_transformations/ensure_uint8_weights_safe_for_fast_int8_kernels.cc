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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSensure_uint8_weights_safe_for_fast_int8_kernelsDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSensure_uint8_weights_safe_for_fast_int8_kernelsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSensure_uint8_weights_safe_for_fast_int8_kernelsDTcc() {
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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

// === Summary ===
//
// TLDR: Some of our 8-bit arithmetic operations require uint8 weight values
// to avoid the value 0, thus ranging only in [1, 255]. This enables faster
// runtime arithmetic kernels on ARM NEON. This is not relevant on most
// other hardware architectures, and will cease to be relevant on ARM NEON
// in the future. These topics are elaborated below ("Context").
//
// Having just one isolated uint8 value equal to 0 is fine. The bad case is when
// two uint8 values are both zero and are less than 16 bytes apart.
//
// By default, toco generates a fatal error when that happens. The user may opt
// in to more lax behavior by passing
//   --allow_nudging_weights_to_use_fast_gemm_kernel.
// This causes toco to nudge such bad 0 values into the value 1, thus avoiding
// the problem in exchange for compromising on accuracy.
//
// The present graph transformation implements both the default fatal-erroring
// behavior, and, when allow_nudging_weights is set, also the lax nudging
// behavior.
//
//
// === Context ===
//
// Since March 2017, we have been using a trick to perform faster
// 8bit matrix multiplications, to our knowledge first implemented in gemmlowp
// here:
//   https://github.com/google/gemmlowp/commit/25b2989415b99e797e1ab977837111b2e231f81f
//
// This trick is explained in Appendix B of our paper,
//   https://arxiv.org/abs/1712.05877
//
// Here is the relevant paragraph:
//
//      For efficient NEON implementation of the matrix multiplication’s
//      core accumulation, we use the following trick.
//      In the multiply-add operation in (10), we first change the
//      operands’ type from uint8 to int8 (which can be done by
//      subtracting 128 from the quantized values and zero-points).
//      Thus the core multiply-add becomes
//
//            int32 += int8 * int8. (B.1)
//
//      As mentioned in section 3, with a minor tweak of the quantized
//      training process, we can ensure that the weights, once
//      quantized as int8 values, never take the value −128. Hence,
//      the product in (B.1) is never −128 ∗ −128, and is therefore
//      always less than 2^14 in absolute value. Hence, (B.1)
//      can accumulate two products on a local int16 accumulator
//      before that needs to be accumulated into the true int32 accumulator.
//      This allows the use of an 8-way SIMD multiplication
//      (SMULL on int8 operands), followed by an 8-way
//      SIMD multiply-add (SMLAL on int8 operands), followed
//      by a pairwise-add-and-accumulate into the int32 accumulators
//      (SADALP).
//
// As that paragraph notes, quantized training should be suitably modified to
// ensure that quantized uint8 weights value only range in [1, 255]. So the
// problem that we are dealing with is only about the existing 8-bit quantized
// models that haven't been trained specifically to get 8-bit weights only in
// [1, 255].
//
// This spreadsheet shows the speed benefit of this trick across many existing
// ARM-architecture CPUs:
//
//    https://docs.google.com/spreadsheets/d/1-0LjdMvW0XtH1bYknC0bQINoFaxjTuL9eplZZcitykI/edit?usp=sharing
//
// Compare Row 18 (fast int8 trick) to Row 20 (regular uint8 kernel).
//
// The introduction of the 'dotprod' extension to ARM NEON, specifically the
// SDOT instruction, renders this eventually moot. See the experimental
// kernels contributed by ARM here,
//
//     https://github.com/google/gemmlowp/pull/116
//
// However, as of April 2018, there don't seem to be any commercially available
// CPU supporting these instructions (yet); we are waiting for
// Cortex-A{75,55}-r1 to become available; the "-r1" is key here. Even if such
// CPUs become available soon, it will presumably take years for them to
// overtake the large volume of existing CPUs not supporting these new
// instructions, especially in current and future low-end devices. All in all,
// we can foresee these 'fast int8 kernels' to remain important to have into
// the 2020s.
//
::tensorflow::Status EnsureUint8WeightsSafeForFastInt8Kernels::Run(
    Model* model, std::size_t op_index, bool* modified) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSensure_uint8_weights_safe_for_fast_int8_kernelsDTcc mht_0(mht_0_v, 281, "", "./tensorflow/lite/toco/graph_transformations/ensure_uint8_weights_safe_for_fast_int8_kernels.cc", "EnsureUint8WeightsSafeForFastInt8Kernels::Run");

  *modified = false;
  const auto& op = *model->operators[op_index];
  int weights_index = 0;
  switch (op.type) {
    case OperatorType::kConv:
      weights_index = 1;
      break;
    case OperatorType::kLstmCell:
      weights_index = 2;
      break;
    case OperatorType::kFullyConnected: {
      weights_index = 1;
      const auto& fc_op = static_cast<const toco::FullyConnectedOperator&>(op);
      CHECK(fc_op.weights_format == FullyConnectedWeightsFormat::kDefault)
          << "This graph transformation expects to run before FC weights get "
             "shuffled.";
      break;
    }
    default:
      // Other operator types are unaffected by this graph transformation,
      // because their runtime implementations don't use the fast int8 trick.
      // In particular that's the case of DepthwiseConv at the moment.
      // We have to update this logic when that changes, e.g. if in the future
      // some DepthwiseConv kernel wants to use the trick.
      //
      // The reason why that's not so likely, hence why it's fairly safe to
      // stay conservative in the list of operators that we handle here, is that
      // the fast int8 kernel trick is only applicable to ops that either are
      // implemented as a GEMM, or use symmetric ranges for both weights and
      // activations. The reason why GEMM is special (can use the trick even
      // without symmetric ranges) is that it is so arithmetic-intense that
      // it can use techniques reducing its implementation to the symmetric
      // ranges case, with limited relative overhead (O(N^2) overhead vs
      // O(N^3) GEMM cost). See https://arxiv.org/pdf/1712.05877, section
      // 2.3 Efficient handling of zero-points.
      //
      // That's why at the moment we only handle operators that use a GEMM
      // (Conv, fully-connected --- note that LSTM merely wraps a
      // fully-connected operator).
      return ::tensorflow::Status::OK();
  }

  const std::string& name = op.inputs[weights_index];
  auto& array = model->GetArray(name);
  if (!array.buffer) {
    return ::tensorflow::Status::OK();
  }
  if (array.data_type != ArrayDataType::kUint8) {
    return ::tensorflow::Status::OK();
  }
  auto& buffer_data = array.GetMutableBuffer<ArrayDataType::kUint8>().data;

  int count_bad = 0;
  int index_of_previous_bad_value = 0;
  bool changed = false;

  for (int i = 0, end = buffer_data.size(); i < end; i++) {
    if (buffer_data[i] == 0) {
      count_bad++;
      if (count_bad > 1) {
        const int distance = i - index_of_previous_bad_value;
        // Semi-arbitrary threshold. The idea is that trouble only occurs
        // when two bad values are very close to each other so that they
        // are jointly used within registers inside some GEMM kernel.
        // The details of that depend on the kernel. Our current fast ARM64
        // kernel, for instance, only has an issue when the distance between
        // consecutive bad values is exactly 8. We do not want to track such
        // kernel details too closely here, so we pick a threshold that's
        // a bit larger than that, to give us room to change kernels in the
        // future without worrying.
        static constexpr int kMinDistanceBetweenBadValues = 16;
        if (distance < kMinDistanceBetweenBadValues) {
          if (allow_nudging_weights() || has_default_ranges_flag()) {
            buffer_data[i] = 1;
            changed = true;
            continue;
          }
          LOG(FATAL) << "Bad value for " << name << " at index " << i
                     << ", previous bad value at index "
                     << index_of_previous_bad_value << ", distance=" << distance
                     << ", kMinDistanceBetweenBadValues="
                     << kMinDistanceBetweenBadValues << ". Consider passing "
                     << "--allow_nudging_weights_to_use_fast_gemm_kernel "
                     << "if you don't care about accuracy.";
        }
      }
      index_of_previous_bad_value = i;
    }
  }

  if (changed) {
    if (has_default_ranges_flag()) {
      std::cerr
          << "Since the specified values of --default_ranges_min and "
             "--default_ranges_max result in values incompatible with TFLite's "
             "fast int8 kernels, "
             "--allow_nudging_weights_to_use_fast_gemm_kernel "
             "has been enabled. This may affect the accuracy of the model."
          << std::endl;
    }
    AddMessageF("Tweaked weights values for %s", LogName(op));
  }

  *modified = changed;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
