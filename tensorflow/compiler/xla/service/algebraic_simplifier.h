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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_ALGEBRAIC_SIMPLIFIER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_ALGEBRAIC_SIMPLIFIER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh() {
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


#include <functional>
#include <utility>

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

class AlgebraicSimplifierOptions {
 public:
  // Platform dependent callback to determine if a reshape `from_shape` to
  // `to_shape` is a bitcast.
  using ReshapeIsBitcastCallback =
      std::function<bool(const Shape& from_shape, const Shape& to_shape)>;
  // Platform dependent callback to determine if a set of reverse dimensions is
  // lowerable
  using ConvIsLowerableCallback = std::function<bool(HloInstruction* window)>;

  explicit AlgebraicSimplifierOptions(
      ReshapeIsBitcastCallback reshape_is_bitcast_callback = {},
      ConvIsLowerableCallback conv_is_lowerable_callback = {})
      : reshape_is_bitcast_callback_(std::move(reshape_is_bitcast_callback)),
        conv_is_lowerable_callback_(std::move(conv_is_lowerable_callback)) {}

  // Use the platform specific callback if set. It is not sensible to return
  // true here if the options are not layout sensitive.
  bool ReshapeIsBitcast(const Shape& from_shape, const Shape& to_shape) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_0(mht_0_v, 217, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "ReshapeIsBitcast");

    if (!is_layout_sensitive_) {
      return false;
    }
    if (!reshape_is_bitcast_callback_) {
      return ShapeUtil::ReshapeIsBitcast(from_shape, to_shape);
    }
    return reshape_is_bitcast_callback_(from_shape, to_shape);
  }

  // Use the platform specific callback if set. Otherwise, return true.
  bool ConvIsLowerable(HloInstruction* reverse_dims) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_1(mht_1_v, 231, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "ConvIsLowerable");

    if (!conv_is_lowerable_callback_) {
      return true;
    }
    return conv_is_lowerable_callback_(reverse_dims);
  }

  // If is_layout_sensitive is true, then the simplifier preserves layout during
  // transformation. Otherwise, layout is ignored.
  void set_is_layout_sensitive(bool is_layout_sensitive) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_2(mht_2_v, 243, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "set_is_layout_sensitive");

    is_layout_sensitive_ = is_layout_sensitive;
  }

  bool is_layout_sensitive() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_3(mht_3_v, 250, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "is_layout_sensitive");
 return is_layout_sensitive_; }

  // Enable dot simplification on platforms where it is profitable.
  void set_enable_dot_strength_reduction(bool enable_dot_strength_reduction) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_4(mht_4_v, 256, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "set_enable_dot_strength_reduction");

    enable_dot_strength_reduction_ = enable_dot_strength_reduction;
  }

  bool enable_dot_strength_reduction() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_5(mht_5_v, 263, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "enable_dot_strength_reduction");

    return enable_dot_strength_reduction_;
  }

  // Enable dot->multiple rewrite for dot as an outer-product
  void set_enable_dot_to_multiply_rewrite(bool enable_dot_to_multiply_rewrite) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_6(mht_6_v, 271, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "set_enable_dot_to_multiply_rewrite");

    enable_dot_to_multiply_rewrite_ = enable_dot_to_multiply_rewrite;
  }

  bool enable_dot_to_multiply_rewrite() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_7(mht_7_v, 278, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "enable_dot_to_multiply_rewrite");

    return enable_dot_to_multiply_rewrite_;
  }

  // Enable convolution simplification on platforms where it is profitable.
  void set_enable_conv_simplification(bool enable_conv_simplification) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_8(mht_8_v, 286, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "set_enable_conv_simplification");

    enable_conv_simplification_ = enable_conv_simplification;
  }
  bool enable_conv_simplification() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_9(mht_9_v, 292, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "enable_conv_simplification");

    return enable_conv_simplification_;
  }

  // Enable convolution operand swapping on platforms where it is supported.
  void set_enable_conv_operand_swap(bool enable_conv_operand_swap) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_10(mht_10_v, 300, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "set_enable_conv_operand_swap");

    enable_conv_operand_swap_ = enable_conv_operand_swap;
  }
  bool enable_conv_operand_swap() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_11(mht_11_v, 306, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "enable_conv_operand_swap");
 return enable_conv_operand_swap_; }

  // Move constant scalar multiply to one operand or output of convolutions with
  // the smallest tensor size, to reduce the number of scalar multiply.
  void set_enable_scalar_multiply_reduction(
      bool enable_scalar_multiply_reduction) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_12(mht_12_v, 314, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "set_enable_scalar_multiply_reduction");

    enable_scalar_multiply_reduction_ = enable_scalar_multiply_reduction;
  }

  bool enable_scalar_multiply_reduction() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_13(mht_13_v, 321, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "enable_scalar_multiply_reduction");

    return enable_scalar_multiply_reduction_;
  }

  // Also the algebraic simplifer to treat floating point values like real
  // numbers.
  void set_enable_floats_are_real(bool enable_floats_are_real) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_14(mht_14_v, 330, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "set_enable_floats_are_real");

    enable_floats_are_real_ = enable_floats_are_real;
  }

  bool enable_floats_are_real() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_15(mht_15_v, 337, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "enable_floats_are_real");
 return enable_floats_are_real_; }

  // If enable_window_reduce_replacement is true, the kReduceWindow instruction
  // can be optimized by replacement with simpler operations.
  void set_enable_window_reduce_to_reduce_replacement(
      bool enable_window_reduce_to_reduce_replacement) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_16(mht_16_v, 345, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "set_enable_window_reduce_to_reduce_replacement");

    enable_window_reduce_to_reduce_replacement_ =
        enable_window_reduce_to_reduce_replacement;
  }

  bool enable_window_reduce_to_reduce_replacement() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_17(mht_17_v, 353, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "enable_window_reduce_to_reduce_replacement");

    return enable_window_reduce_to_reduce_replacement_;
  }

  // Sets the size of a gather operand that can be unrolled into many selects.
  void set_very_small_gather_size(int64_t size) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_18(mht_18_v, 361, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "set_very_small_gather_size");

    very_small_gather_size_ = size;
  }

  int64_t very_small_gather_size() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_19(mht_19_v, 368, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "very_small_gather_size");
 return very_small_gather_size_; }

  void set_cudnn_batchnorm_forward_training_metadata(const std::string& c) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("c: \"" + c + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_20(mht_20_v, 374, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "set_cudnn_batchnorm_forward_training_metadata");

    metadata_.cudnn_batchnorm_forward_training_metadata = c;
  }

  const std::string& get_cudnn_batchnorm_forward_training_metadata() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_21(mht_21_v, 381, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "get_cudnn_batchnorm_forward_training_metadata");

    return metadata_.cudnn_batchnorm_forward_training_metadata;
  }

  void set_enable_reduce_of_reshape(bool enable_reduce_of_reshape) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_22(mht_22_v, 388, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "set_enable_reduce_of_reshape");

    enable_reduce_of_reshape_ = enable_reduce_of_reshape;
  }

  bool enable_reduce_of_reshape() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_23(mht_23_v, 395, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "enable_reduce_of_reshape");
 return enable_reduce_of_reshape_; }

  void set_enable_negative_padding_replacement(
      bool enable_negative_padding_replacement) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_24(mht_24_v, 401, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "set_enable_negative_padding_replacement");

    enable_negative_padding_replacement_ = enable_negative_padding_replacement;
  }

  bool enable_negative_padding_replacement() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_25(mht_25_v, 408, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "enable_negative_padding_replacement");

    return enable_negative_padding_replacement_;
  }

  void set_enable_sink_broadcast(bool enable_sink_broadcast) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_26(mht_26_v, 415, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "set_enable_sink_broadcast");

    enable_sink_broadcast_ = enable_sink_broadcast;
  }

  bool enable_sink_broadcast() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_27(mht_27_v, 422, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "enable_sink_broadcast");
 return enable_sink_broadcast_; }

  void set_replace_transpose_with_bitcast(bool replace_transpose_with_bitcast) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_28(mht_28_v, 427, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "set_replace_transpose_with_bitcast");

    replace_transpose_with_bitcast_ = replace_transpose_with_bitcast;
  }

  bool replace_transpose_with_bitcast() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_29(mht_29_v, 434, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "replace_transpose_with_bitcast");

    return replace_transpose_with_bitcast_;
  }

  // If true, min(x, NaN) = NaN.  If false, min(x, NaN) = x.
  //
  // TODO(b/209827141): Remove this and make minmax_propagate_nan uncondtionally
  // true.
  bool minmax_propagate_nan() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_30(mht_30_v, 445, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "minmax_propagate_nan");
 return minmax_propagate_nan_; }
  void set_minmax_propagate_nan(bool val) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_31(mht_31_v, 449, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "set_minmax_propagate_nan");
 minmax_propagate_nan_ = val; }

 private:
  // Metadata struct can be used to store any metadata information encapsulated
  // with the AlgebraicSimplierOptions that can be later used in an
  // AlgebraicSimplifier pass. For example,
  // cudnn_batchnorm_forward_training_metadata can be used to store the name of
  // a custom call. If the custom call is
  // __cudnn$batchNormalizationForwardTraining, the output with index 2 is
  // guaranteed to be postive. This property has been used to recursively
  // determine if the operand of an instruction is always positive.
  struct Metadata {
    std::string cudnn_batchnorm_forward_training_metadata{""};
    Metadata() {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_32(mht_32_v, 465, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "Metadata");
}
  };
  ReshapeIsBitcastCallback reshape_is_bitcast_callback_;
  ConvIsLowerableCallback conv_is_lowerable_callback_;
  bool is_layout_sensitive_{false};
  bool enable_dot_strength_reduction_{true};
  bool enable_dot_to_multiply_rewrite_{true};
  bool enable_conv_simplification_{true};
  bool enable_conv_operand_swap_{true};
  bool enable_scalar_multiply_reduction_{false};
  bool enable_floats_are_real_{false};
  bool enable_window_reduce_to_reduce_replacement_{true};
  bool enable_reduce_of_reshape_{true};
  bool enable_negative_padding_replacement_{true};
  bool enable_sink_broadcast_{true};
  bool replace_transpose_with_bitcast_{true};
  int64_t very_small_gather_size_{4};
  bool minmax_propagate_nan_{true};
  Metadata metadata_;
};

// A pass which performs algebraic simplifications.
class AlgebraicSimplifier : public HloModulePass {
 public:
  // If is_layout_sensitive is true, then the simplifier preserves layout during
  // transformation. Otherwise, layout is ignored.
  explicit AlgebraicSimplifier(const AlgebraicSimplifierOptions& options)
      : options_(options) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_33(mht_33_v, 495, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "AlgebraicSimplifier");
}
  ~AlgebraicSimplifier() override = default;
  absl::string_view name() const override {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_34(mht_34_v, 500, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "name");
 return "algsimp"; }

  // Run algebraic simplification on the given computation. Returns whether the
  // computation was changed.
  StatusOr<bool> Run(HloModule* module) override;

  // Create constant from literal with tiles and element size updated in the
  // constant's layout.
  std::unique_ptr<HloInstruction> CreateConstantWithLayoutUpdated(
      Literal literal) {
    auto constant = HloInstruction::CreateConstant(std::move(literal));
    UpdateLayout(constant->mutable_shape());
    return constant;
  }

 protected:
  AlgebraicSimplifierOptions options_;
};

// AlgebraicSimplifierVisitor traverses the HLO computation and reduces certain
// algebraic expressions to simplified forms. Note: This only supports
// simplifications that simply look at the operands of an instruction. For the
// more general case a worklist based approach would be needed.
class AlgebraicSimplifierVisitor : public DfsHloRewriteVisitor {
 public:
  explicit AlgebraicSimplifierVisitor(const AlgebraicSimplifierOptions& options,
                                      AlgebraicSimplifier* simplifier)
      : options_(options), simplifier_(simplifier) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_35(mht_35_v, 530, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "AlgebraicSimplifierVisitor");
}

  Status HandleAbs(HloInstruction* abs) override;

  Status HandleAdd(HloInstruction* add) override;

  Status HandleAnd(HloInstruction* logical_and) override;

  Status HandleBitcast(HloInstruction* bitcast) override;

  Status HandleBitcastConvert(HloInstruction* bitcast) override;

  Status HandleBroadcast(HloInstruction* broadcast) override;

  Status HandleCompare(HloInstruction* compare) override;

  Status HandleConcatenate(HloInstruction* concatenate) override;

  Status HandleConstant(HloInstruction* constant) override;

  Status HandleCopy(HloInstruction* copy) override;

  Status HandleConvert(HloInstruction* convert) override;

  Status HandleComplex(HloInstruction* complex) override;

  Status HandleReal(HloInstruction* real) override;

  Status HandleImag(HloInstruction* imag) override;

  Status HandleIota(HloInstruction* instruction) override;

  Status HandleConvolution(HloInstruction* convolution) override;

  Status HandleDivide(HloInstruction* divide) override;

  Status HandleDot(HloInstruction* dot) override;

  Status HandleGather(HloInstruction* gather) override;

  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;

  Status HandleLog(HloInstruction* log) override;

  Status HandleMaximum(HloInstruction* maximum) override;

  Status HandleMinimum(HloInstruction* minimum) override;

  Status HandleClamp(HloInstruction* clamp) override;

  Status HandleMultiply(HloInstruction* multiply) override;

  Status HandleNegate(HloInstruction* negate) override;

  Status HandleNot(HloInstruction* logical_not) override;

  Status HandleOr(HloInstruction* logical_or) override;

  Status HandlePad(HloInstruction* pad) override;

  Status HandlePower(HloInstruction* power) override;

  Status HandleRemainder(HloInstruction* remainder) override;

  Status HandleReshape(HloInstruction* reshape) override;

  Status HandleReduce(HloInstruction* hlo) override;

  Status HandleReduceWindow(HloInstruction* hlo) override;

  Status HandleReverse(HloInstruction* reverse) override;

  Status HandleRsqrt(HloInstruction* rsqrt) override;

  Status HandleSlice(HloInstruction* slice) override;

  Status HandleSqrt(HloInstruction* sqrt) override;

  Status HandleDynamicSlice(HloInstruction* dynamic_slice) override;

  Status HandleDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice) override;
  Status HandleScatter(HloInstruction* scatter) override;

  Status HandleSelect(HloInstruction* select) override;

  Status HandleSort(HloInstruction* sort) override;

  Status HandleTranspose(HloInstruction* transpose) override;

  Status HandleSubtract(HloInstruction* sub) override;

  Status HandleMap(HloInstruction* map) override;

  // Runs the visitor on a computation.
  bool Run(HloComputation* computation,
           const AlgebraicSimplifierOptions& options,
           AlgebraicSimplifier* simplifier);

 private:
  // Removes degenerate dimension from dot.
  StatusOr<bool> RemoveDegenerateDimensionFromDot(HloInstruction* dot);

  // Converts to primitive type if the input hlo is not that type, otherwise
  // returns the original hlo.
  HloInstruction* AsType(HloInstruction* hlo,
                         const PrimitiveType element_type) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_36(mht_36_v, 639, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "AsType");

    if (hlo->shape().element_type() == element_type) {
      return hlo;
    }
    Shape changed_shape =
        ShapeUtil::ChangeElementType(hlo->shape(), element_type);
    simplifier_->UpdateLayout(&changed_shape);
    return computation_->AddInstruction(
        HloInstruction::CreateConvert(changed_shape, hlo));
  }

  // Transposes a dot operand such that the batch dimensions are the most major,
  // and the contracting dimensions are most minor.
  StatusOr<HloInstruction*> NormalizeDotOperandToBatchMajorAndContractingMinor(
      HloInstruction* dot_operand, absl::Span<const int64_t> batch_dimensions,
      absl::Span<const int64_t> contracting_dimensions);

  // Simplify dot(transpose(a), transpose(b)) to transpose(dot(b,a)) (or
  // transpose(dot(a,b)) if only the batch dims are transposed).
  //
  // Requires the dot has been canonicalized by DotDecomposer into
  //
  //   LHS [batch dims..., non-contracting dim, contracting dim]
  //   RHS [batch dims..., contracting dim, non-contracting dim].
  StatusOr<bool> RemoveTransposesFromDotOperands(HloInstruction* dot);

  // Helper method to perform and add reduction on a list of dimensions.
  HloInstruction* AddReduce(HloInstruction* hlo, absl::Span<const int64_t> dims,
                            PrimitiveType type);

  // Move scalar multiply to the smallest side of convolution to
  // reduce multiply computations.
  Status ScalarMultiplyReduction(HloInstruction* dot);

  // Convenience method for replacing an instruction with a bitcast. If operand
  // is not null, then the bitcast will use the specified operand instead of the
  // operand of the instruction.
  void ReplaceWithBitcast(HloInstruction* instruction,
                          HloInstruction* operand = nullptr);

  // Replace old instruction with new instruction if old and new instructions
  // are compatible (have the same shape and replacement preserves sharding).
  // Updates uses and root instruction. Returns whether a replacement was made.
  bool ReplaceInstructionIfCompatible(HloInstruction* old_instruction,
                                      HloInstruction* new_instruction);

  // Returns whether the shape of the output of the given instructions are the
  // same for the purposes of simplification. If options_.is_layout_sensitive()
  // is true, then this tests shape equality including layout
  // (ShapeUtil::Equal). If options_.is_layout_sensitive() is false, then the
  // tests shape compatibility (ShapeUtil::Compatible).
  bool SameShape(const HloInstruction* lhs, const HloInstruction* rhs) const;

  // A Broadcast that feeds an element-wise operation with a unique non-scalar
  // operand can sink to after the operation.
  StatusOr<bool> TryToSinkBroadcastAfterOpWithUniqueNonScalarOperand(
      HloInstruction* broadcast);

  StatusOr<HloInstruction*> OptimizeDotOfConcat(HloInstruction* dot);
  StatusOr<HloInstruction*> OptimizeDotOfConcatHelper(
      HloInstruction* dot, HloInstruction* lhs, int64_t lhs_contracting_dim,
      HloInstruction* rhs, int64_t rhs_contracting_dim, bool swapped);

  StatusOr<HloInstruction*> OptimizeDotOfGather(HloInstruction* dot);

  StatusOr<HloInstruction*> OptimizeDotOfReorderContractingDims(
      HloInstruction* dot);

  HloComputation* GetOrCreateScalarAddComputation(PrimitiveType type) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSalgebraic_simplifierDTh mht_37(mht_37_v, 710, "", "./tensorflow/compiler/xla/service/algebraic_simplifier.h", "GetOrCreateScalarAddComputation");

    HloComputation*& scalar_add_computation = scalar_add_computations_[type];
    if (scalar_add_computation) {
      return scalar_add_computation;
    }

    HloComputation::Builder b("scalar_add_computation");
    Shape shape = ShapeUtil::MakeShape(type, {});
    simplifier_->UpdateLayout(&shape);
    auto scalar_lhs = b.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "scalar_lhs"));
    auto scalar_rhs = b.AddInstruction(
        HloInstruction::CreateParameter(1, shape, "scalar_rhs"));
    auto scalar_op = b.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, scalar_lhs, scalar_rhs));
    scalar_add_computation =
        computation_->parent()->AddEmbeddedComputation(b.Build(scalar_op));
    return scalar_add_computation;
  }

  // Tries to fold a kPad in the input or filter into the convolution
  // instruction's window.
  virtual StatusOr<bool> FoldConvInputPad(HloInstruction* convolution);
  StatusOr<bool> FoldConvFilterPad(HloInstruction* convolution);

  // Tries to swap convolution operands if they would result in a more efficient
  // convolution.
  StatusOr<bool> SwapConvOperands(HloInstruction* convolution);

  // Tries to use a kDot in place of the given convolution.
  StatusOr<bool> SimplifyConvToDot(HloInstruction* convolution);

  // Tries to simplify a slice where the result of the slice is a scalar.
  StatusOr<bool> TrySimplifyScalarSlice(HloInstruction* slice);

  // Tries to convert slice(reshape(X)) into reshape(slice(X))
  StatusOr<bool> TryToReorderSliceAndReshape(HloInstruction* slice);

  // Tries to convert slice(reverse(X)) into reverse(slice(X))
  StatusOr<bool> TryToReorderSliceAndReverse(HloInstruction* slice);

  // Tries to simplify `(and (< a N) (< a K))` in cases where `N <= K` into
  // `(< a N)`. This is crucial for being able to figure out the loop trip
  // count.
  //
  // Assumes that the input is conjunction.
  StatusOr<bool> TrySimplifyTautologicalCompare(HloInstruction* conjunction);

  // Tries to simlplify (bitcast-convert (concat (bitcast-convert A) ...)) where
  // the types of inner and outer bitcast-convert cancel out.
  StatusOr<bool> TrySimplifyTautologicalBitcastConvert(HloInstruction* bitcast);

  // Useful when we want to use the same visitor over multiple computations.
  void ResetState(HloComputation* computation);

  // Current HloComputation instance the AlgebraicSimplifierVisitor is
  // traversing.
  HloComputation* computation_;

  // The backend-specific options selected for the algebraic simplifier.
  const AlgebraicSimplifierOptions& options_;

  // Cached computation for adding two scalars of a given type.
  absl::flat_hash_map<PrimitiveType, HloComputation*> scalar_add_computations_;

  AlgebraicSimplifier* simplifier_ = nullptr;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_ALGEBRAIC_SIMPLIFIER_H_
