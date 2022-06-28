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
class MHTracer_DTPStensorflowPSlitePStocoPStoco_cmdline_flagsDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPStoco_cmdline_flagsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPStoco_cmdline_flagsDTcc() {
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

#include <string>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "absl/types/optional.h"
#include "tensorflow/lite/toco/toco_cmdline_flags.h"
#include "tensorflow/lite/toco/toco_port.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace toco {

bool ParseTocoFlagsFromCommandLineFlags(
    int* argc, char* argv[], std::string* msg,
    ParsedTocoFlags* parsed_toco_flags_ptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPStoco_cmdline_flagsDTcc mht_0(mht_0_v, 202, "", "./tensorflow/lite/toco/toco_cmdline_flags.cc", "ParseTocoFlagsFromCommandLineFlags");

  using tensorflow::Flag;
  ParsedTocoFlags& parsed_flags = *parsed_toco_flags_ptr;
  std::vector<tensorflow::Flag> flags = {
      Flag("input_file", parsed_flags.input_file.bind(),
           parsed_flags.input_file.default_value(),
           "Input file (model of any supported format). For Protobuf "
           "formats, both text and binary are supported regardless of file "
           "extension."),
      Flag("savedmodel_directory", parsed_flags.savedmodel_directory.bind(),
           parsed_flags.savedmodel_directory.default_value(),
           "Deprecated. Full path to the directory containing the SavedModel."),
      Flag("output_file", parsed_flags.output_file.bind(),
           parsed_flags.output_file.default_value(),
           "Output file. "
           "For Protobuf formats, the binary format will be used."),
      Flag("input_format", parsed_flags.input_format.bind(),
           parsed_flags.input_format.default_value(),
           "Input file format. One of: TENSORFLOW_GRAPHDEF, TFLITE."),
      Flag("output_format", parsed_flags.output_format.bind(),
           parsed_flags.output_format.default_value(),
           "Output file format. "
           "One of TENSORFLOW_GRAPHDEF, TFLITE, GRAPHVIZ_DOT."),
      Flag("savedmodel_tagset", parsed_flags.savedmodel_tagset.bind(),
           parsed_flags.savedmodel_tagset.default_value(),
           "Deprecated. Comma-separated set of tags identifying the "
           "MetaGraphDef within the SavedModel to analyze. All tags in the tag "
           "set must be specified."),
      Flag("default_ranges_min", parsed_flags.default_ranges_min.bind(),
           parsed_flags.default_ranges_min.default_value(),
           "If defined, will be used as the default value for the min bound "
           "of min/max ranges used for quantization of uint8 arrays."),
      Flag("default_ranges_max", parsed_flags.default_ranges_max.bind(),
           parsed_flags.default_ranges_max.default_value(),
           "If defined, will be used as the default value for the max bound "
           "of min/max ranges used for quantization of uint8 arrays."),
      Flag("default_int16_ranges_min",
           parsed_flags.default_int16_ranges_min.bind(),
           parsed_flags.default_int16_ranges_min.default_value(),
           "If defined, will be used as the default value for the min bound "
           "of min/max ranges used for quantization of int16 arrays."),
      Flag("default_int16_ranges_max",
           parsed_flags.default_int16_ranges_max.bind(),
           parsed_flags.default_int16_ranges_max.default_value(),
           "If defined, will be used as the default value for the max bound "
           "of min/max ranges used for quantization of int16 arrays."),
      Flag("inference_type", parsed_flags.inference_type.bind(),
           parsed_flags.inference_type.default_value(),
           "Target data type of arrays in the output file (for input_arrays, "
           "this may be overridden by inference_input_type). "
           "One of FLOAT, QUANTIZED_UINT8."),
      Flag("inference_input_type", parsed_flags.inference_input_type.bind(),
           parsed_flags.inference_input_type.default_value(),
           "Target data type of input arrays. "
           "If not specified, inference_type is used. "
           "One of FLOAT, QUANTIZED_UINT8."),
      Flag("input_type", parsed_flags.input_type.bind(),
           parsed_flags.input_type.default_value(),
           "Deprecated ambiguous flag that set both --input_data_types and "
           "--inference_input_type."),
      Flag("input_types", parsed_flags.input_types.bind(),
           parsed_flags.input_types.default_value(),
           "Deprecated ambiguous flag that set both --input_data_types and "
           "--inference_input_type. Was meant to be a "
           "comma-separated list, but this was deprecated before "
           "multiple-input-types was ever properly supported."),

      Flag("drop_fake_quant", parsed_flags.drop_fake_quant.bind(),
           parsed_flags.drop_fake_quant.default_value(),
           "Ignore and discard FakeQuant nodes. For instance, to "
           "generate plain float code without fake-quantization from a "
           "quantized graph."),
      Flag(
          "reorder_across_fake_quant",
          parsed_flags.reorder_across_fake_quant.bind(),
          parsed_flags.reorder_across_fake_quant.default_value(),
          "Normally, FakeQuant nodes must be strict boundaries for graph "
          "transformations, in order to ensure that quantized inference has "
          "the exact same arithmetic behavior as quantized training --- which "
          "is the whole point of quantized training and of FakeQuant nodes in "
          "the first place. "
          "However, that entails subtle requirements on where exactly "
          "FakeQuant nodes must be placed in the graph. Some quantized graphs "
          "have FakeQuant nodes at unexpected locations, that prevent graph "
          "transformations that are necessary in order to generate inference "
          "code for these graphs. Such graphs should be fixed, but as a "
          "temporary work-around, setting this reorder_across_fake_quant flag "
          "allows TOCO to perform necessary graph transformaitons on them, "
          "at the cost of no longer faithfully matching inference and training "
          "arithmetic."),
      Flag("allow_custom_ops", parsed_flags.allow_custom_ops.bind(),
           parsed_flags.allow_custom_ops.default_value(),
           "If true, allow TOCO to create TF Lite Custom operators for all the "
           "unsupported TensorFlow ops."),
      Flag("custom_opdefs", parsed_flags.custom_opdefs.bind(),
           parsed_flags.custom_opdefs.default_value(),
           "List of strings representing custom ops OpDefs that are included "
           "in the GraphDef."),
      Flag("allow_dynamic_tensors", parsed_flags.allow_dynamic_tensors.bind(),
           parsed_flags.allow_dynamic_tensors.default_value(),
           "Boolean flag indicating whether the converter should allow models "
           "with dynamic Tensor shape. When set to False, the converter will "
           "generate runtime memory offsets for activation Tensors (with 128 "
           "bits alignment) and error out on models with undetermined Tensor "
           "shape. (Default: True)"),
      Flag(
          "drop_control_dependency",
          parsed_flags.drop_control_dependency.bind(),
          parsed_flags.drop_control_dependency.default_value(),
          "If true, ignore control dependency requirements in input TensorFlow "
          "GraphDef. Otherwise an error will be raised upon control dependency "
          "inputs."),
      Flag("debug_disable_recurrent_cell_fusion",
           parsed_flags.debug_disable_recurrent_cell_fusion.bind(),
           parsed_flags.debug_disable_recurrent_cell_fusion.default_value(),
           "If true, disable fusion of known identifiable cell subgraphs into "
           "cells. This includes, for example, specific forms of LSTM cell."),
      Flag("propagate_fake_quant_num_bits",
           parsed_flags.propagate_fake_quant_num_bits.bind(),
           parsed_flags.propagate_fake_quant_num_bits.default_value(),
           "If true, use FakeQuant* operator num_bits attributes to adjust "
           "array data_types."),
      Flag("allow_nudging_weights_to_use_fast_gemm_kernel",
           parsed_flags.allow_nudging_weights_to_use_fast_gemm_kernel.bind(),
           parsed_flags.allow_nudging_weights_to_use_fast_gemm_kernel
               .default_value(),
           "Some fast uint8 GEMM kernels require uint8 weights to avoid the "
           "value 0. This flag allows nudging them to 1 to allow proceeding, "
           "with moderate inaccuracy."),
      Flag("dedupe_array_min_size_bytes",
           parsed_flags.dedupe_array_min_size_bytes.bind(),
           parsed_flags.dedupe_array_min_size_bytes.default_value(),
           "Minimum size of constant arrays to deduplicate; arrays smaller "
           "will not be deduplicated."),
      Flag("split_tflite_lstm_inputs",
           parsed_flags.split_tflite_lstm_inputs.bind(),
           parsed_flags.split_tflite_lstm_inputs.default_value(),
           "Split the LSTM inputs from 5 tensors to 18 tensors for TFLite. "
           "Ignored if the output format is not TFLite."),
      Flag("quantize_to_float16", parsed_flags.quantize_to_float16.bind(),
           parsed_flags.quantize_to_float16.default_value(),
           "Used in conjunction with post_training_quantize. Specifies that "
           "the weights should be quantized to fp16 instead of the default "
           "(int8)"),
      Flag("quantize_weights", parsed_flags.quantize_weights.bind(),
           parsed_flags.quantize_weights.default_value(),
           "Deprecated. Please use --post_training_quantize instead."),
      Flag("post_training_quantize", parsed_flags.post_training_quantize.bind(),
           parsed_flags.post_training_quantize.default_value(),
           "Boolean indicating whether to quantize the weights of the "
           "converted float model. Model size will be reduced and there will "
           "be latency improvements (at the cost of accuracy)."),
      // TODO(b/118822804): Unify the argument definition with `tflite_convert`.
      // WARNING: Experimental interface, subject to change
      Flag("enable_select_tf_ops", parsed_flags.enable_select_tf_ops.bind(),
           parsed_flags.enable_select_tf_ops.default_value(), ""),
      // WARNING: Experimental interface, subject to change
      Flag("force_select_tf_ops", parsed_flags.force_select_tf_ops.bind(),
           parsed_flags.force_select_tf_ops.default_value(), ""),
      // WARNING: Experimental interface, subject to change
      Flag("unfold_batchmatmul", parsed_flags.unfold_batchmatmul.bind(),
           parsed_flags.unfold_batchmatmul.default_value(), ""),
      // WARNING: Experimental interface, subject to change
      Flag("accumulation_type", parsed_flags.accumulation_type.bind(),
           parsed_flags.accumulation_type.default_value(),
           "Accumulation type to use with quantize_to_float16"),
      // WARNING: Experimental interface, subject to change
      Flag("allow_bfloat16", parsed_flags.allow_bfloat16.bind(),
           parsed_flags.allow_bfloat16.default_value(), "")};

  bool asked_for_help =
      *argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-help"));
  if (asked_for_help) {
    *msg += tensorflow::Flags::Usage(argv[0], flags);
    return false;
  } else {
    return tensorflow::Flags::Parse(argc, argv, flags);
  }
}

namespace {

// Defines the requirements for a given flag. kUseDefault means the default
// should be used in cases where the value isn't specified by the user.
enum class FlagRequirement {
  kNone,
  kMustBeSpecified,
  kMustNotBeSpecified,
  kUseDefault,
};

// Enforces the FlagRequirements are met for a given flag.
template <typename T>
void EnforceFlagRequirement(const T& flag, const std::string& flag_name,
                            FlagRequirement requirement) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("flag_name: \"" + flag_name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStoco_cmdline_flagsDTcc mht_1(mht_1_v, 400, "", "./tensorflow/lite/toco/toco_cmdline_flags.cc", "EnforceFlagRequirement");

  if (requirement == FlagRequirement::kMustBeSpecified) {
    QCHECK(flag.specified()) << "Missing required flag " << flag_name;
  }
  if (requirement == FlagRequirement::kMustNotBeSpecified) {
    QCHECK(!flag.specified())
        << "Given other flags, this flag should not have been specified: "
        << flag_name;
  }
}

// Gets the value from the flag if specified. Returns default if the
// FlagRequirement is kUseDefault.
template <typename T>
absl::optional<T> GetFlagValue(const Arg<T>& flag,
                               FlagRequirement requirement) {
  if (flag.specified()) return flag.value();
  if (requirement == FlagRequirement::kUseDefault) return flag.default_value();
  return absl::optional<T>();
}

}  // namespace

void ReadTocoFlagsFromCommandLineFlags(const ParsedTocoFlags& parsed_toco_flags,
                                       TocoFlags* toco_flags) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPStoco_cmdline_flagsDTcc mht_2(mht_2_v, 427, "", "./tensorflow/lite/toco/toco_cmdline_flags.cc", "ReadTocoFlagsFromCommandLineFlags");

  namespace port = toco::port;
  port::CheckInitGoogleIsDone("InitGoogle is not done yet");

#define READ_TOCO_FLAG(name, requirement)                                \
  do {                                                                   \
    EnforceFlagRequirement(parsed_toco_flags.name, #name, requirement);  \
    auto flag_value = GetFlagValue(parsed_toco_flags.name, requirement); \
    if (flag_value.has_value()) {                                        \
      toco_flags->set_##name(flag_value.value());                        \
    }                                                                    \
  } while (false)

#define PARSE_TOCO_FLAG(Type, name, requirement)                         \
  do {                                                                   \
    EnforceFlagRequirement(parsed_toco_flags.name, #name, requirement);  \
    auto flag_value = GetFlagValue(parsed_toco_flags.name, requirement); \
    if (flag_value.has_value()) {                                        \
      Type x;                                                            \
      QCHECK(Type##_Parse(flag_value.value(), &x))                       \
          << "Unrecognized " << #Type << " value "                       \
          << parsed_toco_flags.name.value();                             \
      toco_flags->set_##name(x);                                         \
    }                                                                    \
  } while (false)

  PARSE_TOCO_FLAG(FileFormat, input_format, FlagRequirement::kUseDefault);
  PARSE_TOCO_FLAG(FileFormat, output_format, FlagRequirement::kUseDefault);
  PARSE_TOCO_FLAG(IODataType, inference_type, FlagRequirement::kNone);
  PARSE_TOCO_FLAG(IODataType, inference_input_type, FlagRequirement::kNone);
  READ_TOCO_FLAG(default_ranges_min, FlagRequirement::kNone);
  READ_TOCO_FLAG(default_ranges_max, FlagRequirement::kNone);
  READ_TOCO_FLAG(default_int16_ranges_min, FlagRequirement::kNone);
  READ_TOCO_FLAG(default_int16_ranges_max, FlagRequirement::kNone);
  READ_TOCO_FLAG(drop_fake_quant, FlagRequirement::kNone);
  READ_TOCO_FLAG(reorder_across_fake_quant, FlagRequirement::kNone);
  READ_TOCO_FLAG(allow_custom_ops, FlagRequirement::kNone);
  READ_TOCO_FLAG(drop_control_dependency, FlagRequirement::kNone);
  READ_TOCO_FLAG(debug_disable_recurrent_cell_fusion, FlagRequirement::kNone);
  READ_TOCO_FLAG(propagate_fake_quant_num_bits, FlagRequirement::kNone);
  READ_TOCO_FLAG(allow_nudging_weights_to_use_fast_gemm_kernel,
                 FlagRequirement::kNone);
  READ_TOCO_FLAG(dedupe_array_min_size_bytes, FlagRequirement::kNone);
  READ_TOCO_FLAG(split_tflite_lstm_inputs, FlagRequirement::kNone);
  READ_TOCO_FLAG(quantize_weights, FlagRequirement::kNone);
  READ_TOCO_FLAG(quantize_to_float16, FlagRequirement::kNone);
  READ_TOCO_FLAG(post_training_quantize, FlagRequirement::kNone);
  READ_TOCO_FLAG(enable_select_tf_ops, FlagRequirement::kNone);
  READ_TOCO_FLAG(force_select_tf_ops, FlagRequirement::kNone);
  READ_TOCO_FLAG(unfold_batchmatmul, FlagRequirement::kNone);
  PARSE_TOCO_FLAG(IODataType, accumulation_type, FlagRequirement::kNone);
  READ_TOCO_FLAG(allow_bfloat16, FlagRequirement::kNone);

  if (parsed_toco_flags.force_select_tf_ops.value() &&
      !parsed_toco_flags.enable_select_tf_ops.value()) {
    // TODO(ycling): Consider to enforce `enable_select_tf_ops` when
    // `force_select_tf_ops` is true.
    LOG(WARNING) << "--force_select_tf_ops should always be used with "
                    "--enable_select_tf_ops.";
  }

  // Deprecated flag handling.
  if (parsed_toco_flags.input_type.specified()) {
    LOG(WARNING)
        << "--input_type is deprecated. It was an ambiguous flag that set both "
           "--input_data_types and --inference_input_type. If you are trying "
           "to complement the input file with information about the type of "
           "input arrays, use --input_data_type. If you are trying to control "
           "the quantization/dequantization of real-numbers input arrays in "
           "the output file, use --inference_input_type.";
    toco::IODataType input_type;
    QCHECK(toco::IODataType_Parse(parsed_toco_flags.input_type.value(),
                                  &input_type));
    toco_flags->set_inference_input_type(input_type);
  }
  if (parsed_toco_flags.input_types.specified()) {
    LOG(WARNING)
        << "--input_types is deprecated. It was an ambiguous flag that set "
           "both --input_data_types and --inference_input_type. If you are "
           "trying to complement the input file with information about the "
           "type of input arrays, use --input_data_type. If you are trying to "
           "control the quantization/dequantization of real-numbers input "
           "arrays in the output file, use --inference_input_type.";
    std::vector<std::string> input_types =
        absl::StrSplit(parsed_toco_flags.input_types.value(), ',');
    QCHECK(!input_types.empty());
    for (size_t i = 1; i < input_types.size(); i++) {
      QCHECK_EQ(input_types[i], input_types[0]);
    }
    toco::IODataType input_type;
    QCHECK(toco::IODataType_Parse(input_types[0], &input_type));
    toco_flags->set_inference_input_type(input_type);
  }
  if (parsed_toco_flags.quantize_weights.value()) {
    LOG(WARNING)
        << "--quantize_weights is deprecated. Falling back to "
           "--post_training_quantize. Please switch --post_training_quantize.";
    toco_flags->set_post_training_quantize(
        parsed_toco_flags.quantize_weights.value());
  }
  if (parsed_toco_flags.quantize_weights.value()) {
    if (toco_flags->inference_type() == IODataType::QUANTIZED_UINT8) {
      LOG(WARNING)
          << "--post_training_quantize quantizes a graph of inference_type "
             "FLOAT. Overriding inference type QUANTIZED_UINT8 to FLOAT.";
      toco_flags->set_inference_type(IODataType::FLOAT);
    }
  }

#undef READ_TOCO_FLAG
#undef PARSE_TOCO_FLAG
}
}  // namespace toco
