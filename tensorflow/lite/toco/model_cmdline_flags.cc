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
class MHTracer_DTPStensorflowPSlitePStocoPSmodel_cmdline_flagsDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSmodel_cmdline_flagsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSmodel_cmdline_flagsDTcc() {
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
#include "tensorflow/lite/toco/model_cmdline_flags.h"

#include <string>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/lite/toco/args.h"
#include "tensorflow/lite/toco/toco_graphviz_dump_options.h"
#include "tensorflow/lite/toco/toco_port.h"

// "batch" flag only exists internally
#ifdef PLATFORM_GOOGLE
#include "base/commandlineflags.h"
#endif

namespace toco {

bool ParseModelFlagsFromCommandLineFlags(
    int* argc, char* argv[], std::string* msg,
    ParsedModelFlags* parsed_model_flags_ptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodel_cmdline_flagsDTcc mht_0(mht_0_v, 209, "", "./tensorflow/lite/toco/model_cmdline_flags.cc", "ParseModelFlagsFromCommandLineFlags");

  ParsedModelFlags& parsed_flags = *parsed_model_flags_ptr;
  using tensorflow::Flag;
  std::vector<tensorflow::Flag> flags = {
      Flag("input_array", parsed_flags.input_array.bind(),
           parsed_flags.input_array.default_value(),
           "Deprecated: use --input_arrays instead. Name of the input array. "
           "If not specified, will try to read "
           "that information from the input file."),
      Flag("input_arrays", parsed_flags.input_arrays.bind(),
           parsed_flags.input_arrays.default_value(),
           "Names of the input arrays, comma-separated. If not specified, "
           "will try to read that information from the input file."),
      Flag("output_array", parsed_flags.output_array.bind(),
           parsed_flags.output_array.default_value(),
           "Deprecated: use --output_arrays instead. Name of the output array, "
           "when specifying a unique output array. "
           "If not specified, will try to read that information from the "
           "input file."),
      Flag("output_arrays", parsed_flags.output_arrays.bind(),
           parsed_flags.output_arrays.default_value(),
           "Names of the output arrays, comma-separated. "
           "If not specified, will try to read "
           "that information from the input file."),
      Flag("input_shape", parsed_flags.input_shape.bind(),
           parsed_flags.input_shape.default_value(),
           "Deprecated: use --input_shapes instead. Input array shape. For "
           "many models the shape takes the form "
           "batch size, input array height, input array width, input array "
           "depth."),
      Flag("input_shapes", parsed_flags.input_shapes.bind(),
           parsed_flags.input_shapes.default_value(),
           "Shapes corresponding to --input_arrays, colon-separated. For "
           "many models each shape takes the form batch size, input array "
           "height, input array width, input array depth."),
      Flag("batch_size", parsed_flags.batch_size.bind(),
           parsed_flags.batch_size.default_value(),
           "Deprecated. Batch size for the model. Replaces the first dimension "
           "of an input size array if undefined. Use only with SavedModels "
           "when --input_shapes flag is not specified. Always use "
           "--input_shapes flag with frozen graphs."),
      Flag("input_data_type", parsed_flags.input_data_type.bind(),
           parsed_flags.input_data_type.default_value(),
           "Deprecated: use --input_data_types instead. Input array type, if "
           "not already provided in the graph. "
           "Typically needs to be specified when passing arbitrary arrays "
           "to --input_arrays."),
      Flag("input_data_types", parsed_flags.input_data_types.bind(),
           parsed_flags.input_data_types.default_value(),
           "Input arrays types, comma-separated, if not already provided in "
           "the graph. "
           "Typically needs to be specified when passing arbitrary arrays "
           "to --input_arrays."),
      Flag("mean_value", parsed_flags.mean_value.bind(),
           parsed_flags.mean_value.default_value(),
           "Deprecated: use --mean_values instead. mean_value parameter for "
           "image models, used to compute input "
           "activations from input pixel data."),
      Flag("mean_values", parsed_flags.mean_values.bind(),
           parsed_flags.mean_values.default_value(),
           "mean_values parameter for image models, comma-separated list of "
           "doubles, used to compute input activations from input pixel "
           "data. Each entry in the list should match an entry in "
           "--input_arrays."),
      Flag("std_value", parsed_flags.std_value.bind(),
           parsed_flags.std_value.default_value(),
           "Deprecated: use --std_values instead. std_value parameter for "
           "image models, used to compute input "
           "activations from input pixel data."),
      Flag("std_values", parsed_flags.std_values.bind(),
           parsed_flags.std_values.default_value(),
           "std_value parameter for image models, comma-separated list of "
           "doubles, used to compute input activations from input pixel "
           "data. Each entry in the list should match an entry in "
           "--input_arrays."),
      Flag("variable_batch", parsed_flags.variable_batch.bind(),
           parsed_flags.variable_batch.default_value(),
           "If true, the model accepts an arbitrary batch size. Mutually "
           "exclusive "
           "with the 'batch' field: at most one of these two fields can be "
           "set."),
      Flag("rnn_states", parsed_flags.rnn_states.bind(),
           parsed_flags.rnn_states.default_value(), ""),
      Flag("model_checks", parsed_flags.model_checks.bind(),
           parsed_flags.model_checks.default_value(),
           "A list of model checks to be applied to verify the form of the "
           "model.  Applied after the graph transformations after import."),
      Flag("dump_graphviz", parsed_flags.dump_graphviz.bind(),
           parsed_flags.dump_graphviz.default_value(),
           "Dump graphviz during LogDump call. If string is non-empty then "
           "it defines path to dump, otherwise will skip dumping."),
      Flag("dump_graphviz_video", parsed_flags.dump_graphviz_video.bind(),
           parsed_flags.dump_graphviz_video.default_value(),
           "If true, will dump graphviz at each "
           "graph transformation, which may be used to generate a video."),
      Flag("conversion_summary_dir", parsed_flags.conversion_summary_dir.bind(),
           parsed_flags.conversion_summary_dir.default_value(),
           "Local file directory to store the conversion logs."),
      Flag("allow_nonexistent_arrays",
           parsed_flags.allow_nonexistent_arrays.bind(),
           parsed_flags.allow_nonexistent_arrays.default_value(),
           "If true, will allow passing inexistent arrays in --input_arrays "
           "and --output_arrays. This makes little sense, is only useful to "
           "more easily get graph visualizations."),
      Flag("allow_nonascii_arrays", parsed_flags.allow_nonascii_arrays.bind(),
           parsed_flags.allow_nonascii_arrays.default_value(),
           "If true, will allow passing non-ascii-printable characters in "
           "--input_arrays and --output_arrays. By default (if false), only "
           "ascii printable characters are allowed, i.e. character codes "
           "ranging from 32 to 127. This is disallowed by default so as to "
           "catch common copy-and-paste issues where invisible unicode "
           "characters are unwittingly added to these strings."),
      Flag(
          "arrays_extra_info_file", parsed_flags.arrays_extra_info_file.bind(),
          parsed_flags.arrays_extra_info_file.default_value(),
          "Path to an optional file containing a serialized ArraysExtraInfo "
          "proto allowing to pass extra information about arrays not specified "
          "in the input model file, such as extra MinMax information."),
      Flag("model_flags_file", parsed_flags.model_flags_file.bind(),
           parsed_flags.model_flags_file.default_value(),
           "Path to an optional file containing a serialized ModelFlags proto. "
           "Options specified on the command line will override the values in "
           "the proto."),
      Flag("change_concat_input_ranges",
           parsed_flags.change_concat_input_ranges.bind(),
           parsed_flags.change_concat_input_ranges.default_value(),
           "Boolean to change the behavior of min/max ranges for inputs and"
           " output of the concat operators."),
  };
  bool asked_for_help =
      *argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-help"));
  if (asked_for_help) {
    *msg += tensorflow::Flags::Usage(argv[0], flags);
    return false;
  } else {
    if (!tensorflow::Flags::Parse(argc, argv, flags)) return false;
  }
  auto& dump_options = *GraphVizDumpOptions::singleton();
  dump_options.dump_graphviz_video = parsed_flags.dump_graphviz_video.value();
  dump_options.dump_graphviz = parsed_flags.dump_graphviz.value();

  return true;
}

void ReadModelFlagsFromCommandLineFlags(
    const ParsedModelFlags& parsed_model_flags, ModelFlags* model_flags) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodel_cmdline_flagsDTcc mht_1(mht_1_v, 357, "", "./tensorflow/lite/toco/model_cmdline_flags.cc", "ReadModelFlagsFromCommandLineFlags");

  toco::port::CheckInitGoogleIsDone("InitGoogle is not done yet");

  // Load proto containing the initial model flags.
  // Additional flags specified on the command line will overwrite the values.
  if (parsed_model_flags.model_flags_file.specified()) {
    std::string model_flags_file_contents;
    QCHECK(port::file::GetContents(parsed_model_flags.model_flags_file.value(),
                                   &model_flags_file_contents,
                                   port::file::Defaults())
               .ok())
        << "Specified --model_flags_file="
        << parsed_model_flags.model_flags_file.value()
        << " was not found or could not be read";
    QCHECK(ParseFromStringEitherTextOrBinary(model_flags_file_contents,
                                             model_flags))
        << "Specified --model_flags_file="
        << parsed_model_flags.model_flags_file.value()
        << " could not be parsed";
  }

#ifdef PLATFORM_GOOGLE
  CHECK(!((base::WasPresentOnCommandLine("batch") &&
           parsed_model_flags.variable_batch.specified())))
      << "The --batch and --variable_batch flags are mutually exclusive.";
#endif
  CHECK(!(parsed_model_flags.output_array.specified() &&
          parsed_model_flags.output_arrays.specified()))
      << "The --output_array and --vs flags are mutually exclusive.";

  if (parsed_model_flags.output_array.specified()) {
    model_flags->add_output_arrays(parsed_model_flags.output_array.value());
  }

  if (parsed_model_flags.output_arrays.specified()) {
    std::vector<std::string> output_arrays =
        absl::StrSplit(parsed_model_flags.output_arrays.value(), ',');
    for (const std::string& output_array : output_arrays) {
      model_flags->add_output_arrays(output_array);
    }
  }

  const bool uses_single_input_flags =
      parsed_model_flags.input_array.specified() ||
      parsed_model_flags.mean_value.specified() ||
      parsed_model_flags.std_value.specified() ||
      parsed_model_flags.input_shape.specified();

  const bool uses_multi_input_flags =
      parsed_model_flags.input_arrays.specified() ||
      parsed_model_flags.mean_values.specified() ||
      parsed_model_flags.std_values.specified() ||
      parsed_model_flags.input_shapes.specified();

  QCHECK(!(uses_single_input_flags && uses_multi_input_flags))
      << "Use either the singular-form input flags (--input_array, "
         "--input_shape, --mean_value, --std_value) or the plural form input "
         "flags (--input_arrays, --input_shapes, --mean_values, --std_values), "
         "but not both forms within the same command line.";

  if (parsed_model_flags.input_array.specified()) {
    QCHECK(uses_single_input_flags);
    model_flags->add_input_arrays()->set_name(
        parsed_model_flags.input_array.value());
  }
  if (parsed_model_flags.input_arrays.specified()) {
    QCHECK(uses_multi_input_flags);
    for (const auto& input_array :
         absl::StrSplit(parsed_model_flags.input_arrays.value(), ',')) {
      model_flags->add_input_arrays()->set_name(std::string(input_array));
    }
  }
  if (parsed_model_flags.mean_value.specified()) {
    QCHECK(uses_single_input_flags);
    model_flags->mutable_input_arrays(0)->set_mean_value(
        parsed_model_flags.mean_value.value());
  }
  if (parsed_model_flags.mean_values.specified()) {
    QCHECK(uses_multi_input_flags);
    std::vector<std::string> mean_values =
        absl::StrSplit(parsed_model_flags.mean_values.value(), ',');
    QCHECK(static_cast<int>(mean_values.size()) ==
           model_flags->input_arrays_size());
    for (size_t i = 0; i < mean_values.size(); ++i) {
      char* last = nullptr;
      model_flags->mutable_input_arrays(i)->set_mean_value(
          strtod(mean_values[i].data(), &last));
      CHECK(last != mean_values[i].data());
    }
  }
  if (parsed_model_flags.std_value.specified()) {
    QCHECK(uses_single_input_flags);
    model_flags->mutable_input_arrays(0)->set_std_value(
        parsed_model_flags.std_value.value());
  }
  if (parsed_model_flags.std_values.specified()) {
    QCHECK(uses_multi_input_flags);
    std::vector<std::string> std_values =
        absl::StrSplit(parsed_model_flags.std_values.value(), ',');
    QCHECK(static_cast<int>(std_values.size()) ==
           model_flags->input_arrays_size());
    for (size_t i = 0; i < std_values.size(); ++i) {
      char* last = nullptr;
      model_flags->mutable_input_arrays(i)->set_std_value(
          strtod(std_values[i].data(), &last));
      CHECK(last != std_values[i].data());
    }
  }
  if (parsed_model_flags.input_data_type.specified()) {
    QCHECK(uses_single_input_flags);
    IODataType type;
    QCHECK(IODataType_Parse(parsed_model_flags.input_data_type.value(), &type));
    model_flags->mutable_input_arrays(0)->set_data_type(type);
  }
  if (parsed_model_flags.input_data_types.specified()) {
    QCHECK(uses_multi_input_flags);
    std::vector<std::string> input_data_types =
        absl::StrSplit(parsed_model_flags.input_data_types.value(), ',');
    QCHECK(static_cast<int>(input_data_types.size()) ==
           model_flags->input_arrays_size());
    for (size_t i = 0; i < input_data_types.size(); ++i) {
      IODataType type;
      QCHECK(IODataType_Parse(input_data_types[i], &type));
      model_flags->mutable_input_arrays(i)->set_data_type(type);
    }
  }
  if (parsed_model_flags.input_shape.specified()) {
    QCHECK(uses_single_input_flags);
    if (model_flags->input_arrays().empty()) {
      model_flags->add_input_arrays();
    }
    auto* shape = model_flags->mutable_input_arrays(0)->mutable_shape();
    shape->clear_dims();
    const IntList& list = parsed_model_flags.input_shape.value();
    for (auto& dim : list.elements) {
      shape->add_dims(dim);
    }
  }
  if (parsed_model_flags.input_shapes.specified()) {
    QCHECK(uses_multi_input_flags);
    std::vector<std::string> input_shapes =
        absl::StrSplit(parsed_model_flags.input_shapes.value(), ':');
    QCHECK(static_cast<int>(input_shapes.size()) ==
           model_flags->input_arrays_size());
    for (size_t i = 0; i < input_shapes.size(); ++i) {
      auto* shape = model_flags->mutable_input_arrays(i)->mutable_shape();
      shape->clear_dims();
      // Treat an empty input shape as a scalar.
      if (input_shapes[i].empty()) {
        continue;
      }
      for (const auto& dim_str : absl::StrSplit(input_shapes[i], ',')) {
        int size;
        CHECK(absl::SimpleAtoi(dim_str, &size))
            << "Failed to parse input_shape: " << input_shapes[i];
        shape->add_dims(size);
      }
    }
  }

#define READ_MODEL_FLAG(name)                                   \
  do {                                                          \
    if (parsed_model_flags.name.specified()) {                  \
      model_flags->set_##name(parsed_model_flags.name.value()); \
    }                                                           \
  } while (false)

  READ_MODEL_FLAG(variable_batch);

#undef READ_MODEL_FLAG

  for (const auto& element : parsed_model_flags.rnn_states.value().elements) {
    auto* rnn_state_proto = model_flags->add_rnn_states();
    for (const auto& kv_pair : element) {
      const std::string& key = kv_pair.first;
      const std::string& value = kv_pair.second;
      if (key == "state_array") {
        rnn_state_proto->set_state_array(value);
      } else if (key == "back_edge_source_array") {
        rnn_state_proto->set_back_edge_source_array(value);
      } else if (key == "size") {
        int32_t size = 0;
        CHECK(absl::SimpleAtoi(value, &size));
        CHECK_GT(size, 0);
        rnn_state_proto->set_size(size);
      } else if (key == "num_dims") {
        int32_t size = 0;
        CHECK(absl::SimpleAtoi(value, &size));
        CHECK_GT(size, 0);
        rnn_state_proto->set_num_dims(size);
      } else {
        LOG(FATAL) << "Unknown key '" << key << "' in --rnn_states";
      }
    }
    CHECK(rnn_state_proto->has_state_array() &&
          rnn_state_proto->has_back_edge_source_array() &&
          rnn_state_proto->has_size())
        << "--rnn_states must include state_array, back_edge_source_array and "
           "size.";
  }

  for (const auto& element : parsed_model_flags.model_checks.value().elements) {
    auto* model_check_proto = model_flags->add_model_checks();
    for (const auto& kv_pair : element) {
      const std::string& key = kv_pair.first;
      const std::string& value = kv_pair.second;
      if (key == "count_type") {
        model_check_proto->set_count_type(value);
      } else if (key == "count_min") {
        int32_t count = 0;
        CHECK(absl::SimpleAtoi(value, &count));
        CHECK_GE(count, -1);
        model_check_proto->set_count_min(count);
      } else if (key == "count_max") {
        int32_t count = 0;
        CHECK(absl::SimpleAtoi(value, &count));
        CHECK_GE(count, -1);
        model_check_proto->set_count_max(count);
      } else {
        LOG(FATAL) << "Unknown key '" << key << "' in --model_checks";
      }
    }
  }

  if (!model_flags->has_allow_nonascii_arrays()) {
    model_flags->set_allow_nonascii_arrays(
        parsed_model_flags.allow_nonascii_arrays.value());
  }
  if (!model_flags->has_allow_nonexistent_arrays()) {
    model_flags->set_allow_nonexistent_arrays(
        parsed_model_flags.allow_nonexistent_arrays.value());
  }
  if (!model_flags->has_change_concat_input_ranges()) {
    model_flags->set_change_concat_input_ranges(
        parsed_model_flags.change_concat_input_ranges.value());
  }

  if (parsed_model_flags.arrays_extra_info_file.specified()) {
    std::string arrays_extra_info_file_contents;
    CHECK(port::file::GetContents(
              parsed_model_flags.arrays_extra_info_file.value(),
              &arrays_extra_info_file_contents, port::file::Defaults())
              .ok());
    ParseFromStringEitherTextOrBinary(arrays_extra_info_file_contents,
                                      model_flags->mutable_arrays_extra_info());
  }
}

ParsedModelFlags* UncheckedGlobalParsedModelFlags(bool must_already_exist) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodel_cmdline_flagsDTcc mht_2(mht_2_v, 608, "", "./tensorflow/lite/toco/model_cmdline_flags.cc", "UncheckedGlobalParsedModelFlags");

  static auto* flags = [must_already_exist]() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodel_cmdline_flagsDTcc mht_3(mht_3_v, 612, "", "./tensorflow/lite/toco/model_cmdline_flags.cc", "lambda");

    if (must_already_exist) {
      fprintf(stderr, __FILE__
              ":"
              "GlobalParsedModelFlags() used without initialization\n");
      fflush(stderr);
      abort();
    }
    return new toco::ParsedModelFlags;
  }();
  return flags;
}

ParsedModelFlags* GlobalParsedModelFlags() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodel_cmdline_flagsDTcc mht_4(mht_4_v, 628, "", "./tensorflow/lite/toco/model_cmdline_flags.cc", "GlobalParsedModelFlags");

  return UncheckedGlobalParsedModelFlags(true);
}

void ParseModelFlagsOrDie(int* argc, char* argv[]) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStocoPSmodel_cmdline_flagsDTcc mht_5(mht_5_v, 635, "", "./tensorflow/lite/toco/model_cmdline_flags.cc", "ParseModelFlagsOrDie");

  // TODO(aselle): in the future allow Google version to use
  // flags, and only use this mechanism for open source
  auto* flags = UncheckedGlobalParsedModelFlags(false);
  std::string msg;
  bool model_success =
      toco::ParseModelFlagsFromCommandLineFlags(argc, argv, &msg, flags);
  if (!model_success || !msg.empty()) {
    // Log in non-standard way since this happens pre InitGoogle.
    fprintf(stderr, "%s", msg.c_str());
    fflush(stderr);
    abort();
  }
}

}  // namespace toco
