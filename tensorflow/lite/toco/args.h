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
// This abstracts command line arguments in toco.
// Arg<T> is a parseable type that can register a default value, be able to
// parse itself, and keep track of whether it was specified.
#ifndef TENSORFLOW_LITE_TOCO_ARGS_H_
#define TENSORFLOW_LITE_TOCO_ARGS_H_
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
class MHTracer_DTPStensorflowPSlitePStocoPSargsDTh {
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
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSargsDTh() {
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
#include <unordered_map>
#include <vector>
#include "tensorflow/lite/toco/toco_port.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "tensorflow/lite/toco/toco_types.h"

namespace toco {

// Since std::vector<int32> is in the std namespace, and we are not allowed
// to add ParseFlag/UnparseFlag to std, we introduce a simple wrapper type
// to use as the flag type:
struct IntList {
  std::vector<int32> elements;
};
struct StringMapList {
  std::vector<std::unordered_map<std::string, std::string>> elements;
};

// command_line_flags.h don't track whether or not a flag is specified. Arg
// contains the value (which will be default if not specified) and also
// whether the flag is specified.
// TODO(aselle): consider putting doc string and ability to construct the
// tensorflow argument into this, so declaration of parameters can be less
// distributed.
// Every template specialization of Arg is required to implement
// default_value(), specified(), value(), parse(), bind().
template <class T>
class Arg final {
 public:
  explicit Arg(T default_ = T()) : value_(default_) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTh mht_0(mht_0_v, 221, "", "./tensorflow/lite/toco/args.h", "Arg");
}
  virtual ~Arg() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTh mht_1(mht_1_v, 225, "", "./tensorflow/lite/toco/args.h", "~Arg");
}

  // Provide default_value() to arg list
  T default_value() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTh mht_2(mht_2_v, 231, "", "./tensorflow/lite/toco/args.h", "default_value");
 return value_; }
  // Return true if the command line argument was specified on the command line.
  bool specified() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTh mht_3(mht_3_v, 236, "", "./tensorflow/lite/toco/args.h", "specified");
 return specified_; }
  // Const reference to parsed value.
  const T& value() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTh mht_4(mht_4_v, 241, "", "./tensorflow/lite/toco/args.h", "value");
 return value_; }

  // Parsing callback for the tensorflow::Flags code
  bool Parse(T value_in) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTh mht_5(mht_5_v, 247, "", "./tensorflow/lite/toco/args.h", "Parse");

    value_ = value_in;
    specified_ = true;
    return true;
  }

  // Bind the parse member function so tensorflow::Flags can call it.
  std::function<bool(T)> bind() {
    return std::bind(&Arg::Parse, this, std::placeholders::_1);
  }

 private:
  // Becomes true after parsing if the value was specified
  bool specified_ = false;
  // Value of the argument (initialized to the default in the constructor).
  T value_;
};

template <>
class Arg<toco::IntList> final {
 public:
  // Provide default_value() to arg list
  std::string default_value() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTh mht_6(mht_6_v, 272, "", "./tensorflow/lite/toco/args.h", "default_value");
 return ""; }
  // Return true if the command line argument was specified on the command line.
  bool specified() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTh mht_7(mht_7_v, 277, "", "./tensorflow/lite/toco/args.h", "specified");
 return specified_; }
  // Bind the parse member function so tensorflow::Flags can call it.
  bool Parse(std::string text);

  std::function<bool(std::string)> bind() {
    return std::bind(&Arg::Parse, this, std::placeholders::_1);
  }

  const toco::IntList& value() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTh mht_8(mht_8_v, 288, "", "./tensorflow/lite/toco/args.h", "value");
 return parsed_value_; }

 private:
  toco::IntList parsed_value_;
  bool specified_ = false;
};

template <>
class Arg<toco::StringMapList> final {
 public:
  // Provide default_value() to StringMapList
  std::string default_value() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTh mht_9(mht_9_v, 302, "", "./tensorflow/lite/toco/args.h", "default_value");
 return ""; }
  // Return true if the command line argument was specified on the command line.
  bool specified() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTh mht_10(mht_10_v, 307, "", "./tensorflow/lite/toco/args.h", "specified");
 return specified_; }
  // Bind the parse member function so tensorflow::Flags can call it.

  bool Parse(std::string text);

  std::function<bool(std::string)> bind() {
    return std::bind(&Arg::Parse, this, std::placeholders::_1);
  }

  const toco::StringMapList& value() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStocoPSargsDTh mht_11(mht_11_v, 319, "", "./tensorflow/lite/toco/args.h", "value");
 return parsed_value_; }

 private:
  toco::StringMapList parsed_value_;
  bool specified_ = false;
};

// Flags that describe a model. See model_cmdline_flags.cc for details.
struct ParsedModelFlags {
  Arg<std::string> input_array;
  Arg<std::string> input_arrays;
  Arg<std::string> output_array;
  Arg<std::string> output_arrays;
  Arg<std::string> input_shapes;
  Arg<int> batch_size = Arg<int>(1);
  Arg<float> mean_value = Arg<float>(0.f);
  Arg<std::string> mean_values;
  Arg<float> std_value = Arg<float>(1.f);
  Arg<std::string> std_values;
  Arg<std::string> input_data_type;
  Arg<std::string> input_data_types;
  Arg<bool> variable_batch = Arg<bool>(false);
  Arg<toco::IntList> input_shape;
  Arg<toco::StringMapList> rnn_states;
  Arg<toco::StringMapList> model_checks;
  Arg<bool> change_concat_input_ranges = Arg<bool>(true);
  // Debugging output options.
  // TODO(benoitjacob): these shouldn't be ModelFlags.
  Arg<std::string> graphviz_first_array;
  Arg<std::string> graphviz_last_array;
  Arg<std::string> dump_graphviz;
  Arg<bool> dump_graphviz_video = Arg<bool>(false);
  Arg<std::string> conversion_summary_dir;
  Arg<bool> allow_nonexistent_arrays = Arg<bool>(false);
  Arg<bool> allow_nonascii_arrays = Arg<bool>(false);
  Arg<std::string> arrays_extra_info_file;
  Arg<std::string> model_flags_file;
};

// Flags that describe the operation you would like to do (what conversion
// you want). See toco_cmdline_flags.cc for details.
struct ParsedTocoFlags {
  Arg<std::string> input_file;
  Arg<std::string> savedmodel_directory;
  Arg<std::string> output_file;
  Arg<std::string> input_format = Arg<std::string>("TENSORFLOW_GRAPHDEF");
  Arg<std::string> output_format = Arg<std::string>("TFLITE");
  Arg<std::string> savedmodel_tagset;
  // TODO(aselle): command_line_flags  doesn't support doubles
  Arg<float> default_ranges_min = Arg<float>(0.);
  Arg<float> default_ranges_max = Arg<float>(0.);
  Arg<float> default_int16_ranges_min = Arg<float>(0.);
  Arg<float> default_int16_ranges_max = Arg<float>(0.);
  Arg<std::string> inference_type;
  Arg<std::string> inference_input_type;
  Arg<bool> drop_fake_quant = Arg<bool>(false);
  Arg<bool> reorder_across_fake_quant = Arg<bool>(false);
  Arg<bool> allow_custom_ops = Arg<bool>(false);
  Arg<bool> allow_dynamic_tensors = Arg<bool>(true);
  Arg<std::string> custom_opdefs;
  Arg<bool> post_training_quantize = Arg<bool>(false);
  Arg<bool> quantize_to_float16 = Arg<bool>(false);
  // Deprecated flags
  Arg<bool> quantize_weights = Arg<bool>(false);
  Arg<std::string> input_type;
  Arg<std::string> input_types;
  Arg<bool> debug_disable_recurrent_cell_fusion = Arg<bool>(false);
  Arg<bool> drop_control_dependency = Arg<bool>(false);
  Arg<bool> propagate_fake_quant_num_bits = Arg<bool>(false);
  Arg<bool> allow_nudging_weights_to_use_fast_gemm_kernel = Arg<bool>(false);
  Arg<int64_t> dedupe_array_min_size_bytes = Arg<int64_t>(64);
  Arg<bool> split_tflite_lstm_inputs = Arg<bool>(true);
  // WARNING: Experimental interface, subject to change
  Arg<bool> enable_select_tf_ops = Arg<bool>(false);
  // WARNING: Experimental interface, subject to change
  Arg<bool> force_select_tf_ops = Arg<bool>(false);
  // WARNING: Experimental interface, subject to change
  Arg<bool> unfold_batchmatmul = Arg<bool>(true);
  // WARNING: Experimental interface, subject to change
  Arg<std::string> accumulation_type;
  // WARNING: Experimental interface, subject to change
  Arg<bool> allow_bfloat16;
};

}  // namespace toco
#endif  // TENSORFLOW_LITE_TOCO_ARGS_H_
