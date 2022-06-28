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
class MHTracer_DTPStensorflowPSlitePStestingPSkernel_testPSinput_generatorDTcc {
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
   MHTracer_DTPStensorflowPSlitePStestingPSkernel_testPSinput_generatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStestingPSkernel_testPSinput_generatorDTcc() {
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
#include "tensorflow/lite/testing/kernel_test/input_generator.h"

#include <cstdio>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/testing/join.h"
#include "tensorflow/lite/testing/split.h"

namespace tflite {
namespace testing {

namespace {
static constexpr char kDefaultServingSignatureDefKey[] = "serving_default";

template <typename T>
std::vector<T> GenerateRandomTensor(TfLiteIntArray* dims,
                                    const std::function<T(int)>& random_func) {
  int64_t num_elements = 1;
  for (int i = 0; i < dims->size; i++) {
    num_elements *= dims->data[i];
  }

  std::vector<T> result(num_elements);
  for (int i = 0; i < num_elements; i++) {
    result[i] = random_func(i);
  }
  return result;
}

template <typename T>
std::vector<T> GenerateUniform(TfLiteIntArray* dims, float min, float max) {
  auto random_float = [](float min, float max) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStestingPSkernel_testPSinput_generatorDTcc mht_0(mht_0_v, 223, "", "./tensorflow/lite/testing/kernel_test/input_generator.cc", "lambda");

    // TODO(yunluli): Change seed for each invocation if needed.
    // Used rand() instead of rand_r() here to make it runnable on android.
    return min + (max - min) * static_cast<float>(rand()) / RAND_MAX;
  };

  std::function<T(int)> random_t = [&](int) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStestingPSkernel_testPSinput_generatorDTcc mht_1(mht_1_v, 232, "", "./tensorflow/lite/testing/kernel_test/input_generator.cc", "lambda");

    return static_cast<T>(random_float(min, max));
  };
  std::vector<T> data = GenerateRandomTensor(dims, random_t);
  return data;
}

template <typename T>
std::vector<T> GenerateGaussian(TfLiteIntArray* dims, float min, float max) {
  auto random_float = [](float min, float max) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStestingPSkernel_testPSinput_generatorDTcc mht_2(mht_2_v, 244, "", "./tensorflow/lite/testing/kernel_test/input_generator.cc", "lambda");

    static std::default_random_engine generator;
    // We generate a float number within [0, 1) following a mormal distribution
    // with mean = 0.5 and stddev = 1/3, and use it to scale the final random
    // number into the desired range.
    static std::normal_distribution<double> distribution(0.5, 1.0 / 3);
    auto rand_n = distribution(generator);
    while (rand_n < 0 || rand_n >= 1) {
      rand_n = distribution(generator);
    }

    return min + (max - min) * static_cast<float>(rand_n);
  };

  std::function<T(int)> random_t = [&](int) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStestingPSkernel_testPSinput_generatorDTcc mht_3(mht_3_v, 261, "", "./tensorflow/lite/testing/kernel_test/input_generator.cc", "lambda");

    return static_cast<T>(random_float(min, max));
  };
  std::vector<T> data = GenerateRandomTensor(dims, random_t);
  return data;
}

}  // namespace

TfLiteStatus InputGenerator::LoadModel(const string& model_dir) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("model_dir: \"" + model_dir + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPSkernel_testPSinput_generatorDTcc mht_4(mht_4_v, 274, "", "./tensorflow/lite/testing/kernel_test/input_generator.cc", "InputGenerator::LoadModel");

  return LoadModel(model_dir, kDefaultServingSignatureDefKey);
}

TfLiteStatus InputGenerator::LoadModel(const string& model_dir,
                                       const string& signature) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("model_dir: \"" + model_dir + "\"");
   mht_5_v.push_back("signature: \"" + signature + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPSkernel_testPSinput_generatorDTcc mht_5(mht_5_v, 284, "", "./tensorflow/lite/testing/kernel_test/input_generator.cc", "InputGenerator::LoadModel");

  model_ = FlatBufferModel::BuildFromFile(model_dir.c_str());
  if (!model_) {
    fprintf(stderr, "Cannot load model %s", model_dir.c_str());
    return kTfLiteError;
  }

  ::tflite::ops::builtin::BuiltinOpResolver builtin_ops;
  InterpreterBuilder(*model_, builtin_ops)(&interpreter_);
  if (!interpreter_) {
    fprintf(stderr, "Failed to build interpreter.");
    return kTfLiteError;
  }
  signature_runner_ = interpreter_->GetSignatureRunner(signature.c_str());
  if (!signature_runner_) {
    fprintf(stderr, "Failed to get SignatureRunner.\n");
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus InputGenerator::ReadInputsFromFile(const string& filename) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPSkernel_testPSinput_generatorDTcc mht_6(mht_6_v, 310, "", "./tensorflow/lite/testing/kernel_test/input_generator.cc", "InputGenerator::ReadInputsFromFile");

  if (filename.empty()) {
    fprintf(stderr, "Empty input file name.");
    return kTfLiteError;
  }

  std::ifstream input_file(filename);
  string input;
  while (std::getline(input_file, input, '\n')) {
    std::vector<string> parts = Split<string>(input, ":");
    if (parts.size() != 2) {
      fprintf(stderr, "Expected <name>:<value>, got %s", input.c_str());
      return kTfLiteError;
    }
    inputs_.push_back(std::make_pair(parts[0], parts[1]));
  }
  input_file.close();
  return kTfLiteOk;
}

TfLiteStatus InputGenerator::WriteInputsToFile(const string& filename) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPSkernel_testPSinput_generatorDTcc mht_7(mht_7_v, 334, "", "./tensorflow/lite/testing/kernel_test/input_generator.cc", "InputGenerator::WriteInputsToFile");

  if (filename.empty()) {
    fprintf(stderr, "Empty input file name.");
    return kTfLiteError;
  }

  std::ofstream output_file;
  output_file.open(filename, std::fstream::out | std::fstream::trunc);
  if (!output_file) {
    fprintf(stderr, "Failed to open output file %s.", filename.c_str());
    return kTfLiteError;
  }

  for (const auto& input : inputs_) {
    output_file << input.first << ":" << input.second << "\n";
  }
  output_file.close();

  return kTfLiteOk;
}

// TODO(yunluli): Support more tensor types when needed.
TfLiteStatus InputGenerator::GenerateInput(const string& distribution) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("distribution: \"" + distribution + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPSkernel_testPSinput_generatorDTcc mht_8(mht_8_v, 360, "", "./tensorflow/lite/testing/kernel_test/input_generator.cc", "InputGenerator::GenerateInput");

  auto input_tensor_names = signature_runner_->input_names();
  for (const char* name : input_tensor_names) {
    auto* tensor = signature_runner_->input_tensor(name);
    if (distribution == "UNIFORM") {
      switch (tensor->type) {
        case kTfLiteInt8: {
          auto data = GenerateUniform<int8_t>(
              tensor->dims, std::numeric_limits<int8_t>::min(),
              std::numeric_limits<int8_t>::max());
          inputs_.push_back(
              std::make_pair(name, Join(data.data(), data.size(), ",")));
          break;
        }
        case kTfLiteUInt8: {
          auto data = GenerateUniform<uint8_t>(
              tensor->dims, std::numeric_limits<uint8_t>::min(),
              std::numeric_limits<uint8_t>::max());
          inputs_.push_back(
              std::make_pair(name, Join(data.data(), data.size(), ",")));
          break;
        }
        case kTfLiteFloat32: {
          auto data = GenerateUniform<float>(tensor->dims, -1, 1);
          inputs_.push_back(
              std::make_pair(name, Join(data.data(), data.size(), ",")));
          break;
        }
        default:
          fprintf(stderr, "Unsupported input tensor type %s.",
                  TfLiteTypeGetName(tensor->type));
          break;
      }
    } else if (distribution == "GAUSSIAN") {
      switch (tensor->type) {
        case kTfLiteInt8: {
          auto data = GenerateGaussian<int8_t>(
              tensor->dims, std::numeric_limits<int8_t>::min(),
              std::numeric_limits<int8_t>::max());
          inputs_.push_back(
              std::make_pair(name, Join(data.data(), data.size(), ",")));
          break;
        }
        case kTfLiteUInt8: {
          auto data = GenerateGaussian<uint8_t>(
              tensor->dims, std::numeric_limits<uint8_t>::min(),
              std::numeric_limits<uint8_t>::max());
          inputs_.push_back(
              std::make_pair(name, Join(data.data(), data.size(), ",")));
          break;
        }
        case kTfLiteFloat32: {
          auto data = GenerateGaussian<float>(tensor->dims, -1, 1);
          inputs_.push_back(
              std::make_pair(name, Join(data.data(), data.size(), ",")));
          break;
        }
        default:
          fprintf(stderr, "Unsupported input tensor type %s.",
                  TfLiteTypeGetName(tensor->type));
          break;
      }
    } else {
      fprintf(stderr, "Unsupported distribution %s.", distribution.c_str());
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

}  // namespace testing
}  // namespace tflite
