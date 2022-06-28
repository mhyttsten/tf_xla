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
class MHTracer_DTPStensorflowPSlitePStestingPSkernel_testPSdiff_analyzerDTcc {
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
   MHTracer_DTPStensorflowPSlitePStestingPSkernel_testPSdiff_analyzerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStestingPSkernel_testPSdiff_analyzerDTcc() {
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
#include "tensorflow/lite/testing/kernel_test/diff_analyzer.h"

#include <cmath>
#include <fstream>
#include <string>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/testing/split.h"

namespace tflite {
namespace testing {

namespace {
float CalculateNormalizedMaxDiff(const std::vector<float>& base,
                                 const std::vector<float>& test) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStestingPSkernel_testPSdiff_analyzerDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/testing/kernel_test/diff_analyzer.cc", "CalculateNormalizedMaxDiff");

  float diff = 0;
  // For numerical stability in case the tensor is all 0.
  float base_max = 1e-6;

  for (int i = 0; i < base.size(); i++) {
    diff = std::max(diff, std::abs(base[i] - test[i]));
    base_max = std::max(base_max, base[i]);
  }

  return diff / base_max;
}

float CalculateNormalizedL2Norm(const std::vector<float>& base,
                                const std::vector<float>& test) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStestingPSkernel_testPSdiff_analyzerDTcc mht_1(mht_1_v, 215, "", "./tensorflow/lite/testing/kernel_test/diff_analyzer.cc", "CalculateNormalizedL2Norm");

  float l2_error = 0;
  // For numerical stability in case the tensor is all 0.
  float base_max = 1e-6;

  for (int i = 0; i < base.size(); i++) {
    float diff = base[i] - test[i];
    l2_error += diff * diff;
    base_max = std::max(base_max, base[i]);
  }

  l2_error /= base.size();

  return std::sqrt(l2_error) / base_max;
}

TfLiteStatus Populate(const string& filename,
                      std::unordered_map<string, std::vector<float>>* tensors) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPSkernel_testPSdiff_analyzerDTcc mht_2(mht_2_v, 236, "", "./tensorflow/lite/testing/kernel_test/diff_analyzer.cc", "Populate");

  if (filename.empty()) {
    fprintf(stderr, "Empty input file name.");
    return kTfLiteError;
  }

  std::ifstream file(filename);
  string content;
  while (std::getline(file, content, '\n')) {
    auto parts = Split<string>(content, ":");
    if (parts.size() != 2) {
      fprintf(stderr, "Expected <name>:<value>, got %s", content.c_str());
      return kTfLiteError;
    }
    tensors->insert(std::make_pair(parts[0], Split<float>(parts[1], ",")));
  }

  file.close();
  return kTfLiteOk;
}
}  // namespace

TfLiteStatus DiffAnalyzer::ReadFiles(const string& base, const string& test) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("base: \"" + base + "\"");
   mht_3_v.push_back("test: \"" + test + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPSkernel_testPSdiff_analyzerDTcc mht_3(mht_3_v, 263, "", "./tensorflow/lite/testing/kernel_test/diff_analyzer.cc", "DiffAnalyzer::ReadFiles");

  TF_LITE_ENSURE_STATUS(Populate(base, &base_tensors_));
  TF_LITE_ENSURE_STATUS(Populate(test, &test_tensors_));

  if (base_tensors_.size() != test_tensors_.size()) {
    fprintf(stderr, "Golden and test tensor dimensions don't match.");
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus DiffAnalyzer::WriteReport(const string& filename) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPSkernel_testPSdiff_analyzerDTcc mht_4(mht_4_v, 279, "", "./tensorflow/lite/testing/kernel_test/diff_analyzer.cc", "DiffAnalyzer::WriteReport");

  if (filename.empty()) {
    fprintf(stderr, "Empty output file name.");
    return kTfLiteError;
  }

  std::ofstream output_file;
  output_file.open(filename, std::fstream::out | std::fstream::trunc);
  if (!output_file) {
    fprintf(stderr, "Failed to open output file %s.", filename.c_str());
    return kTfLiteError;
  }

  output_file << "Normalized L2 Error"
              << ","
              << "Normalized Max Diff"
              << "\n";
  for (const auto& item : base_tensors_) {
    const auto& name = item.first;
    if (!test_tensors_.count(name)) {
      fprintf(stderr, "Missing tensor %s in test tensors.", name.c_str());
      continue;
    }
    float l2_error =
        CalculateNormalizedL2Norm(base_tensors_[name], test_tensors_[name]);
    float max_diff =
        CalculateNormalizedMaxDiff(base_tensors_[name], test_tensors_[name]);
    output_file << name << ":" << l2_error << "," << max_diff << "\n";
  }

  output_file.close();
  return kTfLiteOk;
}
}  // namespace testing
}  // namespace tflite
