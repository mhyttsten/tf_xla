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
class MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSutilsDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSutilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSutilsDTcc() {
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

#include "tensorflow/lite/tools/evaluation/utils.h"

#include "tensorflow/lite/tools/delegates/delegate_provider.h"

#if !defined(_WIN32)
#include <dirent.h>
#endif
#include <sys/stat.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>

namespace tflite {
namespace evaluation {

std::string StripTrailingSlashes(const std::string& path) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSutilsDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/tools/evaluation/utils.cc", "StripTrailingSlashes");

  int end = path.size();
  while (end > 0 && path[end - 1] == '/') {
    end--;
  }
  return path.substr(0, end);
}

bool ReadFileLines(const std::string& file_path,
                   std::vector<std::string>* lines_output) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("file_path: \"" + file_path + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSutilsDTcc mht_1(mht_1_v, 216, "", "./tensorflow/lite/tools/evaluation/utils.cc", "ReadFileLines");

  if (!lines_output) {
    return false;
  }
  std::ifstream stream(file_path.c_str());
  if (!stream) {
    return false;
  }
  std::string line;
  while (std::getline(stream, line)) {
    lines_output->push_back(line);
  }
  return true;
}

#if !defined(_WIN32)
TfLiteStatus GetSortedFileNames(
    const std::string& directory, std::vector<std::string>* result,
    const std::unordered_set<std::string>& extensions) {
  DIR* dir;
  struct dirent* ent;
  if (result == nullptr) {
    return kTfLiteError;
  }
  result->clear();
  std::string dir_path = StripTrailingSlashes(directory);
  if ((dir = opendir(dir_path.c_str())) != nullptr) {
    while ((ent = readdir(dir)) != nullptr) {
      if (ent->d_type == DT_DIR) continue;
      std::string filename(std::string(ent->d_name));
      size_t lastdot = filename.find_last_of('.');
      std::string ext = lastdot != std::string::npos ? filename.substr(lastdot)
                                                     : std::string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (!extensions.empty() && extensions.find(ext) == extensions.end()) {
        continue;
      }
      result->emplace_back(dir_path + "/" + filename);
    }
    closedir(dir);
  } else {
    return kTfLiteError;
  }
  std::sort(result->begin(), result->end());
  return kTfLiteOk;
}
#endif

TfLiteDelegatePtr CreateNNAPIDelegate() {
#if defined(__ANDROID__)
  return TfLiteDelegatePtr(
      NnApiDelegate(),
      // NnApiDelegate() returns a singleton, so provide a no-op deleter.
      [](TfLiteDelegate*) {});
#else
  return tools::CreateNullDelegate();
#endif  // defined(__ANDROID__)
}

TfLiteDelegatePtr CreateNNAPIDelegate(StatefulNnApiDelegate::Options options) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSutilsDTcc mht_2(mht_2_v, 278, "", "./tensorflow/lite/tools/evaluation/utils.cc", "CreateNNAPIDelegate");

#if defined(__ANDROID__)
  return TfLiteDelegatePtr(
      new StatefulNnApiDelegate(options), [](TfLiteDelegate* delegate) {
        delete reinterpret_cast<StatefulNnApiDelegate*>(delegate);
      });
#else
  return tools::CreateNullDelegate();
#endif  // defined(__ANDROID__)
}

#if TFLITE_SUPPORTS_GPU_DELEGATE
TfLiteDelegatePtr CreateGPUDelegate(TfLiteGpuDelegateOptionsV2* options) {
  return TfLiteDelegatePtr(TfLiteGpuDelegateV2Create(options),
                           &TfLiteGpuDelegateV2Delete);
}
#endif  // TFLITE_SUPPORTS_GPU_DELEGATE

TfLiteDelegatePtr CreateGPUDelegate() {
#if TFLITE_SUPPORTS_GPU_DELEGATE
  TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
  options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
  options.inference_preference =
      TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;

  return CreateGPUDelegate(&options);
#else
  return tools::CreateNullDelegate();
#endif  // TFLITE_SUPPORTS_GPU_DELEGATE
}

TfLiteDelegatePtr CreateHexagonDelegate(
    const std::string& library_directory_path, bool profiling) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("library_directory_path: \"" + library_directory_path + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSutilsDTcc mht_3(mht_3_v, 314, "", "./tensorflow/lite/tools/evaluation/utils.cc", "CreateHexagonDelegate");

#if TFLITE_ENABLE_HEXAGON
  TfLiteHexagonDelegateOptions options = {0};
  options.print_graph_profile = profiling;
  return CreateHexagonDelegate(&options, library_directory_path);
#else
  return tools::CreateNullDelegate();
#endif  // TFLITE_ENABLE_HEXAGON
}

#if TFLITE_ENABLE_HEXAGON
TfLiteDelegatePtr CreateHexagonDelegate(
    const TfLiteHexagonDelegateOptions* options,
    const std::string& library_directory_path) {
  if (library_directory_path.empty()) {
    TfLiteHexagonInit();
  } else {
    TfLiteHexagonInitWithPath(library_directory_path.c_str());
  }

  TfLiteDelegate* delegate = TfLiteHexagonDelegateCreate(options);
  if (!delegate) {
    TfLiteHexagonTearDown();
    return tools::CreateNullDelegate();
  }
  return TfLiteDelegatePtr(delegate, [](TfLiteDelegate* delegate) {
    TfLiteHexagonDelegateDelete(delegate);
    TfLiteHexagonTearDown();
  });
}
#endif

#if defined(__s390x__) || defined(TFLITE_WITHOUT_XNNPACK)
TfLiteDelegatePtr CreateXNNPACKDelegate(int num_threads) {
  return tools::CreateNullDelegate();
}
#else
TfLiteDelegatePtr CreateXNNPACKDelegate() {
  TfLiteXNNPackDelegateOptions xnnpack_options =
      TfLiteXNNPackDelegateOptionsDefault();
  return CreateXNNPACKDelegate(&xnnpack_options);
}

TfLiteDelegatePtr CreateXNNPACKDelegate(
    const TfLiteXNNPackDelegateOptions* xnnpack_options) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSutilsDTcc mht_4(mht_4_v, 361, "", "./tensorflow/lite/tools/evaluation/utils.cc", "CreateXNNPACKDelegate");

  auto xnnpack_delegate = TfLiteXNNPackDelegateCreate(xnnpack_options);
  return TfLiteDelegatePtr(xnnpack_delegate, [](TfLiteDelegate* delegate) {
    TfLiteXNNPackDelegateDelete(delegate);
  });
}

TfLiteDelegatePtr CreateXNNPACKDelegate(int num_threads) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSutilsDTcc mht_5(mht_5_v, 371, "", "./tensorflow/lite/tools/evaluation/utils.cc", "CreateXNNPACKDelegate");

  auto opts = TfLiteXNNPackDelegateOptionsDefault();
  // Note that we don't want to use the thread pool for num_threads == 1.
  opts.num_threads = num_threads > 1 ? num_threads : 0;
  return CreateXNNPACKDelegate(&opts);
}
#endif
}  // namespace evaluation
}  // namespace tflite
