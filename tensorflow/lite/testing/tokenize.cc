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
class MHTracer_DTPStensorflowPSlitePStestingPStokenizeDTcc {
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
   MHTracer_DTPStensorflowPSlitePStestingPStokenizeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStestingPStokenizeDTcc() {
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
#include "tensorflow/lite/testing/tokenize.h"

#include <istream>
#include <string>

#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace testing {

void Tokenize(std::istream* input, TokenProcessor* processor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStestingPStokenizeDTcc mht_0(mht_0_v, 194, "", "./tensorflow/lite/testing/tokenize.cc", "Tokenize");

  enum State { kBuildQuotedToken, kBuildToken, kIdle };

  std::string current_token;
  State state = kIdle;
  auto start_token = [&](char c) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("c: '" + std::string(1, c) + "'");
   MHTracer_DTPStensorflowPSlitePStestingPStokenizeDTcc mht_1(mht_1_v, 203, "", "./tensorflow/lite/testing/tokenize.cc", "lambda");

    state = kBuildToken;
    current_token.clear();
    current_token = c;
  };
  auto issue_token = [&]() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStestingPStokenizeDTcc mht_2(mht_2_v, 211, "", "./tensorflow/lite/testing/tokenize.cc", "lambda");

    state = kIdle;
    processor->ConsumeToken(&current_token);
    current_token.clear();
  };
  auto start_quoted_token = [&]() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStestingPStokenizeDTcc mht_3(mht_3_v, 219, "", "./tensorflow/lite/testing/tokenize.cc", "lambda");

    state = kBuildQuotedToken;
    current_token.clear();
  };
  auto issue_quoted_token = [&]() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStestingPStokenizeDTcc mht_4(mht_4_v, 226, "", "./tensorflow/lite/testing/tokenize.cc", "lambda");

    state = kIdle;
    processor->ConsumeToken(&current_token);
    current_token.clear();
  };
  auto issue_delim = [&](char d) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("d: '" + std::string(1, d) + "'");
   MHTracer_DTPStensorflowPSlitePStestingPStokenizeDTcc mht_5(mht_5_v, 235, "", "./tensorflow/lite/testing/tokenize.cc", "lambda");

    current_token = string(1, d);
    processor->ConsumeToken(&current_token);
    current_token.clear();
  };
  auto is_delim = [](char c) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("c: '" + std::string(1, c) + "'");
   MHTracer_DTPStensorflowPSlitePStestingPStokenizeDTcc mht_6(mht_6_v, 244, "", "./tensorflow/lite/testing/tokenize.cc", "lambda");
 return c == '{' || c == '}' || c == ':'; };
  auto is_quote = [](char c) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("c: '" + std::string(1, c) + "'");
   MHTracer_DTPStensorflowPSlitePStestingPStokenizeDTcc mht_7(mht_7_v, 249, "", "./tensorflow/lite/testing/tokenize.cc", "lambda");
 return c == '"'; };

  for (auto it = std::istreambuf_iterator<char>(*input);
       it != std::istreambuf_iterator<char>(); ++it) {
    switch (state) {
      case kIdle:
        if (is_delim(*it)) {
          issue_delim(*it);
        } else if (is_quote(*it)) {
          start_quoted_token();
        } else if (!isspace(*it)) {
          start_token(*it);
        }
        break;
      case kBuildToken:
        if (is_delim(*it)) {
          issue_token();
          issue_delim(*it);
        } else if (is_quote(*it)) {
          issue_token();
          start_quoted_token();
        } else if (isspace(*it)) {
          issue_token();
        } else {
          current_token += *it;
        }
        break;
      case kBuildQuotedToken:
        if (is_quote(*it)) {
          issue_quoted_token();
        } else {
          current_token += *it;
        }
        break;
    }
  }
  if (state != kIdle) {
    issue_token();
  }
}

}  // namespace testing
}  // namespace tflite
