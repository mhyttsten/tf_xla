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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSmanifest_checking_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSmanifest_checking_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSmanifest_checking_testDTcc() {
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

#include "tensorflow/compiler/xla/tests/manifest_checking_test.h"

#include <fstream>
#include <iterator>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/regexp.h"

namespace xla {

namespace {

// Mapping from test name; i.e. MyTest.MyTestCase to platforms on which it is
// disabled - a sequence of regexps.
using ManifestT = absl::flat_hash_map<std::string, std::vector<std::string>>;

ManifestT ReadManifest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSmanifest_checking_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/xla/tests/manifest_checking_test.cc", "ReadManifest");

  ManifestT manifest;

  absl::string_view path = absl::NullSafeStringView(*DisabledManifestPath());
  if (path.empty()) {
    return manifest;
  }

  // Note: parens are required to disambiguate vs function decl.
  std::ifstream file_stream((std::string(path)));
  std::string contents((std::istreambuf_iterator<char>(file_stream)),
                       std::istreambuf_iterator<char>());

  std::vector<std::string> lines = absl::StrSplit(contents, '\n');
  for (std::string& line : lines) {
    auto comment = line.find("//");
    if (comment != std::string::npos) {
      line = line.substr(0, comment);
    }
    if (line.empty()) {
      continue;
    }
    absl::StripTrailingAsciiWhitespace(&line);
    std::vector<std::string> pieces = absl::StrSplit(line, ' ');
    CHECK_GE(pieces.size(), 1);
    auto& platforms = manifest[pieces[0]];
    for (size_t i = 1; i < pieces.size(); ++i) {
      platforms.push_back(pieces[i]);
    }
  }
  return manifest;
}

}  // namespace

void ManifestCheckingTest::SetUp() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSmanifest_checking_testDTcc mht_1(mht_1_v, 244, "", "./tensorflow/compiler/xla/tests/manifest_checking_test.cc", "ManifestCheckingTest::SetUp");

  const testing::TestInfo* test_info =
      testing::UnitTest::GetInstance()->current_test_info();
  absl::string_view test_case_name = test_info->test_suite_name();
  absl::string_view test_name = test_info->name();
  VLOG(1) << "test_case_name: " << test_case_name;
  VLOG(1) << "test_name: " << test_name;

  // Remove the type suffix from the test case name.
  if (const char* type_param = test_info->type_param()) {
    VLOG(1) << "type_param: " << type_param;
    size_t last_slash = test_case_name.rfind('/');
    test_case_name = test_case_name.substr(0, last_slash);
    VLOG(1) << "test_case_name: " << test_case_name;
  }

  // Remove the test instantiation name if it is present.
  auto first_slash = test_case_name.find('/');
  if (first_slash != test_case_name.npos) {
    test_case_name.remove_prefix(first_slash + 1);
    VLOG(1) << "test_case_name: " << test_case_name;
  }

  ManifestT manifest = ReadManifest();

  // If the test name ends with a slash followed by one or more characters,
  // strip that off.
  auto last_slash = test_name.rfind('/');
  if (last_slash != test_name.npos) {
    test_name = test_name.substr(0, last_slash);
    VLOG(1) << "test_name: " << test_name;
  }

  // First try full match: test_case_name.test_name
  // If that fails, try to find just the test_case_name; this would disable all
  // tests in the test case.
  auto it = manifest.find(absl::StrCat(test_case_name, ".", test_name));
  if (it == manifest.end()) {
    it = manifest.find(test_case_name);
    if (it == manifest.end()) {
      return;
    }
  }

  // Expect a full match vs. one of the platform regexps to disable the test.
  const std::vector<std::string>& disabled_platforms = it->second;
  auto platform_string = *TestPlatform();
  for (const auto& s : disabled_platforms) {
    if (RE2::FullMatch(/*text=*/platform_string, /*re=*/s)) {
      GTEST_SKIP();
      return;
    }
  }

  // We didn't hit in the disabled manifest entries, so don't disable it.
}

}  // namespace xla
