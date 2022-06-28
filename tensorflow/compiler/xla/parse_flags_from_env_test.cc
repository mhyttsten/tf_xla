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
class MHTracer_DTPStensorflowPScompilerPSxlaPSparse_flags_from_env_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSparse_flags_from_env_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSparse_flags_from_env_testDTcc() {
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

// Test for parse_flags_from_env.cc

#include "tensorflow/compiler/xla/parse_flags_from_env.h"

#include <stdio.h>
#include <stdlib.h>

#include <vector>

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace xla {

// Test that XLA flags can be set from the environment.
// Failure messages are accompanied by the text in msg[].
static void TestParseFlagsFromEnv(const char* msg) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("msg: \"" + (msg == nullptr ? std::string("nullptr") : std::string((char*)msg)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSparse_flags_from_env_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/xla/parse_flags_from_env_test.cc", "TestParseFlagsFromEnv");

  // Initialize module under test.
  int* pargc;
  std::vector<char*>* pargv;
  ResetFlagsFromEnvForTesting("TF_XLA_FLAGS", &pargc, &pargv);

  // Check that actual flags can be parsed.
  bool simple = false;
  std::string with_value;
  std::string embedded_quotes;
  std::string single_quoted;
  std::string double_quoted;
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("simple", &simple, ""),
      tensorflow::Flag("with_value", &with_value, ""),
      tensorflow::Flag("embedded_quotes", &embedded_quotes, ""),
      tensorflow::Flag("single_quoted", &single_quoted, ""),
      tensorflow::Flag("double_quoted", &double_quoted, ""),
  };
  bool parsed_ok = ParseFlagsFromEnvAndDieIfUnknown("TF_XLA_FLAGS", flag_list);
  CHECK_EQ(*pargc, 1) << msg;
  const std::vector<char*>& argv_second = *pargv;
  CHECK_NE(argv_second[0], nullptr) << msg;
  CHECK_EQ(argv_second[1], nullptr) << msg;
  CHECK(parsed_ok) << msg;
  CHECK(simple) << msg;
  CHECK_EQ(with_value, "a_value") << msg;
  CHECK_EQ(embedded_quotes, "single'double\"") << msg;
  CHECK_EQ(single_quoted, "single quoted \\\\ \n \"") << msg;
  CHECK_EQ(double_quoted, "double quoted \\ \n '\"") << msg;
}

// The flags settings to test.
static const char kTestFlagString[] =
    "--simple "
    "--with_value=a_value "
    "--embedded_quotes=single'double\" "
    "--single_quoted='single quoted \\\\ \n \"' "
    "--double_quoted=\"double quoted \\\\ \n '\\\"\" ";

// Test that the environment variable is parsed correctly.
TEST(ParseFlagsFromEnv, Basic) {
  // Prepare environment.
  tensorflow::setenv("TF_XLA_FLAGS", kTestFlagString, true /*overwrite*/);
  TestParseFlagsFromEnv("(flags in environment variable)");
}

// Test that a file named by the environment variable is parsed correctly.
TEST(ParseFlagsFromEnv, File) {
  // environment variables where  tmp dir may be specified.
  static const char* kTempVars[] = {"TEST_TMPDIR", "TMP"};
  static const char kTempDir[] = "/tmp";  // default temp dir if all else fails.
  const char* tmp_dir = nullptr;
  for (int i = 0; i != TF_ARRAYSIZE(kTempVars) && tmp_dir == nullptr; i++) {
    tmp_dir = getenv(kTempVars[i]);
  }
  if (tmp_dir == nullptr) {
    tmp_dir = kTempDir;
  }
  std::string tmp_file =
      absl::StrFormat("%s/parse_flags_from_env.%d", tmp_dir, getpid());
  FILE* fp = fopen(tmp_file.c_str(), "w");
  CHECK_NE(fp, nullptr) << "can't write to " << tmp_file;
  for (int i = 0; kTestFlagString[i] != '\0'; i++) {
    putc(kTestFlagString[i], fp);
  }
  fflush(fp);
  CHECK_EQ(ferror(fp), 0) << "writes failed to " << tmp_file;
  fclose(fp);
  // Prepare environment.
  tensorflow::setenv("TF_XLA_FLAGS", tmp_file.c_str(), true /*overwrite*/);
  TestParseFlagsFromEnv("(flags in file)");
  unlink(tmp_file.c_str());
}

// Name of the test binary.
static const char* binary_name;

// Test that when we use both the environment variable and actual
// commend line flags (when the latter is possible), the latter win.
TEST(ParseFlagsFromEnv, EnvAndFlag) {
  static struct {
    const char* env;
    const char* arg;
    const char* expected_value;
  } test[] = {
      {nullptr, nullptr, "1\n"},
      {nullptr, "--int_flag=2", "2\n"},
      {"--int_flag=3", nullptr, "3\n"},
      {"--int_flag=3", "--int_flag=2", "2\n"},  // flag beats environment
  };
  for (int i = 0; i != TF_ARRAYSIZE(test); i++) {
    if (test[i].env == nullptr) {
      // Might be set from previous tests.
      tensorflow::unsetenv("TF_XLA_FLAGS");
    } else {
      tensorflow::setenv("TF_XLA_FLAGS", test[i].env, /*overwrite=*/true);
    }
    tensorflow::SubProcess child;
    std::vector<std::string> argv;
    argv.push_back(binary_name);
    argv.push_back("--recursing");
    if (test[i].arg != nullptr) {
      argv.push_back(test[i].arg);
    }
    child.SetProgram(binary_name, argv);
    child.SetChannelAction(tensorflow::CHAN_STDOUT, tensorflow::ACTION_PIPE);
    child.SetChannelAction(tensorflow::CHAN_STDERR, tensorflow::ACTION_PIPE);
    CHECK(child.Start()) << "test " << i;
    std::string stdout_str;
    std::string stderr_str;
    int child_status = child.Communicate(nullptr, &stdout_str, &stderr_str);
    CHECK_EQ(child_status, 0) << "test " << i << "\nstdout\n"
                              << stdout_str << "\nstderr\n"
                              << stderr_str;
    // On windows, we get CR characters. Remove them.
    stdout_str.erase(std::remove(stdout_str.begin(), stdout_str.end(), '\r'),
                     stdout_str.end());
    CHECK_EQ(stdout_str, test[i].expected_value) << "test " << i;
  }
}

}  // namespace xla

int main(int argc, char* argv[]) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSparse_flags_from_env_testDTcc mht_1(mht_1_v, 334, "", "./tensorflow/compiler/xla/parse_flags_from_env_test.cc", "main");

  // Save name of binary so that it may invoke itself.
  xla::binary_name = argv[0];
  bool recursing = false;
  int32_t int_flag = 1;
  const std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("recursing", &recursing,
                       "Whether the binary is being invoked recursively."),
      tensorflow::Flag("int_flag", &int_flag, "An integer flag to test with"),
  };
  std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parse_ok =
      xla::ParseFlagsFromEnvAndDieIfUnknown("TF_XLA_FLAGS", flag_list);
  if (!parse_ok) {
    LOG(QFATAL) << "can't parse from environment\n" << usage;
  }
  parse_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok) {
    LOG(QFATAL) << usage;
  }
  if (recursing) {
    printf("%d\n", int_flag);
    exit(0);
  }
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
