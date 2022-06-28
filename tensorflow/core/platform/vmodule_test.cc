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
class MHTracer_DTPStensorflowPScorePSplatformPSvmodule_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSvmodule_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSvmodule_testDTcc() {
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

// Test that popens a child process with the VLOG-ing environment variable set
// for the logging framework, and observes VLOG_IS_ON and VLOG macro output.

#include <stdio.h>
#include <string.h>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/test.h"

// Make sure popen and pclose ara available on windows.
#ifdef PLATFORM_WINDOWS
#define popen _popen
#define pclose _pclose
#endif

namespace tensorflow {
namespace {

int RealMain(const char* argv0, bool do_vlog) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("argv0: \"" + (argv0 == nullptr ? std::string("nullptr") : std::string((char*)argv0)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSvmodule_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/platform/vmodule_test.cc", "RealMain");

  if (do_vlog) {
#if !defined(PLATFORM_GOOGLE)
    // Note, we only test this when !defined(PLATFORM_GOOGLE) because
    // VmoduleActivated doesn't exist in that implementation.
    //
    // Also, we call this internal API to simulate what would happen if
    // differently-named translation units attempted to VLOG, so we don't need
    // to create dummy translation unit files.
    bool ok = internal::LogMessage::VmoduleActivated("vmodule_test.cc", 7) &&
              internal::LogMessage::VmoduleActivated("shoobadooba.h", 3);
    if (!ok) {
      fprintf(stderr, "vmodule activated levels not as expected.\n");
      return EXIT_FAILURE;
    }
#endif

    // Print info on which VLOG levels are activated.
    fprintf(stderr, "VLOG_IS_ON(8)? %d\n", VLOG_IS_ON(8));
    fprintf(stderr, "VLOG_IS_ON(7)? %d\n", VLOG_IS_ON(7));
    fprintf(stderr, "VLOG_IS_ON(6)? %d\n", VLOG_IS_ON(6));
    // Do some VLOG-ing.
    VLOG(8) << "VLOG(8)";
    VLOG(7) << "VLOG(7)";
    VLOG(6) << "VLOG(6)";
    LOG(INFO) << "INFO";
    return EXIT_SUCCESS;
  }

  // Popen the child process.
  std::string command = std::string(argv0);
#if defined(PLATFORM_GOOGLE)
  command = command + " do_vlog --vmodule=vmodule_test=7 --alsologtostderr";
#elif defined(PLATFORM_WINDOWS)
  command = "set TF_CPP_VMODULE=vmodule_test=7,shoobadooba=3 && " + command +
            " do_vlog";
#else
  command =
      "TF_CPP_VMODULE=vmodule_test=7,shoobadooba=3 " + command + " do_vlog";
#endif
  command += " 2>&1";
  fprintf(stderr, "Running: \"%s\"\n", command.c_str());
  FILE* f = popen(command.c_str(), "r");
  if (f == nullptr) {
    fprintf(stderr, "Failed to popen child: %s\n", strerror(errno));
    return EXIT_FAILURE;
  }

  // Read data from the child's stdout.
  constexpr int kBufferSizeBytes = 4096;
  char buffer[kBufferSizeBytes];
  size_t result = fread(buffer, sizeof(buffer[0]), kBufferSizeBytes - 1, f);
  if (result == 0) {
    fprintf(stderr, "Failed to read from child stdout: %zu %s\n", result,
            strerror(errno));
    return EXIT_FAILURE;
  }
  buffer[result] = '\0';
  int status = pclose(f);
  if (status == -1) {
    fprintf(stderr, "Failed to close popen child: %s\n", strerror(errno));
    return EXIT_FAILURE;
  }

  // Check output is as expected.
  const char kExpected[] =
      "VLOG_IS_ON(8)? 0\nVLOG_IS_ON(7)? 1\nVLOG_IS_ON(6)? 1\n";
  if (strstr(buffer, kExpected) == nullptr) {
    fprintf(stderr, "error: unexpected output from child: \"%.*s\"\n",
            kBufferSizeBytes, buffer);
    return EXIT_FAILURE;
  }
  bool ok = strstr(buffer, "VLOG(7)\n") != nullptr &&
            strstr(buffer, "VLOG(6)\n") != nullptr &&
            strstr(buffer, "VLOG(8)\n") == nullptr;
  if (!ok) {
    fprintf(stderr, "error: VLOG output not as expected: \"%.*s\"\n",
            kBufferSizeBytes, buffer);
    return EXIT_FAILURE;
  }

  // Success!
  return EXIT_SUCCESS;
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char** argv) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSvmodule_testDTcc mht_1(mht_1_v, 296, "", "./tensorflow/core/platform/vmodule_test.cc", "main");

  testing::InitGoogleTest(&argc, argv);
  bool do_vlog = argc >= 2 && strcmp(argv[1], "do_vlog") == 0;
  return tensorflow::RealMain(argv[0], do_vlog);
}
