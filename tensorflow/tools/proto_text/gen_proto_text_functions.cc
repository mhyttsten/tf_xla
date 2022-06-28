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
class MHTracer_DTPStensorflowPStoolsPSproto_textPSgen_proto_text_functionsDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSproto_textPSgen_proto_text_functionsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSproto_textPSgen_proto_text_functionsDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <stdio.h>
#include <set>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/protobuf_compiler.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/tools/proto_text/gen_proto_text_functions_lib.h"

namespace tensorflow {

namespace {
class CrashOnErrorCollector
    : public tensorflow::protobuf::compiler::MultiFileErrorCollector {
 public:
  ~CrashOnErrorCollector() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSproto_textPSgen_proto_text_functionsDTcc mht_0(mht_0_v, 200, "", "./tensorflow/tools/proto_text/gen_proto_text_functions.cc", "~CrashOnErrorCollector");
}

  void AddError(const string& filename, int line, int column,
                const string& message) override {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("filename: \"" + filename + "\"");
   mht_1_v.push_back("message: \"" + message + "\"");
   MHTracer_DTPStensorflowPStoolsPSproto_textPSgen_proto_text_functionsDTcc mht_1(mht_1_v, 208, "", "./tensorflow/tools/proto_text/gen_proto_text_functions.cc", "AddError");

    LOG(FATAL) << "Unexpected error at " << filename << "@" << line << ":"
               << column << " - " << message;
  }
};

static const char kTensorFlowHeaderPrefix[] = "";

static const char kPlaceholderFile[] =
    "tensorflow/tools/proto_text/placeholder.txt";

bool IsPlaceholderFile(const char* s) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("s: \"" + (s == nullptr ? std::string("nullptr") : std::string((char*)s)) + "\"");
   MHTracer_DTPStensorflowPStoolsPSproto_textPSgen_proto_text_functionsDTcc mht_2(mht_2_v, 223, "", "./tensorflow/tools/proto_text/gen_proto_text_functions.cc", "IsPlaceholderFile");

  string ph(kPlaceholderFile);
  string str(s);
  return str.size() >= strlen(kPlaceholderFile) &&
         ph == str.substr(str.size() - ph.size());
}

}  // namespace

// Main program to take input protos and write output pb_text source files that
// contain generated proto text input and output functions.
//
// Main expects:
// - First argument is output path
// - Second argument is the relative path of the protos to the root. E.g.,
//   for protos built by a rule in tensorflow/core, this will be
//   tensorflow/core.
// - Then any number of source proto file names, plus one source name must be
//   placeholder.txt from this gen tool's package.  placeholder.txt is
//   ignored for proto resolution, but is used to determine the root at which
//   the build tool has placed the source proto files.
//
// Note that this code doesn't use tensorflow's command line parsing, because of
// circular dependencies between libraries if that were done.
//
// This is meant to be invoked by a genrule. See BUILD for more information.
int MainImpl(int argc, char** argv) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSproto_textPSgen_proto_text_functionsDTcc mht_3(mht_3_v, 252, "", "./tensorflow/tools/proto_text/gen_proto_text_functions.cc", "MainImpl");

  if (argc < 4) {
    LOG(ERROR) << "Pass output path, relative path, and at least proto file";
    return -1;
  }

  const string output_root = argv[1];
  const string output_relative_path = kTensorFlowHeaderPrefix + string(argv[2]);

  string src_relative_path;
  bool has_placeholder = false;
  for (int i = 3; i < argc; ++i) {
    if (IsPlaceholderFile(argv[i])) {
      const string s(argv[i]);
      src_relative_path = s.substr(0, s.size() - strlen(kPlaceholderFile));
      has_placeholder = true;
    }
  }
  if (!has_placeholder) {
    LOG(ERROR) << kPlaceholderFile << " must be passed";
    return -1;
  }

  tensorflow::protobuf::compiler::DiskSourceTree source_tree;

  source_tree.MapPath("", src_relative_path.empty() ? "." : src_relative_path);
  CrashOnErrorCollector crash_on_error;
  tensorflow::protobuf::compiler::Importer importer(&source_tree,
                                                    &crash_on_error);

  for (int i = 3; i < argc; i++) {
    if (IsPlaceholderFile(argv[i])) continue;
    const string proto_path = string(argv[i]).substr(src_relative_path.size());

    const tensorflow::protobuf::FileDescriptor* fd =
        importer.Import(proto_path);

    const int index = proto_path.find_last_of('.');
    string proto_path_no_suffix = proto_path.substr(0, index);

    proto_path_no_suffix =
        proto_path_no_suffix.substr(output_relative_path.size());

    const auto code =
        tensorflow::GetProtoTextFunctionCode(*fd, kTensorFlowHeaderPrefix);

    // Three passes, one for each output file.
    for (int pass = 0; pass < 3; ++pass) {
      string suffix;
      string data;
      if (pass == 0) {
        suffix = ".pb_text.h";
        data = code.header;
      } else if (pass == 1) {
        suffix = ".pb_text-impl.h";
        data = code.header_impl;
      } else {
        suffix = ".pb_text.cc";
        data = code.cc;
      }

      const string path = output_root + "/" + proto_path_no_suffix + suffix;
      FILE* f = fopen(path.c_str(), "w");
      if (f == nullptr) {
        // We don't expect this output to be generated. It was specified in the
        // list of sources solely to satisfy a proto import dependency.
        continue;
      }
      if (fwrite(data.c_str(), 1, data.size(), f) != data.size()) {
        fclose(f);
        return -1;
      }
      if (fclose(f) != 0) {
        return -1;
      }
    }
  }
  return 0;
}

}  // namespace tensorflow

int main(int argc, char** argv) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSproto_textPSgen_proto_text_functionsDTcc mht_4(mht_4_v, 337, "", "./tensorflow/tools/proto_text/gen_proto_text_functions.cc", "main");
 return tensorflow::MainImpl(argc, argv); }
