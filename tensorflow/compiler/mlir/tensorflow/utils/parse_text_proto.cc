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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSparse_text_protoDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSparse_text_protoDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSparse_text_protoDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/parse_text_proto.h"

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

namespace {
// Error collector that simply ignores errors reported.
class NoOpErrorCollector : public protobuf::io::ErrorCollector {
 public:
  void AddError(int line, int column, const std::string& message) override {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("message: \"" + message + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSparse_text_protoDTcc mht_0(mht_0_v, 201, "", "./tensorflow/compiler/mlir/tensorflow/utils/parse_text_proto.cc", "AddError");
}
};
}  // namespace

Status ConsumePrefix(absl::string_view str, absl::string_view prefix,
                     absl::string_view* output) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("str: \"" + std::string(str.data(), str.size()) + "\"");
   mht_1_v.push_back("prefix: \"" + std::string(prefix.data(), prefix.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSparse_text_protoDTcc mht_1(mht_1_v, 211, "", "./tensorflow/compiler/mlir/tensorflow/utils/parse_text_proto.cc", "ConsumePrefix");

  if (absl::StartsWith(str, prefix)) {
    *output = str.substr(prefix.size());
    return Status::OK();
  }
  return errors::NotFound("No prefix \"", prefix, "\" in \"", str, "\"");
}

Status ParseTextProto(absl::string_view text_proto,
                      absl::string_view prefix_to_strip,
                      protobuf::Message* parsed_proto) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("text_proto: \"" + std::string(text_proto.data(), text_proto.size()) + "\"");
   mht_2_v.push_back("prefix_to_strip: \"" + std::string(prefix_to_strip.data(), prefix_to_strip.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSparse_text_protoDTcc mht_2(mht_2_v, 226, "", "./tensorflow/compiler/mlir/tensorflow/utils/parse_text_proto.cc", "ParseTextProto");

  protobuf::TextFormat::Parser parser;
  // Don't produce errors when attempting to parse text format as it would fail
  // when the input is actually a binary file.
  NoOpErrorCollector collector;
  parser.RecordErrorsTo(&collector);
  // Attempt to parse as text.
  absl::string_view text_proto_without_prefix = text_proto;
  if (!prefix_to_strip.empty()) {
    TF_RETURN_IF_ERROR(
        ConsumePrefix(text_proto, prefix_to_strip, &text_proto_without_prefix));
  }
  protobuf::io::ArrayInputStream input_stream(text_proto_without_prefix.data(),
                                              text_proto_without_prefix.size());
  if (parser.Parse(&input_stream, parsed_proto)) {
    return Status::OK();
  }
  parsed_proto->Clear();
  return errors::InvalidArgument("Could not parse text proto: ", text_proto);
}

}  // namespace tensorflow
