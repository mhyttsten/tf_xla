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
class MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_proto_to_jsonDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_proto_to_jsonDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_proto_to_jsonDTcc() {
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

// Usage:
//   hlo_proto_to_json --input_file=some_binary_proto
//   --output_file=path_to_dump_output
//
// Reads one serialized Hlo module, convert it into JSON format and dump into
// some output directory. some_binary_proto is obtained by serializing Hlo
// module to disk using the debug options
//
//   --xla_dump_to=DIR --xla_dump_hlo_as_proto

#include <stdio.h>

#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

using std::string;
using tensorflow::Env;

namespace xla {
namespace tools {

StatusOr<std::string> ToJson(const tensorflow::protobuf::Message& message) {
  std::string json_output;
  tensorflow::protobuf::util::JsonPrintOptions json_options;
  json_options.add_whitespace = true;
  json_options.always_print_primitive_fields = true;
  auto status = tensorflow::protobuf::util::MessageToJsonString(
      message, &json_output, json_options);
  if (!status.ok()) {
    return InternalError("MessageToJsonString failed: %s",
                         status.error_message().data());
  }
  return json_output;
}

void RealMain(const std::string& input, const std::string& output) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("input: \"" + input + "\"");
   mht_0_v.push_back("output: \"" + output + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_proto_to_jsonDTcc mht_0(mht_0_v, 231, "", "./tensorflow/compiler/xla/tools/hlo_proto_to_json.cc", "RealMain");

  HloProto hlo_proto;
  TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(), input,
                                          &hlo_proto))
      << "Can't open, read, or parse input file " << input;

  auto statusor = ToJson(hlo_proto);
  QCHECK(statusor.ok()) << "Error converting " << input << " to JSON."
                        << statusor.status();

  TF_CHECK_OK(tensorflow::WriteStringToFile(tensorflow::Env::Default(), output,
                                            statusor.ValueOrDie()));
}

}  // namespace tools
}  // namespace xla

int main(int argc, char** argv) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_proto_to_jsonDTcc mht_1(mht_1_v, 251, "", "./tensorflow/compiler/xla/tools/hlo_proto_to_json.cc", "main");

  std::string input_file, output_file;
  const std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("input_file", &input_file, "file to convert."),
      tensorflow::Flag("output_file", &output_file, "converted file"),
  };
  const std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  tensorflow::port::InitMain(usage.c_str(), &argc, &argv);
  QCHECK(parse_ok && argc == 1) << "\n" << usage;

  QCHECK(!input_file.empty()) << "--input_file is required";
  QCHECK(!output_file.empty()) << "--output_file is required";

  xla::tools::RealMain(input_file, output_file);

  return 0;
}
