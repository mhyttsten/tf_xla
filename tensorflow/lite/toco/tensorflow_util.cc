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
class MHTracer_DTPStensorflowPSlitePStocoPStensorflow_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPStensorflow_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPStensorflow_utilDTcc() {
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
#include "tensorflow/lite/toco/tensorflow_util.h"

#include <string.h>
#include <memory>
#include <set>

#ifdef GOOGLE_PLATFORM
#include "file/logging/log_lines.h"
#endif
#include "google/protobuf/map.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/toco/toco_port.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

using tensorflow::AttrValue;
using tensorflow::GraphDef;

void LogDumpGraphDef(int log_level, const std::string& message,
                     const GraphDef& tf_graph) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("message: \"" + message + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStensorflow_utilDTcc mht_0(mht_0_v, 211, "", "./tensorflow/lite/toco/tensorflow_util.cc", "LogDumpGraphDef");

  if (!VLOG_IS_ON(log_level)) {
    return;
  }
  std::set<std::string> ops;
  for (const auto& node : tf_graph.node()) {
    ops.insert(node.op());
  }
  std::string dump;
  toco::port::AppendF(&dump, R"MSG(
BEGIN DUMP OF TENSORFLOW GRAPHDEF (%s)
There are %d nodes.
There are %zu different op types:
)MSG",
                      message, tf_graph.node_size(), ops.size());
  for (const auto& op : ops) {
    toco::port::AppendF(&dump, "  %s\n", op);
  }
  dump.append(R"MSG(
PROTO DUMP
)MSG");
  for (const auto& node : tf_graph.node()) {
    toco::port::AppendF(&dump, R"MSG(
BEGIN NODE: name = %s
  op = %s
  inputs = [
)MSG",
                        node.name(), node.op());
    for (const auto& input : node.input()) {
      toco::port::AppendF(&dump, "    %s\n", input);
    }
    dump.append("  ]\n");
    for (const auto& attr : node.attr()) {
      toco::port::AppendF(&dump, "  ATTR: name = %s\n", attr.first);
      if (attr.second.value_case() == AttrValue::kFunc) {
        dump.append("    func\n");
      } else if (attr.second.value_case() == AttrValue::kPlaceholder) {
        toco::port::AppendF(&dump, "    placeholder: %s\n",
                            attr.second.placeholder());
      } else if (attr.second.value_case() == AttrValue::kS) {
        dump.append("    string:\n");
        dump.append(R"MSG(
      BEGIN EMBEDDED STRING
)MSG");
        const auto& lines = absl::StrSplit(attr.second.s(), '\n');
        for (const auto& line : lines) {
          toco::port::AppendF(&dump, "      %s\n", line);
        }
        dump.append(R"MSG(
      END EMBEDDED STRING
)MSG");
      } else if (attr.second.value_case() == AttrValue::kI) {
        toco::port::AppendF(&dump, "    int: %lld\n", attr.second.i());
      } else if (attr.second.value_case() == AttrValue::kF) {
        toco::port::AppendF(&dump, "    float: %g\n", attr.second.f());
      } else if (attr.second.value_case() == AttrValue::kB) {
        toco::port::AppendF(&dump, "    bool: %s\n",
                            attr.second.b() ? "true" : "false");
      } else if (attr.second.value_case() == AttrValue::kType) {
        toco::port::AppendF(&dump, "    type: %s\n",
                            tensorflow::DataType_Name(attr.second.type()));
      } else if (attr.second.value_case() == AttrValue::kShape) {
        dump.append("    shape: [ ");
        const auto& shape = attr.second.shape();
        for (int i = 0; i < shape.dim_size(); i++) {
          toco::port::AppendF(&dump, "%lld ", shape.dim(i).size());
        }
        dump.append("]\n");
      } else if (attr.second.value_case() == AttrValue::kTensor) {
        const auto& tensor = attr.second.tensor();
        dump.append("    TENSOR:\n");
        toco::port::AppendF(&dump, "      type: %s\n",
                            tensorflow::DataType_Name(tensor.dtype()));
        const auto& shape = tensor.tensor_shape();
        dump.append("      shape: [ ");
        for (int i = 0; i < shape.dim_size(); i++) {
          toco::port::AppendF(&dump, "%lld ", shape.dim(i).size());
        }
        dump.append("]\n");
        if (!tensor.tensor_content().empty()) {
          toco::port::AppendF(&dump, "      tensor_content: %zu bytes\n",
                              tensor.tensor_content().size());
        }
        if (tensor.dtype() == tensorflow::DT_INT32) {
          CHECK_EQ(0, tensor.tensor_content().size() % sizeof(int32));
          const int size = tensor.tensor_content().size() / sizeof(int32);
          std::vector<int32> data(size);
          toco::port::CopyToBuffer(tensor.tensor_content(),
                                   reinterpret_cast<char*>(data.data()));
          const int kMaxValsToPrint = 4;
          dump.append("        tensor_content as ints: [ ");
          for (int i = 0; i < kMaxValsToPrint && i < size; i++) {
            toco::port::AppendF(&dump, "%d ", data[i]);
          }
          if (size > kMaxValsToPrint) {
            dump.append("... ");
          }
          dump.append("]\n");
        }
        if (tensor.dtype() == tensorflow::DT_FLOAT) {
          CHECK_EQ(0, tensor.tensor_content().size() % sizeof(float));
          const int size = tensor.tensor_content().size() / sizeof(float);
          std::vector<float> data(size);
          toco::port::CopyToBuffer(tensor.tensor_content(),
                                   reinterpret_cast<char*>(data.data()));
          const int kMaxValsToPrint = 4;
          dump.append("        tensor_content as floats: [ ");
          for (int i = 0; i < kMaxValsToPrint && i < size; i++) {
            toco::port::AppendF(&dump, "%g ", data[i]);
          }
          if (size > kMaxValsToPrint) {
            dump.append("... ");
          }
          dump.append("]\n");
        }
        if (tensor.int_val_size()) {
          toco::port::AppendF(&dump, "      int_val: %d ints: [ ",
                              tensor.int_val_size());
          const int kMaxValsToPrint = 4;
          for (int i = 0; i < kMaxValsToPrint && i < tensor.int_val_size();
               i++) {
            toco::port::AppendF(&dump, "%d ", tensor.int_val(i));
          }
          if (tensor.int_val_size() > kMaxValsToPrint) {
            dump.append("... ");
          }
          dump.append("]\n");
        }
        if (tensor.float_val_size()) {
          toco::port::AppendF(&dump, "      float_val: %d floats: [ ",
                              tensor.float_val_size());
          const int kMaxValsToPrint = 4;
          for (int i = 0; i < kMaxValsToPrint && i < tensor.float_val_size();
               i++) {
            toco::port::AppendF(&dump, "%g ", tensor.float_val(i));
          }
          if (tensor.float_val_size() > kMaxValsToPrint) {
            dump.append("... ");
          }
          dump.append("]\n");
        }
        if (tensor.string_val_size()) {
          toco::port::AppendF(&dump, "      string_val: %d strings\n",
                              tensor.string_val_size());
        }
      } else if (attr.second.value_case() == AttrValue::kList) {
        dump.append("  LIST\n");
      }
    }
    dump.append("END NODE\n");
  }
  toco::port::AppendF(&dump, "END DUMP OF TENSORFLOW GRAPHDEF (%s)\n", message);
#if defined(GOOGLE_PLATFORM)
  VLOG_LINES(log_level, dump);
#else
  VLOG(log_level) << dump;
#endif
}
}  // namespace toco
