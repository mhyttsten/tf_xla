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
#ifndef TENSORFLOW_CORE_PROFILER_LIB_TRACEME_ENCODE_H_
#define TENSORFLOW_CORE_PROFILER_LIB_TRACEME_ENCODE_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSlibPStraceme_encodeDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPStraceme_encodeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSlibPStraceme_encodeDTh() {
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


#include <string.h>

#include <initializer_list>
#include <string>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace profiler {

// An argument passed to TraceMeEncode.
struct TraceMeArg {
  // This constructor is required because absl::AlphaNum is non-copyable.
  template <typename Value>
  TraceMeArg(absl::string_view k, Value v) : key(k), value(v) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("k: \"" + std::string(k.data(), k.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPStraceme_encodeDTh mht_0(mht_0_v, 206, "", "./tensorflow/core/profiler/lib/traceme_encode.h", "TraceMeArg");
}

  TF_DISALLOW_COPY_AND_ASSIGN(TraceMeArg);

  absl::string_view key;
  absl::AlphaNum value;
};

namespace traceme_internal {

// Copies the contents of str to the address pointed by out.
// Returns the address after the copy.
// REQUIRED: The address range [out, out + str.size()] must have been allocated.
TF_ATTRIBUTE_ALWAYS_INLINE inline char* Append(char* out,
                                               absl::string_view str) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("out: \"" + (out == nullptr ? std::string("nullptr") : std::string((char*)out)) + "\"");
   mht_1_v.push_back("str: \"" + std::string(str.data(), str.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPStraceme_encodeDTh mht_1(mht_1_v, 225, "", "./tensorflow/core/profiler/lib/traceme_encode.h", "Append");

  DCHECK(!absl::StrContains(str, '#'))
      << "'#' is not a valid character in TraceMeEncode";
  const size_t str_size = str.size();
  if (TF_PREDICT_TRUE(str_size > 0)) {
    memcpy(out, str.data(), str_size);
    out += str_size;
  }
  return out;
}

// Appends args encoded as TraceMe metadata to name.
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string AppendArgs(
    std::string name, std::initializer_list<TraceMeArg> args) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPStraceme_encodeDTh mht_2(mht_2_v, 242, "", "./tensorflow/core/profiler/lib/traceme_encode.h", "AppendArgs");

  if (TF_PREDICT_TRUE(args.size() > 0)) {
    const auto old_size = name.size();
    auto new_size = old_size + args.size() * 2 + 1;
    for (const auto& arg : args) {
      new_size += arg.key.size() + arg.value.size();
    }
    name.resize(new_size);
    char* const begin = &name[0];
    char* out = begin + old_size;
    *out++ = '#';
    for (const auto& arg : args) {
      out = Append(out, arg.key);
      *out++ = '=';
      out = Append(out, arg.value.Piece());
      *out++ = ',';
    }
    *(out - 1) = '#';
    DCHECK_EQ(out, begin + new_size);
  }
  return name;
}

// Appends new_metadata to the metadata part of name.
TF_ATTRIBUTE_ALWAYS_INLINE inline void AppendMetadata(
    std::string* name, absl::string_view new_metadata) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("new_metadata: \"" + std::string(new_metadata.data(), new_metadata.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPStraceme_encodeDTh mht_3(mht_3_v, 271, "", "./tensorflow/core/profiler/lib/traceme_encode.h", "AppendMetadata");

  if (!TF_PREDICT_FALSE(new_metadata.empty())) {
    if (!name->empty() && name->back() == '#') {  // name already has metadata
      name->back() = ',';
      if (TF_PREDICT_TRUE(new_metadata.front() == '#')) {
        new_metadata.remove_prefix(1);
      }
    }
    name->append(new_metadata.data(), new_metadata.size());
  }
}

}  // namespace traceme_internal

// Encodes an event name and arguments into TraceMe metadata.
// Use within a lambda to avoid expensive operations when tracing is disabled.
// Example Usage:
//   TraceMe trace_me([value1]() {
//     return TraceMeEncode("my_trace", {{"key1", value1}, {"key2", 42}});
//   });
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeEncode(
    std::string name, std::initializer_list<TraceMeArg> args) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPStraceme_encodeDTh mht_4(mht_4_v, 296, "", "./tensorflow/core/profiler/lib/traceme_encode.h", "TraceMeEncode");

  return traceme_internal::AppendArgs(std::move(name), args);
}
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeEncode(
    absl::string_view name, std::initializer_list<TraceMeArg> args) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPStraceme_encodeDTh mht_5(mht_5_v, 304, "", "./tensorflow/core/profiler/lib/traceme_encode.h", "TraceMeEncode");

  return traceme_internal::AppendArgs(std::string(name), args);
}
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeEncode(
    const char* name, std::initializer_list<TraceMeArg> args) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPStraceme_encodeDTh mht_6(mht_6_v, 312, "", "./tensorflow/core/profiler/lib/traceme_encode.h", "TraceMeEncode");

  return traceme_internal::AppendArgs(std::string(name), args);
}

// Encodes arguments into TraceMe metadata.
// Use within a lambda to avoid expensive operations when tracing is disabled.
// Example Usage:
//   TraceMe trace_me("my_trace");
//   ...
//   trace_me.AppendMetadata([value1]() {
//     return TraceMeEncode({{"key1", value1}, {"key2", 42}});
//   });
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeEncode(
    std::initializer_list<TraceMeArg> args) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPStraceme_encodeDTh mht_7(mht_7_v, 328, "", "./tensorflow/core/profiler/lib/traceme_encode.h", "TraceMeEncode");

  return traceme_internal::AppendArgs(std::string(), args);
}

// Concatenates op_name and op_type.
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeOp(
    absl::string_view op_name, absl::string_view op_type) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("op_name: \"" + std::string(op_name.data(), op_name.size()) + "\"");
   mht_8_v.push_back("op_type: \"" + std::string(op_type.data(), op_type.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPStraceme_encodeDTh mht_8(mht_8_v, 339, "", "./tensorflow/core/profiler/lib/traceme_encode.h", "TraceMeOp");

  return absl::StrCat(op_name, ":", op_type);
}

TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeOp(const char* op_name,
                                                        const char* op_type) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   mht_9_v.push_back("op_type: \"" + (op_type == nullptr ? std::string("nullptr") : std::string((char*)op_type)) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPStraceme_encodeDTh mht_9(mht_9_v, 349, "", "./tensorflow/core/profiler/lib/traceme_encode.h", "TraceMeOp");

  return absl::StrCat(op_name, ":", op_type);
}

TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeOp(
    std::string&& op_name, absl::string_view op_type) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("op_type: \"" + std::string(op_type.data(), op_type.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPStraceme_encodeDTh mht_10(mht_10_v, 358, "", "./tensorflow/core/profiler/lib/traceme_encode.h", "TraceMeOp");

  absl::StrAppend(&op_name, ":", op_type);
  return op_name;
}

// Concatenates op_name and op_type.
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeOpOverride(
    absl::string_view op_name, absl::string_view op_type) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("op_name: \"" + std::string(op_name.data(), op_name.size()) + "\"");
   mht_11_v.push_back("op_type: \"" + std::string(op_type.data(), op_type.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPStraceme_encodeDTh mht_11(mht_11_v, 370, "", "./tensorflow/core/profiler/lib/traceme_encode.h", "TraceMeOpOverride");

  return absl::StrCat("#tf_op=", op_name, ":", op_type, "#");
}

TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeOpOverride(
    const char* op_name, const char* op_type) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   mht_12_v.push_back("op_type: \"" + (op_type == nullptr ? std::string("nullptr") : std::string((char*)op_type)) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPStraceme_encodeDTh mht_12(mht_12_v, 380, "", "./tensorflow/core/profiler/lib/traceme_encode.h", "TraceMeOpOverride");

  return absl::StrCat("#tf_op=", op_name, ":", op_type, "#");
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_LIB_TRACEME_ENCODE_H_
