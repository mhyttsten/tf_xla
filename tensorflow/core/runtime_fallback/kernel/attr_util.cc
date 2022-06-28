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
class MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSattr_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSattr_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSattr_utilDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/runtime_fallback/kernel/attr_util.h"

#include <assert.h>
#include <stdlib.h>

#include <string>
#include <vector>

#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/runtime_fallback/util/attr_util.h"

namespace tensorflow {

// TODO(annarev): merge this file with attr_util.cc
// after reducing attr_util dependencies.
DataType ParseTFDataType(StringPiece dtype) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSattr_utilDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/runtime_fallback/kernel/attr_util.cc", "ParseTFDataType");

  if (dtype == "DT_INT8") {
    return DataType::DT_INT8;
  } else if (dtype == "DT_INT32") {
    return DataType::DT_INT32;
  } else if (dtype == "DT_INT64") {
    return DataType::DT_INT64;
  } else if (dtype == "DT_FLOAT") {
    return DataType::DT_FLOAT;
  } else if (dtype == "DT_DOUBLE") {
    return DataType::DT_DOUBLE;
  } else {
    assert(false && "Unsupported dtype");
    abort();
  }
}

bool ParseBoolAttrValue(StringPiece attr_value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSattr_utilDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/runtime_fallback/kernel/attr_util.cc", "ParseBoolAttrValue");

  if (attr_value == "false") {
    return false;
  } else if (attr_value == "true") {
    return true;
  } else {
    assert(false && "Bool attribute value invalid");
    abort();
  }
}

Status ParseValue(StringPiece input, bool* value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSattr_utilDTcc mht_2(mht_2_v, 237, "", "./tensorflow/core/runtime_fallback/kernel/attr_util.cc", "ParseValue");

  *value = ParseBoolAttrValue(input);
  return Status::OK();
}

Status ParseValue(StringPiece input, int32* value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSattr_utilDTcc mht_3(mht_3_v, 245, "", "./tensorflow/core/runtime_fallback/kernel/attr_util.cc", "ParseValue");

  bool parse_result = absl::SimpleAtoi(input, value);
  if (!parse_result) {
    return errors::InvalidArgument("Could not parse int32 from ", input);
  }
  return Status::OK();
}

Status ParseValue(StringPiece input, DataType* value) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSattr_utilDTcc mht_4(mht_4_v, 256, "", "./tensorflow/core/runtime_fallback/kernel/attr_util.cc", "ParseValue");

  *value = ParseTFDataType(input);
  return Status::OK();
}

Status ParseValue(StringPiece input, std::string* value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSattr_utilDTcc mht_5(mht_5_v, 264, "", "./tensorflow/core/runtime_fallback/kernel/attr_util.cc", "ParseValue");

  *value = std::string(input);
  return Status::OK();
}

Status ParseValue(StringPiece input, std::vector<int32>* value) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSattr_utilDTcc mht_6(mht_6_v, 272, "", "./tensorflow/core/runtime_fallback/kernel/attr_util.cc", "ParseValue");

  std::vector<std::string> parts = str_util::Split(input, ",");
  value->reserve(parts.size());
  for (const auto& value_str : parts) {
    int32_t value_int;
    bool parse_result = absl::SimpleAtoi(value_str, &value_int);
    if (!parse_result) {
      return errors::InvalidArgument("Could not parse list of integers from ",
                                     input);
    }
    value->push_back(value_int);
  }
  return Status::OK();
}

Status ParseValue(StringPiece input, Padding* value) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSattr_utilDTcc mht_7(mht_7_v, 290, "", "./tensorflow/core/runtime_fallback/kernel/attr_util.cc", "ParseValue");

  return GetPaddingFromString(input, value);
}

Status AddOpAttr(const std::string& name, const std::string& attr_value,
                 tfrt::OpAttrs* opattrs) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("name: \"" + name + "\"");
   mht_8_v.push_back("attr_value: \"" + attr_value + "\"");
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSattr_utilDTcc mht_8(mht_8_v, 300, "", "./tensorflow/core/runtime_fallback/kernel/attr_util.cc", "AddOpAttr");

  Status s;
  // Splits attr_value into type and value
  std::vector<absl::string_view> value_split = tfd::AttrValueSplit(attr_value);
  auto& type = value_split[0];
  auto& value = value_split[1];
  if (type == "bool") {
    bool val;
    s = ParseValue(value, &val);
    opattrs->Set<bool>(name, val);
  } else if (type == "i32") {
    int32_t val;
    s = ParseValue(value, &val);
    opattrs->Set<int32>(name, val);
  } else if (type == "string" || type == "padding") {
    std::string val;
    s = ParseValue(value, &val);
    opattrs->SetString(name, val);
  } else if (type == "tfdtype") {
    DataType val;
    s = ParseValue(value, &val);
    opattrs->Set<tfrt::OpAttrType>(name, tfd::ConvertFromTfDataType(val));
  } else if (type == "list(i32)") {
    std::vector<int32> val;
    s = ParseValue(value, &val);
    opattrs->SetArray<int32>(name, val);
  }
  return s;
}

Status FillOpAttrs(tfrt::RemainingAttributes attrs, tfrt::OpAttrs* opattrs) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSattr_utilDTcc mht_9(mht_9_v, 333, "", "./tensorflow/core/runtime_fallback/kernel/attr_util.cc", "FillOpAttrs");

  int num_tf_attrs = attrs.size() / 2;
  Status status;
  for (int i = 0; i < num_tf_attrs; ++i) {
    // Each TF attribute is represented as a pair of name and value strings.
    std::string name = attrs.GetStringAttribute(i * 2).str();
    std::string attr_value = attrs.GetStringAttribute(i * 2 + 1).str();
    Status s = AddOpAttr(name, attr_value, opattrs);
    status.Update(s);
  }
  return status;
}

}  // namespace tensorflow
