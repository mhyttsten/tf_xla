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

#ifndef TENSORFLOW_LITE_TOOLS_TOOL_PARAMS_H_
#define TENSORFLOW_LITE_TOOLS_TOOL_PARAMS_H_
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
class MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh {
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
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh() {
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

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tflite {
namespace tools {

template <typename T>
class TypedToolParam;

class ToolParam {
 protected:
  enum class ParamType { TYPE_INT32, TYPE_FLOAT, TYPE_BOOL, TYPE_STRING };
  template <typename T>
  static ParamType GetValueType();

 public:
  template <typename T>
  static std::unique_ptr<ToolParam> Create(const T& default_value,
                                           int position = 0) {
    auto* param = new TypedToolParam<T>(default_value);
    param->SetPosition(position);
    return std::unique_ptr<ToolParam>(param);
  }

  template <typename T>
  TypedToolParam<T>* AsTyped() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh mht_0(mht_0_v, 215, "", "./tensorflow/lite/tools/tool_params.h", "AsTyped");

    AssertHasSameType(GetValueType<T>(), type_);
    return static_cast<TypedToolParam<T>*>(this);
  }

  template <typename T>
  const TypedToolParam<T>* AsConstTyped() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh mht_1(mht_1_v, 224, "", "./tensorflow/lite/tools/tool_params.h", "AsConstTyped");

    AssertHasSameType(GetValueType<T>(), type_);
    return static_cast<const TypedToolParam<T>*>(this);
  }

  virtual ~ToolParam() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh mht_2(mht_2_v, 232, "", "./tensorflow/lite/tools/tool_params.h", "~ToolParam");
}
  explicit ToolParam(ParamType type)
      : has_value_set_(false), position_(0), type_(type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh mht_3(mht_3_v, 237, "", "./tensorflow/lite/tools/tool_params.h", "ToolParam");
}

  bool HasValueSet() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh mht_4(mht_4_v, 242, "", "./tensorflow/lite/tools/tool_params.h", "HasValueSet");
 return has_value_set_; }

  int GetPosition() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh mht_5(mht_5_v, 247, "", "./tensorflow/lite/tools/tool_params.h", "GetPosition");
 return position_; }
  void SetPosition(int position) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh mht_6(mht_6_v, 251, "", "./tensorflow/lite/tools/tool_params.h", "SetPosition");
 position_ = position; }

  virtual void Set(const ToolParam&) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh mht_7(mht_7_v, 256, "", "./tensorflow/lite/tools/tool_params.h", "Set");
}

  virtual std::unique_ptr<ToolParam> Clone() const = 0;

 protected:
  bool has_value_set_;

  // Represents the relative ordering among a set of params.
  // Note: in our code, a ToolParam is generally used together with a
  // tflite::Flag so that its value could be set when parsing commandline flags.
  // In this case, the `position_` is simply the index of the particular flag
  // into the list of commandline flags (i.e. named 'argv' in general).
  int position_;

 private:
  static void AssertHasSameType(ParamType a, ParamType b);

  const ParamType type_;
};

template <typename T>
class TypedToolParam : public ToolParam {
 public:
  explicit TypedToolParam(const T& value)
      : ToolParam(GetValueType<T>()), value_(value) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh mht_8(mht_8_v, 283, "", "./tensorflow/lite/tools/tool_params.h", "TypedToolParam");
}

  void Set(const T& value) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh mht_9(mht_9_v, 288, "", "./tensorflow/lite/tools/tool_params.h", "Set");

    value_ = value;
    has_value_set_ = true;
  }

  T Get() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh mht_10(mht_10_v, 296, "", "./tensorflow/lite/tools/tool_params.h", "Get");
 return value_; }

  void Set(const ToolParam& other) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh mht_11(mht_11_v, 301, "", "./tensorflow/lite/tools/tool_params.h", "Set");

    Set(other.AsConstTyped<T>()->Get());
    SetPosition(other.AsConstTyped<T>()->GetPosition());
  }

  std::unique_ptr<ToolParam> Clone() const override {
    return ToolParam::Create<T>(value_, position_);
  }

 private:
  T value_;
};

// A map-like container for holding values of different types.
class ToolParams {
 public:
  // Add a ToolParam instance `value` w/ `name` to this container.
  void AddParam(const std::string& name, std::unique_ptr<ToolParam> value) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh mht_12(mht_12_v, 322, "", "./tensorflow/lite/tools/tool_params.h", "AddParam");

    params_[name] = std::move(value);
  }

  void RemoveParam(const std::string& name) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh mht_13(mht_13_v, 330, "", "./tensorflow/lite/tools/tool_params.h", "RemoveParam");
 params_.erase(name); }

  bool HasParam(const std::string& name) const {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh mht_14(mht_14_v, 336, "", "./tensorflow/lite/tools/tool_params.h", "HasParam");

    return params_.find(name) != params_.end();
  }

  bool Empty() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh mht_15(mht_15_v, 343, "", "./tensorflow/lite/tools/tool_params.h", "Empty");
 return params_.empty(); }

  const ToolParam* GetParam(const std::string& name) const {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh mht_16(mht_16_v, 349, "", "./tensorflow/lite/tools/tool_params.h", "GetParam");

    const auto& entry = params_.find(name);
    if (entry == params_.end()) return nullptr;
    return entry->second.get();
  }

  template <typename T>
  void Set(const std::string& name, const T& value, int position = 0) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh mht_17(mht_17_v, 360, "", "./tensorflow/lite/tools/tool_params.h", "Set");

    AssertParamExists(name);
    params_.at(name)->AsTyped<T>()->Set(value);
    params_.at(name)->AsTyped<T>()->SetPosition(position);
  }

  template <typename T>
  bool HasValueSet(const std::string& name) const {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh mht_18(mht_18_v, 371, "", "./tensorflow/lite/tools/tool_params.h", "HasValueSet");

    AssertParamExists(name);
    return params_.at(name)->AsConstTyped<T>()->HasValueSet();
  }

  template <typename T>
  int GetPosition(const std::string& name) const {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh mht_19(mht_19_v, 381, "", "./tensorflow/lite/tools/tool_params.h", "GetPosition");

    AssertParamExists(name);
    return params_.at(name)->AsConstTyped<T>()->GetPosition();
  }

  template <typename T>
  T Get(const std::string& name) const {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPStool_paramsDTh mht_20(mht_20_v, 391, "", "./tensorflow/lite/tools/tool_params.h", "Get");

    AssertParamExists(name);
    return params_.at(name)->AsConstTyped<T>()->Get();
  }

  // Set the value of all same parameters from 'other'.
  void Set(const ToolParams& other);

  // Merge the value of all parameters from 'other'. 'overwrite' indicates
  // whether the value of the same paratmeter is overwritten or not.
  void Merge(const ToolParams& other, bool overwrite = false);

 private:
  void AssertParamExists(const std::string& name) const;
  std::unordered_map<std::string, std::unique_ptr<ToolParam>> params_;
};

#define LOG_TOOL_PARAM(params, type, name, description, verbose)      \
  do {                                                                \
    TFLITE_MAY_LOG(INFO, (verbose) || params.HasValueSet<type>(name)) \
        << description << ": [" << params.Get<type>(name) << "]";     \
  } while (0)

}  // namespace tools
}  // namespace tflite
#endif  // TENSORFLOW_LITE_TOOLS_TOOL_PARAMS_H_
