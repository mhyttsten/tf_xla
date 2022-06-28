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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSserialization_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSserialization_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSserialization_testDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/gl/serialization.h"

#include <stddef.h>
#include <sys/types.h>

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/object.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

struct ProgramDesc {
  std::vector<Variable> parameters;
  std::vector<Object> objects;
  uint3 workgroup_size;
  uint3 num_workgroups;
  size_t shader_index;
};

struct Handler : public DeserializationHandler {
  absl::Status OnShader(absl::Span<const char> shader_src) final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSserialization_testDTcc mht_0(mht_0_v, 217, "", "./tensorflow/lite/delegates/gpu/gl/serialization_test.cc", "OnShader");

    shaders.push_back(std::string(shader_src.data(), shader_src.size()));
    return absl::OkStatus();
  }

  absl::Status OnProgram(const std::vector<Variable>& parameters,
                         const std::vector<Object>& objects,
                         const uint3& workgroup_size,
                         const uint3& num_workgroups,
                         size_t shader_index) final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSserialization_testDTcc mht_1(mht_1_v, 229, "", "./tensorflow/lite/delegates/gpu/gl/serialization_test.cc", "OnProgram");

    programs.push_back(
        {parameters, objects, workgroup_size, num_workgroups, shader_index});
    return absl::OkStatus();
  }

  void OnOptions(const CompiledModelOptions& o) final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSserialization_testDTcc mht_2(mht_2_v, 238, "", "./tensorflow/lite/delegates/gpu/gl/serialization_test.cc", "OnOptions");
 options = o; }

  std::vector<std::string> shaders;
  std::vector<ProgramDesc> programs;
  CompiledModelOptions options;
};

struct ParameterComparator {
  bool operator()(int32_t value) const {
    return value == absl::get<int32_t>(a.value);
  }

  bool operator()(const int2& value) const {
    auto v = absl::get<int2>(a.value);
    return value.x == v.x && value.y == v.y;
  }

  bool operator()(const int4& value) const {
    auto v = absl::get<int4>(a.value);
    return value.x == v.x && value.y == v.y && value.z == v.z && value.w == v.w;
  }

  bool operator()(const std::vector<int2>& value) const {
    auto v = absl::get<std::vector<int2>>(a.value);
    if (v.size() != value.size()) {
      return false;
    }
    for (int i = 0; i < v.size(); ++i) {
      if (v[i].x != value[i].x || v[i].y != value[i].y) {
        return false;
      }
    }
    return true;
  }

  bool operator()(uint32_t value) const {
    return value == absl::get<uint32_t>(a.value);
  }

  bool operator()(const uint4& value) const {
    auto v = absl::get<uint4>(a.value);
    return value.x == v.x && value.y == v.y && value.z == v.z && value.w == v.w;
  }

  bool operator()(float value) const {
    return value == absl::get<float>(a.value);
  }

  bool operator()(float2 value) const {
    auto v = absl::get<float2>(a.value);
    return value.x == v.x && value.y == v.y;
  }

  bool operator()(const float4& value) const {
    auto v = absl::get<float4>(a.value);
    return value.x == v.x && value.y == v.y && value.z == v.z && value.w == v.w;
  }

  bool operator()(const std::vector<float4>& value) const {
    auto v = absl::get<std::vector<float4>>(a.value);
    if (v.size() != value.size()) {
      return false;
    }
    for (int i = 0; i < v.size(); ++i) {
      if (v[i].x != value[i].x || v[i].y != value[i].y) {
        return false;
      }
    }
    return true;
  }

  Variable a;
};

bool Eq(const Variable& a, const Variable& b) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSserialization_testDTcc mht_3(mht_3_v, 315, "", "./tensorflow/lite/delegates/gpu/gl/serialization_test.cc", "Eq");

  return a.name == b.name && absl::visit(ParameterComparator{a}, b.value);
}

struct ObjectComparator {
  bool operator()(const ObjectData& data) const {
    return absl::get<ObjectData>(a.object) == data;
  }
  bool operator()(const ObjectRef& ref) const {
    return absl::get<ObjectRef>(a.object) == ref;
  }

  Object a;
};

bool Eq(const Object& a, const Object& b) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSserialization_testDTcc mht_4(mht_4_v, 333, "", "./tensorflow/lite/delegates/gpu/gl/serialization_test.cc", "Eq");

  return a.access == b.access && a.binding == b.binding &&
         absl::visit(ObjectComparator{a}, b.object);
}

TEST(Smoke, Read) {
  std::string shader1 = "A";
  std::string shader2 = "B";

  SerializedCompiledModelBuilder builder;
  builder.AddShader(shader1);
  builder.AddShader(shader2);

  std::vector<Variable> parameters;
  parameters.push_back({"1", int32_t(1)});
  parameters.push_back({"2", int2(1, 2)});
  parameters.push_back({"3", int4(1, 2, 3, 4)});
  parameters.push_back({"4", uint32_t(10)});
  parameters.push_back({"5", uint4(10, 20, 30, 40)});
  parameters.push_back({"6", -2.0f});
  parameters.push_back({"7", float2(1, -1)});
  parameters.push_back({"8", float4(1, -1, 2, -2)});
  parameters.push_back(
      {"9", std::vector<int2>{int2(1, 2), int2(3, 4), int2(5, 6)}});

  std::vector<Object> objects;
  objects.push_back(MakeReadonlyBuffer(std::vector<float>{1, 2, 3, 4}));
  objects.push_back(Object{AccessType::WRITE, DataType::FLOAT32,
                           ObjectType::TEXTURE, 5, uint3(1, 2, 3), 100u});
  objects.push_back(Object{AccessType::READ_WRITE, DataType::INT8,
                           ObjectType::BUFFER, 6, uint2(2, 1),
                           std::vector<uint8_t>{7, 9}});
  uint3 num_workgroups(10, 20, 30);
  uint3 workgroup_size(1, 2, 3);
  builder.AddProgram(parameters, objects, workgroup_size, num_workgroups, 1);

  Handler handler;
  CompiledModelOptions options;
  options.dynamic_batch = true;
  ASSERT_TRUE(
      DeserializeCompiledModel(builder.Finalize(options), &handler).ok());
  EXPECT_EQ(num_workgroups.data_, handler.programs[0].num_workgroups.data_);
  EXPECT_EQ(workgroup_size.data_, handler.programs[0].workgroup_size.data_);
  EXPECT_THAT(handler.shaders, ::testing::ElementsAre(shader1, shader2));
  EXPECT_EQ(handler.programs[0].parameters.size(), parameters.size());
  for (int i = 0; i < parameters.size(); ++i) {
    EXPECT_TRUE(Eq(parameters[i], handler.programs[0].parameters[i])) << i;
  }
  EXPECT_EQ(handler.programs[0].objects.size(), objects.size());
  for (int i = 0; i < objects.size(); ++i) {
    EXPECT_TRUE(Eq(objects[i], handler.programs[0].objects[i])) << i;
  }
  EXPECT_TRUE(handler.options.dynamic_batch);
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
