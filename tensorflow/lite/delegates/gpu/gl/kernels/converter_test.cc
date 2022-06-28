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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverter_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverter_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverter_testDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/kernels/converter.h"

#include <algorithm>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/egl_environment.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

inline std::vector<float> GenerateFloats(float multiplier, int size) {
  std::vector<float> v(size);
  for (int i = 0; i < size; ++i) {
    v[i] = multiplier * i * (i % 2 == 0 ? -1 : 1);
  }
  return v;
}

Dimensions ToDimensions(const BHWC& shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverter_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter_test.cc", "ToDimensions");

  return Dimensions(shape.b, shape.h, shape.w, shape.c);
}

absl::Status RunFromTensorTest(const BHWC& shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverter_testDTcc mht_1(mht_1_v, 220, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter_test.cc", "RunFromTensorTest");

  // Create random input and calculate expected output for it.
  std::vector<float> input =
      GenerateFloats(0.01, GetElementsSizeForPHWC4(shape));
  std::vector<float> output(shape.DimensionsProduct(), 0);
  RETURN_IF_ERROR(
      ConvertFromPHWC4(absl::MakeConstSpan(input.data(), input.size()), shape,
                       absl::MakeSpan(output.data(), output.size())));

  std::unique_ptr<EglEnvironment> env;
  RETURN_IF_ERROR(EglEnvironment::NewEglEnvironment(&env));

  // Create input and output buffers
  GlBuffer input_buffer;
  RETURN_IF_ERROR(CreateReadOnlyShaderStorageBuffer(
      absl::MakeConstSpan(input.data(), input.size()), &input_buffer));

  GlBuffer output_buffer;
  RETURN_IF_ERROR(CreateReadWriteShaderStorageBuffer<float>(
      shape.DimensionsProduct(), &output_buffer));

  // Create converter and run it.
  auto builder = NewConverterBuilder(nullptr);
  TensorObjectDef input_def;
  input_def.object_def.data_type = DataType::FLOAT32;
  input_def.object_def.data_layout = DataLayout::DHWC4;
  input_def.object_def.object_type = ObjectType::OPENGL_SSBO;
  input_def.dimensions = ToDimensions(shape);
  TensorObjectDef output_def = input_def;
  output_def.object_def.data_layout = DataLayout::BHWC;
  std::unique_ptr<TensorObjectConverter> converter;
  RETURN_IF_ERROR(builder->MakeConverter(input_def, output_def, &converter));
  RETURN_IF_ERROR(converter->Convert(OpenGlBuffer{input_buffer.id()},
                                     OpenGlBuffer{output_buffer.id()}));

  // Compare outputs.
  std::vector<float> converted_output(output.size(), 0);
  RETURN_IF_ERROR(output_buffer.Read(
      absl::MakeSpan(converted_output.data(), converted_output.size())));
  if (output != converted_output) {
    return absl::InternalError("Outputs don't match");
  }
  return absl::OkStatus();
}

TEST(FromTensor, Smoke) {
  for (int32_t h : {1, 2, 3, 7, 20}) {
    for (int32_t w : {1, 2, 4, 5, 11}) {
      for (int32_t c : {1, 2, 4, 5, 8, 9}) {
        BHWC shape(1, h, w, c);
        auto status = RunFromTensorTest(shape);
        EXPECT_TRUE(status.ok()) << status << ", shape = " << shape.h << " "
                                 << shape.w << " " << shape.c;
      }
    }
  }
}

absl::Status RunToTensorTest(const BHWC& shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverter_testDTcc mht_2(mht_2_v, 281, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter_test.cc", "RunToTensorTest");

  // Create random input and calculate expected output for it.
  std::vector<float> input = GenerateFloats(0.01, shape.DimensionsProduct());
  std::vector<float> output(GetElementsSizeForPHWC4(shape), 0);
  RETURN_IF_ERROR(
      ConvertToPHWC4(absl::MakeConstSpan(input.data(), input.size()), shape,
                     absl::MakeSpan(output.data(), output.size())));

  std::unique_ptr<EglEnvironment> env;
  RETURN_IF_ERROR(EglEnvironment::NewEglEnvironment(&env));

  // Create input and output buffers
  GlBuffer input_buffer;
  RETURN_IF_ERROR(CreateReadOnlyShaderStorageBuffer(
      absl::MakeConstSpan(input.data(), input.size()), &input_buffer));

  GlBuffer output_buffer;
  RETURN_IF_ERROR(CreateReadWriteShaderStorageBuffer<float>(
      GetElementsSizeForPHWC4(shape), &output_buffer));

  // Create converter and run it.
  auto builder = NewConverterBuilder(nullptr);
  TensorObjectDef input_def;
  input_def.object_def.data_type = DataType::FLOAT32;
  input_def.object_def.data_layout = DataLayout::BHWC;
  input_def.object_def.object_type = ObjectType::OPENGL_SSBO;
  input_def.dimensions = ToDimensions(shape);
  TensorObjectDef output_def = input_def;
  output_def.object_def.data_layout = DataLayout::DHWC4;
  std::unique_ptr<TensorObjectConverter> converter;
  RETURN_IF_ERROR(builder->MakeConverter(input_def, output_def, &converter));
  RETURN_IF_ERROR(converter->Convert(OpenGlBuffer{input_buffer.id()},
                                     OpenGlBuffer{output_buffer.id()}));

  // Compare outputs.
  std::vector<float> converted_output(output.size(), 0);
  RETURN_IF_ERROR(output_buffer.Read(
      absl::MakeSpan(converted_output.data(), converted_output.size())));
  if (output != converted_output) {
    return absl::InternalError("Outputs don't match");
  }
  return absl::OkStatus();
}

TEST(ToTensor, Smoke) {
  for (int32_t h : {1, 2, 3, 7, 20}) {
    for (int32_t w : {1, 2, 4, 5, 11}) {
      for (int32_t c : {1, 2, 4, 5, 8, 9}) {
        BHWC shape(1, h, w, c);
        auto status = RunToTensorTest(shape);
        EXPECT_TRUE(status.ok()) << status << ", shape = " << shape.h << " "
                                 << shape.w << " " << shape.c;
      }
    }
  }
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
