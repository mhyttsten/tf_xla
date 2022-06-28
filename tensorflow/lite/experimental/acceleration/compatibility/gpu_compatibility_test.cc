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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPScompatibilityPSgpu_compatibility_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPScompatibilityPSgpu_compatibility_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPScompatibilityPSgpu_compatibility_testDTcc() {
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
#include "tensorflow/lite/experimental/acceleration/compatibility/gpu_compatibility.h"

#include <algorithm>
#include <cstddef>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/acceleration/compatibility/devicedb-sample.h"

namespace {

class GPUCompatibilityTest : public ::testing::Test {
 protected:
  GPUCompatibilityTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPScompatibilityPSgpu_compatibility_testDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/experimental/acceleration/compatibility/gpu_compatibility_test.cc", "GPUCompatibilityTest");

    list_ = tflite::acceleration::GPUCompatibilityList::Create(
        g_tflite_acceleration_devicedb_sample_binary,
        g_tflite_acceleration_devicedb_sample_binary_len);
  }

  std::unique_ptr<tflite::acceleration::GPUCompatibilityList> list_;
};

TEST_F(GPUCompatibilityTest, ReturnsSupportedForFullMatch) {
  ASSERT_TRUE(list_ != nullptr);

  tflite::acceleration::AndroidInfo android_info = {.android_sdk_version = "24",
                                                    .model = "m712c"};

  tflite::gpu::GpuInfo tflite_gpu_info;
  tflite_gpu_info.opengl_info.major_version = 3;
  tflite_gpu_info.opengl_info.minor_version = 1;

  EXPECT_TRUE(list_->Includes(android_info, tflite_gpu_info));
}

TEST_F(GPUCompatibilityTest, ReturnsUnsupportedForFullMatch) {
  ASSERT_TRUE(list_ != nullptr);

  tflite::acceleration::AndroidInfo android_info = {.android_sdk_version = "28",
                                                    .model = "SM-G960F",
                                                    .device = "starlte",
                                                    .manufacturer = "Samsung"};
  tflite::gpu::GpuInfo tflite_gpu_info;
  tflite_gpu_info.opengl_info.renderer_name = "Mali-G72";
  tflite_gpu_info.opengl_info.major_version = 3;
  tflite_gpu_info.opengl_info.minor_version = 2;
  EXPECT_FALSE(list_->Includes(android_info, tflite_gpu_info));
}

TEST_F(GPUCompatibilityTest, ReturnsDefaultOptions) {
  ASSERT_TRUE(list_ != nullptr);
  tflite::acceleration::AndroidInfo android_info;
  tflite::gpu::GpuInfo tflite_gpu_info;
  auto default_options = TfLiteGpuDelegateOptionsV2Default();
  auto best_options = list_->GetBestOptionsFor(android_info, tflite_gpu_info);
  EXPECT_EQ(best_options.is_precision_loss_allowed,
            default_options.is_precision_loss_allowed);
  EXPECT_EQ(best_options.inference_preference,
            default_options.inference_preference);
  EXPECT_EQ(best_options.inference_priority1,
            default_options.inference_priority1);
  EXPECT_EQ(best_options.inference_priority2,
            default_options.inference_priority2);
  EXPECT_EQ(best_options.inference_priority3,
            default_options.inference_priority3);
  EXPECT_EQ(best_options.experimental_flags,
            default_options.experimental_flags);
  EXPECT_EQ(best_options.max_delegated_partitions,
            default_options.max_delegated_partitions);
}

TEST(GPUCompatibility, RecogniseValidCompatibilityListFlatbuffer) {
  EXPECT_TRUE(tflite::acceleration::GPUCompatibilityList::IsValidFlatbuffer(
      g_tflite_acceleration_devicedb_sample_binary,
      g_tflite_acceleration_devicedb_sample_binary_len));
}

TEST(GPUCompatibility, RecogniseInvalidCompatibilityListFlatbuffer) {
  unsigned char invalid_buffer[100];
  std::fill(invalid_buffer, invalid_buffer + 100, ' ');
  EXPECT_FALSE(tflite::acceleration::GPUCompatibilityList::IsValidFlatbuffer(
      invalid_buffer, 100));
}

TEST(GPUCompatibility, CreationWithInvalidCompatibilityListFlatbuffer) {
  unsigned char invalid_buffer[10];
  std::fill(invalid_buffer, invalid_buffer + 10, ' ');
  std::unique_ptr<tflite::acceleration::GPUCompatibilityList> list =
      tflite::acceleration::GPUCompatibilityList::Create(invalid_buffer, 10);
  EXPECT_EQ(list, nullptr);
}

TEST(GPUCompatibility, CreationWithNullCompatibilityListFlatbuffer) {
  std::unique_ptr<tflite::acceleration::GPUCompatibilityList> list =
      tflite::acceleration::GPUCompatibilityList::Create(nullptr, 0);
  EXPECT_EQ(list, nullptr);
}

}  // namespace
