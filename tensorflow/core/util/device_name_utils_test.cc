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
class MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utils_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utils_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utils_testDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/device_name_utils.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

namespace {

bool RoundTripParsedName(const string& original, const string& expected) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("original: \"" + original + "\"");
   mht_0_v.push_back("expected: \"" + expected + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utils_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/util/device_name_utils_test.cc", "RoundTripParsedName");

  DeviceNameUtils::ParsedName p;
  if (!DeviceNameUtils::ParseFullName(original, &p)) {
    return false;
  }
  string round_tripped = DeviceNameUtils::ParsedNameToString(p);
  return (round_tripped == expected);
}

enum NamePart { kJob = 0x01, kReplica = 0x02, kTask = 0x04, kDevice = 0x08 };

bool RoundTripPartialName(int parts_to_test, const std::vector<string>& parts,
                          bool explicitDevice) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utils_testDTcc mht_1(mht_1_v, 214, "", "./tensorflow/core/util/device_name_utils_test.cc", "RoundTripPartialName");

  string original, expected;
  if (parts_to_test & kJob) {
    strings::StrAppend(&original, "/job:", parts[0]);
    strings::StrAppend(&expected, "/job:", parts[0]);
  }
  if (parts_to_test & kReplica) {
    strings::StrAppend(&original, "/replica:", parts[1]);
    strings::StrAppend(&expected, "/replica:", parts[1]);
  }
  if (parts_to_test & kTask) {
    strings::StrAppend(&original, "/task:", parts[2]);
    strings::StrAppend(&expected, "/task:", parts[2]);
  }
  if (parts_to_test & kDevice) {
    if (explicitDevice) {
      strings::StrAppend(&original, "/device:", parts[3]);
      strings::StrAppend(&expected, "/device:", parts[3]);
    } else {
      strings::StrAppend(&original, "/", parts[3]);
      strings::StrAppend(&expected,
                         "/device:", absl::AsciiStrToUpper(parts[3]));
    }
  }
  return RoundTripParsedName(original, expected);
}

}  // namespace

TEST(DeviceNameUtilsTest, Basic) {
  EXPECT_EQ(DeviceNameUtils::FullName("hello", 1, 2, "CPU", 3),
            "/job:hello/replica:1/task:2/device:CPU:3");

  {
    DeviceNameUtils::ParsedName p;
    EXPECT_FALSE(DeviceNameUtils::ParseFullName("foobar", &p));
    EXPECT_FALSE(DeviceNameUtils::ParseFullName(
        "/job:123/replica:1/task:2/device:GPU:3", &p));
    EXPECT_FALSE(
        DeviceNameUtils::ParseFullName("/job:123/replica:1/task:2/gpu:", &p));
    EXPECT_FALSE(DeviceNameUtils::ParseFullName(
        "/job:123/replica:1/task:2/device:gpu:", &p));
    EXPECT_FALSE(DeviceNameUtils::ParseFullName(
        "/job:foo/replica:-1/task:2/device:GPU:3", &p));
    EXPECT_FALSE(DeviceNameUtils::ParseFullName(
        "/job:foo/replica:1/task:-2/device:GPU:3", &p));
    EXPECT_FALSE(
        DeviceNameUtils::ParseFullName("/job:foo/replica:1/task:2/bar:3", &p));
    EXPECT_FALSE(DeviceNameUtils::ParseFullName(
        "/job:foo/replica:1/task:2/device:GPU:3/extra", &p));
    EXPECT_TRUE(DeviceNameUtils::ParseFullName(
        "/job:foo/replica:1/task:2/device:GPU:3", &p));
    EXPECT_TRUE(p.has_job);
    EXPECT_TRUE(p.has_replica);
    EXPECT_TRUE(p.has_task);
    EXPECT_TRUE(p.has_type);
    EXPECT_TRUE(p.has_id);
    EXPECT_EQ(p.job, "foo");
    EXPECT_EQ(p.replica, 1);
    EXPECT_EQ(p.task, 2);
    EXPECT_EQ(p.type, "GPU");
    EXPECT_EQ(p.id, 3);
  }
  {
    // Allow _ in job names.
    DeviceNameUtils::ParsedName p;
    EXPECT_TRUE(DeviceNameUtils::ParseFullName(
        "/job:foo_bar/replica:1/task:2/device:GPU:3", &p));
    EXPECT_TRUE(DeviceNameUtils::ParseFullOrLocalName(
        "/job:foo_bar/replica:1/task:2/device:GPU:3", &p));
    EXPECT_TRUE(p.has_job);
    EXPECT_TRUE(p.has_replica);
    EXPECT_TRUE(p.has_task);
    EXPECT_TRUE(p.has_type);
    EXPECT_TRUE(p.has_id);
    EXPECT_EQ(p.job, "foo_bar");
    EXPECT_EQ(p.replica, 1);
    EXPECT_EQ(p.task, 2);
    EXPECT_EQ(p.type, "GPU");
    EXPECT_EQ(p.id, 3);
  }
  {
    // Allow _ in job names.
    DeviceNameUtils::ParsedName p;
    EXPECT_TRUE(DeviceNameUtils::ParseFullName(
        "/job:foo_bar/replica:1/task:2/device:GPU:3", &p));
    EXPECT_TRUE(p.has_job);
    EXPECT_TRUE(p.has_replica);
    EXPECT_TRUE(p.has_task);
    EXPECT_TRUE(p.has_type);
    EXPECT_TRUE(p.has_id);
    EXPECT_EQ(p.job, "foo_bar");
    EXPECT_EQ(p.replica, 1);
    EXPECT_EQ(p.task, 2);
    EXPECT_EQ(p.type, "GPU");
    EXPECT_EQ(p.id, 3);
  }
  {
    DeviceNameUtils::ParsedName p;
    EXPECT_TRUE(DeviceNameUtils::ParseFullName("/job:*/replica:4/gpu:*", &p));
    EXPECT_FALSE(p.has_job);
    EXPECT_TRUE(p.has_replica);
    EXPECT_FALSE(p.has_task);
    EXPECT_TRUE(p.has_type);
    EXPECT_FALSE(p.has_id);
    EXPECT_EQ(p.replica, 4);
    EXPECT_EQ(p.type, "GPU");
  }
  {
    DeviceNameUtils::ParsedName p;
    EXPECT_TRUE(
        DeviceNameUtils::ParseFullName("/job:*/replica:4/device:GPU:*", &p));
    EXPECT_FALSE(p.has_job);
    EXPECT_TRUE(p.has_replica);
    EXPECT_FALSE(p.has_task);
    EXPECT_TRUE(p.has_type);
    EXPECT_FALSE(p.has_id);
    EXPECT_EQ(p.replica, 4);
    EXPECT_EQ(p.type, "GPU");
  }
  {
    DeviceNameUtils::ParsedName p;
    EXPECT_TRUE(
        DeviceNameUtils::ParseFullName("/job:*/device:GPU/replica:4", &p));
    EXPECT_FALSE(p.has_job);
    EXPECT_TRUE(p.has_replica);
    EXPECT_FALSE(p.has_task);
    EXPECT_TRUE(p.has_type);
    EXPECT_FALSE(p.has_id);
    EXPECT_EQ(p.replica, 4);
    EXPECT_EQ(p.type, "GPU");
  }
  {
    DeviceNameUtils::ParsedName p;
    EXPECT_TRUE(DeviceNameUtils::ParseFullName(
        "/job:*/replica:4/device:myspecialdevice:13", &p));
    EXPECT_FALSE(p.has_job);
    EXPECT_TRUE(p.has_replica);
    EXPECT_FALSE(p.has_task);
    EXPECT_TRUE(p.has_type);
    EXPECT_TRUE(p.has_id);
    EXPECT_EQ(p.replica, 4);
    EXPECT_EQ(p.type, "myspecialdevice");
    EXPECT_EQ(p.id, 13);
  }
  {
    DeviceNameUtils::ParsedName p;
    EXPECT_TRUE(DeviceNameUtils::ParseFullName("/", &p));
    EXPECT_FALSE(p.has_job);
    EXPECT_FALSE(p.has_replica);
    EXPECT_FALSE(p.has_task);
    EXPECT_FALSE(p.has_type);
    EXPECT_FALSE(p.has_id);
  }
  {
    DeviceNameUtils::ParsedName p;
    EXPECT_TRUE(
        DeviceNameUtils::ParseFullName("/job:*/replica:4/device:GPU:5", &p));
    EXPECT_FALSE(p.has_job);
    EXPECT_TRUE(p.has_replica);
    EXPECT_FALSE(p.has_task);
    EXPECT_TRUE(p.has_type);
    EXPECT_TRUE(p.has_id);
    EXPECT_EQ(p.replica, 4);
    EXPECT_EQ(p.type, "GPU");
    EXPECT_EQ(p.id, 5);
  }
  {  // Same result if we reorder the components
    DeviceNameUtils::ParsedName p;
    EXPECT_TRUE(DeviceNameUtils::ParseFullName("/gpu:*/job:*/replica:4", &p));
    EXPECT_FALSE(p.has_job);
    EXPECT_TRUE(p.has_replica);
    EXPECT_FALSE(p.has_task);
    EXPECT_TRUE(p.has_type);
    EXPECT_FALSE(p.has_id);
    EXPECT_EQ(p.replica, 4);
    EXPECT_EQ(p.type, "GPU");
  }

  EXPECT_TRUE(DeviceNameUtils::IsSameAddressSpace(
      "/job:foo/replica:1/task:2/cpu:3",
      "/job:foo/replica:1/task:2/device:GPU:4"));
  EXPECT_FALSE(DeviceNameUtils::IsSameAddressSpace(
      "/job:foo/replica:1/task:2/cpu:3",
      "/job:foo/replica:1/task:3/device:GPU:4"));
  EXPECT_FALSE(DeviceNameUtils::IsSameAddressSpace(
      "/job:foo/replica:1/task:2/cpu:3",
      "/job:foo/replica:10/task:2/device:GPU:4"));
  EXPECT_FALSE(DeviceNameUtils::IsSameAddressSpace(
      "/job:foo/replica:1/task:2/cpu:3",
      "/job:bar/replica:1/task:2/device:GPU:4"));

  EXPECT_EQ(DeviceNameUtils::LocalName("CPU", 1), "/device:CPU:1");
  EXPECT_EQ(DeviceNameUtils::LocalName("GPU", 2), "/device:GPU:2");
  EXPECT_EQ(DeviceNameUtils::LocalName("MySpecialDevice", 13),
            "/device:MySpecialDevice:13");

  EXPECT_EQ(
      DeviceNameUtils::LocalName("/job:foo/replica:1/task:2/device:CPU:3"),
      "/device:CPU:3");

  EXPECT_EQ(DeviceNameUtils::LocalName("/job:foo/replica:1/task:2/cpu:3"),
            "/device:CPU:3");

  EXPECT_EQ(
      DeviceNameUtils::LocalName("/job:foo/replica:1/task:2/device:abc:73"),
      "/device:abc:73");

  {
    DeviceNameUtils::ParsedName p;
    EXPECT_TRUE(DeviceNameUtils::ParseLocalName("CPU:10", &p));
    EXPECT_TRUE(DeviceNameUtils::ParseFullOrLocalName("CPU:10", &p));
    EXPECT_EQ(p.type, "CPU");
    EXPECT_EQ(p.id, 10);
    EXPECT_FALSE(DeviceNameUtils::ParseLocalName("cpu:abc", &p));
    EXPECT_FALSE(DeviceNameUtils::ParseLocalName("abc:", &p));
    EXPECT_FALSE(DeviceNameUtils::ParseLocalName("abc", &p));
    EXPECT_FALSE(DeviceNameUtils::ParseLocalName("myspecialdevice", &p));
    EXPECT_FALSE(DeviceNameUtils::ParseFullOrLocalName("myspecialdevice", &p));
  }

  // Test that all parts are round-tripped correctly.
  {
    for (int i = 0; i < 0x10; ++i) {
      EXPECT_TRUE(RoundTripPartialName(i, {"foo", "3", "2", "CPU:3"},
                                       /*explicitDevice=*/false));
      EXPECT_TRUE(RoundTripPartialName(i, {"foo", "3", "2", "GPU:3"},
                                       /*explicitDevice=*/false));
      EXPECT_TRUE(RoundTripPartialName(i, {"foo", "3", "2", "cpu:3"},
                                       /*explicitDevice=*/false));
      EXPECT_TRUE(RoundTripPartialName(i, {"foo", "3", "2", "gpu:3"},
                                       /*explicitDevice=*/false));
      EXPECT_TRUE(RoundTripPartialName(i, {"foo", "3", "2", "CPU:3"},
                                       /*explicitDevice=*/true));
      EXPECT_TRUE(RoundTripPartialName(i, {"foo", "3", "2", "GPU:3"},
                                       /*explicitDevice=*/true));
      EXPECT_TRUE(RoundTripPartialName(i, {"foo", "3", "2", "cpu:3"},
                                       /*explicitDevice=*/true));
      EXPECT_TRUE(RoundTripPartialName(i, {"foo", "3", "2", "gpu:3"},
                                       /*explicitDevice=*/true));
      EXPECT_TRUE(RoundTripPartialName(i, {"foo", "3", "2", "someDevice:3"},
                                       /*explicitDevice=*/true));
    }
  }
  {
    DeviceNameUtils::ParsedName x, y;
    DeviceNameUtils::ParseFullName("/job:work/replica:1/task:3/device:GPU:*",
                                   &x);
    DeviceNameUtils::ParseFullName("/device:CPU:*", &y);
    EXPECT_FALSE(DeviceNameUtils::AreCompatibleDevNames(x, y));
  }
  {
    DeviceNameUtils::ParsedName x, y;
    DeviceNameUtils::ParseFullName("/job:work/replica:1/task:3", &x);
    DeviceNameUtils::ParseFullName("/device:CPU:*", &y);
    EXPECT_TRUE(DeviceNameUtils::AreCompatibleDevNames(x, y));
  }
}

static bool IsCSHelper(StringPiece pattern, StringPiece actual) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utils_testDTcc mht_2(mht_2_v, 476, "", "./tensorflow/core/util/device_name_utils_test.cc", "IsCSHelper");

  DeviceNameUtils::ParsedName p, a;
  EXPECT_TRUE(DeviceNameUtils::ParseFullName(pattern, &p));
  EXPECT_TRUE(DeviceNameUtils::ParseFullName(actual, &a));
  return DeviceNameUtils::IsCompleteSpecification(p, a);
}

TEST(DeviceNameUtilsTest, IsCompleteSpecification) {
  EXPECT_TRUE(IsCSHelper("/job:*", "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsCSHelper("/job:*/replica:*",
                         "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(
      IsCSHelper("/job:*/task:*", "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsCSHelper("/job:*/replica:*/task:*",
                         "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsCSHelper("/job:*/replica:*/gpu:*",
                         "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_FALSE(
      IsCSHelper("/cpu:*", "/job:worker/replica:1/task:2/device:GPU:3"));
  EXPECT_FALSE(
      IsCSHelper("/device:GPU:2", "/job:worker/replica:1/task:2/device:GPU:1"));
  EXPECT_TRUE(
      IsCSHelper("/gpu:*", "/job:worker/replica:1/task:2/device:GPU:3"));
}

static bool IsSpecHelper(StringPiece pattern, StringPiece actual) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utils_testDTcc mht_3(mht_3_v, 504, "", "./tensorflow/core/util/device_name_utils_test.cc", "IsSpecHelper");

  DeviceNameUtils::ParsedName p, a;
  EXPECT_TRUE(DeviceNameUtils::ParseFullName(pattern, &p));
  EXPECT_TRUE(DeviceNameUtils::ParseFullName(actual, &a));
  return DeviceNameUtils::IsSpecification(p, a);
}

TEST(DeviceNameUtilsTest, IsSpecification) {
  EXPECT_TRUE(
      IsSpecHelper("/job:*", "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsSpecHelper("/job:*", "/job:work/replica:1/device:GPU:3"));
  EXPECT_TRUE(IsSpecHelper("/job:*", "/job:work/replica:1"));
  EXPECT_TRUE(IsSpecHelper("/job:*", "/replica:1"));
  EXPECT_TRUE(IsSpecHelper("/job:*", "/job:work"));
  EXPECT_TRUE(IsSpecHelper("/job:*/replica:*",
                           "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsSpecHelper("/job:work/replica:1/gpu:*",
                           "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsSpecHelper("/job:work/replica:1/device:GPU:3",
                           "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsSpecHelper("/job:work/replica:1/task:2",
                           "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsSpecHelper("/job:work/replica:*/task:2",
                           "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsSpecHelper("/task:*", "/job:*/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsSpecHelper("/task:2", "/job:*/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsSpecHelper("/cpu:*", "/job:*/replica:1/task:2/cpu:1"));
  EXPECT_TRUE(IsSpecHelper("/cpu:0", "/cpu:0"));
  EXPECT_TRUE(
      IsSpecHelper("/gpu:*", "/job:worker/replica:1/task:2/device:GPU:3"));

  EXPECT_FALSE(
      IsSpecHelper("/job:worker/replica:1/task:2/device:GPU:3", "/gpu:*"));
  EXPECT_FALSE(IsSpecHelper("/cpu:*", "/job:*/replica:1/task:2"));
  EXPECT_FALSE(IsSpecHelper("/cpu:*", "/job:*/replica:1/task:2/device:GPU:1"));
  EXPECT_FALSE(
      IsSpecHelper("/cpu:*", "/job:worker/replica:1/task:2/device:GPU:3"));
  EXPECT_FALSE(IsSpecHelper("/device:GPU:2",
                            "/job:worker/replica:1/task:2/device:GPU:1"));
  EXPECT_FALSE(IsSpecHelper("/job:work/replica:*/task:0",
                            "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_FALSE(IsSpecHelper("/job:work/replica:0/task:2",
                            "/job:work/replica:*/task:2/device:GPU:3"));
}

TEST(DeviceNameUtilsTest, SplitDeviceName) {
  string task;
  string device;
  EXPECT_TRUE(DeviceNameUtils::SplitDeviceName(
      "/job:foo/replica:1/task:2/cpu:1", &task, &device));
  EXPECT_EQ("/job:foo/replica:1/task:2", task);
  EXPECT_EQ("CPU:1", device);
  EXPECT_TRUE(DeviceNameUtils::SplitDeviceName(
      "/job:foo/cpu:1/task:2/replica:1", &task, &device));
  EXPECT_EQ("/job:foo/replica:1/task:2", task);
  EXPECT_EQ("CPU:1", device);
  EXPECT_TRUE(
      DeviceNameUtils::SplitDeviceName("/device:GPU:3", &task, &device));
  EXPECT_EQ("", task);
  EXPECT_EQ("GPU:3", device);
  EXPECT_FALSE(DeviceNameUtils::SplitDeviceName("gpu:3", &task, &device));
  EXPECT_FALSE(DeviceNameUtils::SplitDeviceName("/job:foo/task:2/replica:1",
                                                &task, &device));
  EXPECT_TRUE(DeviceNameUtils::SplitDeviceName("/device:myspecialdevice:3",
                                               &task, &device));
  EXPECT_EQ("", task);
  EXPECT_EQ("myspecialdevice:3", device);
}

static DeviceNameUtils::ParsedName Name(const string& str) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("str: \"" + str + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utils_testDTcc mht_4(mht_4_v, 577, "", "./tensorflow/core/util/device_name_utils_test.cc", "Name");

  DeviceNameUtils::ParsedName ret;
  CHECK(DeviceNameUtils::ParseFullName(str, &ret)) << "Invalid name: " << str;
  return ret;
}

static void MergeDevNamesHelperImpl(const string& name_a, const string& name_b,
                                    const string& expected_merge_name,
                                    bool allow_soft_placement) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name_a: \"" + name_a + "\"");
   mht_5_v.push_back("name_b: \"" + name_b + "\"");
   mht_5_v.push_back("expected_merge_name: \"" + expected_merge_name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utils_testDTcc mht_5(mht_5_v, 591, "", "./tensorflow/core/util/device_name_utils_test.cc", "MergeDevNamesHelperImpl");

  DeviceNameUtils::ParsedName target_a = Name(name_a);
  TF_EXPECT_OK(DeviceNameUtils::MergeDevNames(&target_a, Name(name_b),
                                              allow_soft_placement));
  DeviceNameUtils::ParsedName target_b = Name(name_b);
  TF_EXPECT_OK(DeviceNameUtils::MergeDevNames(&target_b, Name(name_a),
                                              allow_soft_placement));
  EXPECT_EQ(target_a, target_b);
  EXPECT_EQ(target_a, Name(expected_merge_name));
  EXPECT_EQ(target_b, Name(expected_merge_name));
}

static void MergeDevNamesHelper(const string& name_a, const string& name_b,
                                const string& expected_merge_name) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name_a: \"" + name_a + "\"");
   mht_6_v.push_back("name_b: \"" + name_b + "\"");
   mht_6_v.push_back("expected_merge_name: \"" + expected_merge_name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utils_testDTcc mht_6(mht_6_v, 610, "", "./tensorflow/core/util/device_name_utils_test.cc", "MergeDevNamesHelper");

  MergeDevNamesHelperImpl(name_a, name_b, expected_merge_name, false);
}

static void MergeDevNamesHelperAllowSoftPlacement(
    const string& name_a, const string& name_b,
    const string& expected_merge_name) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name_a: \"" + name_a + "\"");
   mht_7_v.push_back("name_b: \"" + name_b + "\"");
   mht_7_v.push_back("expected_merge_name: \"" + expected_merge_name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utils_testDTcc mht_7(mht_7_v, 622, "", "./tensorflow/core/util/device_name_utils_test.cc", "MergeDevNamesHelperAllowSoftPlacement");

  MergeDevNamesHelperImpl(name_a, name_b, expected_merge_name, true);
}

static void MergeDevNamesError(const string& name_a, const string& name_b,
                               const string& expected_error_substr) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("name_a: \"" + name_a + "\"");
   mht_8_v.push_back("name_b: \"" + name_b + "\"");
   mht_8_v.push_back("expected_error_substr: \"" + expected_error_substr + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utils_testDTcc mht_8(mht_8_v, 633, "", "./tensorflow/core/util/device_name_utils_test.cc", "MergeDevNamesError");

  DeviceNameUtils::ParsedName target_a = Name(name_a);
  Status s = DeviceNameUtils::MergeDevNames(&target_a, Name(name_b));
  EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
  EXPECT_TRUE(absl::StrContains(s.error_message(), expected_error_substr)) << s;
}

static void MergeOverrideHelper(const string& target, const string& name,
                                const string& expected_merge_name) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("target: \"" + target + "\"");
   mht_9_v.push_back("name: \"" + name + "\"");
   mht_9_v.push_back("expected_merge_name: \"" + expected_merge_name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utils_testDTcc mht_9(mht_9_v, 647, "", "./tensorflow/core/util/device_name_utils_test.cc", "MergeOverrideHelper");

  DeviceNameUtils::ParsedName parsed_target = Name(target);
  TF_EXPECT_OK(
      DeviceNameUtils::MergeOverrideDevNames(&parsed_target, Name(name)));
  DeviceNameUtils::ParsedName parsed_expected = Name(expected_merge_name);

  EXPECT_EQ(parsed_target, parsed_expected)
      << "parsed_target: " << DeviceNameUtils::ParsedNameToString(parsed_target)
      << " expected_name: "
      << DeviceNameUtils::ParsedNameToString(parsed_expected);
}

static void MergeUnsetDevNamesHelper(const string& name_a, const string& name_b,
                                     const string& expected_merge_name_ab,
                                     const string& expected_merge_name_ba) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name_a: \"" + name_a + "\"");
   mht_10_v.push_back("name_b: \"" + name_b + "\"");
   mht_10_v.push_back("expected_merge_name_ab: \"" + expected_merge_name_ab + "\"");
   mht_10_v.push_back("expected_merge_name_ba: \"" + expected_merge_name_ba + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utils_testDTcc mht_10(mht_10_v, 668, "", "./tensorflow/core/util/device_name_utils_test.cc", "MergeUnsetDevNamesHelper");

  DeviceNameUtils::ParsedName target_a = Name(name_a);
  DeviceNameUtils::MergeUnsetDevNames(&target_a, Name(name_b));
  EXPECT_EQ(target_a, Name(expected_merge_name_ab));
  DeviceNameUtils::ParsedName target_b = Name(name_b);
  DeviceNameUtils::MergeUnsetDevNames(&target_b, Name(name_a));
  EXPECT_EQ(target_b, Name(expected_merge_name_ba));
}

TEST(DeviceNameUtilsTest, MergeDevNames) {
  // Idempotence tests.
  MergeDevNamesHelper("", "", "");
  MergeDevNamesHelper("/job:foo/replica:1/task:2/cpu:1",
                      "/job:foo/replica:1/task:2/cpu:1",
                      "/job:foo/replica:1/task:2/cpu:1");

  // Merging with empty device has no effect.
  MergeDevNamesHelper("", "/job:foo", "/job:foo");
  MergeDevNamesHelper("", "/replica:2", "/replica:2");
  MergeDevNamesHelper("", "/task:7", "/task:7");
  MergeDevNamesHelper("", "/device:GPU:1", "/device:GPU:1");

  // Combining disjoint names.
  MergeDevNamesHelper("/job:foo", "/task:7", "/job:foo/task:7");
  MergeDevNamesHelper("/job:foo", "/device:GPU:1", "/job:foo/device:GPU:1");

  // Combining overlapping names.
  MergeDevNamesHelper("/job:foo/replica:0", "/replica:0/task:1",
                      "/job:foo/replica:0/task:1");

  // Wildcard tests.
  MergeDevNamesHelper("", "/gpu:*", "/gpu:*");
  MergeDevNamesHelper("/gpu:*", "/gpu:*", "/gpu:*");
  MergeDevNamesHelper("/device:GPU:1", "/gpu:*", "/device:GPU:1");

  // Incompatible components.
  MergeDevNamesError("/job:foo", "/job:bar", "incompatible jobs");
  MergeDevNamesError("/replica:0", "/replica:1", "incompatible replicas");
  MergeDevNamesError("/task:0", "/task:1", "incompatible tasks");
  MergeDevNamesError("/gpu:*", "/cpu:*", "incompatible types");
  MergeDevNamesError("/device:GPU:0", "/device:GPU:1", "incompatible ids");
}

TEST(DeviceNameUtilsTest, MergeDevNamesAllowSoftPlacement) {
  // Incompatible components with allow_soft_placement.
  MergeDevNamesHelperAllowSoftPlacement("/gpu:*", "/cpu:1", "");
  MergeDevNamesHelperAllowSoftPlacement("/cpu:*", "/device:GPU:1", "");
  MergeDevNamesHelperAllowSoftPlacement("/device:GPU:1", "/device:GPU:2",
                                        "/device:GPU:*");
}

TEST(DeviceNameUtilsTest, MergeOverrideDevNames) {
  // Idempotence tests.
  MergeOverrideHelper("", "", "");
  MergeOverrideHelper("/job:foo/replica:1/task:2/cpu:1",
                      "/job:foo/replica:1/task:2/cpu:1",
                      "/job:foo/replica:1/task:2/cpu:1");

  // Merging with empty device has no effect.
  MergeOverrideHelper("", "/job:foo", "/job:foo");
  MergeOverrideHelper("", "/replica:2", "/replica:2");
  MergeOverrideHelper("", "/task:7", "/task:7");
  MergeOverrideHelper("", "/device:GPU:1", "/device:GPU:1");

  // Combining disjoint names.
  MergeOverrideHelper("/job:foo", "/task:7", "/job:foo/task:7");
  MergeOverrideHelper("/job:foo", "/device:GPU:1", "/job:foo/device:GPU:1");

  // Combining overlapping names.
  MergeOverrideHelper("/job:foo/replica:0", "/replica:0/task:1",
                      "/job:foo/replica:0/task:1");

  // Wildcard tests.
  MergeOverrideHelper("", "/gpu:*", "/gpu:*");
  MergeOverrideHelper("/gpu:*", "/gpu:*", "/gpu:*");
  MergeOverrideHelper("/device:GPU:1", "/gpu:*", "/device:GPU:1");

  // Testing actual override functionality
  MergeOverrideHelper("/gpu:0", "/cpu:1", "/cpu:1");
  MergeOverrideHelper("/gpu:*", "/cpu:1", "/cpu:1");
  MergeOverrideHelper("/cpu:*", "/device:GPU:1", "/gpu:1");
  MergeOverrideHelper("/device:GPU:1", "/device:GPU:2", "/device:GPU:2");

  // Override with regular merging
  MergeOverrideHelper("/job:foo/CPU:*", "/device:GPU:1", "/job:foo/GPU:1");
  MergeOverrideHelper("/cpu:*", "/job:foo/device:GPU:1", "/job:foo/GPU:1");
  MergeOverrideHelper("/task:0/cpu:*", "/device:GPU:1", "/task:0/GPU:1");
  MergeOverrideHelper("/cpu:*", "/task:0/device:GPU:1", "/task:0/GPU:1");
}

TEST(DeviceNameUtilsTest, MergeUnsetDevNames) {
  // Idempotence tests.
  MergeUnsetDevNamesHelper("", "", "", "");
  MergeUnsetDevNamesHelper(
      "/job:foo/replica:1/task:2/cpu:1", "/job:foo/replica:1/task:2/cpu:1",
      "/job:foo/replica:1/task:2/cpu:1", "/job:foo/replica:1/task:2/cpu:1");

  // Merging with empty device has no effect.
  MergeUnsetDevNamesHelper("", "/job:foo", "/job:foo", "/job:foo");
  MergeUnsetDevNamesHelper("", "/replica:2", "/replica:2", "/replica:2");
  MergeUnsetDevNamesHelper("", "/task:7", "/task:7", "/task:7");
  MergeUnsetDevNamesHelper("", "/device:GPU:1", "/device:GPU:1",
                           "/device:GPU:1");

  // Combining disjoint names.
  MergeUnsetDevNamesHelper("/job:foo", "/task:7", "/job:foo/task:7",
                           "/job:foo/task:7");
  MergeUnsetDevNamesHelper("/job:foo", "/device:GPU:1", "/job:foo/device:GPU:1",
                           "/job:foo/device:GPU:1");

  // Combining overlapping names.
  MergeUnsetDevNamesHelper("/job:foo/replica:0", "/replica:0/task:1",
                           "/job:foo/replica:0/task:1",
                           "/job:foo/replica:0/task:1");

  // Wildcard tests.
  MergeUnsetDevNamesHelper("", "/gpu:*", "/gpu:*", "/gpu:*");
  MergeUnsetDevNamesHelper("/gpu:*", "/gpu:*", "/gpu:*", "/gpu:*");
  MergeUnsetDevNamesHelper("/device:GPU:1", "/gpu:*", "/device:GPU:1",
                           "/device:GPU:1");

  // Incompatible components.
  MergeUnsetDevNamesHelper("/job:foo", "/job:bar", "/job:foo", "/job:bar");
  MergeUnsetDevNamesHelper("/replica:0", "/replica:1", "/replica:0",
                           "/replica:1");
  MergeUnsetDevNamesHelper("/task:0", "/task:1", "/task:0", "/task:1");
  MergeUnsetDevNamesHelper("/gpu:*", "/cpu:*", "/gpu:*", "/cpu:*");
  MergeUnsetDevNamesHelper("/device:GPU:0", "/device:GPU:1", "/device:GPU:0",
                           "/device:GPU:1");
  MergeUnsetDevNamesHelper("/job:foo/device:GPU", "/job:bar",
                           "/job:foo/device:GPU", "/job:bar/device:GPU");
}

TEST(DeviceNameUtilsTest, GetNamesForDeviceMappings) {
  DeviceNameUtils::ParsedName p =
      Name("/job:foo/replica:10/task:0/device:GPU:1");
  EXPECT_EQ(absl::StrJoin(DeviceNameUtils::GetNamesForDeviceMappings(p), ","),
            "/job:foo/replica:10/task:0/device:GPU:1,"
            "/job:foo/replica:10/task:0/gpu:1");
  p.has_task = false;
  EXPECT_EQ(absl::StrJoin(DeviceNameUtils::GetNamesForDeviceMappings(p), ","),
            "");
}

TEST(DeviceNameUtilsTest, CanonicalizeDeviceName) {
  string canonical_name;
  {
    // Good basename.
    string basename = "/job:foo/replica:10/task:0/device:CPU:0";
    TF_EXPECT_OK(DeviceNameUtils::CanonicalizeDeviceName(
        "/job:foo/replica:10/task:0/device:CPU:1", basename, &canonical_name));
    EXPECT_EQ("/job:foo/replica:10/task:0/device:CPU:1", canonical_name);
    TF_EXPECT_OK(DeviceNameUtils::CanonicalizeDeviceName(
        "/job:foo/task:0/replica:10/device:CPU:1", basename, &canonical_name));
    EXPECT_EQ("/job:foo/replica:10/task:0/device:CPU:1", canonical_name);
    TF_EXPECT_OK(DeviceNameUtils::CanonicalizeDeviceName(
        "/job:foo/task:0/replica:10/cpu:1", basename, &canonical_name));
    EXPECT_EQ("/job:foo/replica:10/task:0/device:CPU:1", canonical_name);
    TF_EXPECT_OK(DeviceNameUtils::CanonicalizeDeviceName("CPU:0", basename,
                                                         &canonical_name));
    EXPECT_EQ("/job:foo/replica:10/task:0/device:CPU:0", canonical_name);
    Status s = DeviceNameUtils::CanonicalizeDeviceName(
        "/job:foo/task:0/replica/cpu:1", basename, &canonical_name);
    EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
    EXPECT_EQ("", canonical_name);
  }

  {
    // Try out malformed basenames.
    string fullname = "/device:CPU:0";

    Status s = DeviceNameUtils::CanonicalizeDeviceName(
        fullname, "/device:CPU:0", &canonical_name);
    EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
    EXPECT_EQ("", canonical_name);
    s = DeviceNameUtils::CanonicalizeDeviceName(
        fullname, "/job:foo/task:0/replica/cpu:1", &canonical_name);
    EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
    EXPECT_EQ("", canonical_name);
  }
}

static void BM_ParseFullName(::testing::benchmark::State& state) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utils_testDTcc mht_11(mht_11_v, 853, "", "./tensorflow/core/util/device_name_utils_test.cc", "BM_ParseFullName");

  DeviceNameUtils::ParsedName p;
  for (auto s : state) {
    DeviceNameUtils::ParseFullName("/job:worker/replica:3/task:0/cpu:0", &p);
  }
}
BENCHMARK(BM_ParseFullName);

}  // namespace tensorflow
