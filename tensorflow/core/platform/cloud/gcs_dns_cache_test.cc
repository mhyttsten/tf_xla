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
class MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/cloud/gcs_dns_cache.h"

#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class TestHttpRequest : public HttpRequest {
 public:
  void SetUri(const string& uri) override {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("uri: \"" + uri + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "SetUri");
}
  void SetRange(uint64 start, uint64 end) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_1(mht_1_v, 199, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "SetRange");
}
  void AddHeader(const string& name, const string& value) override {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   mht_2_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_2(mht_2_v, 205, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "AddHeader");
}
  void AddResolveOverride(const string& hostname, int64_t port,
                          const string& ip_addr) override {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("hostname: \"" + hostname + "\"");
   mht_3_v.push_back("ip_addr: \"" + ip_addr + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_3(mht_3_v, 212, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "AddResolveOverride");

    EXPECT_EQ(port, 443) << "Unexpected port set for hostname: " << hostname;
    auto itr = resolve_overrides_.find(hostname);
    EXPECT_EQ(itr, resolve_overrides_.end())
        << "Hostname " << hostname << "already in map: " << itr->second;

    resolve_overrides_.insert(
        std::map<string, string>::value_type(hostname, ip_addr));
  }

  void AddAuthBearerHeader(const string& auth_token) override {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("auth_token: \"" + auth_token + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_4(mht_4_v, 226, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "AddAuthBearerHeader");
}
  void SetRequestStats(HttpRequest::RequestStats* stats) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_5(mht_5_v, 230, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "SetRequestStats");
}
  void SetDeleteRequest() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_6(mht_6_v, 234, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "SetDeleteRequest");
}

  Status SetPutFromFile(const string& body_filepath, size_t offset) override {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("body_filepath: \"" + body_filepath + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_7(mht_7_v, 240, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "SetPutFromFile");

    return Status::OK();
  }
  void SetPutEmptyBody() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_8(mht_8_v, 246, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "SetPutEmptyBody");
}
  void SetPostFromBuffer(const char* buffer, size_t size) override {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_9(mht_9_v, 251, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "SetPostFromBuffer");
}
  void SetPostEmptyBody() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_10(mht_10_v, 255, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "SetPostEmptyBody");
}
  void SetResultBuffer(std::vector<char>* out_buffer) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_11(mht_11_v, 259, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "SetResultBuffer");
}
  void SetResultBufferDirect(char* buffer, size_t size) override {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_12(mht_12_v, 264, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "SetResultBufferDirect");
}
  size_t GetResultBufferDirectBytesTransferred() override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_13(mht_13_v, 268, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "GetResultBufferDirectBytesTransferred");
 return 0; }

  string GetResponseHeader(const string& name) const override {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_14(mht_14_v, 274, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "GetResponseHeader");
 return ""; }
  uint64 GetResponseCode() const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_15(mht_15_v, 278, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "GetResponseCode");
 return 0; }
  Status Send() override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_16(mht_16_v, 282, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "Send");
 return Status::OK(); }
  string EscapeString(const string& str) override {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("str: \"" + str + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_17(mht_17_v, 287, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "EscapeString");
 return ""; }

  void SetTimeouts(uint32 connection, uint32 inactivity,
                   uint32 total) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_18(mht_18_v, 293, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "SetTimeouts");
}

  std::map<string, string> resolve_overrides_;
};

// Friend class for testing.
//
// It is written this way (as opposed to using FRIEND_TEST) to avoid a
// non-test-time dependency on gunit.
class GcsDnsCacheTest : public ::testing::Test {
 protected:
  void ResolveNameTest() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_19(mht_19_v, 307, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "ResolveNameTest");

    auto response = GcsDnsCache::ResolveName("www.googleapis.com");
    EXPECT_LT(1, response.size()) << absl::StrJoin(response, ", ");
  }

  void AnnotateRequestTest() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_20(mht_20_v, 315, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "AnnotateRequestTest");

    GcsDnsCache d;
    {
      mutex_lock l(d.mu_);
      d.started_ = true;  // Avoid creating a thread.
      d.addresses_ = {{"192.168.1.1"}, {"172.134.1.1"}};
    }

    TestHttpRequest req;
    d.AnnotateRequest(&req);
    EXPECT_EQ("192.168.1.1", req.resolve_overrides_["www.googleapis.com"]);
    EXPECT_EQ("172.134.1.1", req.resolve_overrides_["storage.googleapis.com"]);
  }

  void SuccessfulCleanupTest() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cache_testDTcc mht_21(mht_21_v, 332, "", "./tensorflow/core/platform/cloud/gcs_dns_cache_test.cc", "SuccessfulCleanupTest");

    // Create a DnsCache object, start the worker thread, ensure it cleans up in
    // a timely manner.
    GcsDnsCache d;
    TestHttpRequest req;
    d.AnnotateRequest(&req);
  }
};

// This sends a DNS name resolution request, thus it is flaky.
// TEST_F(GcsDnsCacheTest, ResolveName) { ResolveNameTest(); }

TEST_F(GcsDnsCacheTest, AnnotateRequest) { AnnotateRequestTest(); }

TEST_F(GcsDnsCacheTest, SuccessfulCleanup) { SuccessfulCleanupTest(); }

}  // namespace tensorflow
