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
class MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc() {
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

#include "tensorflow/core/platform/cloud/curl_http_request.h"

#include <fstream>
#include <string>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

const string kTestContent = "random original scratch content";

class FakeEnv : public EnvWrapper {
 public:
  FakeEnv() : EnvWrapper(Env::Default()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "FakeEnv");
}

  uint64 NowSeconds() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_1(mht_1_v, 207, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "NowSeconds");
 return now_; }
  uint64 now_ = 10000;
};

// A fake proxy that pretends to be libcurl.
class FakeLibCurl : public LibCurl {
 public:
  FakeLibCurl(const string& response_content, uint64 response_code)
      : response_content_(response_content), response_code_(response_code) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("response_content: \"" + response_content + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_2(mht_2_v, 219, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "FakeLibCurl");
}
  FakeLibCurl(const string& response_content, uint64 response_code,
              std::vector<std::tuple<uint64, curl_off_t>> progress_ticks,
              FakeEnv* env)
      : response_content_(response_content),
        response_code_(response_code),
        progress_ticks_(std::move(progress_ticks)),
        env_(env) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("response_content: \"" + response_content + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_3(mht_3_v, 230, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "FakeLibCurl");
}
  FakeLibCurl(const string& response_content, uint64 response_code,
              const std::vector<string>& response_headers)
      : response_content_(response_content),
        response_code_(response_code),
        response_headers_(response_headers) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("response_content: \"" + response_content + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_4(mht_4_v, 239, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "FakeLibCurl");
}
  CURL* curl_easy_init() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_5(mht_5_v, 243, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "curl_easy_init");

    is_initialized_ = true;
    // The reuslt just needs to be non-null.
    return reinterpret_cast<CURL*>(this);
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            uint64 param) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_6(mht_6_v, 252, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "curl_easy_setopt");

    switch (option) {
      case CURLOPT_POST:
        is_post_ = param;
        break;
      case CURLOPT_PUT:
        is_put_ = param;
        break;
      default:
        break;
    }
    return CURLE_OK;
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            const char* param) override {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("param: \"" + (param == nullptr ? std::string("nullptr") : std::string((char*)param)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_7(mht_7_v, 270, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "curl_easy_setopt");

    return curl_easy_setopt(curl, option,
                            reinterpret_cast<void*>(const_cast<char*>(param)));
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            void* param) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_8(mht_8_v, 278, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "curl_easy_setopt");

    switch (option) {
      case CURLOPT_URL:
        url_ = reinterpret_cast<char*>(param);
        break;
      case CURLOPT_RANGE:
        range_ = reinterpret_cast<char*>(param);
        break;
      case CURLOPT_CUSTOMREQUEST:
        custom_request_ = reinterpret_cast<char*>(param);
        break;
      case CURLOPT_HTTPHEADER:
        headers_ = reinterpret_cast<std::vector<string>*>(param);
        break;
      case CURLOPT_ERRORBUFFER:
        error_buffer_ = reinterpret_cast<char*>(param);
        break;
      case CURLOPT_CAINFO:
        ca_info_ = reinterpret_cast<char*>(param);
        break;
      case CURLOPT_WRITEDATA:
        write_data_ = reinterpret_cast<FILE*>(param);
        break;
      case CURLOPT_HEADERDATA:
        header_data_ = reinterpret_cast<FILE*>(param);
        break;
      case CURLOPT_READDATA:
        read_data_ = reinterpret_cast<FILE*>(param);
        break;
      case CURLOPT_XFERINFODATA:
        progress_data_ = param;
        break;
      default:
        break;
    }
    return CURLE_OK;
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            size_t (*param)(void*, size_t, size_t,
                                            FILE*)) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_9(mht_9_v, 320, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "curl_easy_setopt");

    read_callback_ = param;
    return CURLE_OK;
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            size_t (*param)(const void*, size_t, size_t,
                                            void*)) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_10(mht_10_v, 329, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "curl_easy_setopt");

    switch (option) {
      case CURLOPT_WRITEFUNCTION:
        write_callback_ = param;
        break;
      case CURLOPT_HEADERFUNCTION:
        header_callback_ = param;
        break;
      default:
        break;
    }
    return CURLE_OK;
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            int (*param)(void* clientp, curl_off_t dltotal,
                                         curl_off_t dlnow, curl_off_t ultotal,
                                         curl_off_t ulnow)) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_11(mht_11_v, 348, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "curl_easy_setopt");

    progress_callback_ = param;
    return CURLE_OK;
  }
  CURLcode curl_easy_perform(CURL* curl) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_12(mht_12_v, 355, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "curl_easy_perform");

    if (is_post_ || is_put_) {
      char buffer[3];
      int bytes_read;
      posted_content_ = "";
      do {
        bytes_read = read_callback_(buffer, 1, sizeof(buffer), read_data_);
        posted_content_ =
            strings::StrCat(posted_content_, StringPiece(buffer, bytes_read));
      } while (bytes_read > 0);
    }
    if (write_data_ || write_callback_) {
      size_t bytes_handled = write_callback_(
          response_content_.c_str(), 1, response_content_.size(), write_data_);
      // Mimic real libcurl behavior by checking write callback return value.
      if (bytes_handled != response_content_.size()) {
        curl_easy_perform_result_ = CURLE_WRITE_ERROR;
      }
    }
    for (const auto& header : response_headers_) {
      header_callback_(header.c_str(), 1, header.size(), header_data_);
    }
    if (error_buffer_) {
      strncpy(error_buffer_, curl_easy_perform_error_message_.c_str(),
              curl_easy_perform_error_message_.size() + 1);
    }
    for (const auto& tick : progress_ticks_) {
      env_->now_ = std::get<0>(tick);
      if (progress_callback_(progress_data_, 0, std::get<1>(tick), 0, 0)) {
        return CURLE_ABORTED_BY_CALLBACK;
      }
    }
    return curl_easy_perform_result_;
  }
  CURLcode curl_easy_getinfo(CURL* curl, CURLINFO info,
                             uint64* value) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_13(mht_13_v, 393, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "curl_easy_getinfo");

    switch (info) {
      case CURLINFO_RESPONSE_CODE:
        *value = response_code_;
        break;
      default:
        break;
    }
    return CURLE_OK;
  }
  CURLcode curl_easy_getinfo(CURL* curl, CURLINFO info,
                             double* value) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_14(mht_14_v, 407, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "curl_easy_getinfo");

    switch (info) {
      case CURLINFO_SIZE_DOWNLOAD:
        *value = response_content_.size();
        break;
      default:
        break;
    }
    return CURLE_OK;
  }
  void curl_easy_cleanup(CURL* curl) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_15(mht_15_v, 420, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "curl_easy_cleanup");
 is_cleaned_up_ = true; }
  curl_slist* curl_slist_append(curl_slist* list, const char* str) override {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("str: \"" + (str == nullptr ? std::string("nullptr") : std::string((char*)str)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_16(mht_16_v, 425, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "curl_slist_append");

    std::vector<string>* v = list ? reinterpret_cast<std::vector<string>*>(list)
                                  : new std::vector<string>();
    v->push_back(str);
    return reinterpret_cast<curl_slist*>(v);
  }
  char* curl_easy_escape(CURL* curl, const char* str, int length) override {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("str: \"" + (str == nullptr ? std::string("nullptr") : std::string((char*)str)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_17(mht_17_v, 435, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "curl_easy_escape");

    // This function just does a simple replacing of "/" with "%2F" instead of
    // full url encoding.
    const string victim = "/";
    const string encoded = "%2F";

    string temp_str = str;
    std::string::size_type n = 0;
    while ((n = temp_str.find(victim, n)) != std::string::npos) {
      temp_str.replace(n, victim.size(), encoded);
      n += encoded.size();
    }
    char* out_char_str = reinterpret_cast<char*>(
        port::Malloc(sizeof(char) * temp_str.size() + 1));
    std::copy(temp_str.begin(), temp_str.end(), out_char_str);
    out_char_str[temp_str.size()] = '\0';
    return out_char_str;
  }
  void curl_slist_free_all(curl_slist* list) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_18(mht_18_v, 456, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "curl_slist_free_all");

    delete reinterpret_cast<std::vector<string>*>(list);
  }
  void curl_free(void* p) override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_19(mht_19_v, 462, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "curl_free");
 port::Free(p); }

  // Variables defining the behavior of this fake.
  string response_content_;
  uint64 response_code_;
  std::vector<string> response_headers_;

  // Internal variables to store the libcurl state.
  string url_;
  string range_;
  string custom_request_;
  string ca_info_;
  char* error_buffer_ = nullptr;
  bool is_initialized_ = false;
  bool is_cleaned_up_ = false;
  std::vector<string>* headers_ = nullptr;
  bool is_post_ = false;
  bool is_put_ = false;
  void* write_data_ = nullptr;
  size_t (*write_callback_)(const void* ptr, size_t size, size_t nmemb,
                            void* userdata) = nullptr;
  void* header_data_ = nullptr;
  size_t (*header_callback_)(const void* ptr, size_t size, size_t nmemb,
                             void* userdata) = nullptr;
  FILE* read_data_ = nullptr;
  size_t (*read_callback_)(void* ptr, size_t size, size_t nmemb,
                           FILE* userdata) = &fread;
  int (*progress_callback_)(void* clientp, curl_off_t dltotal, curl_off_t dlnow,
                            curl_off_t ultotal, curl_off_t ulnow) = nullptr;
  void* progress_data_ = nullptr;
  // Outcome of performing the request.
  string posted_content_;
  CURLcode curl_easy_perform_result_ = CURLE_OK;
  string curl_easy_perform_error_message_;
  // A vector of <timestamp, progress in bytes> pairs that represent the
  // progress of a transmission.
  std::vector<std::tuple<uint64, curl_off_t>> progress_ticks_;
  FakeEnv* env_ = nullptr;
};

TEST(CurlHttpRequestTest, GetRequest) {
  FakeLibCurl libcurl("get response", 200);
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.insert(scratch.begin(), kTestContent.begin(), kTestContent.end());
  scratch.reserve(100);

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetRange(100, 199);
  http_request.SetResultBuffer(&scratch);
  TF_EXPECT_OK(http_request.Send());

  EXPECT_EQ("get response", string(scratch.begin(), scratch.end()));

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl.is_initialized_);
  EXPECT_EQ("http://www.testuri.com", libcurl.url_);
  EXPECT_EQ("100-199", libcurl.range_);
  EXPECT_EQ("", libcurl.custom_request_);
  EXPECT_EQ("", libcurl.ca_info_);
  EXPECT_EQ(1, libcurl.headers_->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl.headers_)[0]);
  EXPECT_FALSE(libcurl.is_post_);
  EXPECT_EQ(200, http_request.GetResponseCode());
}

TEST(CurlHttpRequestTest, GetRequest_Direct) {
  FakeLibCurl libcurl("get response", 200);
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch(100, 0);

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetRange(100, 199);
  http_request.SetResultBufferDirect(scratch.data(), scratch.capacity());
  TF_EXPECT_OK(http_request.Send());

  string expected_response = "get response";
  size_t response_bytes_transferred =
      http_request.GetResultBufferDirectBytesTransferred();
  EXPECT_EQ(expected_response.size(), response_bytes_transferred);
  EXPECT_EQ(
      "get response",
      string(scratch.begin(), scratch.begin() + response_bytes_transferred));

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl.is_initialized_);
  EXPECT_EQ("http://www.testuri.com", libcurl.url_);
  EXPECT_EQ("100-199", libcurl.range_);
  EXPECT_EQ("", libcurl.custom_request_);
  EXPECT_EQ("", libcurl.ca_info_);
  EXPECT_EQ(1, libcurl.headers_->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl.headers_)[0]);
  EXPECT_FALSE(libcurl.is_post_);
  EXPECT_EQ(200, http_request.GetResponseCode());
}

TEST(CurlHttpRequestTest, GetRequest_CustomCaInfoFlag) {
  static char set_var[] = "CURL_CA_BUNDLE=test";
  putenv(set_var);
  FakeLibCurl libcurl("get response", 200);
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.insert(scratch.begin(), kTestContent.begin(), kTestContent.end());
  scratch.reserve(100);

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetRange(100, 199);
  http_request.SetResultBuffer(&scratch);
  TF_EXPECT_OK(http_request.Send());

  EXPECT_EQ("get response", string(scratch.begin(), scratch.end()));

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl.is_initialized_);
  EXPECT_EQ("http://www.testuri.com", libcurl.url_);
  EXPECT_EQ("100-199", libcurl.range_);
  EXPECT_EQ("", libcurl.custom_request_);
  EXPECT_EQ("test", libcurl.ca_info_);
  EXPECT_EQ(1, libcurl.headers_->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl.headers_)[0]);
  EXPECT_FALSE(libcurl.is_post_);
  EXPECT_EQ(200, http_request.GetResponseCode());
}

TEST(CurlHttpRequestTest, GetRequest_Direct_ResponseTooLarge) {
  FakeLibCurl libcurl("get response", 200);
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch(5, 0);

  http_request.SetUri("http://www.testuri.com");
  http_request.SetResultBufferDirect(scratch.data(), scratch.size());
  const Status& status = http_request.Send();
  EXPECT_EQ(error::FAILED_PRECONDITION, status.code());
  EXPECT_EQ(
      "Error executing an HTTP request: libcurl code 23 meaning "
      "'Failed writing received data to disk/application', error details: "
      "Received 12 response bytes for a 5-byte buffer",
      status.error_message());

  // As long as the request clearly fails, ok to leave truncated response here.
  EXPECT_EQ(5, http_request.GetResultBufferDirectBytesTransferred());
  EXPECT_EQ("get r", string(scratch.begin(), scratch.begin() + 5));
}

TEST(CurlHttpRequestTest, GetRequest_Direct_RangeOutOfBound) {
  FakeLibCurl libcurl("get response", 416);
  CurlHttpRequest http_request(&libcurl);

  const string initialScratch = "abcde";
  std::vector<char> scratch;
  scratch.insert(scratch.end(), initialScratch.begin(), initialScratch.end());

  http_request.SetUri("http://www.testuri.com");
  http_request.SetRange(0, 4);
  http_request.SetResultBufferDirect(scratch.data(), scratch.size());
  TF_EXPECT_OK(http_request.Send());
  EXPECT_EQ(416, http_request.GetResponseCode());

  // Some servers (in particular, GCS) return an error message payload with a
  // 416 Range Not Satisfiable response. We should pretend it's not there when
  // reporting bytes transferred, but it's ok if it writes to scratch.
  EXPECT_EQ(0, http_request.GetResultBufferDirectBytesTransferred());
  EXPECT_EQ("get r", string(scratch.begin(), scratch.end()));
}

TEST(CurlHttpRequestTest, GetRequest_Empty) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.resize(0);

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetRange(100, 199);
  http_request.SetResultBuffer(&scratch);
  TF_EXPECT_OK(http_request.Send());

  EXPECT_TRUE(scratch.empty());

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl.is_initialized_);
  EXPECT_EQ("http://www.testuri.com", libcurl.url_);
  EXPECT_EQ("100-199", libcurl.range_);
  EXPECT_EQ("", libcurl.custom_request_);
  EXPECT_EQ(1, libcurl.headers_->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl.headers_)[0]);
  EXPECT_FALSE(libcurl.is_post_);
  EXPECT_EQ(200, http_request.GetResponseCode());
}

TEST(CurlHttpRequestTest, GetRequest_RangeOutOfBound) {
  FakeLibCurl libcurl("get response", 416);
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.insert(scratch.end(), kTestContent.begin(), kTestContent.end());

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetRange(100, 199);
  http_request.SetResultBuffer(&scratch);
  TF_EXPECT_OK(http_request.Send());

  // Some servers (in particular, GCS) return an error message payload with a
  // 416 Range Not Satisfiable response. We should pretend it's not there.
  EXPECT_TRUE(scratch.empty());
  EXPECT_EQ(416, http_request.GetResponseCode());
}

TEST(CurlHttpRequestTest, GetRequest_503) {
  FakeLibCurl libcurl("get response", 503);
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.insert(scratch.end(), kTestContent.begin(), kTestContent.end());

  http_request.SetUri("http://www.testuri.com");
  http_request.SetResultBuffer(&scratch);
  const auto& status = http_request.Send();
  EXPECT_EQ(error::UNAVAILABLE, status.code());
  EXPECT_EQ(
      "Error executing an HTTP request: HTTP response code 503 with body "
      "'get response'",
      status.error_message());
}

TEST(CurlHttpRequestTest, GetRequest_HttpCode0) {
  FakeLibCurl libcurl("get response", 0);
  libcurl.curl_easy_perform_result_ = CURLE_OPERATION_TIMEDOUT;
  libcurl.curl_easy_perform_error_message_ = "Operation timed out";
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.insert(scratch.end(), kTestContent.begin(), kTestContent.end());

  http_request.SetUri("http://www.testuri.com");
  const auto& status = http_request.Send();
  EXPECT_EQ(error::UNAVAILABLE, status.code());
  EXPECT_EQ(
      "Error executing an HTTP request: libcurl code 28 meaning "
      "'Timeout was reached', error details: Operation timed out",
      status.error_message());
  EXPECT_EQ(0, http_request.GetResponseCode());
}

TEST(CurlHttpRequestTest, GetRequest_CouldntResolveHost) {
  FakeLibCurl libcurl("get response", 0);
  libcurl.curl_easy_perform_result_ = CURLE_COULDNT_RESOLVE_HOST;
  libcurl.curl_easy_perform_error_message_ =
      "Could not resolve host 'metadata'";
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.insert(scratch.end(), kTestContent.begin(), kTestContent.end());

  http_request.SetUri("http://metadata");
  const auto& status = http_request.Send();
  EXPECT_EQ(error::FAILED_PRECONDITION, status.code());
  EXPECT_EQ(
      "Error executing an HTTP request: libcurl code 6 meaning "
      "'Couldn't resolve host name', error details: Could not resolve host "
      "'metadata'",
      status.error_message());
  EXPECT_EQ(0, http_request.GetResponseCode());
}

TEST(CurlHttpRequestTest, GetRequest_SslBadCertfile) {
  FakeLibCurl libcurl("get response", 0);
  libcurl.curl_easy_perform_result_ = CURLE_SSL_CACERT_BADFILE;
  libcurl.curl_easy_perform_error_message_ =
      "error setting certificate verify locations:";
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.insert(scratch.end(), kTestContent.begin(), kTestContent.end());

  http_request.SetUri("http://metadata");
  const auto& status = http_request.Send();
  EXPECT_EQ(error::FAILED_PRECONDITION, status.code());
  EXPECT_EQ(
      "Error executing an HTTP request: libcurl code 77 meaning "
      "'Problem with the SSL CA cert (path? access rights?)', error details: "
      "error setting certificate verify locations:",
      status.error_message());
  EXPECT_EQ(0, http_request.GetResponseCode());
}

TEST(CurlHttpRequestTest, ResponseHeaders) {
  FakeLibCurl libcurl(
      "get response", 200,
      {"Location: abcd", "Content-Type: text", "unparsable header"});
  CurlHttpRequest http_request(&libcurl);

  http_request.SetUri("http://www.testuri.com");
  TF_EXPECT_OK(http_request.Send());

  EXPECT_EQ("abcd", http_request.GetResponseHeader("Location"));
  EXPECT_EQ("text", http_request.GetResponseHeader("Content-Type"));
  EXPECT_EQ("", http_request.GetResponseHeader("Not-Seen-Header"));
}

TEST(CurlHttpRequestTest, PutRequest_WithBody_FromFile) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);

  auto content_filename = io::JoinPath(testing::TmpDir(), "content");
  std::ofstream content(content_filename, std::ofstream::binary);
  content << "post body content";
  content.close();

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  TF_EXPECT_OK(http_request.SetPutFromFile(content_filename, 0));
  TF_EXPECT_OK(http_request.Send());

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl.is_initialized_);
  EXPECT_EQ("http://www.testuri.com", libcurl.url_);
  EXPECT_EQ("", libcurl.custom_request_);
  EXPECT_EQ(2, libcurl.headers_->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl.headers_)[0]);
  EXPECT_EQ("Content-Length: 17", (*libcurl.headers_)[1]);
  EXPECT_TRUE(libcurl.is_put_);
  EXPECT_EQ("post body content", libcurl.posted_content_);

  std::remove(content_filename.c_str());
}

TEST(CurlHttpRequestTest, PutRequest_WithBody_FromFile_NonZeroOffset) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);

  auto content_filename = io::JoinPath(testing::TmpDir(), "content");
  std::ofstream content(content_filename, std::ofstream::binary);
  content << "post body content";
  content.close();

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  TF_EXPECT_OK(http_request.SetPutFromFile(content_filename, 7));
  TF_EXPECT_OK(http_request.Send());

  // Check interactions with libcurl.
  EXPECT_EQ("dy content", libcurl.posted_content_);

  std::remove(content_filename.c_str());
}

TEST(CurlHttpRequestTest, PutRequest_WithoutBody) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetPutEmptyBody();
  TF_EXPECT_OK(http_request.Send());

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl.is_initialized_);
  EXPECT_EQ("http://www.testuri.com", libcurl.url_);
  EXPECT_EQ("", libcurl.custom_request_);
  EXPECT_EQ(3, libcurl.headers_->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl.headers_)[0]);
  EXPECT_EQ("Content-Length: 0", (*libcurl.headers_)[1]);
  EXPECT_EQ("Transfer-Encoding: identity", (*libcurl.headers_)[2]);
  EXPECT_TRUE(libcurl.is_put_);
  EXPECT_EQ("", libcurl.posted_content_);
}

TEST(CurlHttpRequestTest, PostRequest_WithBody_FromMemory) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);

  string content = "post body content";

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetPostFromBuffer(content.c_str(), content.size());
  TF_EXPECT_OK(http_request.Send());

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl.is_initialized_);
  EXPECT_EQ("http://www.testuri.com", libcurl.url_);
  EXPECT_EQ("", libcurl.custom_request_);
  EXPECT_EQ(2, libcurl.headers_->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl.headers_)[0]);
  EXPECT_EQ("Content-Length: 17", (*libcurl.headers_)[1]);
  EXPECT_TRUE(libcurl.is_post_);
  EXPECT_EQ("post body content", libcurl.posted_content_);
}

TEST(CurlHttpRequestTest, PostRequest_WithoutBody) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);
  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetPostEmptyBody();
  TF_EXPECT_OK(http_request.Send());

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl.is_initialized_);
  EXPECT_EQ("http://www.testuri.com", libcurl.url_);
  EXPECT_EQ("", libcurl.custom_request_);
  EXPECT_EQ(3, libcurl.headers_->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl.headers_)[0]);
  EXPECT_EQ("Content-Length: 0", (*libcurl.headers_)[1]);
  EXPECT_EQ("Transfer-Encoding: identity", (*libcurl.headers_)[2]);
  EXPECT_TRUE(libcurl.is_post_);
  EXPECT_EQ("", libcurl.posted_content_);
}

TEST(CurlHttpRequestTest, DeleteRequest) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);
  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetDeleteRequest();
  TF_EXPECT_OK(http_request.Send());

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl.is_initialized_);
  EXPECT_EQ("http://www.testuri.com", libcurl.url_);
  EXPECT_EQ("DELETE", libcurl.custom_request_);
  EXPECT_EQ(1, libcurl.headers_->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl.headers_)[0]);
  EXPECT_FALSE(libcurl.is_post_);
}

TEST(CurlHttpRequestTest, WrongSequenceOfCalls_NoUri) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);
  ASSERT_DEATH((void)http_request.Send(), "URI has not been set");
}

TEST(CurlHttpRequestTest, WrongSequenceOfCalls_TwoSends) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);
  http_request.SetUri("http://www.google.com");
  TF_EXPECT_OK(http_request.Send());
  ASSERT_DEATH((void)http_request.Send(), "The request has already been sent");
}

TEST(CurlHttpRequestTest, WrongSequenceOfCalls_ReusingAfterSend) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);
  http_request.SetUri("http://www.google.com");
  TF_EXPECT_OK(http_request.Send());
  ASSERT_DEATH(http_request.SetUri("http://mail.google.com"),
               "The request has already been sent");
}

TEST(CurlHttpRequestTest, WrongSequenceOfCalls_SettingMethodTwice) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);
  http_request.SetDeleteRequest();
  ASSERT_DEATH(http_request.SetPostEmptyBody(),
               "HTTP method has been already set");
}

TEST(CurlHttpRequestTest, EscapeString) {
  FakeLibCurl libcurl("get response", 200);
  CurlHttpRequest http_request(&libcurl);
  const string test_string = "a/b/c";
  EXPECT_EQ("a%2Fb%2Fc", http_request.EscapeString(test_string));
}

TEST(CurlHttpRequestTest, ErrorReturnsNoResponse) {
  FakeLibCurl libcurl("get response", 500);
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.insert(scratch.begin(), kTestContent.begin(), kTestContent.end());
  scratch.reserve(100);

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetRange(100, 199);
  http_request.SetResultBuffer(&scratch);
  EXPECT_EQ(error::UNAVAILABLE, http_request.Send().code());

  EXPECT_EQ("", string(scratch.begin(), scratch.end()));
}

TEST(CurlHttpRequestTest, ProgressIsOk) {
  // Imitate a steady progress.
  FakeEnv env;
  FakeLibCurl libcurl(
      "test", 200,
      {
          std::make_tuple(100, 0) /* timestamp 100, 0 bytes */,
          std::make_tuple(110, 0) /* timestamp 110, 0 bytes */,
          std::make_tuple(200, 100) /* timestamp 200, 100 bytes */
      },
      &env);
  CurlHttpRequest http_request(&libcurl, &env);
  http_request.SetUri("http://www.testuri.com");
  TF_EXPECT_OK(http_request.Send());
}

TEST(CurlHttpRequestTest, ProgressIsStuck) {
  // Imitate a transmission that got stuck for more than a minute.
  FakeEnv env;
  FakeLibCurl libcurl(
      "test", 200,
      {
          std::make_tuple(100, 10) /* timestamp 100, 10 bytes */,
          std::make_tuple(130, 10) /* timestamp 130, 10 bytes */,
          std::make_tuple(170, 10) /* timestamp 170, 10 bytes */
      },
      &env);
  CurlHttpRequest http_request(&libcurl, &env);
  http_request.SetUri("http://www.testuri.com");
  auto status = http_request.Send();
  EXPECT_EQ(error::UNAVAILABLE, status.code());
  EXPECT_EQ(
      "Error executing an HTTP request: libcurl code 42 meaning 'Operation "
      "was aborted by an application callback', error details: (none)",
      status.error_message());
}

class TestStats : public HttpRequest::RequestStats {
 public:
  ~TestStats() override = default;

  void RecordRequest(const HttpRequest* request, const string& uri,
                     HttpRequest::RequestMethod method) override {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("uri: \"" + uri + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_20(mht_20_v, 999, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "RecordRequest");

    has_recorded_request_ = true;
    record_request_request_ = request;
    record_request_uri_ = uri;
    record_request_method_ = method;
  }

  void RecordResponse(const HttpRequest* request, const string& uri,
                      HttpRequest::RequestMethod method,
                      const Status& result) override {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("uri: \"" + uri + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_21(mht_21_v, 1012, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "RecordResponse");

    has_recorded_response_ = true;
    record_response_request_ = request;
    record_response_uri_ = uri;
    record_response_method_ = method;
    record_response_result_ = result;
  }

  const HttpRequest* record_request_request_ = nullptr;
  string record_request_uri_ = "http://www.testuri.com";
  HttpRequest::RequestMethod record_request_method_ =
      HttpRequest::RequestMethod::kGet;

  const HttpRequest* record_response_request_ = nullptr;
  string record_response_uri_ = "http://www.testuri.com";
  HttpRequest::RequestMethod record_response_method_ =
      HttpRequest::RequestMethod::kGet;
  Status record_response_result_;

  bool has_recorded_request_ = false;
  bool has_recorded_response_ = false;
};

class StatsTestFakeLibCurl : public FakeLibCurl {
 public:
  StatsTestFakeLibCurl(TestStats* stats, const string& response_content,
                       uint64 response_code)
      : FakeLibCurl(response_content, response_code), stats_(stats) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("response_content: \"" + response_content + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_22(mht_22_v, 1043, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "StatsTestFakeLibCurl");
}
  CURLcode curl_easy_perform(CURL* curl) override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_request_testDTcc mht_23(mht_23_v, 1047, "", "./tensorflow/core/platform/cloud/curl_http_request_test.cc", "curl_easy_perform");

    CHECK(!performed_request_);
    performed_request_ = true;
    stats_had_recorded_request_ = stats_->has_recorded_request_;
    stats_had_recorded_response_ = stats_->has_recorded_response_;
    return FakeLibCurl::curl_easy_perform(curl);
  };

  TestStats* stats_;
  bool performed_request_ = false;
  bool stats_had_recorded_request_;
  bool stats_had_recorded_response_;
};

TEST(CurlHttpRequestTest, StatsGetSuccessful) {
  TestStats stats;
  StatsTestFakeLibCurl libcurl(&stats, "get response", 200);
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.insert(scratch.begin(), kTestContent.begin(), kTestContent.end());
  scratch.reserve(100);

  http_request.SetRequestStats(&stats);

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetRange(100, 199);
  http_request.SetResultBuffer(&scratch);
  TF_EXPECT_OK(http_request.Send());

  EXPECT_EQ("get response", string(scratch.begin(), scratch.end()));

  // Check interaction with stats.
  ASSERT_TRUE(stats.has_recorded_request_);
  EXPECT_EQ(&http_request, stats.record_request_request_);
  EXPECT_EQ("http://www.testuri.com", stats.record_request_uri_);
  EXPECT_EQ(HttpRequest::RequestMethod::kGet, stats.record_request_method_);

  ASSERT_TRUE(stats.has_recorded_response_);
  EXPECT_EQ(&http_request, stats.record_response_request_);
  EXPECT_EQ("http://www.testuri.com", stats.record_response_uri_);
  EXPECT_EQ(HttpRequest::RequestMethod::kGet, stats.record_response_method_);
  TF_EXPECT_OK(stats.record_response_result_);

  // Check interaction with libcurl.
  EXPECT_TRUE(libcurl.performed_request_);
  EXPECT_TRUE(libcurl.stats_had_recorded_request_);
  EXPECT_FALSE(libcurl.stats_had_recorded_response_);
}

TEST(CurlHttpRequestTest, StatsGetNotFound) {
  TestStats stats;
  StatsTestFakeLibCurl libcurl(&stats, "get other response", 404);
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.insert(scratch.begin(), kTestContent.begin(), kTestContent.end());
  scratch.reserve(100);

  http_request.SetRequestStats(&stats);

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetRange(100, 199);
  http_request.SetResultBuffer(&scratch);
  Status s = http_request.Send();

  // Check interaction with stats.
  ASSERT_TRUE(stats.has_recorded_request_);
  EXPECT_EQ(&http_request, stats.record_request_request_);
  EXPECT_EQ("http://www.testuri.com", stats.record_request_uri_);
  EXPECT_EQ(HttpRequest::RequestMethod::kGet, stats.record_request_method_);

  ASSERT_TRUE(stats.has_recorded_response_);
  EXPECT_EQ(&http_request, stats.record_response_request_);
  EXPECT_EQ("http://www.testuri.com", stats.record_response_uri_);
  EXPECT_EQ(HttpRequest::RequestMethod::kGet, stats.record_response_method_);
  EXPECT_TRUE(errors::IsNotFound(stats.record_response_result_));
  EXPECT_EQ(s, stats.record_response_result_);

  // Check interaction with libcurl.
  EXPECT_TRUE(libcurl.performed_request_);
  EXPECT_TRUE(libcurl.stats_had_recorded_request_);
  EXPECT_FALSE(libcurl.stats_had_recorded_response_);
}

TEST(CurlHttpRequestTest, StatsPost) {
  TestStats stats;

  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);

  http_request.SetRequestStats(&stats);

  string content = "post body content";

  http_request.SetUri("http://www.testuri.com");
  http_request.SetPostFromBuffer(content.c_str(), content.size());
  TF_EXPECT_OK(http_request.Send());

  // Check interaction with stats.
  ASSERT_TRUE(stats.has_recorded_request_);
  EXPECT_EQ(&http_request, stats.record_request_request_);
  EXPECT_EQ("http://www.testuri.com", stats.record_request_uri_);
  EXPECT_EQ(HttpRequest::RequestMethod::kPost, stats.record_request_method_);

  ASSERT_TRUE(stats.has_recorded_response_);
  EXPECT_EQ(&http_request, stats.record_response_request_);
  EXPECT_EQ("http://www.testuri.com", stats.record_response_uri_);
  EXPECT_EQ(HttpRequest::RequestMethod::kPost, stats.record_response_method_);
  TF_EXPECT_OK(stats.record_response_result_);
}

TEST(CurlHttpRequestTest, StatsDelete) {
  TestStats stats;

  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);
  http_request.SetRequestStats(&stats);
  http_request.SetUri("http://www.testuri.com");
  http_request.SetDeleteRequest();
  TF_EXPECT_OK(http_request.Send());

  // Check interaction with stats.
  ASSERT_TRUE(stats.has_recorded_request_);
  EXPECT_EQ(&http_request, stats.record_request_request_);
  EXPECT_EQ("http://www.testuri.com", stats.record_request_uri_);
  EXPECT_EQ(HttpRequest::RequestMethod::kDelete, stats.record_request_method_);

  ASSERT_TRUE(stats.has_recorded_response_);
  EXPECT_EQ(&http_request, stats.record_response_request_);
  EXPECT_EQ("http://www.testuri.com", stats.record_response_uri_);
  EXPECT_EQ(HttpRequest::RequestMethod::kDelete, stats.record_response_method_);
  TF_EXPECT_OK(stats.record_response_result_);
}

TEST(CurlHttpRequestTest, StatsPut) {
  TestStats stats;

  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);
  http_request.SetRequestStats(&stats);
  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetPutEmptyBody();
  TF_EXPECT_OK(http_request.Send());

  // Check interaction with stats.
  ASSERT_TRUE(stats.has_recorded_request_);
  EXPECT_EQ(&http_request, stats.record_request_request_);
  EXPECT_EQ("http://www.testuri.com", stats.record_request_uri_);
  EXPECT_EQ(HttpRequest::RequestMethod::kPut, stats.record_request_method_);

  ASSERT_TRUE(stats.has_recorded_response_);
  EXPECT_EQ(&http_request, stats.record_response_request_);
  EXPECT_EQ("http://www.testuri.com", stats.record_response_uri_);
  EXPECT_EQ(HttpRequest::RequestMethod::kPut, stats.record_response_method_);
  TF_EXPECT_OK(stats.record_response_result_);
}

}  // namespace
}  // namespace tensorflow
