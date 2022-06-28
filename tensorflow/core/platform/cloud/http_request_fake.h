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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_HTTP_REQUEST_FAKE_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_HTTP_REQUEST_FAKE_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh() {
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


#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include <curl/curl.h>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/cloud/curl_http_request.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

/// Fake HttpRequest for testing.
class FakeHttpRequest : public CurlHttpRequest {
 public:
  /// Return the response for the given request.
  FakeHttpRequest(const string& request, const string& response)
      : FakeHttpRequest(request, response, Status::OK(), nullptr, {}, 200) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("request: \"" + request + "\"");
   mht_0_v.push_back("response: \"" + response + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_0(mht_0_v, 213, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "FakeHttpRequest");
}

  /// Return the response with headers for the given request.
  FakeHttpRequest(const string& request, const string& response,
                  const std::map<string, string>& response_headers)
      : FakeHttpRequest(request, response, Status::OK(), nullptr,
                        response_headers, 200) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("request: \"" + request + "\"");
   mht_1_v.push_back("response: \"" + response + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_1(mht_1_v, 224, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "FakeHttpRequest");
}

  /// \brief Return the response for the request and capture the POST body.
  ///
  /// Post body is not expected to be a part of the 'request' parameter.
  FakeHttpRequest(const string& request, const string& response,
                  string* captured_post_body)
      : FakeHttpRequest(request, response, Status::OK(), captured_post_body, {},
                        200) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("request: \"" + request + "\"");
   mht_2_v.push_back("response: \"" + response + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_2(mht_2_v, 237, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "FakeHttpRequest");
}

  /// \brief Return the response and the status for the given request.
  FakeHttpRequest(const string& request, const string& response,
                  Status response_status, uint64 response_code)
      : FakeHttpRequest(request, response, response_status, nullptr, {},
                        response_code) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("request: \"" + request + "\"");
   mht_3_v.push_back("response: \"" + response + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_3(mht_3_v, 248, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "FakeHttpRequest");
}

  /// \brief Return the response and the status for the given request
  ///  and capture the POST body.
  ///
  /// Post body is not expected to be a part of the 'request' parameter.
  FakeHttpRequest(const string& request, const string& response,
                  Status response_status, string* captured_post_body,
                  const std::map<string, string>& response_headers,
                  uint64 response_code)
      : expected_request_(request),
        response_(response),
        response_status_(response_status),
        captured_post_body_(captured_post_body),
        response_headers_(response_headers),
        response_code_(response_code) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("request: \"" + request + "\"");
   mht_4_v.push_back("response: \"" + response + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_4(mht_4_v, 268, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "FakeHttpRequest");
}

  void SetUri(const string& uri) override {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("uri: \"" + uri + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_5(mht_5_v, 274, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "SetUri");

    actual_uri_ += "Uri: " + uri + "\n";
  }
  void SetRange(uint64 start, uint64 end) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_6(mht_6_v, 280, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "SetRange");

    actual_request_ += strings::StrCat("Range: ", start, "-", end, "\n");
  }
  void AddHeader(const string& name, const string& value) override {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + name + "\"");
   mht_7_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_7(mht_7_v, 288, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "AddHeader");

    actual_request_ += "Header " + name + ": " + value + "\n";
  }
  void AddAuthBearerHeader(const string& auth_token) override {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("auth_token: \"" + auth_token + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_8(mht_8_v, 295, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "AddAuthBearerHeader");

    actual_request_ += "Auth Token: " + auth_token + "\n";
  }
  void SetDeleteRequest() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_9(mht_9_v, 301, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "SetDeleteRequest");
 actual_request_ += "Delete: yes\n"; }
  Status SetPutFromFile(const string& body_filepath, size_t offset) override {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("body_filepath: \"" + body_filepath + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_10(mht_10_v, 306, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "SetPutFromFile");

    std::ifstream stream(body_filepath);
    const string& content = string(std::istreambuf_iterator<char>(stream),
                                   std::istreambuf_iterator<char>())
                                .substr(offset);
    actual_request_ += "Put body: " + content + "\n";
    return Status::OK();
  }
  void SetPostFromBuffer(const char* buffer, size_t size) override {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_11(mht_11_v, 318, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "SetPostFromBuffer");

    if (captured_post_body_) {
      *captured_post_body_ = string(buffer, size);
    } else {
      actual_request_ +=
          strings::StrCat("Post body: ", StringPiece(buffer, size), "\n");
    }
  }
  void SetPutEmptyBody() override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_12(mht_12_v, 329, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "SetPutEmptyBody");
 actual_request_ += "Put: yes\n"; }
  void SetPostEmptyBody() override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_13(mht_13_v, 333, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "SetPostEmptyBody");

    if (captured_post_body_) {
      *captured_post_body_ = "<empty>";
    } else {
      actual_request_ += "Post: yes\n";
    }
  }
  void SetResultBuffer(std::vector<char>* buffer) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_14(mht_14_v, 343, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "SetResultBuffer");

    buffer->clear();
    buffer_ = buffer;
  }
  void SetResultBufferDirect(char* buffer, size_t size) override {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_15(mht_15_v, 351, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "SetResultBufferDirect");

    direct_result_buffer_ = buffer;
    direct_result_buffer_size_ = size;
  }
  size_t GetResultBufferDirectBytesTransferred() override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_16(mht_16_v, 358, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "GetResultBufferDirectBytesTransferred");

    return direct_result_bytes_transferred_;
  }
  Status Send() override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_17(mht_17_v, 364, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "Send");

    EXPECT_EQ(expected_request_, actual_request())
        << "Unexpected HTTP request.";
    if (buffer_) {
      buffer_->insert(buffer_->begin(), response_.data(),
                      response_.data() + response_.size());
    } else if (direct_result_buffer_ != nullptr) {
      size_t bytes_to_copy =
          std::min<size_t>(direct_result_buffer_size_, response_.size());
      memcpy(direct_result_buffer_, response_.data(), bytes_to_copy);
      direct_result_bytes_transferred_ += bytes_to_copy;
    }
    return response_status_;
  }

  // This function just does a simple replacing of "/" with "%2F" instead of
  // full url encoding.
  string EscapeString(const string& str) override {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("str: \"" + str + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_18(mht_18_v, 385, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "EscapeString");

    const string victim = "/";
    const string encoded = "%2F";

    string copy_str = str;
    std::string::size_type n = 0;
    while ((n = copy_str.find(victim, n)) != std::string::npos) {
      copy_str.replace(n, victim.size(), encoded);
      n += encoded.size();
    }
    return copy_str;
  }

  string GetResponseHeader(const string& name) const override {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_19(mht_19_v, 402, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "GetResponseHeader");

    const auto header = response_headers_.find(name);
    return header != response_headers_.end() ? header->second : "";
  }

  virtual uint64 GetResponseCode() const override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_20(mht_20_v, 410, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "GetResponseCode");
 return response_code_; }

  void SetTimeouts(uint32 connection, uint32 inactivity,
                   uint32 total) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_21(mht_21_v, 416, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "SetTimeouts");

    actual_request_ += strings::StrCat("Timeouts: ", connection, " ",
                                       inactivity, " ", total, "\n");
  }

 private:
  string actual_request() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_22(mht_22_v, 425, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "actual_request");

    string s;
    s.append(actual_uri_);
    s.append(actual_request_);
    return s;
  }

  std::vector<char>* buffer_ = nullptr;
  char* direct_result_buffer_ = nullptr;
  size_t direct_result_buffer_size_ = 0;
  size_t direct_result_bytes_transferred_ = 0;
  string expected_request_;
  string actual_uri_;
  string actual_request_;
  string response_;
  Status response_status_;
  string* captured_post_body_ = nullptr;
  std::map<string, string> response_headers_;
  uint64 response_code_ = 0;
};

/// Fake HttpRequest factory for testing.
class FakeHttpRequestFactory : public HttpRequest::Factory {
 public:
  FakeHttpRequestFactory(const std::vector<HttpRequest*>* requests)
      : requests_(requests) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_23(mht_23_v, 453, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "FakeHttpRequestFactory");
}

  ~FakeHttpRequestFactory() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_24(mht_24_v, 458, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "~FakeHttpRequestFactory");

    EXPECT_EQ(current_index_, requests_->size())
        << "Not all expected requests were made.";
  }

  HttpRequest* Create() override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_request_fakeDTh mht_25(mht_25_v, 466, "", "./tensorflow/core/platform/cloud/http_request_fake.h", "Create");

    EXPECT_LT(current_index_, requests_->size())
        << "Too many calls of HttpRequest factory.";
    return (*requests_)[current_index_++];
  }

 private:
  const std::vector<HttpRequest*>* requests_;
  int current_index_ = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_HTTP_REQUEST_FAKE_H_
