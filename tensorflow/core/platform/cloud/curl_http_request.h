/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_CURL_HTTP_REQUEST_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_CURL_HTTP_REQUEST_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTh() {
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


#include <string>
#include <unordered_map>
#include <vector>

#include <curl/curl.h>
#include "tensorflow/core/platform/cloud/http_request.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class LibCurl;  // libcurl interface as a class, for dependency injection.

/// \brief A basic HTTP client based on the libcurl library.
///
/// The usage pattern for the class reflects the one of the libcurl library:
/// create a request object, set request parameters and call Send().
///
/// For example:
///   std::unique_ptr<HttpRequest> request(http_request_factory->Create());
///   request->SetUri("http://www.google.com");
///   request->SetResultsBuffer(out_buffer);
///   request->Send();
class CurlHttpRequest : public HttpRequest {
 public:
  class Factory : public HttpRequest::Factory {
   public:
    virtual ~Factory() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTh mht_0(mht_0_v, 220, "", "./tensorflow/core/platform/cloud/curl_http_request.h", "~Factory");
}
    virtual HttpRequest* Create() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTh mht_1(mht_1_v, 224, "", "./tensorflow/core/platform/cloud/curl_http_request.h", "Create");
 return new CurlHttpRequest(); }
  };

  CurlHttpRequest();
  explicit CurlHttpRequest(LibCurl* libcurl)
      : CurlHttpRequest(libcurl, Env::Default()) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTh mht_2(mht_2_v, 232, "", "./tensorflow/core/platform/cloud/curl_http_request.h", "CurlHttpRequest");
}
  CurlHttpRequest(LibCurl* libcurl, Env* env);
  ~CurlHttpRequest() override;

  /// Sets the request URI.
  void SetUri(const string& uri) override;

  /// \brief Sets the Range header.
  ///
  /// Used for random seeks, for example "0-999" returns the first 1000 bytes
  /// (note that the right border is included).
  void SetRange(uint64 start, uint64 end) override;

  /// Sets a request header.
  void AddHeader(const string& name, const string& value) override;

  void AddResolveOverride(const string& hostname, int64_t port,
                          const string& ip_addr) override;

  /// Sets the 'Authorization' header to the value of 'Bearer ' + auth_token.
  void AddAuthBearerHeader(const string& auth_token) override;

  void SetRequestStats(RequestStats* stats) override;

  /// Makes the request a DELETE request.
  void SetDeleteRequest() override;

  /// \brief Makes the request a PUT request.
  ///
  /// The request body will be taken from the specified file starting from
  /// the given offset.
  Status SetPutFromFile(const string& body_filepath, size_t offset) override;

  /// Makes the request a PUT request with an empty body.
  void SetPutEmptyBody() override;

  /// \brief Makes the request a POST request.
  ///
  /// The request body will be taken from the specified buffer.
  void SetPostFromBuffer(const char* buffer, size_t size) override;

  /// Makes the request a POST request with an empty body.
  void SetPostEmptyBody() override;

  /// \brief Specifies the buffer for receiving the response body.
  ///
  /// Size of out_buffer after an access will be exactly the number of bytes
  /// read. Existing content of the vector will be cleared.
  void SetResultBuffer(std::vector<char>* out_buffer) override;

  /// \brief Specifies the buffer for receiving the response body, when the
  /// caller knows the maximum size of the response body.
  ///
  /// This method allows the caller to receive the response body without an
  /// additional intermediate buffer allocation and copy.  This method should
  /// be called before calling Send(). After Send() has succeeded, the caller
  /// should use the GetResultBufferDirectBytesTransferred() method in order
  /// to learn how many bytes were transferred.
  ///
  /// Using this method is mutually exclusive with using SetResultBuffer().
  void SetResultBufferDirect(char* buffer, size_t size) override;

  /// \brief Distinguish response type (direct vs. implicit).
  bool IsDirectResponse() const;

  /// \brief Returns the number of bytes (of the response body) that were
  /// transferred, when using the SetResultBufferDirect() method. The returned
  /// value will always be less than or equal to the 'size' parameter that
  /// was passed to SetResultBufferDirect(). If the actual HTTP response body
  /// was greater than 'size' bytes, then this transfer method will only copy
  /// the first 'size' bytes, and the rest will be ignored.
  size_t GetResultBufferDirectBytesTransferred() override;

  /// \brief Returns the response headers of a completed request.
  ///
  /// If the header is not found, returns an empty string.
  string GetResponseHeader(const string& name) const override;

  /// Returns the response code of a completed request.
  uint64 GetResponseCode() const override;

  /// \brief Sends the formed request.
  ///
  /// If the result buffer was defined, the response will be written there.
  /// The object is not designed to be re-used after Send() is executed.
  Status Send() override;

  // Url encodes str and returns a new string.
  string EscapeString(const string& str) override;

  void SetTimeouts(uint32 connection, uint32 inactivity, uint32 total) override;

 private:
  /// A write callback in the form which can be accepted by libcurl.
  static size_t WriteCallback(const void* ptr, size_t size, size_t nmemb,
                              void* userdata);

  /// Processes response body content received when using SetResultBufferDirect.
  static size_t WriteCallbackDirect(const void* ptr, size_t size, size_t nmemb,
                                    void* userdata);
  /// A read callback in the form which can be accepted by libcurl.
  static size_t ReadCallback(void* ptr, size_t size, size_t nmemb,
                             FILE* userdata);
  /// A header callback in the form which can be accepted by libcurl.
  static size_t HeaderCallback(const void* ptr, size_t size, size_t nmemb,
                               void* this_object);
  /// A progress meter callback in the form which can be accepted by libcurl.
  static int ProgressCallback(void* this_object, curl_off_t dltotal,
                              curl_off_t dlnow, curl_off_t ultotal,
                              curl_off_t ulnow);
  void CheckMethodNotSet() const;
  void CheckNotSent() const;
  StringPiece GetResponse() const;

  /// Helper to convert the given CURLcode and error buffer, representing the
  /// result of performing a transfer, into a Status with an error message.
  Status CURLcodeToStatus(CURLcode code, const char* error_buffer);

  LibCurl* libcurl_;
  Env* env_;

  FILE* put_body_ = nullptr;

  StringPiece post_body_buffer_;
  size_t post_body_read_ = 0;

  std::vector<char>* response_buffer_ = nullptr;

  struct DirectResponseState {
    char* buffer_;
    size_t buffer_size_;
    size_t bytes_transferred_;
    size_t bytes_received_;
  };
  DirectResponseState direct_response_ = {};

  CURL* curl_ = nullptr;
  curl_slist* curl_headers_ = nullptr;
  curl_slist* resolve_list_ = nullptr;

  RequestStats* stats_ = nullptr;

  std::vector<char> default_response_buffer_;

  std::unordered_map<string, string> response_headers_;
  uint64 response_code_ = 0;

  // The timestamp of the last activity related to the request execution, in
  // seconds since epoch.
  uint64 last_progress_timestamp_ = 0;
  // The last progress in terms of bytes transmitted.
  curl_off_t last_progress_bytes_ = 0;

  // The maximum period of request inactivity.
  uint32 inactivity_timeout_secs_ = 60;  // 1 minute

  // Timeout for the connection phase.
  uint32 connect_timeout_secs_ = 120;  // 2 minutes

  // Timeout for the whole request. Set only to prevent hanging indefinitely.
  uint32 request_timeout_secs_ = 3600;  // 1 hour

  // Members to enforce the usage flow.
  bool is_uri_set_ = false;
  bool is_method_set_ = false;
  bool is_sent_ = false;

  // Store the URI to help disambiguate requests when errors occur.
  string uri_;
  RequestMethod method_ = RequestMethod::kGet;

  // Limit the size of an http response that is copied into an error message.
  const size_t response_to_error_limit_ = 500;

  TF_DISALLOW_COPY_AND_ASSIGN(CurlHttpRequest);
};

/// \brief A proxy to the libcurl C interface as a dependency injection measure.
///
/// This class is meant as a very thin wrapper for the libcurl C library.
class LibCurl {
 public:
  virtual ~LibCurl() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTh mht_3(mht_3_v, 417, "", "./tensorflow/core/platform/cloud/curl_http_request.h", "~LibCurl");
}

  virtual CURL* curl_easy_init() = 0;
  virtual CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                                    uint64 param) TF_MUST_USE_RESULT = 0;
  virtual CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                                    const char* param) TF_MUST_USE_RESULT = 0;
  virtual CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                                    void* param) TF_MUST_USE_RESULT = 0;
  virtual CURLcode curl_easy_setopt(
      CURL* curl, CURLoption option,
      size_t (*param)(void*, size_t, size_t, FILE*)) TF_MUST_USE_RESULT = 0;
  virtual CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                                    size_t (*param)(const void*, size_t, size_t,
                                                    void*))
      TF_MUST_USE_RESULT = 0;
  virtual CURLcode curl_easy_setopt(
      CURL* curl, CURLoption option,
      int (*param)(void* clientp, curl_off_t dltotal, curl_off_t dlnow,
                   curl_off_t ultotal,
                   curl_off_t ulnow)) TF_MUST_USE_RESULT = 0;
  virtual CURLcode curl_easy_perform(CURL* curl) TF_MUST_USE_RESULT = 0;
  virtual CURLcode curl_easy_getinfo(CURL* curl, CURLINFO info,
                                     uint64* value) TF_MUST_USE_RESULT = 0;
  virtual CURLcode curl_easy_getinfo(CURL* curl, CURLINFO info,
                                     double* value) TF_MUST_USE_RESULT = 0;
  virtual void curl_easy_cleanup(CURL* curl) = 0;
  virtual curl_slist* curl_slist_append(curl_slist* list, const char* str) = 0;
  virtual void curl_slist_free_all(curl_slist* list) = 0;
  virtual char* curl_easy_escape(CURL* curl, const char* str, int length) = 0;
  virtual void curl_free(void* p) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_CURL_HTTP_REQUEST_H_
