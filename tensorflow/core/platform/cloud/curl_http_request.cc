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
class MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc() {
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

#include <algorithm>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/scanner.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/env_var.h"

#define CHECK_CURL_OK(expr) CHECK_EQ(expr, CURLE_OK)

namespace tensorflow {

namespace {

// Set to 1 to enable verbose debug output from curl.
constexpr uint64 kVerboseOutput = 0;

// Proxy to the real libcurl implementation.
class LibCurlProxy : public LibCurl {
 public:
  static LibCurlProxy* Load() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "Load");

    static LibCurlProxy* libcurl = []() -> LibCurlProxy* {
      curl_global_init(CURL_GLOBAL_ALL);
      return new LibCurlProxy;
    }();
    return libcurl;
  }

  CURL* curl_easy_init() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_1(mht_1_v, 222, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "curl_easy_init");
 return ::curl_easy_init(); }

  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            uint64 param) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_2(mht_2_v, 228, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "curl_easy_setopt");

    return ::curl_easy_setopt(curl, option, param);
  }

  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            const char* param) override {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("param: \"" + (param == nullptr ? std::string("nullptr") : std::string((char*)param)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_3(mht_3_v, 237, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "curl_easy_setopt");

    return ::curl_easy_setopt(curl, option, param);
  }

  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            void* param) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_4(mht_4_v, 245, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "curl_easy_setopt");

    return ::curl_easy_setopt(curl, option, param);
  }

  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            size_t (*param)(void*, size_t, size_t,
                                            FILE*)) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_5(mht_5_v, 254, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "curl_easy_setopt");

    return ::curl_easy_setopt(curl, option, param);
  }

  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            size_t (*param)(const void*, size_t, size_t,
                                            void*)) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_6(mht_6_v, 263, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "curl_easy_setopt");

    return ::curl_easy_setopt(curl, option, param);
  }

  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            int (*param)(void* clientp, curl_off_t dltotal,
                                         curl_off_t dlnow, curl_off_t ultotal,
                                         curl_off_t ulnow)) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_7(mht_7_v, 273, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "curl_easy_setopt");

    return ::curl_easy_setopt(curl, option, param);
  }

  CURLcode curl_easy_perform(CURL* curl) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_8(mht_8_v, 280, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "curl_easy_perform");

    return ::curl_easy_perform(curl);
  }

  CURLcode curl_easy_getinfo(CURL* curl, CURLINFO info,
                             uint64* value) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_9(mht_9_v, 288, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "curl_easy_getinfo");

    return ::curl_easy_getinfo(curl, info, value);
  }

  CURLcode curl_easy_getinfo(CURL* curl, CURLINFO info,
                             double* value) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_10(mht_10_v, 296, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "curl_easy_getinfo");

    return ::curl_easy_getinfo(curl, info, value);
  }

  void curl_easy_cleanup(CURL* curl) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_11(mht_11_v, 303, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "curl_easy_cleanup");

    return ::curl_easy_cleanup(curl);
  }

  char* curl_easy_escape(CURL* curl, const char* str, int length) override {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("str: \"" + (str == nullptr ? std::string("nullptr") : std::string((char*)str)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_12(mht_12_v, 311, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "curl_easy_escape");

    return ::curl_easy_escape(curl, str, length);
  }

  curl_slist* curl_slist_append(curl_slist* list, const char* str) override {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("str: \"" + (str == nullptr ? std::string("nullptr") : std::string((char*)str)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_13(mht_13_v, 319, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "curl_slist_append");

    return ::curl_slist_append(list, str);
  }

  void curl_slist_free_all(curl_slist* list) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_14(mht_14_v, 326, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "curl_slist_free_all");

    return ::curl_slist_free_all(list);
  }

  void curl_free(void* p) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_15(mht_15_v, 333, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "curl_free");
 ::curl_free(p); }
};
}  // namespace

CurlHttpRequest::CurlHttpRequest() : CurlHttpRequest(LibCurlProxy::Load()) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_16(mht_16_v, 340, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::CurlHttpRequest");
}

CurlHttpRequest::CurlHttpRequest(LibCurl* libcurl, Env* env)
    : libcurl_(libcurl), env_(env) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_17(mht_17_v, 346, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::CurlHttpRequest");

  default_response_buffer_.reserve(CURL_MAX_WRITE_SIZE);

  curl_ = libcurl_->curl_easy_init();
  CHECK(curl_ != nullptr) << "Couldn't initialize a curl session.";

  // NOTE: The cURL CA bundle path is, by default, set to
  //   etc/ssl/certs/ca-certificates.crt in tensorflow/third_party/curl.BUILD.
  //   It can be customized with the CURL_CA_BUNDLE environment variable.
  //   See also: https://curl.haxx.se/libcurl/c/CURLOPT_CAINFO.html.
  std::string value = "";
  TF_CHECK_OK(ReadStringFromEnvVar("CURL_CA_BUNDLE", "", &value));
  if (!value.empty()) {
    CHECK_CURL_OK(
        libcurl_->curl_easy_setopt(curl_, CURLOPT_CAINFO, value.c_str()));
  }
  CHECK_CURL_OK(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_VERBOSE, kVerboseOutput));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(
      curl_, CURLOPT_USERAGENT,
      strings::StrCat("TensorFlow/", TF_VERSION_STRING).c_str()));
  // Do not use signals for timeouts - does not work in multi-threaded programs.
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_NOSIGNAL, 1L));

  // TODO(b/74351157): Enable HTTP/2.
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_HTTP_VERSION,
                                           CURL_HTTP_VERSION_1_1));

  // Set up the progress meter.
  CHECK_CURL_OK(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_NOPROGRESS, uint64{0}));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_XFERINFODATA, this));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_XFERINFOFUNCTION,
                                           &CurlHttpRequest::ProgressCallback));

  // If response buffer is not set, libcurl will print results to stdout,
  // so we always set it.
  SetResultBuffer(&default_response_buffer_);
}

CurlHttpRequest::~CurlHttpRequest() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_18(mht_18_v, 389, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::~CurlHttpRequest");

  if (curl_headers_) {
    libcurl_->curl_slist_free_all(curl_headers_);
  }
  if (resolve_list_) {
    libcurl_->curl_slist_free_all(resolve_list_);
  }
  if (put_body_) {
    if (fclose(put_body_) != 0) {
      LOG(ERROR) << "fclose() failed: " << strerror(errno);
    }
  }
  if (curl_) {
    libcurl_->curl_easy_cleanup(curl_);
  }
}

string CurlHttpRequest::EscapeString(const string& str) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("str: \"" + str + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_19(mht_19_v, 410, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::EscapeString");

  char* out_char_str = libcurl_->curl_easy_escape(curl_, str.c_str(), 0);
  string out_str(out_char_str);
  libcurl_->curl_free(out_char_str);
  return out_str;
}

void CurlHttpRequest::SetUri(const string& uri) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("uri: \"" + uri + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_20(mht_20_v, 421, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::SetUri");

  CheckNotSent();
  is_uri_set_ = true;
  uri_ = uri;
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_URL, uri.c_str()));
}

void CurlHttpRequest::SetRange(uint64 start, uint64 end) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_21(mht_21_v, 431, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::SetRange");

  CheckNotSent();
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(
      curl_, CURLOPT_RANGE, strings::StrCat(start, "-", end).c_str()));
}

void CurlHttpRequest::AddHeader(const string& name, const string& value) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("name: \"" + name + "\"");
   mht_22_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_22(mht_22_v, 442, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::AddHeader");

  CheckNotSent();
  curl_headers_ = libcurl_->curl_slist_append(
      curl_headers_, strings::StrCat(name, ": ", value).c_str());
}

void CurlHttpRequest::AddResolveOverride(const string& hostname, int64_t port,
                                         const string& ip_addr) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("hostname: \"" + hostname + "\"");
   mht_23_v.push_back("ip_addr: \"" + ip_addr + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_23(mht_23_v, 454, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::AddResolveOverride");

  CheckNotSent();
  // Resolve values are hostname:port:IP.add.ress
  resolve_list_ = libcurl_->curl_slist_append(
      resolve_list_,
      strings::StrCat(hostname, ":", port, ":", ip_addr).c_str());
}

void CurlHttpRequest::AddAuthBearerHeader(const string& auth_token) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("auth_token: \"" + auth_token + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_24(mht_24_v, 466, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::AddAuthBearerHeader");

  CheckNotSent();
  if (!auth_token.empty()) {
    AddHeader("Authorization", strings::StrCat("Bearer ", auth_token));
  }
}

void CurlHttpRequest::SetRequestStats(RequestStats* stats) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_25(mht_25_v, 476, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::SetRequestStats");

  CheckNotSent();
  CHECK(stats_ == nullptr) << "SetRequestStats already called";
  stats_ = stats;
}

void CurlHttpRequest::SetDeleteRequest() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_26(mht_26_v, 485, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::SetDeleteRequest");

  CheckNotSent();
  CheckMethodNotSet();
  is_method_set_ = true;
  method_ = RequestMethod::kDelete;
  CHECK_CURL_OK(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, "DELETE"));
}

Status CurlHttpRequest::SetPutFromFile(const string& body_filepath,
                                       size_t offset) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("body_filepath: \"" + body_filepath + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_27(mht_27_v, 499, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::SetPutFromFile");

  CheckNotSent();
  CheckMethodNotSet();
  is_method_set_ = true;
  method_ = RequestMethod::kPut;
  if (put_body_) {
    if (fclose(put_body_) != 0) {
      LOG(ERROR) << "fclose() failed: " << strerror(errno);
    }
  }
  put_body_ = fopen(body_filepath.c_str(), "r");
  if (!put_body_) {
    return errors::InvalidArgument("Couldn't open the specified file: " +
                                   body_filepath);
  }
  fseek(put_body_, 0, SEEK_END);
  const auto size = ftell(put_body_) - offset;
  fseek(put_body_, offset, SEEK_SET);

  curl_headers_ = libcurl_->curl_slist_append(
      curl_headers_, strings::StrCat("Content-Length: ", size).c_str());
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_PUT, 1));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_READDATA,
                                           reinterpret_cast<void*>(put_body_)));
  // Using the default CURLOPT_READFUNCTION, which is doing an fread() on the
  // FILE * userdata set with CURLOPT_READDATA.
  return Status::OK();
}

void CurlHttpRequest::SetPutEmptyBody() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_28(mht_28_v, 531, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::SetPutEmptyBody");

  CheckNotSent();
  CheckMethodNotSet();
  is_method_set_ = true;
  method_ = RequestMethod::kPut;
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_PUT, 1));
  AddHeader("Content-Length", "0");
  AddHeader("Transfer-Encoding", "identity");
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_READDATA,
                                           reinterpret_cast<void*>(this)));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_READFUNCTION,
                                           &CurlHttpRequest::ReadCallback));
}

void CurlHttpRequest::SetPostFromBuffer(const char* buffer, size_t size) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_29(mht_29_v, 549, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::SetPostFromBuffer");

  CheckNotSent();
  CheckMethodNotSet();
  is_method_set_ = true;
  method_ = RequestMethod::kPost;
  curl_headers_ = libcurl_->curl_slist_append(
      curl_headers_, strings::StrCat("Content-Length: ", size).c_str());
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_POST, 1));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_READDATA,
                                           reinterpret_cast<void*>(this)));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_READFUNCTION,
                                           &CurlHttpRequest::ReadCallback));
  post_body_buffer_ = StringPiece(buffer, size);
}

void CurlHttpRequest::SetPostEmptyBody() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_30(mht_30_v, 567, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::SetPostEmptyBody");

  CheckNotSent();
  CheckMethodNotSet();
  is_method_set_ = true;
  method_ = RequestMethod::kPost;
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_POST, 1));
  AddHeader("Content-Length", "0");
  AddHeader("Transfer-Encoding", "identity");
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_READDATA,
                                           reinterpret_cast<void*>(this)));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_READFUNCTION,
                                           &CurlHttpRequest::ReadCallback));
}

void CurlHttpRequest::SetResultBuffer(std::vector<char>* out_buffer) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_31(mht_31_v, 584, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::SetResultBuffer");

  CheckNotSent();
  CHECK(out_buffer != nullptr);

  out_buffer->clear();
  response_buffer_ = out_buffer;

  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_WRITEDATA,
                                           reinterpret_cast<void*>(this)));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION,
                                           &CurlHttpRequest::WriteCallback));
}

void CurlHttpRequest::SetResultBufferDirect(char* buffer, size_t size) {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_32(mht_32_v, 601, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::SetResultBufferDirect");

  CHECK(buffer != nullptr);
  CheckNotSent();

  direct_response_ = DirectResponseState{buffer, size, 0, 0};
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_WRITEDATA,
                                           reinterpret_cast<void*>(this)));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(
      curl_, CURLOPT_WRITEFUNCTION, &CurlHttpRequest::WriteCallbackDirect));
}

bool CurlHttpRequest::IsDirectResponse() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_33(mht_33_v, 615, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::IsDirectResponse");

  return direct_response_.buffer_ != nullptr;
}

size_t CurlHttpRequest::WriteCallbackDirect(const void* ptr, size_t size,
                                            size_t nmemb, void* userdata) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_34(mht_34_v, 623, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::WriteCallbackDirect");

  CHECK(ptr != nullptr);
  auto that = reinterpret_cast<CurlHttpRequest*>(userdata);
  DirectResponseState* state = &that->direct_response_;
  CHECK(state->buffer_ != nullptr);
  CHECK(state->bytes_transferred_ <= state->buffer_size_);

  size_t curl_bytes_received = size * nmemb;
  size_t user_buffer_bytes_available =
      state->buffer_size_ - state->bytes_transferred_;
  size_t bytes_to_copy =
      std::min<size_t>(curl_bytes_received, user_buffer_bytes_available);
  memcpy(&state->buffer_[state->bytes_transferred_], ptr, bytes_to_copy);
  state->bytes_transferred_ += bytes_to_copy;
  state->bytes_received_ += curl_bytes_received;
  // If we didn't have room to store the full response, returning less than
  // curl_bytes_received here will abort the transfer and curl_easy_perform()
  // will return CURLE_WRITE_ERROR. We will detect and handle this error there,
  // and can use state->bytes_received_ as stored above for logging purposes.
  return bytes_to_copy;
}

size_t CurlHttpRequest::GetResultBufferDirectBytesTransferred() {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_35(mht_35_v, 648, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::GetResultBufferDirectBytesTransferred");

  CHECK(direct_response_.buffer_ != nullptr);
  return direct_response_.bytes_transferred_;
}

void CurlHttpRequest::SetTimeouts(uint32 connection, uint32 inactivity,
                                  uint32 total) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_36(mht_36_v, 657, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::SetTimeouts");

  CheckNotSent();
  connect_timeout_secs_ = connection;
  inactivity_timeout_secs_ = inactivity;
  request_timeout_secs_ = total;
}

size_t CurlHttpRequest::WriteCallback(const void* ptr, size_t size,
                                      size_t nmemb, void* this_object) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_37(mht_37_v, 668, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::WriteCallback");

  CHECK(ptr);
  auto that = reinterpret_cast<CurlHttpRequest*>(this_object);
  CHECK(that->response_buffer_);
  const size_t bytes_to_copy = size * nmemb;
  that->response_buffer_->insert(
      that->response_buffer_->end(), reinterpret_cast<const char*>(ptr),
      reinterpret_cast<const char*>(ptr) + bytes_to_copy);

  return bytes_to_copy;
}

size_t CurlHttpRequest::ReadCallback(void* ptr, size_t size, size_t nmemb,
                                     FILE* this_object) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_38(mht_38_v, 684, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::ReadCallback");

  CHECK(ptr);
  auto that = reinterpret_cast<CurlHttpRequest*>(this_object);
  CHECK(that->post_body_read_ <= that->post_body_buffer_.size());
  const size_t bytes_to_copy = std::min(
      size * nmemb, that->post_body_buffer_.size() - that->post_body_read_);
  memcpy(ptr, that->post_body_buffer_.data() + that->post_body_read_,
         bytes_to_copy);
  that->post_body_read_ += bytes_to_copy;
  return bytes_to_copy;
}

size_t CurlHttpRequest::HeaderCallback(const void* ptr, size_t size,
                                       size_t nmemb, void* this_object) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_39(mht_39_v, 700, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::HeaderCallback");

  CHECK(ptr);
  auto that = reinterpret_cast<CurlHttpRequest*>(this_object);
  StringPiece header(reinterpret_cast<const char*>(ptr), size * nmemb);
  StringPiece name, value;
  // The supplied header has the form "<name>: <value>", parse it.
  if (strings::Scanner(header)
          .ScanEscapedUntil(':')
          .StopCapture()
          .OneLiteral(": ")
          .GetResult(&value, &name)) {
    string str_value(value);
    absl::StripTrailingAsciiWhitespace(&str_value);
    that->response_headers_[string(name)] = str_value;
  }
  return size * nmemb;
}

Status CurlHttpRequest::Send() {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_40(mht_40_v, 721, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::Send");

  CheckNotSent();
  CHECK(is_uri_set_) << "URI has not been set.";

  is_sent_ = true;

  if (curl_headers_) {
    CHECK_CURL_OK(
        libcurl_->curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, curl_headers_));
  }
  if (resolve_list_) {
    CHECK_CURL_OK(
        libcurl_->curl_easy_setopt(curl_, CURLOPT_RESOLVE, resolve_list_));
  }
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_HEADERDATA,
                                           reinterpret_cast<void*>(this)));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_HEADERFUNCTION,
                                           &CurlHttpRequest::HeaderCallback));

  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_TIMEOUT,
                                           request_timeout_secs_));
  CHECK_CURL_OK(libcurl_->curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT,
                                           connect_timeout_secs_));

  char error_buffer[CURL_ERROR_SIZE] = {0};
  CHECK_CURL_OK(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_ERRORBUFFER, error_buffer));

  if (stats_ != nullptr) {
    stats_->RecordRequest(this, uri_, method_);
  }

  const CURLcode curl_result = libcurl_->curl_easy_perform(curl_);
  TF_RETURN_IF_ERROR(CURLcodeToStatus(curl_result, error_buffer));

  double written_size = 0;
  CHECK_CURL_OK(libcurl_->curl_easy_getinfo(curl_, CURLINFO_SIZE_DOWNLOAD,
                                            &written_size));

  CHECK_CURL_OK(libcurl_->curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE,
                                            &response_code_));

  auto get_error_message = [this]() -> string {
    string error_message = strings::StrCat(
        "Error executing an HTTP request: HTTP response code ", response_code_);
    StringPiece body = GetResponse();
    if (!body.empty()) {
      return strings::StrCat(
          error_message, " with body '",
          body.substr(0, std::min(body.size(), response_to_error_limit_)), "'");
    }
    return error_message;
  };

  Status result;
  switch (response_code_) {
    // The group of response codes indicating that the request achieved
    // the expected goal.
    case 200:  // OK
    case 201:  // Created
    case 204:  // No Content
    case 206:  // Partial Content
      result = Status::OK();
      break;

    case 416:  // Requested Range Not Satisfiable
      // The requested range had no overlap with the available range.
      // This doesn't indicate an error, but we should produce an empty response
      // body. (Not all servers do; GCS returns a short error message body.)
      response_buffer_->clear();
      if (IsDirectResponse()) {
        direct_response_.bytes_transferred_ = 0;
      }
      result = Status::OK();
      break;

    // INVALID_ARGUMENT indicates a problem with how the request is constructed.
    case 400:  // Bad Request
    case 406:  // Not Acceptable
    case 411:  // Length Required
    case 414:  // URI Too Long
      result = errors::InvalidArgument(get_error_message());
      break;

    // PERMISSION_DENIED indicates an authentication or an authorization issue.
    case 401:  // Unauthorized
    case 403:  // Forbidden
    case 407:  // Proxy Authorization Required
      result = errors::PermissionDenied(get_error_message());
      break;

    // NOT_FOUND indicates that the requested resource does not exist.
    case 404:  // Not found
    case 410:  // Gone
      result = errors::NotFound(get_error_message());
      break;

    // FAILED_PRECONDITION indicates that the request failed because some
    // of the underlying assumptions were not satisfied. The request
    // shouldn't be retried unless the external context has changed.
    case 302:  // Found
    case 303:  // See Other
    case 304:  // Not Modified
    case 307:  // Temporary Redirect
    case 412:  // Precondition Failed
    case 413:  // Payload Too Large
      result = errors::FailedPrecondition(get_error_message());
      break;

    // UNAVAILABLE indicates a problem that can go away if the request
    // is just retried without any modification. 308 return codes are intended
    // for write requests that can be retried. See the documentation and the
    // official library:
    // https://cloud.google.com/storage/docs/json_api/v1/how-tos/resumable-upload
    // https://github.com/google/apitools/blob/master/apitools/base/py/transfer.py
    case 308:  // Resume Incomplete
    case 409:  // Conflict
    case 429:  // Too Many Requests
    case 500:  // Internal Server Error
    case 502:  // Bad Gateway
    case 503:  // Service Unavailable
    default:   // All other HTTP response codes also should be retried.
      result = errors::Unavailable(get_error_message());
      break;
  }
  if (!result.ok()) {
    response_buffer_->clear();
  }

  if (stats_ != nullptr) {
    stats_->RecordResponse(this, uri_, method_, result);
  }

  return result;
}

void CurlHttpRequest::CheckMethodNotSet() const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_41(mht_41_v, 860, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::CheckMethodNotSet");

  CHECK(!is_method_set_) << "HTTP method has been already set.";
}

void CurlHttpRequest::CheckNotSent() const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_42(mht_42_v, 867, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::CheckNotSent");

  CHECK(!is_sent_) << "The request has already been sent.";
}

StringPiece CurlHttpRequest::GetResponse() const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_43(mht_43_v, 874, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::GetResponse");

  StringPiece response;
  if (IsDirectResponse()) {
    response = StringPiece(direct_response_.buffer_,
                           direct_response_.bytes_transferred_);
  } else {
    response = StringPiece(response_buffer_->data(), response_buffer_->size());
  }
  return response;
}

string CurlHttpRequest::GetResponseHeader(const string& name) const {
   std::vector<std::string> mht_44_v;
   mht_44_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_44(mht_44_v, 889, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::GetResponseHeader");

  const auto& header = response_headers_.find(name);
  return header != response_headers_.end() ? header->second : "";
}

uint64 CurlHttpRequest::GetResponseCode() const {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_45(mht_45_v, 897, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::GetResponseCode");
 return response_code_; }

// Cancels the transmission if no progress has been made for too long.
int CurlHttpRequest::ProgressCallback(void* this_object, curl_off_t dltotal,
                                      curl_off_t dlnow, curl_off_t ultotal,
                                      curl_off_t ulnow) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_46(mht_46_v, 905, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::ProgressCallback");

  auto that = reinterpret_cast<CurlHttpRequest*>(this_object);
  const auto now = that->env_->NowSeconds();
  const auto current_progress = dlnow + ulnow;
  if (that->last_progress_timestamp_ == 0 ||
      current_progress > that->last_progress_bytes_) {
    // This is the first time the callback is called or some progress
    // was made since the last tick.
    that->last_progress_timestamp_ = now;
    that->last_progress_bytes_ = current_progress;
    return 0;
  }

  if (now - that->last_progress_timestamp_ > that->inactivity_timeout_secs_) {
    double lookup_time = -1;
    const auto lookup_time_status = that->libcurl_->curl_easy_getinfo(
        that->curl_, CURLINFO_NAMELOOKUP_TIME, &lookup_time);

    double connect_time = -1;
    const auto connect_time_status = that->libcurl_->curl_easy_getinfo(
        that->curl_, CURLINFO_CONNECT_TIME, &connect_time);

    double pretransfer_time = -1;
    const auto pretransfer_time_status = that->libcurl_->curl_easy_getinfo(
        that->curl_, CURLINFO_PRETRANSFER_TIME, &pretransfer_time);

    double starttransfer_time = -1;
    const auto starttransfer_time_status = that->libcurl_->curl_easy_getinfo(
        that->curl_, CURLINFO_STARTTRANSFER_TIME, &starttransfer_time);

    LOG(ERROR) << "The transmission  of request " << this_object
               << " (URI: " << that->uri_ << ") has been stuck at "
               << current_progress << " of " << dltotal + ultotal
               << " bytes for " << now - that->last_progress_timestamp_
               << " seconds and will be aborted. CURL timing information: "
               << "lookup time: " << lookup_time << " ("
               << curl_easy_strerror(lookup_time_status)
               << "), connect time: " << connect_time << " ("
               << curl_easy_strerror(connect_time_status)
               << "), pre-transfer time: " << pretransfer_time << " ("
               << curl_easy_strerror(pretransfer_time_status)
               << "), start-transfer time: " << starttransfer_time << " ("
               << curl_easy_strerror(starttransfer_time_status) << ")";
    return 1;  // Will abort the request.
  }

  // No progress was made since the last call, but we should wait a bit longer.
  return 0;
}

Status CurlHttpRequest::CURLcodeToStatus(CURLcode code,
                                         const char* error_buffer) {
   std::vector<std::string> mht_47_v;
   mht_47_v.push_back("error_buffer: \"" + (error_buffer == nullptr ? std::string("nullptr") : std::string((char*)error_buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPScurl_http_requestDTcc mht_47(mht_47_v, 960, "", "./tensorflow/core/platform/cloud/curl_http_request.cc", "CurlHttpRequest::CURLcodeToStatus");

  if (code == CURLE_OK) {
    return Status::OK();
  }
  string error_message = strings::StrCat(
      "Error executing an HTTP request: libcurl code ", code, " meaning '",
      curl_easy_strerror(code), "', error details: ");
  // Special-case response-too-large errors as FAILED_PRECONDITION.
  if (code == CURLE_WRITE_ERROR && IsDirectResponse() &&
      direct_response_.bytes_received_ > direct_response_.buffer_size_) {
    string overflow_message = strings::StrCat(
        "Received ", direct_response_.bytes_received_, " response bytes ",
        "for a ", direct_response_.buffer_size_, "-byte buffer");
    uint64 response_code = 0;
    const CURLcode get_response_result = libcurl_->curl_easy_getinfo(
        curl_, CURLINFO_RESPONSE_CODE, &response_code);
    // Special-case 416 Range Not Satisfied responses; they sometimes have
    // a response body (e.g. GCS sends one with an error message) but we
    // pretend as though they don't, so actually ignore this error.
    if (get_response_result == CURLE_OK && response_code == 416) {
      return Status::OK();
    }
    return errors::FailedPrecondition(
        strings::StrCat(error_message, overflow_message));
  }
  // Domain resolution errors and certificate problems aren't going to improve
  // on retry, so we return a FailedPrecondition (as the caller must take action
  // before this can succeed).
  if (code == CURLE_COULDNT_RESOLVE_HOST || code == CURLE_SSL_CACERT_BADFILE) {
    return errors::FailedPrecondition(
        strings::StrCat(error_message, error_buffer));
  }
  // Return Unavailable to retry by default. There may be other permanent
  // failures that should be distinguished.
  return errors::Unavailable(
      strings::StrCat(error_message, *error_buffer ? error_buffer : "(none)"));
}

}  // namespace tensorflow
