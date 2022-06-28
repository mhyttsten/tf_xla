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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_HTTP_REQUEST_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_HTTP_REQUEST_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_requestDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_requestDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_requestDTh() {
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

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

/// \brief An abstract basic HTTP client.
///
/// The usage pattern for the class is based on the libcurl library:
/// create a request object, set request parameters and call Send().
///
/// For example:
///   HttpRequest request;
///   request.SetUri("http://www.google.com");
///   request.SetResultsBuffer(out_buffer);
///   request.Send();
class HttpRequest {
 public:
  class Factory {
   public:
    virtual ~Factory() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_requestDTh mht_0(mht_0_v, 216, "", "./tensorflow/core/platform/cloud/http_request.h", "~Factory");
}
    virtual HttpRequest* Create() = 0;
  };

  /// RequestMethod is used to capture what type of HTTP request is made and
  /// is used in conjunction with RequestStats for instrumentation and
  /// monitoring of HTTP requests and their responses.
  enum class RequestMethod : char {
    kGet,
    kPost,
    kPut,
    kDelete,
  };

  /// RequestMethodName converts a RequestMethod to the canonical method string.
  inline static const char* RequestMethodName(RequestMethod m) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_requestDTh mht_1(mht_1_v, 234, "", "./tensorflow/core/platform/cloud/http_request.h", "RequestMethodName");

    switch (m) {
      case RequestMethod::kGet:
        return "GET";
      case RequestMethod::kPost:
        return "POST";
      case RequestMethod::kPut:
        return "PUT";
      case RequestMethod::kDelete:
        return "DELETE";
      default:
        return "???";
    }
  }

  /// RequestStats is a class that can be used to instrument an Http Request.
  class RequestStats {
   public:
    virtual ~RequestStats() = default;

    /// RecordRequest is called right before a request is sent on the wire.
    virtual void RecordRequest(const HttpRequest* request, const string& uri,
                               RequestMethod method) = 0;

    /// RecordResponse is called after the response has been received.
    virtual void RecordResponse(const HttpRequest* request, const string& uri,
                                RequestMethod method, const Status& result) = 0;
  };

  HttpRequest() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_requestDTh mht_2(mht_2_v, 266, "", "./tensorflow/core/platform/cloud/http_request.h", "HttpRequest");
}
  virtual ~HttpRequest() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPShttp_requestDTh mht_3(mht_3_v, 270, "", "./tensorflow/core/platform/cloud/http_request.h", "~HttpRequest");
}

  /// Sets the request URI.
  virtual void SetUri(const string& uri) = 0;

  /// \brief Sets the Range header.
  ///
  /// Used for random seeks, for example "0-999" returns the first 1000 bytes
  /// (note that the right border is included).
  virtual void SetRange(uint64 start, uint64 end) = 0;

  /// Sets a request header.
  virtual void AddHeader(const string& name, const string& value) = 0;

  /// Sets a DNS resolve mapping (to skip DNS resolution).
  ///
  /// Note: because GCS is available over HTTPS, we cannot replace the hostname
  /// in the URI with an IP address, as that will cause the certificate check
  /// to fail.
  virtual void AddResolveOverride(const string& hostname, int64_t port,
                                  const string& ip_addr) = 0;

  /// Sets the 'Authorization' header to the value of 'Bearer ' + auth_token.
  virtual void AddAuthBearerHeader(const string& auth_token) = 0;

  /// Sets the RequestStats object to use to record the request and response.
  virtual void SetRequestStats(RequestStats* stats) = 0;

  /// Makes the request a DELETE request.
  virtual void SetDeleteRequest() = 0;

  /// \brief Makes the request a PUT request.
  ///
  /// The request body will be taken from the specified file starting from
  /// the given offset.
  virtual Status SetPutFromFile(const string& body_filepath, size_t offset) = 0;

  /// Makes the request a PUT request with an empty body.
  virtual void SetPutEmptyBody() = 0;

  /// \brief Makes the request a POST request.
  ///
  /// The request body will be taken from the specified buffer.
  virtual void SetPostFromBuffer(const char* buffer, size_t size) = 0;

  /// Makes the request a POST request with an empty body.
  virtual void SetPostEmptyBody() = 0;

  /// \brief Specifies the buffer for receiving the response body.
  ///
  /// Size of out_buffer after an access will be exactly the number of bytes
  /// read. Existing content of the vector will be cleared.
  virtual void SetResultBuffer(std::vector<char>* out_buffer) = 0;

  /// \brief Specifies the buffer for receiving the response body.
  ///
  /// This method should be used when a caller knows the upper bound of the
  /// size of the response data.  The caller provides a pre-allocated buffer
  /// and its size. After the Send() method is called, the
  /// GetResultBufferDirectBytesTransferred() method may be used to learn to the
  /// number of bytes that were transferred using this method.
  virtual void SetResultBufferDirect(char* buffer, size_t size) = 0;

  /// \brief Returns the number of bytes transferred, when using
  /// SetResultBufferDirect(). This method may only be used when using
  /// SetResultBufferDirect().
  virtual size_t GetResultBufferDirectBytesTransferred() = 0;

  /// \brief Returns the response headers of a completed request.
  ///
  /// If the header is not found, returns an empty string.
  virtual string GetResponseHeader(const string& name) const = 0;

  /// Returns the response code of a completed request.
  virtual uint64 GetResponseCode() const = 0;

  /// \brief Sends the formed request.
  ///
  /// If the result buffer was defined, the response will be written there.
  /// The object is not designed to be re-used after Send() is executed.
  virtual Status Send() = 0;

  // Url encodes str and returns a new string.
  virtual string EscapeString(const string& str) = 0;

  /// \brief Set timeouts for this request.
  ///
  /// The connection parameter controls how long we should wait for the
  /// connection to be established. The inactivity parameter controls how long
  /// we should wait between additional responses from the server. Finally the
  /// total parameter controls the maximum total connection time to prevent
  /// hanging indefinitely.
  virtual void SetTimeouts(uint32 connection, uint32 inactivity,
                           uint32 total) = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(HttpRequest);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_HTTP_REQUEST_H_
