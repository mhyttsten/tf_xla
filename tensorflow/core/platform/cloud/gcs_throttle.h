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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_GCS_THROTTLE_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_GCS_THROTTLE_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_throttleDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_throttleDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_throttleDTh() {
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


#include "tensorflow/core/platform/env.h"

namespace tensorflow {

/**
 * GcsThrottleConfig is used to configure the GcsThrottle.
 */
struct GcsThrottleConfig {
  /**
   * enabled is true if GcsThrottle should throttle requests, false otherwise.
   */
  bool enabled = false;

  /**
   * token_rate is the number of tokens accrued every second that can be used
   * for making requests to the GCS service.
   */
  int64_t token_rate =
      100000;  // Approximately 800 MBits/second bandwidth-only.

  /**
   * bucket_size is the maximum number of available tokens the GcsThrottle can
   * accrue.
   */
  int64_t bucket_size = 10000000;  // 10 million tokens total

  /**
   * tokens_per_request determines the number of tokens consumed for every
   * request.
   *
   * Note: tokens are also consumed in proportion to the response size.
   */
  int64_t tokens_per_request = 100;

  /**
   * initial_tokens determines how many tokens should be available immediately
   * after the GcsThrottle is constructed.
   */
  int64_t initial_tokens = 0;
};

/**
 * GcsThrottle is used to ensure fair use of the available GCS capacity.
 *
 * GcsThrottle operates around a concept of tokens. Tokens are consumed when
 * making requests to the GCS service. Tokens are consumed both based on the
 * number of requests made, as well as the bandwidth consumed (response sizes).
 *
 * GcsThrottle is thread safe and can be used from multiple threads.
 */
class GcsThrottle {
 public:
  /**
   * Constructs a GcsThrottle.
   */
  explicit GcsThrottle(EnvTime* env_time = nullptr);

  /**
   * AdmitRequest updates the GcsThrottle to record a request will be made.
   *
   * AdmitRequest should be called before any request is made. AdmitRequest
   * returns false if the request should be denied. If AdmitRequest
   * returns false, no tokens are consumed. If true is returned, the configured
   * number of tokens are consumed.
   */
  bool AdmitRequest();

  /**
   * RecordResponse updates the GcsThrottle to record a request has been made.
   *
   * RecordResponse should be called after the response has been received.
   * RecordResponse will update the internal state based on the number of bytes
   * in the response.
   *
   * Note: we split up the request and the response in this fashion in order to
   * avoid penalizing consumers who are using large readahead buffers at higher
   * layers of the I/O stack.
   */
  void RecordResponse(size_t num_bytes);

  /**
   * SetConfig sets the configuration for GcsThrottle and re-initializes state.
   *
   * After calling this, the token pool will be config.initial_tokens.
   */
  void SetConfig(GcsThrottleConfig config);

  /**
   * available_tokens gives a snapshot of how many tokens are available.
   *
   * The returned value should not be used to make admission decisions. The
   * purpose of this function is to make available to monitoring or other
   * instrumentation the number of available tokens in the pool.
   */
  inline int64_t available_tokens() TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    UpdateState();
    return available_tokens_;
  }

  /**
   * is_enabled determines if the throttle is enabled.
   *
   * If !is_enabled(), AdmitRequest() will always return true. To enable the
   * throttle, call SetConfig passing in a configuration that has enabled set to
   * true.
   */
  bool is_enabled() TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    return config_.enabled;
  }

 private:
  /**
   * UpdateState updates the available_tokens_ and last_updated_secs_ variables.
   *
   * UpdateState should be called in order to mark the passage of time, and
   * therefore add tokens to the available_tokens_ pool.
   */
  void UpdateState() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  inline uint64 request_bytes_to_tokens(size_t num_bytes) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_throttleDTh mht_0(mht_0_v, 309, "", "./tensorflow/core/platform/cloud/gcs_throttle.h", "request_bytes_to_tokens");

    return num_bytes >> 10;
  }

  mutex mu_;

  /**
   * last_updated_secs_ records the number of seconds since the Unix epoch that
   * the internal state of the GcsThrottle was updated. This is important when
   * determining the number of tokens to add to the available_tokens_ pool.
   */
  uint64 last_updated_secs_ TF_GUARDED_BY(mu_) = 0;

  /**
   * available_tokens_ records how many tokens are available to be consumed.
   *
   * Note: it is possible for available_tokens_ to become negative. If a
   * response comes back that consumes more than the available tokens, the count
   * will go negative, and block future requests until we have available tokens.
   */
  int64_t available_tokens_ TF_GUARDED_BY(mu_) = 0;

  EnvTime* const env_time_;
  GcsThrottleConfig config_ TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_GCS_THROTTLE_H_
