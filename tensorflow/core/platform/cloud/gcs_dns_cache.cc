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
class MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cacheDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cacheDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cacheDTcc() {
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
#ifndef _WIN32
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#else
#include <Windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#endif
#include <sys/types.h>

namespace tensorflow {

namespace {

const std::vector<string>& kCachedDomainNames =
    *new std::vector<string>{"www.googleapis.com", "storage.googleapis.com"};

inline void print_getaddrinfo_error(const string& name, int error_code) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cacheDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/platform/cloud/gcs_dns_cache.cc", "print_getaddrinfo_error");

#ifndef _WIN32
  if (error_code == EAI_SYSTEM) {
    LOG(ERROR) << "Error resolving " << name
               << " (EAI_SYSTEM): " << strerror(errno);
  } else {
    LOG(ERROR) << "Error resolving " << name << ": "
               << gai_strerror(error_code);
  }
#else
  // TODO:WSAGetLastError is better than gai_strerror
  LOG(ERROR) << "Error resolving " << name << ": " << gai_strerror(error_code);
#endif
}

// Selects one item at random from a vector of items, using a uniform
// distribution.
template <typename T>
const T& SelectRandomItemUniform(std::default_random_engine* random,
                                 const std::vector<T>& items) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cacheDTcc mht_1(mht_1_v, 228, "", "./tensorflow/core/platform/cloud/gcs_dns_cache.cc", "SelectRandomItemUniform");

  CHECK_GT(items.size(), 0);
  std::uniform_int_distribution<size_t> distribution(0u, items.size() - 1u);
  size_t choice_index = distribution(*random);
  return items[choice_index];
}
}  // namespace

GcsDnsCache::GcsDnsCache(Env* env, int64_t refresh_rate_secs)
    : env_(env), refresh_rate_secs_(refresh_rate_secs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cacheDTcc mht_2(mht_2_v, 240, "", "./tensorflow/core/platform/cloud/gcs_dns_cache.cc", "GcsDnsCache::GcsDnsCache");
}

void GcsDnsCache::AnnotateRequest(HttpRequest* request) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cacheDTcc mht_3(mht_3_v, 245, "", "./tensorflow/core/platform/cloud/gcs_dns_cache.cc", "GcsDnsCache::AnnotateRequest");

  // TODO(saeta): Denylist failing IP addresses.
  mutex_lock l(mu_);
  if (!started_) {
    VLOG(1) << "Starting GCS DNS cache.";
    DCHECK(!worker_) << "Worker thread already exists!";
    // Perform DNS resolutions to warm the cache.
    addresses_ = ResolveNames(kCachedDomainNames);

    // Note: we opt to use a thread instead of a delayed closure.
    worker_.reset(env_->StartThread({}, "gcs_dns_worker",
                                    [this]() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cacheDTcc mht_4(mht_4_v, 259, "", "./tensorflow/core/platform/cloud/gcs_dns_cache.cc", "lambda");
 return WorkerThread(); }));
    started_ = true;
  }

  CHECK_EQ(kCachedDomainNames.size(), addresses_.size());
  for (size_t i = 0; i < kCachedDomainNames.size(); ++i) {
    const string& name = kCachedDomainNames[i];
    const std::vector<string>& addresses = addresses_[i];
    if (!addresses.empty()) {
      const string& chosen_address =
          SelectRandomItemUniform(&random_, addresses);
      request->AddResolveOverride(name, 443, chosen_address);
      VLOG(1) << "Annotated DNS mapping: " << name << " --> " << chosen_address;
    } else {
      LOG(WARNING) << "No IP addresses available for " << name;
    }
  }
}

/* static */ std::vector<string> GcsDnsCache::ResolveName(const string& name) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cacheDTcc mht_5(mht_5_v, 282, "", "./tensorflow/core/platform/cloud/gcs_dns_cache.cc", "GcsDnsCache::ResolveName");

  VLOG(1) << "Resolving DNS name: " << name;

  addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;  // Only use IPv4 for now.
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* result = nullptr;
  int return_code = getaddrinfo(name.c_str(), nullptr, &hints, &result);

  std::vector<string> output;
  if (return_code == 0) {
    for (const addrinfo* i = result; i != nullptr; i = i->ai_next) {
      if (i->ai_family != AF_INET || i->ai_addr->sa_family != AF_INET) {
        LOG(WARNING) << "Non-IPv4 address returned. ai_family: " << i->ai_family
                     << ". sa_family: " << i->ai_addr->sa_family << ".";
        continue;
      }
      char buf[INET_ADDRSTRLEN];
      void* address_ptr =
          &(reinterpret_cast<sockaddr_in*>(i->ai_addr)->sin_addr);
      const char* formatted = nullptr;
      if ((formatted = inet_ntop(i->ai_addr->sa_family, address_ptr, buf,
                                 INET_ADDRSTRLEN)) == nullptr) {
        LOG(ERROR) << "Error converting response to IP address for " << name
                   << ": " << strerror(errno);
      } else {
        output.emplace_back(buf);
        VLOG(1) << "... address: " << buf;
      }
    }
  } else {
    print_getaddrinfo_error(name, return_code);
  }
  if (result != nullptr) {
    freeaddrinfo(result);
  }
  return output;
}

// Performs DNS resolution for a set of DNS names. The return vector contains
// one element for each element in 'names', and each element is itself a
// vector of IP addresses (in textual form).
//
// If DNS resolution fails for any name, then that slot in the return vector
// will still be present, but will be an empty vector.
//
// Ensures: names.size() == return_value.size()

std::vector<std::vector<string>> GcsDnsCache::ResolveNames(
    const std::vector<string>& names) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cacheDTcc mht_6(mht_6_v, 335, "", "./tensorflow/core/platform/cloud/gcs_dns_cache.cc", "GcsDnsCache::ResolveNames");

  std::vector<std::vector<string>> all_addresses;
  all_addresses.reserve(names.size());
  for (const string& name : names) {
    all_addresses.push_back(ResolveName(name));
  }
  return all_addresses;
}

void GcsDnsCache::WorkerThread() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgcs_dns_cacheDTcc mht_7(mht_7_v, 347, "", "./tensorflow/core/platform/cloud/gcs_dns_cache.cc", "GcsDnsCache::WorkerThread");

  while (true) {
    {
      // Don't immediately re-resolve the addresses.
      mutex_lock l(mu_);
      if (cancelled_) return;
      cond_var_.wait_for(l, std::chrono::seconds(refresh_rate_secs_));
      if (cancelled_) return;
    }

    // Resolve DNS values
    auto new_addresses = ResolveNames(kCachedDomainNames);

    {
      mutex_lock l(mu_);
      // Update instance variables.
      addresses_.swap(new_addresses);
    }
  }
}

}  // namespace tensorflow
