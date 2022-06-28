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
class MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_placer_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_placer_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_placer_testDTcc() {
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

#include "tensorflow/core/grappler/costs/virtual_placer.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"

namespace tensorflow {
namespace grappler {

TEST(VirtualPlacerTest, LocalDevices) {
  // Create a virtual cluster with a local CPU and a local GPU
  std::unordered_map<string, DeviceProperties> devices;
  DeviceProperties cpu_device;
  cpu_device.set_type("CPU");
  devices["/job:localhost/replica:0/task:0/cpu:0"] = cpu_device;
  DeviceProperties gpu_device;
  gpu_device.set_type("GPU");
  devices["/job:localhost/replica:0/task:0/device:GPU:0"] = gpu_device;
  VirtualCluster cluster(devices);
  VirtualPlacer placer(devices);

  NodeDef node;
  node.set_op("Conv2D");
  // node.device() is empty, but GPU is default device if there is.
  EXPECT_EQ("GPU", placer.get_device(node).type());
  EXPECT_EQ("/job:localhost/replica:0/task:0/device:GPU:0",
            placer.get_canonical_device_name(node));

  node.set_device("CPU");
  EXPECT_EQ("CPU", placer.get_device(node).type());
  EXPECT_EQ("/job:localhost/replica:0/task:0/cpu:0",
            placer.get_canonical_device_name(node));

  node.set_device("GPU:0");
  EXPECT_EQ("GPU", placer.get_device(node).type());
  EXPECT_EQ("/job:localhost/replica:0/task:0/device:GPU:0",
            placer.get_canonical_device_name(node));
}

TEST(VirtualPlacerTest, ShortNames) {
  // Create a virtual cluster with a local CPU and a local GPU
  std::unordered_map<string, DeviceProperties> devices;
  DeviceProperties cpu_device;
  cpu_device.set_type("CPU");
  devices["/CPU:0"] = cpu_device;
  DeviceProperties gpu_device;
  gpu_device.set_type("GPU");
  devices["/GPU:0"] = gpu_device;
  VirtualCluster cluster(devices);
  VirtualPlacer placer(devices);

  NodeDef node;
  node.set_op("Conv2D");
  // node.device() is empty, but GPU is default device if there is.
  EXPECT_EQ("GPU", placer.get_device(node).type());
  EXPECT_EQ("/GPU:0", placer.get_canonical_device_name(node));

  node.set_device("CPU");
  EXPECT_EQ("CPU", placer.get_device(node).type());
  EXPECT_EQ("/CPU:0", placer.get_canonical_device_name(node));

  node.set_device("GPU:0");
  EXPECT_EQ("GPU", placer.get_device(node).type());
  EXPECT_EQ("/GPU:0", placer.get_canonical_device_name(node));
}

TEST(VirtualPlacerTest, PlacementOnNonDefaultDevice) {
  // Create a virtual cluster with a CPU and a device:TPU
  // Test that placement on TPU works
  // In contrast with GPU, TPU is not selected as default device at the moment.

  std::unordered_map<string, DeviceProperties> devices;
  DeviceProperties cpu_device;
  cpu_device.set_type("CPU");
  devices["/job:localhost/replica:0/task:0/cpu:0"] = cpu_device;
  DeviceProperties tpu_device;
  tpu_device.set_type("TPU");
  devices["/job:localhost/replica:0/task:0/device:TPU:0"] = tpu_device;
  VirtualCluster cluster(devices);
  VirtualPlacer placer(devices);

  NodeDef node;
  node.set_op("Conv2D");
  // node.device() is empty, and CPU is default device.
  EXPECT_EQ("CPU", placer.get_device(node).type());
  EXPECT_EQ("/job:localhost/replica:0/task:0/cpu:0",
            placer.get_canonical_device_name(node));

  node.set_device("/device:TPU:0");
  EXPECT_EQ("TPU", placer.get_device(node).type());
  EXPECT_EQ("/job:localhost/replica:0/task:0/device:TPU:0",
            placer.get_canonical_device_name(node));
}

TEST(VirtualPlacerTest, EmptyJobName) {
  // Virtual placer choose job name from the devices in cluster if a device name
  // of an op is empty. In case there are more than one kind of job name
  // or job names are missing in the devices in cluster, we use local_host.
  for (const string& job_name : {"localhost", "worker", "worker_train"}) {
    std::unordered_map<string, DeviceProperties> devices;
    DeviceProperties cpu_device;
    cpu_device.set_type("CPU");
    devices[strings::StrCat("/job:", job_name, "/replica:0/task:0/cpu:0")] =
        cpu_device;
    DeviceProperties gpu_device;
    gpu_device.set_type("GPU");
    devices[strings::StrCat("/job:", job_name,
                            "/replica:0/task:0/device:GPU:0")] = gpu_device;
    VirtualCluster cluster(devices);
    VirtualPlacer placer(devices);

    NodeDef node;
    node.set_op("Conv2D");
    node.set_device("/device:CPU:0");
    EXPECT_EQ(strings::StrCat("/job:", job_name, "/replica:0/task:0/cpu:0"),
              placer.get_canonical_device_name(node));
    node.set_device("/device:GPU:0");
    EXPECT_EQ(
        strings::StrCat("/job:", job_name, "/replica:0/task:0/device:GPU:0"),
        placer.get_canonical_device_name(node));
  }

  // When more than one job names are used, we use default "localhost"
  // This may be improved later.
  std::unordered_map<string, DeviceProperties> devices;
  DeviceProperties cpu_device;
  cpu_device.set_type("CPU");
  devices["/job:localhost/replica:0/task:0/cpu:0"] = cpu_device;
  devices["/job:ps/replica:0/task:0/cpu:0"] = cpu_device;
  devices["/job:worker/replica:0/task:0/cpu:0"] = cpu_device;
  VirtualCluster cluster(devices);
  VirtualPlacer placer(devices);

  NodeDef node;
  node.set_op("Conv2D");
  node.set_device("/device:CPU:0");
  EXPECT_EQ("/job:localhost/replica:0/task:0/cpu:0",
            placer.get_canonical_device_name(node));
}

string GetDefaultDeviceName(
    const std::unordered_map<string, DeviceProperties>& devices) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_placer_testDTcc mht_0(mht_0_v, 327, "", "./tensorflow/core/grappler/costs/virtual_placer_test.cc", "GetDefaultDeviceName");

  VirtualCluster cluster(devices);
  VirtualPlacer placer(devices);
  NodeDef node;
  node.set_op("Conv2D");
  // Device is not set to the node, so get_canonical_device_name() will return
  // the default_device_.
  return placer.get_canonical_device_name(node);
}

TEST(VirtualPlacerTest, DefaultDevice) {
  std::unordered_map<string, DeviceProperties> devices;
  DeviceProperties cpu_device;
  cpu_device.set_type("CPU");
  devices["/job:worker/replica:0/task:0/cpu:0"] = cpu_device;

  // CPU is default when there is only CPU.
  EXPECT_EQ("/job:worker/replica:0/task:0/cpu:0",
            GetDefaultDeviceName(devices));

  DeviceProperties gpu_device;
  gpu_device.set_type("GPU");

  // If there is any GPU, then gpu:0 is default device.
  for (int i = 0; i < 8; i++) {
    devices[strings::StrCat("/job:worker/replica:0/task:0/gpu:", i)] =
        gpu_device;
    EXPECT_EQ("/job:worker/replica:0/task:0/gpu:0",
              GetDefaultDeviceName(devices));
  }
}

TEST(VirtualPlacerTest, MultiReplica) {
  // Create a cluster with 8 workers, each with 8 GPUs.
  std::unordered_map<string, DeviceProperties> devices;
  DeviceProperties cpu_device;
  cpu_device.set_type("CPU");
  DeviceProperties gpu_device;
  gpu_device.set_type("GPU");
  for (int i = 0; i < 8; i++) {
    devices[strings::StrCat("/job:worker/replica:", i, "/task:0/cpu:0")] =
        cpu_device;
    for (int j = 0; j < 8; j++) {
      devices[strings::StrCat("/job:worker/replica:", i, "/task:0/gpu:", j)] =
          gpu_device;
    }
  }

  std::unique_ptr<VirtualCluster> cluster(new VirtualCluster(devices));
  std::unique_ptr<VirtualPlacer> placer(new VirtualPlacer(devices));

  auto get_device_name = [&placer](const string& device) -> string {
    NodeDef node;
    node.set_op("Conv2D");
    node.set_device(device);
    return placer->get_canonical_device_name(node);
  };

  // Validate device name is correct when we pass only replica ID and device
  // name.
  EXPECT_EQ("/job:worker/replica:0/task:0/cpu:0",
            get_device_name("/replica:0/cpu:0"));
  EXPECT_EQ("/job:worker/replica:2/task:0/cpu:0",
            get_device_name("/replica:2/cpu:0"));
  EXPECT_EQ("/job:worker/replica:7/task:0/cpu:0",
            get_device_name("/replica:7/cpu:0"));
  EXPECT_EQ("/job:worker/replica:3/task:0/gpu:0",
            get_device_name("/replica:3/gpu:0"));
  EXPECT_EQ("/job:worker/replica:5/task:0/gpu:3",
            get_device_name("/replica:5/gpu:3"));
  EXPECT_EQ("/job:worker/replica:4/task:0/gpu:7",
            get_device_name("/replica:4/gpu:7"));

  // Now add PS replicas; with multiple job names present in the cluster,
  // device names in nodes should specify job names correctly.
  for (int i = 0; i < 4; i++) {
    devices[strings::StrCat("/job:ps/replica:", i, "/task:0/cpu:0")] =
        cpu_device;
  }
  cluster.reset(new VirtualCluster(devices));
  placer.reset(new VirtualPlacer(cluster->GetDevices()));
  EXPECT_EQ("/job:worker/replica:0/task:0/cpu:0",
            get_device_name("/job:worker/replica:0/cpu:0"));
  EXPECT_EQ("/job:worker/replica:7/task:0/gpu:3",
            get_device_name("/job:worker/replica:7/gpu:3"));
  EXPECT_EQ("/job:ps/replica:0/task:0/cpu:0",
            get_device_name("/job:ps/replica:0/cpu:0"));
  EXPECT_EQ("/job:ps/replica:1/task:0/cpu:0",
            get_device_name("/job:ps/replica:1/cpu:0"));
  EXPECT_EQ("/job:ps/replica:2/task:0/cpu:0",
            get_device_name("/job:ps/replica:2/cpu:0"));
  EXPECT_EQ("/job:ps/replica:3/task:0/cpu:0",
            get_device_name("/job:ps/replica:3/cpu:0"));
}

TEST(VirtualPlacerTest, FallBackUnknown) {
  // Virtual placer falls back to "UNKNOWN" only if there are no devices in the
  // cluster.
  std::unordered_map<string, DeviceProperties> devices;
  VirtualCluster cluster(devices);
  VirtualPlacer placer(devices);

  NodeDef node;
  node.set_op("Conv2D");

  // Device falls back to UNKNOWN since the cluster has no devices.
  EXPECT_EQ("UNKNOWN", placer.get_device(node).type());
  EXPECT_EQ("UNKNOWN", placer.get_canonical_device_name(node));
}

TEST(VirtualPlacerTest, FallBackCPU) {
  std::unordered_map<string, DeviceProperties> devices;
  DeviceProperties cpu_device;
  cpu_device.set_type("CPU");
  devices["/job:my_job/replica:0/task:0/cpu:0"] = cpu_device;
  VirtualCluster cluster(devices);
  VirtualPlacer placer(devices);

  NodeDef node;
  node.set_op("Conv2D");

  // Device falls back to CPU since there is no GPU.
  EXPECT_EQ("CPU", placer.get_device(node).type());
  EXPECT_EQ("/job:my_job/replica:0/task:0/cpu:0",
            placer.get_canonical_device_name(node));
}

TEST(VirtualPlacerTest, RemoteDevices) {
  std::unordered_map<string, DeviceProperties> devices;
  DeviceProperties cpu_device;
  cpu_device.set_type("CPU");
  devices["/job:my_job/replica:0/task:0/cpu:0"] = cpu_device;
  DeviceProperties gpu_device;
  gpu_device.set_type("GPU");
  devices["/job:my_job/replica:0/task:0/device:GPU:0"] = gpu_device;
  VirtualCluster cluster(devices);
  VirtualPlacer placer(devices);

  NodeDef node;
  node.set_op("Conv2D");

  // Device falls back to GPU.
  EXPECT_EQ("GPU", placer.get_device(node).type());
  EXPECT_EQ("/job:my_job/replica:0/task:0/device:GPU:0",
            placer.get_canonical_device_name(node));

  node.set_device("/job:my_job/replica:0/task:0/cpu:0");
  EXPECT_EQ("CPU", placer.get_device(node).type());
  EXPECT_EQ("/job:my_job/replica:0/task:0/cpu:0",
            placer.get_canonical_device_name(node));

  node.set_device("/job:my_job/replica:0/task:0/device:GPU:0");
  EXPECT_EQ("GPU", placer.get_device(node).type());
  EXPECT_EQ("/job:my_job/replica:0/task:0/device:GPU:0",
            placer.get_canonical_device_name(node));

  // There is no local cpu available. Device falls back to GPU.
  node.set_device("CPU");
  EXPECT_EQ("GPU", placer.get_device(node).type());
  EXPECT_EQ("/job:my_job/replica:0/task:0/device:GPU:0",
            placer.get_canonical_device_name(node));

  node.set_device("GPU:0");
  // There is no local GPU available. Fall back to default GPU.
  EXPECT_EQ("GPU", placer.get_device(node).type());
  EXPECT_EQ("/job:my_job/replica:0/task:0/device:GPU:0",
            placer.get_canonical_device_name(node));

  // This isn't a valid name. Fall back to GPU.
  node.set_device("/job:my_job/replica:0/task:0");
  EXPECT_EQ("GPU", placer.get_device(node).type());
  EXPECT_EQ("/job:my_job/replica:0/task:0/device:GPU:0",
            placer.get_canonical_device_name(node));
}

}  // end namespace grappler
}  // end namespace tensorflow
