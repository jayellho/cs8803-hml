/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <common/Type.h>
#include <topology/BasicTopology.h>

namespace NetworkAnalytical {

class Mesh2D final : public BasicTopology {
  public:
    /**
     * Constructor.
     *
     * @param npus_count number of npus in the FullyConnected topology
     * @param bandwidth bandwidth of each link
     * @param latency latency of each link
     */
    Mesh2D(int width, int height, Bandwidth bandwidth, Latency latency) noexcept;

    /**
     * Implementation of route function in Topology.
     */
    [[nodiscard]] Route route(DeviceId src, DeviceId dest) const noexcept override;
  
  private:
    int width;
    int height;
};

}  // namespace NetworkAnalytical
