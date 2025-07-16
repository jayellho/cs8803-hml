/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include <topology/Mesh2D.h>
#include <cassert>
#include <tuple>
#include <iostream>

using namespace NetworkAnalytical;

Mesh2D::Mesh2D(const int width, const int height, Bandwidth bandwidth, Latency latency) noexcept
    : BasicTopology(width * height, width * height, bandwidth, latency), width(width), height(height) {
    // check validity
    assert(width > 0);
    assert(height > 0);
    assert(bandwidth > 0);
    assert(latency >= 0);

    // set topology type
    basic_topology_type = TopologyBuildingBlock::Mesh2D;

    //// Lab 4B Part 1 =================================================
    //// Objective: Implement the physical connections of the 2D mesh topology.

    //// Hint: you'll use the connect(src, dest, bandwidth, latency) function to connect two NPUs.
    //// Hint: refer to other topology implementations (Ring, FullyConnected, Switch) for your reference.

    //// TODO: implement physical connections of the 2D mesh topology
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int index = r * width + c;
            // Connect to the right neighbor if it exists.
            if (c < width - 1) {
                int right_index = r * width + (c + 1);
                connect(index, right_index, bandwidth, latency);
            }
            // Connect to the bottom neighbor if it exists.
            if (r < height - 1) {
                int bottom_index = (r + 1) * width + c;
                connect(index, bottom_index, bandwidth, latency);
            }
        }
    }
    

    // // connect npus in a ring
    // for (auto i = 0; i < npus_count - 1; i++) {
    //     connect(i, i + 1, bandwidth, latency, bidirectional);
    // }
    // connect(npus_count - 1, 0, bandwidth, latency, bidirectional);

    //// ==============================================================
}

Route Mesh2D::route(const DeviceId src, const DeviceId dest) const noexcept {
    // assert npus are in valid range
    assert(0 <= src && src < npus_count);
    assert(0 <= dest && dest < npus_count);

    // construct route
    auto route = Route();

    //// Lab 4B Part 1 =================================================
    //// Objective: Implement xy-routing from src to dest.
    
    //// Hint: you'll use route.push_back(devices[NpuID]) to add an NPU to the route.
    //// Hint: note that route should include both src and dest, i.e., your code should ideally start with
    ////    route.push_back(devices[src]); and end with route.push_back(devices[dest]);
    //// Hint: refer to other topology implementations (Ring, FullyConnected, Switch) for your reference.

    //// TODO: translate src and dest to (x, y) coordinates
    //// Hint: this is to determine if you need x or y axis routing.
    //// Hint: use width and height
    int src_x = src % width;
    int src_y = src / width;
    int dest_x = dest % width;
    int dest_y = dest / width;


    //// TODO: route x-axis if required
    //// Hint: you may translate (x, y) coordinates back to the NpuID to add to the route.
    route.push_back(devices[src]);
    
    int cur_x = src_x;
    int cur_y = src_y;
    while (cur_x != dest_x) {
        if (dest_x > cur_x)
            cur_x++;
        else
            cur_x--;
        int id = cur_y * width + cur_x;
        route.push_back(devices[id]);
    }

    //// TODO: route y-axis if required
    //// Hint: you may translate (x, y) coordinates back to the NpuID to add to the route.

    while (cur_y != dest_y) {
        if (dest_y > cur_y)
            cur_y++;
        else
            cur_y--;
        int id = cur_y * width + cur_x;
        route.push_back(devices[id]);
    }

    //// ==============================================================

    // return constructed route
    return route;
}
