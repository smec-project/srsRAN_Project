/*
 *
 * Copyright 2021-2024 Software Radio Systems Limited
 *
 * This file is part of srsRAN.
 *
 * srsRAN is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * srsRAN is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * A copy of the GNU Affero General Public License can be found in
 * the LICENSE file in the top-level directory of this distribution
 * and at http://www.gnu.org/licenses/.
 *
 */

#pragma once

#include "srsran/ran/slot_point.h"
#include "srsran/ran/rnti.h"
#include "srsran/du/du_cell_config.h"
#include "srsran/support/srsran_assert.h"
// #include "srsran/support/format/format_utils.h"
#include "srsran/srslog/srslog.h"
#include "srsran/ran/du_types.h"
#include <string>
#include <thread>
#include <atomic>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <mutex>

namespace srsran {

/// Structure to hold UE scheduling metrics
struct ue_scheduling_metrics {
    du_ue_index_t ue_index;
    rnti_t        crnti;
    unsigned      nof_prbs;
    slot_point    slot;
};

/// \brief Class responsible for sending scheduler metrics over TCP connection.
class scheduler_metrics_sender
{
public:
  scheduler_metrics_sender();
  ~scheduler_metrics_sender() { stop(); }

  /// \brief Initialize the TCP server.
  /// \param[in] port TCP port to listen on.
  /// \return True on success, false otherwise.
  bool init(int port = 5556);

  /// \brief Stop the TCP server and cleanup resources.
  void stop();

  /// \brief Send UE scheduling metrics to connected client.
  /// \param[in] metrics UE scheduling metrics to send.
  /// \return True if message was sent successfully, false otherwise.
  bool send_metrics(const ue_scheduling_metrics& metrics);

  bool send_message(const std::string& msg);

private:
  /// \brief Format metrics into JSON string.
  /// \param[in] metrics UE scheduling metrics to format.
  /// \return Formatted JSON string.
  std::string format_metrics(const ue_scheduling_metrics& metrics);

  /// \brief Thread function to handle incoming connections.
  void accept_connections();

  std::mutex mutex;
  int                    server_fd;
  int                    client_fd;
  std::atomic<bool>      running;
  std::thread           accept_thread;
  srslog::basic_logger& logger;
};

} // namespace srsran