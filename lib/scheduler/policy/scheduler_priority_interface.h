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

#ifndef SRSRAN_SCHEDULER_PRIORITY_INTERFACE_H
#define SRSRAN_SCHEDULER_PRIORITY_INTERFACE_H

#include "srsran/ran/du_types.h"

namespace srsran {

/// Parameters for UE priority update.
struct ue_priority_update {
    du_ue_index_t ue_index;
    double priority_value;
};

/// Interface for scheduler priority handling.
class scheduler_priority_handler 
{
public:
    /// Updates priority for a specific UE.
    virtual void update_ue_priority(const ue_priority_update& update) = 0;
    
    /// Resets priority for a specific UE to default.
    virtual void reset_ue_priority(du_ue_index_t ue_index) = 0;
    
    virtual ~scheduler_priority_handler() = default;
};

} // namespace srsran

#endif // SRSRAN_SCHEDULER_PRIORITY_INTERFACE_H 