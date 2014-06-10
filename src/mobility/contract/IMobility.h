/* -*- mode:c++ -*- ********************************************************
 * file:        IMobility.h
 *
 * author:      Levente Meszaros
 *
 * copyright:   (C) 2004 Telecommunication Networks Group (TKN) at
 *              Technische Universitaet Berlin, Germany.
 *
 *              (C) 2005 Andras Varga
 *
 *              This program is free software; you can redistribute it
 *              and/or modify it under the terms of the GNU General Public
 *              License as published by the Free Software Foundation; either
 *              version 2 of the License, or (at your option) any later
 *              version.
 *              For further information see file COPYING
 *              in the top level directory
 ***************************************************************************
 * part of:     framework implementation developed by tkn
 *************************************************************************/


#ifndef __INET_IMOBILITY_H
#define __INET_IMOBILITY_H

#include "INETDefs.h"

#include "Coord.h"
#include "EulerAngles.h"


/**
 * @brief Abstract base class defining the public interface that must be provided by all mobility modules.
 *
 * @ingroup mobility
 * @author Levente Meszaros
 */
class INET_API IMobility
{
  public:
    /** @brief A signal used to publish mobility state changes. */
    static simsignal_t mobilityStateChangedSignal;

  public:
    virtual ~IMobility() {}

    /** @brief Returns the current position at the current simulation time. */
    virtual Coord getCurrentPosition() = 0;

    /** @brief Returns the current speed at the current simulation time. */
    virtual Coord getCurrentSpeed() = 0;

    /** @brief Returns the current acceleration at the current simulation time. */
    // virtual Coord getCurrentAcceleration() = 0;

    /** @brief Returns the current angular position at the current simulation time. */
    virtual EulerAngles getCurrentAngularPosition() = 0;

    /** @brief Returns the current angular speed at the current simulation time. */
    virtual EulerAngles getCurrentAngularSpeed() = 0;

    /** @brief Returns the current angular acceleration at the current simulation time. */
    // virtual Orientation getCurrentAngularAcceleration() = 0;
};

#endif
