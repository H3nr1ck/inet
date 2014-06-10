//
// Copyright (C) 2013 OpenSim Ltd.
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, see <http://www.gnu.org/licenses/>.
//

#ifndef __INET_IRADIOFRAME_H
#define __INET_IRADIOFRAME_H

#include "IPhysicalLayerFrame.h"
#include "IRadioSignalTransmission.h"

/**
 * This purely virtual interface provides an abstraction for different radio frames.
 */
class INET_API IRadioFrame : public IPhysicalLayerFrame, public IPrintableObject
{
    public:
        /**
         * Returns the radio signal transmission that this radio frame represents.
         * This function never returns NULL.
         */
        virtual const IRadioSignalTransmission *getTransmission() const = 0;

        /**
         * TODO: fill in the background?
         * This function may return NULL if this is not yet computed.
         */
        // virtual const IRadioSignalArrival *getArrival() const = 0;

        /**
         * TODO: fill in the background?
         * This function may return NULL if this is not yet computed.
         */
        // virtual const IRadioSignalReception *getReception() const = 0;

        /**
         * TODO: fill in the background?
         * This function may return NULL if this is not yet computed.
         */
        // virtual const IRadioSignalReceptionDecision *getReceptionDecision() const = 0;
};

#endif
