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

#ifndef __INET_IRADIOSIGNALRECEPTION_H
#define __INET_IRADIOSIGNALRECEPTION_H

#include "IRadioSignalTransmission.h"

/**
 * This interface represents the reception of a transmission at a receiver. There's
 * one instance per transmission of this interface for each receiver.
 *
 * This interface is strictly immutable to safely support parallel computation.
 */
class INET_API IRadioSignalReception : public IPrintableObject
{
    public:
        /**
         * Returns the receiver that received the corresponding transmission
         * from the radio channel. This function never returns NULL.
         */
        virtual const IRadio *getReceiver() const = 0;

        /**
         * Returns the transmission that corresponds to this reception at the
         * receiver. This function never returns NULL.
         */
        virtual const IRadioSignalTransmission *getTransmission() const = 0;

        /**
         * Returns the time when the receiver started to receive the corresponding
         * transmission. It is the start of the first bit's reception.
         */
        virtual const simtime_t getStartTime() const = 0;

        /**
         * Returns the time when the receiver ended to receive the corresponding
         * transmission. It is the end of the last bit's reception.
         */
        virtual const simtime_t getEndTime() const = 0;

        /**
         * Returns the antenna's position when the receiver started to receive
         * the corresponding transmission.
         */
        virtual const Coord getStartPosition() const = 0;

        /**
         * Returns the antenna's position when the receiver ended to receive the
         * corresponding transmission.
         */
        virtual const Coord getEndPosition() const = 0;

        /**
         * Returns the antenna's orientation when the receiver started to receive
         * the corresponding transmission.
         */
        virtual const EulerAngles getStartOrientation() const = 0;

        /**
         * Returns the antenna's orientation when the receiver ended to receive
         * the corresponding transmission.
         */
        virtual const EulerAngles getEndOrientation() const = 0;
};

#endif
