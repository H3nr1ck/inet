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

#ifndef __INET_SCALARRADIO_H
#define __INET_SCALARRADIO_H

#include "Radio.h"

class INET_API ScalarRadio : public Radio
{
    protected:
        void handleUpperCommand(cMessage *message);

    public:
        ScalarRadio();
        ScalarRadio(RadioMode radioMode, const IRadioAntenna *antenna, const IRadioSignalTransmitter *transmitter, const IRadioSignalReceiver *receiver, IRadioChannel *channel);

        virtual void setBitrate(bps bitrate);
};

#endif
