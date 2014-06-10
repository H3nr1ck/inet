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

#ifndef __INET_RADIO_H
#define __INET_RADIO_H

#include "IRadio.h"
#include "IRadioChannel.h"
#include "IRadioAntenna.h"
#include "PhysicalLayerBase.h"
#include "RadioFrame.h"

/**
 * The standard implementation of the radio interface.
 */
// TODO: support capturing a stronger transmission
class INET_API Radio : public PhysicalLayerBase, public virtual IRadio
{
    protected:
        /**
         * An identifier which is globally unique for the whole lifetime of the
         * simulation among all radios.
         */
        const int id;

        /** @name Parameters that determine the behavior of the radio channel. */
        //@{
        /**
         * The radio antenna model is never NULL.
         */
        const IRadioAntenna *antenna;
        /**
         * The transmitter model is never NULL.
         */
        const IRadioSignalTransmitter *transmitter;
        /**
         * The receiver model is never NULL.
         */
        const IRadioSignalReceiver *receiver;
        /**
         * The radio channel model is never.
         */
        IRadioChannel *channel;
        /**
         * Simulation time required to switch from one radio mode to another.
         */
        simtime_t switchingTimes[RADIO_MODE_SWITCHING][RADIO_MODE_SWITCHING];
        //@}

        /** Gates */
        //@{
        cGate *upperLayerOut;
        cGate *upperLayerIn;
        cGate *radioIn;
        //@}

        /** State */
        //@{
        /**
         * The current radio mode.
         */
        RadioMode radioMode;
        /**
         * The radio is switching to this radio radio mode if a switch is in
         * progress, otherwise this is the same as the current radio mode.
         */
        RadioMode nextRadioMode;
        /**
         * The radio is switching from this radio mode to another if a switch is
         * in progress, otherwise this is the same as the current radio mode.
         */
        RadioMode previousRadioMode;
        /**
         * The current reception state.
         */
        ReceptionState receptionState;
        /**
         * The current transmission state.
         */
        TransmissionState transmissionState;
        //@}

        /** @name Timer */
        //@{
        /**
         * The timer that is scheduled to the end of the current transmission.
         * If this timer is not scheduled then no transmission is in progress.
         */
        cMessage *endTransmissionTimer;
        /**
         * The timer that is scheduled to the end of the current reception.
         * If this timer is NULL then no attempted reception is in progress but
         * there still may be incoming receptions which are not attempted.
         */
        cMessage *endReceptionTimer;
        /**
         * The timer that is scheduled to the end of the radio mode switch.
         */
        cMessage *endSwitchTimer;
        //@}

    private:
        void parseRadioModeSwitchingTimes();
        void startRadioModeSwitch(RadioMode newRadioMode, simtime_t switchingTime);
        void completeRadioModeSwitch(RadioMode newRadioMode);

    protected:
        virtual int numInitStages() const { return NUM_INIT_STAGES; }
        virtual void initialize(int stage);

        virtual void handleMessageWhenDown(cMessage *message);
        virtual void handleMessageWhenUp(cMessage *message);
        virtual void handleSelfMessage(cMessage *message);
        virtual void handleUpperCommand(cMessage *command);
        virtual void handleLowerCommand(cMessage *command);
        virtual bool handleOperationStage(LifecycleOperation *operation, int stage, IDoneCallback *doneCallback);

        virtual void startTransmission(cPacket *macFrame);
        virtual void endTransmission();

        virtual void startReception(RadioFrame *radioFrame);
        virtual void endReception(cMessage *message);

        virtual bool isListeningPossible();
        virtual void updateTransceiverState();

    public:
        Radio();
        Radio(RadioMode radioMode, const IRadioAntenna *antenna, const IRadioSignalTransmitter *transmitter, const IRadioSignalReceiver *receiver, IRadioChannel *channel);
        virtual ~Radio();

        virtual int getId() const { return id; }

        virtual void printToStream(std::ostream &stream) const;

        virtual const IRadioAntenna *getAntenna() const { return antenna; }
        virtual const IRadioSignalTransmitter *getTransmitter() const { return transmitter; }
        virtual const IRadioSignalReceiver *getReceiver() const { return receiver; }
        virtual const IRadioChannel *getChannel() const { return channel; }

        virtual const cGate *getRadioGate() const { return radioIn; }

        virtual RadioMode getRadioMode() const { return radioMode; }
        virtual void setRadioMode(RadioMode newRadioMode);

        virtual ReceptionState getReceptionState() const { return receptionState; }
        virtual TransmissionState getTransmissionState() const { return transmissionState; }

        virtual const IRadioSignalTransmission *getTransmissionInProgress() const;
        virtual const IRadioSignalTransmission *getReceptionInProgress() const;
};

#endif
