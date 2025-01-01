'use client';

import { AnimatePresence, motion } from "framer-motion";
import {
  LiveKitRoom,
  useVoiceAssistant,
  BarVisualizer,
  RoomAudioRenderer,
  VoiceAssistantControlBar,
  AgentState,
  DisconnectButton,
} from "@livekit/components-react";
import { useCallback, useEffect, useState } from "react";
import { MediaDeviceFailure } from "livekit-client";
import type { ConnectionDetails } from "../api/connection-details/route";
import { NoAgentNotification } from "../components/NoAgentNotification";
import { CloseIcon } from "../components/CloseIcon";

export default function VoicePage() {
  const [connectionDetails, updateConnectionDetails] = useState<
    ConnectionDetails | undefined
  >(undefined);
  const [agentState, setAgentState] = useState<AgentState>("disconnected");

  const onConnectButtonClicked = useCallback(async () => {
    const url = new URL(
      process.env.NEXT_PUBLIC_CONN_DETAILS_ENDPOINT ??
      "/api/connection-details",
      window.location.origin
    );
    const response = await fetch(url.toString());
    const connectionDetailsData = await response.json();
    updateConnectionDetails(connectionDetailsData);
  }, []);

  return (
    <div className="flex-1 flex flex-col">
      <div className="flex-1 flex items-center justify-center p-4">
        <LiveKitRoom
          token={connectionDetails?.participantToken}
          serverUrl={connectionDetails?.serverUrl}
          connect={connectionDetails !== undefined}
          audio={true}
          video={false}
          onMediaDeviceFailure={onDeviceFailure}
          onDisconnected={() => {
            updateConnectionDetails(undefined);
          }}
          className="w-full max-w-3xl mx-auto flex flex-col gap-8"
        >
          <SimpleVoiceAssistant onStateChange={setAgentState} />
          <ControlBar
            onConnectButtonClicked={onConnectButtonClicked}
            agentState={agentState}
          />
          <RoomAudioRenderer />
          <NoAgentNotification state={agentState} />
        </LiveKitRoom>
      </div>
    </div>
  );
}

function SimpleVoiceAssistant(props: {
  onStateChange: (state: AgentState) => void;
}) {
  const { state, audioTrack } = useVoiceAssistant();
  useEffect(() => {
    props.onStateChange(state);
  }, [props, state]);

  return (
    <div className="relative h-[200px] w-full bg-black/50 rounded-lg border border-green-500/20 overflow-hidden">
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="w-[400px] h-[80px]">
          <BarVisualizer
            state={state}
            barCount={5}
            trackRef={audioTrack}
            className="w-full h-full"
            options={{ 
              minHeight: 40,
              maxHeight: 80,
              color: '#22c55e',
              backgroundColor: '#1a1a1a',
              gap: 20,
              radius: 40,
              barWidth: 60,
            }}
          />
        </div>
      </div>

      {/* Status Messages */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        {state === "connecting" && (
          <div className="bg-black/50 px-4 py-2 rounded">
            <div className="text-green-500 font-mono animate-pulse">
              Connecting...
            </div>
          </div>
        )}
        {state === "listening" && (
          <div className="text-green-500/50 font-mono">
            Listening...
          </div>
        )}
        {state === "speaking" && (
          <div className="text-green-500/50 font-mono">
            Speaking...
          </div>
        )}
      </div>
    </div>
  );
}

function ControlBar(props: {
  onConnectButtonClicked: () => void;
  agentState: AgentState;
}) {
  return (
    <div className="relative h-[60px] flex items-center justify-center">
      <AnimatePresence>
        {props.agentState === "disconnected" && (
          <motion.button
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className="px-6 py-3 bg-green-500 text-black font-mono rounded hover:bg-green-400 transition-colors uppercase"
            onClick={() => props.onConnectButtonClicked()}
          >
            Start a conversation
          </motion.button>
        )}
      </AnimatePresence>
      <AnimatePresence>
        {props.agentState !== "disconnected" &&
          props.agentState !== "connecting" && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
              className="flex items-center gap-2"
            >
              <VoiceAssistantControlBar controls={{ leave: false }} />
              <DisconnectButton>
                <CloseIcon />
              </DisconnectButton>
            </motion.div>
          )}
      </AnimatePresence>
    </div>
  );
}

function onDeviceFailure(error?: MediaDeviceFailure) {
  console.error(error);
  alert(
    "Error acquiring microphone permissions. Please make sure you grant the necessary permissions in your browser and reload the tab"
  );
} 