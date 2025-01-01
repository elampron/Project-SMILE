import type { AgentState } from "@livekit/components-react";
import { useEffect, useRef, useState } from "react";

interface NoAgentNotificationProps extends React.PropsWithChildren<object> {
  state: AgentState;
}

export function NoAgentNotification(props: NoAgentNotificationProps) {
  const timeToWaitMs = 10_000;
  const timeoutRef = useRef<number | null>(null);
  const [showNotification, setShowNotification] = useState(false);
  const agentHasConnected = useRef(false);

  if (
    ["listening", "thinking", "speaking"].includes(props.state) &&
    agentHasConnected.current == false
  ) {
    agentHasConnected.current = true;
  }

  useEffect(() => {
    if (props.state === "connecting") {
      timeoutRef.current = window.setTimeout(() => {
        if (
          props.state === "connecting" &&
          agentHasConnected.current === false
        ) {
          setShowNotification(true);
        }
      }, timeToWaitMs);
    } else {
      if (timeoutRef.current) {
        window.clearTimeout(timeoutRef.current);
      }
      setShowNotification(false);
    }

    return () => {
      if (timeoutRef.current) {
        window.clearTimeout(timeoutRef.current);
      }
    };
  }, [props.state]);

  return (
    <>
      {showNotification ? (
        <div className="fixed text-sm left-1/2 max-w-[90vw] -translate-x-1/2 flex top-6 items-center gap-4 bg-black/80 border border-green-500/20 px-4 py-3 rounded-lg text-green-500">
          <div>
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              className="text-green-500"
            >
              <path
                fillRule="evenodd"
                clipRule="evenodd"
                d="M9.85068 3.63564C10.8197 2.00589 13.1793 2.00589 14.1484 3.63564L21.6323 16.2223C22.6232 17.8888 21.4223 20 19.4835 20H4.51555C2.57676 20 1.37584 17.8888 2.36671 16.2223L9.85068 3.63564ZM12 8.5C12.2761 8.5 12.5 8.72386 12.5 9V13.5C12.5 13.7761 12.2761 14 12 14C11.7239 14 11.5 13.7761 11.5 13.5V9C11.5 8.72386 11.7239 8.5 12 8.5ZM12.75 16C12.75 16.4142 12.4142 16.75 12 16.75C11.5858 16.75 11.25 16.4142 11.25 16C11.25 15.5858 11.5858 15.25 12 15.25C12.4142 15.25 12.75 15.5858 12.75 16Z"
                fill="currentColor"
              />
            </svg>
          </div>
          <p className="text-pretty w-max font-mono">
            It&apos;s quiet... too quiet. Is your agent lost? Ensure your agent
            is properly configured and running on your machine.
          </p>
          <a
            href="https://docs.livekit.io/agents/quickstarts/s2s/"
            target="_blank"
            className="underline whitespace-nowrap hover:text-green-400 transition-colors"
          >
            View guide
          </a>
          <button 
            onClick={() => setShowNotification(false)}
            className="hover:text-green-400 transition-colors"
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 16 16"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M3.16602 3.16666L12.8327 12.8333M12.8327 3.16666L3.16602 12.8333"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="square"
              />
            </svg>
          </button>
        </div>
      ) : null}
    </>
  );
} 