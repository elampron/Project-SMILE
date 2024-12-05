import asyncio
import logging
import os
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero, turn_detector
from livekit_plugin_smiles import LLM as SmilesLLM

load_dotenv()
logger = logging.getLogger("smiles-assistant")

def prewarm(proc: JobProcess):
    """Load models and resources before processing."""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    """Main entry point for the SMILES voice assistant."""
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are SMILES, an AI assistant. Your interface with users will be voice. "
            "You should use natural, conversational responses while being helpful and informative."
        ),
    )

    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"Starting SMILES assistant for participant {participant.identity}")

    # Configure STT model based on participant type
    dg_model = "nova-2-general"
    if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        dg_model = "nova-2-phonecall"

    # Initialize the voice pipeline agent with SMILES LLM
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model=dg_model),
        llm=SmilesLLM(
            api_url=os.getenv("SMILES_API_URL", "http://localhost:8000"),
            api_key=os.getenv("SMILES_API_KEY")
        ),
        tts=openai.TTS(),
        turn_detector=turn_detector.EOUModel(),
        chat_ctx=initial_ctx,
    )

    agent.start(ctx.room, participant)

    # Set up metrics collection
    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def _on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: ${summary}")

    ctx.add_shutdown_callback(log_usage)

    # Handle text chat messages
    chat = rtc.ChatManager(ctx.room)

    async def answer_from_text(txt: str):
        chat_ctx = agent.chat_ctx.copy()
        chat_ctx.append(role="user", text=txt)
        stream = agent.llm.chat(chat_ctx=chat_ctx)
        await agent.say(stream)

    @chat.on("message_received")
    def on_chat_received(msg: rtc.ChatMessage):
        if msg.message:
            asyncio.create_task(answer_from_text(msg.message))

    # Initial greeting
    await agent.say("Hello! I'm SMILES, how can I assist you today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
