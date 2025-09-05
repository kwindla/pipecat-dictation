import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.runner.types import RunnerArguments
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.frames.frames import LLMRunFrame, LLMMessagesAppendFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi import (
    RTVIConfig,
    RTVIObserver,
    RTVIProcessor,
    RTVIUserTranscriptionMessage,
    BotInterruptionFrame,
)
from pipecat.frames.frames import StartFrame, StopFrame
from pipecat_macos_transport.local_mac_transport import (
    LocalMacTransport,
    LocalMacTransportParams,
)

load_dotenv(override=True)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot with Gemini TTS")

    stt_speechmatics = SpeechmaticsSTTService(
        api_key=os.getenv("SPEECHMATICS_API_KEY"),
        params=SpeechmaticsSTTService.InputParams(
            language=Language.EN,
        ),
    )

    stt_deepgram = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    stt = stt_deepgram

    llm_google = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.5-flash",
    )
    llm_openai = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4.1",
    )
    llm = llm_openai

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
    rtvi_observer = RTVIObserver(rtvi)

    # System message that instructs the AI on how to speak
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant. Respond to what the user said in a creative and helpful way.""",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,
            stt,  # STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # Gemini TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[rtvi_observer],
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        rtvi.set_bot_ready()
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    @rtvi.event_handler("on_client_message")
    async def on_client_message(rtvi, message):
        # called for rtvi-ai messages of type "client-message"
        logger.info(f"!!! Client message: {message}")
        # New style: typed client messages surfaced via RTVI as message.type
        if getattr(message, "type", None) == "mute-unmute":
            try:
                mute = bool(getattr(message, "data", {}).get("mute"))
                if mute:
                    await transport.input().process_frame(StopFrame(), FrameDirection.DOWNSTREAM)
                    logger.info("Input muted via client-message (typed)")
                else:
                    await transport.input().process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
                    logger.info("Input unmuted via client-message (typed)")
            except Exception:
                logger.exception("Failed to toggle mute from client-message (typed)")
        elif message.type == "llm-input":
            # Two shapes supported:
            # 1) { messages: [...] } for direct LLM message injection
            # 2) { type: "mute-unmute", mute: true|false } for input control
            d = message.data or {}
            if isinstance(d, dict) and d.get("type") == "mute-unmute":
                try:
                    mute = bool(d.get("mute"))
                    if mute:
                        await transport.input().process_frame(
                            StopFrame(), FrameDirection.DOWNSTREAM
                        )
                        logger.info("Input muted via client-message")
                    else:
                        await transport.input().process_frame(
                            StartFrame(), FrameDirection.DOWNSTREAM
                        )
                        logger.info("Input unmuted via client-message")
                except Exception:
                    logger.exception("Failed to toggle mute from client-message")
            else:
                messages = d.get("messages", [])
                if len(messages) > 0:
                    text = messages[0].get("content", "")
                    await rtvi_observer.push_transport_message_urgent(
                        RTVIUserTranscriptionMessage(
                            data={
                                "text": text,
                                "user_id": "",
                                "timestamp": str(datetime.now()),
                                "final": True,
                            }
                        )
                    )
                    await task.queue_frames(
                        [
                            BotInterruptionFrame(),
                            LLMMessagesAppendFrame(messages=messages, run_llm=True),
                        ]
                    )

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


if __name__ == "__main__":
    logger.info("Using new AEC transport")

    params = LocalMacTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    )
    transport = LocalMacTransport(params=params)
    asyncio.run(run_bot(transport, RunnerArguments()))
