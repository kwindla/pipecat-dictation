#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat_window_functions import (
    handle_list_windows,
    handle_remember_window,
    handle_send_text_to_window,
    handle_run_actions,
    handle_run_actions_file,
    handle_start_action_recording,
    handle_stop_action_recording,
    handle_save_sequence,
    handle_list_sequences,
    handle_delete_sequence,
    handle_run_sequence,
    list_windows_schema,
    remember_window_schema,
    send_text_to_window_schema,
    run_actions_schema,
    run_actions_file_schema,
    start_action_recording_schema,
    stop_action_recording_schema,
    save_sequence_schema,
    list_sequences_schema,
    delete_sequence_schema,
    run_sequence_schema,
)
from pipecat.frames.frames import TranscriptionMessage
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai_realtime_beta import (
    InputAudioNoiseReduction,
    InputAudioTranscription,
    OpenAIRealtimeBetaLLMService,
    SemanticTurnDetection,
    SessionProperties,
)
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor


load_dotenv(override=True)

# Load system instruction from file
with open("prompt-realtime-api.txt", "r") as f:
    SYSTEM_INSTRUCTION = f.read()

# Create tools schema with window control functions
tools = ToolsSchema(
    standard_tools=[
        list_windows_schema,
        remember_window_schema,
        send_text_to_window_schema,
        run_actions_schema,
        run_actions_file_schema,
        start_action_recording_schema,
        stop_action_recording_schema,
        save_sequence_schema,
        list_sequences_schema,
        delete_sequence_schema,
        run_sequence_schema,
    ]
)


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    session_properties = SessionProperties(
        input_audio_transcription=InputAudioTranscription(),
        # Set openai TurnDetection parameters. Not setting this at all will turn it
        # on by default
        turn_detection=SemanticTurnDetection(),
        # Or set to False to disable openai turn detection and use transport VAD
        # turn_detection=False,
        input_audio_noise_reduction=InputAudioNoiseReduction(type="near_field"),
        instructions=SYSTEM_INSTRUCTION,
    )

    llm = OpenAIRealtimeBetaLLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        session_properties=session_properties,
        start_audio_paused=False,
        model="gpt-realtime",
    )

    # Register window control and sequence functions
    llm.register_function("list_windows", handle_list_windows)
    llm.register_function("remember_window", handle_remember_window)
    llm.register_function("send_text_to_window", handle_send_text_to_window)
    llm.register_function("run_actions", handle_run_actions)
    llm.register_function("run_actions_file", handle_run_actions_file)
    llm.register_function("start_action_recording", handle_start_action_recording)
    llm.register_function("stop_action_recording", handle_stop_action_recording)
    llm.register_function("save_sequence", handle_save_sequence)
    llm.register_function("list_sequences", handle_list_sequences)
    llm.register_function("delete_sequence", handle_delete_sequence)
    llm.register_function("run_sequence", handle_run_sequence)

    transcript = TranscriptProcessor()

    # Create a standard OpenAI LLM context object using the normal messages format. The
    # OpenAIRealtimeBetaLLMService will convert this internally to messages that the
    # openai WebSocket API can understand.
    context = OpenAILLMContext(
        [{"role": "user", "content": "Say hello!"}],
        tools,
    )

    context_aggregator = llm.create_context_aggregator(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            context_aggregator.user(),
            rtvi,
            llm,  # LLM
            transcript.user(),  # Placed after the LLM, as LLM pushes TranscriptionFrames downstream
            transport.output(),  # Transport bot output
            transcript.assistant(),  # After the transcript output, to time with the audio output
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation.
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    # Register event handler for transcript updates
    @transcript.event_handler("on_transcript_update")
    async def on_transcript_update(processor, frame):
        for msg in frame.messages:
            if isinstance(msg, TranscriptionMessage):
                timestamp = f"[{msg.timestamp}] " if msg.timestamp else ""
                line = f"{timestamp}{msg.role}: {msg.content}"
                logger.info(f"Transcript: {line}")

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
