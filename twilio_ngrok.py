from gevent import monkey

monkey.patch_all()

import logging
import argparse
import tempfile
import os
import time
import sys
from agents import OpenAIChat, TwilioCaller
from audio_input import get_whisper_model
from twilio_io import TwilioServer
from conversation import run_conversation
from pyngrok import ngrok


def main(port, remote_host, start_ngrok, phone_number):
    if start_ngrok:
        ngrok_http = ngrok.connect(port)
        remote_host = ngrok_http.public_url.split("//")[1]

    static_dir = os.path.join(tempfile.gettempdir(), "twilio_static")
    os.makedirs(static_dir, exist_ok=True)

    logging.info(
        f"Starting server at {remote_host} from local:{port}, serving static content from {static_dir}, will call {phone_number}"
    )
    logging.info(f"Set call webhook to https://{remote_host}/incoming-voice")

    input(" >>> Press enter to start the call after ensuring the webhook is set. <<< ")

    tws = TwilioServer(remote_host=remote_host, port=port, static_dir=static_dir)
    tws.start()
    agent_a = OpenAIChat(
        system_prompt="""
    You are a Mercedes-Benz customer service bot that handles inquiries about vehicle purchases, test drives, and parts orders.

When stating numbers, space them out clearly (e.g., "The C 3 0 0 starts at $4 3,0 0 0"). Never use abbreviations – say "Mercedes-Benz" instead of "MB", "miles per gallon" instead of "MPG".

If asked for unavailable information, provide a reasonable Mercedes-Benz appropriate response.

Customer Details:

Name: Ahmed Al-Mansoori

Contact: +971 50 123 4567

Address: Dubai Marina, UAE

Interest:

Purchase: GLE 450 (AED 350,000 budget)

Test Drive: EQS Sedan

Parts Order: Brake pads for 2020 C 200

For Test Drives:
"Hello, I'd like to schedule a test drive for the EQS. I'm available Thursday after 3 PM at your Sheikh Zayed Road location."

For Purchases:
"I'm interested in the GLE 4 5 0 with the AMG Line package. My budget is 3 5 0,0 0 0 AED. Can you confirm availability in Obsidian Black?"

For Parts:
"I need front brake pads for my 2 0 2 0 C 2 0 0 (VIN: WDD2050471A123456). Do you offer same-day installation?"

Maintain a professional, knowledgeable tone that reflects the Mercedes-Benz brand. Prioritize clarity and premium customer service in all responses.

    """,# need to be maintained later (simple example)
        init_phrase="Hi, I would like to order a pizza.",
    )

    def run_chat(sess):
        agent_b = TwilioCaller(sess, thinking_phrase="One moment.")
        while not agent_b.session.media_stream_connected():
            time.sleep(0.1)
        run_conversation(agent_a, agent_b)
        sys.exit(0)

    tws.on_session = run_chat
    tws.start_call(phone_number)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--phone_number", type=str)
    parser.add_argument("--preload_whisper", action="store_true")
    parser.add_argument("--start_ngrok", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--remote_host", type=str, default="localhost")
    args = parser.parse_args()
    if args.preload_whisper:
        get_whisper_model()
    main(args.port, args.remote_host, args.start_ngrok, args.phone_number)
