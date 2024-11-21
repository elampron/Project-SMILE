from flask import Flask, request, redirect
import ngrok
from twilio.twiml.messaging_response import MessagingResponse
from app.agents.smile import Smile
import logging
import os
from app.configs.settings import settings
# Logging setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

smile = Smile()

@app.route("/sms", methods=['GET', 'POST'])
def sms_reply():
    """Respond to incoming calls with a simple text message."""
    # Start our TwiML response
    resp = MessagingResponse()

    # Get the message from the user
    user_message = request.values.get('Body', None)

    # Send the message to the AI not streaming
    response_content = ""
    for chunk in smile.stream(user_message):
        response_content += chunk
    resp.message(response_content)

    return str(resp)


if __name__ == "__main__":

    app.run(debug=True)
