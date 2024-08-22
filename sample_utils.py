
import os
import streamlit as st
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client



def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.  # noqa: E501
    We considered Open Relay Project (https://www.metered.ca/tools/openrelay/) too,
    but it is not stable and hardly works as some people reported like https://github.com/aiortc/aiortc/issues/832#issuecomment-1482420656  # noqa: E501
    See https://github.com/whitphx/streamlit-webrtc/issues/1213
    """

    try:
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
    except KeyError:
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    try:
        token = client.tokens.create()
    except TwilioRestException as e:
        st.warning(
            f"Error occurred while accessing Twilio API. Fallback to a free STUN server from Google. ({e})"  # noqa: E501
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]
    return token.ice_servers



def perform_cleanup(PKL_PATH):
    try:
        if os.path.exists(PKL_PATH):
            os.remove(PKL_PATH)
            print(f"Deleted file: {PKL_PATH}")
        else:
            print(f"File does not exist: {PKL_PATH}")
    except FileNotFoundError as e:
        print(f"File does not exist: {PKL_PATH}")

    except Exception as e:
        print(f"error: {e}")
