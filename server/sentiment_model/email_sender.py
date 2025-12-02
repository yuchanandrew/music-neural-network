import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import os

def send_success_email(count):
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(env_path)

    sender_email = os.getenv("SENDER_EMAIL")
    receiver_email = os.getenv("RECEIVER_EMAIL")
    password = os.getenv("PASSWORD")

    subject = f"[{count}] Batch Successfully Processed!"
    body = "If you are seeing this email, there is nothing to be worried about! Stay tuned for the next batch..."

    message = EmailMessage()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    message.set_content(body)

    # with open("pagespeed_results.csv", "rb") as f:
    #     message.add_attachment(f.read(), maintype="text", subtype="csv", filename=f.name)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.send_message(message)

    print("Email sent successfully!")

def send_failure_email():
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(env_path)

    sender_email = os.getenv("SENDER_EMAIL")
    receiver_email = os.getenv("RECEIVER_EMAIL")
    password = os.getenv("PASSWORD")

    subject = f"[❗] Batch Failed"
    body = "Unfortunately, the current batch has failed. Supervision required to resume data processing."

    message = EmailMessage()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    message.set_content(body)

    # with open("pagespeed_results.csv", "rb") as f:
    #     message.add_attachment(f.read(), maintype="text", subtype="csv", filename=f.name)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.send_message(message)

    print("Email sent successfully!")