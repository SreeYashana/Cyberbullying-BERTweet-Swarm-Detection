import smtplib
from email.message import EmailMessage

SENDER = "sreeyashana.gscj@gmail.com"
APP_PWD = "iyauuoczwngmgski"
TO = "sreeyashana4@gmail.com"

msg = EmailMessage()
msg["Subject"] = "SMTP TEST - SafeX"
msg["From"] = SENDER
msg["To"] = TO
msg.set_content("If you got this mail, Gmail SMTP app-password works ✅")

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
    smtp.login(SENDER, APP_PWD)
    smtp.send_message(msg)

print("✅ sent")
