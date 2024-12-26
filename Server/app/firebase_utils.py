import firebase_admin
from firebase_admin import credentials, db

def initialize_firebase():
    try:

        cred = credentials.Certificate("config/drowsines-key-firebase.json")
        
        firebase_admin.initialize_app(cred, {
            "databaseURL": "https://drowsines-e2e79-default-rtdb.firebaseio.com/"
        })

        print("Firebase initialized successfully!")

    except Exception as e:
        print(f"Error initializing Firebase: {e}")

def send_status_to_firebase(status, confidence):
    try:

        ref = db.reference("/pengendara")
        ref.set(status)
        ref = db.reference("/confidence")
        ref.set(confidence)

        print(f"Status '{status}' : '{confidence}' berhasil dikirim ke Firebase.")

    except Exception as e:
        print(f"Error sending status to Firebase: {e}")
        