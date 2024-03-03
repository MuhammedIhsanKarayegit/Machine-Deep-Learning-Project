import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendacerealtime-6060d-default-rtdb.firebaseio.com/"
})


ref = db.reference('Persons')

data = {
    "00001":
    {
        "name": "Muhammed İhsan Karayeğit",
        "profession": "Software Engineer",
        "old": 22,
        "last_attendance_time": "2024-03-3 12:08:32"
    },

    "00002":
    {
        "name": "Barış Cebeci",
        "profession": "Software Engineer",
        "old": 22,
        "last_attendance_time": "2024-03-3 12:08:32"
    }
}

for key, value in data.items():
    ref.child(key).set(value)