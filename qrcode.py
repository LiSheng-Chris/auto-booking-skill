import tagui as t
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud import storage
import datetime
# import webbrowser
# import threading

cred = credentials.Certificate('./static/e-charger-303306-510a928eb8dd.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

client = storage.Client.from_service_account_json(
    './static/e-charger-303306-510a928eb8dd.json')

t.init()
t.url('https://eservices.healthhub.sg/')
t.wait(6)
t.snap('//a[contains(@href, "singpassmobile.sg/qrlogin")]', 'qrcode.png')
t.close()

bucket = client.bucket("e-charger-303306.appspot.com")
blob = bucket.blob("qrcode.png")
blob.upload_from_filename('qrcode.png')
url = blob.generate_signed_url(version="v4", expiration=datetime.timedelta(minutes=60), method="GET")
db.collection(u'QRCode').document(u'4OGsFShm0OmuTq8a5c7J').set({u'qrCodeUrl': url})

# qrcode_ref = db.collection(u'QRCode').document(u'4OGsFShm0OmuTq8a5c7J')
# qrcode = qrcode_ref.get()
# qrcode_url = qrcode.to_dict()['qrCodeUrl']

# if (qrcode_url):
#     webbrowser.open(qrcode_url)
#     qrcode_ref.set({u'qrCodeUrl': ''})
# else:
#     print("null")


# callback_done = threading.Event()

# doc_ref = db.collection(u'QRCode').document(u'4OGsFShm0OmuTq8a5c7J')

# # Create a callback on_snapshot function to capture changes
# def on_snapshot(doc_snapshot, changes, read_time):
#     for doc in doc_snapshot:
#         qrcode_url = doc.to_dict()['qrCodeUrl']
#         if (qrcode_url):
#             webbrowser.open(qrcode_url)
#             doc_ref.set({u'qrCodeUrl': ''})
#     callback_done.set()

# # Watch the document
# doc_watch = doc_ref.on_snapshot(on_snapshot)
