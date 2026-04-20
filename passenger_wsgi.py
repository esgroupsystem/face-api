from asgiref.wsgi import WsgiToAsgi
from main import app

application = WsgiToAsgi(app)