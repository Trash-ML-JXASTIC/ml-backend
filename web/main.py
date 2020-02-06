from http.server import HTTPServer,BaseHTTPRequestHandler
import json

data = {'result':'this is a test'}
host = ('localhost', 8080)

class Request(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
		self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
	def do_POST(self):
		ds = self.rfile.read(int(self.headers['content-length']))
		self.send_response(200)
		self.send_header('Access-Control-Allow-Origin', '*')
		self.send_header('Content-type', 'application/json')
		self.send_header('Content-length', int(self.headers['content-length']))
		self.send_header('test','This is test!')
		self.end_headers()
		self.wfile.write(ds)

if __name__ == '__main__':
    server = HTTPServer(host, Request)
    print('Starting server, listen at: %s:%s' % host)
    server.serve_forever()
