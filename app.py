import tornado.ioloop
import tornado.web
import tornado.iostream
import tornado.escape
import tornado.template

import football_predict


class MainHandler(tornado.web.RequestHandler):
    def get(self):
    	return self.render('indexserver.html', title="Football data", items=["test"])

    def post(self):
    	self.set_header("Content-Type", "text/plain")



application = tornado.web.Application([
    (r"/", MainHandler),
    (r"/fun", AnotherHandler)
])

if __name__ == "__main__":
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()