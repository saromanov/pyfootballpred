import tornado.ioloop
import tornado.web
import tornado.iostream
import tornado.escape
import tornado.template

import football_predict


class MainHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    def get(self):
        #Demonstration
        team1, team2 = football_predict.getRandomTeams()
        name1 = team1[0]
        players1 = list(team1[1].values())
        name2 = team2[0]
        players2 = list(team2[1].values())
        return self.render('index.html', title="Football data", \
        	name1=name1, name2=name2, players1=players1,players2=players2)

    def post(self):
    	self.set_header("Content-Type", "text/plain")


class PredictHandler(tornado.web.RequestHandler):
	def get(self):
		pass



application = tornado.web.Application([
    (r"/", MainHandler),
    (r"/predict", PredictHandler)
])

if __name__ == "__main__":
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()