from football_predict import *
import sklearn

manage = ManageData(path='../teams')

def finder1():
	fnd = Finder('dribbles').greater(10, sort=True)\
							.viewBy('goals')\
							.viewBy('yellow')\
							.show()


def finder2():
	fnd = Finder('dribbles').greater(50)\
							.viewBy('goals')\
							.show()
def finder3():
	""" Example with team param """

	#Find the game where was a greather then 5 misses
	fnd = Finder('dribbles').greater(10).show()
	print(fnd, ...)

def finder4():
	fnd = Finder('goals').greater(10).show()
	print(fnd)

def finder5():
	fnd = Finder('goal').between(5,10)


#Statistics examples

def stat1():
	stat = Statistics(manage.data)
	print(stat.compareTeams('everton', 'arsenal'))

def predict1():
	stat = Statistics(manage.data)
	result = stat.fit(['age', 'dribbles', 'totalpasses'], 'goals').predict([[350, 13, 30], [220,18,7]])
	print(result)


#Analysis for text broadcasting

def textgame1():
	txtgame = TextGame(games='./matches')
	#txtgame.similarGames('Manchester 5-0 City', 10)
	print(len(txtgame.getGames('City')))

def textgame2():
	fun1 = LiveGameAnalysis(data='./matches')
	print(fun1.mostFreqEvents(28, 37))

def textgame3():
	fun1 = LiveGameAnalysis(data='./matches')
	#print(fun1.similarGames('Manchester 5-0 City', 20,45))
	print(list(fun1.getEvents('goal')))
def textgame4():
	fun1 = LiveGameAnalysis(data='./matches')
	print(list(fun1.getEventsByTime(1, 5)))

