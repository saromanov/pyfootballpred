from football_predict import *

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


#Statistics examples

def stat1():
	stat = Statistics(manage.data)
	print(stat.compareTeams('everton', 'arsenal'))

def predict1():
	stat = Statistics(manage.data)
	stat.fit(['age', 'dribbles', 'totalpasses'], 'goals').predict([350, 13, 30])


#Analysis for text broadcasting

def textgame1():
	pass

predict1()
