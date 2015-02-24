from football_predict import *

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
	data = ManageData(path='../teams')
	stat = Statistics(data)

def predict1():
	pass
